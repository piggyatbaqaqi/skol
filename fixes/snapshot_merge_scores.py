#!/usr/bin/env python3
"""Snapshot ``metrics.n_terms_above_5`` for every merge suspect.

**Run this before anything touches the merge-suspect population.**

``bin/llm_annotate_features`` replaces the whole status doc when it
annotates a treatment, and ``skipped_merge_suspect`` is neither
``success`` nor ``error``, so ``filter_already_annotated`` lets those
treatments through.  Annotating one therefore **wipes**
``metrics.n_terms_above_5``.  That cascades: once the status is gone,
``fetch_prior_merge_skip_ids`` no longer recognises the treatment as a
suspect, and a later ``select_for_annotation
--exclude-suspected-merges`` silently re-admits it into the annotatable
pool — quietly contaminating the very population separation the
sampling design depends on.

The score is recoverable by recomputation, but only while the
treatments DB is unchanged, and the snapshot costs one scan.

It doubles as the severity-ranked work queue for merge-suspect review:
the score is graded (10 → 915, median 22), not a boolean.

One-shot, hence ``fixes/`` rather than ``bin/``.

Usage::

    fixes/snapshot_merge_scores --experiment production_v4 \\
        --output data/merge_suspects_20260823.tsv
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import (
    Any, Dict, Iterator, List, Optional, Sequence, Tuple,
)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'bin'))

STATUS_MERGE_SUSPECT = 'skipped_merge_suspect'

_TSV_HEADER = 'treatment_id\tn_terms_above_5'


def parse_metrics(raw: Any) -> Dict[str, Any]:
    """Coerce a status doc's ``metrics`` field to a mapping.

    Live status docs store ``metrics`` inconsistently — sometimes a
    real mapping, sometimes its Python ``repr`` (single quotes, so not
    JSON).  Anything that does not resolve to a mapping yields ``{}``
    rather than raising: a malformed doc must not abort a scan whose
    whole purpose is to preserve data.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    # JSON first (cheaper, stricter); fall back to the Python repr
    # form, which is what the live docs actually contain.
    for loads in (json.loads, ast.literal_eval):
        try:
            value = loads(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def iter_merge_scores(
    status_db: Any,
) -> Iterator[Tuple[str, Optional[int]]]:
    """Yield ``(treatment_id, n_terms_above_5)`` for merge suspects.

    A suspect whose score is missing or unparseable yields ``None``
    rather than being skipped — dropping it would understate the
    population and make the snapshot an incomplete restore.
    """
    for row in status_db.view('_all_docs', include_docs=True):
        doc = getattr(row, 'doc', None)
        if not doc or doc.get('status') != STATUS_MERGE_SUSPECT:
            continue
        raw = parse_metrics(doc.get('metrics')).get('n_terms_above_5')
        try:
            score: Optional[int] = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            score = None
        yield row.id, score


def format_tsv(rows: Sequence[Tuple[str, Optional[int]]]) -> str:
    """Render rows as TSV, highest score first.

    Descending order makes the file a ready-made severity queue.
    Rows with no score sort last and render as an empty field.
    """
    # -1 sorts unscored rows after every real score, which start at
    # the merge threshold and are therefore always positive.
    ordered = sorted(
        rows, key=lambda r: (-(r[1] if r[1] is not None else -1), r[0]),
    )
    lines = [_TSV_HEADER]
    lines.extend(
        f'{tid}\t{"" if score is None else score}'
        for tid, score in ordered
    )
    return '\n'.join(lines) + '\n'


def main(argv: Optional[Sequence[str]] = None) -> int:
    from env_config import common_parser, get_env_config

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output', default=None, metavar='PATH',
        help='Write TSV here (default: stdout).',
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)

    if not config.get('experiment_name'):
        print("error: --experiment is required", file=sys.stderr)
        return 2

    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )
    # Reuse the status-DB resolver from bin/llm_annotate_features so
    # the naming-convention fallback stays in one place.
    from llm_annotate_features import resolve_status_db_name
    experiment = config['experiment_name']
    experiments_db = server['skol_experiments']
    if experiment not in experiments_db:
        print(f"error: experiment {experiment!r} not found in "
              f"skol_experiments", file=sys.stderr)
        return 2
    db_name = resolve_status_db_name(
        experiment, experiments_db[experiment],
        verbosity=int(config.get('verbosity', 1) or 0),
    )
    if db_name not in server:
        print(f"error: {db_name} not found", file=sys.stderr)
        return 2

    rows: List[Tuple[str, Optional[int]]] = list(
        iter_merge_scores(server[db_name])
    )
    missing = sum(1 for _tid, score in rows if score is None)
    text = format_tsv(rows)

    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
        print(f"{len(rows)} merge suspects -> {args.output}",
              file=sys.stderr)
    else:
        sys.stdout.write(text)
    if missing:
        print(f"warning: {missing} suspects had no n_terms_above_5",
              file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
