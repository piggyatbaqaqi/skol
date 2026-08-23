#!/usr/bin/env python3
"""Sample treatments and check that their spans still resolve.

The guard that makes ``Span.head`` an active defence rather than a
passive field.  Span offsets are indexed to the treatment's
``attachment_name`` inside its ``annotations_db``; resolving them
against the wrong attachment returns plausible prose rather than an
error (§16 in ``docs/data_quality_production_v4_model.md``).  This
samples the corpus, resolves each span through
:mod:`span_resolver`, and reports anything that fails.

Run it after a re-extraction, after any change to how ``*_spans``
are written, and periodically — a re-extraction that rewrites
``article.txt.ann`` without rewriting the offsets is exactly the
drift this catches.

Exit status is 1 if any span fails to resolve, so it can gate a
pipeline step.

Usage::

    bin/verify_spans --experiment production_v4 --sample 200
"""

import argparse
import dataclasses
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402
from span_resolver import (  # noqa: E402
    SpanResolutionError,
    resolve_span,
)

_SPAN_FIELDS = (
    'nomenclature_spans', 'description_spans', 'diagnosis_spans',
    'etymology_spans', 'distribution_spans', 'materials_examined_spans',
    'type_designation_spans', 'biology_spans', 'notes_spans',
    'figure_caption_spans',
)

_DEFAULT_SAMPLE = 200


@dataclasses.dataclass(frozen=True)
class SpanCheck:
    """The outcome of resolving one span."""

    treatment_id: str
    field: str
    index: int
    ok: bool
    reason: str


def check_treatment(
    treatment: Dict[str, Any], server: Any,
) -> List[SpanCheck]:
    """Resolve every span on one treatment.

    A treatment whose coordinate space cannot be determined fails
    each of its spans with that reason rather than raising, so one
    bad document does not abort the sample.
    """
    out: List[SpanCheck] = []
    treatment_id = treatment.get('_id', '?')
    for field in _SPAN_FIELDS:
        for index, span in enumerate(treatment.get(field) or []):
            try:
                resolve_span(treatment, span, server)
            except SpanResolutionError as exc:
                out.append(SpanCheck(
                    treatment_id, field, index, False, str(exc),
                ))
            else:
                out.append(SpanCheck(treatment_id, field, index, True, ''))
    return out


def summarise(checks: List[SpanCheck]) -> Tuple[int, int, float]:
    """Return ``(total, ok, percentage_ok)``."""
    total = len(checks)
    ok = sum(1 for c in checks if c.ok)
    return total, ok, (100.0 * ok / total) if total else 0.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--sample', type=int, default=_DEFAULT_SAMPLE, metavar='N',
        help=f'Treatments to check (default {_DEFAULT_SAMPLE}).  '
             f'0 checks every treatment.',
    )
    parser.add_argument(
        '--seed', type=int, default=None, metavar='N',
        help='Seed the sample for a reproducible run.',
    )
    parser.add_argument(
        '--show', type=int, default=10, metavar='N',
        help='Failures to print in full (default 10).',
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)

    if not config.get('experiment_name'):
        print("error: --experiment is required", file=sys.stderr)
        return 2
    if args.sample < 0:
        print("error: --sample must not be negative", file=sys.stderr)
        return 2

    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )
    db_name = config['treatments_db_name']
    if db_name not in server:
        print(f"error: {db_name} not found", file=sys.stderr)
        return 2
    db = server[db_name]

    ids = [r.id for r in db.view('_all_docs') if r.id.startswith('taxon_')]
    if args.sample and args.sample < len(ids):
        ids = random.Random(args.seed).sample(ids, args.sample)

    checks: List[SpanCheck] = []
    for treatment_id in ids:
        checks.extend(check_treatment(db[treatment_id], server))

    total, ok, rate = summarise(checks)
    failures = [c for c in checks if not c.ok]
    print(f"{len(ids)} treatments, {total} spans, {ok} resolved "
          f"({rate:.1f} %)")
    for check in failures[:args.show]:
        print(f"  FAIL {check.treatment_id[:22]} {check.field}"
              f"[{check.index}]: {check.reason}")
    if len(failures) > args.show:
        print(f"  ... and {len(failures) - args.show} more")
    if not total:
        print("error: no spans found to check", file=sys.stderr)
        return 2
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
