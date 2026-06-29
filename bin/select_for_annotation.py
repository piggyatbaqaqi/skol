#!/usr/bin/env python3
"""Sample treatments from a treatments_prose database, banded by complexity score.

Phase 1 deliverable 2 of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.4.

Reads the configured experiment's ``treatments_prose`` database,
scores each treatment with
``treatments_to_structured.complexity.complexity_score``, filters out
treatments with score 0 (those lack the prose we'd annotate), and
selects a random sample of ``--n`` treatment IDs split across
complexity bands.

Usage::

    bin/select_for_annotation --experiment production_v4 --n 100
    bin/select_for_annotation --experiment production_v4 --n 100 \\
        --bands low:25,mid:50,high:25
    bin/select_for_annotation --experiment production_v4 --n 100 \\
        --bands low:25,mid:50,high:25 --seed 42

Output: one treatment ID per line on stdout, suitable for piping
into the bootstrap annotator (Phase 1 deliverable 5):

    bin/select_for_annotation --experiment production_v4 --n 100 \\
        | bin/llm_annotate_features --experiment production_v4
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402
from treatments_to_structured.complexity import (  # noqa: E402
    complexity_score,
)
from treatments_to_structured.select import (  # noqa: E402
    parse_band_spec,
    select_treatments,
)


def _connect_server(config: Dict[str, Any]) -> Any:
    """Connect to CouchDB using the project's resolved config."""
    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'],
        config['couchdb_password'],
    )
    return server


def score_treatments_in_db(
    treatments_db: Any,
    verbosity: int = 1,
) -> List[Tuple[str, float]]:
    """Iterate every doc in ``treatments_db``, score each, return
    ``(treatment_id, score)`` for treatments with score > 0.

    Skips _design docs and any doc whose read raises (transient CouchDB
    errors — the build_sources_stats convention).  Treatments scoring
    0 (no description / diagnosis prose) are filtered out: they can't
    be annotated and would only crowd the low band.
    """
    scored: List[Tuple[str, float]] = []
    count = 0
    for doc_id in treatments_db:
        if doc_id.startswith('_design/'):
            continue
        count += 1
        if verbosity >= 2 and count % 1000 == 0:
            print(
                f"  Scored {count} treatments...",
                file=sys.stderr,
            )
        try:
            doc = treatments_db[doc_id]
        except Exception:
            continue
        score = complexity_score(doc)
        if score > 0:
            scored.append((doc_id, score))
    if verbosity >= 1:
        print(
            f"  Scored {count} treatments total; "
            f"{len(scored)} with non-zero score.",
            file=sys.stderr,
        )
    return scored


def fetch_annotated_treatment_ids(candidate_db: Any) -> Set[str]:
    """Return the set of treatment IDs that already have at least
    one annotation in ``candidate_db``.

    Annotation _ids are ``<treatment_id>:<feature_label>:<offset>``.
    A single ``_all_docs`` scan reads every key; we take the prefix
    before the first ``:`` as the treatment_id.  Cheap on the
    hundreds-of-treatments scale Phase 1 + Phase 2 are targeting;
    if this ever scales past tens of thousands, swap to a view.
    """
    ids: Set[str] = set()
    for row in candidate_db.view('_all_docs').rows:
        key = row.id if hasattr(row, 'id') else row.get('id', '')
        if not key or ':' not in key or key.startswith('_design/'):
            continue
        ids.add(key.split(':', 1)[0])
    return ids


def filter_excluded(
    scored: List[Tuple[str, float]],
    excluded_ids: Set[str],
) -> List[Tuple[str, float]]:
    """Drop any scored entry whose treatment_id is in
    ``excluded_ids``.  Preserves the input order of survivors so the
    downstream banding sees the same complexity distribution shape."""
    if not excluded_ids:
        return scored
    return [(tid, sc) for tid, sc in scored if tid not in excluded_ids]


def _resolve_band_specs(
    bands_flag: str,
    n: int,
) -> List[Tuple[str, int]]:
    """Resolve ``--bands`` flag into ``[(name, count), ...]``.

    Empty string / None → single ``('all', n)`` band (no banding).
    Otherwise parse the spec and validate quotas sum to ``n``.
    """
    if not bands_flag:
        return [('all', n)]
    band_specs = parse_band_spec(bands_flag)
    total_quota = sum(count for _, count in band_specs)
    if total_quota != n:
        raise ValueError(
            f"band quotas sum to {total_quota}, but --n is {n}"
        )
    return band_specs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--n', type=int, required=True,
        help='Total number of treatments to select.',
    )
    parser.add_argument(
        '--bands', default=None,
        help=(
            "Comma-separated '<name>:<count>' band quotas (e.g., "
            "'low:25,mid:50,high:25').  Counts must sum to --n.  "
            "Population is sorted by complexity score and split into "
            "equal-size slices, one per band; each band samples its "
            "quota at random from its slice.  Default: single 'all' "
            "band (no banding)."
        ),
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help=(
            "Random seed for reproducibility.  Omit for "
            "nondeterministic sampling."
        ),
    )
    parser.add_argument(
        '--exclude-annotated', action='store_true',
        help=(
            'Skip treatments that already have annotations in the '
            "experiment's features_candidate DB.  Use when extending "
            'an existing selection — guarantees the --n count is N '
            'NEW treatments, not N including repeats of work in '
            'progress.  Requires --experiment (so the candidate DB '
            'can be resolved).'
        ),
    )
    # --verbosity is provided by common_parser() — don't re-declare.
    args = parser.parse_args()

    try:
        band_specs = _resolve_band_specs(args.bands, args.n)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = get_env_config(cli_args=args)
    # common_parser declares --verbosity without an argparse default,
    # so verbosity is None when the flag isn't passed.
    # get_env_config defaults to 1 via _get_env('VERBOSITY', '1').
    # Use the config-resolved value (NOT verbosity) so the
    # "no flag" case doesn't TypeError on `verbosity >= 1`.
    verbosity = int(config.get('verbosity', 1) or 0)
    treatments_db_name = (
        config.get('treatments_prose_db_name')
        or config.get('treatments_db_name')
    )
    if not treatments_db_name:
        print(
            "error: could not resolve treatments_prose DB name.  "
            "Pass --experiment <NAME> (preferred) or set "
            "TREATMENTS_PROSE_DB_NAME.",
            file=sys.stderr,
        )
        return 2

    server = _connect_server(config)
    if treatments_db_name not in server:
        print(
            f"error: treatments DB {treatments_db_name!r} not found "
            "on the server.",
            file=sys.stderr,
        )
        return 2

    treatments_db = server[treatments_db_name]
    if verbosity >= 1:
        print(
            f"Scanning {treatments_db_name}...",
            file=sys.stderr,
        )

    scored = score_treatments_in_db(treatments_db, verbosity)
    if not scored:
        print(
            "error: no treatments with non-zero complexity score were "
            "found.  Either the database is empty or every treatment "
            "has null/empty description and diagnosis.",
            file=sys.stderr,
        )
        return 2

    if args.exclude_annotated:
        experiment = config.get('experiment_name')
        if not experiment:
            print(
                "error: --exclude-annotated requires --experiment "
                "(needed to resolve the candidate DB name)",
                file=sys.stderr,
            )
            return 2
        try:
            exp_doc = server['skol_experiments'][experiment]
        except Exception:
            print(
                f"error: experiment {experiment!r} not found in "
                f"skol_experiments",
                file=sys.stderr,
            )
            return 2
        # Reuse the candidate-DB resolver from bin/llm_annotate_features
        # so the fallback naming convention stays in one place.
        from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
            resolve_candidate_db_name,
        )
        candidate_db_name = resolve_candidate_db_name(
            experiment, exp_doc, verbosity=verbosity,
        )
        if candidate_db_name in server:
            excluded_ids = fetch_annotated_treatment_ids(
                server[candidate_db_name],
            )
            before = len(scored)
            scored = filter_excluded(scored, excluded_ids)
            if verbosity >= 1:
                print(
                    f"  --exclude-annotated: dropped "
                    f"{before - len(scored)} already-annotated "
                    f"treatments from the selection pool "
                    f"({len(excluded_ids)} in candidate DB; "
                    f"{len(scored)} remain to sample from)",
                    file=sys.stderr,
                )
        elif verbosity >= 1:
            print(
                f"  --exclude-annotated: candidate DB "
                f"{candidate_db_name!r} does not exist yet; "
                f"no exclusions applied",
                file=sys.stderr,
            )

    rng = (
        random.Random(args.seed)
        if args.seed is not None
        else random.Random()
    )
    try:
        selected = select_treatments(scored, band_specs, rng)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for treatment_id in selected:
        print(treatment_id)
    return 0


if __name__ == '__main__':
    sys.exit(main())
