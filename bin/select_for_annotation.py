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

The same IDs are ALSO written to
``data/annotation_rounds/<experiment>_round<N>.txt`` (override with
``--output``).  That file is the durable record of what a round
covered — necessary because a selection stops being reproducible
from ``--seed`` as soon as the annotator runs: ``--exclude-annotated``
reads the candidate DB live, so the same command afterwards returns a
different set.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402
from treatments_to_structured.complexity import (  # noqa: E402
    complexity_score,
)
from treatments_to_structured.merge_metric import (  # noqa: E402
    treatment_merge_metric,
)
from treatments_to_structured.select import (  # noqa: E402
    band_report,
    parse_band_spec,
    select_treatments,
)
from treatments_to_structured.status import (  # noqa: E402
    STATUS_SKIPPED_MERGE_SUSPECT,
    make_skip_status_doc,
    status_doc_id,
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
) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
    """Iterate every doc in ``treatments_db``, score each, return
    ``(scored, merge_metrics)`` where:

      * ``scored`` — list of ``(treatment_id, complexity_score)``
        tuples for treatments with score > 0.
      * ``merge_metrics`` — dict mapping treatment_id → merge
        metric value.  Populated in the same doc-read pass so no
        second scan is needed.

    Skips _design docs and any doc whose read raises (transient CouchDB
    errors — the build_sources_stats convention).  Treatments scoring
    0 (no description / diagnosis prose) are filtered out: they can't
    be annotated and would only crowd the low band.
    """
    scored: List[Tuple[str, float]] = []
    merge_metrics: Dict[str, int] = {}
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
            merge_metrics[doc_id] = treatment_merge_metric(doc)
    if verbosity >= 1:
        print(
            f"  Scored {count} treatments total; "
            f"{len(scored)} with non-zero score.",
            file=sys.stderr,
        )
    return scored, merge_metrics


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


def fetch_prior_merge_skip_ids(status_db: Any) -> Set[str]:
    """Return the set of treatment IDs previously flagged as
    suspected merges (``status == skipped_merge_suspect``).

    Used by ``--exclude-suspected-merges`` in its default
    (non-``--force``) mode: skip whatever was previously flagged
    without recomputing the metric, so re-runs converge quickly.
    ``--force`` bypasses this and recomputes fresh.
    """
    ids: Set[str] = set()
    for row in status_db.view('_all_docs', include_docs=True).rows:
        if row.doc is None:
            continue
        if row.doc.get('status') == STATUS_SKIPPED_MERGE_SUSPECT:
            ids.add(row.doc['treatment_id'])
    return ids


def apply_merge_filter(
    scored: List[Tuple[str, float]],
    merge_metrics: Dict[str, int],
    threshold: int,
    already_flagged: Set[str],
) -> Tuple[
    List[Tuple[str, float]],
    List[Tuple[str, int]],
]:
    """Filter suspected-merge treatments out of the scored pool.

    Args:
        scored: The scored population to filter.
        merge_metrics: id → metric-value map (from
            ``score_treatments_in_db``).
        threshold: Metric value at or above which a treatment is
            flagged as a suspected merge.
        already_flagged: IDs previously written to the status DB
            as ``STATUS_SKIPPED_MERGE_SUSPECT`` — filtered out
            without recomputation.  Pass an empty set to force
            re-evaluation of every treatment (the ``--force``
            path).

    Returns:
        ``(surviving, newly_flagged)`` where:
          * ``surviving`` — scored entries that survived the
            filter (kept the same tuple shape so banding /
            selection still works).
          * ``newly_flagged`` — list of ``(treatment_id,
            metric_value)`` for treatments the caller must
            write skip status docs for.  Excludes
            already_flagged (those have status docs already).
    """
    surviving: List[Tuple[str, float]] = []
    newly_flagged: List[Tuple[str, int]] = []
    for tid, score in scored:
        if tid in already_flagged:
            # Was previously flagged; drop from pool without
            # rewriting its status doc.
            continue
        metric_value = merge_metrics.get(tid, 0)
        if metric_value >= threshold:
            newly_flagged.append((tid, metric_value))
            continue
        surviving.append((tid, score))
    return surviving, newly_flagged


_SELECTIONS_DIR = (
    Path(__file__).resolve().parent.parent / 'data' / 'annotation_rounds'
)


def default_output_path(experiment: str, base_dir: Path) -> Path:
    """Next unused ``{experiment}_round{N}.txt`` under ``base_dir``.

    A selection is the only durable record of which treatments a
    round covered, and it stops being reproducible from ``--seed``
    the moment the annotator runs: ``--exclude-annotated`` reads the
    candidate DB live, so re-running the same command afterwards
    yields a different set.  Hence a default path rather than
    relying on the operator to redirect stdout.

    Numbering is per-experiment and continues past the highest file
    present, rather than filling the lowest gap.  Round numbers are
    sequential history, not free slots: rounds 1-3 of production_v4
    happened before selections were kept as files, so gap-filling
    would have numbered the run after round4 as round1.  A missing
    file means that round's selection wasn't captured, not that the
    number is available.

    Never returns a path that already exists, so a past round cannot
    be clobbered.
    """
    existing = [
        int(m.group(1))
        for p in base_dir.glob(f'{experiment}_round*.txt')
        if (m := re.fullmatch(
            rf'{re.escape(experiment)}_round(\d+)\.txt', p.name))
    ]
    return base_dir / f'{experiment}_round{max(existing, default=0) + 1}.txt'


def write_selection(treatment_ids: List[str], path: Path) -> None:
    """Write one treatment ID per line, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for treatment_id in treatment_ids:
            f.write(f'{treatment_id}\n')


def resolve_seed(seed: Optional[int]) -> Tuple[int, bool]:
    """Return ``(seed, was_generated)``, never a missing seed.

    ``--seed`` defaults to ``None``, and the old behaviour was a bare
    ``random.Random()`` seeded from OS entropy — leaving the draw
    unreproducible *in principle*, with nothing to record.  A sidecar
    saying ``"seed": null`` documents an irretrievable gap rather than
    preventing one, so a seed is generated when none is given.

    ``--seed`` therefore means "pin this value", not "switch on
    reproducibility".

    Note a recorded seed alone does **not** reproduce a draw:
    ``--exclude-annotated`` reads the candidate DB live, so the
    surviving population moves as soon as the annotator runs.  Seed
    *and* funnel are both needed, which is why the sidecar carries
    both, and why the round file stays the only durable record of
    actual membership.
    """
    if seed is not None:
        return seed, False
    # SystemRandom so a generated seed does not depend on any global
    # random state a caller may have set.  2**31 keeps it small enough
    # to retype from a log without transcription errors.
    return random.SystemRandom().randrange(2 ** 31), True


def build_round_metadata(
    *,
    experiment: str,
    seed: int,
    seed_generated: bool,
    n_requested: int,
    n_selected: int,
    band_specs: List[Tuple[str, int]],
    band_rows: List[Dict[str, Any]],
    funnel: List[Dict[str, Any]],
    merge_threshold: int,
    force_recompute: bool,
    selector_argv: List[str],
    drawn_at: str,
) -> Dict[str, Any]:
    """Build the round sidecar: how this selection was actually made.

    Bias is **not a boolean and not a hand-typed label**.  Every stage
    of the funnel biases the draw — the always-on complexity filter,
    ``--exclude-suspected-merges`` (default on), ``--exclude-annotated``,
    and ``--bands`` — so "random" only ever means *uniform over the
    survivors*.  The honest record is the funnel itself, and
    ``selection`` is derived from ``band_specs`` rather than asserted.

    ``output_order`` is load-bearing, not decoration:
    ``select_treatments`` emits band-by-band, so on a stratified round
    the first N lines come entirely from the first band — a maximally
    biased subset that looks like an innocent ``head -N``.

    Everything returned must be JSON-serialisable; band specs are
    tuples in memory and lists on disk.
    """
    stratified = len(band_specs) > 1
    return {
        'experiment': experiment,
        'selector_argv': list(selector_argv),
        'seed': seed,
        'seed_generated': seed_generated,
        'n_requested': n_requested,
        'n_selected': n_selected,
        'selection': 'stratified' if stratified else 'uniform',
        'output_order': 'band-by-band' if stratified else 'uniform',
        'bands': (
            [[name, quota] for name, quota in band_specs]
            if stratified else None
        ),
        'band_slices': list(band_rows),
        'population_funnel': list(funnel),
        'merge_threshold': merge_threshold,
        'force_recompute': force_recompute,
        'drawn_at': drawn_at,
    }


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
        '--output', default=None, metavar='PATH',
        help=(
            'Write the selection here, one treatment ID per line.  '
            'Default: data/annotation_rounds/'
            '<experiment>_round<N>.txt with the next free N.  IDs '
            'are still written to stdout as well, so the pipe into '
            'bin/llm_annotate_features keeps working.  The file is '
            'the durable record of what a round covered: once the '
            'annotator has run, --exclude-annotated makes the same '
            '--seed produce a different selection.'
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
    # Multi-species-merge filter.  Default ON: hand-inspected
    # merges get flagged via features_status without leaking into
    # bootstrap runs.  --no-exclude-suspected-merges to bypass.
    parser.add_argument(
        '--exclude-suspected-merges',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop treatments whose merge-detection metric "
            "(treatments_to_structured.merge_metric."
            "n_terms_above_k) exceeds --merge-threshold, and "
            "write features_status skip docs recording the "
            "decision.  Default: on.  Also honors prior skip "
            "decisions from the status DB unless --force is "
            "passed (see --force behavior)."
        ),
    )
    parser.add_argument(
        '--merge-threshold', type=int, default=10, metavar='N',
        help=(
            'Treatments whose n_terms_above_5 metric is >= this '
            'value are flagged as suspected merges when '
            '--exclude-suspected-merges is on.  Default 10, '
            'calibrated 2026-07-01 on the 56-treatment sample.  '
            'Higher threshold = fewer flagged = more permissive.'
        ),
    )
    # --verbosity, --force, --experiment are provided by
    # common_parser().  --force here means: recompute the merge
    # metric fresh instead of trusting cached decisions from
    # features_status.  Useful after tweaking --merge-threshold.
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

    dry_run = bool(config.get('dry_run', False))
    if dry_run and verbosity >= 1:
        print("[dry-run] nothing will be written", file=sys.stderr)

    total_docs = treatments_db.info()['doc_count']
    scored, merge_metrics = score_treatments_in_db(
        treatments_db, verbosity,
    )
    # Every filter stage is a bias source, so the honest record of how
    # a round was drawn is the funnel, not a 'random'/'biased' label.
    funnel: List[Dict[str, Any]] = [
        {'stage': 'all_treatments', 'n': total_docs},
        {'stage': 'complexity_gt_0', 'n': len(scored)},
    ]
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
            funnel.append(
                {'stage': 'not_already_annotated', 'n': len(scored)}
            )
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

    if args.exclude_suspected_merges:
        experiment = config.get('experiment_name')
        if not experiment:
            print(
                "error: --exclude-suspected-merges requires "
                "--experiment (needed to resolve the status DB name)",
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
        # Reuse the status-DB resolver from bin/llm_annotate_features.
        from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
            resolve_status_db_name,
        )
        status_db_name = resolve_status_db_name(
            experiment, exp_doc, verbosity=verbosity,
        )
        # Ensure the status DB exists — we may need to write skip
        # docs even on the first run.
        if status_db_name in server:
            status_db = server[status_db_name]
        elif dry_run:
            # Filtering still works: score_treatments_in_db already
            # returned the merge metrics, so only the persistence of
            # skip decisions is skipped.
            if verbosity >= 1:
                print(
                    f"  [dry-run] would create status DB "
                    f"{status_db_name!r}; no prior skip decisions "
                    f"to trust",
                    file=sys.stderr,
                )
            status_db = None
        else:
            if verbosity >= 1:
                print(
                    f"  Creating status DB {status_db_name!r}",
                    file=sys.stderr,
                )
            status_db = server.create(status_db_name)

        # --force bypasses the "trust cached decisions" path and
        # recomputes the metric fresh.  Without --force, previously
        # flagged treatments are trusted (no re-computation cost).
        force_recompute = bool(config.get('force', False))
        if force_recompute or status_db is None:
            already_flagged: Set[str] = set()
        else:
            already_flagged = fetch_prior_merge_skip_ids(status_db)

        before = len(scored)
        scored, newly_flagged = apply_merge_filter(
            scored, merge_metrics, args.merge_threshold,
            already_flagged=already_flagged,
        )
        funnel.append(
            {'stage': 'not_merge_suspect', 'n': len(scored)}
        )
        # Write skip docs for newly-flagged.  Overwrites any prior
        # doc with fresh metric + threshold + timestamp (matters
        # under --force).
        from datetime import datetime, timezone
        decided_at = datetime.now(timezone.utc).isoformat()
        for tid, metric_value in (
            [] if dry_run else newly_flagged
        ):
            doc = make_skip_status_doc(
                tid,
                metric_value=metric_value,
                threshold=args.merge_threshold,
                metric_name='n_terms_above_5',
                decided_at=decided_at,
            )
            existing_id = status_doc_id(tid)
            if existing_id in status_db:
                doc['_rev'] = status_db[existing_id]['_rev']
            try:
                status_db.save(doc)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  ERROR writing skip doc for {tid}: {exc}",
                    file=sys.stderr,
                )
        if verbosity >= 1:
            print(
                f"  --exclude-suspected-merges "
                f"(threshold={args.merge_threshold}, "
                f"{'--force' if force_recompute else 'trust prior'}): "
                f"dropped {before - len(scored)} treatments "
                f"({len(already_flagged)} previously flagged; "
                f"{len(newly_flagged)} newly flagged; "
                f"{len(scored)} remain to sample from)",
                file=sys.stderr,
            )

    seed, seed_generated = resolve_seed(args.seed)
    if verbosity >= 1 and seed_generated:
        print(
            f"  No --seed given; generated {seed} and recording it "
            f"(an unrecorded seed makes the draw unreproducible).",
            file=sys.stderr,
        )
    rng = random.Random(seed)
    try:
        selected = select_treatments(scored, band_specs, rng)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = default_output_path(
            config.get('experiment_name') or 'selection',
            _SELECTIONS_DIR,
        )
    from datetime import datetime as _dt, timezone as _tz
    meta = build_round_metadata(
        experiment=config.get('experiment_name') or 'selection',
        seed=seed,
        seed_generated=seed_generated,
        n_requested=args.n,
        n_selected=len(selected),
        band_specs=band_specs,
        band_rows=band_report(scored, band_specs),
        funnel=funnel,
        merge_threshold=args.merge_threshold,
        force_recompute=bool(config.get('force', False)),
        selector_argv=list(sys.argv[1:]),
        drawn_at=_dt.now(_tz.utc).isoformat(),
    )
    meta_path = out_path.with_suffix('.meta.json')

    if verbosity >= 1:
        print("  Population funnel:", file=sys.stderr)
        for stage in funnel:
            print(f"    {stage['stage']:<24} {stage['n']:>8}",
                  file=sys.stderr)
        for row in meta['band_slices']:
            print(
                f"    band {row['name']:<12} quota={row['quota']:<6} "
                f"n={row['slice_n']:<8} "
                f"score {row['score_min']:g}..{row['score_max']:g}",
                file=sys.stderr,
            )
        print(f"    seed={seed}"
              f"{' (generated)' if seed_generated else ''}  "
              f"selection={meta['selection']}  "
              f"output_order={meta['output_order']}",
              file=sys.stderr)

    if dry_run:
        if verbosity >= 1:
            print(
                f"  [dry-run] would write {out_path} and {meta_path}",
                file=sys.stderr,
            )
    else:
        write_selection(selected, out_path)
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=False) + '\n',
            encoding='utf-8',
        )
        if verbosity >= 1:
            print(
                f"  Selection written to {out_path}\n"
                f"  Provenance written to {meta_path}",
                file=sys.stderr,
            )

    for treatment_id in selected:
        print(treatment_id)
    return 0


if __name__ == '__main__':
    sys.exit(main())
