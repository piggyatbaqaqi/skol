#!/usr/bin/env python3
"""Export an experiment's candidate (or hand) annotations to a
directory of brat collection files for interactive review.

Phase 1 deliverable 6 companion of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.4.

For each treatment selected, writes three things to the output
directory:

  ``<treatment_id>.txt``   the synthetic brat document (rendered
                           via ``brat_render.render`` against the
                           treatment in the treatments_prose DB).
  ``<treatment_id>.ann``   the annotations as brat T-lines
                           (rendered via ``annotations_to_brat``).
  ``annotation.conf``      brat collection config listing every
                           entity type encountered, so brat opens
                           the directory cleanly without manual
                           setup.

By default exports from the candidate DB
(``skol_exp_<exp>_02_50_features_candidate``); pass
``--source hand`` to export from the hand-reviewed DB
(``skol_exp_<exp>_02_55_features_hand``) for re-review or
inspection of prior-reviewed work.

Filename convention is ``<treatment_id>.ann`` to match the
input format ``bin/brat_ingest`` expects.

Usage::

    # Export every treatment the bootstrap annotator has annotated:
    bin/brat_export --experiment production_v4 \\
        --output-dir /home/operator/brat_review/production_v4/

    # Restrict to specific treatments:
    bin/brat_export --experiment production_v4 \\
        --output-dir /tmp/review/ \\
        --doc-id taxon_841d5cbed,taxon_2114314b

    # Re-export the hand-reviewed state (after one round of review):
    bin/brat_export --experiment production_v4 \\
        --output-dir /tmp/review_round_2/ --source hand
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402

from treatments_to_structured.brat_render import (  # noqa: E402
    annotations_to_brat,
    render,
)


# ---------------------------------------------------------------------------
# Testable helpers
# ---------------------------------------------------------------------------


def list_annotated_treatment_ids(
    annotations_db: Any,
) -> List[str]:
    """Enumerate distinct treatment IDs that have at least one
    annotation in the given DB (candidate or hand).

    The annotation ``_id`` format is
    ``<treatment_id>:<feature_label>:<start>``; a single
    ``_all_docs`` scan reads every key and the prefix before the
    first ``:`` is the treatment ID.  Cheap for hundreds of
    treatments; if this ever scales past tens of thousands, swap
    to a CouchDB view.

    Returns sorted IDs (stable, so re-exports are diff-friendly).
    """
    rows = annotations_db.view('_all_docs').rows
    ids: Set[str] = set()
    for row in rows:
        key = row.id if hasattr(row, 'id') else row.get('id', '')
        if not key or ':' not in key:
            continue
        tid = key.split(':', 1)[0]
        ids.add(tid)
    return sorted(ids)


def fetch_reviewer_touched_ids(status_db: Any) -> Set[str]:
    """Return the set of treatment IDs whose status docs carry a
    ``reviewer_action`` sub-doc — i.e., the reviewer opened the
    treatment in brat and hit save, regardless of whether they
    kept any annotations.

    Written by ``bin/brat_ingest`` since 2026-07-01.  Fills the
    gap that ``list_annotated_treatment_ids`` on features_hand
    can't: treatments where the reviewer rejected every
    annotation leave features_hand empty for that treatment_id,
    so a hand-DB scan alone can't distinguish 'reviewed and
    rejected' from 'not yet reviewed'.  The status doc's
    ``reviewer_action`` field makes the distinction explicit.
    """
    ids: Set[str] = set()
    rows = status_db.view('_all_docs', include_docs=True).rows
    for row in rows:
        if row.doc is None:
            continue
        if 'reviewer_action' in row.doc:
            tid = row.doc.get('treatment_id') or row.id
            ids.add(tid)
    return ids


def fetch_anns_for_treatment(
    annotations_db: Any,
    treatment_id: str,
) -> List[Dict[str, Any]]:
    """Fetch every annotation doc for one treatment.

    Same prefix-range scan as bin/brat_ingest.fetch_candidate_anns_
    for_treatment — kept duplicated here rather than imported so
    brat_export is a self-contained tool (and so the import graph
    doesn't pull bin/brat_ingest's couchdb / argparse footprint
    into brat_export's tests).
    """
    prefix = f'{treatment_id}:'
    rows = annotations_db.view(
        '_all_docs',
        startkey=prefix,
        endkey=prefix + '￰',
        include_docs=True,
    ).rows
    return [dict(r.doc) for r in rows if r.doc is not None]


def collect_entity_types(
    annotations_by_treatment: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Collect the distinct entity types (= feature labels with
    spaces normalized to underscores) appearing in the export, so
    we can write a brat ``annotation.conf`` that lets brat open
    the directory without manual config.

    Returns a sorted list so the conf file is deterministic
    (operator can diff configs across exports).
    """
    types: Set[str] = set()
    for anns in annotations_by_treatment.values():
        for ann in anns:
            label = ann.get('feature_label')
            if not label:
                continue
            types.add(label.replace(' ', '_'))
    return sorted(types)


def render_annotation_conf(entity_types: List[str]) -> str:
    """Build the contents of a brat ``annotation.conf`` file.

    Emits all four sections brat looks for —
    ``[entities]`` (populated with observed feature_labels),
    plus empty ``[relations]``, ``[events]``, ``[attributes]``.
    The empty sections silence brat's "missing section, config
    may be wrong" startup warnings; Phase 1 only uses entities,
    but a partial config is read as "incomplete" rather than
    "intentionally entity-only".

    The header comment captures provenance so the operator knows
    which entity types came from which export run.
    """
    lines = [
        '# brat annotation config — generated by bin/brat_export',
        '# Lists every feature_label observed in the exported',
        '# annotations.  Add new types here if the reviewer needs',
        '# them; bin/brat_ingest will pick them up on the next',
        '# ingest pass.',
        '',
        '[entities]',
    ]
    if entity_types:
        lines.extend(entity_types)
    else:
        lines.append('# (no entity types found in source DB)')
    # Empty sections so brat doesn't warn about partial config.
    # Phase 1 doesn't use relations / events / attributes; future
    # phases can extend.
    lines.extend([
        '',
        '[relations]',
        '',
        '[events]',
        '',
        '[attributes]',
    ])
    return '\n'.join(lines) + '\n'


def select_treatment_ids(
    annotations_db: Any,
    doc_id_filter: Optional[List[str]],
) -> List[str]:
    """Decide which treatment IDs to export.

    Without ``--doc-id``: every treatment that has any annotation
    in the source DB.  With ``--doc-id``: just those IDs (raises
    if any of them have no annotations, so the operator catches
    typos immediately).
    """
    all_ids = list_annotated_treatment_ids(annotations_db)
    if doc_id_filter is None:
        if not all_ids:
            raise ValueError(
                "source DB has no annotations; nothing to export"
            )
        return all_ids
    all_set = set(all_ids)
    missing = [t for t in doc_id_filter if t not in all_set]
    if missing:
        raise ValueError(
            f"--doc-id {missing!r} have no annotations in the "
            f"source DB"
        )
    return [t for t in doc_id_filter]


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output-dir', required=True, metavar='DIR',
        help=(
            'Directory to write .txt / .ann / annotation.conf '
            'files into.  Created if it does not exist.'
        ),
    )
    parser.add_argument(
        '--source', default='candidate',
        choices=('candidate', 'hand'),
        help=(
            "Which DB to export from.  'candidate' "
            '(default) reads the bootstrap output '
            '(features_candidate).  '
            "'hand' reads the human-reviewed DB "
            '(features_hand) — useful for re-review or '
            'inspection of prior-reviewed work.'
        ),
    )
    parser.add_argument(
        '--exclude-reviewed', action='store_true',
        help=(
            'Skip treatments that already have annotations in '
            'features_hand.  Use when extending an existing '
            'review batch — the exported directory then contains '
            'only treatments still needing review, so opening it '
            'in brat gives a clean queue with no already-done '
            'work to skip past.  No-op when --source hand '
            '(everything would be excluded).'
        ),
    )
    args = parser.parse_args()

    config = get_env_config(cli_args=args)
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    force = bool(config.get('force', False))
    experiment = config.get('experiment_name')
    if not experiment:
        print("error: --experiment is required", file=sys.stderr)
        return 2

    doc_id_filter = config.get('doc_ids') or None

    # CouchDB
    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )

    # treatments_prose DB (always read; provides the synth doc text)
    treatments_db_name = (
        config.get('treatments_prose_db_name')
        or config.get('treatments_db_name')
    )
    if not treatments_db_name or treatments_db_name not in server:
        print(
            f"error: treatments_prose DB {treatments_db_name!r} "
            f"not found on the server",
            file=sys.stderr,
        )
        return 2
    treatments_db = server[treatments_db_name]

    # Resolve source DB (candidate or hand)
    try:
        exp_doc = server['skol_experiments'][experiment]
    except Exception:
        print(
            f"error: experiment {experiment!r} not found in "
            f"skol_experiments",
            file=sys.stderr,
        )
        return 2

    if args.source == 'candidate':
        # Resolve via the same helper bin/llm_annotate_features
        # uses, so the fallback naming convention stays in one
        # place.
        from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
            resolve_candidate_db_name,
        )
        source_db_name = resolve_candidate_db_name(
            experiment, exp_doc, verbosity=verbosity,
        )
    else:  # 'hand'
        from brat_ingest import (  # type: ignore[import]  # noqa: E402
            resolve_hand_db_name,
        )
        source_db_name = resolve_hand_db_name(
            experiment, exp_doc, verbosity=verbosity,
        )
    if source_db_name not in server:
        print(
            f"error: source DB {source_db_name!r} not found "
            f"(run bin/llm_annotate_features first, or pass "
            f"--source candidate if you meant the bootstrap "
            f"output)",
            file=sys.stderr,
        )
        return 2
    source_db = server[source_db_name]

    # Treatment selection
    try:
        treatment_ids = select_treatment_ids(
            source_db, doc_id_filter,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Optional exclusion: drop treatments already reviewed.
    # 'Reviewed' comes from two sources:
    #   1. features_hand — treatments with any kept/added annotation
    #   2. features_status.reviewer_action — treatments the reviewer
    #      touched even if all Claude annotations were rejected
    #      (features_hand has zero entries for them).  Written by
    #      bin/brat_ingest since 2026-07-01.
    # Union of the two = complete "already reviewed" set.
    # No-op when --source is 'hand' (would exclude everything).
    if args.exclude_reviewed and args.source != 'hand':
        from brat_ingest import (  # type: ignore[import]  # noqa: E402
            resolve_hand_db_name,
        )
        from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
            resolve_status_db_name,
        )
        hand_db_name = resolve_hand_db_name(
            experiment, exp_doc, verbosity=verbosity,
        )
        status_db_name = resolve_status_db_name(
            experiment, exp_doc, verbosity=verbosity,
        )
        reviewed_ids: Set[str] = set()
        if hand_db_name in server:
            reviewed_ids |= list_annotated_treatment_ids(
                server[hand_db_name],
            )
        if status_db_name in server:
            reviewed_ids |= fetch_reviewer_touched_ids(
                server[status_db_name],
            )
        before = len(treatment_ids)
        treatment_ids = [
            tid for tid in treatment_ids
            if tid not in reviewed_ids
        ]
        if verbosity >= 1:
            n_dropped = before - len(treatment_ids)
            if reviewed_ids:
                print(
                    f"  --exclude-reviewed: dropped "
                    f"{n_dropped} treatments "
                    f"({len(reviewed_ids)} reviewed via "
                    f"features_hand + features_status; "
                    f"{len(treatment_ids)} remain to export)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  --exclude-reviewed: no prior reviewed "
                    f"treatments found (features_hand + "
                    f"features_status both empty); no exclusions",
                    file=sys.stderr,
                )

    if not treatment_ids:
        print(
            "No treatments to export after filters applied.",
            file=sys.stderr,
        )
        return 0

    if verbosity >= 1:
        print(
            f"Exporting {len(treatment_ids)} treatments from "
            f"{source_db_name} → {args.output_dir!r}",
            file=sys.stderr,
        )

    # Create output dir
    if not dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    # Per-treatment fetch + render + write
    annotations_by_treatment: Dict[str, List[Dict[str, Any]]] = {}
    errors = 0
    written = 0
    skipped = 0
    for tid in treatment_ids:
        try:
            treatment = dict(treatments_db[tid])
        except Exception:  # noqa: BLE001
            print(
                f"  ERROR {tid}: not found in {treatments_db_name}",
                file=sys.stderr,
            )
            errors += 1
            continue

        synth_txt, span_map = render(treatment)
        if not synth_txt:
            print(
                f"  WARN {tid}: empty synth doc (no description / "
                f"diagnosis); skipping",
                file=sys.stderr,
            )
            skipped += 1
            continue

        anns = fetch_anns_for_treatment(source_db, tid)
        annotations_by_treatment[tid] = anns

        try:
            ann_text = annotations_to_brat(anns, span_map)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  ERROR {tid}: annotations_to_brat: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            errors += 1
            continue

        txt_path = os.path.join(args.output_dir, f'{tid}.txt')
        ann_path = os.path.join(args.output_dir, f'{tid}.ann')

        # Existing-file guard (default refuses to overwrite so
        # operator review-in-progress doesn't get clobbered).
        if not force and not dry_run:
            existing = [
                p for p in (txt_path, ann_path)
                if os.path.exists(p)
            ]
            if existing:
                print(
                    f"  SKIP {tid}: output files already exist "
                    f"({existing!r}); pass --force to overwrite",
                    file=sys.stderr,
                )
                skipped += 1
                continue

        if dry_run:
            if verbosity >= 1:
                print(
                    f"  [dry-run] {tid}: would write "
                    f"{len(synth_txt)}-char .txt + "
                    f"{len(anns)} annotation(s)",
                    file=sys.stderr,
                )
            written += 1
            continue

        with open(txt_path, 'w') as f:
            f.write(synth_txt)
        with open(ann_path, 'w') as f:
            f.write(ann_text)
        if verbosity >= 1:
            print(
                f"  {tid}: {len(anns)} annotation(s)",
                file=sys.stderr,
            )
        written += 1

    # Write annotation.conf
    entity_types = collect_entity_types(annotations_by_treatment)
    conf_path = os.path.join(args.output_dir, 'annotation.conf')
    if not dry_run:
        with open(conf_path, 'w') as f:
            f.write(render_annotation_conf(entity_types))

    # End-of-run summary
    print(
        f"\nDone: {written} written, {skipped} skipped, "
        f"{errors} errors  ({len(entity_types)} distinct "
        f"entity types in annotation.conf)"
    )
    if written > 0 and not dry_run:
        print(f"Open in brat at:  {args.output_dir}")
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
