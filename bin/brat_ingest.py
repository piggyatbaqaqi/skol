#!/usr/bin/env python3
"""Read a reviewer's edited brat .ann files and write the diff
result to the experiment's features_hand DB.

Phase 1 deliverable 6 of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.4.

For each treatment whose .ann file is processed:
  1. Re-render the synthetic brat .txt from the Treatment in the
     treatments_prose DB (gets us the SpanMap for offset
     translation).
  2. Parse the reviewer's .ann file
     (treatments_to_structured.brat_render.parse_brat_ann) into
     annotation dicts in the durable storage shape.
  3. Diff against the candidate-DB annotations for the same
     treatment (see treatments_to_structured.brat_ingest).
  4. Write `kept` and `added` annotations to the features_hand
     DB with reviewer metadata (reviewer, reviewed_at,
     reviewer_action) attached.  `deleted` annotations are
     simply absent from the hand DB.

Filename convention: brat .ann file is named ``<treatment_id>.ann``.
The CLI accepts either ``--ann-dir DIR`` (scan for taxon_*.ann) or
``--ann-file PATH``.

Usage::

    # Process all reviewed .ann files in a directory:
    bin/brat_ingest --experiment production_v4 \\
        --ann-dir /home/operator/brat_review/production_v4/

    # Single file:
    bin/brat_ingest --experiment production_v4 \\
        --ann-file /tmp/taxon_841d5cbed.ann

    # Restrict --ann-dir to specific treatment IDs:
    bin/brat_ingest --experiment production_v4 \\
        --ann-dir /home/operator/brat_review/production_v4/ \\
        --doc-id taxon_841d5cbed,taxon_2114314b
"""

import argparse
import getpass
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402

from treatments_to_structured.brat_ingest import (  # noqa: E402
    DiffResult,
    diff_annotations,
    make_reviewed_doc,
    treatment_id_from_ann_filename,
)
from treatments_to_structured.brat_render import (  # noqa: E402
    parse_brat_ann,
    render,
)
from treatments_to_structured.status import (  # noqa: E402
    status_doc_id,
)


# ---------------------------------------------------------------------------
# Testable helpers
# ---------------------------------------------------------------------------


def resolve_hand_db_name(
    experiment_name: str,
    experiment_doc: Dict[str, Any],
    *,
    verbosity: int = 1,
    warn_stream: TextIO = sys.stderr,
) -> str:
    """Find the hand DB name for an experiment.

    Mirrors ``resolve_candidate_db_name`` / ``resolve_status_db_name``
    in bin/llm_annotate_features.py.  Prefers
    ``experiment.databases.features_hand``; falls back to
    ``skol_exp_<name>_02_55_features_hand`` with a one-line
    warning.
    """
    dbs = experiment_doc.get('databases') or {}
    explicit = dbs.get('features_hand')
    if explicit:
        return explicit
    fallback = (
        f'skol_exp_{experiment_name}_02_55_features_hand'
    )
    if verbosity >= 1:
        print(
            f"NOTE: experiment.databases.features_hand not set; "
            f"using naming-convention fallback {fallback!r}.  Run "
            f"`bin/manage_experiment update {experiment_name}` to "
            f"make this canonical.",
            file=warn_stream,
        )
    return fallback


def discover_ann_files(
    ann_dir: Optional[str],
    ann_file: Optional[str],
    doc_id_filter: Optional[List[str]],
) -> List[Tuple[str, str]]:
    """Resolve the list of (treatment_id, ann_file_path) pairs to
    process.

    Args:
        ann_dir: Directory to scan for ``taxon_*.ann`` files
            (mutually exclusive with ann_file).
        ann_file: Single .ann file path.
        doc_id_filter: Restrict --ann-dir scan to these treatment
            IDs (no-op if ann_dir is None).

    Raises ValueError on missing source, conflicting source, or
    empty result.
    """
    if ann_dir is None and ann_file is None:
        raise ValueError(
            "must specify either --ann-dir or --ann-file"
        )
    if ann_dir is not None and ann_file is not None:
        raise ValueError(
            "--ann-dir and --ann-file are mutually exclusive"
        )
    if ann_file is not None:
        if not os.path.isfile(ann_file):
            raise ValueError(f"--ann-file not found: {ann_file!r}")
        tid = treatment_id_from_ann_filename(ann_file)
        return [(tid, ann_file)]
    # ann_dir mode — narrow type after the early returns above.
    assert ann_dir is not None
    if not os.path.isdir(ann_dir):
        raise ValueError(
            f"--ann-dir not a directory: {ann_dir!r}"
        )
    pairs: List[Tuple[str, str]] = []
    for entry in sorted(os.listdir(ann_dir)):
        if not entry.endswith('.ann'):
            continue
        path = os.path.join(ann_dir, entry)
        if not os.path.isfile(path):
            continue
        tid = treatment_id_from_ann_filename(path)
        if doc_id_filter and tid not in doc_id_filter:
            continue
        pairs.append((tid, path))
    if not pairs:
        if doc_id_filter:
            raise ValueError(
                f"no .ann files in {ann_dir!r} matched --doc-id "
                f"filter {doc_id_filter!r}"
            )
        raise ValueError(
            f"no .ann files found in {ann_dir!r}"
        )
    return pairs


def fetch_candidate_anns_for_treatment(
    candidate_db: Any,
    treatment_id: str,
) -> List[Dict[str, Any]]:
    """Fetch every candidate annotation for one treatment.

    Annotation docs are keyed
    ``<treatment_id>:<feature_label>:<offset>``, so a single
    ``_all_docs`` range query with the treatment-id prefix gets
    them all.  Returns the full doc dicts (so the diff sees
    ``_id`` / ``_rev`` / ``model`` / ``created_at``).
    """
    prefix = f'{treatment_id}:'
    rows = candidate_db.view(
        '_all_docs',
        startkey=prefix,
        endkey=prefix + '￰',
        include_docs=True,
    ).rows
    return [dict(r.doc) for r in rows if r.doc is not None]


def reviewer_label_from_env() -> str:
    """Derive a default reviewer identifier when --reviewer isn't
    passed.  Format: ``<user>@<host>``."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = os.environ.get('USER', 'unknown')
    try:
        host = socket.gethostname()
    except Exception:  # noqa: BLE001
        host = 'unknown'
    return f'{user}@{host}'


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
        '--ann-dir', default=None, metavar='DIR',
        help=(
            'Directory of brat .ann files.  Each file is named '
            '<treatment_id>.ann.  Mutually exclusive with --ann-file.'
        ),
    )
    parser.add_argument(
        '--ann-file', default=None, metavar='PATH',
        help=(
            'Single brat .ann file.  Treatment ID derived from '
            'filename.  Mutually exclusive with --ann-dir.'
        ),
    )
    parser.add_argument(
        '--reviewer', default=None, metavar='NAME',
        help=(
            'Free-form reviewer identifier stored on each '
            'reviewed annotation.  Defaults to <user>@<host>.'
        ),
    )
    args = parser.parse_args()

    config = get_env_config(cli_args=args)
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    experiment = config.get('experiment_name')
    if not experiment:
        print("error: --experiment is required", file=sys.stderr)
        return 2

    # Resolve source-file selection
    doc_id_filter = config.get('doc_ids') or None
    try:
        ann_pairs = discover_ann_files(
            args.ann_dir, args.ann_file, doc_id_filter,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    reviewer = args.reviewer or reviewer_label_from_env()
    if verbosity >= 1:
        print(
            f"Reviewer: {reviewer}  "
            f"({len(ann_pairs)} .ann file(s) to ingest)",
            file=sys.stderr,
        )

    # CouchDB
    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )

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

    try:
        exp_doc = server['skol_experiments'][experiment]
    except Exception:
        print(
            f"error: experiment {experiment!r} not found in "
            f"skol_experiments",
            file=sys.stderr,
        )
        return 2

    # Resolve candidate + reviewed DBs.  Candidate must exist (we
    # diff against it); reviewed is created on demand.
    from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
        resolve_candidate_db_name,
    )
    candidate_db_name = resolve_candidate_db_name(
        experiment, exp_doc, verbosity=verbosity,
    )
    if candidate_db_name not in server:
        print(
            f"error: candidate DB {candidate_db_name!r} not found "
            f"— run bin/llm_annotate_features first",
            file=sys.stderr,
        )
        return 2
    candidate_db = server[candidate_db_name]

    hand_db_name = resolve_hand_db_name(
        experiment, exp_doc, verbosity=verbosity,
    )
    if hand_db_name in server:
        hand_db = server[hand_db_name]
    elif dry_run:
        if verbosity >= 1:
            print(
                f"[dry-run] would create hand DB "
                f"{hand_db_name!r}",
                file=sys.stderr,
            )
        hand_db = None
    else:
        if verbosity >= 1:
            print(
                f"Creating hand DB {hand_db_name!r}",
                file=sys.stderr,
            )
        hand_db = server.create(hand_db_name)

    # Resolve the status DB — brat_ingest writes a `reviewer_action`
    # sub-doc to each treatment's status doc, so 'reviewed but
    # empty' cases (all annotations rejected) are still detectable
    # by downstream tools that filter on 'already reviewed'.
    # Without this, features_hand alone can't distinguish
    # 'reviewer said no to everything' from 'not yet reviewed'.
    from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
        resolve_status_db_name,
    )
    status_db_name = resolve_status_db_name(
        experiment, exp_doc, verbosity=verbosity,
    )
    if status_db_name in server:
        status_db = server[status_db_name]
    else:
        # If the status DB doesn't exist, the treatment was
        # (probably) not bootstrapped via Design Y — skip the
        # reviewer_action write.  brat_ingest still succeeds; the
        # user just misses the 'reviewed-empty' detection benefit.
        if verbosity >= 1:
            print(
                f"NOTE: status DB {status_db_name!r} does not "
                f"exist; skipping reviewer_action write "
                f"(features_hand alone is still authoritative)",
                file=sys.stderr,
            )
        status_db = None

    # Per-treatment processing
    totals = {'kept': 0, 'added': 0, 'deleted': 0, 'errors': 0}
    reviewed_at = datetime.now(timezone.utc).isoformat()

    for tid, ann_path in ann_pairs:
        try:
            treatment = dict(treatments_db[tid])
        except Exception:  # noqa: BLE001
            print(
                f"  ERROR {tid}: not found in {treatments_db_name}",
                file=sys.stderr,
            )
            totals['errors'] += 1
            continue

        synth_txt, span_map = render(treatment)
        if not synth_txt:
            print(
                f"  ERROR {tid}: treatment has no description / "
                f"diagnosis; synthetic doc is empty",
                file=sys.stderr,
            )
            totals['errors'] += 1
            continue

        # Parse the reviewer's .ann
        try:
            with open(ann_path, 'r') as f:
                ann_text = f.read()
            reviewed_anns = parse_brat_ann(ann_text, span_map)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  ERROR {tid}: parse_brat_ann({ann_path!r}): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            totals['errors'] += 1
            continue

        # Diff against the candidate DB
        candidate_anns = fetch_candidate_anns_for_treatment(
            candidate_db, tid,
        )
        result: DiffResult = diff_annotations(
            reviewed_anns, candidate_anns,
        )

        totals['kept'] += len(result.kept)
        totals['added'] += len(result.added)
        totals['deleted'] += len(result.deleted)

        # Build candidate lookup so kept docs can pull bootstrap
        # provenance through.
        from treatments_to_structured.brat_ingest import (  # noqa: E402
            annotation_key,
        )
        candidates_by_key = {
            annotation_key(c): c for c in candidate_anns
        }
        doc_id = (treatment.get('ingest') or {}).get('_id') or ''

        # Write to hand DB
        if not dry_run and hand_db is not None:
            # Delete stale reviewed docs first.  Each ingest is the
            # source-of-truth for the treatment's reviewed state —
            # if the reviewer relabeled or deleted an annotation on
            # this pass, the prior reviewed doc must go.  Without
            # this, e.g., relabeling Spores → Basidiospores leaves
            # the stale Spores doc forever.
            kept_added_ids = set()
            for r in result.kept + result.added:
                kept_added_ids.add(
                    f"{tid}:{r['feature_label']}:{int(r['start'])}"
                )
            prefix = f'{tid}:'
            stale_rows = hand_db.view(
                '_all_docs',
                startkey=prefix,
                endkey=prefix + '￰',
                include_docs=True,
            ).rows
            for row in stale_rows:
                if row.doc is None:
                    continue
                doc = dict(row.doc)
                if doc['_id'] in kept_added_ids:
                    continue
                try:
                    hand_db.delete(doc)
                    if verbosity >= 2:
                        print(
                            f"    × stale reviewed doc removed: "
                            f"{doc['_id']}",
                            file=sys.stderr,
                        )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  ERROR deleting stale reviewed doc "
                        f"{doc['_id']}: {exc}",
                        file=sys.stderr,
                    )
            for r in result.kept:
                rd = make_reviewed_doc(
                    r, treatment_id=tid, doc_id=doc_id,
                    reviewer=reviewer, reviewed_at=reviewed_at,
                    action='kept',
                    candidate_match=candidates_by_key.get(
                        annotation_key(r),
                    ),
                )
                if rd['_id'] in hand_db:
                    rd['_rev'] = hand_db[rd['_id']]['_rev']
                try:
                    hand_db.save(rd)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  ERROR saving {rd['_id']}: {exc}",
                        file=sys.stderr,
                    )
            for r in result.added:
                rd = make_reviewed_doc(
                    r, treatment_id=tid, doc_id=doc_id,
                    reviewer=reviewer, reviewed_at=reviewed_at,
                    action='added',
                    candidate_match=None,
                )
                if rd['_id'] in hand_db:
                    rd['_rev'] = hand_db[rd['_id']]['_rev']
                try:
                    hand_db.save(rd)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  ERROR saving {rd['_id']}: {exc}",
                        file=sys.stderr,
                    )

        # Update the treatment's status doc with a reviewer_action
        # sub-doc so downstream tools can detect "reviewed even
        # if empty" (all annotations rejected → features_hand has
        # no entries for this treatment_id).  See the module
        # docstring for the design rationale.
        if not dry_run and status_db is not None:
            sd_id = status_doc_id(tid)
            try:
                sd = dict(status_db[sd_id])
            except Exception:  # noqa: BLE001
                # No status doc for this treatment — pre-Design-Y
                # data or manually-imported treatment.  Skip; the
                # user's features_hand entries are still
                # authoritative.
                sd = None
            if sd is not None:
                sd['reviewer_action'] = {
                    'reviewer': reviewer,
                    'reviewed_at': reviewed_at,
                    'kept_count': len(result.kept),
                    'added_count': len(result.added),
                    'deleted_count': len(result.deleted),
                }
                try:
                    status_db.save(sd)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  ERROR writing reviewer_action for "
                        f"{tid}: {exc}",
                        file=sys.stderr,
                    )

        if verbosity >= 1:
            print(
                f"  {tid}: {result.summary()}",
                file=sys.stderr,
            )
        if verbosity >= 2:
            for r in result.added:
                print(
                    f"    + {r['feature_label']} "
                    f"[{r['start']}, {r['end']}]",
                    file=sys.stderr,
                )
            for r in result.deleted:
                print(
                    f"    - {r['feature_label']} "
                    f"[{r['start']}, {r['end']}]",
                    file=sys.stderr,
                )

    # End-of-run summary
    print(
        f"\nDone: {totals['kept']} kept, {totals['added']} added, "
        f"{totals['deleted']} deleted across "
        f"{len(ann_pairs) - totals['errors']}/{len(ann_pairs)} "
        f"treatments"
        + (f' ({totals["errors"]} errors)' if totals['errors'] else '')
    )
    return 0 if totals['errors'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
