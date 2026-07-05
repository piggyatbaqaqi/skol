#!/usr/bin/env python3
"""Build a per-treatment triage CSV for pre-review batching.

Phase 1 tooling for the second round of hand review.  See
docs/schema_constrained_pipeline.md §10.4 and the memo entries in
docs/data_quality_production_v4_model.md §§1, 6, 8, 10, 11.

For every treatment that has been bootstrapped (i.e., has a doc in
the ``features_status`` DB), computes:

  * bootstrap outcome + review status (reviewed / unreviewed via
    ``reviewer_action`` sub-doc or non-empty features_hand entries);
  * the merge-detection metric (``treatment_merge_metric``);
  * every triage signal from ``triage_signals.treatment_signals``
    (multi-Diagnosis, multi-sp.-nov., numbered couplets, Latin
    alternation, mid-sentence start, etc.);
  * a compact ``predicted_issues`` flag string that concatenates
    the triggered §-numbered heuristics.

The CSV lets an operator sort by predicted_issues and merge_metric
so the highest-signal treatments float to the top of the review
queue.  Column layout is one-signal-per-column so operators can
eyeball raw values, not just the summary verdict.

Usage::

    bin/triage_treatments --experiment production_v4 \\
        --output /tmp/triage_production_v4.csv

    # Filter to specific treatments:
    bin/triage_treatments --experiment production_v4 \\
        --doc-id taxon_abc,taxon_def --output /tmp/triage.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser, get_env_config  # noqa: E402
from treatments_to_structured.merge_metric import (  # noqa: E402
    treatment_merge_metric,
)
from treatments_to_structured.triage_signals import (  # noqa: E402
    predicted_issues,
    treatment_signals,
)


# ---------------------------------------------------------------------------
# CSV column layout
# ---------------------------------------------------------------------------

CSV_COLUMNS: List[str] = [
    'treatment_id',
    'reviewed',
    'reviewer',
    'bootstrap_status',
    'claude_annotation_count',
    'kept_count',
    'added_count',
    'deleted_count',
    'merge_metric',
    'desc_length',
    'diag_length',
    'n_diagnosis_headers',
    'n_description_headers',
    'n_sp_nov',
    'n_key_couplets',
    'desc_starts_mid_sentence',
    'latin_block_count',
    'latin_between_english',
    'mid_body_description_header',
    'tail_clipped',
    'diag_starts_mid_sentence',
    'authored_binomial_in_desc',
    'synthetic_nomenclature',
    'predicted_issues',
    'first_line',
]


# ---------------------------------------------------------------------------
# Testable helpers
# ---------------------------------------------------------------------------


def build_reviewed_hand_ids(hand_db: Any) -> Set[str]:
    """Enumerate treatment IDs with any annotation in features_hand.

    Same prefix-parsing trick brat_export uses: annotation _id is
    ``<treatment_id>:<label>:<offset>``.  A hand DB entry means
    the reviewer kept or added at least one annotation for the
    treatment.  Combine with ``reviewer_action`` markers (from the
    status DB) for the full 'has been reviewed' set — see
    ``review_status``.
    """
    ids: Set[str] = set()
    for row in hand_db.view('_all_docs').rows:
        key = row.id if hasattr(row, 'id') else row.get('id', '')
        if not key or ':' not in key:
            continue
        ids.add(key.split(':', 1)[0])
    return ids


def review_status(
    status_doc: Optional[Dict[str, Any]],
    reviewed_hand_ids: Set[str],
    treatment_id: str,
) -> Dict[str, Any]:
    """Determine whether a treatment has been reviewed and by whom.

    A treatment is 'reviewed' if EITHER:
      * its status doc carries a ``reviewer_action`` sub-doc
        (written by brat_ingest since 2026-07-01, covers even the
        'all annotations rejected → features_hand empty' case), or
      * it has any entries in features_hand.

    Returns a dict with keys ``reviewed`` (bool), ``reviewer``
    (str or ''), and the three reviewer_action count fields.
    Counts are 0 when reviewer_action is absent even if the
    treatment is reviewed via the features_hand path — the
    hand-DB path predates the counts.
    """
    action = None
    if status_doc:
        action = status_doc.get('reviewer_action')
    if action:
        return {
            'reviewed': True,
            'reviewer': action.get('reviewer', ''),
            'kept_count': int(action.get('kept_count', 0) or 0),
            'added_count': int(action.get('added_count', 0) or 0),
            'deleted_count': int(action.get('deleted_count', 0) or 0),
        }
    if treatment_id in reviewed_hand_ids:
        return {
            'reviewed': True,
            'reviewer': '',
            'kept_count': 0,
            'added_count': 0,
            'deleted_count': 0,
        }
    return {
        'reviewed': False,
        'reviewer': '',
        'kept_count': 0,
        'added_count': 0,
        'deleted_count': 0,
    }


def _first_nonempty_line(text: str, cap: int = 80) -> str:
    """First non-blank line of ``text``, truncated to ``cap`` chars
    for the CSV.  Blank / None returns ''.
    """
    if not text:
        return ''
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            if len(stripped) > cap:
                return stripped[: cap - 1] + '…'
            return stripped
    return ''


def build_row(
    treatment_id: str,
    treatment_doc: Optional[Dict[str, Any]],
    status_doc: Optional[Dict[str, Any]],
    reviewed_hand_ids: Set[str],
    merge_threshold: int,
    *,
    authored_binomial: Optional[bool] = None,
) -> Dict[str, Any]:
    """Compose one CSV row for a single treatment.

    Missing treatment_doc (bootstrapped but the prose doc was
    later deleted, or a status-only entry) produces a row with
    all signal columns 0 and a ``predicted_issues`` value of
    ``'no_prose_doc'`` so the operator sees the gap.

    ``authored_binomial`` is the caller-supplied gn_client result
    (or None if gn services are unavailable / not queried).
    Threaded in rather than computed here so the network cost
    happens once per treatment in ``main`` where the
    service-unavailability warning can be issued once per
    invocation.
    """
    if treatment_doc is None:
        signals = {
            'desc_length': 0,
            'diag_length': 0,
            'n_diagnosis_headers': 0,
            'n_description_headers': 0,
            'n_sp_nov': 0,
            'n_key_couplets': 0,
            'desc_starts_mid_sentence': False,
            'latin_block_count': 0,
            'latin_between_english': False,
            'mid_body_description_header': False,
            'tail_clipped': False,
            'diag_starts_mid_sentence': False,
            'authored_binomial_in_desc': False,
            'synthetic_nomenclature': False,
        }
        merge_metric = 0
        first_line = ''
        issues = 'no_prose_doc'
    else:
        signals = treatment_signals(
            treatment_doc,
            authored_binomial_in_desc=authored_binomial,
        )
        merge_metric = treatment_merge_metric(treatment_doc)
        first_line = _first_nonempty_line(
            treatment_doc.get('description') or ''
        )
        issues = predicted_issues(
            signals, merge_metric, merge_threshold=merge_threshold,
        )

    review = review_status(
        status_doc, reviewed_hand_ids, treatment_id,
    )

    bootstrap_status = (
        (status_doc or {}).get('status', 'unknown')
    )
    claude_ann_count = int(
        (status_doc or {}).get('annotation_count', 0) or 0
    )

    return {
        'treatment_id': treatment_id,
        'reviewed': review['reviewed'],
        'reviewer': review['reviewer'],
        'bootstrap_status': bootstrap_status,
        'claude_annotation_count': claude_ann_count,
        'kept_count': review['kept_count'],
        'added_count': review['added_count'],
        'deleted_count': review['deleted_count'],
        'merge_metric': merge_metric,
        'desc_length': signals['desc_length'],
        'diag_length': signals['diag_length'],
        'n_diagnosis_headers': signals['n_diagnosis_headers'],
        'n_description_headers': signals['n_description_headers'],
        'n_sp_nov': signals['n_sp_nov'],
        'n_key_couplets': signals['n_key_couplets'],
        'desc_starts_mid_sentence':
            signals['desc_starts_mid_sentence'],
        'latin_block_count': signals['latin_block_count'],
        'latin_between_english':
            signals['latin_between_english'],
        'mid_body_description_header':
            signals['mid_body_description_header'],
        'tail_clipped': signals['tail_clipped'],
        'diag_starts_mid_sentence':
            signals['diag_starts_mid_sentence'],
        'authored_binomial_in_desc':
            signals['authored_binomial_in_desc'],
        'synthetic_nomenclature':
            signals['synthetic_nomenclature'],
        'predicted_issues': issues,
        'first_line': first_line,
    }


def sort_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sort the CSV for review-priority: unreviewed first, then
    highest merge_metric, then flagged issues, then treatment_id.

    Keeps reviewed rows at the bottom so the operator's queue is
    the top of the file.  Within unreviewed, high-signal treatments
    surface first.
    """
    def key(r: Dict[str, Any]) -> Any:
        return (
            0 if not r['reviewed'] else 1,
            -int(r.get('merge_metric', 0) or 0),
            0 if r.get('predicted_issues') else 1,
            r['treatment_id'],
        )
    return sorted(rows, key=key)


def write_csv(
    rows: Iterable[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write rows to ``output_path`` in the canonical column order.

    Uses ``csv.DictWriter`` with ``extrasaction='ignore'`` so a
    row carrying extra keys writes cleanly (defensive against
    schema drift in ``build_row``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_COLUMNS, extrasaction='ignore',
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--output', required=True, metavar='FILE',
        help='Path to write the triage CSV.',
    )
    parser.add_argument(
        '--merge-threshold', type=int, default=10,
        help=(
            'Metric threshold above which the merge_metric flag '
            'fires in predicted_issues.  Default 10 (same as '
            'bin/select_for_annotation).'
        ),
    )
    parser.add_argument(
        '--include-skipped', action='store_true',
        help=(
            "Include treatments with status 'skipped_merge_suspect' "
            '(written by bin/select_for_annotation when the merge '
            'filter fires pre-bootstrap).  Default off — the CSV is '
            'meant for the review queue of the actually-bootstrapped '
            'treatments, not the tens of thousands of skip records.'
        ),
    )
    args = parser.parse_args()

    config = get_env_config(cli_args=args)
    verbosity = int(config.get('verbosity', 1) or 0)
    experiment = config.get('experiment_name')
    if not experiment:
        print("error: --experiment is required", file=sys.stderr)
        return 2

    doc_id_filter = config.get('doc_ids') or None

    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )

    # treatments_prose (for signal computation)
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

    # Resolve status + hand DB names (same helpers other bin/
    # tools use so fallback naming stays in one place).
    from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
        resolve_status_db_name,
    )
    from brat_ingest import (  # type: ignore[import]  # noqa: E402
        resolve_hand_db_name,
    )
    status_db_name = resolve_status_db_name(
        experiment, exp_doc, verbosity=verbosity,
    )
    hand_db_name = resolve_hand_db_name(
        experiment, exp_doc, verbosity=verbosity,
    )

    if status_db_name not in server:
        print(
            f"error: status DB {status_db_name!r} not found — run "
            f"bin/llm_annotate_features first",
            file=sys.stderr,
        )
        return 2
    status_db = server[status_db_name]

    reviewed_hand_ids: Set[str] = set()
    if hand_db_name in server:
        reviewed_hand_ids = build_reviewed_hand_ids(
            server[hand_db_name],
        )

    # Enumerate treatments — iterate the status DB rather than
    # treatments_prose so the CSV represents exactly what's been
    # bootstrapped ('the 56').  A treatment_prose doc without a
    # status doc means we haven't attempted it yet — deliberately
    # out of scope for this triage pass.
    #
    # By default we drop STATUS_SKIPPED_MERGE_SUSPECT entries
    # (7000+ of them written by select_for_annotation --exclude-
    # suspected-merges).  They're not in the review queue; they
    # were filtered out pre-bootstrap.  --include-skipped surfaces
    # them for the "why did select_for_annotation drop these"
    # sanity check.
    include_skipped = bool(args.include_skipped)
    treatment_ids: List[str] = []
    for row in status_db.view('_all_docs', include_docs=True).rows:
        tid = row.id if hasattr(row, 'id') else row.get('id', '')
        if not tid or tid.startswith('_design/'):
            continue
        doc = row.doc if hasattr(row, 'doc') else None
        if (
            not include_skipped
            and doc is not None
            and doc.get('status') == 'skipped_merge_suspect'
        ):
            continue
        treatment_ids.append(tid)

    if doc_id_filter:
        wanted = {t.strip() for t in doc_id_filter if t.strip()}
        treatment_ids = [t for t in treatment_ids if t in wanted]

    if verbosity >= 1:
        print(
            f"triage: {len(treatment_ids)} treatments to score "
            f"(status DB: {status_db_name!r})",
            file=sys.stderr,
        )

    # §6 idea #2: gnfinder + gnparser detector for authored
    # binomials in Description.  Configured URLs come from
    # env_config (skol-gnservices deb defaults localhost).
    # First failure to reach the services is warned once; all
    # subsequent treatments get authored_binomial=None (not
    # fired) without repeated noise.
    from treatments_to_structured import gn_client  # noqa: E402
    gnfinder_url = config.get(
        'gnfinder_url', gn_client.DEFAULT_GNFINDER_URL,
    )
    gnparser_url = config.get(
        'gnparser_url', gn_client.DEFAULT_GNPARSER_URL,
    )
    gn_unavailable = False

    rows: List[Dict[str, Any]] = []
    for tid in treatment_ids:
        try:
            status_doc = dict(status_db[tid])
        except Exception:
            status_doc = None
        try:
            treatment_doc = dict(treatments_db[tid])
        except Exception:
            treatment_doc = None

        authored_binomial: Optional[bool] = None
        if treatment_doc is not None and not gn_unavailable:
            desc = treatment_doc.get('description') or ''
            try:
                authored_binomial = (
                    gn_client.authored_binomial_in_text(
                        desc, gnfinder_url, gnparser_url,
                    )
                )
            except gn_client.GnServiceUnavailable as exc:
                print(
                    f"warning: gn services unavailable "
                    f"({exc}); §6:authored_binomial will not "
                    f"fire for the remainder of this run",
                    file=sys.stderr,
                )
                gn_unavailable = True

        rows.append(build_row(
            treatment_id=tid,
            treatment_doc=treatment_doc,
            status_doc=status_doc,
            reviewed_hand_ids=reviewed_hand_ids,
            merge_threshold=int(args.merge_threshold),
            authored_binomial=authored_binomial,
        ))

    rows = sort_rows(rows)
    write_csv(rows, Path(args.output))

    if verbosity >= 1:
        n_reviewed = sum(1 for r in rows if r['reviewed'])
        n_flagged = sum(1 for r in rows if r['predicted_issues'])
        print(
            f"triage: wrote {len(rows)} rows to {args.output} "
            f"({n_reviewed} reviewed, "
            f"{n_flagged} flagged with predicted_issues)",
            file=sys.stderr,
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
