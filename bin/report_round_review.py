#!/usr/bin/env python3
"""Report an annotation round's label-validation result, per T5.

``treatments_to_structured.review_stats`` has carried the honest
precision/recall logic since 2026-09-01 and **nothing called it** — round
5's figures were computed ad hoc in a session and typed into
``data/annotation_rounds/README.md``.  That is the shape §12.3.45 caught
with the dropped-span recovery script: a capability described in a
docstring with no way to invoke it.  This script is the caller.

It exists mainly to make one rule mechanical rather than remembered:

* **Precision** is pooled, with a **treatment-level** bootstrap
  interval.  The naive annotation-level interval is ~1.6x too narrow
  because a treatment's annotations are not independent draws.
* **Recall gets no interval at all.**  At n=50 the clustered half-width
  is ~15 pp — 83 % +/- 15 spans 68-98 % and supports no conclusion.  Most
  treatments add nothing and one adds many; round 1's 36.3 % was a single
  document contributing 136 of 263 additions.  So this reports the
  *distribution* — median additions, the fraction needing any, top-k
  concentration — and says why the ratio is not quotable.

The reviewed set is the treatments with hand annotations, **not** the
treatments exported to brat: one exported and never opened is not
evidence, and counting it would inflate "needed no changes".

Usage::

    bin/report_round_review --experiment production_v4 --round 7
    bin/report_round_review --experiment production_v4 --round 7 \\
        --round-file data/annotation_rounds/production_v4_round7.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import couchdb  # noqa: E402

from env_config import common_parser, get_env_config  # noqa: E402
from treatments_to_structured.review_stats import (  # noqa: E402
    TreatmentCounts,
    per_treatment_counts,
    precision_bootstrap,
    recall_distribution,
)


def reviewed_treatment_ids(
    hand: Iterable[Dict[str, Any]],
    round_treatment_ids: Sequence[str],
) -> List[str]:
    """The round's treatments that actually carry hand annotations.

    ``features_hand`` spans every round, so scoping by round file is
    what keeps a round-7 report from absorbing round 5's review.  Order
    follows the round file so two reports for the same round diff
    cleanly.
    """
    touched = {
        ann.get('treatment_id') for ann in hand
        if ann.get('treatment_id') is not None
    }
    return [tid for tid in round_treatment_ids if tid in touched]


def format_report(
    counts: Sequence[TreatmentCounts],
    round_no: Any,
    n_resamples: int = 2000,
) -> str:
    """Render the T5 report.  Pure, so the shape is testable."""
    if not counts:
        return (
            f'Round {round_no}: no reviewed treatments.\n'
            f'  Nothing in features_hand matches this round file.  Run '
            f'bin/brat_ingest after a review sitting — review that has '
            f'not been ingested is invisible from the database side.'
        )

    lines: List[str] = [f'Round {round_no} label validation', '']
    lines.append(f'  Treatments reviewed: {len(counts)}')
    candidates = sum(c.candidates for c in counts)
    lines.append(f'  Candidate annotations in scope: {candidates}')
    lines.append('')

    boot = precision_bootstrap(counts, n_resamples=n_resamples)
    lines.append('  PRECISION — pooled, treatment-level bootstrap CI')
    if boot is None:
        lines.append('    undefined: no candidate annotations to keep or '
                     'reject.')
    else:
        point, lo, hi = boot
        kept = sum(c.kept for c in counts)
        lines.append(f'    {point:.2%}  [{lo:.2%}, {hi:.2%}]')
        lines.append(f'    {kept} kept of {candidates} proposed')
        lines.append('    The interval resamples TREATMENTS, not '
                     'annotations: an')
        lines.append('    annotation-level interval is ~1.6x too narrow '
                     'here.')
    lines.append('')

    rec = recall_distribution(counts)
    lines.append('  RECALL — distribution only, NO INTERVAL')
    lines.append('    T5: at these sample sizes a recall interval spans '
                 '~30 pp and')
    lines.append('    supports no conclusion.  Quote the shape, never '
                 'the ratio alone.')
    pooled = rec['pooled_recall']
    pooled_text = f'{pooled:.2%}' if pooled is not None else 'undefined'
    lines.append(f'    pooled ratio (context only):   {pooled_text}'
                 f'  = {rec["kept"]} kept / '
                 f'({rec["kept"]} kept + {rec["added"]} added)')
    lines.append(f'    median additions/treatment:    '
                 f'{rec["median_additions"]}')
    lines.append(f'    treatments needing >=1 add:    '
                 f'{rec["fraction_needing_additions"]:.1%}')
    lines.append(f'    top-1 share of all additions:  '
                 f'{rec["top_1_share"]:.1%}')
    lines.append(f'    top-5 share of all additions:  '
                 f'{rec["top_5_share"]:.1%}')
    lines.append(f'    largest single contribution:   '
                 f'{rec["max_additions"]} additions')
    if rec['top_1_share'] >= 0.4 and rec['added']:
        lines.append('    ^ one treatment dominates the additions, which '
                     'is the round-1')
        lines.append('      failure mode: the pooled ratio describes that '
                     'document, not')
        lines.append('      the corpus.')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--round', type=int, default=None, metavar='N', dest='round_no',
        help='Round number.  Used to locate the round file and to label '
             'the report.',
    )
    parser.add_argument(
        '--round-file', default=None, metavar='PATH',
        help='Round file listing the drawn treatment ids.  Default: '
             'data/annotation_rounds/<experiment>_round<N>.txt',
    )
    parser.add_argument(
        '--n-resamples', type=int, default=2000, metavar='N',
        help='Bootstrap resamples for the precision interval '
             '(default 2000).',
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)
    experiment = config.get('experiment') or 'production_v4'

    if args.round_file:
        round_path = Path(args.round_file)
    elif args.round_no is not None:
        round_path = (Path(__file__).resolve().parent.parent / 'data'
                      / 'annotation_rounds'
                      / f'{experiment}_round{args.round_no}.txt')
    else:
        print('error: pass --round N or --round-file PATH',
              file=sys.stderr)
        return 2
    if not round_path.exists():
        print(f'{round_path}: not found', file=sys.stderr)
        return 1
    round_ids = [
        line.strip() for line in round_path.read_text().splitlines()
        if line.strip()
    ]

    hand_db = (config.get('features_hand_db_name')
               or f'skol_exp_{experiment}_02_55_features_hand')
    candidate_db = (config.get('features_candidate_db_name')
                    or f'skol_exp_{experiment}_02_50_features_candidate')
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (config['couchdb_username'],
                                   config['couchdb_password'])
    for name in (hand_db, candidate_db):
        if name not in server:
            print(f'{name}: not found', file=sys.stderr)
            return 1

    def load(name: str) -> List[Dict[str, Any]]:
        return [
            dict(row.doc)
            for row in server[name].view('_all_docs', include_docs=True)
            if row.doc is not None and not row.id.startswith('_design/')
        ]

    hand = load(hand_db)
    candidate = load(candidate_db)
    reviewed = reviewed_treatment_ids(hand, round_ids)
    counts = per_treatment_counts(hand, candidate, reviewed)
    label = args.round_no if args.round_no is not None else round_path.name
    print(format_report(counts, label, n_resamples=args.n_resamples))
    return 0


if __name__ == '__main__':
    sys.exit(main())
