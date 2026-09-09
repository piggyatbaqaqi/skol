"""Tests for bin/report_round_review — the T5 reporting CLI.

The module `treatments_to_structured.review_stats` has had the honest
precision/recall logic (and tests) since 2026-09-01, and nothing called
it: round 5's figures were computed by hand in a session and typed into
`data/annotation_rounds/README.md`.  This is the same shape as the
dropped-span recovery script §12.3.45 found advertised and absent, so
this file exists before the next round is reviewed rather than after.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_round_review import (  # type: ignore[import]  # noqa: E402
    format_report,
    reviewed_treatment_ids,
)
from treatments_to_structured.review_stats import (  # noqa: E402
    TreatmentCounts,
)


def _hand(tid: str, action: str, n: int = 1) -> List[Dict[str, Any]]:
    return [{'treatment_id': tid, 'reviewer_action': action}] * n


class TestReviewedTreatmentIds:
    def test_only_treatments_with_hand_annotations_count(self) -> None:
        """The reviewed set is what the reviewer touched, not what was
        exported.  A treatment exported to brat and never opened has no
        hand annotations and must not enter the denominator."""
        hand = _hand('taxon_a', 'kept', 3) + _hand('taxon_b', 'added')
        assert reviewed_treatment_ids(hand, ['taxon_a', 'taxon_b',
                                             'taxon_c']) == [
            'taxon_a', 'taxon_b',
        ]

    def test_order_follows_the_round_file(self) -> None:
        """Order is the round file's, so a report can be diffed against
        an earlier one for the same round."""
        hand = _hand('taxon_b', 'kept') + _hand('taxon_a', 'kept')
        assert reviewed_treatment_ids(hand, ['taxon_a', 'taxon_b']) == [
            'taxon_a', 'taxon_b',
        ]

    def test_hand_annotation_outside_the_round_is_ignored(self) -> None:
        """features_hand spans every round; a report for round 7 must
        not silently absorb round 5's reviewed treatments."""
        hand = _hand('taxon_other', 'kept', 9) + _hand('taxon_a', 'kept')
        assert reviewed_treatment_ids(hand, ['taxon_a']) == ['taxon_a']

    def test_empty_review_returns_empty(self) -> None:
        assert reviewed_treatment_ids([], ['taxon_a']) == []


class TestFormatReport:
    def test_precision_is_reported_with_a_treatment_level_interval(
        self,
    ) -> None:
        counts = [
            TreatmentCounts('taxon_a', kept=9, added=0, candidates=10),
            TreatmentCounts('taxon_b', kept=10, added=1, candidates=10),
        ]
        out = format_report(counts, round_no=7, n_resamples=200)
        assert 'precision' in out.lower()
        assert 'treatment-level bootstrap' in out

    def test_recall_is_never_given_an_interval(self) -> None:
        """The load-bearing rule from T5.  At n=50 a recall interval
        spans 68-98 %, so the report must show the distribution and say
        why there is no interval — a reader who sees a bare percentage
        will quote it with one."""
        counts = [
            TreatmentCounts('taxon_a', kept=9, added=0, candidates=10),
            TreatmentCounts('taxon_b', kept=10, added=136, candidates=10),
        ]
        out = format_report(counts, round_no=7, n_resamples=200)
        recall_block = out[out.lower().index('recall'):]
        assert 'no interval' in recall_block.lower()
        # The distribution, not the ratio, is what is quotable.
        assert 'median additions' in recall_block.lower()
        assert 'top-1' in recall_block.lower()

    def test_the_concentration_that_broke_round_1_is_visible(self) -> None:
        """Round 1's 36.3 % recall was one document contributing 136 of
        263 additions.  A report that hides concentration would let that
        happen again unnoticed."""
        counts = [
            TreatmentCounts('taxon_a', kept=1, added=0, candidates=1),
            TreatmentCounts('taxon_b', kept=1, added=136, candidates=1),
        ]
        out = format_report(counts, round_no=1, n_resamples=200)
        assert '100.0%' in out.replace(' ', '') or '100%' in out
        assert '136' in out

    def test_zero_addition_treatments_stay_in_the_denominator(
        self,
    ) -> None:
        """A treatment read and left unchanged is evidence.  The report
        must count it, since the zeros dominate the distribution."""
        counts = [
            TreatmentCounts('taxon_a', kept=5, added=0, candidates=5),
            TreatmentCounts('taxon_b', kept=5, added=0, candidates=5),
            TreatmentCounts('taxon_c', kept=5, added=1, candidates=5),
        ]
        out = format_report(counts, round_no=7, n_resamples=200)
        assert 'treatments reviewed: 3' in out.lower()

    def test_empty_review_says_so_rather_than_dividing_by_zero(
        self,
    ) -> None:
        out = format_report([], round_no=8, n_resamples=200)
        assert 'no reviewed treatments' in out.lower()
