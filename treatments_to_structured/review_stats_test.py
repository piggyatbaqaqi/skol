#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.review_stats``."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.review_stats import (  # noqa: E402
    TreatmentCounts,
    per_treatment_counts,
    precision_bootstrap,
    recall_distribution,
)


def _c(tid, kept, added, candidates):
    return TreatmentCounts(treatment_id=tid, kept=kept, added=added,
                           candidates=candidates)


class TestPerTreatmentCounts:
    def test_splits_kept_from_added(self):
        hand = [{'treatment_id': 'a', 'reviewer_action': 'kept'},
                {'treatment_id': 'a', 'reviewer_action': 'added'},
                {'treatment_id': 'a', 'reviewer_action': 'kept'}]
        cand = [{'treatment_id': 'a'}, {'treatment_id': 'a'}]
        got = per_treatment_counts(hand, cand, ['a'])
        assert got == [_c('a', kept=2, added=1, candidates=2)]

    def test_a_reviewed_treatment_with_no_annotations_is_kept(self):
        """It was reviewed and yielded nothing.  Dropping it would
        remove the zero from the recall distribution, which is where
        most of the mass lies -- 53 of 85 treatments added nothing in
        the round-4 measurement.
        """
        got = per_treatment_counts([], [], ['ghost'])
        assert got == [_c('ghost', kept=0, added=0, candidates=0)]

    def test_treatments_outside_the_review_set_are_ignored(self):
        hand = [{'treatment_id': 'other', 'reviewer_action': 'kept'}]
        assert per_treatment_counts(hand, [], ['a']) == [
            _c('a', kept=0, added=0, candidates=0)]

    def test_unknown_reviewer_action_is_not_silently_counted(self):
        """`kept` and `added` are the only actions brat_ingest writes.
        A third value means the schema changed and the statistic is no
        longer measuring what it claims.
        """
        hand = [{'treatment_id': 'a', 'reviewer_action': 'edited'}]
        with pytest.raises(ValueError, match='edited'):
            per_treatment_counts(hand, [], ['a'])


class TestPrecisionBootstrap:
    def test_resamples_treatments_not_annotations(self):
        """**The whole point.**  Annotations within a treatment are
        highly correlated -- the measured design effect is 38.8x for
        recall -- so an annotation-level interval is ~6x too narrow.

        One treatment holding every error must therefore produce a wide
        interval, not a narrow one: resampling treatments sometimes
        draws it several times and sometimes not at all.
        """
        counts = [_c('bad', kept=0, added=0, candidates=10)] + [
            _c(f'g{i}', kept=10, added=0, candidates=10) for i in range(9)]
        point, lo, hi = precision_bootstrap(counts, n_resamples=400, seed=1)
        assert point == pytest.approx(0.9, abs=1e-9)
        assert hi - lo > 0.2

    def test_a_perfect_review_has_a_degenerate_interval(self):
        counts = [_c(f'g{i}', kept=5, added=0, candidates=5) for i in range(6)]
        point, lo, hi = precision_bootstrap(counts, n_resamples=100, seed=1)
        assert (point, lo, hi) == (1.0, 1.0, 1.0)

    def test_treatments_with_no_candidates_do_not_enter_precision(self):
        """Precision is over *candidates*.  A treatment the model said
        nothing about cannot make it better or worse, and including it
        as a 0/0 would either crash or silently bias.
        """
        counts = [_c('a', kept=4, added=0, candidates=4),
                  _c('empty', kept=0, added=3, candidates=0)]
        point, _, _ = precision_bootstrap(counts, n_resamples=50, seed=1)
        assert point == 1.0

    def test_is_reproducible_from_the_seed(self):
        counts = [_c(f't{i}', kept=i, added=0, candidates=i + 1)
                  for i in range(8)]
        a = precision_bootstrap(counts, n_resamples=200, seed=7)
        b = precision_bootstrap(counts, n_resamples=200, seed=7)
        assert a == b

    def test_no_candidates_anywhere_yields_none(self):
        counts = [_c('a', kept=0, added=2, candidates=0)]
        assert precision_bootstrap(counts, n_resamples=10, seed=1) is None


class TestRecallDistribution:
    def test_reports_a_distribution_and_never_an_interval(self):
        """The plan forbids a pooled recall point estimate with a CI:
        at n=50 the clustered half-width is +/-15pp, which spans
        68-98% and supports no conclusion.  The API must not offer one.
        """
        counts = [_c('a', kept=5, added=0, candidates=5),
                  _c('b', kept=3, added=7, candidates=3)]
        got = recall_distribution(counts)
        assert 'ci' not in got and 'interval' not in got
        assert got['median_additions'] == 3.5
        assert got['fraction_needing_additions'] == pytest.approx(0.5)

    def test_top_k_concentration(self):
        """R1's 36.3% recall was one document contributing 136 of 263
        additions.  Concentration is the robust statistic; the ratio is
        not.
        """
        counts = ([_c('big', kept=1, added=90, candidates=1)]
                  + [_c(f'x{i}', kept=1, added=1, candidates=1)
                     for i in range(10)])
        got = recall_distribution(counts)
        assert got['top_1_share'] == pytest.approx(0.9)
        assert got['top_5_share'] == pytest.approx(0.94)

    def test_pooled_ratio_is_reported_as_a_raw_count_pair(self):
        counts = [_c('a', kept=8, added=2, candidates=8)]
        got = recall_distribution(counts)
        assert got['kept'] == 8 and got['added'] == 2
        assert got['pooled_recall'] == pytest.approx(0.8)

    def test_all_zero_review(self):
        counts = [_c('a', kept=0, added=0, candidates=0)]
        got = recall_distribution(counts)
        assert got['median_additions'] == 0
        assert got['fraction_needing_additions'] == 0.0
        assert got['pooled_recall'] is None


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
