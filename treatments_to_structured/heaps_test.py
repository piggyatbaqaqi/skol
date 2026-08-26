#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.heaps``."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.heaps import (  # noqa: E402
    cumulative_curve,
    labels_by_treatment,
    permutation_band,
    timestamp_collisions,
)

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason='T4: implementation follows test confirmation',
)


def _ann(tid: str, label: str, ts: str = '2026-01-01T00:00:00') -> dict:
    return {'treatment_id': tid, 'feature_label': label,
            'created_at': ts}


class TestLabelsByTreatment:
    def test_groups_distinct_labels(self) -> None:
        got = labels_by_treatment([
            _ann('a', 'Pileus'), _ann('a', 'Stipe'),
            _ann('b', 'Pileus'),
        ])
        assert got == {'a': {'Pileus', 'Stipe'}, 'b': {'Pileus'}}

    def test_repeats_within_a_treatment_collapse(self) -> None:
        """Two Pileus spans are one vocabulary item.  Counting them
        twice would inflate the curve wherever D7's repeated-label
        merges appear.
        """
        got = labels_by_treatment([_ann('a', 'Pileus')] * 5)
        assert got == {'a': {'Pileus'}}

    def test_canonicalizer_collapses_drift(self) -> None:
        got = labels_by_treatment(
            [_ann('a', 'Colonies'), _ann('a', 'Colony')],
            canonicalizer={'Colonies': 'Colony'},
        )
        assert got == {'a': {'Colony'}}

    def test_annotations_missing_a_field_are_skipped(self) -> None:
        got = labels_by_treatment([
            {'treatment_id': 'a'}, {'feature_label': 'Pileus'},
            _ann('b', 'Stipe'),
        ])
        assert got == {'b': {'Stipe'}}


class TestCumulativeCurve:
    def test_counts_first_appearances(self) -> None:
        by = {'a': {'P', 'S'}, 'b': {'P', 'L'}, 'c': {'X'}}
        xs, ys = cumulative_curve(['a', 'b', 'c'], by)
        assert xs == [1, 2, 3]
        assert ys == [2, 3, 4]

    def test_order_changes_the_curve_but_not_its_endpoint(self) -> None:
        """The endpoint is the vocabulary; the path to it is what the
        ordering artefact distorts.
        """
        by = {'a': {'P', 'S'}, 'b': {'P', 'L'}, 'c': {'X'}}
        _, ys1 = cumulative_curve(['a', 'b', 'c'], by)
        _, ys2 = cumulative_curve(['c', 'b', 'a'], by)
        assert ys1 != ys2
        assert ys1[-1] == ys2[-1] == 4

    def test_a_treatment_with_no_labels_still_advances_x(self) -> None:
        """It was sampled.  Dropping it would compress the curve and
        flatter the vocabulary -- and round 5 has 14 such treatments.
        """
        by = {'a': {'P'}, 'b': set(), 'c': {'S'}}
        xs, ys = cumulative_curve(['a', 'b', 'c'], by)
        assert xs == [1, 2, 3]
        assert ys == [1, 1, 2]

    def test_a_treatment_absent_from_the_map_still_advances_x(
        self,
    ) -> None:
        """Same reasoning: a treatment that produced no annotations at
        all never reaches ``labels_by_treatment``, but it was drawn.
        """
        xs, ys = cumulative_curve(['a', 'ghost'], {'a': {'P'}})
        assert (xs, ys) == ([1, 2], [1, 1])

    def test_never_keys_on_time(self) -> None:
        """The old implementation grouped labels by ``created_at`` and
        added them for *every* treatment sharing that timestamp, so a
        collision double-counted.  Identical timestamps must now be
        irrelevant.
        """
        by = {'a': {'P'}, 'b': {'S'}}
        xs, ys = cumulative_curve(['a', 'b'], by)
        assert ys == [1, 2]

    def test_empty_order(self) -> None:
        assert cumulative_curve([], {'a': {'P'}}) == ([], [])


class TestPermutationBand:
    def test_returns_a_band_over_the_same_x(self) -> None:
        by = {c: {c.upper()} for c in 'abcdefgh'}
        xs, mean, lo, hi = permutation_band(by, n_permutations=50)
        assert xs == list(range(1, 9))
        assert len(mean) == len(lo) == len(hi) == 8
        assert all(a <= b <= c for a, b, c in zip(lo, mean, hi))

    def test_the_endpoint_is_certain(self) -> None:
        """Every permutation ends on the full vocabulary, so the band
        must pinch shut at the right-hand end.  A band that does not is
        a sign the permutation is dropping treatments.
        """
        by = {'a': {'P', 'S'}, 'b': {'L'}, 'c': {'X'}}
        _, mean, lo, hi = permutation_band(by, n_permutations=50)
        assert lo[-1] == mean[-1] == hi[-1] == 4

    def test_is_reproducible_from_the_seed(self) -> None:
        by = {c: {c.upper()} for c in 'abcdef'}
        a = permutation_band(by, n_permutations=20, seed=7)
        b = permutation_band(by, n_permutations=20, seed=7)
        assert a == b

    def test_a_different_seed_gives_a_different_band(self) -> None:
        by = {c: {c.upper(), 'shared'} for c in 'abcdefghij'}
        a = permutation_band(by, n_permutations=20, seed=1)
        b = permutation_band(by, n_permutations=20, seed=2)
        assert a[1] != b[1]

    def test_single_treatment(self) -> None:
        xs, mean, lo, hi = permutation_band({'a': {'P'}},
                                            n_permutations=5)
        assert (xs, mean, lo, hi) == ([1], [1.0], [1.0], [1.0])


class TestTimestampCollisions:
    def test_healthy_data_reports_none(self) -> None:
        assert timestamp_collisions([
            _ann('a', 'P', '2026-01-01T00:00:00'),
            _ann('b', 'S', '2026-01-01T00:00:01'),
        ]) == {}

    def test_a_shared_timestamp_is_reported(self) -> None:
        got = timestamp_collisions([
            _ann('a', 'P', '2026-01-01T00:00:00'),
            _ann('b', 'S', '2026-01-01T00:00:00'),
        ])
        assert got == {'2026-01-01T00:00:00': {'a', 'b'}}

    def test_one_treatment_many_annotations_is_not_a_collision(
        self,
    ) -> None:
        """Every annotation from one Claude call shares a timestamp by
        construction; that is the normal case, not the fault.
        """
        assert timestamp_collisions([
            _ann('a', 'P', '2026-01-01T00:00:00'),
            _ann('a', 'S', '2026-01-01T00:00:00'),
        ]) == {}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
