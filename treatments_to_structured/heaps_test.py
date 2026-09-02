#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.heaps``."""

import collections
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.heaps import (  # noqa: E402
    cumulative_curve,
    fit_beta,
    instances_by_treatment,
    labels_by_treatment,
    out_of_sample_coverage,
    permutation_band,
    timestamp_collisions,
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
        """The fixture has to make order *matter*.

        One rich treatment holding the whole vocabulary and five poor
        ones each repeating a piece of it: rich-first jumps straight to
        5, rich-last climbs 1,2,3,4,5.  A fixture where every treatment
        contributes exactly one new label produces the same curve under
        every permutation, so the band is seed-independent and the test
        would pass vacuously.
        """
        by = {'rich': {'A', 'B', 'C', 'D', 'E'},
              'p1': {'A'}, 'p2': {'B'}, 'p3': {'C'},
              'p4': {'D'}, 'p5': {'E'}}
        a = permutation_band(by, n_permutations=20, seed=1)
        b = permutation_band(by, n_permutations=20, seed=2)
        assert a[1] != b[1]

    def test_order_is_irrelevant_when_every_treatment_is_novel(
        self,
    ) -> None:
        """The converse, pinned so the fixture above is not mistaken
        for arbitrary: if each treatment contributes exactly one new
        label, every permutation gives the same curve and the band has
        zero width.
        """
        by = {c: {c.upper()} for c in 'abcdef'}
        _, mean, lo, hi = permutation_band(by, n_permutations=20)
        assert lo == mean == hi

    def test_the_drawn_population_can_be_given_explicitly(
        self,
    ) -> None:
        """`cumulative_curve` counts a label-less treatment because it
        was sampled; the band must agree, and it cannot know about one
        that never reached `by_treatment`.

        Round 5 makes this concrete: 1 000 drawn, 877 with labels.
        Permuting only the 877 puts the curve's x-axis 123 short and
        steepens it -- exactly the flattery this whole exercise is
        meant to remove.
        """
        by = {'a': {'P'}, 'b': {'S'}}
        xs, mean, lo, hi = permutation_band(
            by, n_permutations=20, ids=['a', 'b', 'ghost1', 'ghost2'])
        assert xs == [1, 2, 3, 4]
        assert mean[-1] == lo[-1] == hi[-1] == 2

    def test_ids_default_to_the_treatments_with_labels(self) -> None:
        by = {'a': {'P'}, 'b': {'S'}}
        assert permutation_band(by, n_permutations=5)[0] == [1, 2]

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


class TestCallableCanonicalizer:
    """A dict cannot express a *rule*.

    ``feature_label_rules.canonicalize`` settles whole families —
    the sexual/asexual head noun, for one — that no finite map
    enumerates, so the curve has to be able to take a function.
    """

    def test_callable_is_applied_to_every_label(self) -> None:
        got = labels_by_treatment(
            [_ann('a', 'Sexual state'), _ann('a', 'Sexual morph')],
            canonicalizer=lambda label: label.replace('state', 'morph'),
        )
        assert got == {'a': {'Sexual morph'}}

    def test_a_mapping_still_works(self) -> None:
        """The dict form is what the notebook and every existing
        caller pass; it must not regress."""
        got = labels_by_treatment(
            [_ann('a', 'Colonies'), _ann('a', 'Colony')],
            canonicalizer={'Colonies': 'Colony'},
        )
        assert got == {'a': {'Colony'}}

    def test_identity_callable_changes_nothing(self) -> None:
        got = labels_by_treatment(
            [_ann('a', 'Pileus')], canonicalizer=lambda label: label,
        )
        assert got == {'a': {'Pileus'}}


class TestFitBeta:
    """Heaps' exponent, fitted where the power law actually holds.

    The head of a vocabulary curve is not Heaps: at n=1 every label
    is new, so V tracks n and the slope approaches 1.  Fitting from
    n=1 therefore reports a steeper beta than the corpus has --
    measured 0.645 against 0.601 on round 5.  ``min_n`` is the
    window, and it is the whole point of the function.
    """

    def _power_law(self, k: float, beta: float, n: int = 1000):
        xs = list(range(1, n + 1))
        return xs, [k * x ** beta for x in xs]

    def test_recovers_an_exact_power_law(self) -> None:
        xs, ys = self._power_law(15.2, 0.601)
        k, beta = fit_beta(xs, ys, min_n=200)
        assert abs(beta - 0.601) < 1e-6
        assert abs(k - 15.2) < 1e-6

    def test_the_window_excludes_the_head(self) -> None:
        """A curve that is linear up to n=50 and Heaps-like after.
        Fitting from n=1 must report the steeper slope; that is the
        defect this window exists to avoid."""
        xs = list(range(1, 1001))
        ys = [float(x) if x <= 50 else 50 * (x / 50) ** 0.6 for x in xs]
        _, from_head = fit_beta(xs, ys, min_n=1)
        _, from_tail = fit_beta(xs, ys, min_n=200)
        assert from_head > from_tail
        assert abs(from_tail - 0.6) < 1e-6

    def test_non_positive_and_short_curves_are_dropped(self) -> None:
        xs, ys = self._power_law(10.0, 0.5, n=300)
        ys[0] = 0.0  # a leading zero must not reach the logarithm
        k, beta = fit_beta(xs, ys, min_n=200)
        assert abs(beta - 0.5) < 1e-6

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_beta([1, 2, 3], [1.0, 2.0, 3.0], min_n=200)


@pytest.mark.xfail(strict=True, reason='coverage measure is a skeleton')
class TestInstancesByTreatment:
    """Like ``labels_by_treatment`` but counting, not collapsing.

    The coverage measure needs both: distinct labels answer "does this
    treatment use vocabulary we have seen", instance counts answer
    "how much of what it says do we already understand", and the two
    differ whenever a treatment repeats a label.
    """

    def test_repeats_are_counted_not_collapsed(self) -> None:
        got = instances_by_treatment([_ann('a', 'Pileus')] * 3
                                     + [_ann('a', 'Stipe')])
        assert got == {'a': collections.Counter({'Pileus': 3, 'Stipe': 1})}

    def test_canonicalizer_applies(self) -> None:
        got = instances_by_treatment(
            [_ann('a', 'Colonies'), _ann('a', 'Colony')],
            canonicalizer={'Colonies': 'Colony'},
        )
        assert got == {'a': collections.Counter({'Colony': 2})}

    def test_callable_canonicalizer_applies(self) -> None:
        got = instances_by_treatment(
            [_ann('a', 'Sexual state')],
            canonicalizer=lambda label: label.replace('state', 'morph'),
        )
        assert got == {'a': collections.Counter({'Sexual morph': 1})}


@pytest.mark.xfail(strict=True, reason='coverage measure is a skeleton')
class TestOutOfSampleCoverage:
    """The measurement that validated the method.

    Every earlier coverage number was **in-sample** — a permutation
    over one round, asking what its own treatments look like to a
    vocabulary built from the rest of that round.  This asks the
    honest question instead: given the vocabulary learned from one
    round, how much of a *different* round does it already cover?

    Round 6 answered 91.3 % against an in-sample prediction of 91.7 %.
    """

    def _held_out(self):
        return {
            'a': {'Pileus', 'Stipe'},        # both known
            'b': {'Pileus', 'Basidia'},      # half known
            'c': {'Cystidia'},               # unknown
            'd': set(),                      # sampled, produced nothing
        }

    def _instances(self):
        return {
            'a': collections.Counter({'Pileus': 3, 'Stipe': 1}),
            'b': collections.Counter({'Pileus': 1, 'Basidia': 9}),
            'c': collections.Counter({'Cystidia': 2}),
        }

    def test_everything_known_is_full_coverage(self) -> None:
        got = out_of_sample_coverage(
            {'Pileus', 'Stipe'}, {'a': {'Pileus', 'Stipe'}},
            {'a': collections.Counter({'Pileus': 2})},
        )
        assert got.type_coverage == 1.0
        assert got.instance_coverage == 1.0
        assert got.pooled_instance_coverage == 1.0
        assert got.novel_labels == frozenset()

    def test_nothing_known_is_zero_coverage(self) -> None:
        got = out_of_sample_coverage(set(), {'a': {'Pileus'}},
                                     {'a': collections.Counter({'Pileus': 1})})
        assert got.type_coverage == 0.0
        assert got.pooled_instance_coverage == 0.0
        assert got.novel_labels == frozenset({'Pileus'})

    def test_mean_over_treatments_differs_from_pooled(self) -> None:
        """**The reason both are reported.**  Treatment `b` carries 10
        instances to `a`'s 4, so pooling weights it more heavily than
        the per-treatment mean does.  Reporting one number invites the
        reader to assume it is the other."""
        got = out_of_sample_coverage(
            {'Pileus', 'Stipe'}, self._held_out(), self._instances(),
        )
        # per-treatment: a=1.0, b=1/10, c=0.0  -> mean 0.3667
        assert abs(got.instance_coverage - (1.0 + 0.1 + 0.0) / 3) < 1e-9
        # pooled: (4 + 1 + 0) known of 16 total
        assert abs(got.pooled_instance_coverage - 5 / 16) < 1e-9
        assert got.instance_coverage != got.pooled_instance_coverage

    def test_type_coverage_is_the_mean_over_treatments(self) -> None:
        got = out_of_sample_coverage(
            {'Pileus', 'Stipe'}, self._held_out(), self._instances(),
        )
        assert abs(got.type_coverage - (1.0 + 0.5 + 0.0) / 3) < 1e-9

    def test_label_less_treatments_are_excluded_not_scored_zero(
            self) -> None:
        """Treatment `d` was sampled and produced nothing.  It carries
        no evidence about coverage; scoring it 0 would understate, and
        scoring it 1 would overstate."""
        got = out_of_sample_coverage(
            {'Pileus', 'Stipe'}, self._held_out(), self._instances(),
        )
        assert got.treatments == 3
        assert got.instances == 16

    def test_novel_labels_are_reported(self) -> None:
        got = out_of_sample_coverage(
            {'Pileus', 'Stipe'}, self._held_out(), self._instances(),
        )
        assert got.novel_labels == frozenset({'Basidia', 'Cystidia'})

    def test_instances_are_optional(self) -> None:
        """A caller with only distinct labels still gets the type
        measure; the instance fields say None rather than lying."""
        got = out_of_sample_coverage({'Pileus'}, {'a': {'Pileus'}})
        assert got.type_coverage == 1.0
        assert got.instance_coverage is None
        assert got.pooled_instance_coverage is None
        assert got.instances == 0

    def test_empty_held_out_set_does_not_divide_by_zero(self) -> None:
        got = out_of_sample_coverage({'Pileus'}, {})
        assert got.treatments == 0
        assert got.type_coverage is None
