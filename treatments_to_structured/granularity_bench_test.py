#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.granularity_bench`` (R1).

R1 asks whether running Mistral per *clause* beats running it per
*description*.  The card framed that as two arms.  Reading the existing
generator first says it has to be three, and says why:

  * ``_translate_description`` chunks at 3 840 chars, so **87.1 % of
    descriptions are already exactly one model call** (mean 1.23).
  * The existing 1 285 outputs are **median 294 / mean 439 generated
    tokens**, under ``max_new_tokens=1024``, ``do_sample=False``.
  * ``_raw_generate`` tokenizes one prompt and calls ``model.generate``
    — **batch size 1**, always.

Per-clause therefore emits roughly the *same* total output tokens
(9.4 clauses x ~45 ≈ 420, against ~540 for a whole description).
Sequentially it cannot win.  The only thing that can win is batching,
which uniform short prompts permit and one long prompt does not.  So
the harness measures A (whole, as-is), B (per-clause, sequential) and
C (per-clause, batched), and B exists precisely to show that
granularity alone is not the lever.

**B and C must be compared, not assumed equal.**  Greedy decoding is
mathematically deterministic, but batching changes cuBLAS kernel
selection, reductions run in a different order, and at a step where the
top two logits are nearly tied that flips the argmax — after which the
continuation diverges.  Arm C claims "the same work, faster"; if the
work is not the same, part of the speed-up is an unpriced quality
change, and R4c will score only one of these paths.  So the outputs are
retained and compared.

**Comparison is exact-then-structural, not edit distance.**  Edit
distance would score reordered JSON keys (harmless — same object) and a
changed value (serious) on the same scale.

**Generation is injected.**  Everything here is exercised against a
fake generator with a controllable cost model, so the arithmetic, the
report and the failure handling are testable without a GPU — and the
GPU driver stays a thin shell over the same functions.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.granularity_bench import (  # noqa: E402
    Timing,
    Workload,
    build_workloads,
    compare_arms,
    run_workload,
    summarize,
)


def _spans(n: int, chars: int = 100) -> list:
    return [{'source_text': 'x' * chars, 'feature_label': f'Organ{i}'}
            for i in range(n)]


def _treatment(chars: int = 1300) -> dict:
    return {'_id': 'taxon_a', 'description': 'y' * chars}


class _FakeGen:
    """A generator whose cost is dominated by output tokens, as a real
    autoregressive decode is, with a fixed per-call overhead and a
    batch that shares one decode loop."""

    def __init__(self, per_call: float = 0.5, per_token: float = 0.02,
                 tokens_per_item: int = 45):
        self.per_call = per_call
        self.per_token = per_token
        self.tokens_per_item = tokens_per_item
        self.calls = 0
        self.items = 0
        self.max_batch = 0
        self.elapsed = 0.0

    def __call__(self, prompts: list) -> list:
        self.calls += 1
        self.items += len(prompts)
        self.max_batch = max(self.max_batch, len(prompts))
        # One decode loop for the batch; longest item sets the length.
        self.elapsed += self.per_call + self.per_token * self.tokens_per_item
        return ['{}'] * len(prompts)


class TestBuildWorkloads:
    def test_three_arms_not_two(self) -> None:
        """The card said two.  87.1 % of descriptions are already one
        call, so a sequential per-clause arm cannot be the lever; the
        harness must be able to show that rather than assume it."""
        w = build_workloads(_treatment(), _spans(9))
        assert [x.arm for x in w] == ['whole', 'clause_seq', 'clause_batch']

    def test_the_whole_arm_is_one_item_per_description(self) -> None:
        w = {x.arm: x for x in build_workloads(_treatment(), _spans(9))}
        assert len(w['whole'].items) == 1

    def test_both_clause_arms_carry_the_same_items(self) -> None:
        """B and C differ only in batch size.  If they differed in
        content the comparison would measure two things at once."""
        w = {x.arm: x for x in build_workloads(_treatment(), _spans(9))}
        assert w['clause_seq'].items == w['clause_batch'].items
        assert len(w['clause_seq'].items) == 9

    def test_only_the_batch_arm_batches(self) -> None:
        w = {x.arm: x for x in build_workloads(_treatment(), _spans(9),
                                               batch_size=8)}
        assert w['whole'].batch_size == 1
        assert w['clause_seq'].batch_size == 1
        assert w['clause_batch'].batch_size == 8

    def test_the_clause_prompt_carries_its_label(self) -> None:
        """Phase 1 supplies the key -- attribute_path is depth 1 on
        98 % of spans -- so the clause prompt must say which organ it
        is filling, or the arm is not testing the proposal."""
        w = {x.arm: x for x in build_workloads(_treatment(), _spans(2))}
        assert any('Organ0' in it for it in w['clause_seq'].items)

    def test_a_treatment_with_no_spans_yields_no_clause_work(self) -> None:
        """3.4 % coverage: most treatments have no Claude-labelled
        spans at all, and a benchmark that silently scored them as
        instant would flatter the clause arms."""
        w = {x.arm: x for x in build_workloads(_treatment(), [])}
        assert w['clause_seq'].items == []
        assert w['whole'].items != []

    def test_an_empty_description_yields_no_whole_work(self) -> None:
        w = {x.arm: x for x in build_workloads({'_id': 'taxon_a'},
                                               _spans(3))}
        assert w['whole'].items == []


class TestRunWorkload:
    def test_sequential_makes_one_call_per_item(self) -> None:
        gen = _FakeGen()
        run_workload(Workload('clause_seq', ['a'] * 9, batch_size=1), gen)
        assert gen.calls == 9 and gen.max_batch == 1

    def test_batched_makes_ceil_calls(self) -> None:
        gen = _FakeGen()
        run_workload(Workload('clause_batch', ['a'] * 9, batch_size=4), gen)
        assert gen.calls == 3 and gen.max_batch == 4

    def test_a_partial_final_batch_is_not_dropped(self) -> None:
        gen = _FakeGen()
        run_workload(Workload('clause_batch', ['a'] * 9, batch_size=4), gen)
        assert gen.items == 9

    def test_timing_records_items_and_calls(self) -> None:
        t = run_workload(Workload('clause_batch', ['a'] * 9, 4), _FakeGen())
        assert t.arm == 'clause_batch'
        assert t.items == 9 and t.calls == 3
        assert t.seconds > 0

    def test_an_empty_workload_costs_nothing_and_does_not_divide_by_zero(
            self) -> None:
        t = run_workload(Workload('clause_seq', [], 1), _FakeGen())
        assert t.items == 0 and t.calls == 0
        assert t.seconds == 0.0

    def test_a_failing_generation_is_counted_not_fatal(self) -> None:
        """A benchmark that aborts on one bad prompt reports nothing.
        Failures must land in the timing so the arm is comparable."""
        def boom(prompts: list) -> list:
            raise RuntimeError('cuda oom')
        t = run_workload(Workload('clause_batch', ['a'] * 4, 2), boom)
        assert t.failures == 2 and t.items == 0

    def test_output_tokens_are_recorded(self) -> None:
        """Output tokens are what an autoregressive decode actually
        costs, so the report must be able to normalise by them rather
        than only by wall clock."""
        t = run_workload(Workload('whole', ['a'], 1),
                         lambda p: ['{"a": 1}' * 10] * len(p))
        assert t.output_chars > 0


class TestSummarize:
    def _timings(self) -> list:
        return [Timing('whole', items=50, calls=62, seconds=100.0,
                       output_chars=66000, failures=0),
                Timing('clause_seq', items=470, calls=470, seconds=90.0,
                       output_chars=21000, failures=0),
                Timing('clause_batch', items=470, calls=59, seconds=20.0,
                       output_chars=21000, failures=0)]

    def test_speedup_is_against_the_whole_arm(self) -> None:
        """`whole` is the incumbent that was called prohibitively slow,
        so it is the denominator."""
        r = summarize(self._timings())
        assert r['whole']['speedup'] == pytest.approx(1.0)
        assert r['clause_batch']['speedup'] == pytest.approx(5.0)

    def test_the_verdict_uses_the_cards_3x_kill_criterion(self) -> None:
        r = summarize(self._timings())
        assert r['clause_batch']['passes'] is True
        assert r['clause_seq']['passes'] is False

    def test_seconds_per_treatment_is_reported_not_per_item(self) -> None:
        """Per-item would flatter the clause arms 9.4-fold for free.
        The comparable unit is one treatment processed end to end."""
        r = summarize(self._timings(), treatments=50)
        assert r['clause_batch']['seconds_per_treatment'] == pytest.approx(0.4)
        assert r['whole']['seconds_per_treatment'] == pytest.approx(2.0)

    def test_an_arm_with_failures_is_flagged_not_silently_ranked(
            self) -> None:
        t = self._timings()
        t[2] = Timing('clause_batch', 400, 50, 20.0, 18000, failures=70)
        r = summarize(t)
        assert r['clause_batch']['failures'] == 70
        assert r['clause_batch']['passes'] is False

    def test_a_zero_second_baseline_does_not_divide_by_zero(self) -> None:
        r = summarize([Timing('whole', 0, 0, 0.0, 0, 0),
                       Timing('clause_batch', 9, 2, 1.0, 100, 0)])
        assert r['clause_batch']['speedup'] is None


class TestOutputsAreRetained:
    """`output_chars` alone made a B-vs-C comparison impossible by
    construction.  The outputs themselves are ~94 KB for 470 clauses."""

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_outputs_come_back_in_order(self) -> None:
        out = run_workload(Workload('clause_seq', ['a', 'b', 'c'], 1),
                           lambda p: [x.upper() for x in p])
        assert out.outputs == ('A', 'B', 'C')

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_outputs_stay_aligned_across_batches(self) -> None:
        """Position is the join key between B and C, so batching must
        not reorder or regroup them."""
        out = run_workload(Workload('clause_batch', list('abcde'), 2),
                           lambda p: [x.upper() for x in p])
        assert out.outputs == ('A', 'B', 'C', 'D', 'E')

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_a_failed_batch_leaves_None_placeholders(self) -> None:
        """Dropping the failures instead would shift every later output
        left by one and silently compare unrelated clauses."""
        calls = {'n': 0}

        def flaky(prompts: list) -> list:
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('cuda oom')
            return [x.upper() for x in prompts]

        out = run_workload(Workload('clause_batch', list('abcdef'), 2), flaky)
        assert out.outputs == ('A', 'B', None, None, 'E', 'F')

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_outputs_length_always_matches_the_workload(self) -> None:
        def boom(prompts: list) -> list:
            raise RuntimeError('nope')
        out = run_workload(Workload('clause_seq', list('abc'), 1), boom)
        assert len(out.outputs) == 3

    def test_an_empty_workload_has_no_outputs(self) -> None:
        assert run_workload(Workload('clause_seq', [], 1),
                            lambda p: []).outputs == ()


class TestCompareArms:
    """Exact match first, structural agreement second, and the genuine
    disagreements kept apart from the unparseable ones."""

    def _t(self, arm: str, outputs: tuple) -> Timing:
        return Timing(arm, items=len(outputs), calls=len(outputs),
                      seconds=1.0, output_chars=0, outputs=outputs)

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_identical_outputs_are_all_exact(self) -> None:
        a = self._t('clause_seq', ('{"a": 1}', '{"b": 2}'))
        r = compare_arms(a, self._t('clause_batch', ('{"a": 1}', '{"b": 2}')))
        assert r['n'] == 2 and r['exact'] == 2
        assert r['exact_rate'] == pytest.approx(1.0)
        assert r['differ'] == 0

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_key_order_and_whitespace_agree_structurally(self) -> None:
        """Same object, different serialisation.  Edit distance would
        call this a difference; it is not one."""
        a = self._t('clause_seq', ('{"a": 1, "b": 2}',))
        b = self._t('clause_batch', ('{"b":2,\n "a":1}',))
        r = compare_arms(a, b)
        assert r['exact'] == 0
        assert r['structural'] == 1
        assert r['agree_rate'] == pytest.approx(1.0)
        assert r['differ'] == 0

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_a_changed_value_is_a_real_disagreement(self) -> None:
        r = compare_arms(self._t('clause_seq', ('{"a": 1}',)),
                         self._t('clause_batch', ('{"a": 2}',)))
        assert r['differ'] == 1 and r['structural'] == 0

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_a_missing_key_is_a_real_disagreement(self) -> None:
        r = compare_arms(self._t('clause_seq', ('{"a": 1, "b": 2}',)),
                         self._t('clause_batch', ('{"a": 1}',)))
        assert r['differ'] == 1

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_unparseable_output_is_not_scored_as_disagreement(self) -> None:
        """A model that emitted prose is a different problem from a
        model that emitted different JSON, and blending them would hide
        whichever is rarer."""
        r = compare_arms(self._t('clause_seq', ('{"a": 1}',)),
                         self._t('clause_batch', ('sorry, I cannot',)))
        assert r['unparseable'] == 1
        assert r['differ'] == 0 and r['structural'] == 0

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_positions_either_side_failed_are_excluded_and_counted(
            self) -> None:
        a = self._t('clause_seq', ('{"a": 1}', None, '{"c": 3}'))
        b = self._t('clause_batch', ('{"a": 1}', '{"b": 2}', None))
        r = compare_arms(a, b)
        assert r['n'] == 1
        assert r['incomparable'] == 2

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_misaligned_lengths_are_refused_not_zipped(self) -> None:
        """zip() would truncate silently and compare clause i against
        clause i for a prefix, reporting a clean result on a broken
        run."""
        with pytest.raises(ValueError):
            compare_arms(self._t('clause_seq', ('{}', '{}')),
                         self._t('clause_batch', ('{}',)))

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_no_comparable_positions_does_not_divide_by_zero(self) -> None:
        r = compare_arms(self._t('clause_seq', (None,)),
                         self._t('clause_batch', (None,)))
        assert r['n'] == 0
        assert r['exact_rate'] is None and r['agree_rate'] is None

    @pytest.mark.xfail(strict=True, reason="R1 outputs/compare_arms not implemented")
    def test_the_arms_compared_are_named_in_the_result(self) -> None:
        """The result gets read months later; which pair it describes
        must not depend on argument order being remembered."""
        r = compare_arms(self._t('clause_seq', ('{}',)),
                         self._t('clause_batch', ('{}',)))
        assert r['left'] == 'clause_seq' and r['right'] == 'clause_batch'
