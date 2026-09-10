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
