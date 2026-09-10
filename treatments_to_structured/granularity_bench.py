#!/usr/bin/env python3
"""R1: is Mistral-per-clause faster than Mistral-per-description?

See ``granularity_bench_test.py`` for the design and the measurements
that produced it.

Three arms, because the two the card proposed cannot separate the
mechanism from the granularity:

  * ``whole``        — one call per description, as the pipeline runs today
  * ``clause_seq``   — one call per clause, sequential
  * ``clause_batch`` — one call per batch of clauses

``clause_seq`` is expected to *lose*.  87.1 % of descriptions are
already exactly one model call, and per-clause emits roughly the same
total output tokens (9.4 clauses × ~45 against ~540 for a description),
so granularity on its own has nothing to give.  What batching gives is
one decode loop shared across many short sequences — which uniform
clause prompts permit and one long description does not.  Running the
losing arm is what lets the result say *which* of the two it was.

Generation is injected so the arithmetic is testable without a GPU.
"""

import json
import time
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

#: Prompt scaffold for one clause.  Phase 1 already supplies the key
#: (``attribute_path`` is depth 1 on 98 % of spans), so the clause arm
#: asks the model to fill a slot rather than to parse a description.
CLAUSE_PROMPT = (
    'Extract the sub-attributes of {label} from this phrase as JSON.\n'
    '{text}'
)

#: The card's go/no-go: under this multiple of the incumbent, the
#: per-clause premise fails.
KILL_SPEEDUP = 3.0


@dataclass(frozen=True)
class Workload:
    """One arm's work: the prompts, and how many go per model call."""

    arm: str
    items: List[str] = field(default_factory=list)
    batch_size: int = 1


@dataclass(frozen=True)
class Timing:
    """What one arm cost.  ``failures`` is carried rather than raised so
    a bad prompt does not cost the whole run."""

    #: ``items`` counts prompts that SUCCEEDED and ``failures`` counts
    #: model calls that raised — a failed call drops its whole batch, so
    #: prompts lost is ``len(workload.items) - items`` rather than
    #: anything derivable from ``failures`` alone.
    arm: str
    items: int = 0
    calls: int = 0
    seconds: float = 0.0
    output_chars: int = 0
    failures: int = 0
    #: One entry per prompt in the workload, positionally aligned with
    #: it -- ``None`` where the call that carried it failed.  Retained
    #: so ``clause_seq`` and ``clause_batch`` can be compared: batching
    #: changes cuBLAS kernel selection, which can flip an argmax at a
    #: near-tie, so identical output is expected but not guaranteed and
    #: has to be measured rather than assumed.
    outputs: Tuple[Optional[str], ...] = ()


def build_workloads(
    treatment: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
) -> List[Workload]:
    """Assemble the three arms for one treatment.

    The two clause arms are given the *same* item list deliberately.
    They differ only in ``batch_size``, so any difference between them
    is attributable to batching and to nothing else.

    A treatment with no spans contributes no clause work rather than
    zero-cost clause work: only 3.4 % of the corpus carries
    Claude-labelled spans, and scoring the remainder as instant would
    flatter the clause arms precisely where the corpus is thinnest.
    """
    description = str(treatment.get('description') or '').strip()
    whole: List[str] = [description] if description else []

    clauses: List[str] = []
    for span in spans:
        text = str(span.get('source_text') or '').strip()
        if not text:
            continue
        label = str(span.get('feature_label') or '').strip() or 'this feature'
        clauses.append(CLAUSE_PROMPT.format(label=label, text=text))

    return [
        Workload('whole', whole, 1),
        Workload('clause_seq', list(clauses), 1),
        Workload('clause_batch', list(clauses), batch_size),
    ]


def run_workload(
    workload: Workload,
    generate_fn: Callable[[List[str]], List[str]],
) -> Timing:
    """Run one arm, timing it.

    ``generate_fn`` takes a list of prompts and returns a list of
    decoded strings.  A batch that raises is counted into ``failures``
    and the run continues: a benchmark that aborts on one bad prompt
    reports nothing, and the arms have to stay comparable.
    """
    size = max(1, workload.batch_size)
    items = list(workload.items)
    if not items:
        return Timing(workload.arm)

    calls = done = failures = chars = 0
    outputs: List[Optional[str]] = []
    started = time.perf_counter()
    for i in range(0, len(items), size):
        batch = items[i:i + size]
        calls += 1
        try:
            out = generate_fn(batch)
        except Exception:                    # noqa: BLE001 - see docstring
            failures += 1
            # Placeholders, not omissions: position is the join key
            # between clause_seq and clause_batch, so a dropped entry
            # would shift every later output left and compare unrelated
            # clauses while looking clean.
            outputs.extend([None] * len(batch))
            continue
        done += len(batch)
        chars += sum(len(str(o)) for o in out)
        outputs.extend(str(o) for o in out)
    seconds = time.perf_counter() - started

    return Timing(workload.arm, items=done, calls=calls, seconds=seconds,
                  output_chars=chars, failures=failures,
                  outputs=tuple(outputs))


def summarize(
    timings: Sequence[Timing],
    *,
    treatments: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Report per arm, normalised against the ``whole`` incumbent.

    ``whole`` is the denominator because it is the path that was called
    prohibitively slow; the question is whether anything beats *it*.

    ``seconds_per_treatment`` is reported rather than per item, because
    per item would flatter the clause arms 9.4-fold for nothing — a
    treatment processed end to end is the only comparable unit.
    """
    by_arm = {t.arm: t for t in timings}
    base = by_arm.get('whole')
    baseline = base.seconds if base and base.seconds > 0 else None

    report: Dict[str, Dict[str, Any]] = {}
    for t in timings:
        speedup = (baseline / t.seconds
                   if baseline is not None and t.seconds > 0 else None)
        row: Dict[str, Any] = {
            'items': t.items,
            'calls': t.calls,
            'seconds': t.seconds,
            'output_chars': t.output_chars,
            'failures': t.failures,
            'speedup': speedup,
            # An arm that dropped work has not earned a verdict, however
            # fast the surviving prompts were.
            'passes': (speedup is not None
                       and speedup >= KILL_SPEEDUP
                       and t.failures == 0),
        }
        if treatments:
            row['seconds_per_treatment'] = t.seconds / treatments
        report[t.arm] = row
    return report


def compare_arms(
    left: Timing,
    right: Timing,
) -> Dict[str, Any]:
    """Compare two arms' outputs position by position.

    Only meaningful for ``clause_seq`` vs ``clause_batch``: they are
    built from the same items, so a difference is attributable to
    batching.  ``whole`` differs by construction and is not comparable
    this way.

    Three rungs, deliberately not one edit distance: **exact** (byte
    identical), **structural** (parses to the same object — reordered
    keys and whitespace are the same JSON, and an edit distance would
    have scored that alongside a changed value), and **differ**.  Two
    populations are kept out of ``differ`` so neither hides the other:
    ``unparseable``, where a side emitted something that is not JSON,
    and ``incomparable``, where a side's call failed.
    """
    if len(left.outputs) != len(right.outputs):
        raise ValueError(
            f'arms are not aligned: {left.arm} has {len(left.outputs)} '
            f'outputs, {right.arm} has {len(right.outputs)}.  Comparing '
            f'them would silently pair unrelated clauses.')

    n = exact = structural = differ = unparseable = incomparable = 0
    for a, b in zip(left.outputs, right.outputs):
        if a is None or b is None:
            incomparable += 1
            continue
        n += 1
        if a == b:
            exact += 1
            continue
        try:
            pa, pb = json.loads(a), json.loads(b)
        except (ValueError, TypeError):
            unparseable += 1
            continue
        if pa == pb:
            structural += 1
        else:
            differ += 1

    return {
        'left': left.arm,
        'right': right.arm,
        'n': n,
        'exact': exact,
        'structural': structural,
        'differ': differ,
        'unparseable': unparseable,
        'incomparable': incomparable,
        'exact_rate': (exact / n) if n else None,
        # The number that answers "is C doing the same work as B".  A
        # low exact rate with a high agree rate is benign: kernel
        # non-determinism reordered the serialisation, not the content.
        'agree_rate': ((exact + structural) / n) if n else None,
    }
