#!/usr/bin/env python3
"""R1: is Mistral-per-clause faster than Mistral-per-description?

Skeleton.  See ``granularity_bench_test.py`` for the design and the
measurements that produced it.

Three arms, because the two the card proposed cannot separate the
mechanism from the granularity:

  * ``whole``        — one call per description, as the pipeline runs today
  * ``clause_seq``   — one call per clause, sequential
  * ``clause_batch`` — one call per batch of clauses

Generation is injected so the arithmetic is testable without a GPU.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

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

    arm: str
    items: int = 0
    calls: int = 0
    seconds: float = 0.0
    output_chars: int = 0
    failures: int = 0


def build_workloads(
    treatment: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
) -> List[Workload]:
    """Assemble the three arms for one treatment."""
    raise NotImplementedError


def run_workload(
    workload: Workload,
    generate_fn: Callable[[List[str]], List[str]],
) -> Timing:
    """Run one arm, timing it.  ``generate_fn`` takes a list of prompts
    and returns a list of decoded strings."""
    raise NotImplementedError


def summarize(
    timings: Sequence[Timing],
    *,
    treatments: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Report per arm, normalised against the ``whole`` incumbent."""
    raise NotImplementedError
