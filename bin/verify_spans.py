#!/usr/bin/env python3
"""Skeleton — implementation lands in the next commit (TDD)."""

import dataclasses
from typing import Any, Dict, List, Tuple


@dataclasses.dataclass(frozen=True)
class SpanCheck:
    treatment_id: str
    field: str
    index: int
    ok: bool
    reason: str


def check_treatment(treatment: Dict[str, Any],
                    server: Any) -> List[SpanCheck]:
    raise NotImplementedError


def summarise(checks: List[SpanCheck]) -> Tuple[int, int, float]:
    raise NotImplementedError
