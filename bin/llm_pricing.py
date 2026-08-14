#!/usr/bin/env python3
"""Claude API list prices — the single source of truth.

Skeleton: the shape is here so callers and tests can import it; the
behaviour lands in the follow-up commit.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Iterator, List, Mapping


class UnknownModelError(ValueError):
    """Raised when a model has no pricing entry."""


@dataclasses.dataclass(frozen=True)
class Price:
    """USD per 1,000,000 tokens, input and output."""

    input: float
    output: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        raise NotImplementedError


class ModelPricing:
    """An immutable registry of per-model list prices."""

    def __init__(self, prices: Mapping[str, Price]) -> None:
        self._prices: Dict[str, Price] = dict(prices)

    def __contains__(self, model: object) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def models(self) -> List[str]:
        raise NotImplementedError

    def for_model(self, model: str) -> Price:
        raise NotImplementedError


PRICING = ModelPricing({})
