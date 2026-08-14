#!/usr/bin/env python3
"""Claude API list prices — the single source of truth.

Import ``PRICING`` and call ``for_model()``; do not copy the table.
Two hand-synced copies (``llm_annotate_features`` and ``llm_relabel``)
are what let every Opus row sit a generation out of date at $15/$75
while the real rate was $5/$25, quoting operators roughly 3x the true
cost of a run.

There is deliberately **no fallback** for an unpriced model.  An
estimate is the number someone commits budget on, so an approximation
is worse than a stopped run: the previous behaviour substituted a
hardcoded rate for anything it didn't recognise, which is precisely
how the stale rows survived — the wrong figure was indistinguishable
from a right one.  Add the model below instead.

Rates verified against the model catalog 2026-08-14.  These are list
prices and they move; treat any row older than a model release as
suspect.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Iterator, List, Mapping


class UnknownModelError(ValueError):
    """Raised when a model has no pricing entry.

    Subclasses ``ValueError`` so callers that already catch that keep
    working.
    """


@dataclasses.dataclass(frozen=True)
class Price:
    """USD per 1,000,000 tokens, input and output.

    Frozen: a rate is a fact about the world, not mutable state, and
    the registry hands out shared instances.
    """

    input: float
    output: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Total USD for a given token split."""
        return (
            input_tokens * self.input + output_tokens * self.output
        ) / 1_000_000


class ModelPricing:
    """An immutable registry of per-model list prices."""

    def __init__(self, prices: Mapping[str, Price]) -> None:
        # Copy: a caller mutating the mapping it passed in must not
        # silently edit the shared registry.
        self._prices: Dict[str, Price] = dict(prices)

    def __contains__(self, model: object) -> bool:
        return model in self._prices

    def __iter__(self) -> Iterator[str]:
        return iter(self._prices)

    def __len__(self) -> int:
        return len(self._prices)

    def models(self) -> List[str]:
        """Known model IDs, sorted — for error messages and listings."""
        return sorted(self._prices)

    def for_model(self, model: str) -> Price:
        """The price for ``model``.

        Raises:
            UnknownModelError: if the model has no entry.  This is the
                designed behaviour, not a defect — see the module
                docstring on why there is no fallback.
        """
        try:
            return self._prices[model]
        except KeyError:
            raise UnknownModelError(
                f"no pricing entry for model {model!r}; refusing to "
                f"estimate a cost that would be a guess. Known "
                f"models: {', '.join(self.models())}. Add the model "
                f"to PRICING in bin/llm_pricing.py with its list "
                f"price."
            ) from None


_OPUS = Price(input=5.00, output=25.00)
# Sonnet 5 is listed at its standard rate, not the introductory
# $2/$10 that lapses 2026-08-31 — an estimate that silently expires is
# worse than one that over-quotes slightly.
_SONNET = Price(input=3.00, output=15.00)
_HAIKU = Price(input=1.00, output=5.00)
_FABLE = Price(input=10.00, output=50.00)

PRICING = ModelPricing({
    'claude-haiku-4-5': _HAIKU,
    # The catalog lists the alias and the dated ID; accept either.
    'claude-haiku-4-5-20251001': _HAIKU,
    'claude-sonnet-4-6': _SONNET,
    'claude-sonnet-5': _SONNET,
    'claude-opus-4-6': _OPUS,
    'claude-opus-4-7': _OPUS,
    'claude-opus-4-8': _OPUS,
    'claude-opus-5': _OPUS,
    'claude-fable-5': _FABLE,
})
