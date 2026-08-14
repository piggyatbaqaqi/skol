#!/usr/bin/env python3
"""Tests for bin/llm_pricing.py."""
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_pricing  # type: ignore[import]  # noqa: E402
from llm_pricing import (  # type: ignore[import]  # noqa: E402
    PRICING,
    ModelPricing,
    Price,
    UnknownModelError,
)


class TestPrice:
    """A per-MTok rate pair, and the arithmetic on it."""

    def test_fields(self) -> None:
        price = Price(input=5.00, output=25.00)
        assert price.input == 5.00
        assert price.output == 25.00

    def test_is_frozen(self) -> None:
        """Prices are facts about the world, not mutable state — a
        caller must not be able to edit the shared table in place."""
        price = Price(input=5.00, output=25.00)
        with pytest.raises(dataclasses.FrozenInstanceError):
            price.input = 1.00  # type: ignore[misc]

    def test_cost_usd(self) -> None:
        price = Price(input=5.00, output=25.00)
        # 1M input × $5 + 500k output × $25 = 5.00 + 12.50
        assert price.cost_usd(1_000_000, 500_000) == pytest.approx(17.50)

    def test_cost_usd_is_zero_for_no_tokens(self) -> None:
        assert Price(5.00, 25.00).cost_usd(0, 0) == 0.0


class TestModelPricingContainer:
    """Container behaviour, exercised on a small local table so the
    tests don't depend on which models are currently priced."""

    def _table(self) -> ModelPricing:
        return ModelPricing({
            'model-a': Price(1.00, 5.00),
            'model-b': Price(3.00, 15.00),
        })

    def test_contains(self) -> None:
        table = self._table()
        assert 'model-a' in table
        assert 'model-z' not in table

    def test_iter_yields_model_names(self) -> None:
        assert set(self._table()) == {'model-a', 'model-b'}

    def test_len(self) -> None:
        assert len(self._table()) == 2

    def test_models_is_sorted(self) -> None:
        assert self._table().models() == ['model-a', 'model-b']

    def test_for_model_returns_the_price(self) -> None:
        assert self._table().for_model('model-a') == Price(1.00, 5.00)

    def test_constructor_copies_the_mapping(self) -> None:
        """A caller mutating the dict it passed in must not silently
        edit the registry."""
        source = {'model-a': Price(1.00, 5.00)}
        table = ModelPricing(source)
        source['model-z'] = Price(99.00, 99.00)
        assert 'model-z' not in table


class TestUnknownModel:
    """No fallback by design: an unpriced model must fail outright
    rather than be approximated from a neighbour.  A guessed rate is
    indistinguishable from a real one, which is how a 3x-stale figure
    survived unnoticed for months."""

    def test_raises_unknown_model_error(self) -> None:
        with pytest.raises(UnknownModelError):
            PRICING.for_model('claude-future-99-99')

    def test_is_a_valueerror(self) -> None:
        """Callers that already catch ValueError keep working."""
        assert issubclass(UnknownModelError, ValueError)

    def test_error_names_the_offending_model(self) -> None:
        with pytest.raises(UnknownModelError, match='claude-future-99'):
            PRICING.for_model('claude-future-99')

    def test_error_lists_known_models(self) -> None:
        """A typo'd --llm-model is the common case; show what it could
        have meant."""
        with pytest.raises(UnknownModelError, match='claude-opus-4-7'):
            PRICING.for_model('claude-opus-4-7-typo')

    def test_no_catch_all_entry(self) -> None:
        """Guard against a future 'default'/'*' row quietly restoring
        approximate pricing."""
        for key in ('default', '*', 'unknown', ''):
            assert key not in PRICING


class TestShippedTable:
    """The rates themselves.  Verified against the model catalog
    2026-08-14; a row older than a model release is suspect."""

    def test_opus_tier(self) -> None:
        for model in ('claude-opus-4-6', 'claude-opus-4-7',
                      'claude-opus-4-8', 'claude-opus-5'):
            assert PRICING.for_model(model) == Price(5.00, 25.00), model

    def test_sonnet_tier(self) -> None:
        """Sonnet 5 is listed at its standard $3/$15, not the
        introductory $2/$10 that lapses 2026-08-31 — an estimate that
        silently expires is worse than one that over-quotes."""
        for model in ('claude-sonnet-4-6', 'claude-sonnet-5'):
            assert PRICING.for_model(model) == Price(3.00, 15.00), model

    def test_haiku_tier(self) -> None:
        assert PRICING.for_model('claude-haiku-4-5') == Price(1.00, 5.00)

    def test_haiku_dated_id_matches_its_alias(self) -> None:
        """The catalog lists both; a caller passing either must not
        fall through to UnknownModelError."""
        assert (PRICING.for_model('claude-haiku-4-5-20251001')
                == PRICING.for_model('claude-haiku-4-5'))

    def test_fable_tier(self) -> None:
        assert PRICING.for_model('claude-fable-5') == Price(10.00, 50.00)

    def test_covers_both_former_default_models(self) -> None:
        """The two tables this replaced defaulted to different models;
        neither consumer may lose its own default."""
        # llm_annotate_features._DEFAULT_MODEL
        assert 'claude-opus-4-7' in PRICING
        # llm_relabel._DEFAULT_MODEL
        assert 'claude-haiku-4-5-20251001' in PRICING

    def test_no_retired_opus_rate_survives(self) -> None:
        """The specific regression: $15/$75 was Opus's price a
        generation ago and silently outlived it."""
        for model in PRICING:
            price = PRICING.for_model(model)
            assert (price.input, price.output) != (15.00, 75.00), model


_XFAIL_MIGRATION = pytest.mark.xfail(
    reason=(
        "2026-08-14: llm_annotate_features and llm_relabel still carry "
        "their own _PRICING; the migration lands in the next commit."
    ),
    strict=True,
)


class TestSingleSourceOfTruth:
    """The point of the refactor: exactly one table."""

    @_XFAIL_MIGRATION
    def test_consumers_import_rather_than_redefine(self) -> None:
        import llm_annotate_features
        import llm_relabel
        for module in (llm_annotate_features, llm_relabel):
            assert getattr(module, '_PRICING', None) is None, module.__name__
            assert module.PRICING is llm_pricing.PRICING, module.__name__
