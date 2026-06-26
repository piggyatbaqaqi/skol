"""Tests for treatments_to_structured.select.

Pure-logic coverage — no CouchDB.  See
docs/schema_constrained_pipeline.md §10.4 deliverable 2.
"""

import random
from typing import List, Tuple

import pytest

from treatments_to_structured.select import (
    parse_band_spec,
    select_treatments,
)


# ---------------------------------------------------------------------------
# parse_band_spec
# ---------------------------------------------------------------------------


class TestParseBandSpec:
    """Parse ``'<name>:<count>,...'`` into ``[(name, count), ...]``."""

    def test_single_band(self) -> None:
        assert parse_band_spec("all:100") == [("all", 100)]

    def test_three_bands(self) -> None:
        assert parse_band_spec("low:25,mid:50,high:25") == [
            ("low", 25), ("mid", 50), ("high", 25),
        ]

    def test_arbitrary_count_of_bands(self) -> None:
        """The format isn't restricted to three bands."""
        assert parse_band_spec("a:1,b:2,c:3,d:4,e:5") == [
            ("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5),
        ]

    def test_whitespace_tolerated_around_tokens(self) -> None:
        assert parse_band_spec("low : 25 , mid : 50") == [
            ("low", 25), ("mid", 50),
        ]

    def test_rejects_zero_count(self) -> None:
        with pytest.raises(ValueError):
            parse_band_spec("low:0")

    def test_rejects_negative_count(self) -> None:
        with pytest.raises(ValueError):
            parse_band_spec("low:-5")

    def test_rejects_missing_colon(self) -> None:
        with pytest.raises(ValueError) as exc:
            parse_band_spec("low,mid:50")
        assert "missing ':<count>'" in str(exc.value)

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError):
            parse_band_spec("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError):
            parse_band_spec("   ")

    def test_rejects_non_integer_count(self) -> None:
        with pytest.raises(ValueError) as exc:
            parse_band_spec("low:five")
        assert "non-integer" in str(exc.value)

    def test_rejects_empty_band_name(self) -> None:
        with pytest.raises(ValueError) as exc:
            parse_band_spec(":25")
        assert "empty band name" in str(exc.value)


# ---------------------------------------------------------------------------
# select_treatments
# ---------------------------------------------------------------------------


def _scored(n: int) -> List[Tuple[str, float]]:
    """Make a scored list: ``id_0`` → 0.0, ..., ``id_<n-1>`` → ``<n-1>.0``."""
    return [(f'id_{i}', float(i)) for i in range(n)]


class TestSelectTreatments:
    """Sample from score-banded populations."""

    def test_single_band_selects_quota_ids(self) -> None:
        scored = _scored(100)
        rng = random.Random(42)
        result = select_treatments(scored, [("all", 10)], rng)
        assert len(result) == 10
        all_ids = {doc_id for doc_id, _ in scored}
        assert set(result) <= all_ids

    def test_three_bands_partition_by_score(self) -> None:
        """Each band's selected IDs come from its score quantile."""
        scored = _scored(99)  # divides cleanly into 3 × 33
        rng = random.Random(42)
        result = select_treatments(
            scored, [("low", 1), ("mid", 1), ("high", 1)], rng,
        )
        assert len(result) == 3
        score_map = dict(scored)
        s_low, s_mid, s_high = [score_map[id_] for id_ in result]
        # Slices: ids 0–32 (low), 33–65 (mid), 66–98 (high)
        assert s_low < 33
        assert 33 <= s_mid < 66
        assert s_high >= 66

    def test_uneven_population_distributes_remainder_to_final_band(
        self,
    ) -> None:
        """Population 100, 3 bands: slices end up 33 / 33 / 34."""
        scored = _scored(100)
        rng = random.Random(42)
        # Ask for all 100 to verify slice sizes by quota acceptance.
        result = select_treatments(
            scored, [("low", 33), ("mid", 33), ("high", 34)], rng,
        )
        assert len(result) == 100

    def test_deterministic_with_same_seed(self) -> None:
        scored = _scored(100)
        r1 = select_treatments(scored, [("all", 10)], random.Random(42))
        r2 = select_treatments(scored, [("all", 10)], random.Random(42))
        assert r1 == r2

    def test_different_seeds_yield_different_samples(self) -> None:
        scored = _scored(100)
        r1 = select_treatments(scored, [("all", 10)], random.Random(1))
        r2 = select_treatments(scored, [("all", 10)], random.Random(2))
        assert r1 != r2

    def test_total_quota_larger_than_population_raises(self) -> None:
        scored = _scored(5)
        rng = random.Random(42)
        with pytest.raises(ValueError) as exc:
            select_treatments(scored, [("all", 10)], rng)
        assert '10' in str(exc.value)
        assert '5' in str(exc.value)

    def test_band_quota_larger_than_slice_population_raises(self) -> None:
        """If we ask for 50 from a band's slice that has ~33, fail."""
        scored = _scored(99)
        rng = random.Random(42)
        with pytest.raises(ValueError) as exc:
            select_treatments(
                scored, [("low", 50), ("mid", 1), ("high", 1)], rng,
            )
        assert 'low' in str(exc.value)

    def test_empty_scored_raises(self) -> None:
        rng = random.Random(42)
        with pytest.raises(ValueError):
            select_treatments([], [("all", 1)], rng)

    def test_empty_band_specs_raises(self) -> None:
        rng = random.Random(42)
        with pytest.raises(ValueError):
            select_treatments(_scored(10), [], rng)

    def test_output_order_follows_band_specs(self) -> None:
        """First band's selections come first in output."""
        scored = _scored(99)
        rng = random.Random(42)
        result = select_treatments(
            scored, [("low", 2), ("high", 2)], rng,
        )
        # n_bands=2: low slice = id_0..48, high slice = id_49..98
        score_map = dict(scored)
        for id_ in result[:2]:
            assert score_map[id_] < 49
        for id_ in result[2:]:
            assert score_map[id_] >= 49

    def test_total_quota_equals_population_returns_everyone(self) -> None:
        """Quota = population is the edge case where every ID is sampled."""
        scored = _scored(10)
        rng = random.Random(42)
        result = select_treatments(scored, [("all", 10)], rng)
        assert sorted(result) == sorted(id_ for id_, _ in scored)

    def test_ties_in_score_handled(self) -> None:
        """Multiple treatments with identical scores all get banded
        somewhere — no crash, no duplication."""
        scored = [(f'id_{i}', 5.0) for i in range(10)]
        rng = random.Random(42)
        result = select_treatments(
            scored, [("low", 2), ("high", 2)], rng,
        )
        assert len(result) == 4
        assert len(set(result)) == 4  # no duplicates
