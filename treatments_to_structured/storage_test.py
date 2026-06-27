"""Tests for treatments_to_structured.storage.

Phase 1 deliverable 4.5: pins the global golden DB name so
promotion / eval tooling can rely on a single source of truth.
"""

from treatments_to_structured.storage import GOLDEN_FEATURES_DB_NAME


class TestGoldenFeaturesDbName:
    """The single canonical name for the global golden-features DB."""

    def test_matches_skol_golden_naming_convention(self) -> None:
        """Global goldens carry the ``skol_golden_`` prefix and NO
        pipeline-order ``XX_YY`` — matches
        ``skol_golden_ann_hand`` etc."""
        assert GOLDEN_FEATURES_DB_NAME.startswith('skol_golden_')
        # No pipeline-order prefix on a global golden.
        assert '02_50' not in GOLDEN_FEATURES_DB_NAME

    def test_specific_value(self) -> None:
        """If this changes, downstream consumers (promote_to_golden,
        eval tooling) need to be re-deployed in lockstep — the
        regression on this exact string is intentional."""
        assert GOLDEN_FEATURES_DB_NAME == 'skol_golden_features'
