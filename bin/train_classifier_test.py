"""Tests for bin/train_classifier.py helpers.

Focused on the precedence rule for training-database resolution: a
caller-supplied env_config value (which flows from CLI > env > the
experiment doc's ``databases.training`` field > env_config's hardcoded
default) must win over the value baked into MODEL_CONFIGS. This matches
the CLI-parameter priority chain documented in CLAUDE.md and parallels
the doc-field plumbing fixed for ``databases.treatments`` in Step 1.E
of docs/golden_v2_plan.md.

Also pins the v3 schema alignment for ``logistic_sections_v3``: the
class_weights dict must enumerate the full ``ACTIVE_TAGS_19`` label
space so the v3 baseline trains every active class (Step 1.C of
docs/production_v3_plan.md).
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent),
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestors.yedda_tags import (  # type: ignore[import]  # noqa: E402
    ACTIVE_TAGS_19,
)
from train_classifier import (  # type: ignore[import]  # noqa: E402
    MODEL_CONFIGS,
    resolve_training_database,
)


class TestResolveTrainingDatabase:
    def test_config_value_wins_over_model_default(self) -> None:
        """When env_config supplies a training_database, it overrides the
        value hardcoded in the model_config — that is the whole point of
        the experiment-doc plumbing."""
        model_config: Dict[str, Any] = {
            "couchdb_training_database": "skol_training",
        }
        config: Dict[str, Any] = {"training_database": "skol_training_v2"}
        assert resolve_training_database(model_config, config) == "skol_training_v2"

    def test_config_default_still_wins(self) -> None:
        """env_config always provides training_database (defaulting to
        ``skol_training``). Even when the value happens to equal the
        env_config default, it should be honored — otherwise we silently
        fall back to whatever MODEL_CONFIGS hardcodes, defeating the
        priority chain. Callers (v1 experiment docs) are expected to
        carry ``databases.training`` explicitly per precedence option (b)."""
        model_config: Dict[str, Any] = {
            "couchdb_training_database": "skol_training_taxpub_v1",
        }
        config: Dict[str, Any] = {"training_database": "skol_training"}
        assert resolve_training_database(model_config, config) == "skol_training"

    def test_missing_config_key_falls_back_to_model(self) -> None:
        """If the env_config dict somehow lacks ``training_database``
        (e.g. an old test fixture or a non-experiment invocation), we
        fall back to the value in MODEL_CONFIGS rather than crashing."""
        model_config: Dict[str, Any] = {
            "couchdb_training_database": "skol_training_taxpub_v1",
        }
        config: Dict[str, Any] = {}
        assert (
            resolve_training_database(model_config, config)
            == "skol_training_taxpub_v1"
        )

    def test_empty_string_config_falls_back(self) -> None:
        """An empty-string training_database (e.g. an env var set to '')
        should fall back to the model default, not be treated as a
        literal DB name."""
        model_config: Dict[str, Any] = {
            "couchdb_training_database": "skol_training_taxpub_v1",
        }
        config: Dict[str, Any] = {"training_database": ""}
        assert (
            resolve_training_database(model_config, config)
            == "skol_training_taxpub_v1"
        )

    def test_both_missing_raises(self) -> None:
        """Neither model nor config supplies a value — fail loudly rather
        than silently training against an undefined database."""
        with pytest.raises((KeyError, ValueError)):
            resolve_training_database({}, {})


class TestLogisticSectionsV3ClassWeightsCoverActive19:
    """Step 1.C of docs/production_v3_plan.md — ``logistic_sections_v3``
    is the v3 baseline. Its class_weights dict must enumerate every
    tag in ACTIVE_TAGS_19 so the trained classifier has a weight for
    each active label.

    Older 3-class MODEL_CONFIGs (``logistic_sections``,
    ``logistic_sections_taxpub_v1``) are intentionally NOT expanded —
    they exist as v1/v1-jats baselines for comparison."""

    def test_logistic_sections_v3_keys_match_active_19(self) -> None:
        """class_weights keys are exactly the 19 ACTIVE_TAGS_19
        string values — no missing tag, no extra tag."""
        cfg = MODEL_CONFIGS["logistic_sections_v3"]
        active_values = {t.value for t in ACTIVE_TAGS_19}
        assert set(cfg["class_weights"].keys()) == active_values

    def test_logistic_sections_v3_weights_are_positive(self) -> None:
        """Every class weight is a positive float. Zero would zero
        out that class's contribution; negative is nonsense."""
        cfg = MODEL_CONFIGS["logistic_sections_v3"]
        for tag, weight in cfg["class_weights"].items():
            assert weight > 0, (
                f"class_weights[{tag!r}] = {weight!r} must be > 0"
            )

    def test_v1_baselines_remain_3class(self) -> None:
        """``logistic_sections`` and ``logistic_sections_taxpub_v1``
        keep their 3-key class_weights — they are v1 baselines for
        the v3 comparison report and must not be silently expanded."""
        for name in ("logistic_sections", "logistic_sections_taxpub_v1"):
            cfg = MODEL_CONFIGS[name]
            assert len(cfg["class_weights"]) == 3, (
                f"{name} must stay 3-class (v1 baseline); "
                f"got {len(cfg['class_weights'])} keys"
            )

    def test_misc_exposition_has_lowest_weight(self) -> None:
        """Inverse-frequency property (Step 3.B): the most-common tag
        ``Misc-exposition`` is the reference point — every other tag
        weight is ≥ its weight. Catches a future weight refresh that
        accidentally inverts the relationship."""
        weights = MODEL_CONFIGS["logistic_sections_v3"]["class_weights"]
        misc = weights["Misc-exposition"]
        for tag, w in weights.items():
            assert w >= misc, (
                f"{tag!r} weight {w} is below Misc-exposition {misc}"
            )

    def test_rare_tags_outweigh_common_tags(self) -> None:
        """Spot check: Phylogeny (188 blocks in the combined corpus)
        and Materials-and-methods (1 001 blocks) must outweigh
        Nomenclature (14 074 blocks) and Description (12 865) —
        otherwise the inverse-frequency calc is broken."""
        weights = MODEL_CONFIGS["logistic_sections_v3"]["class_weights"]
        assert weights["Phylogeny"] > weights["Nomenclature"]
        assert weights["Materials-and-methods"] > weights["Description"]
        assert weights["Diagnosis"] > weights["Notes"]
