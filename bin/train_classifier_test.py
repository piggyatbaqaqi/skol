"""Tests for bin/train_classifier.py helpers.

Focused on the precedence rule for training-database resolution: a
caller-supplied env_config value (which flows from CLI > env > the
experiment doc's ``databases.training`` field > env_config's hardcoded
default) must win over the value baked into MODEL_CONFIGS. This matches
the CLI-parameter priority chain documented in CLAUDE.md and parallels
the doc-field plumbing fixed for ``databases.treatments`` in Step 1.E
of docs/golden_v2_plan.md.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_classifier import resolve_training_database  # type: ignore[import]  # noqa: E402


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
