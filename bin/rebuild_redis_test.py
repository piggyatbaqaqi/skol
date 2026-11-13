"""Tests for bin/rebuild_redis.py's component table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from rebuild_redis import COMPONENTS  # type: ignore[import]  # noqa: E402


class TestVocabTreeComponent:
    def test_runs_build_vocab_tree(self) -> None:
        assert COMPONENTS['vocab_tree']['script'] == 'build_vocab_tree'

    @pytest.mark.xfail(strict=True,
                       reason="skol:ui:menus_latest still listed")
    def test_no_menus_latest_pointer(self) -> None:
        """docs/plans/experiment-record-canonical.md, Step 1: the
        experiment record names the menus key; there is no shared
        pointer to rebuild."""
        assert 'skol:ui:menus_latest' not in COMPONENTS['vocab_tree']['keys']
