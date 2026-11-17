"""Tests for bin/rebuild_redis.py's component table."""

import fnmatch
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_redis import COMPONENTS  # type: ignore[import]  # noqa: E402


class TestVocabTreeComponent:
    def test_runs_build_vocab_tree(self) -> None:
        assert COMPONENTS['vocab_tree']['script'] == 'build_vocab_tree'

    def test_no_menus_latest_pointer(self) -> None:
        """docs/plans/experiment-record-canonical.md, Step 1: the
        experiment record names the menus key; there is no shared
        pointer to rebuild."""
        assert 'skol:ui:menus_latest' not in COMPONENTS['vocab_tree']['keys']

    @pytest.mark.xfail(strict=True,
                       reason='counts hash not yet in the vocab_tree patterns')
    def test_patterns_cover_the_counts_hash(self) -> None:
        """The counts hash is '{<menus key>}:counts' (hash-tagged into the
        tree key's slot), which 'skol:ui:menus_*' does not match."""
        counts = '{skol:ui:menus_production_v4_1}:counts'
        assert any(fnmatch.fnmatchcase(counts, p)
                   for p in COMPONENTS['vocab_tree']['keys'])
