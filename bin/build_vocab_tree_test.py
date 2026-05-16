"""Tests for bin/build_vocab_tree.py.

Focused on the per-document top-level-key tracking and singleton pruning
introduced to eliminate vocabulary menu items that only appear in a single
JSON description.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_vocab_tree import VocabularyTree  # type: ignore[import]  # noqa: E402


class TestTopLevelDocCounts:
    def test_count_starts_at_zero(self) -> None:
        tree = VocabularyTree()
        assert dict(tree.top_level_doc_counts) == {}

    def test_increment_once_per_document(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({"pileus": {}, "stipe": {}})
        tree.add_document_json({"pileus": {}, "gills": {}})
        tree.add_document_json({"pileus": {}, "stipe": {}})
        # pileus appears in 3 docs, stipe in 2, gills in 1.
        assert tree.top_level_doc_counts["pileus"] == 3
        assert tree.top_level_doc_counts["stipe"] == 2
        assert tree.top_level_doc_counts["gills"] == 1

    def test_duplicate_key_within_one_doc_counts_once(self) -> None:
        """Two top-level dicts cannot share a key in one Python dict literal,
        but the count must reflect *documents*, not invocations."""
        tree = VocabularyTree()
        tree.add_document_json({"pileus": {"color": "brown"}})
        # The inner 'color' key is depth-1, not top-level.
        assert tree.top_level_doc_counts.get("color", 0) == 0
        assert tree.top_level_doc_counts["pileus"] == 1

    def test_top_level_list_of_dicts_each_count(self) -> None:
        """A document whose root is a list of dicts: each dict's top-level
        keys count as that document's top-level keys."""
        tree = VocabularyTree()
        tree.add_document_json([{"pileus": {}}, {"stipe": {}}])
        assert tree.top_level_doc_counts["pileus"] == 1
        assert tree.top_level_doc_counts["stipe"] == 1

    def test_invalid_terms_skipped(self) -> None:
        tree = VocabularyTree()
        # '{' and ': ' fragments are filtered by _is_valid_term.
        tree.add_document_json({"pileus": {}, '{"': {}, "  ": {}})
        assert tree.top_level_doc_counts["pileus"] == 1
        assert '{"' not in tree.top_level_doc_counts
        assert "  " not in tree.top_level_doc_counts


class TestPruneTopLevelSingletons:
    def test_removes_singletons(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({"pileus": {}, "rare_term": {}})
        tree.add_document_json({"pileus": {}, "stipe": {}})
        tree.add_document_json({"pileus": {}, "stipe": {}})
        pruned = tree.prune_top_level_singletons()
        assert pruned == ["rare_term"]
        assert "rare_term" not in tree.tree
        assert "pileus" in tree.tree
        assert "stipe" in tree.tree

    def test_keeps_keys_seen_twice(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({"shared": {}})
        tree.add_document_json({"shared": {}})
        pruned = tree.prune_top_level_singletons()
        assert pruned == []
        assert "shared" in tree.tree

    def test_idempotent(self) -> None:
        """Running prune twice yields the same result and removes nothing
        the second time."""
        tree = VocabularyTree()
        tree.add_document_json({"a": {}, "b": {}})
        tree.add_document_json({"a": {}})
        first = tree.prune_top_level_singletons()
        second = tree.prune_top_level_singletons()
        assert first == ["b"]
        assert second == []

    def test_prune_recomputes_stats(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({"pileus": {"shape": "convex"}, "rare": {"x": "y"}})
        tree.add_document_json({"pileus": {"shape": "flat"}})
        before_nodes = tree.stats["total_nodes"]
        tree.prune_top_level_singletons()
        assert tree.stats["total_nodes"] < before_nodes
        # After pruning, 'rare' subtree (2 nodes: 'rare' itself + 'x' + 'y'?) is gone.
        assert "rare" not in tree.tree

    def test_pruned_keys_disappear_from_get_children(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({"keeper": {}, "loner": {}})
        tree.add_document_json({"keeper": {}})
        tree.prune_top_level_singletons()
        children = tree.get_children([])
        assert "keeper" in children
        assert "loner" not in children


class TestAddDocumentJsonStillCallsAddJson:
    """The new method must still populate the tree the same way as add_json."""

    def test_full_subtree_populated(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({"pileus": {"shape": "convex"}})
        tree.add_document_json({"pileus": {"shape": "flat"}})
        # Without pruning, both shapes should be present.
        children = tree.get_children(["pileus", "shape"])
        assert "convex" in children
        assert "flat" in children
