"""Tests for bin/build_vocab_tree.py.

Focused on the per-document top-level-key tracking and singleton pruning
introduced to eliminate vocabulary menu items that only appear in a single
JSON description.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


from build_vocab_tree import (  # type: ignore[import]  # noqa: E402
    VocabularyTree,
    display_form,
    normalize_key,
    normalize_value,
)


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
        # Counts are filed under the grouping key ('gills' -> 'gill').
        assert tree.top_level_doc_counts[normalize_key("gills")] == 1

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
        # Reported as the menus show it: '_' displays as a space.
        assert pruned == ["rare term"]
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


# ---------------------------------------------------------------------------
# Surface normalization (2026-11-12).  Census of production_v4_1's 43k
# records: beyond lower()+strip(), keys still split across plurals,
# `wrappers`, >shape<, ./texture and _ / - / space; present/absent was
# written six ways.  Spellings are grouped by normalize_key and each group
# is shown under its most frequent spelling -- the corpus's own choice, the
# same "most frequent term is canonical" rule as VocabularyNormalizer:
# shape (114,476 vs shapes 152) but spores (9,661 vs spore 427).
# ---------------------------------------------------------------------------


class TestNormalizeKey:
    """The grouping key.  Never shown to users."""

    def same(self, *spellings: str) -> bool:
        return len({normalize_key(s) for s in spellings}) == 1

    def test_case_whitespace_and_wrappers(self) -> None:
        assert self.same('Shape', '  shape ', '`Shape`', '``shape``',
                         '>shape<', './shape', ':shape', '"shape"',
                         "shape'", '? shape', ',shape')

    def test_separators_are_equivalent(self) -> None:
        assert self.same('clamp_connections', 'clamp-connections',
                         'clamp connections', 'Clamp  Connections')

    def test_english_plurals_group_with_singulars(self) -> None:
        assert self.same('Shape', 'Shapes')
        assert self.same('spore', 'Spores')
        assert self.same('conidiogenous cell', 'Conidiogenous_Cells')

    def test_latin_pairs_group(self) -> None:
        for singular, plural in (
                ('conidium', 'conidia'), ('ascus', 'asci'),
                ('hypha', 'hyphae'), ('basidium', 'basidia'),
                ('perithecium', 'perithecia'), ('apothecium', 'apothecia'),
                ('cystidium', 'cystidia'), ('sporangium', 'sporangia')):
            assert self.same(singular, plural), (singular, plural)

    def test_words_that_only_look_plural_stay_apart(self) -> None:
        for word in ('species', 'series', 'glomus', 'basis', 'apparatus',
                     'thickness', 'gibbous', 'lens', 'gas'):
            assert normalize_key(word) != normalize_key(word[:-1]), word

    def test_reaction_signs_are_kept(self) -> None:
        """IKI- (Melzer's negative) and KOH+ are results, not stray
        punctuation; IKI- must not merge with IKI or IKI+."""
        assert len({normalize_key(s) for s in ('iki-', 'iki', 'iki+')}) == 3
        assert self.same('IKI-', 'iki-')
        assert self.same('KOH+', 'koh+')

    def test_non_string_keys(self) -> None:
        assert normalize_key(3) == normalize_key('3')

    def test_idempotent(self) -> None:
        for raw in ('`Spore_Shapes`', 'Species', './Texture', 'Conidia',
                    'IKI-', 'd-xylose'):
            once = normalize_key(raw)
            assert normalize_key(once) == once, raw


class TestDisplayForm:
    """How one spelling would be shown, before the majority vote."""

    def test_lowercase_trimmed_unwrapped(self) -> None:
        assert display_form('  `Shape` ') == 'shape'
        assert display_form('>Surface<') == 'surface'
        assert display_form('./Texture') == 'texture'

    def test_underscores_display_as_spaces(self) -> None:
        assert display_form('Clamp_Connections') == 'clamp connections'

    def test_hyphens_are_kept(self) -> None:
        assert display_form('Thin-Walled') == 'thin-walled'
        assert display_form('D-Xylose') == 'd-xylose'
        assert display_form('IKI-') == 'iki-'

    def test_plural_is_kept(self) -> None:
        """Choosing singular or plural is the majority vote's job."""
        assert display_form('Spores') == 'spores'

    def test_result_is_lowercase(self) -> None:
        """The menus' Django endpoint looks paths up with part.lower()."""
        for raw in ('SURFACE', 'Clamp_Connections', 'KOH+'):
            assert display_form(raw) == display_form(raw).lower()


class TestNormalizeValue:

    def test_present_spellings(self) -> None:
        for raw in ('yes', 'Yes', 'YES', 'true', 'True', 'present',
                    'Present', ' yes '):
            assert normalize_value(raw) == 'present', raw

    def test_absent_spellings(self) -> None:
        for raw in ('no', 'No', 'false', 'False', 'absent', 'Absent'):
            assert normalize_value(raw) == 'absent', raw

    def test_inner_punctuation_kept(self) -> None:
        assert normalize_value('Thin-walled') == 'thin-walled'
        assert normalize_value('6–8 µm') == '6–8 µm'

    def test_surrounding_quotes_and_backticks(self) -> None:
        assert normalize_value('`hyaline`') == 'hyaline'
        assert normalize_value('"smooth"') == 'smooth'

    def test_no_plural_folding(self) -> None:
        assert normalize_value('Spores') == 'spores'

    def test_booleans_only_as_whole_values(self) -> None:
        assert normalize_value('not present') == 'not present'
        assert normalize_value('yes, rarely') == 'yes, rarely'

    def test_idempotent(self) -> None:
        for raw in ('Yes', 'Thin-walled', '`hyaline`'):
            once = normalize_value(raw)
            assert normalize_value(once) == once, raw


def build(*docs: object) -> VocabularyTree:
    tree = VocabularyTree()
    for doc in docs:
        tree.add_document_json(doc)
    return tree


class TestMostFrequentSpellingWins:

    def test_singular_majority(self) -> None:
        tree = build({'Shape': ['convex']}, {'shape': ['flat']},
                     {'Shapes': ['round']})
        assert tree.get_children([]) == ['shape']
        assert tree.get_children(['shape']) == ['convex', 'flat', 'round']

    def test_plural_majority(self) -> None:
        tree = build({'Spores': ['ovoid']}, {'spores': ['globose']},
                     {'Spore': ['ellipsoid']})
        assert tree.get_children([]) == ['spores']

    def test_latin_plural_majority(self) -> None:
        tree = build({'Conidia': ['hyaline']}, {'conidia': ['smooth']},
                     {'Conidium': ['curved']})
        assert tree.get_children([]) == ['conidia']

    def test_separator_majority(self) -> None:
        tree = build({'clamp connections': ['present']},
                     {'Clamp connections': ['absent']},
                     {'clamp-connections': ['present']},
                     {'clamp_connections': ['present']})
        assert tree.get_children([]) == ['clamp connections']

    def test_hyphenated_majority_survives(self) -> None:
        tree = build({'Thin-walled': ['yes']}, {'thin-walled': ['yes']},
                     {'thin walled': ['yes']})
        assert tree.get_children([]) == ['thin-walled']

    def test_tie_breaks_alphabetically(self) -> None:
        tree = build({'Spore': ['ovoid']}, {'Spores': ['ovoid']})
        assert tree.get_children([]) == ['spore']

    def test_counted_across_the_whole_tree(self) -> None:
        """One spelling per group everywhere in the menus, not per level."""
        tree = build({'Spores': ['ovoid']}, {'spores': ['ovoid']},
                     {'Asci': {'Spore': ['8']}})
        assert tree.get_children(['asci']) == ['spores']

    def test_boolean_values_merge(self) -> None:
        tree = build({'Clamps': ['yes', 'True', 'present', 'No', 'absent']},
                     {'Clamps': ['x']})
        assert tree.get_children(['clamps']) == ['absent', 'present', 'x']


def test_reaction_signs_stay_separate() -> None:
    """Guard: grouping must not merge IKI- (negative) with IKI or IKI+.
    Passes today (plain lower()) and must keep passing."""
    tree = build({'IKI-': ['yes']}, {'IKI+': ['yes']}, {'IKI': ['x']})
    assert tree.get_children([]) == ['iki', 'iki+', 'iki-']


class TestLookupsAcceptAnySpelling:

    def tree(self) -> VocabularyTree:
        return build({'Spores': {'Outline': ['ovoid']}},
                     {'spores': {'outline': ['globose']}},
                     {'Spore': {'Outlines': ['ellipsoid']}})

    def test_get_children(self) -> None:
        tree = self.tree()
        for spelling in ('spores', 'Spores', 'spore', '`Spore`', 'SPORES'):
            assert tree.get_children([spelling]) == ['outline'], spelling

    def test_get_subtree(self) -> None:
        assert self.tree().get_subtree(['spore', 'Outlines']) == {
            'ellipsoid': {}, 'globose': {}, 'ovoid': {}}

    def test_document_counts_merge_spellings(self) -> None:
        tree = build({'Shape': ['convex']}, {'shapes': ['round']})
        assert tree.prune_top_level_singletons() == []
        assert tree.get_children([]) == ['shape']

    def test_saved_tree_uses_display_spellings(self) -> None:
        """What build_vocab_tree writes to Redis, and the Django menu
        endpoint reads with part.lower()."""
        saved = self.tree().to_dict()['tree']
        assert list(saved) == ['spores']
        assert sorted(saved['spores']) == ['outline']

    def test_reloaded_tree_still_accepts_any_spelling(self) -> None:
        reloaded = VocabularyTree.from_dict(self.tree().to_dict())
        assert reloaded.get_children(['spore']) == ['outline']
