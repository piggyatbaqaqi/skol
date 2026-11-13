"""Tests for bin/build_vocab_tree.py.

Focused on the per-document top-level-key tracking and singleton pruning
introduced to eliminate vocabulary menu items that only appear in a single
JSON description.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))


import pytest  # noqa: E402

import build_vocab_tree  # type: ignore[import]  # noqa: E402
from build_vocab_tree import (  # type: ignore[import]  # noqa: E402
    VocabularyTree,
    build_parser,
    display_form,
    normalize_key,
    normalize_value,
    resolve_source_db,
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


class TestBuildParser:
    """The pipelines run ``build_vocab_tree --experiment X``; the script's
    own parser did not know --experiment and exited 2."""

    def test_accepts_experiment(self) -> None:
        args = build_parser().parse_args(['--experiment', 'production_v4_1'])
        assert args.experiment_name == 'production_v4_1'

    def test_script_flags_still_parse(self) -> None:
        args = build_parser().parse_args(
            ['--db', 'x', '--dry-run', '--limit', '5', '--output-json', 'o',
             '--show-sample', '--verbosity', '2', '--ttl', '60'])
        assert (args.db, args.dry_run, args.limit, args.output_json,
                args.show_sample, args.verbosity, args.ttl) == (
            'x', True, 5, 'o', True, 2, 60)

    def test_rejects_unknown_flags(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--experimnet', 'x'])


class TestResolveSourceDb:
    CONFIG = {
        'treatments_db_name':
            'skol_exp_production_v4_1_02_00_treatments_prose',
        'treatments_structured_db_name':
            'skol_exp_production_v4_1_03_00_treatments_structured',
    }

    def test_explicit_db_wins(self) -> None:
        assert resolve_source_db('mine', self.CONFIG) == 'mine'

    def test_experiment_structured_db_not_prose(self) -> None:
        assert resolve_source_db(None, self.CONFIG) == (
            'skol_exp_production_v4_1_03_00_treatments_structured')

    def test_fallback(self) -> None:
        assert resolve_source_db(None, {}) == 'skol_taxa_full_dev'


LATEST = 'skol:ui:menus_latest'


class FakeRedis:
    """The Redis string commands save_to_redis / load_from_redis use."""

    def __init__(self, data: Optional[Dict[str, str]] = None) -> None:
        self.data: Dict[str, str] = dict(data or {})

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self.data[key] = value
        return True

    def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self.data)


def _patched(r: FakeRedis) -> mock._patch:  # type: ignore[type-arg]
    return mock.patch.object(build_vocab_tree, 'create_redis_client',
                             return_value=r)


def _save(r: FakeRedis, **kwargs: object) -> str:
    tree = VocabularyTree()
    tree.add_document_json({'spores': {'shape': 'ellipsoid'}})
    with _patched(r):
        key: str = build_vocab_tree.save_to_redis(
            tree, verbosity=0, **kwargs)
    return key


class TestSaveToRedisExperimentKey:
    """With --experiment the tree goes to the record's menus key."""

    def test_tree_written_at_the_record_key(self) -> None:
        r = FakeRedis()
        key = _save(r, key_override='skol:ui:menus_production_v4_1')
        assert key == 'skol:ui:menus_production_v4_1'
        assert 'spores' in json.loads(r.data[key])['tree']

    def test_menus_latest_untouched(self) -> None:
        r = FakeRedis({LATEST: 'skol:ui:menus_old'})
        _save(r, key_override='skol:ui:menus_production_v4_1')
        assert r.data[LATEST] == 'skol:ui:menus_old'

    def test_record_key_holding_a_pointer_becomes_the_tree(self) -> None:
        """The 2026-11-13 stopgap left a pointer at a record's menus
        key; a build replaces it with the tree."""
        r = FakeRedis({'skol:ui:menus_v41_pointer':
                       'skol:ui:menus_production_v4_1'})
        _save(r, key_override='skol:ui:menus_v41_pointer')
        data = json.loads(r.data['skol:ui:menus_v41_pointer'])
        assert 'spores' in data['tree']


class TestSaveToRedisNoExperiment:
    """Without --experiment: a timestamped key, nothing else."""

    def test_versioned_key(self) -> None:
        r = FakeRedis()
        key = _save(r, version='2026_11_13_00_00')
        assert key == 'skol:ui:menus_2026_11_13_00_00'
        assert 'spores' in json.loads(r.data[key])['tree']

    def test_no_pointer_written(self) -> None:
        r = FakeRedis()
        _save(r, version='2026_11_13_00_00')
        assert set(r.data) == {'skol:ui:menus_2026_11_13_00_00'}


class TestLoadFromRedis:
    def test_explicit_version_loads(self) -> None:
        r = FakeRedis()
        _save(r, version='v1')
        with _patched(r):
            tree, meta = build_vocab_tree.load_from_redis(version='v1')
        assert tree.get_children([]) == ['spores']

    def test_no_version_does_not_read_menus_latest(self) -> None:
        r = FakeRedis()
        _save(r, version='v1')
        r.data[LATEST] = 'skol:ui:menus_v1'
        with _patched(r), pytest.raises(ValueError):
            build_vocab_tree.load_from_redis()
