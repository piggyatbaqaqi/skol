"""Tests for bin/build_vocab_tree.py.

Covers surface normalization, per-document support counts, pruning by
minimum support, frequency ordering, the count map, and the Redis
storage of the tree and its counts.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))


import pytest  # noqa: E402
from redis.crc import key_slot  # noqa: E402
from redis.exceptions import CrossSlotTransactionError  # noqa: E402

import build_vocab_tree  # type: ignore[import]  # noqa: E402
from build_vocab_tree import (  # type: ignore[import]  # noqa: E402
    VocabularyTree,
    build_parser,
    counts_key,
    display_form,
    normalize_key,
    resolve_source_db,
)
from treatments_to_structured.vocab_canonical import (  # noqa: E402
    normalize_value,
)


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
        assert tree.get_support(['shape']) == 2
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
    """The Redis commands save_to_redis / load_from_redis use: strings,
    plus the transaction that writes the tree with its counts hash."""

    def __init__(self, data: Optional[Dict[str, str]] = None) -> None:
        self.data: Dict[str, str] = dict(data or {})
        self.hashes: Dict[str, Dict[str, object]] = {}
        self.batches: list = []  # type: ignore[type-arg]

    def pipeline(self, transaction: bool = True) -> 'FakePipeline':
        return FakePipeline(self, transaction)

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


# ---------------------------------------------------------------------------
# Proposal 1 of docs/plans/vocab-tree-narrowing.md: per-node document
# support, frequency ordering, a minimum support at every level, and a
# separate path -> count map stored beside the tree.
# ---------------------------------------------------------------------------

def _support_tree() -> VocabularyTree:
    """Three documents.  ``conidia`` and ``conidium`` are one group;
    ``pileus`` and ``Pileus`` in doc 3 are one key seen once."""
    tree = VocabularyTree()
    tree.add_document_json({'conidia': {'shape': 'ellipsoid'},
                            'pileus': {'color': 'brown'}})
    tree.add_document_json({'conidium': {'shape': ['ellipsoid',
                                                   'ellipsoid']}})
    tree.add_document_json({'pileus': {'color': 'brown'},
                            'Pileus': {'color': 'brown'}})
    return tree


class TestSupportCounts:
    """A node's support is the number of documents containing its path."""

    def test_spellings_of_one_group_merge(self) -> None:
        assert _support_tree().get_support(['conidium']) == 2
        assert _support_tree().get_support(['conidia']) == 2

    def test_path_counts_once_per_document(self) -> None:
        tree = _support_tree()
        # doc 3 has pileus twice (two spellings): one document.
        assert tree.get_support(['pileus']) == 2
        assert tree.get_support(['pileus', 'color']) == 2

    def test_leaf_values_are_counted(self) -> None:
        tree = _support_tree()
        # doc 2 lists 'ellipsoid' twice: still one document.
        assert tree.get_support(['conidium', 'shape', 'ellipsoid']) == 2
        assert tree.get_support(['pileus', 'color', 'brown']) == 2

    def test_lists_recurse(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json([{'ascus': {'shape': 'clavate'}},
                                {'ascus': {'shape': 'cylindrical'}}])
        assert tree.get_support(['ascus']) == 1
        assert tree.get_support(['ascus', 'shape', 'clavate']) == 1
        assert tree.get_support(['ascus', 'shape', 'cylindrical']) == 1

    def test_missing_path_has_no_support(self) -> None:
        assert _support_tree().get_support(['stipe']) == 0

    def test_child_never_exceeds_parent(self) -> None:
        tree = _support_tree()
        for path in (['conidium', 'shape'], ['pileus', 'color'],
                     ['pileus', 'color', 'brown']):
            assert tree.get_support(path) <= tree.get_support(path[:-1])


def _prune_tree() -> VocabularyTree:
    """pileus in 3 docs, stipe in 1; under pileus, color in 3, odour in 1;
    under color, brown in 2, mauve in 1."""
    tree = VocabularyTree()
    tree.add_document_json({'pileus': {'color': 'brown', 'odour': 'faint'},
                            'stipe': {'surface': 'smooth'}})
    tree.add_document_json({'pileus': {'color': 'brown'}})
    tree.add_document_json({'pileus': {'color': 'mauve'}})
    return tree


class TestPruneBelowSupport:
    """Replaces prune_top_level_singletons: the threshold applies at
    every level, and a pruned node takes its subtree with it."""

    def test_removes_low_support_at_every_level(self) -> None:
        tree = _prune_tree()
        tree.prune_below_support(2)
        assert tree.get_children([]) == ['pileus']
        assert tree.get_children(['pileus']) == ['color']
        assert tree.get_children(['pileus', 'color']) == ['brown']

    def test_subtree_goes_with_its_node(self) -> None:
        tree = _prune_tree()
        tree.prune_below_support(2)
        assert tree.get_children(['stipe']) == []
        assert tree.get_children(['stipe', 'surface']) == []

    def test_one_changes_nothing(self) -> None:
        tree = _prune_tree()
        before = tree.to_dict()
        assert tree.prune_below_support(1) == 0
        assert tree.to_dict() == before

    def test_returns_nodes_removed_and_recomputes_stats(self) -> None:
        tree = _prune_tree()
        before = tree.stats['total_nodes']
        removed = tree.prune_below_support(2)
        # stipe, surface, smooth, odour, faint, mauve.
        assert removed == 6
        assert tree.stats['total_nodes'] == before - 6

    def test_counts_for_pruned_paths_are_dropped(self) -> None:
        tree = _prune_tree()
        tree.prune_below_support(2)
        assert tree.get_support(['stipe']) == 0
        assert '["stipe"]' not in tree.count_map()
        assert '["pileus", "odour"]' not in tree.count_map()


class TestChildrenByFrequency:
    """Children come back by descending support, ties alphabetical."""

    def test_descending_support_then_alphabetical(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({'stipe': {}, 'pileus': {}, 'ascus': {}})
        tree.add_document_json({'stipe': {}, 'pileus': {}})
        tree.add_document_json({'stipe': {}, 'basidium': {}})
        assert tree.get_children([]) == ['stipe', 'pileus', 'ascus',
                                         'basidium']

    def test_nested_levels_too(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({'pileus': {'odour': 'faint',
                                           'color': 'brown'}})
        tree.add_document_json({'pileus': {'odour': 'faint',
                                           'color': 'ochre'}})
        tree.add_document_json({'pileus': {'odour': 'faint',
                                           'color': 'ochre'}})
        tree.add_document_json({'pileus': {'odour': 'spicy'}})
        assert tree.get_children(['pileus']) == ['odour', 'color']
        assert tree.get_children(['pileus', 'color']) == ['ochre', 'brown']


class TestCountMap:
    """The separate path -> count map: each field is the JSON-encoded
    display path, because a key can contain '.' (``fig. 4``)."""

    def test_fields_are_json_display_paths(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({'Conidia': {'shape': 'ellipsoid'}})
        tree.add_document_json({'conidia': {'shape': 'ellipsoid'}})
        assert tree.count_map() == {
            '["conidia"]': 2,
            '["conidia", "shape"]': 2,
            '["conidia", "shape", "ellipsoid"]': 2,
        }

    def test_key_with_a_dot(self) -> None:
        tree = VocabularyTree()
        tree.add_document_json({'fig. 4': {'scale': '10 μm'}})
        assert tree.count_map()['["fig. 4", "scale", "10 μm"]'] == 1

    def test_values_are_ints(self) -> None:
        assert all(isinstance(v, int)
                   for v in _prune_tree().count_map().values())


class FakePipeline:
    """Buffers commands; applies them together on execute()."""

    def __init__(self, r: FakeRedis, transaction: bool) -> None:
        self.r = r
        self.transaction = transaction
        self.ops: list = []  # type: ignore[type-arg]

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self.ops.append(('set', key, value))

    def delete(self, *keys: str) -> None:
        self.ops.append(('delete', keys))

    def hset(self, key: str, mapping: Dict[str, int]) -> None:
        self.ops.append(('hset', key, mapping))

    def expire(self, key: str, ttl: int) -> None:
        self.ops.append(('expire', key, ttl))

    def execute(self) -> None:
        self.r.batches.append((self.transaction, list(self.ops)))
        for op in self.ops:
            if op[0] == 'set':
                self.r.data[op[1]] = op[2]
            elif op[0] == 'delete':
                for k in op[1]:
                    self.r.data.pop(k, None)
                    self.r.hashes.pop(k, None)
            elif op[0] == 'hset':
                self.r.hashes.setdefault(op[1], {}).update(op[2])


class CountingRedis(FakeRedis):
    """FakeRedis whose EXISTS also sees hashes, as real Redis does."""

    def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self.data or k in self.hashes)


class ClusterPipeline(FakePipeline):
    """Like redis-py's cluster pipeline: a transaction whose keys span
    hash slots raises CrossSlotTransactionError."""

    def execute(self) -> None:
        slots = {key_slot(op[1].encode()) if isinstance(op[1], str)
                 else key_slot(op[1][0].encode()) for op in self.ops}
        if self.transaction and len(slots) > 1:
            raise CrossSlotTransactionError(
                'All keys involved in a cluster transaction must map to '
                'the same slot')
        super().execute()


class ClusterRedis(CountingRedis):
    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return ClusterPipeline(self, transaction)


MENUS = 'skol:ui:menus_production_v4_1'


def _save_counted(r: CountingRedis) -> VocabularyTree:
    tree = _prune_tree()
    with _patched(r):
        build_vocab_tree.save_to_redis(tree, key_override=MENUS,
                                       verbosity=0)
    return tree


class TestSaveCountMap:
    def test_counts_key_hash_tags_the_menus_key(self) -> None:
        assert counts_key(MENUS) == '{' + MENUS + '}:counts'

    def test_counts_key_shares_the_tree_keys_slot(self) -> None:
        """Production's Redis is a cluster: a transaction over keys in
        different hash slots fails (2026-11-17, CrossSlotTransactionError)."""
        for key in (MENUS, 'skol:ui:menus_production',
                    'skol:ui:menus_2026_11_17_22_00'):
            assert key_slot(key.encode()) == key_slot(
                counts_key(key).encode()), key

    def test_save_succeeds_on_a_cluster(self) -> None:
        r = ClusterRedis()
        tree = _save_counted(r)
        assert 'pileus' in json.loads(r.data[MENUS])['tree']
        assert r.hashes[counts_key(MENUS)] == tree.count_map()

    def test_counts_hash_written_beside_the_tree(self) -> None:
        r = CountingRedis()
        tree = _save_counted(r)
        assert 'pileus' in json.loads(r.data[MENUS])['tree']
        assert r.hashes[counts_key(MENUS)] == tree.count_map()

    def test_tree_and_counts_in_one_transaction(self) -> None:
        r = CountingRedis()
        _save_counted(r)
        writes = [(txn, {op[0] for op in ops}) for txn, ops in r.batches]
        assert any(txn and {'set', 'hset'} <= kinds
                   for txn, kinds in writes)

    def test_stale_fields_removed(self) -> None:
        r = CountingRedis()
        r.hashes[counts_key(MENUS)] = {'["gone"]': 9}
        _save_counted(r)
        assert '["gone"]' not in r.hashes[counts_key(MENUS)]


class TestMinSupportFlag:
    """--min-support beats $VOCAB_MIN_SUPPORT beats the default of 2."""

    def _parse(self, argv: list, env: Dict[str, str]) -> int:  # type: ignore[type-arg]
        with mock.patch.dict('os.environ', env, clear=False):
            value: int = build_parser().parse_args(argv).min_support
        return value

    def test_default_is_two(self) -> None:
        with mock.patch.dict('os.environ', {}, clear=False):
            os.environ.pop('VOCAB_MIN_SUPPORT', None)
            assert build_parser().parse_args([]).min_support == 2

    def test_environment_beats_default(self) -> None:
        assert self._parse([], {'VOCAB_MIN_SUPPORT': '5'}) == 5

    def test_cli_beats_environment(self) -> None:
        assert self._parse(['--min-support', '3'],
                           {'VOCAB_MIN_SUPPORT': '5'}) == 3

    def test_below_one_rejected(self) -> None:
        assert self._parse(['--min-support', '1'], {}) == 1
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--min-support', '0'])


# ---------------------------------------------------------------------------
# P2a: the tree is built from json_canonical (docs/plans/
# vocab-tree-narrowing.md, Decisions for P2a; Trello AZeJMCR3)
# ---------------------------------------------------------------------------

from build_vocab_tree import (  # type: ignore[import]  # noqa: E402
    ALSO_SUFFIX,
    PinMissingError,
    StaleCanonicalError,
    also_key,
    resolve_version,
    tree_from_docs,
)
from treatments_to_structured.vocab_canonical import (  # noqa: E402
    group_key,
    source_hash,
)

VERSION = {'synonyms': 'a' * 40, 'spellings': 'b' * 40}


def _canonical_doc(annotated: Dict[str, object],
                   canonical: Dict[str, object],
                   edits: Optional[list] = None,  # type: ignore[type-arg]
                   version: Optional[Dict[str, str]] = None
                   ) -> Dict[str, object]:
    return {'json_annotated': annotated, 'json_canonical': canonical,
            'canonical_edits': edits or [],
            'canonical_map_version': dict(version or VERSION),
            'canonical_source': source_hash(annotated)}


def _cap_doc() -> Dict[str, object]:
    return _canonical_doc(
        {'Cap': {'Color': ['brown']}},
        {'Pileus': {'Color': ['brown']}},
        [{'path': ['Pileus'], 'from': 'Cap', 'to': 'Pileus',
          'rule': 'synonym', 'folds': []}])


def _pileus_doc() -> Dict[str, object]:
    return _canonical_doc({'Pileus': {'Color': ['yellow']}},
                          {'Pileus': {'Color': ['yellow']}})


def test_also_suffix() -> None:
    assert ALSO_SUFFIX == ':also'


class TestMenuSharesTheLanguageRules:
    def test_normalize_key_is_group_key(self) -> None:
        for raw in ('Colonies', 'Colour', 'Conidia', '`Spore_Shapes`',
                    'Species', 'IKI-'):
            assert normalize_key(raw) == group_key(raw), raw

    def test_new_folds_reach_the_menu(self) -> None:
        assert normalize_key('Colonies') == normalize_key('colony')
        assert normalize_key('Colour') == normalize_key('color')


class TestTreeFromCanonical:
    def test_reads_json_canonical_not_json_annotated(self) -> None:
        tree = tree_from_docs([_cap_doc(), _pileus_doc()], VERSION,
                              min_support=1)
        top = tree.get_children([])
        assert 'Pileus' in top
        assert 'Cap' not in top and 'cap' not in top
        assert tree.get_support(['Pileus']) == 2

    def test_shown_in_the_canonical_spelling(self) -> None:
        """Case kept: the spelling table chose the idiomatic form,
        so the menu no longer lowercases it."""
        tree = tree_from_docs([_pileus_doc()], VERSION, min_support=1)
        assert tree.get_children([]) == ['Pileus']
        assert tree.get_children(['pileus']) == ['Color']

    def test_documents_without_annotation_are_ignored(self) -> None:
        tree = tree_from_docs([_pileus_doc(), {'_id': 'x'}], VERSION,
                              min_support=1)
        assert tree.get_support(['Pileus']) == 1

    def test_design_documents_are_ignored(self) -> None:
        tree = tree_from_docs([_pileus_doc(),
                               {'_id': '_design/v', 'json_annotated': {}}],
                              VERSION, min_support=1)
        assert tree.get_support(['Pileus']) == 1

    def test_min_support_still_prunes(self) -> None:
        tree = tree_from_docs([_cap_doc(), _pileus_doc()], VERSION,
                              min_support=2)
        assert tree.get_children(['Pileus', 'Color']) == []


class TestStaleDocumentsRefuse:
    """Decision 2026-11-18: no tree while any document is stale."""

    def test_missing_canonical(self) -> None:
        doc: Dict[str, object] = {'json_annotated': {'Pileus': {}}}
        with pytest.raises(StaleCanonicalError) as caught:
            tree_from_docs([_pileus_doc(), doc], VERSION, min_support=1)
        assert (caught.value.stale, caught.value.checked) == (1, 2)

    def test_other_version(self) -> None:
        doc = _canonical_doc({'Pileus': {}}, {'Pileus': {}},
                             version={**VERSION, 'spellings': 'c' * 40})
        with pytest.raises(StaleCanonicalError):
            tree_from_docs([doc], VERSION)

    def test_annotation_changed_since(self) -> None:
        doc = _pileus_doc()
        doc['json_annotated'] = {'Stipe': {}}
        with pytest.raises(StaleCanonicalError):
            tree_from_docs([doc], VERSION)

    def test_message_names_the_count_and_the_fix(self) -> None:
        with pytest.raises(StaleCanonicalError) as caught:
            tree_from_docs([{'json_annotated': {'Pileus': {}}}], VERSION,
                           experiment='production')
        message = str(caught.value)
        assert '1 of 1' in message
        assert 'runstep production canonicalize_json' in message


class TestResolveVersion:
    def test_the_records_pin(self) -> None:
        config = {'experiment_name': 'production',
                  'vocab_canonical': dict(VERSION)}
        assert resolve_version(config) == VERSION

    def test_experiment_without_a_pin(self) -> None:
        with pytest.raises(PinMissingError, match='--pin'):
            resolve_version({'experiment_name': 'production'})

    def test_no_experiment_uses_the_installed_files(self) -> None:
        with mock.patch('build_vocab_tree.installed_version',
                        return_value=dict(VERSION)):
            assert resolve_version({}) == VERSION


class TestSynonymsShownInTheMenu:
    """Pileus (3,097; also: cap): synonym renames only."""

    def test_also_map(self) -> None:
        tree = tree_from_docs([_cap_doc(), _pileus_doc()], VERSION,
                              min_support=1)
        assert tree.also_map() == {json.dumps(['Pileus']): ['Cap']}

    def test_one_name_per_synonym_group(self) -> None:
        """Its most frequent spelling, as written."""
        docs = [_cap_doc(), _cap_doc(), _canonical_doc(
            {'cap': {}}, {'Pileus': {}},
            [{'path': ['Pileus'], 'from': 'cap', 'to': 'Pileus',
              'rule': 'synonym', 'folds': []}])]
        tree = tree_from_docs(docs, VERSION, min_support=1)
        assert tree.also_map() == {json.dumps(['Pileus']): ['Cap']}

    def test_spelling_edits_are_not_shown(self) -> None:
        doc = _canonical_doc(
            {'conidia': {}}, {'Conidia': {}},
            [{'path': ['Conidia'], 'from': 'conidia', 'to': 'Conidia',
              'rule': 'spelling', 'folds': ['case']}])
        tree = tree_from_docs([doc], VERSION, min_support=1)
        assert tree.also_map() == {}

    def test_pruned_paths_drop_their_synonyms(self) -> None:
        tree = tree_from_docs([_cap_doc()], VERSION, min_support=2)
        assert tree.also_map() == {}

    def test_also_key_shares_the_tree_keys_slot(self) -> None:
        assert also_key(MENUS) == '{' + MENUS + '}:also'
        assert key_slot(MENUS.encode()) == key_slot(also_key(MENUS).encode())

    def test_saved_beside_the_tree_in_one_transaction(self) -> None:
        r = ClusterRedis()
        tree = tree_from_docs([_cap_doc(), _pileus_doc()], VERSION,
                              min_support=1)
        with _patched(r):
            build_vocab_tree.save_to_redis(tree, key_override=MENUS,
                                           verbosity=0)
        assert r.hashes[also_key(MENUS)] == {
            json.dumps(['Pileus']): json.dumps(['Cap'])}

    def test_stale_synonyms_removed(self) -> None:
        r = CountingRedis()
        r.hashes[also_key(MENUS)] = {'["gone"]': '["x"]'}
        tree = tree_from_docs([_pileus_doc()], VERSION, min_support=1)
        with _patched(r):
            build_vocab_tree.save_to_redis(tree, key_override=MENUS,
                                           verbosity=0)
        assert '["gone"]' not in r.hashes.get(also_key(MENUS), {})


def test_value_spellings_are_one_value() -> None:
    """Decision B (2026-11-23): the menu groups values by value_group,
    as Structured Splits does, and shows each in its most common
    written form (user, 2026-11-23)."""
    tree = build({'Asci': {'Spore count': ['8-spored']}},
                 {'Asci': {'Spore count': ['8spored']}},
                 {'Asci': {'Spore count': ['eight-spored']}},
                 {'Asci': {'Spore count': ['0\u20131-septate']}})
    assert tree.get_children(['asci', 'spore count']) == [
        '8-spored', '0\u20131-septate']
    assert tree.get_support(['asci', 'spore count', '8-spored']) == 3


def test_values_show_their_most_common_written_form() -> None:
    """User, 2026-11-23: the menu shows written forms, case kept in a
    tree of json_canonical; any spelling still finds the value."""
    tree = VocabularyTree(keep_case=True)
    for colour in ['Brown', 'Brown', 'brown']:
        tree.add_document_json({'Pileus': {'Color': [colour]}})
    assert tree.get_children(['Pileus', 'Color']) == ['Brown']
    assert tree.get_support(['Pileus', 'Color', 'brown']) == 3
    reloaded = VocabularyTree.from_dict(tree.to_dict())
    assert reloaded.get_children(['Pileus', 'Color']) == ['Brown']
