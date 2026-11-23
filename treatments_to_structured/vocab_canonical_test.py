"""Tests for treatments_to_structured/vocab_canonical.py.

Proposal 2a of docs/plans/vocab-tree-narrowing.md (Trello AZeJMCR3):
``json_canonical`` is derived from ``json_annotated`` by language rules,
a hand-curated synonym map and a generated spelling table, with every
changed key recorded in ``canonical_edits``.

Run with: python -m pytest treatments_to_structured/vocab_canonical_test.py -v
"""
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from treatments_to_structured.vocab_canonical import (  # noqa: E402
    FOLDS,
    NON_SYNONYMS_NAME,
    RULE_FUNCTIONS,
    RULE_TABLES,
    RULES_FINGERPRINT,
    RULES_VERSION,
    SPELLINGS_NAME,
    SYNONYMS_NAME,
    MapError,
    blob_hash,
    canonical_fields,
    canonicalize,
    check_non_synonyms,
    display_form,
    fingerprint,
    folds,
    group_key,
    installed_version,
    is_fresh,
    load_non_synonyms,
    load_spellings,
    load_synonyms,
    normalize_value,
    parse_annotated,
    rules_fingerprint,
    source_hash,
    spelling_candidate,
    value_group,
)


SYNONYMS = {'cap': ('pileus', 'Pileus')}
SPELLINGS = {
    'conidium': 'Conidia',
    'pileus': 'Pileus',
    'color': 'Color',
    'shape': 'Shape',
    'size': 'Size',
    'culture characteristic': 'Culture characteristics',
}
VERSION = {'synonyms': 'a' * 40, 'spellings': 'b' * 40}


@pytest.fixture(autouse=True)
def _repo_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests of the repo's own data must not read an installed copy."""
    monkeypatch.delenv('SKOL_SHARE', raising=False)


def _synonym_entry(**overrides: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        'from': 'cap', 'to': 'Pileus', 'rule': 'synonym',
        'support': {'from': 114, 'to': 3097},
        'experiment': 'production', 'added': '2026-11-16'}
    entry.update(overrides)
    return entry


def _write_synonyms(path: Path, entries: List[Dict[str, Any]]) -> Path:
    path.write_text(json.dumps({'_comment': 'test', 'entries': entries}))
    return path


def _write_spellings(path: Path, spellings: Dict[str, str]) -> Path:
    path.write_text(json.dumps({'_comment': 'test', '_generated': {},
                                'spellings': spellings}))
    return path


# ---------------------------------------------------------------------------
# Language rules
# ---------------------------------------------------------------------------


def test_folds_are_reported_in_this_order() -> None:
    assert list(FOLDS) == ['wrapper', 'case', 'separator', 'plural',
                           'latin_plural', 'british']


class TestGroupKey:
    """The language rules, moved from build_vocab_tree.normalize_key."""

    def same(self, *spellings: str) -> bool:
        return len({group_key(s) for s in spellings}) == 1

    def test_case_whitespace_and_wrappers(self) -> None:
        assert self.same('Shape', '  shape ', '`Shape`', '>shape<',
                         './shape', ':shape', '"shape"', '? shape')

    def test_separators_are_equivalent(self) -> None:
        assert self.same('clamp_connections', 'clamp-connections',
                         'Clamp  Connections')

    def test_english_and_latin_plurals(self) -> None:
        assert self.same('Spore', 'Spores')
        assert self.same('Conidium', 'conidia')
        assert self.same('Ascus', 'Asci')

    def test_ies_plural_folds(self) -> None:
        """colony (1,265 documents) vs colonies (1,293)."""
        assert self.same('Colony', 'Colonies', 'colonies')
        assert self.same('body', 'bodies')
        assert self.same('Fruiting bodies', 'fruiting body')

    def test_species_and_series_are_not_ies_plurals(self) -> None:
        assert group_key('Species') == 'species'
        assert group_key('series') == 'series'

    def test_british_spellings_fold(self) -> None:
        assert self.same('Colour', 'color', 'Colours', 'colors')
        assert self.same('Odour', 'odor')
        assert self.same('Spore colour', 'spore color')

    def test_british_fold_is_whole_words_only(self) -> None:
        assert group_key('colouration') != group_key('coloration')

    def test_words_that_only_look_plural_stay_apart(self) -> None:
        for word in ('glomus', 'basis', 'apparatus', 'thickness',
                     'gibbous', 'lens', 'gas'):
            assert group_key(word) != group_key(word[:-1]), word

    def test_reaction_signs_are_kept(self) -> None:
        assert len({group_key(s) for s in ('IKI-', 'IKI', 'IKI+')}) == 3

    def test_idempotent(self) -> None:
        for raw in ('`Spore_Shapes`', 'Species', 'Colonies', 'Colour',
                    'Conidia', 'IKI-'):
            once = group_key(raw)
            assert group_key(once) == once, raw


class TestDisplayFormAndValues:
    def test_display_form(self) -> None:
        assert display_form('`Spore_Shape`') == 'spore shape'

    def test_normalize_value(self) -> None:
        assert normalize_value(' Yes ') == 'present'
        assert normalize_value('Absent') == 'absent'
        assert normalize_value('Thin-walled') == 'thin-walled'


class TestSpellingCandidate:
    @pytest.mark.parametrize('raw,expected', [
        ('`Species`', 'Species'),
        ('Culture_characteristics', 'Culture characteristics'),
        ('  Pileus  ', 'Pileus'),
        ('PILEUS', 'PILEUS'),
        ('Conidiogenous  cells', 'Conidiogenous cells'),
    ])
    def test_candidate(self, raw: str, expected: str) -> None:
        assert spelling_candidate(raw) == expected


class TestFolds:
    @pytest.mark.parametrize('raw,target,expected', [
        ('Conidium', 'Conidia', ['latin_plural']),
        ('conidia', 'Conidia', ['case']),
        ('conidium', 'Conidia', ['case', 'latin_plural']),
        ('Culture_characteristics', 'Culture characteristics',
         ['separator']),
        ('Culture Characteristics', 'Culture characteristics', ['case']),
        ('`Species`', 'Species', ['wrapper']),
        (' Species ', 'Species', ['wrapper']),
        ('Spore', 'Spores', ['plural']),
        ('Colonies', 'Colony', ['plural']),
        ('Colour', 'Color', ['british']),
        ('conidiogenous_Cell', 'Conidiogenous cells',
         ['case', 'separator', 'plural']),
        ('PILEUS', 'Pileus', ['case']),
        ('Pileus', 'Pileus', []),
    ])
    def test_folds(self, raw: str, target: str,
                   expected: List[str]) -> None:
        assert folds(raw, target) == expected


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------


class TestBlobHash:
    def test_matches_git_hash_object(self, tmp_path: Path) -> None:
        """``printf 'hello\\n' | git hash-object --stdin``."""
        path = tmp_path / 'x'
        path.write_bytes(b'hello\n')
        assert blob_hash(path) == 'ce013625030ba8dba906f756967f9e9ca394464a'


class TestLoadSynonyms:
    def test_loads_groups_and_target_spelling(self, tmp_path: Path) -> None:
        path = _write_synonyms(tmp_path / 's.json', [_synonym_entry()])
        assert load_synonyms(path) == {'cap': ('pileus', 'Pileus')}

    def test_source_in_any_spelling(self, tmp_path: Path) -> None:
        path = _write_synonyms(tmp_path / 's.json',
                               [_synonym_entry(**{'from': '`Caps`'})])
        assert set(load_synonyms(path)) == {'cap'}

    def test_chains_are_refused(self, tmp_path: Path) -> None:
        """cap -> pileus and pileus -> x: apply-once would leave the
        result depending on order."""
        path = _write_synonyms(tmp_path / 's.json', [
            _synonym_entry(),
            _synonym_entry(**{'from': 'Pileus', 'to': 'Hat'})])
        with pytest.raises(MapError):
            load_synonyms(path)

    def test_self_map_is_refused(self, tmp_path: Path) -> None:
        path = _write_synonyms(tmp_path / 's.json', [
            _synonym_entry(**{'from': 'pileus', 'to': 'Pileus'})])
        with pytest.raises(MapError):
            load_synonyms(path)

    def test_conflicting_targets_are_refused(self, tmp_path: Path) -> None:
        path = _write_synonyms(tmp_path / 's.json', [
            _synonym_entry(),
            _synonym_entry(**{'from': 'Caps', 'to': 'Stipe'})])
        with pytest.raises(MapError):
            load_synonyms(path)

    @pytest.mark.parametrize('missing', ['support', 'experiment', 'added'])
    def test_evidence_is_required(self, tmp_path: Path,
                                  missing: str) -> None:
        entry = _synonym_entry()
        del entry[missing]
        path = _write_synonyms(tmp_path / 's.json', [entry])
        with pytest.raises(MapError):
            load_synonyms(path)

    def test_rule_must_be_synonym(self, tmp_path: Path) -> None:
        path = _write_synonyms(tmp_path / 's.json',
                               [_synonym_entry(rule='hyponym')])
        with pytest.raises(MapError):
            load_synonyms(path)

    def test_default_path_is_in_the_share(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_synonyms(tmp_path / SYNONYMS_NAME, [_synonym_entry()])
        monkeypatch.setenv('SKOL_SHARE', str(tmp_path))
        assert load_synonyms() == {'cap': ('pileus', 'Pileus')}


class TestLoadSpellings:
    def test_loads(self, tmp_path: Path) -> None:
        path = _write_spellings(tmp_path / 'sp.json',
                                {'conidium': 'Conidia'})
        assert load_spellings(path) == {'conidium': 'Conidia'}

    def test_stale_entry_is_refused(self, tmp_path: Path) -> None:
        """A spelling outside its group means the table predates a
        change in the language rules: regenerate it."""
        path = _write_spellings(tmp_path / 'sp.json',
                                {'conidium': 'Asci'})
        with pytest.raises(MapError):
            load_spellings(path)

    def test_default_path_is_in_the_share(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_spellings(tmp_path / SPELLINGS_NAME, {'pileus': 'Pileus'})
        monkeypatch.setenv('SKOL_SHARE', str(tmp_path))
        assert load_spellings() == {'pileus': 'Pileus'}


class TestNonSynonyms:
    PAIRS = [{'a': 'Sexual morph', 'b': 'Asexual morph',
              'why': 'opposites'}]

    def test_loads_a_list(self, tmp_path: Path) -> None:
        path = tmp_path / 'n.json'
        path.write_text(json.dumps(self.PAIRS))
        assert load_non_synonyms(path) == self.PAIRS

    def test_why_is_required(self, tmp_path: Path) -> None:
        path = tmp_path / 'n.json'
        path.write_text(json.dumps([{'a': 'x', 'b': 'y'}]))
        with pytest.raises(MapError):
            load_non_synonyms(path)

    def test_default_path_is_in_the_share(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / NON_SYNONYMS_NAME).write_text(json.dumps(self.PAIRS))
        monkeypatch.setenv('SKOL_SHARE', str(tmp_path))
        assert load_non_synonyms() == self.PAIRS

    def test_clean(self) -> None:
        assert check_non_synonyms(SYNONYMS, self.PAIRS) == []

    def test_map_entry_merging_a_pair(self) -> None:
        bad = {'asexual morph': ('sexual morph', 'Sexual morph')}
        problems = check_non_synonyms(bad, self.PAIRS)
        assert len(problems) == 1
        assert 'Sexual morph' in problems[0]
        assert 'Asexual morph' in problems[0]

    def test_reverse_direction(self) -> None:
        bad = {'sexual morph': ('asexual morph', 'Asexual morph')}
        assert len(check_non_synonyms(bad, self.PAIRS)) == 1

    def test_shared_target(self) -> None:
        bad = {'sexual morph': ('morph', 'Morph'),
               'asexual morph': ('morph', 'Morph')}
        assert len(check_non_synonyms(bad, self.PAIRS)) == 1

    def test_language_rules_merging_a_pair(self) -> None:
        pairs = [{'a': 'Spores', 'b': 'spore', 'why': 'test'}]
        assert len(check_non_synonyms({}, pairs)) == 1


class TestInstalledVersion:
    def test_hashes_of_the_installed_files(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        syn = _write_synonyms(tmp_path / SYNONYMS_NAME, [_synonym_entry()])
        spel = _write_spellings(tmp_path / SPELLINGS_NAME,
                                {'pileus': 'Pileus'})
        monkeypatch.setenv('SKOL_SHARE', str(tmp_path))
        assert installed_version() == {'synonyms': blob_hash(syn),
                                       'spellings': blob_hash(spel),
                                       'rules': str(RULES_VERSION)}

    def test_explicit_share(self, tmp_path: Path) -> None:
        syn = _write_synonyms(tmp_path / SYNONYMS_NAME, [_synonym_entry()])
        spel = _write_spellings(tmp_path / SPELLINGS_NAME, {})
        assert installed_version(tmp_path) == {
            'synonyms': blob_hash(syn), 'spellings': blob_hash(spel),
            'rules': str(RULES_VERSION)}

    def test_document_made_before_the_rules_version_is_stale(
            self, tmp_path: Path) -> None:
        # Documents written before this change pin only the two files.
        _write_synonyms(tmp_path / SYNONYMS_NAME, [_synonym_entry()])
        _write_spellings(tmp_path / SPELLINGS_NAME, {})
        installed = installed_version(tmp_path)
        old = {k: v for k, v in installed.items() if k != 'rules'}
        annotated = {'Pileus': {'Color': ['brown']}}
        doc = {'json_annotated': annotated, 'json_canonical': annotated,
               'canonical_map_version': old,
               'canonical_source': source_hash(annotated)}
        assert not is_fresh(doc, installed)


def _body_a(x: int) -> int:
    """A docstring."""
    return x + 1  # a comment


def _body_b(x: int) -> int:
    return (x
            + 1)


def _body_c(x: int) -> int:
    return x + 2


class TestRulesFingerprint:
    def test_recorded_for_this_rules_version(self) -> None:
        # If this fails, a language rule changed: bump RULES_VERSION and
        # set RULES_FINGERPRINT to the value printed here.
        assert rules_fingerprint() == RULES_FINGERPRINT, (
            f'rules changed: bump RULES_VERSION (now {RULES_VERSION}) and '
            f'set RULES_FINGERPRINT = {rules_fingerprint()!r}')

    def test_version_is_a_positive_integer(self) -> None:
        assert isinstance(RULES_VERSION, int) and RULES_VERSION >= 1

    def test_every_rule_table_is_covered(self) -> None:
        import treatments_to_structured.vocab_canonical as vc
        tables = {name for name in vars(vc)
                  if re.fullmatch(r'_?[A-Z][A-Z_]*', name)
                  and name not in {'RULES_VERSION', 'RULES_FINGERPRINT',
                                   'RULE_FUNCTIONS', 'RULE_TABLES',
                                   'SYNONYMS_NAME', 'SPELLINGS_NAME',
                                   'NON_SYNONYMS_NAME', '_SYNONYM_FIELDS'}}
        assert tables <= set(RULE_TABLES)

    def test_the_grouping_and_value_rules_are_covered(self) -> None:
        assert group_key in RULE_FUNCTIONS
        assert normalize_value in RULE_FUNCTIONS
        assert folds in RULE_FUNCTIONS

    def test_docstrings_comments_and_layout_do_not_count(self) -> None:
        assert fingerprint([_body_a], {}) == fingerprint([_body_b], {})

    def test_a_changed_body_counts(self) -> None:
        assert fingerprint([_body_a], {}) != fingerprint([_body_c], {})

    def test_a_changed_table_counts(self) -> None:
        assert (fingerprint([], {'t': {'a': 'b'}})
                != fingerprint([], {'t': {'a': 'c'}}))

    def test_table_order_does_not_count(self) -> None:
        assert (fingerprint([], {'t': {'a': 1, 'b': 2}, 's': frozenset('xy')})
                == fingerprint([], {'s': frozenset('yx'),
                                    't': {'b': 2, 'a': 1}}))


class TestRepoData:
    """The files in data/, as shipped."""

    def test_they_load(self) -> None:
        assert load_synonyms()
        assert load_spellings()
        assert load_non_synonyms()

    def test_no_entry_merges_a_recorded_non_synonym(self) -> None:
        assert check_non_synonyms(load_synonyms(), load_non_synonyms()) == []

    def test_seed_entry(self) -> None:
        assert load_synonyms()['cap'] == ('pileus', 'Pileus')

    def test_spellings_follow_the_corpus(self) -> None:
        """The idiomatic forms measured 2026-11-18."""
        spellings = load_spellings()
        assert spellings['conidium'] == 'Conidia'
        assert spellings['pileus'] == 'Pileus'
        assert spellings['ascus'] == 'Asci'


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestSourceAndParse:
    def test_source_hash_ignores_key_order(self) -> None:
        assert (source_hash({'a': 1, 'b': [2, 3]})
                == source_hash({'b': [2, 3], 'a': 1}))

    def test_source_hash_sees_values(self) -> None:
        assert source_hash({'a': 1}) != source_hash({'a': 2})

    def test_parse_dict(self) -> None:
        assert parse_annotated({'json_annotated': {'a': 1}}) == {'a': 1}

    def test_parse_json_string(self) -> None:
        assert parse_annotated({'json_annotated': '{"a": 1}'}) == {'a': 1}

    @pytest.mark.parametrize('value', ['not json', '[1, 2]', [1], None, ''])
    def test_unusable(self, value: Any) -> None:
        assert parse_annotated({'json_annotated': value}) is None

    def test_missing(self) -> None:
        assert parse_annotated({}) is None


class TestIsFresh:
    def _doc(self) -> Dict[str, Any]:
        annotated = {'Cap': {'Color': ['brown']}}
        return {'json_annotated': annotated,
                'json_canonical': {'Pileus': {'Color': ['brown']}},
                'canonical_map_version': dict(VERSION),
                'canonical_source': source_hash(annotated)}

    def test_fresh(self) -> None:
        assert is_fresh(self._doc(), VERSION)

    def test_no_canonical(self) -> None:
        doc = self._doc()
        del doc['json_canonical']
        assert not is_fresh(doc, VERSION)

    def test_other_version(self) -> None:
        assert not is_fresh(self._doc(), {**VERSION, 'spellings': 'c' * 40})

    def test_annotated_changed_since(self) -> None:
        doc = self._doc()
        doc['json_annotated'] = {'Cap': {'Color': ['yellow']}}
        assert not is_fresh(doc, VERSION)


class TestCanonicalize:
    def run(self, annotated: Dict[str, Any]) -> Any:
        return canonicalize(annotated, SYNONYMS, SPELLINGS)

    def test_canonical_spelling_is_unchanged(self) -> None:
        annotated = {'Conidia': {'Shape': ['ellipsoid']}}
        assert self.run(annotated) == (annotated, [])

    def test_variant_takes_the_table_spelling(self) -> None:
        canonical, edits = self.run({'conidium': {'Shape': ['ellipsoid']}})
        assert canonical == {'Conidia': {'Shape': ['ellipsoid']}}
        assert edits == [{'path': ['Conidia'], 'from': 'conidium',
                          'to': 'Conidia', 'rule': 'spelling',
                          'folds': ['case', 'latin_plural']}]

    def test_nested_keys(self) -> None:
        canonical, edits = self.run({'Pileus': {'colour': ['brown']}})
        assert canonical == {'Pileus': {'Color': ['brown']}}
        assert edits == [{'path': ['Pileus', 'Color'], 'from': 'colour',
                          'to': 'Color', 'rule': 'spelling',
                          'folds': ['case', 'british']}]

    def test_synonym(self) -> None:
        canonical, edits = self.run({'Cap': {'Color': ['brown']}})
        assert canonical == {'Pileus': {'Color': ['brown']}}
        assert edits == [{'path': ['Pileus'], 'from': 'Cap',
                          'to': 'Pileus', 'rule': 'synonym', 'folds': []}]

    def test_synonym_target_without_a_table_spelling(self) -> None:
        """The map's own ``to`` is the spelling then."""
        canonical, _ = canonicalize({'cap': {}}, SYNONYMS, {})
        assert canonical == {'Pileus': {}}

    def test_group_not_in_the_table_is_left_as_written(self) -> None:
        annotated = {'Hymenophore': {'`Odd key`': ['x']}}
        assert self.run(annotated) == (annotated, [])

    def test_values_are_left_as_written(self) -> None:
        annotated = {'Conidia': {'Color': ['Colour brown', 'conidia']}}
        assert self.run(annotated) == (annotated, [])

    def test_lists_of_objects(self) -> None:
        canonical, edits = self.run({'Conidia': [{'colour': 'x'}]})
        assert canonical == {'Conidia': [{'Color': 'x'}]}
        assert edits[0]['path'] == ['Conidia', 0, 'Color']

    def test_input_is_not_modified(self) -> None:
        annotated = {'Cap': {'colour': ['brown']}}
        before = copy.deepcopy(annotated)
        self.run(annotated)
        assert annotated == before


class TestCollisions:
    """Decision 2026-11-17: merge children; conflicting leaf values are
    kept as a list; the merge is recorded."""

    def run(self, annotated: Dict[str, Any]) -> Any:
        return canonicalize(annotated, SYNONYMS, SPELLINGS)

    def test_children_merge(self) -> None:
        canonical, edits = self.run({
            'Cap': {'Color': ['brown']},
            'Pileus': {'Color': ['yellow'], 'Shape': ['convex']}})
        assert canonical == {'Pileus': {'Color': ['brown', 'yellow'],
                                        'Shape': ['convex']}}
        assert {'path': ['Pileus'], 'from': ['Cap', 'Pileus'],
                'to': 'Pileus', 'rule': 'merge'} in edits
        assert {'path': ['Pileus'], 'from': 'Cap', 'to': 'Pileus',
                'rule': 'synonym', 'folds': []} in edits

    def test_conflicting_scalars_become_a_list(self) -> None:
        canonical, _ = self.run({'Cap': {'Size': '5 cm'},
                                 'Pileus': {'Size': '6 cm'}})
        assert canonical == {'Pileus': {'Size': ['5 cm', '6 cm']}}

    def test_equal_scalars_stay_scalar(self) -> None:
        canonical, _ = self.run({'Cap': {'Size': '5 cm'},
                                 'Pileus': {'Size': '5 cm'}})
        assert canonical == {'Pileus': {'Size': '5 cm'}}

    def test_list_values_are_deduplicated_in_order(self) -> None:
        canonical, _ = self.run({'Cap': {'Color': ['brown']},
                                 'Pileus': {'Color': ['brown', 'yellow']}})
        assert canonical == {'Pileus': {'Color': ['brown', 'yellow']}}

    def test_spelling_variants_collide_too(self) -> None:
        canonical, edits = self.run({'conidia': {'Shape': ['ovoid']},
                                     'Conidia': {'Size': ['5 um']}})
        assert canonical == {'Conidia': {'Shape': ['ovoid'],
                                         'Size': ['5 um']}}
        assert any(e['rule'] == 'merge' for e in edits)


class TestCanonicalFields:
    def test_fields(self) -> None:
        doc = {'_id': 't1', 'json_annotated': '{"Cap": {"Color": ["brown"]}}'}
        fields = canonical_fields(doc, SYNONYMS, SPELLINGS, VERSION)
        assert fields is not None
        assert fields['json_canonical'] == {'Pileus': {'Color': ['brown']}}
        assert fields['canonical_edits'][0]['rule'] == 'synonym'
        assert fields['canonical_map_version'] == VERSION
        assert fields['canonical_source'] == source_hash(
            {'Cap': {'Color': ['brown']}})

    def test_written_document_is_fresh(self) -> None:
        doc: Dict[str, Any] = {'json_annotated': {'Cap': {}}}
        fields = canonical_fields(doc, SYNONYMS, SPELLINGS, VERSION)
        assert fields is not None
        doc.update(fields)
        assert is_fresh(doc, VERSION)

    def test_no_annotation(self) -> None:
        assert canonical_fields({'_id': 't2'}, SYNONYMS, SPELLINGS,
                                VERSION) is None


class TestValueGroup:
    @pytest.mark.parametrize('raw', [
        'Convex', ' `thin-walled` ', 'yes', 'No', '6-8 um', 'IKI-'])
    def test_builds_on_normalize_value(self, raw: str) -> None:
        assert value_group(raw) == normalize_value(raw)

    @pytest.mark.parametrize('raw, grouped', [
        ('0\u20131-septate', '0-1-septate'),     # en dash
        ('IKI\u2212', 'iki-'),                   # minus sign
        ('2\u20143 \u03bcm', '2-3 \u03bcm'),     # em dash
        ('1\u20102', '1-2'),                     # Unicode hyphen
    ])
    def test_dashes(self, raw: str, grouped: str) -> None:
        assert value_group(raw) == grouped

    @pytest.mark.parametrize('raw, grouped', [
        ('8spored', '8-spored'),
        ('1Septate', '1-septate'),
        ('2\u20133seriate', '2-3-seriate'),
        ('medianly 1septate', 'medianly 1-septate'),
        ('2celled', '2-celled'),
    ])
    def test_a_digit_joined_to_an_adjective(self, raw: str,
                                            grouped: str) -> None:
        assert value_group(raw) == grouped

    @pytest.mark.parametrize('raw', ['5days', '2guttules', '10um', '3x'])
    def test_other_joined_digits_are_kept(self, raw: str) -> None:
        assert value_group(raw) == normalize_value(raw)

    @pytest.mark.parametrize('raw, grouped', [
        ('eight-spored', '8-spored'),
        ('One-celled', '1-celled'),
        ('two-layered', '2-layered'),
        ('one-septate', '1-septate'),
        ('twelve-spored', '12-spored'),
    ])
    def test_number_words(self, raw: str, grouped: str) -> None:
        assert value_group(raw) == grouped

    @pytest.mark.parametrize('raw', [
        'eight spores', 'someone-celled', 'eighteen-spored', 'one',
        'Octosporous', 'Agaricus bisporus'])
    def test_other_words_are_kept(self, raw: str) -> None:
        assert value_group(raw) == normalize_value(raw)

    def test_reaction_signs_stay_apart(self) -> None:
        assert len({value_group(v) for v in ('IKI-', 'IKI+', 'IKI')}) == 3

    @pytest.mark.parametrize('raw', [
        'Eight\u2013spored', '8spored', '0\u20131septate', ' yes '])
    def test_idempotent(self, raw: str) -> None:
        assert value_group(value_group(raw)) == value_group(raw)


def test_value_group_is_covered_by_the_rules_fingerprint() -> None:
    assert value_group in RULE_FUNCTIONS
