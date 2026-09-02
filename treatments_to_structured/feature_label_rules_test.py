#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.feature_label_rules``.

**The control sets are the point of this file.**  A canonicalization
rule is only as good as its refusals: a missed merge costs one
duplicate label, a wrong merge silently deletes a biological concept
and nothing downstream fails.  So every rule here is tested against

* **positive controls** — every entry of
  ``docs/feature_label_canonicalization.json``, which no rule may
  contradict; and
* **negative controls** — every pair recorded in
  ``docs/feature_label_non_synonyms.md`` as deliberately *not*
  merged, which no rule may collapse.

The negative pairs are transcribed here rather than parsed out of the
prose so that editing the doc cannot silently weaken the test.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.feature_label_rules import (  # noqa: E402
    canonical_morph,
    canonicalize,
    load_canonicalization,
    split_medium_context,
)


# Pairs docs/feature_label_non_synonyms.md records as NOT synonyms.
# Each tuple is a group whose members must stay mutually distinct.
NON_SYNONYMS = [
    ('Sexual morph', 'Asexual morph'),
    ('Macroconidia', 'Microconidia'),
    ('Endoperidium', 'Exoperidium'),
    ('Generative hyphae', 'Vegetative hyphae'),
    ('Primary branches', 'Tertiary branches'),
    ('Spores', 'Basidiospores', 'Ascospores'),
    ('Basidiomata', 'Ascomata'),
    ('Fertile hyphae', 'Generative hyphae'),
    ('Apothecia', 'Hypothecium'),
    ('Hymenium', 'Subhymenium', 'Epihymenium'),
    ('Peridium', 'Exoperidium'),
    ('Cystidia', 'Cheilocystidia', 'Pleurocystidia',
     'Pileocystidia', 'Gloeocystidia'),
    ('Cheilocystidia', 'Cheilolamprocystidia'),
    ('Pleurocystidia', 'Pleurolamprocystidia', 'Pleuropseudocystidia'),
    ('Conidiomata', 'Pycnidia'),
    # "Same feature, different substrate or medium" -- the medium is
    # the entire point of the observation.
    ('Colony on MEA', 'Colony on OA', 'Colony on PDA'),
    ('Culture on CMA', 'Culture on MEA', 'Culture on DG18'),
    ('Asci in culture MEA', 'Asci in culture V8'),
]


class TestCanonicalMorph:
    """The sexual/asexual family: one modifier, four head nouns."""

    @pytest.mark.parametrize('label', [
        'Sexual morph', 'Sexual stage', 'Sexual state', 'Sexual phase',
        'SexualMorph', 'sexual stage', 'Sexual  Stage',
    ])
    def test_sexual_forms_reach_the_canonical(self, label: str) -> None:
        assert canonical_morph(label) == 'Sexual morph'

    @pytest.mark.parametrize('label', [
        'Asexual morph', 'Asexual stage', 'Asexual state',
        'AsexualMorph', 'asexual state',
    ])
    def test_asexual_forms_reach_the_canonical(self, label: str) -> None:
        assert canonical_morph(label) == 'Asexual morph'

    def test_the_modifier_is_never_crossed(self) -> None:
        """`Sexual morph`/`Asexual morph` score 0.96 -- the highest
        similarity in the corpus and the most dangerous pair.  The
        rule normalises the *head noun* only."""
        assert canonical_morph('Sexual stage') != canonical_morph(
            'Asexual stage')

    @pytest.mark.parametrize('label', [
        'Synasexual morph',          # a second anamorph, not the morph
        'Synasexual morph conidia',
        'Anamorph Stromata',         # a structure of the anamorph
        'Sexual structures',         # different granularity
        'Sexual reproduction',
        'Sexual organs',
        'Asexual Propagules',
        'Sexual Mating',
    ])
    def test_qualified_and_adjacent_labels_are_left_alone(
            self, label: str) -> None:
        """The rule fires only on a bare modifier + head noun.
        Anything carrying a further noun is a different feature."""
        assert canonical_morph(label) is None

    def test_unrelated_labels_return_none(self) -> None:
        assert canonical_morph('Pileus') is None


class TestSplitMediumContext:
    """Decomposition, never a merge.

    docs/feature_label_non_synonyms.md forbids collapsing the
    medium family and names the fix: "a separate `context` field,
    not a longer label".  This function performs that split and
    nothing else -- the medium survives in the second element.
    """

    @pytest.mark.parametrize('label,base,context', [
        ('Colony on MEA', 'Colony', 'MEA'),
        ('Colonies on SNA', 'Colonies', 'SNA'),
        ('Colony morphology on PDA', 'Colony morphology', 'PDA'),
        ('Conidia in culture', 'Conidia', 'culture'),
        ('Asci in culture MEA', 'Asci', 'culture MEA'),
        ('Colony in culture', 'Colony', 'culture'),
    ])
    def test_splits_the_condition_out(
            self, label: str, base: str, context: str) -> None:
        assert split_medium_context(label) == (base, context)

    @pytest.mark.parametrize('label,context', [
        ('Conidia in vitro', 'in vitro'),
        ('Ascomata in situ', 'in situ'),
    ])
    def test_latin_phrases_keep_their_preposition(
            self, label: str, context: str) -> None:
        """A stored context of `vitro` reads as a parsing bug."""
        assert split_medium_context(label)[1] == context

    def test_labels_without_a_condition_are_unchanged(self) -> None:
        assert split_medium_context('Colony') == ('Colony', None)
        assert split_medium_context('Pileus') == ('Pileus', None)

    def test_different_media_never_become_the_same_pair(self) -> None:
        """The whole objection to a strip-and-merge rule."""
        assert (split_medium_context('Colony on MEA')
                != split_medium_context('Colony on OA'))

    def test_conidiogenous_cells_keeps_its_head(self) -> None:
        assert split_medium_context('Conidiogenous cells in culture') == (
            'Conidiogenous cells', 'culture')


class TestCanonicalizeAgainstTheHandMap:
    """Positive controls: the hand map is the record of decisions
    already made, so no rule may disagree with one."""

    def test_every_map_entry_survives_canonicalize(self) -> None:
        mapping = load_canonicalization()
        for raw, canonical in mapping.items():
            assert canonicalize(raw, mapping) == canonical, raw

    def test_every_map_target_is_a_fixed_point(self) -> None:
        """Applying the rules to an already-canonical label must not
        move it, or repeated passes would drift."""
        mapping = load_canonicalization()
        for canonical in set(mapping.values()):
            assert canonicalize(canonical, mapping) == canonical

    def test_the_map_carries_the_forms_the_corpus_actually_has(
            self) -> None:
        """`Sexual state` (7 treatments) and `AsexualMorph` (1) are
        real candidate-DB labels; `Sexual stage`/`Asexual stage` are
        preventive, added 2026-09-01 after an operator hand-labelled
        two such spans with the canonical forms."""
        mapping = load_canonicalization()
        for raw, canonical in (
            ('Sexual state', 'Sexual morph'),
            ('Asexual state', 'Asexual morph'),
            ('Sexual stage', 'Sexual morph'),
            ('Asexual stage', 'Asexual morph'),
            ('AsexualMorph', 'Asexual morph'),
        ):
            assert mapping.get(raw) == canonical, raw


class TestCanonicalizeAgainstTheNonSynonyms:
    """Negative controls: the refusals are the safety argument."""

    @pytest.mark.parametrize('group', NON_SYNONYMS)
    def test_recorded_non_synonyms_stay_distinct(self, group) -> None:
        mapping = load_canonicalization()
        canon = [canonicalize(label, mapping) for label in group]
        assert len(set(canon)) == len(group), canon

    def test_the_medium_family_is_not_collapsed_by_canonicalize(
            self) -> None:
        """`canonicalize` must not reach for `split_medium_context`:
        until the `context` field exists, dropping the medium loses
        the observation."""
        mapping = load_canonicalization()
        assert canonicalize('Colony on MEA', mapping) == 'Colony on MEA'


class TestLoadCanonicalization:
    def test_comment_keys_are_dropped(self) -> None:
        mapping = load_canonicalization()
        assert not [k for k in mapping if k.startswith('_')]

    def test_reads_the_repo_map_by_default(self) -> None:
        path = (Path(__file__).resolve().parent.parent / 'docs'
                / 'feature_label_canonicalization.json')
        with path.open() as handle:
            raw = json.load(handle)
        expected = {k: v for k, v in raw.items() if not k.startswith('_')}
        assert load_canonicalization() == expected
