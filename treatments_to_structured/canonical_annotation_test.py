#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.canonical_annotation``.

The transform that turns a raw bootstrap annotation into one or more
canonical ones: a top-level feature label plus named sub-attributes,
per the schema decision of 2026-09-02 (see
``docs/feature_label_singletons.md``).

**Every rule here is deterministic and guarded.**  The alternative
considered and rejected was prompt instructions — the annotator sees 9
seed labels and is told to invent names, so it cannot know whether a
label is new, and the prompt already carries a "one feature per span"
rule that the compounds violate.

Both control sets from ``feature_label_rules_test`` apply again at the
end of this file: no rule may contradict the hand map, and no rule may
collapse a recorded non-synonym.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.canonical_annotation import (  # noqa: E402
    CanonicalLabel,
    canonical_records,
    canonicalize_label,
    fold_case,
    presence_from_span,
    split_compound,
    split_condition,
    strip_sub_attribute,
)


# A miniature vocabulary standing in for the corpus: `known` is every
# label, `established` only those with df >= 5.  The guard on
# sub-attribute stripping is that the *head* must be established --
# otherwise a df-1 label becomes the parent of a df-1 label and the
# hierarchy is invented rather than found.
KNOWN: Dict[str, str] = {
    'colony': 'Colony',
    'colony reverse': 'Colony reverse',
    'ascomata': 'Ascomata',
    'asci': 'Asci',
    'conidia': 'Conidia',
    'gamma conidia': 'Gamma conidia',
    'beta conidia': 'Beta conidia',
    'basidia': 'Basidia',
    'cheilocystidia': 'Cheilocystidia',
    'lower surface': 'Lower surface',
    'biofilm': 'Biofilm',
    'generative hyphae': 'Generative hyphae',
    'megaconidia': 'Megaconidia',
}
ESTABLISHED: Dict[str, str] = {
    'colony': 'Colony',
    'ascomata': 'Ascomata',
    'asci': 'Asci',
    'conidia': 'Conidia',
    'gamma conidia': 'Gamma conidia',
    'beta conidia': 'Beta conidia',
    'basidia': 'Basidia',
    'cheilocystidia': 'Cheilocystidia',
    'lower surface': 'Lower surface',
}


def _ann(label: str, **over: object) -> Dict[str, object]:
    base: Dict[str, object] = {
        'feature_label': label,
        'field': 'description',
        'start': 48,
        'end': 96,
        'source_text': 'Colonies on MEA reaching 40 mm in 7 days.',
        'source_spans': [{'start': 100, 'end': 148}],
        'treatment_id': 'taxon_abc',
        'doc_id': 'ingest_xyz',
        'model': 'claude-opus-4-7',
        'created_at': '2026-09-01T00:00:00Z',
        'round': 6,
    }
    base.update(over)
    return base


@pytest.mark.xfail(strict=True, reason='canonical_annotation is a skeleton')
class TestFoldCase:
    @pytest.mark.parametrize('label,expected', [
        ('Lower Surface', 'Lower surface'),
        ('COLONY', 'Colony'),
        ('Colony', 'Colony'),
    ])
    def test_folds_onto_an_existing_label(
            self, label: str, expected: str) -> None:
        assert fold_case(label, KNOWN) == expected

    def test_does_not_invent_a_label(self) -> None:
        """A label with no case-variant in the vocabulary is left
        exactly as it came."""
        assert fold_case('Venae Externae', KNOWN) == 'Venae Externae'


@pytest.mark.xfail(strict=True, reason='canonical_annotation is a skeleton')
class TestSplitCondition:
    @pytest.mark.parametrize('label,base,media,condition', [
        ('Colony on MEA', 'Colony', ('MEA',), None),
        ('Colony MEA', 'Colony', ('MEA',), None),
        ('Colony on OA and PCA', 'Colony', ('OA', 'PCA'), None),
        ('Conidia in culture', 'Conidia', (), 'in culture'),
        ('Asci in culture MEA', 'Asci', ('MEA',), 'in culture'),
        ('Conidia in vitro', 'Conidia', (), 'in vitro'),
        ('Conidia in vivo', 'Conidia', (), 'in vivo'),
        ('Conidia on host', 'Conidia', (), 'on host'),
        ('Colony', 'Colony', (), None),
        ('Ascomata', 'Ascomata', (), None),
    ])
    def test_named_dimensions_come_out_of_the_label(
            self, label: str, base: str, media: Tuple[str, ...],
            condition: Optional[str]) -> None:
        assert split_condition(label) == (base, media, condition)

    def test_two_media_stay_two_values(self) -> None:
        """`Colony on OA and PCA` is one feature observed on two
        media, not two features.  It must never reach the compound
        splitter."""
        base, media, _ = split_condition('Colony on OA and PCA')
        assert base == 'Colony' and len(media) == 2

    def test_the_mycological_sense_of_context_is_untouched(self) -> None:
        """`Pileus context` is flesh, not a growth condition.  The
        field this replaces was named `context`, which collided with
        `context_color` in schemas/pileus.json -- the reason for the
        rename."""
        assert split_condition('Pileus context') == (
            'Pileus context', (), None)


@pytest.mark.xfail(strict=True, reason='canonical_annotation is a skeleton')
class TestStripSubAttribute:
    @pytest.mark.parametrize('label,base,sub', [
        ('Ascomata height', 'Ascomata', 'height'),
        ('Colony colour', 'Colony', 'colour'),
        ('Colony reverse', 'Colony', 'reverse'),
    ])
    def test_an_established_head_keeps_the_label(
            self, label: str, base: str, sub: str) -> None:
        assert strip_sub_attribute(label, ESTABLISHED) == (base, sub)

    def test_a_rare_head_is_not_promoted_to_a_parent(self) -> None:
        """`Biofilm` has df 1 in the real corpus.  Stripping
        `Biofilm Architecture` onto it would invent a hierarchy from
        two equally rare labels."""
        assert strip_sub_attribute('Biofilm Architecture', ESTABLISHED) == (
            'Biofilm Architecture', None)

    def test_a_head_that_is_not_a_label_at_all_is_left_alone(self) -> None:
        """The case that protects the hyphal-system family:
        `Generative` is not a feature, so `Generative hyphae` survives
        whole -- which docs/feature_label_non_synonyms.md requires."""
        assert strip_sub_attribute('Generative hyphae', ESTABLISHED) == (
            'Generative hyphae', None)

    def test_single_word_labels_are_left_alone(self) -> None:
        assert strip_sub_attribute('Ascomata', ESTABLISHED) == (
            'Ascomata', None)


@pytest.mark.xfail(strict=True, reason='canonical_annotation is a skeleton')
class TestSplitCompound:
    @pytest.mark.parametrize('label,parts', [
        ('Basidia and cheilocystidia', ['Basidia', 'Cheilocystidia']),
        ('Gamma and beta conidia', ['Gamma conidia', 'Beta conidia']),
        ('Beta or gamma conidia', ['Beta conidia', 'Gamma conidia']),
    ])
    def test_splits_when_every_half_resolves(
            self, label: str, parts: list) -> None:
        assert split_compound(label, KNOWN) == parts

    @pytest.mark.parametrize('label', [
        'Mega- and microconidia',        # microconidia is not a label here
        'Kinetosome and centriole',      # neither half is
        'Ascomata and Pycnidia co-occurrence',   # compound + sub-attribute
    ])
    def test_refuses_when_a_half_does_not_resolve(
            self, label: str) -> None:
        """An honest refusal.  Splitting on faith would mint labels
        rather than consolidate them."""
        assert split_compound(label, KNOWN) is None

    def test_a_plain_label_is_not_a_compound(self) -> None:
        assert split_compound('Ascomata', KNOWN) is None


@pytest.mark.xfail(strict=True, reason='canonical_annotation is a skeleton')
class TestPresenceFromSpan:
    @pytest.mark.parametrize('text', [
        'gamma and beta conidia are not observed',
        'Micro- or macropycnidia not seen.',
        'Chlamydospores absent.',
        'none observed in culture',
        'Sexual morph lacking.',
    ])
    def test_absence_is_detected(self, text: str) -> None:
        assert presence_from_span(text) == 'absent'

    def test_ordinary_description_says_nothing(self) -> None:
        """Presence is the default and is not recorded -- marking it
        would put a redundant key on every annotation in the corpus."""
        assert presence_from_span(
            'Colonies on MEA reaching 40 mm in 7 days.') is None


@pytest.mark.xfail(strict=True, reason='canonical_annotation is a skeleton')
class TestCanonicalizeLabel:
    """The pipeline, where order is the whole design."""

    def _run(self, label: str):
        return canonicalize_label(
            label, known=KNOWN, established=ESTABLISHED)

    def test_condition_is_taken_before_compound_splitting(self) -> None:
        """`Colony on OA and PCA` must come out as ONE label with two
        media.  If the compound splitter saw the "and" first it would
        try to make two features out of one observation."""
        got = self._run('Colony on OA and PCA')
        assert got == [CanonicalLabel(
            label='Colony', media=('OA', 'PCA'),
            transforms=('condition',))]

    def test_a_compound_becomes_several_labels(self) -> None:
        got = self._run('Gamma and beta conidia')
        assert [c.label for c in got] == ['Gamma conidia', 'Beta conidia']
        assert all('compound' in c.transforms for c in got)

    def test_sub_attribute_survives_the_pipeline(self) -> None:
        got = self._run('Ascomata height')
        assert got == [CanonicalLabel(
            label='Ascomata', sub_attribute='height',
            transforms=('sub_attribute',))]

    def test_case_folding_runs_first(self) -> None:
        """`Colony Reverse` folds to `Colony reverse`, which then
        strips to `Colony` + `reverse`.  Folding after stripping would
        have missed it."""
        got = self._run('Colony Reverse')
        assert got == [CanonicalLabel(
            label='Colony', sub_attribute='reverse',
            transforms=('case_fold', 'sub_attribute'))]

    def test_an_untouched_label_records_no_transforms(self) -> None:
        assert self._run('Ascomata') == [CanonicalLabel(label='Ascomata')]


@pytest.mark.xfail(strict=True, reason='canonical_annotation is a skeleton')
class TestCanonicalRecords:
    def test_passthrough_fields_survive(self) -> None:
        rec, = canonical_records(
            _ann('Colony on MEA'), known=KNOWN, established=ESTABLISHED)
        for key in ('field', 'start', 'end', 'source_text',
                    'source_spans', 'treatment_id', 'doc_id', 'model',
                    'created_at', 'round'):
            assert key in rec

    def test_id_follows_the_candidate_scheme(self) -> None:
        rec, = canonical_records(
            _ann('Colony on MEA'), known=KNOWN, established=ESTABLISHED)
        assert rec['_id'] == 'taxon_abc:Colony:48'

    def test_raw_label_is_kept_for_traceability(self) -> None:
        """The derived DB has to be diffable against the candidate DB
        -- that is the whole argument for deriving instead of
        mutating."""
        rec, = canonical_records(
            _ann('Colony on MEA'), known=KNOWN, established=ESTABLISHED)
        assert rec['raw_label'] == 'Colony on MEA'
        assert rec['feature_label'] == 'Colony'
        assert rec['medium'] == ['MEA']

    def test_a_compound_annotation_becomes_two_records(self) -> None:
        recs = canonical_records(
            _ann('Gamma and beta conidia',
                 source_text='gamma and beta conidia are not observed'),
            known=KNOWN, established=ESTABLISHED)
        assert len(recs) == 2
        assert {r['feature_label'] for r in recs} == {
            'Gamma conidia', 'Beta conidia'}
        assert {r['_id'] for r in recs} == {
            'taxon_abc:Gamma conidia:48', 'taxon_abc:Beta conidia:48'}

    def test_absence_is_recorded_as_a_value_not_a_suppression(
            self) -> None:
        """Absence is diagnostic -- operator, 2026-09-03.  It becomes
        a value on the label, never a reason to drop it."""
        recs = canonical_records(
            _ann('Gamma and beta conidia',
                 source_text='gamma and beta conidia are not observed'),
            known=KNOWN, established=ESTABLISHED)
        assert all(r['presence'] == 'absent' for r in recs)

    def test_absent_dimensions_omit_their_keys(self) -> None:
        rec, = canonical_records(
            _ann('Ascomata'), known=KNOWN, established=ESTABLISHED)
        for key in ('medium', 'condition', 'sub_attribute', 'presence',
                    'raw_label'):
            assert key not in rec, key

    def test_transforms_are_recorded_when_anything_fired(self) -> None:
        rec, = canonical_records(
            _ann('Colony Reverse'), known=KNOWN, established=ESTABLISHED)
        assert rec['transforms'] == ['case_fold', 'sub_attribute']


@pytest.mark.xfail(strict=True, reason='canonical_annotation is a skeleton')
class TestControlSets:
    """The refusals, again, against the real map and the real
    non-synonym list rather than the miniature vocabulary."""

    def _real(self):
        from treatments_to_structured.feature_label_rules import (
            load_canonicalization,
        )
        mapping = load_canonicalization()
        known = {v.lower(): v for v in mapping.values()}
        known.update({
            name.lower(): name for name in (
                'Sexual morph', 'Asexual morph', 'Macroconidia',
                'Microconidia', 'Cystidia', 'Cheilocystidia',
                'Conidiomata', 'Pycnidia', 'Hymenium', 'Subhymenium',
                'Generative hyphae', 'Vegetative hyphae', 'Spores',
                'Basidiospores', 'Ascospores', 'Basidiomata', 'Ascomata',
            )
        })
        return mapping, known

    @pytest.mark.parametrize('group', [
        ('Sexual morph', 'Asexual morph'),
        ('Macroconidia', 'Microconidia'),
        ('Cystidia', 'Cheilocystidia'),
        ('Conidiomata', 'Pycnidia'),
        ('Hymenium', 'Subhymenium'),
        ('Generative hyphae', 'Vegetative hyphae'),
        ('Spores', 'Basidiospores', 'Ascospores'),
        ('Basidiomata', 'Ascomata'),
    ])
    def test_no_recorded_non_synonym_collapses(self, group) -> None:
        _, known = self._real()
        out = []
        for label in group:
            got = canonicalize_label(label, known=known, established=known)
            assert len(got) == 1, (label, got)
            out.append(got[0].label)
        assert len(set(out)) == len(group), out

    def test_every_hand_map_target_is_a_fixed_point(self) -> None:
        """A canonical label must survive the pipeline unchanged, or
        repeated passes would drift."""
        mapping, known = self._real()
        for target in set(mapping.values()):
            got = canonicalize_label(
                target, known=known, established=known)
            assert [c.label for c in got] == [target], (target, got)
