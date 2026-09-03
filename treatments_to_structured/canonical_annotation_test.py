#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.canonical_annotation``.

The transform that turns a raw bootstrap annotation into one or more
canonical ones: a top-level feature label plus a **path** into the
attribute tree, per the schema decision of 2026-09-02/03 (see
``docs/feature_label_singletons.md``).

**Deterministic, and that was the operator's call.**  The alternative
was prompt instructions, and it cannot work: the annotator sees 9 seed
labels and rule 2 tells it to *invent* names, so it has no way to know
a label is new — and the prompt's rule 3 already asks for one feature
per span, which the compounds violate.

**A path, not a flat sub-attribute.**  ``build_vocab_tree.add_json``
already represents position as ``path + [key]`` at arbitrary depth, so
``['Peridium', 'hyphae', 'width']`` needs no schema change where a flat
field would need a second one.  The path applies to the *whole clause*,
which is why it lives on the span record rather than inside the label
string.

**Map-wins precedence.**  A label that is itself a hand-map target is
returned whole, so the fixed-point property holds by construction
rather than by accident of whether its head clears the support guard.

Both control sets run at the end of this file: no rule may contradict
the hand map, and no rule may collapse a recorded non-synonym.
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
    vocabulary_index,
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
    'partial veil': 'Partial veil',
    'partial veil microscopic': 'Partial veil microscopic',
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
    'partial veil': 'Partial veil',
}

# Hand-map targets: decisions a human already made.
PROTECTED = frozenset({'partial veil microscopic'})

SOURCE_DB = 'skol_exp_production_v4_02_50_features_candidate'


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


def _canon(label: str, protected=frozenset()):
    return canonicalize_label(
        label, known=KNOWN, established=ESTABLISHED, protected=protected)


def _records(annotation, protected=frozenset()):
    return canonical_records(
        annotation, known=KNOWN, established=ESTABLISHED,
        protected=protected, source_db=SOURCE_DB)


class TestCanonicalLabel:
    def test_label_is_the_head_of_the_path(self) -> None:
        assert CanonicalLabel(path=('Ascomata', 'height')).label == 'Ascomata'

    def test_a_depth_one_path_is_still_a_path(self) -> None:
        assert CanonicalLabel(path=('Ascomata',)).label == 'Ascomata'


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
        """`Colony on OA and PCA` is one feature observed on two media,
        not two features.  It must never reach the compound splitter."""
        base, media, _ = split_condition('Colony on OA and PCA')
        assert base == 'Colony' and len(media) == 2

    def test_the_mycological_sense_of_context_is_untouched(self) -> None:
        """`Pileus context` is flesh, not a growth condition.  The field
        this replaces was named `context`, which collided with
        `context_color` in schemas/pileus.json -- the reason for the
        rename to medium/condition."""
        assert split_condition('Pileus context') == (
            'Pileus context', (), None)


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
        `Biofilm Architecture` onto it would invent a hierarchy from two
        equally rare labels."""
        assert strip_sub_attribute('Biofilm Architecture', ESTABLISHED) == (
            'Biofilm Architecture', None)

    def test_a_head_that_is_not_a_label_at_all_is_left_alone(self) -> None:
        """The case that protects the hyphal-system family: `Generative`
        is not a feature, so `Generative hyphae` survives whole -- which
        docs/feature_label_non_synonyms.md requires."""
        assert strip_sub_attribute('Generative hyphae', ESTABLISHED) == (
            'Generative hyphae', None)

    def test_single_word_labels_are_left_alone(self) -> None:
        assert strip_sub_attribute('Ascomata', ESTABLISHED) == (
            'Ascomata', None)


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


class TestCanonicalizeLabel:
    """The pipeline, where order is the whole design."""

    def test_condition_is_taken_before_compound_splitting(self) -> None:
        """`Colony on OA and PCA` must come out as ONE label with two
        media.  If the compound splitter saw the "and" first it would
        make two features out of one observation."""
        assert _canon('Colony on OA and PCA') == [CanonicalLabel(
            path=('Colony',), media=('OA', 'PCA'),
            transforms=('condition',))]

    def test_a_compound_becomes_several_labels(self) -> None:
        got = _canon('Gamma and beta conidia')
        assert [c.label for c in got] == ['Gamma conidia', 'Beta conidia']
        assert all('compound' in c.transforms for c in got)

    def test_a_sub_attribute_becomes_a_two_step_path(self) -> None:
        assert _canon('Ascomata height') == [CanonicalLabel(
            path=('Ascomata', 'height'), transforms=('sub_attribute',))]

    def test_case_folding_runs_first(self) -> None:
        """`Colony Reverse` folds to `Colony reverse`, which then strips
        to a path.  Folding after stripping would have missed it."""
        assert _canon('Colony Reverse') == [CanonicalLabel(
            path=('Colony', 'reverse'),
            transforms=('case_fold', 'sub_attribute'))]

    def test_an_untouched_label_records_no_transforms(self) -> None:
        assert _canon('Ascomata') == [CanonicalLabel(path=('Ascomata',))]


class TestMapWinsPrecedence:
    """A hand-map target is a decision already made.

    Without this, `Partial veil microscopic` survives only because
    `Partial veil` sits at df 3 -- an accident of support, not a design.
    Step 2 of the plan rewrites those targets as paths deliberately;
    until then no rule may touch them.
    """

    def test_a_protected_target_is_not_decomposed(self) -> None:
        assert _canon('Partial veil microscopic', PROTECTED) == [
            CanonicalLabel(path=('Partial veil microscopic',))]

    def test_the_same_label_decomposes_when_unprotected(self) -> None:
        """The guard is doing real work here, not nothing."""
        assert _canon('Partial veil microscopic') == [CanonicalLabel(
            path=('Partial veil', 'microscopic'),
            transforms=('sub_attribute',))]

    def test_protection_is_case_insensitive(self) -> None:
        assert _canon('PARTIAL VEIL MICROSCOPIC', PROTECTED)[0].label == (
            'Partial veil microscopic')


class TestCanonicalRecords:
    def test_passthrough_fields_survive(self) -> None:
        rec, = _records(_ann('Colony on MEA'))
        for key in ('field', 'start', 'end', 'source_text',
                    'source_spans', 'treatment_id', 'doc_id', 'model',
                    'created_at', 'round'):
            assert key in rec

    def test_id_follows_the_candidate_scheme(self) -> None:
        rec, = _records(_ann('Colony on MEA'))
        assert rec['_id'] == 'taxon_abc:Colony:48'

    def test_raw_label_and_source_are_kept_for_traceability(self) -> None:
        """The derived DB has to be diffable against the raw one -- that
        is the whole argument for deriving instead of mutating."""
        rec, = _records(_ann('Colony on MEA'))
        assert rec['raw_label'] == 'Colony on MEA'
        assert rec['source_db'] == SOURCE_DB
        assert rec['feature_label'] == 'Colony'
        assert rec['medium'] == ['MEA']

    def test_attribute_path_is_always_present(self) -> None:
        """Including at depth 1.  A consumer walking paths should not
        need a special case at the root."""
        plain, = _records(_ann('Ascomata'))
        assert plain['attribute_path'] == ['Ascomata']
        deep, = _records(_ann('Ascomata height'))
        assert deep['attribute_path'] == ['Ascomata', 'height']

    def test_a_compound_annotation_becomes_two_records(self) -> None:
        recs = _records(_ann(
            'Gamma and beta conidia',
            source_text='gamma and beta conidia are not observed'))
        assert len(recs) == 2
        assert {r['feature_label'] for r in recs} == {
            'Gamma conidia', 'Beta conidia'}
        assert {r['_id'] for r in recs} == {
            'taxon_abc:Gamma conidia:48', 'taxon_abc:Beta conidia:48'}

    def test_both_records_of_a_compound_share_the_span(self) -> None:
        """One clause, two features.  The path applies to the whole
        clause, so the span is not divided."""
        recs = _records(_ann(
            'Gamma and beta conidia',
            source_text='gamma and beta conidia are not observed'))
        assert {r['start'] for r in recs} == {48}
        assert {r['end'] for r in recs} == {96}

    def test_absence_is_recorded_as_a_value_not_a_suppression(self) -> None:
        """Absence is diagnostic -- operator, 2026-09-03.  It becomes a
        value on the label, never a reason to drop it."""
        recs = _records(_ann(
            'Gamma and beta conidia',
            source_text='gamma and beta conidia are not observed'))
        assert all(r['presence'] == 'absent' for r in recs)

    def test_absent_dimensions_omit_their_keys(self) -> None:
        rec, = _records(_ann('Ascomata'))
        for key in ('medium', 'condition', 'presence', 'raw_label',
                    'transforms'):
            assert key not in rec, key

    def test_transforms_are_recorded_when_anything_fired(self) -> None:
        rec, = _records(_ann('Colony Reverse'))
        assert rec['transforms'] == ['case_fold', 'sub_attribute']

    def test_an_empty_label_yields_nothing(self) -> None:
        assert _records(_ann('   ')) == []


class TestVocabularyIndex:
    """Support is *treatment* frequency, the unit corpus_vocabulary uses
    and for the same reason: forty repeats inside one document are one
    piece of evidence."""

    def _anns(self):
        return [
            {'treatment_id': 't1', 'feature_label': 'Pileus'},
            {'treatment_id': 't1', 'feature_label': 'Pileus'},
            {'treatment_id': 't2', 'feature_label': 'Pileus'},
            {'treatment_id': 't3', 'feature_label': 'Stipe'},
        ]

    def test_indexes_by_lower_case(self) -> None:
        assert vocabulary_index(self._anns())['pileus'] == 'Pileus'

    def test_repeats_within_one_treatment_count_once(self) -> None:
        assert vocabulary_index(self._anns(), min_df=2) == {
            'pileus': 'Pileus'}

    def test_min_df_filters(self) -> None:
        assert vocabulary_index(self._anns(), min_df=3) == {}

    def test_canonicalizer_applies_before_counting(self) -> None:
        got = vocabulary_index(
            self._anns(), canonicalizer={'Stipe': 'Pileus'}, min_df=3)
        assert got == {'pileus': 'Pileus'}


class TestControlSets:
    """The refusals, against the real map and the real non-synonym list
    rather than the miniature vocabulary."""

    def _real(self):
        from treatments_to_structured.feature_label_rules import (
            load_canonicalization,
        )
        mapping = load_canonicalization()
        # Values are paths as of step 2; the head is the label.
        known = {path[0].lower(): path[0] for path in mapping.values()}
        known.update({
            name.lower(): name for name in (
                'Sexual morph', 'Asexual morph', 'Macroconidia',
                'Microconidia', 'Cystidia', 'Cheilocystidia',
                'Conidiomata', 'Pycnidia', 'Hymenium', 'Subhymenium',
                'Generative hyphae', 'Vegetative hyphae', 'Spores',
                'Basidiospores', 'Ascospores', 'Basidiomata', 'Ascomata',
            )
        })
        # Only ATOMIC targets need protection: a multi-step path is
        # already what the rules would produce.
        protected = frozenset(
            path[0].lower() for path in mapping.values() if len(path) == 1
        )
        return mapping, known, protected

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
        _, known, protected = self._real()
        out = []
        for label in group:
            got = canonicalize_label(
                label, known=known, established=known, protected=protected)
            assert len(got) == 1, (label, got)
            out.append(got[0].label)
        assert len(set(out)) == len(group), out

    def test_every_hand_map_target_is_a_fixed_point(self) -> None:
        """Holds **by construction** under map-wins precedence, not by
        accident of whether a head clears the support guard."""
        mapping, known, protected = self._real()
        for path in set(mapping.values()):
            got = canonicalize_label(
                path[0], known=known, established=known,
                protected=protected)
            assert [c.label for c in got] == [path[0]], (path, got)


class TestVocabularyIndexSurfaceForm:
    """**Which spelling wins matters, and the first version got it
    backwards.**

    ``vocabulary_index`` built ``{label.lower(): label}`` as a dict
    comprehension, so among case variants the *last* one iterated won —
    effectively arbitrary.  Measured on the real corpus 2026-09-03: 33
    keys resolved to the rarer spelling, and ``fold_case`` then folded
    everything onto it.  ``Conidiogenous Cells`` (1 occurrence) beat
    ``Conidiogenous cells`` (550), mislabelling 565 canonical records.

    The frequent form wins.  Ties break toward the hand map's
    canonical form, and then lexicographically so the index is
    reproducible run to run.
    """

    def _variants(self):
        return (
            [{'treatment_id': f't{i}', 'feature_label': 'Conidiogenous cells'}
             for i in range(20)]
            + [{'treatment_id': 't99',
                'feature_label': 'Conidiogenous Cells'}]
        )

    def test_the_frequent_spelling_wins(self) -> None:
        got = vocabulary_index(self._variants())
        assert got['conidiogenous cells'] == 'Conidiogenous cells'

    def test_order_of_arrival_does_not_decide(self) -> None:
        """The rare form arriving last must not win, which is exactly
        how the first version failed."""
        reversed_arrival = list(reversed(self._variants()))
        assert vocabulary_index(reversed_arrival)[
            'conidiogenous cells'] == 'Conidiogenous cells'

    def test_a_tie_is_broken_deterministically(self) -> None:
        anns = [
            {'treatment_id': 't1', 'feature_label': 'Spore Print'},
            {'treatment_id': 't2', 'feature_label': 'Spore print'},
        ]
        first = vocabulary_index(anns)['spore print']
        second = vocabulary_index(list(reversed(anns)))['spore print']
        assert first == second

    def test_support_still_counts_treatments_not_occurrences(self) -> None:
        """The winner is decided on the same unit the guard uses."""
        anns = (
            [{'treatment_id': 't1', 'feature_label': 'Colony'}] * 40
            + [{'treatment_id': f't{i}', 'feature_label': 'colony'}
               for i in range(2, 5)]
        )
        assert vocabulary_index(anns)['colony'] == 'colony'


@pytest.mark.xfail(strict=True, reason='canonicalize_label ignores the map')
class TestTheMapIsConsulted:
    """**The map must be applied to the raw label, terminally.**

    Found 2026-09-03 by rebuilding after the path migration:
    `canonicalize_label` never consulted the map at all.  It reached
    the map's effect only by accident, because ``known`` is built from
    map-canonicalized labels and ``fold_case`` looks up there — which
    reproduces a *case* rename and nothing else.

    So every non-case entry was bypassed: `Colonies` stayed `Colonies`
    beside `Colony` (127 annotations), `Odor` stayed `Odor` (18),
    `Stroma` stayed `Stroma` (34), and `Culture characteristics` was
    *stripped* to `('Culture', 'characteristics')` (343) because its
    head is established — the rules overruling a human decision, which
    is the one thing map-wins exists to prevent.  674 annotations in
    all.
    """

    PATHS = {
        'Colonies': ('Colony',),
        'Culture characteristics': ('Cultural characteristics',),
        'Pileus context microstructure': ('Pileus', 'context',
                                          'microscopic'),
    }

    def _canon_mapped(self, label: str):
        return canonicalize_label(
            label, known=KNOWN, established=ESTABLISHED,
            protected=frozenset(), paths=self.PATHS)

    def test_a_mapped_label_takes_the_map_path(self) -> None:
        assert self._canon_mapped('Colonies') == [
            CanonicalLabel(path=('Colony',), transforms=('map',))]

    def test_the_map_beats_the_strip_rule(self) -> None:
        """`Culture` is established, so the strip rule would decompose
        this.  The map says otherwise and the map is a human decision."""
        assert self._canon_mapped('Culture characteristics') == [
            CanonicalLabel(path=('Cultural characteristics',),
                           transforms=('map',))]

    def test_a_multi_step_map_path_is_taken_whole(self) -> None:
        assert self._canon_mapped('Pileus context microstructure') == [
            CanonicalLabel(path=('Pileus', 'context', 'microscopic'),
                           transforms=('map',))]

    def test_an_unmapped_label_still_reaches_the_rules(self) -> None:
        assert self._canon_mapped('Ascomata height') == [
            CanonicalLabel(path=('Ascomata', 'height'),
                           transforms=('sub_attribute',))]


class TestMapParameterIsOptional:
    def test_no_map_means_the_rules_alone(self) -> None:
        """The parameter is optional so every existing caller and test
        keeps its meaning."""
        assert _canon('Ascomata height') == [CanonicalLabel(
            path=('Ascomata', 'height'), transforms=('sub_attribute',))]
