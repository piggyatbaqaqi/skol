"""Tests for treatments_to_structured.brat_render.

Covers the three coordinate systems (source plaintext / field-relative
/ synth-doc) and the round-trip from Treatment → synthetic .txt →
brat .ann → annotation dicts.
"""

from typing import Any, Dict, List, Optional

import pytest

from treatments_to_structured.brat_render import (
    FieldExtent,
    SpanMap,
    annotations_to_brat,
    brat_safe_type,
    parse_brat_ann,
    render,
)


def _make_treatment(
    description: Optional[str] = None,
    diagnosis: Optional[str] = None,
    description_spans: Optional[List[Dict[str, Any]]] = None,
    diagnosis_spans: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Production Treatment shape, minimal."""
    return {
        '_id': 'taxon_test',
        'description': description,
        'diagnosis': diagnosis,
        'description_spans': description_spans or [],
        'diagnosis_spans': diagnosis_spans or [],
    }


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


class TestRender:
    """Synthesize the brat .txt + SpanMap."""

    def test_both_fields_present(self) -> None:
        txt, span_map = render(_make_treatment(
            description='Pileus brown 3 cm.',
            description_spans=[
                {'start_char': 1000, 'end_char': 1018},
            ],
            diagnosis='Differs from M. brevicaulis.',
            diagnosis_spans=[
                {'start_char': 2000, 'end_char': 2028},
            ],
        ))
        assert txt.startswith('=== description ===\n\n')
        assert 'Pileus brown 3 cm.' in txt
        assert '=== diagnosis ===' in txt
        assert 'Differs from M. brevicaulis.' in txt
        # Description section comes first, then diagnosis section
        assert txt.index('=== description ===') < txt.index('=== diagnosis ===')

    def test_description_only_no_diagnosis_header(self) -> None:
        txt, span_map = render(_make_treatment(
            description='Pileus brown 3 cm.',
            description_spans=[{'start_char': 1000, 'end_char': 1018}],
        ))
        assert txt.startswith('=== description ===\n\n')
        assert '=== diagnosis ===' not in txt

    def test_diagnosis_only_no_description_header(self) -> None:
        txt, span_map = render(_make_treatment(
            diagnosis='Differs from M. brevicaulis.',
            diagnosis_spans=[{'start_char': 2000, 'end_char': 2028}],
        ))
        assert txt.startswith('=== diagnosis ===\n\n')
        assert '=== description ===' not in txt

    def test_both_null_renders_empty(self) -> None:
        txt, span_map = render(_make_treatment())
        assert txt == ''
        assert span_map.field_extents == []
        assert span_map.synth_text == ''

    def test_null_description_treated_as_absent(self) -> None:
        """description: null behaves like description missing."""
        txt, span_map = render(_make_treatment(
            description=None,
            diagnosis='Differs.',
            diagnosis_spans=[{'start_char': 2000, 'end_char': 2008}],
        ))
        assert '=== description ===' not in txt
        assert '=== diagnosis ===' in txt
        assert [e.field for e in span_map.field_extents] == ['diagnosis']

    def test_empty_string_description_treated_as_absent(self) -> None:
        txt, span_map = render(_make_treatment(
            description='',
            diagnosis='Differs.',
            diagnosis_spans=[{'start_char': 2000, 'end_char': 2008}],
        ))
        assert '=== description ===' not in txt

    def test_span_map_synth_offsets_locate_field_content(self) -> None:
        """synth_text[synth_start:synth_end] == the field prose."""
        txt, span_map = render(_make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 100, 'end_char': 113}],
        ))
        ext = span_map.field_extents[0]
        assert ext.field == 'description'
        assert txt[ext.synth_start:ext.synth_end] == 'Pileus brown.'

    def test_string_offsets_in_spans_normalized_to_int(self) -> None:
        """Production data has start_char/end_char sometimes as STRINGS
        (per the Trichoderma sample, '24693' not 24693).  render
        normalizes at module boundary."""
        txt, span_map = render(_make_treatment(
            description='Pileus brown.',
            description_spans=[{
                'start_char': '100', 'end_char': '113',
            }],
        ))
        ext = span_map.field_extents[0]
        assert isinstance(ext.source_spans[0]['start'], int)
        assert isinstance(ext.source_spans[0]['end'], int)
        assert ext.source_spans[0] == {'start': 100, 'end': 113}

    def test_multiple_source_spans_normalized(self) -> None:
        """A field with multiple source-plaintext spans (page-break
        interruption) records each as a dict with int start/end."""
        txt, span_map = render(_make_treatment(
            description='A B',
            description_spans=[
                {'start_char': 100, 'end_char': 101},
                {'start_char': 200, 'end_char': 202},
            ],
        ))
        ext = span_map.field_extents[0]
        assert len(ext.source_spans) == 2
        assert ext.source_spans[0] == {'start': 100, 'end': 101}
        assert ext.source_spans[1] == {'start': 200, 'end': 202}


# ---------------------------------------------------------------------------
# annotations_to_brat
# ---------------------------------------------------------------------------


class TestAnnotationsToBrat:
    """Build the brat .ann file body from annotation dicts."""

    def test_empty_input_renders_empty_string(self) -> None:
        _, span_map = render(_make_treatment(
            description='ignored',
            description_spans=[{'start_char': 0, 'end_char': 7}],
        ))
        assert annotations_to_brat([], span_map) == ''

    def test_single_annotation_t1_line(self) -> None:
        _, span_map = render(_make_treatment(
            description='Pileus brown 3 cm wide.',
            description_spans=[{'start_char': 100, 'end_char': 123}],
        ))
        ss = span_map.field_extents[0].synth_start
        annotations = [{
            'feature_label': 'Pileus',
            'field': 'description',
            'start': 0, 'end': 23,
        }]
        ann = annotations_to_brat(annotations, span_map)
        # field-relative (0, 23) translates to synth (ss, ss+23).
        assert ann == (
            f'T1\tPileus {ss} {ss + 23}\tPileus brown 3 cm wide.\n'
        )

    def test_multiple_annotations_sequential_ids(self) -> None:
        _, span_map = render(_make_treatment(
            description='A. B.',
            description_spans=[{'start_char': 0, 'end_char': 5}],
        ))
        annotations = [
            {'feature_label': 'Pileus', 'field': 'description',
             'start': 0, 'end': 2},
            {'feature_label': 'Stipe', 'field': 'description',
             'start': 3, 'end': 5},
        ]
        ann = annotations_to_brat(annotations, span_map)
        lines = ann.strip().split('\n')
        assert lines[0].startswith('T1\t')
        assert lines[1].startswith('T2\t')
        assert 'Pileus' in lines[0]
        assert 'Stipe' in lines[1]

    def test_annotation_text_escapes_embedded_newlines(self) -> None:
        """T-line text can't contain literal newlines (brat reads
        one T-line per file line), so we escape with literal
        backslash-n.  The custom skol brat fork unescapes back to
        a real newline before verifying against the .txt file
        (which has the real newline at the offsets).  Same
        convention as bin/yedda_to_brat.py.

        Without this: brat would either (a) reject the T-line as
        malformed if we kept the newline, or (b) reject the text
        as not matching the .txt offsets if we replaced with a
        space.
        """
        _, span_map = render(_make_treatment(
            description='Line one\nLine two',
            description_spans=[{'start_char': 0, 'end_char': 17}],
        ))
        annotations = [{
            'feature_label': 'Pileus',
            'field': 'description',
            'start': 0, 'end': 17,
        }]
        ann = annotations_to_brat(annotations, span_map)
        # One newline per T-line plus the trailing one.
        assert ann.count('\n') == 1
        # The embedded \n is escaped to literal backslash + n
        # (brat unescapes on read).
        assert 'Line one\\nLine two' in ann

    def test_feature_label_with_space_becomes_underscore(
        self,
    ) -> None:
        """Brat T-line types are single-token (no whitespace).
        Phase 1's bootstrap routinely produces multi-word labels
        ('Basal mycelium', 'Universal veil on pileus'), so on the
        wire the space becomes an underscore.  parse_brat_ann
        reverses the substitution."""
        _, span_map = render(_make_treatment(
            description='ignored',
            description_spans=[{'start_char': 0, 'end_char': 7}],
        ))
        ann_text = annotations_to_brat([{
            'feature_label': 'Pileus surface',  # space!
            'field': 'description',
            'start': 0, 'end': 7,
        }], span_map)
        # Wire format has the underscore...
        assert 'Pileus_surface' in ann_text
        # ...and round-trip restores the space.
        round_tripped = parse_brat_ann(ann_text, span_map)
        assert round_tripped[0]['feature_label'] == 'Pileus surface'

    def test_feature_label_with_tab_still_raises(self) -> None:
        """Tabs are an actual format violation (delimiter), not
        a normalization concern.  Still rejected."""
        _, span_map = render(_make_treatment(
            description='ignored',
            description_spans=[{'start_char': 0, 'end_char': 7}],
        ))
        with pytest.raises(ValueError) as exc:
            annotations_to_brat([{
                'feature_label': 'a\tb',
                'field': 'description',
                'start': 0, 'end': 7,
            }], span_map)
        assert 'tab' in str(exc.value).lower()

    def test_label_with_parens_sanitized_to_brat_safe(
        self,
    ) -> None:
        """Parens / commas / periods get stripped on the wire so
        brat doesn't auto-mangle (and silently corrupt the
        reviewer's saved .ann).  Real case observed live on
        2026-06-29: 'Partial veil (microscopic)' produced the
        brat warning 'is not appropriate for storage'."""
        _, span_map = render(_make_treatment(
            description='ignored',
            description_spans=[{'start_char': 0, 'end_char': 7}],
        ))
        ann_text = annotations_to_brat([{
            'feature_label': 'Partial veil (microscopic)',
            'field': 'description',
            'start': 0, 'end': 7,
        }], span_map)
        # Wire form: brat-safe, no parens, no double underscores.
        assert 'Partial_veil_microscopic' in ann_text
        assert '(' not in ann_text.split('\t')[1].split(' ')[0]
        # Round-trip: parse restores spaces.  Parens are
        # information-lost (we can't recover them); the label
        # becomes 'Partial veil microscopic'.
        round_tripped = parse_brat_ann(ann_text, span_map)
        assert (
            round_tripped[0]['feature_label']
            == 'Partial veil microscopic'
        )


class TestBratSafeType:
    """The brat-storage-regex-conforming label sanitizer."""

    def test_ascii_alnum_label_unchanged(self) -> None:
        """Simple labels with no special chars pass through."""
        assert brat_safe_type('Pileus') == 'Pileus'

    def test_underscore_label_unchanged(self) -> None:
        """Already-sanitized labels with internal underscores
        pass through (idempotency precondition)."""
        assert brat_safe_type('Basal_mycelium') == 'Basal_mycelium'

    def test_space_becomes_underscore(self) -> None:
        assert (
            brat_safe_type('Basal mycelium') == 'Basal_mycelium'
        )

    def test_parens_stripped(self) -> None:
        """Brat regex ^[a-zA-Z0-9_-]*$ doesn't allow parens."""
        assert (
            brat_safe_type('Pileus (cap)') == 'Pileus_cap'
        )

    def test_commas_stripped(self) -> None:
        assert (
            brat_safe_type('Veil, on pileus')
            == 'Veil_on_pileus'
        )

    def test_multi_special_chars_collapse_to_one(self) -> None:
        """The Phase 1 motivating case: 'Universal veil
        (microscopic, on pileus)' must produce a clean wire
        form with no consecutive underscores."""
        result = brat_safe_type(
            'Universal veil (microscopic, on pileus)',
        )
        assert result == 'Universal_veil_microscopic_on_pileus'
        # No double underscores anywhere.
        assert '__' not in result
        # No leading or trailing underscore.
        assert not result.startswith('_')
        assert not result.endswith('_')

    def test_period_stripped(self) -> None:
        """Common in Claude output like 'av. = 98' but unusual
        in feature labels.  Still sanitized."""
        assert brat_safe_type('a.b') == 'a_b'

    def test_idempotent(self) -> None:
        """Applying twice gives the same result as applying once.
        Important property because parse_claude_response and
        annotations_to_brat both sanitize defensively."""
        cases = [
            'Pileus',
            'Basal mycelium',
            'Pileus (cap)',
            'Universal veil (microscopic, on pileus)',
            'Veil, on pileus',
        ]
        for label in cases:
            once = brat_safe_type(label)
            twice = brat_safe_type(once)
            assert once == twice, (
                f'{label!r} not idempotent: '
                f'once={once!r} twice={twice!r}'
            )

    def test_hyphen_preserved(self) -> None:
        """Hyphens are in the brat regex's allowed set
        ([a-zA-Z0-9_-]), so they pass through unchanged."""
        assert brat_safe_type('thick-walled') == 'thick-walled'

    def test_empty_input(self) -> None:
        """Empty stays empty (no crash on edge case)."""
        assert brat_safe_type('') == ''

    def test_all_special_chars_becomes_empty(self) -> None:
        """A label that's entirely special chars (would be
        rejected upstream by 'non-empty' validation, but be
        defensive) collapses to empty after stripping."""
        assert brat_safe_type('()') == ''
        assert brat_safe_type(' , . ') == ''

    def test_unknown_field_raises(self) -> None:
        """Annotation pointing at a field not in span_map → error."""
        _, span_map = render(_make_treatment(
            description='ignored',
            description_spans=[{'start_char': 0, 'end_char': 7}],
        ))
        with pytest.raises(ValueError):
            annotations_to_brat([{
                'feature_label': 'Pileus',
                'field': 'diagnosis',  # not rendered
                'start': 0, 'end': 5,
            }], span_map)


# ---------------------------------------------------------------------------
# parse_brat_ann
# ---------------------------------------------------------------------------


class TestParseBratAnn:
    """Read brat .ann back into annotation dicts."""

    def test_empty_input(self) -> None:
        _, span_map = render(_make_treatment(
            description='ignored',
            description_spans=[{'start_char': 0, 'end_char': 7}],
        ))
        assert parse_brat_ann('', span_map) == []

    def test_skips_non_T_lines(self) -> None:
        """Brat also stores R-lines (relations) and A-lines
        (attributes) — Phase 1 ignores both."""
        _, span_map = render(_make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 0, 'end_char': 13}],
        ))
        ss = span_map.field_extents[0].synth_start
        ann_text = (
            f'T1\tPileus {ss} {ss + 13}\tPileus brown.\n'
            'R1\tArg1:T1 Arg2:T1\n'
            'A1\tConfidence T1 High\n'
            '\n'
        )
        result = parse_brat_ann(ann_text, span_map)
        assert len(result) == 1
        assert result[0]['feature_label'] == 'Pileus'

    def test_field_relative_offsets_correctly_computed(self) -> None:
        """An annotation in the description block translates to
        field='description' with field-relative offsets at (0, 13)."""
        _, span_map = render(_make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 100, 'end_char': 113}],
        ))
        ss = span_map.field_extents[0].synth_start
        ann_text = f'T1\tPileus {ss} {ss + 13}\tPileus brown.\n'
        result = parse_brat_ann(ann_text, span_map)
        assert result[0]['field'] == 'description'
        assert result[0]['start'] == 0
        assert result[0]['end'] == 13

    def test_source_text_extracted_from_synth(self) -> None:
        _, span_map = render(_make_treatment(
            description='Pileus brown 3 cm wide.',
            description_spans=[{'start_char': 100, 'end_char': 123}],
        ))
        ss = span_map.field_extents[0].synth_start
        ann_text = (
            f'T1\tPileus {ss} {ss + 23}\tPileus brown 3 cm wide.\n'
        )
        result = parse_brat_ann(ann_text, span_map)
        assert result[0]['source_text'] == 'Pileus brown 3 cm wide.'

    def test_source_spans_translated_for_single_source_span(self) -> None:
        """Field text from one source-plaintext range: annotation
        translates to one source_spans entry."""
        _, span_map = render(_make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 100, 'end_char': 113}],
        ))
        ss = span_map.field_extents[0].synth_start
        # Field-relative (0, 13) → source plaintext (100, 113)
        ann_text = f'T1\tPileus {ss} {ss + 13}\tPileus brown.\n'
        result = parse_brat_ann(ann_text, span_map)
        assert result[0]['source_spans'] == [{'start': 100, 'end': 113}]

    def test_partial_annotation_within_source_span(self) -> None:
        """Annotation covers part of one source span: source_spans
        narrows to the relevant sub-range."""
        _, span_map = render(_make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 100, 'end_char': 113}],
        ))
        ss = span_map.field_extents[0].synth_start
        # Field-relative (7, 12) → source (107, 112), i.e. 'brown'
        ann_text = f'T1\tPileus {ss + 7} {ss + 12}\tbrown\n'
        result = parse_brat_ann(ann_text, span_map)
        assert result[0]['source_spans'] == [{'start': 107, 'end': 112}]

    def test_annotation_crossing_source_span_boundary(self) -> None:
        """A treatment whose description was concatenated from TWO
        source-plaintext spans (e.g., page break in middle): an
        annotation straddling the boundary produces source_spans with
        two entries."""
        _, span_map = render(_make_treatment(
            description='AAAAAAAAAABBBBBBBBBB',   # 20 chars
            description_spans=[
                {'start_char': 100, 'end_char': 110},  # AAAAAAAAAA
                {'start_char': 200, 'end_char': 210},  # BBBBBBBBBB
            ],
        ))
        ss = span_map.field_extents[0].synth_start
        # Field-relative (5, 15) = 'AAAAA' + 'BBBBB'
        # → source (105, 110) + (200, 205)
        ann_text = f'T1\tTest {ss + 5} {ss + 15}\tAAAAABBBBB\n'
        result = parse_brat_ann(ann_text, span_map)
        assert result[0]['source_spans'] == [
            {'start': 105, 'end': 110},
            {'start': 200, 'end': 205},
        ]

    def test_annotation_in_diagnosis_field(self) -> None:
        """Offsets falling within the diagnosis section map to
        field='diagnosis' with appropriate field-relative offsets."""
        _, span_map = render(_make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 100, 'end_char': 113}],
            diagnosis='Differs.',
            diagnosis_spans=[{'start_char': 200, 'end_char': 208}],
        ))
        # description block: "=== description ===\n\nPileus brown.\n\n"
        #   header = 22, content = 13, trailer = 2  → ends at 37
        # diagnosis block starts at synth offset 37 with header (22 chars),
        # then content at 59
        diagnosis_ext = next(
            e for e in span_map.field_extents if e.field == 'diagnosis'
        )
        ann_text = (
            f'T1\tDistinction {diagnosis_ext.synth_start} '
            f'{diagnosis_ext.synth_end}\tDiffers.\n'
        )
        result = parse_brat_ann(ann_text, span_map)
        assert result[0]['field'] == 'diagnosis'
        assert result[0]['start'] == 0
        assert result[0]['end'] == 8
        assert result[0]['source_spans'] == [
            {'start': 200, 'end': 208},
        ]

    def test_offsets_in_header_raise(self) -> None:
        """An annotation that lands inside ``=== description ===``
        rather than inside the field content is a user error — raise
        so the operator can fix it."""
        _, span_map = render(_make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 0, 'end_char': 13}],
        ))
        # Synth offsets (0, 19) cover '=== description ===' header
        ann_text = 'T1\tPileus 0 19\t=== description ===\n'
        with pytest.raises(ValueError):
            parse_brat_ann(ann_text, span_map)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Write then read = original (on the round-tripping fields)."""

    def test_single_field_single_source_span(self) -> None:
        treatment = _make_treatment(
            description='Pileus brown 3 cm wide.',
            description_spans=[{'start_char': 100, 'end_char': 123}],
        )
        _, span_map = render(treatment)
        original = [{
            'feature_label': 'Pileus',
            'field': 'description',
            'start': 0, 'end': 23,
        }]
        ann_text = annotations_to_brat(original, span_map)
        parsed = parse_brat_ann(ann_text, span_map)
        assert len(parsed) == 1
        p = parsed[0]
        assert p['feature_label'] == 'Pileus'
        assert p['field'] == 'description'
        assert p['start'] == 0
        assert p['end'] == 23
        assert p['source_text'] == 'Pileus brown 3 cm wide.'
        assert p['source_spans'] == [{'start': 100, 'end': 123}]

    def test_round_trip_with_multiple_annotations(self) -> None:
        treatment = _make_treatment(
            description='Pileus brown. Stipe 3 cm.',
            description_spans=[{'start_char': 0, 'end_char': 25}],
        )
        _, span_map = render(treatment)
        original = [
            {'feature_label': 'Pileus', 'field': 'description',
             'start': 0, 'end': 13},
            {'feature_label': 'Stipe', 'field': 'description',
             'start': 14, 'end': 25},
        ]
        ann_text = annotations_to_brat(original, span_map)
        parsed = parse_brat_ann(ann_text, span_map)
        labels = [p['feature_label'] for p in parsed]
        assert labels == ['Pileus', 'Stipe']
        assert all(p['field'] == 'description' for p in parsed)

    def test_round_trip_preserves_discontiguous_source_spans(self) -> None:
        """Annotation straddling a source-span boundary preserves the
        two-element source_spans on round-trip."""
        treatment = _make_treatment(
            description='AAAAAAAAAABBBBBBBBBB',
            description_spans=[
                {'start_char': 100, 'end_char': 110},
                {'start_char': 200, 'end_char': 210},
            ],
        )
        _, span_map = render(treatment)
        original = [{
            'feature_label': 'Test',
            'field': 'description',
            'start': 5, 'end': 15,
        }]
        ann_text = annotations_to_brat(original, span_map)
        parsed = parse_brat_ann(ann_text, span_map)
        assert parsed[0]['source_spans'] == [
            {'start': 105, 'end': 110},
            {'start': 200, 'end': 205},
        ]
