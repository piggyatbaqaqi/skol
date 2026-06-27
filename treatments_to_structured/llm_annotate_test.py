"""Tests for treatments_to_structured.llm_annotate pure logic.

No Anthropic API key or network required — Claude responses are
constructed in-test.
"""

import json
from typing import Any, Dict

import pytest

from treatments_to_structured.brat_render import render
from treatments_to_structured.llm_annotate import (
    ClaudeResponseError,
    annotation_doc_id,
    build_user_prompt,
    parse_claude_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_treatment(
    description: str = 'Pileus brown 3 cm wide.',
    description_spans: Any = None,
    diagnosis: Any = None,
    diagnosis_spans: Any = None,
) -> Dict[str, Any]:
    return {
        '_id': 'taxon_test',
        'description': description,
        'description_spans': description_spans or [
            {'start_char': 100, 'end_char': 100 + len(description)},
        ],
        'diagnosis': diagnosis,
        'diagnosis_spans': diagnosis_spans or [],
    }


_PILEUS_SCHEMA_SAMPLE: Dict[str, Any] = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'title': 'Pileus',
    'description': (
        "The cap.  Hymenophore features and stipe features are "
        "explicitly NOT pileus."
    ),
    'type': 'object',
    'properties': {
        'size_mm': {'type': 'object'},
        'color': {'type': 'string'},
    },
}


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """Compose the user-turn prompt for one Treatment."""

    def test_includes_synth_text(self) -> None:
        synth = (
            '=== description ===\n\nPileus brown 3 cm wide.\n\n'
        )
        prompt = build_user_prompt(synth, _PILEUS_SCHEMA_SAMPLE, 'Pileus')
        assert 'Pileus brown 3 cm wide.' in prompt
        assert '=== description ===' in prompt

    def test_includes_full_schema_json(self) -> None:
        """The schema travels in the prompt — particularly the
        anatomical-boundary section that disambiguates pileus from
        hymenophore/stipe."""
        prompt = build_user_prompt('text', _PILEUS_SCHEMA_SAMPLE, 'Pileus')
        # Pretty-printed JSON dump of the schema should appear
        assert '"title": "Pileus"' in prompt
        assert 'Hymenophore features' in prompt

    def test_specifies_output_envelope(self) -> None:
        prompt = build_user_prompt('text', _PILEUS_SCHEMA_SAMPLE, 'Pileus')
        # Must instruct the response shape exactly so parse can rely on it
        assert '"spans"' in prompt
        assert '"text"' in prompt

    def test_feature_label_appears_in_prompt(self) -> None:
        prompt = build_user_prompt('text', _PILEUS_SCHEMA_SAMPLE, 'Pileus')
        assert 'Pileus' in prompt

    def test_works_for_arbitrary_feature_label(self) -> None:
        prompt = build_user_prompt('text', _PILEUS_SCHEMA_SAMPLE, 'Lamellae')
        assert 'Lamellae' in prompt


# ---------------------------------------------------------------------------
# parse_claude_response
# ---------------------------------------------------------------------------


class TestParseClaudeResponse:
    """Translate Claude's JSON response into annotation dicts."""

    def _setup(self) -> tuple:
        """Build a SpanMap from a small Treatment to drive the tests."""
        treatment = _make_treatment(
            description='Pileus brown 3 cm wide.',
            description_spans=[{'start_char': 100, 'end_char': 123}],
        )
        _, span_map = render(treatment)
        return span_map

    def test_one_span_recovers_offsets_and_source_spans(self) -> None:
        span_map = self._setup()
        response = json.dumps({
            'spans': [{'text': 'Pileus brown 3 cm wide.'}],
        })
        anns = parse_claude_response(
            response, span_map, 'Pileus', 'claude-opus-4-7',
            'taxon_test', 'src_doc', '2026-06-27T00:00:00Z',
        )
        assert len(anns) == 1
        ann = anns[0]
        assert ann['feature_label'] == 'Pileus'
        assert ann['field'] == 'description'
        assert ann['start'] == 0
        assert ann['end'] == 23
        assert ann['source_text'] == 'Pileus brown 3 cm wide.'
        assert ann['source_spans'] == [{'start': 100, 'end': 123}]
        assert ann['model'] == 'claude-opus-4-7'
        assert ann['created_at'] == '2026-06-27T00:00:00Z'
        assert ann['treatment_id'] == 'taxon_test'
        assert ann['doc_id'] == 'src_doc'

    def test_empty_spans_returns_empty_list(self) -> None:
        """Claude reporting 'no pileus mentions' is a legitimate signal."""
        span_map = self._setup()
        response = json.dumps({'spans': []})
        anns = parse_claude_response(
            response, span_map, 'Pileus', 'm', 'tid', 'did', 'ts',
        )
        assert anns == []

    def test_substring_span_recovers_partial_range(self) -> None:
        span_map = self._setup()
        response = json.dumps({'spans': [{'text': 'brown'}]})
        anns = parse_claude_response(
            response, span_map, 'Pileus', 'm', 'tid', 'did', 'ts',
        )
        # 'brown' is at field-relative offset 7-12
        assert anns[0]['start'] == 7
        assert anns[0]['end'] == 12
        assert anns[0]['source_spans'] == [{'start': 107, 'end': 112}]

    def test_multiple_spans_advance_cursor_left_to_right(self) -> None:
        """Two spans, the second of which is later in the doc, both
        get distinct offsets."""
        treatment = _make_treatment(
            description='Pileus brown.  Pileus also wide.',
            description_spans=[{'start_char': 100, 'end_char': 132}],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [
                {'text': 'Pileus brown.'},
                {'text': 'Pileus also wide.'},
            ],
        })
        anns = parse_claude_response(
            response, span_map, 'Pileus', 'm', 'tid', 'did', 'ts',
        )
        # First 'Pileus' at field-rel 0, second at field-rel 15
        assert anns[0]['start'] == 0
        assert anns[1]['start'] == 15

    def test_response_with_json_fences_tolerated(self) -> None:
        """LLMs sometimes wrap output in ```json fences despite the
        prompt's no-fence instruction.  Strip them tolerantly."""
        span_map = self._setup()
        response = (
            '```json\n'
            '{"spans": [{"text": "brown"}]}\n'
            '```'
        )
        anns = parse_claude_response(
            response, span_map, 'Pileus', 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1
        assert anns[0]['source_text'] == 'brown'

    def test_response_with_bare_fences_tolerated(self) -> None:
        span_map = self._setup()
        response = (
            '```\n'
            '{"spans": [{"text": "brown"}]}\n'
            '```'
        )
        anns = parse_claude_response(
            response, span_map, 'Pileus', 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1

    def test_invalid_json_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                'not json at all', span_map, 'Pileus', 'm',
                'tid', 'did', 'ts',
            )
        assert 'JSON' in str(exc.value)

    def test_missing_spans_key_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"annotations": []}', span_map, 'Pileus', 'm',
                'tid', 'did', 'ts',
            )
        assert "'spans'" in str(exc.value)

    def test_spans_not_a_list_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"spans": "Pileus brown."}', span_map, 'Pileus', 'm',
                'tid', 'did', 'ts',
            )
        assert 'list' in str(exc.value).lower()

    def test_span_missing_text_key_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"spans": [{"foo": "bar"}]}', span_map, 'Pileus', 'm',
                'tid', 'did', 'ts',
            )
        assert "'text'" in str(exc.value)

    def test_empty_text_string_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError):
            parse_claude_response(
                '{"spans": [{"text": ""}]}', span_map, 'Pileus', 'm',
                'tid', 'did', 'ts',
            )

    def test_text_not_found_in_synth_raises(self) -> None:
        """Claude hallucinated text that's not in the synthetic doc.
        Better to fail than silently invent offsets."""
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"spans": [{"text": "Lamellae cream-colored."}]}',
                span_map, 'Pileus', 'm', 'tid', 'did', 'ts',
            )
        assert 'not found' in str(exc.value)

    def test_span_crossing_field_boundary_raises(self) -> None:
        """An annotation that overlaps the synthetic-doc gap between
        description and diagnosis isn't a valid Phase 1 annotation —
        fail loudly so the operator can split it manually."""
        treatment = _make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 100, 'end_char': 113}],
            diagnosis='Pileus differs.',
            diagnosis_spans=[{'start_char': 200, 'end_char': 215}],
        )
        _, span_map = render(treatment)
        # A string that exists across the field boundary doesn't exist
        # as a single substring (the section markers intervene), so we
        # need to be more careful here.  This test just demonstrates
        # that the response parser refuses to invent cross-field spans.
        response = json.dumps({
            'spans': [{'text': 'brown.\n\n=== diagnosis ===\n\nPileus'}],
        })
        with pytest.raises(ClaudeResponseError):
            parse_claude_response(
                response, span_map, 'Pileus', 'm', 'tid', 'did', 'ts',
            )


# ---------------------------------------------------------------------------
# annotation_doc_id
# ---------------------------------------------------------------------------


class TestAnnotationDocId:
    """Construct the CouchDB _id for an annotation doc."""

    def test_format_is_treatment_feature_start(self) -> None:
        assert annotation_doc_id('taxon_abc', 'Pileus', 42) == (
            'taxon_abc:Pileus:42'
        )

    def test_distinct_offsets_distinct_ids(self) -> None:
        """Two pileus mentions in the same treatment get distinct
        IDs — critical for storing both without collision."""
        a = annotation_doc_id('taxon_x', 'Pileus', 0)
        b = annotation_doc_id('taxon_x', 'Pileus', 100)
        assert a != b

    def test_same_offset_same_id_for_overwrite_idempotency(self) -> None:
        """Re-running the annotator with an updated prompt should
        overwrite the previous annotation at the same offset rather
        than appending — covered by stable _id derivation."""
        first = annotation_doc_id('taxon_x', 'Pileus', 42)
        second = annotation_doc_id('taxon_x', 'Pileus', 42)
        assert first == second
