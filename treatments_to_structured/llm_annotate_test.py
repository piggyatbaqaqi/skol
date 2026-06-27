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


# Minimal seed used by the prompt-construction tests.  Real seeds
# live in treatments_to_structured/seeds/*.json; this fixture
# captures just the shape.
_SEED_SAMPLE: Dict[str, Any] = {
    'name': 'fungi',
    'description': (
        "Seed examples of mycological anatomical features.  "
        "Not exhaustive — invent canonical names for features "
        "not in this list."
    ),
    'examples': [
        {'name': 'Pileus', 'description': 'The cap.'},
        {'name': 'Stipe', 'description': 'The stem.'},
        {'name': 'Basidia', 'description': 'Spore-bearing cells.'},
    ],
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
        prompt = build_user_prompt(synth, _SEED_SAMPLE)
        assert 'Pileus brown 3 cm wide.' in prompt
        assert '=== description ===' in prompt

    def test_includes_seed_examples_as_label_list(self) -> None:
        """Each seed example shows up in the prompt by name — they
        ARE the example label vocabulary."""
        prompt = build_user_prompt('text', _SEED_SAMPLE)
        assert 'Pileus' in prompt
        assert 'Stipe' in prompt
        assert 'Basidia' in prompt

    def test_includes_seed_description_for_boundary_discipline(
        self,
    ) -> None:
        """The seed's description carries anatomical-boundary
        guidance (what to treat as a feature; what NOT to)."""
        prompt = build_user_prompt('text', _SEED_SAMPLE)
        assert 'Not exhaustive' in prompt

    def test_specifies_open_ended_label_rule(self) -> None:
        """The prompt must instruct Claude that labels NOT in the
        seed are acceptable — that's the test we're going to verify
        live with Hymenophore."""
        prompt = build_user_prompt('text', _SEED_SAMPLE)
        # The exact wording can drift; pin a clear keyword that
        # represents the rule: invent a canonical name.
        assert 'invent' in prompt.lower()

    def test_specifies_output_envelope(self) -> None:
        prompt = build_user_prompt('text', _SEED_SAMPLE)
        # Both fields per span must be specified.
        assert '"spans"' in prompt
        assert '"text"' in prompt
        assert '"feature_label"' in prompt

    def test_works_for_arbitrary_seed(self) -> None:
        """Swapping seeds (e.g., for non-fungal kingdoms) is the
        whole point of the seed-based design."""
        plant_seed = {
            'name': 'plants',
            'description': 'Plant features.',
            'examples': [
                {'name': 'Lamina', 'description': 'Leaf blade.'},
                {'name': 'Petiole', 'description': 'Leaf stalk.'},
            ],
        }
        prompt = build_user_prompt('text', plant_seed)
        assert 'Lamina' in prompt
        assert 'Petiole' in prompt
        # No fungal labels should leak in.
        assert 'Basidia' not in prompt


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
            'spans': [{
                'text': 'Pileus brown 3 cm wide.',
                'feature_label': 'Pileus',
            }],
        })
        anns = parse_claude_response(
            response, span_map, 'claude-opus-4-7',
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
        """Claude reporting 'no features mentioned' is a legitimate
        signal."""
        span_map = self._setup()
        response = json.dumps({'spans': []})
        anns = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert anns == []

    def test_multiple_features_in_one_response(self) -> None:
        """The whole point of the multi-feature bootstrap: ONE
        Claude call returns spans tagged with DIFFERENT labels."""
        treatment = _make_treatment(
            description='Pileus brown.  Stipe long.',
            description_spans=[{'start_char': 0, 'end_char': 26}],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [
                {'text': 'Pileus brown.', 'feature_label': 'Pileus'},
                {'text': 'Stipe long.', 'feature_label': 'Stipe'},
            ],
        })
        anns = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert [a['feature_label'] for a in anns] == ['Pileus', 'Stipe']

    def test_invented_label_outside_seed_accepted(self) -> None:
        """The prompt's open-ended rule lets Claude invent a label
        not in the seed.  Parse must accept any non-empty string —
        validation/canonicalization happens during review, not
        here."""
        treatment = _make_treatment(
            description='Hymenophore poroid, depressed around apex.',
            description_spans=[{'start_char': 0, 'end_char': 42}],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [{
                'text': 'Hymenophore poroid, depressed around apex.',
                'feature_label': 'Hymenophore',
            }],
        })
        anns = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert anns[0]['feature_label'] == 'Hymenophore'

    def test_substring_span_recovers_partial_range(self) -> None:
        span_map = self._setup()
        response = json.dumps({
            'spans': [{'text': 'brown', 'feature_label': 'Pileus'}],
        })
        anns = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        # 'brown' is at field-relative offset 7-12
        assert anns[0]['start'] == 7
        assert anns[0]['end'] == 12
        assert anns[0]['source_spans'] == [{'start': 107, 'end': 112}]

    def test_multiple_spans_advance_cursor_left_to_right(self) -> None:
        treatment = _make_treatment(
            description='Pileus brown.  Pileus also wide.',
            description_spans=[{'start_char': 100, 'end_char': 132}],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [
                {'text': 'Pileus brown.', 'feature_label': 'Pileus'},
                {'text': 'Pileus also wide.', 'feature_label': 'Pileus'},
            ],
        })
        anns = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
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
            '{"spans": [{"text": "brown", "feature_label": "Pileus"}]}\n'
            '```'
        )
        anns = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1
        assert anns[0]['source_text'] == 'brown'

    def test_response_with_bare_fences_tolerated(self) -> None:
        span_map = self._setup()
        response = (
            '```\n'
            '{"spans": [{"text": "brown", "feature_label": "Pileus"}]}\n'
            '```'
        )
        anns = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1

    def test_invalid_json_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                'not json at all', span_map, 'm',
                'tid', 'did', 'ts',
            )
        assert 'JSON' in str(exc.value)

    def test_missing_spans_key_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"annotations": []}', span_map, 'm',
                'tid', 'did', 'ts',
            )
        assert "'spans'" in str(exc.value)

    def test_spans_not_a_list_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"spans": "Pileus brown."}', span_map, 'm',
                'tid', 'did', 'ts',
            )
        assert 'list' in str(exc.value).lower()

    def test_span_missing_text_key_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"spans": [{"feature_label": "Pileus"}]}',
                span_map, 'm', 'tid', 'did', 'ts',
            )
        assert "'text'" in str(exc.value)

    def test_span_missing_feature_label_raises(self) -> None:
        """In the multi-feature design, feature_label is per-span;
        omitting it is a contract violation."""
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"spans": [{"text": "brown"}]}',
                span_map, 'm', 'tid', 'did', 'ts',
            )
        assert "'feature_label'" in str(exc.value)

    def test_empty_text_string_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError):
            parse_claude_response(
                '{"spans": [{"text": "", "feature_label": "Pileus"}]}',
                span_map, 'm', 'tid', 'did', 'ts',
            )

    def test_empty_feature_label_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError):
            parse_claude_response(
                '{"spans": [{"text": "brown", "feature_label": ""}]}',
                span_map, 'm', 'tid', 'did', 'ts',
            )

    def test_whitespace_only_feature_label_raises(self) -> None:
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError):
            parse_claude_response(
                '{"spans": [{"text": "brown", "feature_label": "   "}]}',
                span_map, 'm', 'tid', 'did', 'ts',
            )

    def test_feature_label_stripped(self) -> None:
        """Leading/trailing whitespace on feature_label is trimmed
        — LLMs sometimes emit ' Pileus ' with stray padding."""
        span_map = self._setup()
        response = json.dumps({
            'spans': [{
                'text': 'brown',
                'feature_label': '  Pileus  ',
            }],
        })
        anns = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert anns[0]['feature_label'] == 'Pileus'

    def test_text_not_found_in_synth_raises(self) -> None:
        """Claude hallucinated text that's not in the synthetic doc.
        Better to fail than silently invent offsets."""
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                ('{"spans": [{"text": "Lamellae cream-colored.", '
                 '"feature_label": "Lamellae"}]}'),
                span_map, 'm', 'tid', 'did', 'ts',
            )
        assert 'not found' in str(exc.value)

    def test_span_crossing_field_boundary_raises(self) -> None:
        """An annotation that overlaps the synthetic-doc gap between
        description and diagnosis isn't a valid annotation — fail
        loudly so the operator can split it manually."""
        treatment = _make_treatment(
            description='Pileus brown.',
            description_spans=[{'start_char': 100, 'end_char': 113}],
            diagnosis='Pileus differs.',
            diagnosis_spans=[{'start_char': 200, 'end_char': 215}],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [{
                'text': 'brown.\n\n=== diagnosis ===\n\nPileus',
                'feature_label': 'Pileus',
            }],
        })
        with pytest.raises(ClaudeResponseError):
            parse_claude_response(
                response, span_map, 'm', 'tid', 'did', 'ts',
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
        """Two Pileus mentions in the same treatment get distinct
        IDs — critical for storing both without collision."""
        a = annotation_doc_id('taxon_x', 'Pileus', 0)
        b = annotation_doc_id('taxon_x', 'Pileus', 100)
        assert a != b

    def test_different_features_distinct_ids(self) -> None:
        """The bootstrap pass now emits multiple feature labels per
        treatment; two annotations at the same offset (impossible
        in practice, but the id space allows for it) under different
        labels must not collide."""
        a = annotation_doc_id('taxon_x', 'Pileus', 50)
        b = annotation_doc_id('taxon_x', 'Stipe', 50)
        assert a != b

    def test_same_offset_same_id_for_overwrite_idempotency(self) -> None:
        """Re-running the annotator with an updated prompt should
        overwrite the previous annotation at the same offset rather
        than appending — covered by stable _id derivation."""
        first = annotation_doc_id('taxon_x', 'Pileus', 42)
        second = annotation_doc_id('taxon_x', 'Pileus', 42)
        assert first == second
