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
        anns, _ = parse_claude_response(
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
        anns, _ = parse_claude_response(
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
        anns, _ = parse_claude_response(
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
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert anns[0]['feature_label'] == 'Hymenophore'

    def test_substring_span_recovers_partial_range(self) -> None:
        span_map = self._setup()
        response = json.dumps({
            'spans': [{'text': 'brown', 'feature_label': 'Pileus'}],
        })
        anns, _ = parse_claude_response(
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
        anns, _ = parse_claude_response(
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
        anns, _ = parse_claude_response(
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
        anns, _ = parse_claude_response(
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
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert anns[0]['feature_label'] == 'Pileus'

    def test_feature_label_sanitized_to_brat_safe(self) -> None:
        """Labels with parens / commas / periods are sanitized at
        parse time so the candidate DB has brat-safe labels from
        the start.  Otherwise the export → review → ingest
        round-trip would diff every paren'd label as delete+add
        (candidate has parens, reviewer's brat-saved form does
        not).  Real case from the 2026-06-29 live brat test."""
        span_map = self._setup()
        response = json.dumps({
            'spans': [{
                'text': 'brown',
                'feature_label': 'Partial veil (microscopic)',
            }],
        })
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        # No parens or other special chars in the stored label.
        assert anns[0]['feature_label'] == 'Partial veil microscopic'

    def test_feature_label_all_special_chars_drops_span(
        self,
    ) -> None:
        """Defensive: a label that's entirely special chars
        sanitizes to empty.  Better to raise (envelope-level)
        than store a blank-labelled annotation."""
        span_map = self._setup()
        with pytest.raises(ClaudeResponseError) as exc:
            parse_claude_response(
                '{"spans": [{"text": "brown", "feature_label": "(.)"}]}',
                span_map, 'm', 'tid', 'did', 'ts',
            )
        assert 'sanitized to empty' in str(exc.value)

    def test_text_not_found_in_synth_drops_span(self) -> None:
        """Per-span isolation: Claude hallucinated text that's not
        in the synthetic doc.  The span is dropped (recorded in
        dropped_spans for offline recovery) rather than raising —
        other valid spans in the same response still get stored."""
        span_map = self._setup()
        anns, dropped = parse_claude_response(
            ('{"spans": [{"text": "Lamellae cream-colored.", '
             '"feature_label": "Lamellae"}]}'),
            span_map, 'm', 'tid', 'did', 'ts',
        )
        assert anns == []
        assert len(dropped) == 1
        assert dropped[0]['feature_label'] == 'Lamellae'
        assert dropped[0]['claude_text'] == 'Lamellae cream-colored.'
        assert 'not found' in dropped[0]['reason']

    def test_text_not_found_does_not_kill_sibling_spans(
        self,
    ) -> None:
        """The whole point of per-span isolation: one bad span in
        a response does not block the others from being stored."""
        span_map = self._setup()
        # Two spans: first hallucinated, second valid.
        response = json.dumps({
            'spans': [
                {
                    'text': 'Lamellae cream-colored.',
                    'feature_label': 'Lamellae',
                },
                {
                    'text': 'Pileus brown 3 cm wide.',
                    'feature_label': 'Pileus',
                },
            ],
        })
        anns, dropped = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1
        assert anns[0]['feature_label'] == 'Pileus'
        assert len(dropped) == 1
        assert dropped[0]['feature_label'] == 'Lamellae'

    def test_span_crossing_field_boundary_drops_span(self) -> None:
        """An annotation that overlaps the synthetic-doc gap between
        description and diagnosis is dropped rather than raising.
        The reviewer can resolve manually if it represents a real
        anatomical mention."""
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
        anns, dropped = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert anns == []
        assert len(dropped) == 1
        assert 'field boundary' in dropped[0]['reason']

    # ------------------------------------------------------------------
    # Whitespace-tolerant fallback search.
    #
    # LLMs routinely normalize unicode whitespace (narrow no-break
    # space U+202F, non-breaking space U+00A0, thin space U+2009,
    # newlines collapsed to spaces) when echoing source text in JSON
    # output.  Exact str.find then fails on visually-identical text.
    # The fuzzy fallback rebuilds a regex where any whitespace run
    # matches any non-empty whitespace run in the source.
    # ------------------------------------------------------------------

    def test_narrow_no_break_space_matches_plain_space(self) -> None:
        """The Persoonia / Fungal Planet case: source uses U+202F
        for unit spacing ('av. = 98'); Claude's echo collapses
        U+202F to U+0020.  Real bug observed on Calonectria
        pentaseptata 2026-06-28.

        Source uses explicit \\u202f escapes so the test's bytes
        are unambiguous (some editors silently insert U+202F when
        you type spaces around symbols).
        """
        source = 'av.\u202f=\u202f98 mm, smooth.'
        treatment = _make_treatment(
            description=source,
            description_spans=[
                {'start_char': 0, 'end_char': len(source)},
            ],
        )
        _, span_map = render(treatment)
        # Claude's echo: U+202F → U+0020.
        response = json.dumps({
            'spans': [{
                'text': 'av. = 98 mm, smooth.',
                'feature_label': 'Spores',
            }],
        })
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1
        # source_text MUST be the source's verbatim bytes (U+202F
        # preserved), NOT Claude's normalized echo.  Downstream brat
        # rendering uses source_text against the actual plaintext
        # attachment, so source bytes must win.
        assert '\u202f' in anns[0]['source_text']
        assert anns[0]['source_text'] == source

    def test_non_breaking_space_matches_plain_space(self) -> None:
        """U+00A0 (NBSP) is common in journals that copy-paste from
        Word.  Same normalization pattern as U+202F."""
        source = 'Pileus\u00a03\u00a0cm wide.'
        treatment = _make_treatment(
            description=source,
            description_spans=[
                {'start_char': 0, 'end_char': len(source)},
            ],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [{
                'text': 'Pileus 3 cm wide.',
                'feature_label': 'Pileus',
            }],
        })
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1
        assert '\u00a0' in anns[0]['source_text']
        assert anns[0]['source_text'] == source

    def test_newline_in_source_matches_space_in_claude(self) -> None:
        """Multi-line descriptions: source has line breaks where
        Claude's echo collapses to a single space."""
        source = 'Pileus brown,\n3 cm wide,\nsmooth.'
        treatment = _make_treatment(
            description=source,
            description_spans=[
                {'start_char': 0, 'end_char': len(source)},
            ],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [{
                'text': 'Pileus brown, 3 cm wide, smooth.',
                'feature_label': 'Pileus',
            }],
        })
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1
        # Original source preserved — newlines and all.
        assert '\n' in anns[0]['source_text']
        assert anns[0]['source_text'] == source

    def test_multiple_whitespace_chars_collapse_to_one(self) -> None:
        """Source has double spaces; Claude returns single.  Should
        still match."""
        treatment = _make_treatment(
            description='Pileus  brown  3  cm.',
            description_spans=[{'start_char': 0, 'end_char': 21}],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [{
                'text': 'Pileus brown 3 cm.',
                'feature_label': 'Pileus',
            }],
        })
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 1

    def test_fuzzy_does_not_trigger_when_exact_matches(self) -> None:
        """Regression: when exact str.find succeeds, the fuzzy path
        must NOT run.  The fuzzy regex is more permissive and could
        match a different (incorrect) span elsewhere in a treatment
        that has whitespace variation across paragraphs."""
        # Source has 'Pileus brown' at offset 0, and a later
        # 'Pileus  brown' (double space) at offset 30.  Claude
        # returns 'Pileus brown'.  Exact-find should match at 0.
        # If fuzzy ran first or in addition, it could match at 30
        # (since '\\s+' matches one or many).
        treatment = _make_treatment(
            description=(
                'Pileus brown 3 cm wide.\n\n'
                'Pileus  brown 5 cm wide.'  # double-space variant
            ),
            description_spans=[{'start_char': 0, 'end_char': 49}],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [{
                'text': 'Pileus brown',
                'feature_label': 'Pileus',
            }],
        })
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        # Must hit the EXACT match at start, not the fuzzy match
        # at offset 25 (after '\n\n').
        assert anns[0]['start'] == 0
        assert anns[0]['end'] == 12

    def test_fuzzy_still_drops_on_non_whitespace_differences(
        self,
    ) -> None:
        """The fallback is whitespace-tolerant ONLY.  Word-level
        paraphrases / hallucinations land in dropped_spans (NOT
        raised) so sibling spans survive.  The drop reason names
        both failure modes (exact + fuzzy) so the operator and
        the offline-recovery script can distinguish whitespace
        normalization from genuine hallucination."""
        span_map = self._setup()
        anns, dropped = parse_claude_response(
            ('{"spans": [{"text": "Pileus red 3 cm wide.", '
             '"feature_label": "Pileus"}]}'),
            span_map, 'm', 'tid', 'did', 'ts',
        )
        assert anns == []
        assert len(dropped) == 1
        assert 'whitespace-tolerant' in dropped[0]['reason']

    def test_fuzzy_match_offsets_advance_cursor(self) -> None:
        """After a fuzzy match, the cursor should advance to the END
        of the matched span in the source (NOT cursor + len(wanted),
        which would be wrong when the source spans were longer due
        to extra whitespace)."""
        treatment = _make_treatment(
            description=(
                'av. = 98 µm.  Another av. = 50 µm later.'
            ),
            description_spans=[{'start_char': 0, 'end_char': 41}],
        )
        _, span_map = render(treatment)
        response = json.dumps({
            'spans': [
                {  # fuzzy match; source uses U+202F
                    'text': 'av. = 98 µm.',
                    'feature_label': 'First',
                },
                {  # exact match; should land AFTER the first
                    'text': 'av. = 50 µm later.',
                    'feature_label': 'Second',
                },
            ],
        })
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        assert len(anns) == 2
        # Second annotation must start AFTER first ends; if cursor
        # advancement was buggy the second find would re-hit the
        # first 'av. =' inside the first span.
        assert anns[1]['start'] > anns[0]['end']


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


# ---------------------------------------------------------------------------
# Growth-condition context
# ---------------------------------------------------------------------------


class TestGrowthConditionContext:
    """`Colony on MEA` carries two facts in one string: the feature
    and the medium it was observed on.

    docs/feature_label_non_synonyms.md refuses to collapse that family
    -- "the medium is the entire point of the observation" -- and
    names the fix: "a separate `context` field, not a longer label".
    This is that field.  It is **additive**: `feature_label` keys both
    the candidate doc and the hand doc
    (`<treatment_id>:<feature_label>:<start>`), so rewriting it here
    would re-key every affected annotation and read as delete+add
    against every existing export.
    """

    def _span_map(self, description: str):
        treatment = _make_treatment(
            description=description,
            description_spans=[
                {'start_char': 0, 'end_char': len(description)},
            ],
        )
        _, span_map = render(treatment)
        return span_map

    def _parse(self, description: str, label: str):
        span_map = self._span_map(description)
        response = json.dumps({
            'spans': [{'text': description, 'feature_label': label}],
        })
        anns, _ = parse_claude_response(
            response, span_map, 'm', 'tid', 'did', 'ts',
        )
        return anns

    def test_medium_is_split_into_its_own_field(self) -> None:
        anns = self._parse(
            'Colonies on MEA reaching 40 mm diam in 7 days.',
            'Colony on MEA',
        )
        assert anns[0]['context'] == 'MEA'

    def test_in_culture_is_a_condition_too(self) -> None:
        anns = self._parse(
            'Conidia in culture narrower than on the host.',
            'Conidia in culture',
        )
        assert anns[0]['context'] == 'culture'

    def test_the_label_itself_is_not_rewritten(self) -> None:
        """Identity does not move.  The decomposition is available to
        consumers through treatments_to_structured.feature_label_rules
        when they want the base form."""
        anns = self._parse(
            'Colonies on MEA reaching 40 mm diam in 7 days.',
            'Colony on MEA',
        )
        assert anns[0]['feature_label'] == 'Colony on MEA'

    def test_labels_without_a_condition_omit_the_key(self) -> None:
        """Keys are omitted rather than set to None when nothing is
        known -- the convention brat_ingest already uses for round
        provenance."""
        anns = self._parse('Pileus brown 3 cm wide.', 'Pileus')
        assert 'context' not in anns[0]
