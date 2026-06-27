"""Tests for bin/llm_annotate_features CLI helpers.

The pure parse / prompt logic is tested in
treatments_to_structured/llm_annotate_test.py.  This file covers the
CLI-side glue: schema loading, candidate-DB resolution (including the
4.5 fallback), treatment-ID input from stdin / --doc-id, the
skip-existing filter, and the annotate-one-treatment loop with a
mock Anthropic client.
"""

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
    annotate_one_treatment,
    estimate_tokens,
    filter_already_annotated,
    load_schema,
    read_treatment_ids,
    resolve_candidate_db_name,
)


# ---------------------------------------------------------------------------
# load_schema
# ---------------------------------------------------------------------------


class TestLoadSchema:
    """Resolves schemas/<NAME>.json relative to the package."""

    def test_loads_pileus_schema(self) -> None:
        schema = load_schema('pileus')
        assert schema['title'] == 'Pileus'
        assert 'properties' in schema

    def test_missing_schema_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_schema('this_feature_does_not_exist_anywhere')


# ---------------------------------------------------------------------------
# resolve_candidate_db_name
# ---------------------------------------------------------------------------


class TestResolveCandidateDbName:
    """Phase 1 deliverable 4.5 + its fallback for unmigrated docs."""

    def test_uses_databases_features_candidate_when_set(self) -> None:
        exp = {
            'databases': {
                'features_candidate': 'skol_exp_X_features_candidate',
            },
        }
        warn = io.StringIO()
        name = resolve_candidate_db_name(
            'production_v4', exp, verbosity=1, warn_stream=warn,
        )
        assert name == 'skol_exp_X_features_candidate'
        # No warning when the field is set
        assert warn.getvalue() == ''

    def test_falls_back_to_naming_convention_when_unset(self) -> None:
        exp = {'databases': {}}
        warn = io.StringIO()
        name = resolve_candidate_db_name(
            'production_v4', exp, verbosity=1, warn_stream=warn,
        )
        assert name == 'skol_exp_production_v4_features_candidate'
        assert 'NOTE' in warn.getvalue()
        assert 'production_v4' in warn.getvalue()

    def test_fallback_handles_missing_databases_block(self) -> None:
        """Some old experiment docs don't even have a `databases`
        key.  The fallback should still produce a sensible name."""
        warn = io.StringIO()
        name = resolve_candidate_db_name(
            'legacy', {}, verbosity=0, warn_stream=warn,
        )
        assert name == 'skol_exp_legacy_features_candidate'

    def test_silent_at_verbosity_zero(self) -> None:
        warn = io.StringIO()
        resolve_candidate_db_name(
            'v', {'databases': {}}, verbosity=0, warn_stream=warn,
        )
        assert warn.getvalue() == ''


# ---------------------------------------------------------------------------
# read_treatment_ids
# ---------------------------------------------------------------------------


class TestReadTreatmentIds:
    """--doc-id (already parsed to list by common_parser) wins over
    stdin; stdin is only consumed when not a TTY."""

    def test_doc_ids_list_passed_through(self) -> None:
        ids = read_treatment_ids(
            ['a', 'b', 'c'], io.StringIO(''), stdin_isatty=False,
        )
        assert ids == ['a', 'b', 'c']

    def test_doc_ids_filters_empty_entries_defensively(self) -> None:
        """common_parser strips already, but be defensive."""
        ids = read_treatment_ids(
            ['a', '', '  ', 'b'], io.StringIO(''), stdin_isatty=False,
        )
        assert ids == ['a', 'b']

    def test_doc_ids_all_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            read_treatment_ids(
                ['', '  '], io.StringIO(''), stdin_isatty=False,
            )

    def test_stdin_one_id_per_line(self) -> None:
        stdin = io.StringIO('taxon_a\ntaxon_b\ntaxon_c\n')
        ids = read_treatment_ids(None, stdin, stdin_isatty=False)
        assert ids == ['taxon_a', 'taxon_b', 'taxon_c']

    def test_stdin_blank_lines_skipped(self) -> None:
        stdin = io.StringIO('taxon_a\n\n  \ntaxon_b\n')
        ids = read_treatment_ids(None, stdin, stdin_isatty=False)
        assert ids == ['taxon_a', 'taxon_b']

    def test_empty_stdin_raises(self) -> None:
        with pytest.raises(ValueError):
            read_treatment_ids(None, io.StringIO(''), stdin_isatty=False)

    def test_no_input_no_stdin_when_tty_raises(self) -> None:
        """Don't block waiting for typed-in IDs from an interactive
        shell — operator wants the error message."""
        with pytest.raises(ValueError) as exc:
            read_treatment_ids(None, io.StringIO(''), stdin_isatty=True)
        assert 'no treatment IDs' in str(exc.value)

    def test_string_input_raises_type_error(self) -> None:
        """Defensive guard against the bug from the first live run:
        passing the raw ``args.doc_ids`` string instead of the parsed
        ``config['doc_ids']`` list silently iterated characters and
        spammed 'skipping <single char>: not found' lines.  TypeError
        now surfaces the bug immediately with the actionable fix in
        the message."""
        with pytest.raises(TypeError) as exc:
            read_treatment_ids(
                'taxon_a,taxon_b', io.StringIO(''), stdin_isatty=False,
            )
        assert 'doc_ids' in str(exc.value)
        assert 'list' in str(exc.value)


# ---------------------------------------------------------------------------
# filter_already_annotated
# ---------------------------------------------------------------------------


class _FakeView:
    """Mimics couchdb.Database.view() return shape."""

    def __init__(self, rows: List[Any]) -> None:
        self.rows = rows


class _FakeCandidateDb:
    """Stand-in for the candidate annotations DB."""

    def __init__(
        self, existing_prefixes: List[str] = (),
    ) -> None:
        self.existing_prefixes = set(existing_prefixes)

    def view(self, _name: str, **kwargs: Any) -> _FakeView:
        startkey = kwargs['startkey']
        # Any prefix in existing_prefixes that's a substring of
        # startkey counts as "this annotation exists".
        if any(p == startkey for p in self.existing_prefixes):
            return _FakeView(rows=[{'id': 'fake', 'key': startkey}])
        return _FakeView(rows=[])


class TestFilterAlreadyAnnotated:
    """Drop treatment IDs that already have annotations in the
    candidate DB for the same feature."""

    def test_keeps_unannotated_only(self) -> None:
        db = _FakeCandidateDb(existing_prefixes=[
            'taxon_a:Pileus:',
        ])
        result = filter_already_annotated(
            ['taxon_a', 'taxon_b'], db, 'Pileus',
        )
        assert result == ['taxon_b']

    def test_different_feature_label_does_not_collide(self) -> None:
        """An existing 'Lamellae' annotation for taxon_a doesn't
        prevent us from annotating taxon_a for 'Pileus'."""
        db = _FakeCandidateDb(existing_prefixes=[
            'taxon_a:Lamellae:',
        ])
        result = filter_already_annotated(
            ['taxon_a'], db, 'Pileus',
        )
        assert result == ['taxon_a']

    def test_empty_input(self) -> None:
        db = _FakeCandidateDb()
        assert filter_already_annotated([], db, 'Pileus') == []


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def _make_mock_count_tokens_client(tokens_per_prompt: int) -> Any:
    """A MagicMock that responds to messages.count_tokens with a
    fixed input-token count per call."""
    client = MagicMock()
    client.messages.count_tokens.return_value = MagicMock(
        input_tokens=tokens_per_prompt,
    )
    return client


class TestEstimateTokens:
    """Sum input tokens via count_tokens; estimate output + cost."""

    def test_single_prompt(self) -> None:
        client = _make_mock_count_tokens_client(1000)
        stats = estimate_tokens(
            client, [('tid_a', 'prompt text')], 'claude-opus-4-7',
        )
        assert stats['doc_count'] == 1
        assert stats['total_input_tokens'] == 1000
        assert stats['est_output_tokens'] == 250

    def test_three_prompts_sum_tokens(self) -> None:
        client = _make_mock_count_tokens_client(500)
        stats = estimate_tokens(
            client,
            [('a', 'p'), ('b', 'p'), ('c', 'p')],
            'claude-opus-4-7',
        )
        assert stats['total_input_tokens'] == 1500
        assert stats['est_output_tokens'] == 375

    def test_cost_calculated_from_pricing_table(self) -> None:
        client = _make_mock_count_tokens_client(1_000_000)
        stats = estimate_tokens(
            client, [('a', 'p')], 'claude-opus-4-7',
        )
        # 1M input tokens × $15.00/1M = $15.00 input cost
        # 250k output tokens × $75.00/1M = $18.75 output cost
        assert stats['est_input_cost_usd'] == 15.00
        assert stats['est_output_cost_usd'] == 18.75
        assert stats['est_total_cost_usd'] == 33.75

    def test_unknown_model_uses_sonnet_pricing(self) -> None:
        """Defensive fallback so a typo in --llm-model doesn't crash
        the estimate; falls back to opus-grade pricing as a
        conservative upper bound (better to over-estimate)."""
        client = _make_mock_count_tokens_client(1_000_000)
        stats = estimate_tokens(
            client, [('a', 'p')], 'claude-future-99-99',
        )
        assert stats['est_total_cost_usd'] > 0


# ---------------------------------------------------------------------------
# annotate_one_treatment
# ---------------------------------------------------------------------------


def _make_treatment(
    description: str = 'Pileus brown 3 cm.',
    description_spans: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        '_id': 'taxon_test',
        'description': description,
        'description_spans': description_spans or [
            {'start_char': 100, 'end_char': 100 + len(description)},
        ],
        'diagnosis': None,
        'diagnosis_spans': [],
        'ingest': {'_id': 'src_doc_xyz'},
    }


def _make_mock_messages_client(claude_response_text: str) -> Any:
    """A MagicMock client whose messages.create returns a single-block
    response carrying the given text."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=claude_response_text)]
    client.messages.create.return_value = response
    return client


_PILEUS_SCHEMA = {
    'title': 'Pileus',
    'description': 'The cap.',
    'type': 'object',
    'properties': {},
}


class TestAnnotateOneTreatment:
    """End-to-end with a mocked Anthropic client."""

    def test_happy_path_returns_annotations(self) -> None:
        treatment = _make_treatment()
        claude_response = json.dumps({
            'spans': [{'text': 'Pileus brown 3 cm.'}],
        })
        client = _make_mock_messages_client(claude_response)
        result = annotate_one_treatment(
            client, treatment, _PILEUS_SCHEMA, 'Pileus',
            'claude-opus-4-7',
        )
        assert isinstance(result, list)
        assert len(result) == 1
        ann = result[0]
        assert ann['feature_label'] == 'Pileus'
        assert ann['field'] == 'description'
        assert ann['treatment_id'] == 'taxon_test'
        assert ann['doc_id'] == 'src_doc_xyz'
        assert ann['model'] == 'claude-opus-4-7'

    def test_empty_treatment_returns_empty_list(self) -> None:
        """A treatment with neither description nor diagnosis renders
        to an empty synth doc; annotate skips the API call entirely."""
        empty = {
            '_id': 'taxon_empty',
            'description': None,
            'description_spans': [],
            'diagnosis': None,
            'diagnosis_spans': [],
        }
        client = _make_mock_messages_client('')
        result = annotate_one_treatment(
            client, empty, _PILEUS_SCHEMA, 'Pileus',
            'claude-opus-4-7',
        )
        assert result == []
        # No API call should have been made.
        client.messages.create.assert_not_called()

    def test_no_spans_returned(self) -> None:
        """Claude says 'no pileus mentions' — that's a legitimate
        outcome, not an error."""
        treatment = _make_treatment(
            description='No anatomy here, just metadata.',
        )
        claude_response = json.dumps({'spans': []})
        client = _make_mock_messages_client(claude_response)
        result = annotate_one_treatment(
            client, treatment, _PILEUS_SCHEMA, 'Pileus',
            'claude-opus-4-7',
        )
        assert result == []

    def test_invalid_response_returns_exception(self) -> None:
        """Bad JSON from Claude → exception returned (NOT raised),
        so the parallel worker pool keeps its other futures alive."""
        treatment = _make_treatment()
        client = _make_mock_messages_client('not valid json')
        result = annotate_one_treatment(
            client, treatment, _PILEUS_SCHEMA, 'Pileus',
            'claude-opus-4-7',
        )
        assert isinstance(result, Exception)
