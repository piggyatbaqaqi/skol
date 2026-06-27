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
    load_seed,
    read_treatment_ids,
    resolve_candidate_db_name,
)


# ---------------------------------------------------------------------------
# load_seed
# ---------------------------------------------------------------------------


class TestLoadSeed:
    """Resolves seeds/<NAME>.json relative to the package."""

    def test_loads_fungi_seed(self) -> None:
        seed = load_seed('fungi')
        assert seed['name'] == 'fungi'
        assert 'examples' in seed
        assert any(
            ex.get('name') == 'Pileus' for ex in seed['examples']
        )

    def test_fungi_seed_intentionally_omits_hymenophore(self) -> None:
        """Phase-1 deliberate test point: Hymenophore is left out of
        the seed so we can verify Claude invents the label for the
        Aureoboletus pores/tubes block.  If a future commit adds
        Hymenophore, update or drop this test consciously."""
        seed = load_seed('fungi')
        names = {ex['name'] for ex in seed['examples']}
        assert 'Hymenophore' not in names

    def test_missing_seed_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_seed('this_kingdom_does_not_exist_anywhere')


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
        # 02_50 sorts between the 02_00 treatments_prose extraction
        # and the 03_00 treatments_structured SLM output.
        assert name == (
            'skol_exp_production_v4_02_50_features_candidate'
        )
        assert 'NOTE' in warn.getvalue()
        assert 'production_v4' in warn.getvalue()

    def test_fallback_handles_missing_databases_block(self) -> None:
        """Some old experiment docs don't even have a `databases`
        key.  The fallback should still produce a sensible name."""
        warn = io.StringIO()
        name = resolve_candidate_db_name(
            'legacy', {}, verbosity=0, warn_stream=warn,
        )
        assert name == 'skol_exp_legacy_02_50_features_candidate'

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
    """Drop treatment IDs that already have ANY annotation in the
    candidate DB.  The multi-feature bootstrap annotates the whole
    treatment in one call, so per-feature scoping no longer makes
    sense — re-running on an annotated treatment would just
    duplicate work."""

    def test_keeps_unannotated_only(self) -> None:
        db = _FakeCandidateDb(existing_prefixes=[
            'taxon_a:',
        ])
        result = filter_already_annotated(
            ['taxon_a', 'taxon_b'], db,
        )
        assert result == ['taxon_b']

    def test_annotated_under_any_label_counts_as_done(self) -> None:
        """Existing annotations (any feature label) on a treatment
        mean we don't re-run.  The bootstrap pass writes ALL
        feature labels per treatment in one go; a prior run already
        produced them."""
        db = _FakeCandidateDb(existing_prefixes=[
            'taxon_a:',  # any prior annotation, regardless of feature
        ])
        result = filter_already_annotated(['taxon_a'], db)
        assert result == []

    def test_empty_input(self) -> None:
        db = _FakeCandidateDb()
        assert filter_already_annotated([], db) == []


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


_TEST_SEED = {
    'name': 'fungi',
    'description': "Test seed: not exhaustive.",
    'examples': [
        {'name': 'Pileus', 'description': 'The cap.'},
        {'name': 'Stipe', 'description': 'The stem.'},
    ],
}


class TestAnnotateOneTreatment:
    """End-to-end with a mocked Anthropic client."""

    def test_happy_path_returns_annotations(self) -> None:
        treatment = _make_treatment()
        claude_response = json.dumps({
            'spans': [{
                'text': 'Pileus brown 3 cm.',
                'feature_label': 'Pileus',
            }],
        })
        client = _make_mock_messages_client(claude_response)
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert isinstance(result, list)
        assert len(result) == 1
        ann = result[0]
        assert ann['feature_label'] == 'Pileus'
        assert ann['field'] == 'description'
        assert ann['treatment_id'] == 'taxon_test'
        assert ann['doc_id'] == 'src_doc_xyz'
        assert ann['model'] == 'claude-opus-4-7'

    def test_multiple_features_in_one_call(self) -> None:
        """The whole point of the pivot: one API call returns
        spans for multiple distinct feature labels."""
        treatment = _make_treatment(
            description='Pileus brown 3 cm.  Stipe 5 cm long.',
        )
        claude_response = json.dumps({
            'spans': [
                {
                    'text': 'Pileus brown 3 cm.',
                    'feature_label': 'Pileus',
                },
                {
                    'text': 'Stipe 5 cm long.',
                    'feature_label': 'Stipe',
                },
            ],
        })
        client = _make_mock_messages_client(claude_response)
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        labels = [a['feature_label'] for a in result]
        assert labels == ['Pileus', 'Stipe']

    def test_invented_label_propagates(self) -> None:
        """If Claude returns a label not in the seed (e.g.
        Hymenophore, the deliberate test point on the live run),
        annotate_one_treatment passes it through unchanged."""
        treatment = _make_treatment(
            description='Hymenophore poroid, depressed around apex.',
        )
        claude_response = json.dumps({
            'spans': [{
                'text': 'Hymenophore poroid, depressed around apex.',
                'feature_label': 'Hymenophore',
            }],
        })
        client = _make_mock_messages_client(claude_response)
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result[0]['feature_label'] == 'Hymenophore'

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
            client, empty, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result == []
        # No API call should have been made.
        client.messages.create.assert_not_called()

    def test_no_spans_returned(self) -> None:
        """Claude says 'no anatomical features mentioned' — that's
        a legitimate outcome, not an error."""
        treatment = _make_treatment(
            description='No anatomy here, just metadata.',
        )
        claude_response = json.dumps({'spans': []})
        client = _make_mock_messages_client(claude_response)
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result == []

    def test_invalid_response_returns_exception(self) -> None:
        """Bad JSON from Claude → exception returned (NOT raised),
        so the parallel worker pool keeps its other futures alive."""
        treatment = _make_treatment()
        client = _make_mock_messages_client('not valid json')
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert isinstance(result, Exception)
