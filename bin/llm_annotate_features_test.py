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
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_annotate_features import (  # type: ignore[import]  # noqa: E402
    annotate_one_treatment,
    estimate_tokens,
    filter_already_annotated,
    load_seed,
    iter_treatment_ids,
    read_treatment_ids,
    resolve_id_filter,
    resolve_candidate_db_name,
    resolve_status_db_name,
)
from treatments_to_structured.status import (  # noqa: E402
    AnnotationResult,
    STATUS_ERROR,
    STATUS_PARTIAL,
    STATUS_SUCCESS,
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


def _canonicalization() -> Dict[str, str]:
    """The hand-maintained drift map, read straight from docs/."""
    path = (
        Path(__file__).resolve().parent.parent
        / 'docs' / 'feature_label_canonicalization.json'
    )
    with path.open() as f:
        return {
            k: v for k, v in json.load(f).items()
            if not k.startswith('_')
        }


class TestSeedCanonicalVocabulary:
    """The seed is what teaches Claude which label to reach for, so a
    seed label the canonicalization map immediately rewrites would be
    self-defeating — and seed *prose* teaches vocabulary just as the
    labels do.  'Sporocarp cap' in the Pileus description, with no
    Sporocarp entry to anchor it, is why the corpus ended up split
    10/8 between Sporophore and Sporocarp."""

    def test_no_seed_label_is_rewritten_by_canonicalization(self) -> None:
        """Holds today; pinned so a future seed addition can't quietly
        teach a label we then rewrite."""
        canon = _canonicalization()
        names = {e['name'] for e in load_seed('fungi')['examples']}
        overlap = names & set(canon)
        assert not overlap, sorted(overlap)

    def test_seed_has_a_sporophore_entry(self) -> None:
        names = {e['name'] for e in load_seed('fungi')['examples']}
        assert 'Sporophore' in names

    def test_sporophore_entry_names_the_clade_specific_forms(self) -> None:
        """Per the 2026-08-17 decision: use the generic only when the
        treatment is generic; preserve whichever clade-specific term
        the treatment itself uses.  The entry has to say so, or the
        annotator will generalise."""
        entry = next(
            e for e in load_seed('fungi')['examples']
            if e['name'] == 'Sporophore'
        )
        for term in ('Basidiomata', 'Ascomata', 'Conidiomata'):
            assert term in entry['description'], term

    def test_seed_prose_uses_the_canonical_generic(self) -> None:
        seed = load_seed('fungi')
        prose = seed['description'] + ' '.join(
            e['description'] for e in seed['examples']
        )
        assert 'porocarp' not in prose, 'seed prose still says Sporocarp'


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
# resolve_status_db_name (Design Y sibling DB)
# ---------------------------------------------------------------------------


class TestResolveStatusDbName:
    """Sibling DB to features_candidate; same fallback shape."""

    def test_uses_databases_features_status_when_set(self) -> None:
        exp = {
            'databases': {
                'features_status': 'skol_exp_X_features_status',
            },
        }
        warn = io.StringIO()
        name = resolve_status_db_name(
            'production_v4', exp, verbosity=1, warn_stream=warn,
        )
        assert name == 'skol_exp_X_features_status'
        assert warn.getvalue() == ''

    def test_falls_back_to_naming_convention_when_unset(self) -> None:
        exp = {'databases': {}}
        warn = io.StringIO()
        name = resolve_status_db_name(
            'production_v4', exp, verbosity=1, warn_stream=warn,
        )
        # Same 02_50 slot as the candidate DB — keeps the pair
        # sorted together in Fauxton.
        assert name == 'skol_exp_production_v4_02_50_features_status'
        assert 'NOTE' in warn.getvalue()

    def test_silent_at_verbosity_zero(self) -> None:
        warn = io.StringIO()
        resolve_status_db_name(
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


class _FakeStatusDb:
    """Stand-in for the status DB.  Keyed by treatment_id; each
    value is a dict with at least a 'status' key."""

    def __init__(self, docs: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.docs = docs or {}

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self.docs:
            # Real couchdb raises ResourceNotFound; the production
            # code catches Exception broadly so any exception is
            # fine for the test contract.
            raise KeyError(doc_id)
        return self.docs[doc_id]


class TestFilterAlreadyAnnotated:
    """Status-aware skip/retry: filter against the features_status
    DB rather than per-annotation lookups."""

    def test_default_skips_success_retries_partial_and_error(
        self,
    ) -> None:
        """Default mode: success treatments are skipped; partial
        and error are retried (re-prompt might succeed, or the
        offline-recovery script may have updated dropped_spans)."""
        db = _FakeStatusDb({
            'taxon_success': {'status': 'success'},
            'taxon_partial': {'status': 'partial'},
            'taxon_error': {'status': 'error'},
        })
        # taxon_new has no status doc → process
        result = filter_already_annotated(
            ['taxon_success', 'taxon_partial', 'taxon_error',
             'taxon_new'],
            db, mode='default',
        )
        assert result == ['taxon_partial', 'taxon_error', 'taxon_new']

    def test_skip_existing_drops_any_status(self) -> None:
        """--skip-existing widens skip to ANY status doc — operator
        opted out of automatic retries."""
        db = _FakeStatusDb({
            'taxon_success': {'status': 'success'},
            'taxon_partial': {'status': 'partial'},
            'taxon_error': {'status': 'error'},
        })
        result = filter_already_annotated(
            ['taxon_success', 'taxon_partial', 'taxon_error',
             'taxon_new'],
            db, mode='skip_existing',
        )
        assert result == ['taxon_new']

    def test_force_processes_everything(self) -> None:
        """--force ignores status entirely — every input ID is
        processed.  Used after a prompt or seed change."""
        db = _FakeStatusDb({
            'taxon_success': {'status': 'success'},
        })
        result = filter_already_annotated(
            ['taxon_success', 'taxon_new'], db, mode='force',
        )
        assert result == ['taxon_success', 'taxon_new']

    def test_input_order_preserved(self) -> None:
        """Order matters: the worker pool's progress display, the
        log file, and any per-treatment summary expect the
        original input ordering."""
        db = _FakeStatusDb()
        result = filter_already_annotated(
            ['c', 'a', 'b'], db, mode='default',
        )
        assert result == ['c', 'a', 'b']

    def test_empty_input(self) -> None:
        db = _FakeStatusDb()
        assert filter_already_annotated([], db, mode='default') == []
        assert (
            filter_already_annotated([], db, mode='skip_existing')
            == []
        )
        assert (
            filter_already_annotated([], db, mode='force') == []
        )

    def test_missing_status_field_treated_as_retry(self) -> None:
        """Status doc with no 'status' field (corrupt or pre-v2
        record) → not success → retry under default mode.  Better
        to over-retry than to silently skip a treatment we can't
        confirm completed."""
        db = _FakeStatusDb({
            'taxon_weird': {'attempt_count': 1},  # no 'status'
        })
        result = filter_already_annotated(
            ['taxon_weird'], db, mode='default',
        )
        assert result == ['taxon_weird']


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
        # Output estimate is 1/2 of input — calibrated 2026-06-29
        # from measured 47.12% ratio in the live 6-treatment sample.
        assert stats['est_output_tokens'] == 500

    def test_three_prompts_sum_tokens(self) -> None:
        client = _make_mock_count_tokens_client(500)
        stats = estimate_tokens(
            client,
            [('a', 'p'), ('b', 'p'), ('c', 'p')],
            'claude-opus-4-7',
        )
        assert stats['total_input_tokens'] == 1500
        assert stats['est_output_tokens'] == 750

    def test_cost_calculated_from_pricing_table(self) -> None:
        client = _make_mock_count_tokens_client(1_000_000)
        stats = estimate_tokens(
            client, [('a', 'p')], 'claude-opus-4-7',
        )
        # Opus-tier is $5.00 / $25.00 per MTok, not $15 / $75.
        # 1M input tokens × $5.00/1M = $5.00 input cost
        # 500k output tokens × $25.00/1M = $12.50 output cost
        assert stats['est_input_cost_usd'] == 5.00
        assert stats['est_output_cost_usd'] == 12.50
        assert stats['est_total_cost_usd'] == 17.50

    def test_unknown_model_raises_rather_than_guessing(self) -> None:
        """A cost estimate is the number an operator commits budget
        on, so an unpriced model must stop the run rather than quote
        a guess.  Guessing is how the $15/$75 rows survived: the
        wrong number looked like a real one."""
        client = _make_mock_count_tokens_client(1_000_000)
        with pytest.raises(ValueError):
            estimate_tokens(
                client, [('a', 'p')], 'claude-future-99-99',
            )


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


def _make_mock_messages_client(
    claude_response_text: str,
    *,
    input_tokens: Optional[int] = 100,
    output_tokens: Optional[int] = 25,
) -> Any:
    """A MagicMock client whose messages.create returns a single-block
    response carrying the given text.

    Token counts default to small non-None values so the
    instrumentation path (response.usage.input_tokens /
    output_tokens) is exercised by every test.  Pass None for
    either to simulate an SDK that doesn't return usage (covered
    by a dedicated regression test)."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=claude_response_text)]
    if input_tokens is None and output_tokens is None:
        # Strip the usage attribute entirely so getattr returns None.
        del response.usage
    else:
        response.usage = MagicMock(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
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
    """End-to-end with a mocked Anthropic client.

    Returns AnnotationResult (always — never raises) carrying
    status, annotations, dropped_spans, error_message.  See
    treatments_to_structured/status.py for the schema.
    """

    def test_happy_path_returns_success_result(self) -> None:
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
        assert isinstance(result, AnnotationResult)
        assert result.status == STATUS_SUCCESS
        assert result.treatment_id == 'taxon_test'
        assert result.error_message is None
        assert result.dropped_spans == []
        assert len(result.annotations) == 1
        ann = result.annotations[0]
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
        assert result.status == STATUS_SUCCESS
        labels = [a['feature_label'] for a in result.annotations]
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
        assert result.status == STATUS_SUCCESS
        assert result.annotations[0]['feature_label'] == 'Hymenophore'

    def test_empty_treatment_returns_success_no_annotations(
        self,
    ) -> None:
        """A treatment with neither description nor diagnosis
        renders to an empty synth doc; annotate skips the API call
        and returns a success result with zero annotations.  Same
        classification as 'Claude found nothing'."""
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
        assert result.status == STATUS_SUCCESS
        assert result.annotations == []
        assert result.dropped_spans == []
        # No API call should have been made.
        client.messages.create.assert_not_called()

    def test_no_spans_returned_is_success(self) -> None:
        """Claude says 'no anatomical features mentioned' — that's
        a legitimate outcome, classified as success (NOT error,
        NOT partial)."""
        treatment = _make_treatment(
            description='No anatomy here, just metadata.',
        )
        claude_response = json.dumps({'spans': []})
        client = _make_mock_messages_client(claude_response)
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result.status == STATUS_SUCCESS
        assert result.annotations == []
        assert result.dropped_spans == []

    def test_invalid_response_returns_error_result(self) -> None:
        """Bad JSON from Claude → status='error' with diagnostic
        in error_message.  No raise — parallel worker pool stays
        healthy."""
        treatment = _make_treatment()
        client = _make_mock_messages_client('not valid json')
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert isinstance(result, AnnotationResult)
        assert result.status == STATUS_ERROR
        assert result.error_message is not None
        assert 'JSON' in result.error_message or 'json' in result.error_message
        assert result.annotations == []

    def test_partial_when_some_spans_fail_recovery(self) -> None:
        """If Claude returns N spans and one fails offset recovery
        (e.g., hallucinated text not in the source), the surviving
        spans are stored in annotations and the failed one lands
        in dropped_spans.  Status: 'partial'."""
        treatment = _make_treatment(
            description='Pileus brown 3 cm.',
        )
        claude_response = json.dumps({
            'spans': [
                {
                    'text': 'Pileus brown 3 cm.',
                    'feature_label': 'Pileus',
                },
                {
                    # Hallucinated — not in the source.
                    'text': 'Lamellae cream-colored.',
                    'feature_label': 'Lamellae',
                },
            ],
        })
        client = _make_mock_messages_client(claude_response)
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result.status == STATUS_PARTIAL
        assert len(result.annotations) == 1
        assert result.annotations[0]['feature_label'] == 'Pileus'
        assert len(result.dropped_spans) == 1
        assert result.dropped_spans[0]['feature_label'] == 'Lamellae'

    def test_anthropic_api_error_returns_error_result(self) -> None:
        """A network / API error from the SDK becomes an error
        AnnotationResult rather than propagating.  Critical for
        parallel worker isolation."""
        treatment = _make_treatment()
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError(
            'simulated network failure',
        )
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result.status == STATUS_ERROR
        assert 'simulated network failure' in result.error_message

    # ------------------------------------------------------------------
    # metrics instrumentation
    # ------------------------------------------------------------------

    def test_metrics_collected_on_happy_path(self) -> None:
        """Every successful run carries the full metrics dict —
        the Heaps' Law notebook and the cost/perf regressions
        depend on these being present whenever Claude was actually
        called."""
        treatment = _make_treatment()
        claude_response = json.dumps({
            'spans': [{
                'text': 'Pileus brown 3 cm.',
                'feature_label': 'Pileus',
            }],
        })
        client = _make_mock_messages_client(
            claude_response, input_tokens=3942, output_tokens=587,
        )
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result.metrics is not None
        # All six instrumentation fields populated.
        m = result.metrics
        assert m['input_tokens'] == 3942
        assert m['output_tokens'] == 587
        assert m['synth_doc_chars'] > 0
        assert m['complexity_score'] >= 0.0
        assert m['api_latency_seconds'] is not None
        assert m['api_latency_seconds'] >= 0.0
        assert m['wall_clock_seconds'] is not None
        # Wall-clock must be at least api-latency (wall-clock
        # subsumes the API call plus pre/post-processing).
        assert (
            m['wall_clock_seconds'] >= m['api_latency_seconds']
        )

    def test_metrics_on_empty_synth_doc_skips_api_fields(
        self,
    ) -> None:
        """A treatment with neither description nor diagnosis
        skips the API call.  Metrics captures the work that DID
        happen (complexity, synth_doc_chars=0, wall_clock) and
        leaves the API-dependent fields as None.  Useful in the
        notebook for filtering 'real' runs from empty-shortcut
        runs."""
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
        assert result.metrics is not None
        assert result.metrics['synth_doc_chars'] == 0
        assert result.metrics['api_latency_seconds'] is None
        assert result.metrics['input_tokens'] is None
        assert result.metrics['output_tokens'] is None
        # Wall-clock and complexity still populated.
        assert result.metrics['wall_clock_seconds'] is not None
        assert 'complexity_score' in result.metrics
        client.messages.create.assert_not_called()

    def test_metrics_on_api_error_carries_partial_data(self) -> None:
        """When the API call raises, metrics carries what was
        collected before the raise — complexity, synth_doc_chars,
        wall_clock — but the API-dependent fields stay None.
        Useful diagnostic: 'did the API fail on big inputs?'"""
        treatment = _make_treatment()
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError(
            'simulated network failure',
        )
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result.status == STATUS_ERROR
        assert result.metrics is not None
        assert result.metrics['synth_doc_chars'] > 0
        assert result.metrics['complexity_score'] >= 0.0
        assert result.metrics['input_tokens'] is None
        assert result.metrics['wall_clock_seconds'] is not None

    def test_metrics_when_sdk_omits_usage(self) -> None:
        """Defensive: if the SDK ever returns a response without
        a `usage` attribute (older versions, mocks that forget),
        the worker doesn't crash — input/output tokens stay None
        and the run still classifies as success."""
        treatment = _make_treatment()
        claude_response = json.dumps({
            'spans': [{
                'text': 'Pileus brown 3 cm.',
                'feature_label': 'Pileus',
            }],
        })
        client = _make_mock_messages_client(
            claude_response,
            input_tokens=None, output_tokens=None,
        )
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result.status == STATUS_SUCCESS
        assert result.metrics is not None
        assert result.metrics['input_tokens'] is None
        assert result.metrics['output_tokens'] is None
        # api_latency and wall_clock are still populated — they
        # don't depend on usage.
        assert result.metrics['api_latency_seconds'] is not None

    def test_empty_response_content_returns_clean_error(
        self,
    ) -> None:
        """Claude can return response.content == [] when it
        refuses an input (e.g., the 2026-06-29 Colletotrichum
        treatment whose diagnosis was hundreds of U+FFFD
        replacement chars from corrupt OCR — Claude returned 1
        output token and zero content blocks, which used to raise
        IndexError in our worker).  We now surface this as a
        STATUS_ERROR with a clear diagnostic instead."""
        treatment = _make_treatment()
        client = MagicMock()
        response = MagicMock()
        response.content = []  # the refusal case
        response.usage = MagicMock(input_tokens=2346, output_tokens=1)
        response.stop_reason = 'end_turn'
        client.messages.create.return_value = response
        result = annotate_one_treatment(
            client, treatment, _TEST_SEED, 'claude-opus-4-7',
        )
        assert result.status == STATUS_ERROR
        assert 'empty content' in result.error_message
        assert 'stop_reason' in result.error_message
        # output_tokens captured so the operator can see the
        # 1-token-no-output pattern that signals refusal.
        assert result.metrics['output_tokens'] == 1
        # stop_reason captured in metrics too.
        assert result.metrics.get('stop_reason') == 'end_turn'


# ---------------------------------------------------------------------------
# iter_treatment_ids — the streaming half of read_treatment_ids
# ---------------------------------------------------------------------------


class _RecordingStream:
    """Stream that permits ONLY ``readline``.

    ``for line in sys.stdin`` block-buffers when stdin is a pipe, so an
    id pasted into a live session would sit unprocessed until the
    buffer filled.  This fake makes that mistake fail loudly: ``read``
    and ``__iter__`` raise, and ``readline`` calls are counted so a
    test can assert the generator is lazy.
    """

    def __init__(self, lines: List[str]) -> None:
        self._lines = list(lines)
        self.readline_calls = 0

    def readline(self) -> str:
        self.readline_calls += 1
        return self._lines.pop(0) if self._lines else ''

    def read(self, *_a: Any, **_kw: Any) -> str:
        raise AssertionError('read() would block on a pipe; use readline()')

    def __iter__(self) -> Any:
        raise AssertionError('iteration block-buffers; use readline()')


class TestIterTreatmentIds:
    """Lazy, line-at-a-time, and never reads ahead."""

    def test_yields_first_id_after_a_single_readline(self) -> None:
        """The property the whole streaming mode depends on."""
        stream = _RecordingStream(['taxon_a\n', 'taxon_b\n'])
        gen = iter_treatment_ids(stream)
        assert next(gen) == 'taxon_a'
        assert stream.readline_calls == 1

    def test_strips_and_skips_blank_lines(self) -> None:
        stream = _RecordingStream(['  taxon_a  \n', '\n', '   \n',
                                   'taxon_b\n'])
        assert list(iter_treatment_ids(stream)) == ['taxon_a', 'taxon_b']

    def test_empty_stream_yields_nothing(self) -> None:
        assert list(iter_treatment_ids(_RecordingStream([]))) == []

    def test_stops_at_eof_not_on_a_blank_line(self) -> None:
        """A blank line is skipped; only '' (EOF) terminates."""
        stream = _RecordingStream(['taxon_a\n', '\n', 'taxon_b\n'])
        assert list(iter_treatment_ids(stream)) == ['taxon_a', 'taxon_b']


class TestReadTreatmentIdsStillBatches:
    """The batch contract must survive the refactor."""

    def test_reads_every_line_to_eof(self) -> None:
        stream = io.StringIO('taxon_a\ntaxon_b\ntaxon_c\n')
        assert read_treatment_ids(
            None, stream, stdin_isatty=False,
        ) == ['taxon_a', 'taxon_b', 'taxon_c']

    def test_doc_ids_still_win_over_stdin(self) -> None:
        stream = io.StringIO('taxon_from_stdin\n')
        assert read_treatment_ids(
            ['taxon_from_flag'], stream, stdin_isatty=False,
        ) == ['taxon_from_flag']

    def test_empty_stdin_still_raises(self) -> None:
        with pytest.raises(ValueError):
            read_treatment_ids(None, io.StringIO(''), stdin_isatty=False)


class TestResolveIdFilter:
    """Optional filter: '-' means stdin, absent means no filter."""

    def test_no_doc_ids_does_not_read_stdin(self) -> None:
        """The cron regression, and the important test in this file.

        cron gives a non-TTY /dev/null stdin.  A tool that read stdin
        whenever it was not a TTY would consume nothing, filter to
        nothing, and process nothing — silently breaking every
        scheduled invocation while exiting 0.
        """
        stream = _RecordingStream(['taxon_should_not_be_read\n'])
        assert resolve_id_filter(
            None, stream, stdin_isatty=False,
        ) is None
        assert stream.readline_calls == 0

    def test_empty_list_also_means_no_filter(self) -> None:
        stream = _RecordingStream(['taxon_x\n'])
        assert resolve_id_filter([], stream, stdin_isatty=False) is None
        assert stream.readline_calls == 0

    def test_sentinel_reads_stdin(self) -> None:
        assert resolve_id_filter(
            ['-'], io.StringIO('taxon_a\ntaxon_b\n'), stdin_isatty=False,
        ) == ['taxon_a', 'taxon_b']

    def test_literal_ids_pass_through(self) -> None:
        stream = _RecordingStream(['taxon_x\n'])
        assert resolve_id_filter(
            ['taxon_a', 'taxon_b'], stream, stdin_isatty=False,
        ) == ['taxon_a', 'taxon_b']
        assert stream.readline_calls == 0

    def test_sentinel_with_empty_stdin_raises(self) -> None:
        """Asking for stdin and getting nothing is an operator error."""
        with pytest.raises(ValueError):
            resolve_id_filter(['-'], io.StringIO(''), stdin_isatty=False)
