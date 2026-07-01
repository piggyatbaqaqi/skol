"""Tests for treatments_to_structured.status."""

import pytest

from treatments_to_structured.status import (
    AnnotationResult,
    STATUS_ERROR,
    STATUS_PARTIAL,
    STATUS_SKIPPED_MERGE_SUSPECT,
    STATUS_SUCCESS,
    classify_result,
    make_skip_status_doc,
    make_status_doc,
    status_doc_id,
)


# ---------------------------------------------------------------------------
# AnnotationResult
# ---------------------------------------------------------------------------


class TestAnnotationResult:
    """In-memory carrier from annotate_one_treatment to main."""

    def test_minimal_success_construction(self) -> None:
        result = AnnotationResult(
            treatment_id='taxon_a', status=STATUS_SUCCESS,
        )
        assert result.treatment_id == 'taxon_a'
        assert result.status == 'success'
        assert result.annotations == []
        assert result.dropped_spans == []
        assert result.error_message is None

    def test_partial_carries_drops(self) -> None:
        result = AnnotationResult(
            treatment_id='taxon_a',
            status=STATUS_PARTIAL,
            annotations=[{'feature_label': 'Pileus'}],
            dropped_spans=[
                {
                    'feature_label': 'Spores',
                    'claude_text': 'av. = 98 µm',
                    'reason': 'not found',
                },
            ],
        )
        assert result.status == 'partial'
        assert len(result.annotations) == 1
        assert len(result.dropped_spans) == 1

    def test_error_carries_message(self) -> None:
        result = AnnotationResult(
            treatment_id='taxon_a',
            status=STATUS_ERROR,
            error_message='Invalid JSON from Claude',
        )
        assert result.status == 'error'
        assert result.error_message == 'Invalid JSON from Claude'

    def test_invalid_status_raises(self) -> None:
        """Typo guard: status must be one of the three known values."""
        with pytest.raises(ValueError) as exc:
            AnnotationResult(treatment_id='t', status='succes')
        assert "'succes'" in str(exc.value)
        assert 'success' in str(exc.value)
        assert 'partial' in str(exc.value)
        assert 'error' in str(exc.value)


# ---------------------------------------------------------------------------
# status_doc_id
# ---------------------------------------------------------------------------


class TestStatusDocId:
    """The CouchDB _id for a treatment's status doc."""

    def test_is_treatment_id_verbatim(self) -> None:
        """One status doc per treatment; keying on treatment_id
        verbatim means O(1) lookup via couchdb.Database.__getitem__
        without needing a view."""
        assert (
            status_doc_id('taxon_abc') == 'taxon_abc'
        )

    def test_distinct_treatments_distinct_ids(self) -> None:
        a = status_doc_id('taxon_aaa')
        b = status_doc_id('taxon_bbb')
        assert a != b


# ---------------------------------------------------------------------------
# classify_result
# ---------------------------------------------------------------------------


class TestClassifyResult:
    """The state-machine rule for turning a run outcome into a
    status string."""

    def test_no_annotations_no_drops_no_error_is_success(self) -> None:
        """Claude returned {"spans": []} — legitimate 'no features
        here' signal.  Not an error, not partial — success."""
        assert classify_result([], [], None) == STATUS_SUCCESS

    def test_annotations_no_drops_is_success(self) -> None:
        assert classify_result(
            [{'feature_label': 'Pileus'}], [], None,
        ) == STATUS_SUCCESS

    def test_any_drop_is_partial(self) -> None:
        assert classify_result(
            [{'feature_label': 'Pileus'}],
            [{'feature_label': 'Spores', 'claude_text': '...',
              'reason': '...'}],
            None,
        ) == STATUS_PARTIAL

    def test_all_dropped_no_annotations_is_partial(self) -> None:
        """Edge case: Claude returned spans but every single one
        failed recovery.  Still partial (NOT success), so the
        offline-recovery script picks it up."""
        assert classify_result(
            [],
            [{'feature_label': 'X', 'claude_text': '...',
              'reason': '...'}],
            None,
        ) == STATUS_PARTIAL

    def test_error_message_wins_over_other_state(self) -> None:
        """Catastrophic failure takes precedence — partial/success
        classifications would be misleading when the run didn't
        actually complete."""
        assert classify_result(
            [], [], 'Invalid JSON from Claude',
        ) == STATUS_ERROR


# ---------------------------------------------------------------------------
# make_status_doc
# ---------------------------------------------------------------------------


class TestMakeStatusDoc:
    """Build the CouchDB doc from an AnnotationResult."""

    def test_success_doc_shape(self) -> None:
        result = AnnotationResult(
            treatment_id='taxon_a',
            status=STATUS_SUCCESS,
            annotations=[
                {'feature_label': 'Pileus'},
                {'feature_label': 'Stipe'},
            ],
        )
        doc = make_status_doc(
            result, 'claude-opus-4-7', '2026-06-28T12:00:00Z',
        )
        assert doc['_id'] == 'taxon_a'
        assert doc['treatment_id'] == 'taxon_a'
        assert doc['status'] == 'success'
        assert doc['annotation_count'] == 2
        assert doc['dropped_span_count'] == 0
        assert doc['dropped_spans'] == []
        assert doc['error_message'] is None
        assert doc['attempt_count'] == 1
        assert doc['last_attempt_at'] == '2026-06-28T12:00:00Z'
        assert doc['model'] == 'claude-opus-4-7'

    def test_partial_doc_preserves_dropped_spans(self) -> None:
        """The dropped_spans list is the recovery queue — the
        offline fixes/ script reads it directly.  Must be
        preserved byte-for-byte (claude_text, feature_label,
        reason)."""
        result = AnnotationResult(
            treatment_id='taxon_a',
            status=STATUS_PARTIAL,
            annotations=[{'feature_label': 'Pileus'}],
            dropped_spans=[
                {
                    'feature_label': 'Spores',
                    'claude_text': 'av. = 98 mm',
                    'reason': 'not found in synthetic doc',
                },
            ],
        )
        doc = make_status_doc(
            result, 'claude-opus-4-7', '2026-06-28T12:00:00Z',
        )
        assert doc['status'] == 'partial'
        assert doc['annotation_count'] == 1
        assert doc['dropped_span_count'] == 1
        assert len(doc['dropped_spans']) == 1
        d = doc['dropped_spans'][0]
        assert d['feature_label'] == 'Spores'
        assert d['claude_text'] == 'av. = 98 mm'
        assert d['reason'] == 'not found in synthetic doc'

    def test_error_doc_carries_message(self) -> None:
        result = AnnotationResult(
            treatment_id='taxon_a',
            status=STATUS_ERROR,
            error_message='Claude returned invalid JSON',
        )
        doc = make_status_doc(
            result, 'claude-opus-4-7', '2026-06-28T12:00:00Z',
        )
        assert doc['status'] == 'error'
        assert doc['error_message'] == 'Claude returned invalid JSON'
        assert doc['annotation_count'] == 0
        assert doc['dropped_span_count'] == 0

    def test_attempt_count_passes_through(self) -> None:
        """Caller is responsible for reading the existing doc and
        incrementing.  make_status_doc just stamps the number
        it's given."""
        result = AnnotationResult(
            treatment_id='taxon_a', status=STATUS_SUCCESS,
        )
        doc = make_status_doc(
            result, 'm', 't', attempt_count=3,
        )
        assert doc['attempt_count'] == 3

    def test_no_rev_in_returned_doc(self) -> None:
        """make_status_doc never emits _rev — the caller is
        responsible for fetching the existing doc and merging
        _rev for overwrite.  Otherwise a stale _rev would clobber
        concurrent updates silently."""
        result = AnnotationResult(
            treatment_id='taxon_a', status=STATUS_SUCCESS,
        )
        doc = make_status_doc(result, 'm', 't')
        assert '_rev' not in doc

    def test_dropped_spans_list_is_copied(self) -> None:
        """The returned doc's dropped_spans must be independent of
        the AnnotationResult's list — mutating one shouldn't
        affect the other (the result is often passed back to a
        worker thread for logging)."""
        drops = [
            {'feature_label': 'X', 'claude_text': '...',
             'reason': '...'},
        ]
        result = AnnotationResult(
            treatment_id='taxon_a', status=STATUS_PARTIAL,
            dropped_spans=drops,
        )
        doc = make_status_doc(result, 'm', 't')
        # Mutate the original
        drops.append({
            'feature_label': 'Y', 'claude_text': '!',
            'reason': '?',
        })
        # Doc was not affected
        assert len(doc['dropped_spans']) == 1

    # ------------------------------------------------------------------
    # metrics sub-dict — cost / perf instrumentation
    # ------------------------------------------------------------------

    def test_metrics_omitted_when_result_has_none(self) -> None:
        """Pre-instrumentation behavior preserved: when the result
        carries no metrics, the doc has no 'metrics' key (not
        null).  Lets Mango queries distinguish old docs from new:
        `selector: {metrics: {$exists: false}}`."""
        result = AnnotationResult(
            treatment_id='taxon_a', status=STATUS_SUCCESS,
        )
        doc = make_status_doc(result, 'm', 't')
        assert 'metrics' not in doc

    def test_metrics_passed_through(self) -> None:
        """When the worker collected metrics, they flow through
        verbatim into the status doc.  Notebook code reads them
        directly off the status doc — no transformation."""
        result = AnnotationResult(
            treatment_id='taxon_a',
            status=STATUS_SUCCESS,
            metrics={
                'wall_clock_seconds': 12.3,
                'api_latency_seconds': 11.8,
                'input_tokens': 3942,
                'output_tokens': 587,
                'synth_doc_chars': 2104,
                'complexity_score': 0.85,
            },
        )
        doc = make_status_doc(result, 'm', 't')
        assert doc['metrics']['wall_clock_seconds'] == 12.3
        assert doc['metrics']['api_latency_seconds'] == 11.8
        assert doc['metrics']['input_tokens'] == 3942
        assert doc['metrics']['output_tokens'] == 587
        assert doc['metrics']['synth_doc_chars'] == 2104
        assert doc['metrics']['complexity_score'] == 0.85

    def test_metrics_dict_is_copied(self) -> None:
        """Same defensive-copy pattern as dropped_spans: mutating
        the AnnotationResult's metrics after make_status_doc must
        not affect the doc that's about to be saved."""
        m = {'wall_clock_seconds': 1.0}
        result = AnnotationResult(
            treatment_id='taxon_a', status=STATUS_SUCCESS,
            metrics=m,
        )
        doc = make_status_doc(result, 'm', 't')
        m['wall_clock_seconds'] = 999.0
        assert doc['metrics']['wall_clock_seconds'] == 1.0

    def test_metrics_partial_in_error_result(self) -> None:
        """The annotator collects what it can before an exception
        propagates — e.g., complexity_score and synth_doc_chars
        are computed BEFORE the API call, so they're present on
        error results even when input_tokens stays None.  Partial
        metrics are still useful for diagnostics ('did the API
        fail on big inputs?')."""
        result = AnnotationResult(
            treatment_id='taxon_a',
            status=STATUS_ERROR,
            error_message='simulated API failure',
            metrics={
                'complexity_score': 0.5,
                'synth_doc_chars': 2000,
                'api_latency_seconds': None,
                'input_tokens': None,
                'output_tokens': None,
                'wall_clock_seconds': 0.05,
            },
        )
        doc = make_status_doc(result, 'm', 't')
        assert doc['status'] == 'error'
        assert doc['metrics']['complexity_score'] == 0.5
        assert doc['metrics']['input_tokens'] is None


# ---------------------------------------------------------------------------
# make_skip_status_doc — for the pre-annotation skip case
# (bin/select_for_annotation's merge-suspect filter)
# ---------------------------------------------------------------------------


class TestMakeSkipStatusDoc:
    """The skip-status-doc shape.  Distinguishes 'never attempted'
    (this) from 'attempted and failed/succeeded' (make_status_doc)."""

    def test_status_and_never_attempted_fields(self) -> None:
        doc = make_skip_status_doc(
            'taxon_a',
            metric_value=42,
            threshold=10,
            metric_name='n_terms_above_5',
            decided_at='2026-07-01T00:00:00Z',
        )
        assert doc['_id'] == 'taxon_a'
        assert doc['treatment_id'] == 'taxon_a'
        assert doc['status'] == STATUS_SKIPPED_MERGE_SUSPECT
        # Never-attempted markers.
        assert doc['annotation_count'] == 0
        assert doc['dropped_span_count'] == 0
        assert doc['dropped_spans'] == []
        assert doc['error_message'] is None
        assert doc['attempt_count'] == 0
        assert doc['last_attempt_at'] is None
        assert doc['model'] is None

    def test_metrics_carry_decision_context(self) -> None:
        """Metric value + name + threshold + timestamp are all
        needed to reproduce (or re-evaluate) the skip decision if
        the threshold changes later."""
        doc = make_skip_status_doc(
            'taxon_a',
            metric_value=42,
            threshold=10,
            metric_name='n_terms_above_5',
            decided_at='2026-07-01T00:00:00Z',
        )
        assert doc['metrics']['n_terms_above_5'] == 42
        assert doc['metrics']['merge_threshold'] == 10
        assert doc['metrics']['decided_at'] == '2026-07-01T00:00:00Z'

    def test_metric_name_parameterized(self) -> None:
        """The metric name is passed in rather than hardcoded so
        future detectors (Zipf slope, entropy, ...) can reuse the
        same skip-doc shape without a new function."""
        doc = make_skip_status_doc(
            'taxon_a',
            metric_value=0.75,
            threshold=0.5,
            metric_name='zipf_slope',
            decided_at='ts',
        )
        assert doc['metrics']['zipf_slope'] == 0.75
        assert 'n_terms_above_5' not in doc['metrics']

    def test_no_rev_in_returned_doc(self) -> None:
        """Same rule as make_status_doc: caller manages _rev for
        overwrite.  A --force pass may need to overwrite a prior
        skip doc with a re-evaluated metric."""
        doc = make_skip_status_doc(
            'taxon_a',
            metric_value=42, threshold=10,
            metric_name='n_terms_above_5', decided_at='ts',
        )
        assert '_rev' not in doc
