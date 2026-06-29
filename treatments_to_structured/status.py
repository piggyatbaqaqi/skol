"""Per-treatment annotation-run status records for the Phase 1
bootstrap annotator.

Phase 1 deliverable 5 (continued) of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.4 + the
data_quality_production_v4_model.md design notes.

Design Y from the per-span-isolation discussion: one status doc
per Treatment ever attempted, written to a sibling DB
(``skol_exp_<exp>_02_50_features_status``).  Together with the
candidate-annotations DB, this gives the operator a queryable
view over what's been processed, what succeeded, what partially
failed (some spans dropped), and what errored entirely — without
having to grep run-time JSONL logs.

Why a sibling DB rather than fields on annotation docs:
errored-with-zero-annotations treatments have no annotation doc
to attach status to; the sibling DB is the only design where
those cases are visible at all.

Status values:
  * ``success`` — all spans Claude returned were recovered to
    annotations and stored.  Includes the legitimate
    "Claude returned {"spans": []} (no features)" case.
  * ``partial`` — Claude returned at least one span we couldn't
    recover (whitespace-normalization too aggressive, span
    crossed a field boundary, etc.).  ``dropped_spans`` lists
    them with a ``reason``; the offline-recovery fixes/ script
    consumes this as a queue.  Some spans may have been stored
    successfully — they live in the candidate DB.
  * ``error`` — catastrophic failure before any spans could be
    parsed.  Invalid JSON from Claude, envelope-shape violations,
    network errors, etc.  No annotations were stored.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


STATUS_SUCCESS = 'success'
STATUS_PARTIAL = 'partial'
STATUS_ERROR = 'error'

_VALID_STATUSES = frozenset({
    STATUS_SUCCESS, STATUS_PARTIAL, STATUS_ERROR,
})


@dataclass
class AnnotationResult:
    """In-memory carrier from ``annotate_one_treatment`` to the
    main loop.  Mirrors the CouchDB status doc shape but is
    intentionally a Python dataclass so workers can pass it
    through ``concurrent.futures`` without touching CouchDB.

    The main loop calls ``to_status_doc`` to convert before
    writing.

    ``metrics`` carries cost / performance instrumentation
    collected during the run — see ``make_status_doc`` for the
    expected shape (wall_clock_seconds, api_latency_seconds,
    input_tokens, output_tokens, synth_doc_chars,
    complexity_score).  ``None`` when the run errored before
    instrumentation could be collected.
    """

    treatment_id: str
    status: str
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    dropped_spans: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f'status must be one of {sorted(_VALID_STATUSES)}; '
                f'got {self.status!r}'
            )


def status_doc_id(treatment_id: str) -> str:
    """The CouchDB ``_id`` for a treatment's status doc.

    One status doc per treatment — the doc is keyed on
    ``treatment_id`` directly.  Same-treatment re-runs overwrite
    in place; the ``attempt_count`` field tracks the history.

    Using the treatment_id verbatim (rather than a prefixed key
    like ``status:<treatment_id>``) means a Mango query for
    ``_id`` ranges does the obvious thing and a single CouchDB
    ``get`` is O(1) without a view.
    """
    return treatment_id


def make_status_doc(
    result: AnnotationResult,
    model: str,
    created_at: str,
    attempt_count: int = 1,
) -> Dict[str, Any]:
    """Build the CouchDB status doc for a treatment-run result.

    Args:
        result: ``AnnotationResult`` from
            ``annotate_one_treatment``.  Carries the optional
            ``metrics`` sub-dict (cost / perf instrumentation);
            see the field on AnnotationResult for the expected
            keys.  When None, the metrics field is omitted from
            the doc (rather than emitted as null) — keeps old
            status docs and new ones easy to distinguish in
            Mango queries (``selector: {metrics: {$exists: false}}``).
        model: Claude model name (e.g., ``"claude-opus-4-7"``).
        created_at: ISO-8601 timestamp.
        attempt_count: Number of times this treatment has been
            attempted, including this run.  Caller is
            responsible for reading the prior doc and
            incrementing.

    Returns:
        Dict ready for CouchDB save.  ``_id`` is set to
        ``treatment_id`` via ``status_doc_id``.  ``_rev`` is NOT
        included — caller must merge with the existing doc's
        ``_rev`` for overwrite (or use ``--force`` semantics).
    """
    doc: Dict[str, Any] = {
        '_id': status_doc_id(result.treatment_id),
        'treatment_id': result.treatment_id,
        'status': result.status,
        'annotation_count': len(result.annotations),
        'dropped_span_count': len(result.dropped_spans),
        'dropped_spans': list(result.dropped_spans),
        'error_message': result.error_message,
        'attempt_count': attempt_count,
        'last_attempt_at': created_at,
        'model': model,
    }
    # Include metrics only when collected — omitting (rather than
    # null-ing) keeps the pre-instrumentation status docs easy to
    # identify in queries: `selector: {metrics: {$exists: false}}`
    # selects rows that need recomputation if metrics are ever
    # backfilled.
    if result.metrics is not None:
        doc['metrics'] = dict(result.metrics)
    return doc


def classify_result(
    annotations: List[Dict[str, Any]],
    dropped_spans: List[Dict[str, Any]],
    error_message: Optional[str],
) -> str:
    """Pick the status string from a run's outcome.

    Rules (apply in order):

    1. If ``error_message`` is set, it's a catastrophic run-level
       failure — ``error`` regardless of what's in the other
       lists (which should both be empty in that case).
    2. If ``dropped_spans`` is non-empty, ``partial`` — at least
       one span Claude returned didn't make it to the candidate
       DB.  Holds even when ``annotations`` is also empty
       (every span was dropped).
    3. Otherwise ``success``.  This includes Claude returning
       ``{"spans": []}`` legitimately (no annotatable features) —
       a "Claude looked and found nothing" result is a valid
       success.
    """
    if error_message is not None:
        return STATUS_ERROR
    if dropped_spans:
        return STATUS_PARTIAL
    return STATUS_SUCCESS


__all__ = (
    'AnnotationResult',
    'STATUS_SUCCESS',
    'STATUS_PARTIAL',
    'STATUS_ERROR',
    'status_doc_id',
    'make_status_doc',
    'classify_result',
)
