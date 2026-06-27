"""Pure-logic helpers for the Phase 1 Claude-API bootstrap annotator.

Phase 1 deliverable 5 of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.4.

Two halves of the annotation pipeline live here as pure functions
so they can be tested without an Anthropic API key or network:

  * ``build_user_prompt`` — assemble the user-turn prompt sent to
    Claude for one Treatment.  Combines the feature schema (as
    structural guidance) with the synthetic brat ``.txt``.
  * ``parse_claude_response`` — parse Claude's JSON response into
    annotation dicts, recover synth-doc offsets via string search,
    translate to the durable storage shape (field-relative offsets
    + source-plaintext source_spans) using a SpanMap from
    ``brat_render.render``.

The CLI (``bin/llm_annotate_features.py``) wires these together
with CouchDB I/O, the Anthropic SDK, parallel workers, and the
``--estimate`` / ``--dry-run`` / ``--skip-existing`` flags.
"""

import json
import re
from typing import Any, Dict, List, Optional

from treatments_to_structured.brat_render import (
    SpanMap,
    _field_relative_to_source_spans,
    _synth_to_field_relative,
)


_SYSTEM_PROMPT = (
    "You are a precise mycological text annotator.  Identify spans of "
    "text that describe a single named anatomical feature, exactly as "
    "instructed in the user message.  Output only the JSON envelope "
    "specified in the user message — no preamble, no markdown fences, "
    "no commentary."
)


def build_user_prompt(
    synth_txt: str,
    schema: Dict[str, Any],
    feature_label: str,
) -> str:
    """Assemble the user-turn prompt for one Treatment.

    Combines the feature schema (full JSON, including the
    anatomical-boundary guidance from the schema's ``description``)
    with the synthetic brat ``.txt`` of the Treatment.  Asks Claude
    to return a JSON object with a ``"spans"`` array, each entry a
    ``{"text": "..."}`` carrying the verbatim source text of one
    feature mention.

    Source-text-only output (rather than start/end offsets) keeps
    the contract robust: offsets are recovered downstream by
    string-searching ``synth_txt`` in left-to-right order, which
    sidesteps off-by-one and Unicode-width ambiguities that LLM
    offset arithmetic is notoriously bad at.
    """
    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
    return (
        f"Annotate every span in the treatment below that describes "
        f"the **{feature_label}** feature, as defined by the schema.\n"
        f"\n"
        f"Read the schema's `description` field carefully — it specifies "
        f"both what counts AS this feature AND what does NOT (the\n"
        f"anatomical-boundary section).  Spans for other features must "
        f"NOT be returned.\n"
        f"\n"
        f"SCHEMA:\n"
        f"```json\n"
        f"{schema_json}\n"
        f"```\n"
        f"\n"
        f"OUTPUT FORMAT (return exactly this JSON envelope, nothing else):\n"
        f"```json\n"
        f'{{"spans": [{{"text": "..."}}, ...]}}\n'
        f"```\n"
        f"\n"
        f"Each `text` value must be a verbatim substring of the treatment "
        f"text below (whitespace and punctuation included; do not "
        f"paraphrase or normalize).  Use as many or as few entries as the "
        f"text supports; empty `spans` is valid when the feature is not "
        f"mentioned.\n"
        f"\n"
        f"TREATMENT TEXT:\n"
        f"{synth_txt}"
    )


# Strips leading / trailing ```json or ``` fences from a model response
# that ignored the no-fences instruction.  Tolerant by design: production
# LLM outputs occasionally include markdown despite explicit prompts.
_FENCE_RE = re.compile(
    r'^\s*```(?:json)?\s*\n(?P<body>.*?)\n```\s*$',
    re.DOTALL,
)


def _strip_json_fences(text: str) -> str:
    """Strip a leading/trailing ```json fence pair if present."""
    m = _FENCE_RE.match(text.strip())
    if m:
        return m.group('body')
    return text.strip()


class ClaudeResponseError(ValueError):
    """Raised when Claude's response can't be parsed into annotations.

    Carries the offending response (truncated) for operator-side debugging.
    """


def parse_claude_response(
    response_text: str,
    span_map: SpanMap,
    feature_label: str,
    model_name: str,
    treatment_id: str,
    doc_id: str,
    created_at: str,
) -> List[Dict[str, Any]]:
    """Parse Claude's JSON response into annotation dicts.

    For each ``{"text": "..."}`` span Claude emits, locates that
    text in ``span_map.synth_text`` left-to-right (each search
    starts after the previous match's end so repeated phrases
    don't all collapse to the same offsets), then translates the
    recovered synth-doc offsets into the durable storage shape:
    ``(field, field-relative start, field-relative end,
    source_spans, source_text)`` per the
    docs/schema_constrained_pipeline.md §10.3 schema.

    Args:
        response_text: Raw text from Claude's message response.
            May or may not include ```json fences (tolerated).
        span_map: From ``brat_render.render`` against the same
            Treatment.
        feature_label: The label to attach to every annotation
            (e.g. "Pileus").
        model_name: Recorded on each annotation for provenance.
        treatment_id: The Treatment's ``_id``.
        doc_id: The source ingest doc's ``_id`` (from
            ``treatment.ingest._id``).
        created_at: ISO-8601 timestamp for the annotation batch.

    Returns:
        List of annotation dicts matching the §10.3 schema.  Empty
        list if Claude returned ``{"spans": []}`` — a legitimate
        signal that the feature isn't mentioned in this Treatment.

    Raises:
        ClaudeResponseError: invalid JSON, wrong envelope shape, or
            any span's ``text`` not found in the synth doc.
    """
    cleaned = _strip_json_fences(response_text)
    try:
        envelope = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ClaudeResponseError(
            f"response is not valid JSON: {exc}; "
            f"response (truncated): {cleaned[:200]!r}"
        ) from exc

    if not isinstance(envelope, dict) or 'spans' not in envelope:
        raise ClaudeResponseError(
            f"response missing 'spans' key in envelope; "
            f"got top-level keys: "
            f"{list(envelope) if isinstance(envelope, dict) else type(envelope).__name__}"
        )

    spans = envelope['spans']
    if not isinstance(spans, list):
        raise ClaudeResponseError(
            f"'spans' must be a list; got {type(spans).__name__}"
        )

    annotations: List[Dict[str, Any]] = []
    cursor = 0  # next synth-doc position to search from
    synth = span_map.synth_text
    for i, span in enumerate(spans):
        if not isinstance(span, dict) or 'text' not in span:
            raise ClaudeResponseError(
                f"spans[{i}] missing 'text' key; got {span!r}"
            )
        wanted = span['text']
        if not isinstance(wanted, str) or not wanted:
            raise ClaudeResponseError(
                f"spans[{i}].text must be a non-empty string; got {wanted!r}"
            )

        # Left-to-right search from cursor; if not found there, try
        # from the start (Claude may have emitted spans out of
        # document order, which is fine; we still want a hit).
        idx = synth.find(wanted, cursor)
        if idx < 0:
            idx = synth.find(wanted)
        if idx < 0:
            raise ClaudeResponseError(
                f"spans[{i}].text not found in synthetic doc: "
                f"{wanted[:120]!r}"
            )

        synth_start = idx
        synth_end = idx + len(wanted)
        cursor = synth_end

        try:
            field, fr_start, fr_end, ext = _synth_to_field_relative(
                synth_start, synth_end, span_map,
            )
        except ValueError as exc:
            raise ClaudeResponseError(
                f"spans[{i}] at synth offsets ({synth_start}, "
                f"{synth_end}) crosses a field boundary or lands in "
                f"a section header: {exc}"
            ) from exc

        source_spans = _field_relative_to_source_spans(
            fr_start, fr_end, ext.source_spans,
        )

        annotations.append({
            'feature_label': feature_label,
            'field': field,
            'start': fr_start,
            'end': fr_end,
            'source_text': wanted,
            'source_spans': source_spans,
            'model': model_name,
            'created_at': created_at,
            'treatment_id': treatment_id,
            'doc_id': doc_id,
        })
    return annotations


def annotation_doc_id(
    treatment_id: str,
    feature_label: str,
    field_relative_start: int,
) -> str:
    """Build the CouchDB ``_id`` for an annotation doc.

    Format: ``<treatment_id>:<feature_label>:<start>`` — the offset
    component disambiguates multiple spans of the same feature
    within one Treatment (one Pileus mention per Treatment is the
    common case, but treatments do exist with two or more).

    Used by the bootstrap annotator's idempotent-overwrite semantics:
    re-running with an updated prompt overwrites the previous
    annotation at the same offset.
    """
    return f'{treatment_id}:{feature_label}:{field_relative_start}'


__all__ = (
    'build_user_prompt',
    'parse_claude_response',
    'annotation_doc_id',
    'ClaudeResponseError',
)
