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
from typing import Any, Dict, List, Tuple

from treatments_to_structured.feature_label_rules import (
    split_medium_context,
)
from treatments_to_structured.brat_render import (
    SpanMap,
    _field_relative_to_source_spans,
    _synth_to_field_relative,
    brat_safe_type,
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
    seed: Dict[str, Any],
) -> str:
    """Assemble the user-turn prompt for one Treatment.

    Takes a seed file (an open-ended example list, NOT an
    exhaustive vocabulary — see ``seeds/fungi.json``) plus the
    synthetic brat ``.txt``.  Asks Claude to return a JSON object
    with a ``"spans"`` array, each entry a ``{"text": "...",
    "feature_label": "..."}`` carrying the verbatim source text of
    one feature mention and the label Claude chose for it.

    Open-ended labelling: Claude is told to *use* the example
    labels when applicable and to *invent* canonical anatomical
    names for features not in the seed.  This keeps the
    architecture taxon-agnostic — swapping ``seeds/fungi.json`` for
    ``seeds/plants.json`` is enough to retarget the annotator.

    Source-text-only output (rather than start/end offsets) keeps
    the contract robust: offsets are recovered downstream by
    string-searching ``synth_txt`` in left-to-right order, which
    sidesteps off-by-one and Unicode-width ambiguities that LLM
    offset arithmetic is notoriously bad at.
    """
    seed_examples = seed.get('examples') or []
    examples_block = '\n'.join(
        f"  - **{ex['name']}** — {ex['description']}"
        for ex in seed_examples
    )
    seed_description = seed.get('description', '')
    return (
        "Annotate every span of text in the treatment below that "
        "describes one specific anatomical feature of the specimen.\n"
        "\n"
        "LABELLING RULES:\n"
        "  1. Use the example labels below when the corresponding "
        "feature is described.\n"
        "  2. For features NOT in the example list, invent a similar "
        "canonical anatomical name (e.g., 'Hymenophore', "
        "'Pileipellis', 'Conidiophores').  Use the most specific "
        "term the treatment uses when it offers one.\n"
        "  3. Each span describes ONE feature.  If a block of text "
        "discusses multiple features, split it into separate "
        "annotations.\n"
        "  4. Read the seed's `description` for the anatomical-"
        "boundary discipline you should apply throughout.\n"
        "\n"
        f"SEED CONTEXT:\n"
        f"{seed_description}\n"
        "\n"
        "EXAMPLE LABELS:\n"
        f"{examples_block}\n"
        "\n"
        "OUTPUT FORMAT (return exactly this JSON envelope, nothing "
        "else — no preamble, no markdown fences, no commentary):\n"
        "```json\n"
        '{"spans": [{"text": "...", "feature_label": "..."}, ...]}\n'
        "```\n"
        "\n"
        "Each `text` value must be a verbatim substring of the "
        "treatment text below (whitespace and punctuation included; "
        "do not paraphrase or normalize).  Use as many or as few "
        "entries as the text supports; empty `spans` is valid when "
        "no anatomical features are described.\n"
        "\n"
        "TREATMENT TEXT:\n"
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
    model_name: str,
    treatment_id: str,
    doc_id: str,
    created_at: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse Claude's JSON response into annotation dicts.

    For each ``{"text": "...", "feature_label": "..."}`` span Claude
    emits, locates the text in ``span_map.synth_text`` left-to-right
    (each search starts after the previous match's end so repeated
    phrases don't all collapse to the same offsets), then translates
    the recovered synth-doc offsets into the durable storage shape:
    ``(field, field-relative start, field-relative end,
    source_spans, source_text)`` per
    docs/schema_constrained_pipeline.md §10.3.

    Per-span isolation: a span that fails offset recovery or
    field-boundary translation is collected into ``dropped_spans``
    rather than raising — the surviving spans still get stored.
    This trades atomic-success semantics for partial-storage +
    offline-recovery semantics (see ``status.py`` and the
    fixes/-side recovery script).

    Envelope-level failures (invalid JSON, missing ``spans`` key,
    spans not a list) still raise — those are catastrophic, not
    per-span.

    Args:
        response_text: Raw text from Claude's message response.
            May or may not include ```json fences (tolerated).
        span_map: From ``brat_render.render`` against the same
            Treatment.
        model_name: Recorded on each annotation for provenance.
        treatment_id: The Treatment's ``_id``.
        doc_id: The source ingest doc's ``_id`` (from
            ``treatment.ingest._id``).
        created_at: ISO-8601 timestamp for the annotation batch.

    Returns:
        ``(annotations, dropped_spans)`` tuple:

        * ``annotations`` — list of annotation dicts matching the
          §10.3 schema.  Each carries the ``feature_label`` Claude
          chose for the span (may be a seed label OR an invented
          canonical name).  Empty when Claude returned
          ``{"spans": []}`` (legitimate "no features" signal) OR
          when every span failed recovery.
        * ``dropped_spans`` — list of ``{"feature_label", "claude_text",
          "reason"}`` dicts capturing spans Claude returned that
          we couldn't recover offsets for.  Empty in the happy
          path.  Becomes the recovery queue for the offline
          ``fixes/`` script.

    Raises:
        ClaudeResponseError: invalid JSON, missing ``spans`` key,
            ``spans`` not a list, individual span missing
            ``text`` or ``feature_label`` keys, or empty/whitespace
            ``text`` or ``feature_label``.  These are envelope-
            level contract violations — Claude returned something
            we can't reason about at all.
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
    dropped_spans: List[Dict[str, Any]] = []
    cursor = 0  # next synth-doc position to search from
    synth = span_map.synth_text
    for i, span in enumerate(spans):
        # Envelope-shape violations on individual spans STILL
        # raise — these mean Claude returned something we can't
        # reason about at the per-span level (missing keys,
        # non-string values).  Recovery is impossible without
        # re-prompting Claude entirely.
        if not isinstance(span, dict) or 'text' not in span:
            raise ClaudeResponseError(
                f"spans[{i}] missing 'text' key; got {span!r}"
            )
        if 'feature_label' not in span:
            raise ClaudeResponseError(
                f"spans[{i}] missing 'feature_label' key; got {span!r}"
            )
        wanted = span['text']
        feature_label = span['feature_label']
        if not isinstance(wanted, str) or not wanted:
            raise ClaudeResponseError(
                f"spans[{i}].text must be a non-empty string; got {wanted!r}"
            )
        if not isinstance(feature_label, str) or not feature_label.strip():
            raise ClaudeResponseError(
                f"spans[{i}].feature_label must be a non-empty string; "
                f"got {feature_label!r}"
            )
        # Canonicalize to brat-safe form at parse time so the
        # candidate DB and the brat wire form share one vocabulary.
        # Without this, the export → review → ingest round-trip
        # would diff every paren'd label as delete+add because the
        # candidate label ("Universal veil (microscopic, on pileus)")
        # differs from the brat-mangled form parse_brat_ann reads
        # back.  brat_safe_type returns the wire form (underscores);
        # the candidate DB stores spaces — matching existing data
        # and what parse_brat_ann produces on the inverse direction.
        # See treatments_to_structured.brat_render.brat_safe_type for
        # the sanitization rules.
        wire_form = brat_safe_type(feature_label.strip())
        if not wire_form:
            raise ClaudeResponseError(
                f"spans[{i}].feature_label sanitized to empty "
                f"string; original was {span['feature_label']!r}"
            )
        feature_label = wire_form.replace('_', ' ')

        # Left-to-right search from cursor; if not found there, try
        # from the start (Claude may have emitted spans out of
        # document order, which is fine; we still want a hit).
        idx = synth.find(wanted, cursor)
        if idx < 0:
            idx = synth.find(wanted)

        if idx >= 0:
            synth_start = idx
            synth_end = idx + len(wanted)
        else:
            # Fallback: whitespace-tolerant regex search.  LLMs
            # routinely normalize narrow no-break space (U+202F),
            # non-breaking space (U+00A0), thin space (U+2009),
            # and other unicode whitespace to U+0020 when echoing
            # source text.  Persoonia / Fungal Planet treatments
            # use U+202F for unit spacing ("av. = 98");
            # exact str.find then fails on whitespace-only
            # differences that are visually identical.
            #
            # Build a regex from the wanted text where any
            # whitespace run matches any non-empty whitespace run
            # in the source.  We search in the ORIGINAL synth_text
            # so the recovered start/end remain in original
            # coordinates — no offset translation needed.
            #
            # Splitting on `\s+` before escaping (rather than
            # substituting after) sidesteps a Python 3.13 surprise:
            # re.escape() escapes literal spaces to `\ ` (backslash-
            # space), so a naive `re.sub(r'\s+', r'\\s+', escaped)`
            # leaves the backslash in front of each space and the
            # resulting pattern matches a LITERAL backslash, not
            # whitespace.  Split-then-rejoin avoids that.
            parts = re.split(r'\s+', wanted)
            fuzzy = r'\s+'.join(re.escape(p) for p in parts)
            fuzzy_re = re.compile(fuzzy)
            m = fuzzy_re.search(synth, cursor)
            if m is None:
                m = fuzzy_re.search(synth)
            if m is None:
                # Per-span recovery failure → drop and continue.
                # Offline ``fixes/`` script reads dropped_spans to
                # retry with more aggressive normalization
                # (NFKD, dash unification, etc.) without a new
                # API call.
                dropped_spans.append({
                    'feature_label': feature_label,
                    'claude_text': wanted,
                    'reason': (
                        'text not found in synthetic doc '
                        '(exact and whitespace-tolerant search '
                        'both failed)'
                    ),
                })
                continue
            synth_start = m.start()
            synth_end = m.end()
        cursor = synth_end

        try:
            field, fr_start, fr_end, ext = _synth_to_field_relative(
                synth_start, synth_end, span_map,
            )
        except ValueError as exc:
            # Per-span recovery failure → drop and continue.
            # Span crossed a field boundary or landed in a section
            # header.  The reviewer can resolve manually if it
            # represents a real anatomical mention.
            dropped_spans.append({
                'feature_label': feature_label,
                'claude_text': wanted,
                'reason': (
                    f'span at synth offsets ({synth_start}, '
                    f'{synth_end}) crosses a field boundary or '
                    f'lands in a section header: {exc}'
                ),
            })
            continue

        source_spans = _field_relative_to_source_spans(
            fr_start, fr_end, ext.source_spans,
        )

        # Store source_text from the SOURCE (synth slice), NOT from
        # Claude's echo.  In the exact-match case these are equal;
        # in the fuzzy-match case Claude's `wanted` has normalized
        # whitespace while the source preserves original chars
        # (U+202F, U+00A0, newlines).  Downstream brat rendering
        # uses source_text to lay out spans against the actual
        # plaintext attachment, so source bytes must win.
        annotation = {
            'feature_label': feature_label,
            'field': field,
            'start': fr_start,
            'end': fr_end,
            'source_text': synth[synth_start:synth_end],
            'source_spans': source_spans,
            'model': model_name,
            'created_at': created_at,
            'treatment_id': treatment_id,
            'doc_id': doc_id,
        }
        # The growth condition gets its own field.  `feature_label`
        # is left exactly as emitted: it keys this doc, so splitting
        # the label here would re-key the annotation.  See
        # feature_label_rules.split_medium_context.
        _, context = split_medium_context(feature_label)
        if context is not None:
            annotation['context'] = context
        annotations.append(annotation)
    return annotations, dropped_spans


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
