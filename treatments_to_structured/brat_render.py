"""Synthetic brat document construction and annotation round-trip.

Phase 1 deliverable 4 of treatments_to_structured.  See
docs/schema_constrained_pipeline.md §10.2 and §10.4.

Builds the synthetic brat ``.txt`` for a Treatment (concatenating
the ``description`` and ``diagnosis`` prose with section markers),
emits brat ``.ann`` files from in-memory annotation lists, and
round-trips reviewer-edited ``.ann`` files back to annotation lists.

The module bridges three coordinate systems:

  1. Source-plaintext offsets (durable storage in
     ``annotation.source_spans``) — what survives any future
     re-render or re-extraction.
  2. Field-relative offsets (what we store as
     ``annotation.start`` / ``annotation.end``) — independent of
     the synthetic-doc layout, so layout changes don't migrate
     stored annotations.
  3. Synthetic-doc offsets (what brat actually sees in the .txt /
     .ann pair) — derived on the fly via ``SpanMap``.

The ``SpanMap`` returned by ``render`` carries the mapping between
all three so ``annotations_to_brat`` and ``parse_brat_ann`` can
translate without re-reading the Treatment.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Fields we annotate in Phase 1.  Order matters: drives the section
# order in the synthetic doc.
_RENDERED_FIELDS: Tuple[str, ...] = ('description', 'diagnosis')


@dataclass
class FieldExtent:
    """One field's content position in the synthetic doc plus the
    source-plaintext provenance that built it.

    ``synth_start`` / ``synth_end`` are byte/char offsets into the
    synthetic ``.txt`` ``render`` produced — what brat sees.

    ``source_spans`` are the (start, end) ranges in source
    plaintext that, in order, concatenate to form the field's
    rendered text.  Each entry is a dict ``{'start': int, 'end':
    int}`` with int values (production data sometimes carries
    these as strings; ``render`` normalizes).
    """

    field: str
    synth_start: int
    synth_end: int
    source_spans: List[Dict[str, int]]


@dataclass
class SpanMap:
    """Bridges synth-doc / (field, field-relative) / source-plaintext.

    ``field_extents`` enumerates each rendered field in synthetic-doc
    order.  ``synth_text`` is the full rendered text (kept so
    ``parse_brat_ann`` can slice out ``source_text`` from offsets
    without the caller having to thread the synthetic doc separately).
    """

    field_extents: List[FieldExtent]
    synth_text: str


def _normalize_source_span(span: Dict[str, Any]) -> Dict[str, int]:
    """Coerce ``start_char`` / ``end_char`` to ints.

    Production Treatment docs carry the offsets as either int (e.g.
    nomenclature_spans) or string (e.g. description_spans on some
    older docs).  Normalize at module boundary so downstream code
    can treat them uniformly.
    """
    return {
        'start': int(span['start_char']),
        'end': int(span['end_char']),
    }


def _normalize_source_spans(
    spans: List[Dict[str, Any]],
) -> List[Dict[str, int]]:
    return [_normalize_source_span(s) for s in spans]


def render(treatment: Dict[str, Any]) -> Tuple[str, SpanMap]:
    """Build the synthetic brat ``.txt`` and a ``SpanMap``.

    Output layout (when both fields are populated)::

        === description ===

        <description prose>

        === diagnosis ===

        <diagnosis prose>

    Missing / null / empty fields are skipped — no section header,
    no trailing blank lines.  A treatment with neither field
    populated renders to empty string and an empty ``SpanMap``.

    The returned ``SpanMap`` lets ``annotations_to_brat`` /
    ``parse_brat_ann`` translate between coordinate systems without
    re-reading the Treatment.

    Args:
        treatment: A Treatment document.  Reads ``description``,
            ``description_spans``, ``diagnosis``, ``diagnosis_spans``
            top-level fields.

    Returns:
        ``(synthetic_txt, span_map)``.
    """
    field_extents: List[FieldExtent] = []
    parts: List[str] = []
    cursor = 0

    for field_name in _RENDERED_FIELDS:
        text = treatment.get(field_name) or ''
        if not text:
            continue
        # Header: "=== <field_name> ===\n\n"
        header = f'=== {field_name} ===\n\n'
        # Trailing: "\n\n" so the next header (if any) is separated
        # from this field's content by a blank line, and the doc
        # ends with a single trailing newline overall.
        trailer = '\n\n'

        # Add a separator before the header if this isn't the first
        # section — keeps the doc from starting with a blank line.
        if parts:
            # Previous part ended with the trailer "\n\n"; that
            # already provides the blank line separator.  Nothing
            # more to add.
            pass

        parts.append(header)
        cursor += len(header)

        synth_start = cursor
        parts.append(text)
        cursor += len(text)
        synth_end = cursor

        parts.append(trailer)
        cursor += len(trailer)

        raw_spans = treatment.get(f'{field_name}_spans') or []
        field_extents.append(FieldExtent(
            field=field_name,
            synth_start=synth_start,
            synth_end=synth_end,
            source_spans=_normalize_source_spans(raw_spans),
        ))

    synth_text = ''.join(parts)
    return synth_text, SpanMap(
        field_extents=field_extents,
        synth_text=synth_text,
    )


def _field_relative_to_source_spans(
    field_relative_start: int,
    field_relative_end: int,
    source_spans: List[Dict[str, int]],
) -> List[Dict[str, int]]:
    """Map a field-relative range to one or more source-plaintext ranges.

    Walks ``source_spans`` in declaration order, accumulating their
    lengths.  For each source span that overlaps the field-relative
    range, emits a corresponding source-plaintext sub-range.

    Assumption: the field text is the in-order concatenation of
    ``plaintext[span.start:span.end]`` for each entry in
    ``source_spans``.  This holds for v4-extracted Treatments to
    within a small discrepancy on some docs (joiner characters
    between source-plaintext chunks aren't accounted for); Phase 1
    accepts the approximation.

    Returns at least one range when the input range is non-empty
    and overlaps any source span; an empty list otherwise.
    """
    result: List[Dict[str, int]] = []
    cumulative = 0
    for span in source_spans:
        span_len = span['end'] - span['start']
        span_field_start = cumulative
        span_field_end = cumulative + span_len

        overlap_start = max(field_relative_start, span_field_start)
        overlap_end = min(field_relative_end, span_field_end)
        if overlap_start < overlap_end:
            src_offset_lo = span['start'] + (overlap_start - cumulative)
            src_offset_hi = span['start'] + (overlap_end - cumulative)
            result.append({'start': src_offset_lo, 'end': src_offset_hi})

        cumulative += span_len
        if cumulative >= field_relative_end:
            break

    return result


def _field_relative_to_synth(
    field: str,
    field_relative_start: int,
    field_relative_end: int,
    span_map: SpanMap,
) -> Tuple[int, int]:
    """Translate (field, field-relative) → synth-doc-relative."""
    for ext in span_map.field_extents:
        if ext.field == field:
            return (
                ext.synth_start + field_relative_start,
                ext.synth_start + field_relative_end,
            )
    raise ValueError(
        f"field {field!r} not present in span_map "
        f"(rendered fields: "
        f"{[e.field for e in span_map.field_extents]})"
    )


def _synth_to_field_relative(
    synth_start: int,
    synth_end: int,
    span_map: SpanMap,
) -> Tuple[str, int, int, FieldExtent]:
    """Translate synth-doc-relative → (field, field-relative, ext).

    Raises ValueError if the range falls outside any field's content
    (e.g., lands in a header), or crosses a field boundary.
    """
    for ext in span_map.field_extents:
        if (
            synth_start >= ext.synth_start
            and synth_end <= ext.synth_end
        ):
            return (
                ext.field,
                synth_start - ext.synth_start,
                synth_end - ext.synth_start,
                ext,
            )
    raise ValueError(
        f"synthetic offsets ({synth_start}, {synth_end}) don't fall "
        f"entirely within any field's content; possible field-boundary "
        f"crossing or annotation inside a section header"
    )


# Brat T-line format:  T<id>\t<type> <start> <end>\t<text>
# For disjoint entities, multiple "start end" pairs separated by ';'.
# Phase 1 only emits contiguous entities (single pair).
_BRAT_T_LINE_RE = re.compile(
    r'^T(?P<num>\d+)\t'
    r'(?P<type>\S+) '
    r'(?P<start>\d+) (?P<end>\d+)\t'
    r'(?P<text>.*)$'
)

# Brat's storage regex for entity types: '^[a-zA-Z0-9_-]*$'.
# Anything else gets auto-mangled by brat (with a startup warning),
# and the mangled form is what gets saved when the reviewer hits
# save — so the round-trip silently degrades.  We sanitize at
# write time so brat has nothing to mangle and the wire form is
# stable across the export → review → ingest cycle.
_BRAT_TYPE_INVALID_RE = re.compile(r'[^A-Za-z0-9_-]+')


def brat_safe_type(label: str) -> str:
    """Convert a feature_label into a brat-safe entity-type token.

    Three transformations:

      1. Replace whitespace with underscore (already required for
         brat T-line single-token types).
      2. Replace any run of non-``[A-Za-z0-9_-]`` characters with
         a single underscore — strips parens, commas, periods,
         etc. that LLM-invented labels routinely include.
      3. Collapse adjacent underscores and strip leading/trailing
         underscores so labels read cleanly in brat's sidebar.

    Lossy: ``Partial veil (microscopic)`` becomes
    ``Partial_veil_microscopic`` — parens are dropped, not encoded.
    Parens carry no semantic load in mycological vocabulary; the
    biological identity is the words, not the punctuation.

    Idempotent: ``brat_safe_type(brat_safe_type(x)) == brat_safe_type(x)``.
    """
    if not label:
        return label
    s = _BRAT_TYPE_INVALID_RE.sub('_', label)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')


def annotations_to_brat(
    annotations: List[Dict[str, Any]],
    span_map: SpanMap,
) -> str:
    """Build a brat ``.ann`` file body from annotation dicts.

    Each annotation contributes one T-line.  IDs are assigned
    sequentially (T1, T2, ...) in input order.

    Args:
        annotations: List of dicts with at least ``feature_label``,
            ``field``, ``start``, ``end`` (field-relative).  Other
            keys (``source_text``, ``model``, etc.) are ignored —
            brat doesn't carry them.
        span_map: From ``render``; used to translate field-relative
            offsets to synth-doc offsets.

    Returns:
        Brat ``.ann`` file content (T-lines separated by ``\\n``,
        terminated by ``\\n``).  Empty input → empty string.

    Raises:
        ValueError: if any annotation's field is not present in the
            span_map, or if a feature_label contains whitespace
            (which brat T-line syntax can't represent).
    """
    if not annotations:
        return ''
    lines: List[str] = []
    for i, ann in enumerate(annotations, start=1):
        label = ann['feature_label']
        # Brat T-line types must be single tokens that match
        # ``^[a-zA-Z0-9_-]*$``.  Phase 1's Claude bootstrap
        # routinely produces multi-word labels with parens /
        # commas / periods ("Universal veil (microscopic, on
        # pileus)").  brat_safe_type does the full conversion
        # in one shot; without it, brat auto-mangles at load
        # time and the round-trip silently degrades.  Tabs are
        # rejected up-front — they're a format violation, not
        # a normalization concern.
        if '\t' in label:
            raise ValueError(
                f"feature_label {label!r} contains tab; brat "
                f"T-line syntax can't represent tabs in types"
            )
        wire_label = brat_safe_type(label)
        synth_start, synth_end = _field_relative_to_synth(
            ann['field'], ann['start'], ann['end'], span_map,
        )
        text = span_map.synth_text[synth_start:synth_end]
        # The custom skol brat fork unescapes `\n` (literal
        # backslash + n) in the T-line text field to a real
        # newline before verifying against the .txt at the
        # offsets.  Without this escape, our newline-to-space
        # substitution would mismatch the .txt content (which
        # has the real newline) and brat would reject the line
        # with "Unable to parse".  Convention established by
        # bin/yedda_to_brat.py.
        text = text.replace('\n', '\\n').replace('\r', '\\r')
        lines.append(
            f"T{i}\t{wire_label} {synth_start} {synth_end}\t{text}"
        )
    return '\n'.join(lines) + '\n'


def parse_brat_ann(
    ann_text: str,
    span_map: SpanMap,
) -> List[Dict[str, Any]]:
    """Read a brat ``.ann`` file body into annotation dicts.

    Synthesizes the durable fields each annotation needs:
      * ``field`` and (``start``, ``end``) are field-relative offsets
        derived from the synthetic-doc offsets in the T-line
      * ``source_text`` is sliced from ``span_map.synth_text``
      * ``source_spans`` is computed from the field's source
        provenance via ``_field_relative_to_source_spans``

    Skips lines that aren't T-entity lines (brat also stores
    relations, attributes, etc. — out of scope for Phase 1).  Skips
    blank lines.

    Args:
        ann_text: Full content of a brat ``.ann`` file.
        span_map: From ``render`` against the same Treatment whose
            ``.txt`` brat was editing.

    Returns:
        List of annotation dicts in T-line order.
    """
    annotations: List[Dict[str, Any]] = []
    for line in ann_text.splitlines():
        if not line or not line.startswith('T'):
            continue
        m = _BRAT_T_LINE_RE.match(line)
        if not m:
            continue
        synth_start = int(m.group('start'))
        synth_end = int(m.group('end'))
        # Reverse the wire-format substitution from
        # annotations_to_brat — underscores in the on-wire type
        # come back as spaces in the feature_label.  Round-trip is
        # lossless for labels that contain spaces; labels that
        # originally contained underscores would be returned as
        # spaces (uncommon for anatomical names, acceptable
        # ambiguity for Phase 1).
        feature_label = m.group('type').replace('_', ' ')

        field, field_start, field_end, ext = _synth_to_field_relative(
            synth_start, synth_end, span_map,
        )
        source_text = span_map.synth_text[synth_start:synth_end]
        source_spans = _field_relative_to_source_spans(
            field_start, field_end, ext.source_spans,
        )

        annotations.append({
            'feature_label': feature_label,
            'field': field,
            'start': field_start,
            'end': field_end,
            'source_text': source_text,
            'source_spans': source_spans,
        })
    return annotations


__all__ = (
    'FieldExtent',
    'SpanMap',
    'render',
    'annotations_to_brat',
    'parse_brat_ann',
)
