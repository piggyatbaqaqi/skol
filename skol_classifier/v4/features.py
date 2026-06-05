"""Line-feature assembler for the v4 classifier.

Builds the 791-dim per-line feature vector enumerated in
docs/v4_classifier_plan.md §Feature engineering::

    sbert[768] + particles[12] + layout[8]
        + page_header_score[2] + section_header_flag[1] = 791

This module is a pure transform — no I/O, no model loading, no Redis
writes.  It takes the three Step-1 artifacts (the SBERT cache
populated in Step 0, the spans + page-header attachments produced in
Step 1.D) and returns a :class:`LineFeatures` record per line.

Particle vocab is locked at 11 labels + 1 SP_NOV flag (= 12 slots):

    TaxonName, Author, DOI, MB-number, Page-ref, GBIF-ID,
    ISSN, CBS-number, Author-footnote, Iconography-header,
    Fungarium-code, SP_NOV_flag

``section-header`` lives in its own dedicated feature
(``section_header_flag[1]``).  ``PDF-page-marker`` is implied by the
page-header score, so it doesn't need its own particle slot.

The ``sbert_lookup`` callable is the cache-miss escape hatch:
returning ``None`` falls back to a zero vector, so the assembler
never depends on the cache being warm — populating it is the
caller's job (typically via ``bin/embed_lines.py`` before training,
or on-the-fly at inference time).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, Optional, Sequence, Tuple,
)

import numpy as np

from ingestors.spans import Span


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARTICLE_VOCAB: Tuple[str, ...] = (
    'TaxonName',
    'Author',
    'DOI',
    'MB-number',
    'Page-ref',
    'GBIF-ID',
    'ISSN',
    'CBS-number',
    'Author-footnote',
    'Iconography-header',
    'Fungarium-code',
)
# Particle vector length = 11 vocab counts + 1 SP_NOV flag.
_PARTICLE_DIM = len(PARTICLE_VOCAB) + 1
_SP_NOV_INDEX = _PARTICLE_DIM - 1

_LAYOUT_DIM = 8
_PAGE_HEADER_DIM = 2
_SECTION_HEADER_DIM = 1

_SBERT_DIM = 768

FEATURE_DIM: int = (
    _SBERT_DIM + _PARTICLE_DIM + _LAYOUT_DIM
    + _PAGE_HEADER_DIM + _SECTION_HEADER_DIM
)
"""Total per-line feature width = 768 + 12 + 8 + 2 + 1 = 791.
Canonical source of truth — both CRFs default to this.  Re-derived
from the individual block sizes so the assertion stays correct if
any block is resized in a later step."""

PARTICLE_SLICE = slice(_SBERT_DIM, _SBERT_DIM + _PARTICLE_DIM)
"""Canonical slice of the 791-d feature vector covering the
particle block (indices 768..779 today).  Re-derived from the
block-size constants so it tracks any future resize.  Step 7.γ's
ablation zeros this slice at inference to measure how much the
spans pipeline contributes to F1."""

_IS_SHORT_THRESHOLD = 30


# ---------------------------------------------------------------------------
# LineFeatures dataclass
# ---------------------------------------------------------------------------


SbertLookup = Callable[[str], Optional[np.ndarray]]


@dataclass(frozen=True)
class LineFeatures:
    """Per-line feature record split into the five v4 feature blocks.

    Kept as a dataclass (rather than a single ``np.ndarray``) so
    Step-7 ablations can zero out one block at a time before
    concatenation — e.g. ``feats.particles[:] = 0`` to measure the
    particle block's contribution.
    """
    sbert: np.ndarray                # (768,) float32
    particles: np.ndarray            # (12,)  float32
    layout: np.ndarray               # (8,)   float32
    page_header_score: np.ndarray    # (2,)   float32
    section_header_flag: np.ndarray  # (1,)   float32

    def concat(self) -> np.ndarray:
        """Return the 791-dim feature vector the CRFs consume.

        Block order is fixed: sbert, particles, layout,
        page_header_score, section_header_flag.
        """
        return np.concatenate([
            self.sbert,
            self.particles,
            self.layout,
            self.page_header_score,
            self.section_header_flag,
        ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Layout block
# ---------------------------------------------------------------------------


def layout_features(
    line_text: str,
    line_index: int,
    doc_lines: Sequence[str],
) -> np.ndarray:
    """Eight layout signals for a single line.

    Order: length, indent_pct, allcaps_pct, digit_pct,
    trailing_digit_flag, is_short, blank_before, blank_after.
    """
    n_chars = len(line_text)
    leading_ws = len(line_text) - len(line_text.lstrip())
    indent_pct = leading_ws / n_chars if n_chars else 0.0

    letters = [ch for ch in line_text if ch.isalpha()]
    allcaps_pct = (
        sum(1 for ch in letters if ch.isupper()) / len(letters)
        if letters else 0.0
    )

    digits = sum(1 for ch in line_text if ch.isdigit())
    digit_pct = digits / n_chars if n_chars else 0.0

    stripped = line_text.rstrip()
    trailing_digit_flag = (
        1.0 if stripped and stripped[-1].isdigit() else 0.0
    )

    is_short = (
        1.0 if len(line_text.strip()) < _IS_SHORT_THRESHOLD else 0.0
    )

    n_lines = len(doc_lines)
    blank_before = (
        1.0
        if line_index <= 0 or not doc_lines[line_index - 1].strip()
        else 0.0
    )
    blank_after = (
        1.0
        if line_index >= n_lines - 1
        or not doc_lines[line_index + 1].strip()
        else 0.0
    )

    return np.asarray([
        float(n_chars),
        indent_pct,
        allcaps_pct,
        digit_pct,
        trailing_digit_flag,
        is_short,
        blank_before,
        blank_after,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Particle block
# ---------------------------------------------------------------------------


def _overlaps_line(
    span: Span, line_start: int, line_end: int,
) -> bool:
    """Half-open overlap: ``span`` intersects ``[line_start,
    line_end)`` iff ``span.start < line_end and span.end > line_start``.
    """
    return span.start < line_end and span.end > line_start


def particle_counts(
    spans: Sequence[Span],
    line_start: int,
    line_end: int,
) -> np.ndarray:
    """Build the 12-dim particle vector for one line.

    The first 11 slots are counts of overlapping spans whose
    ``label`` matches the corresponding ``PARTICLE_VOCAB`` entry.
    Slot 11 is the SP_NOV flag: 1.0 if any overlapping TaxonName
    span has ``metadata.get('annot_nomen_type') == 'SP_NOV'``.
    """
    out = np.zeros(_PARTICLE_DIM, dtype=np.float32)
    for span in spans:
        if not _overlaps_line(span, line_start, line_end):
            continue
        if span.label in PARTICLE_VOCAB:
            idx = PARTICLE_VOCAB.index(span.label)
            out[idx] += 1.0
            if span.label == 'TaxonName':
                nomen = (span.metadata or {}).get('annot_nomen_type')
                if nomen == 'SP_NOV':
                    out[_SP_NOV_INDEX] = 1.0
    return out


# ---------------------------------------------------------------------------
# Page-header score block
# ---------------------------------------------------------------------------


def page_header_score(
    line_index: int,
    page_headers: Dict[str, Any],
) -> np.ndarray:
    """Two-dim block: clamped per-line confidence + binary "in any
    region" flag.

    Defensive: an out-of-range ``line_index`` or missing
    ``per_line_confidence`` key yields ``[0.0, 0.0]`` — happens when
    ``detect_page_headers`` was called on a different line
    tokenization than the caller's ``doc_lines``.
    """
    conf_list = page_headers.get('per_line_confidence') or []
    if 0 <= line_index < len(conf_list):
        raw = float(conf_list[line_index])
    else:
        raw = 0.0
    confidence = max(0.0, min(1.0, raw))
    binary_flag = 1.0 if confidence > 0.0 else 0.0
    return np.asarray([confidence, binary_flag], dtype=np.float32)


# ---------------------------------------------------------------------------
# Section-header flag block
# ---------------------------------------------------------------------------


def section_header_flag(
    spans: Sequence[Span],
    line_start: int,
    line_end: int,
) -> np.ndarray:
    """One-dim block: 1.0 if any ``section-header`` span overlaps
    the line, else 0.0."""
    for span in spans:
        if span.label != 'section-header':
            continue
        if _overlaps_line(span, line_start, line_end):
            return np.asarray([1.0], dtype=np.float32)
    return np.asarray([0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Line-start table
# ---------------------------------------------------------------------------


def compute_line_starts(doc_lines: Sequence[str]) -> Tuple[int, ...]:
    """Cumulative char-offset table: ``starts[i]`` is the byte offset
    where ``doc_lines[i]`` begins in the original plaintext, assuming
    one ``'\\n'`` separator between consecutive lines.

    Pass this in via ``build_line_features(..., line_starts=...)`` to
    compute once and reuse across every line of the doc instead of
    re-deriving it per call.
    """
    starts = [0]
    for line in doc_lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)  # +1 for '\n'
    return tuple(starts)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_line_features(
    line_text: str,
    line_index: int,
    doc_lines: Sequence[str],
    spans: Sequence[Span],
    page_headers: Dict[str, Any],
    sbert_lookup: SbertLookup,
    *,
    sbert_dim: int = 768,
    line_starts: Optional[Sequence[int]] = None,
) -> LineFeatures:
    """Assemble all five v4 feature blocks for a single line.

    ``sbert_lookup`` is invoked with ``line_text``; on cache miss
    (return value ``None``) we fall back to a zero-vector of
    ``sbert_dim`` floats.

    ``line_starts`` is the cumulative-offset table from
    :func:`compute_line_starts`; pass it in to skip recomputing on
    every line.
    """
    starts = line_starts if line_starts is not None else compute_line_starts(doc_lines)
    line_start = int(starts[line_index])
    line_end = line_start + len(line_text)

    sbert = sbert_lookup(line_text)
    if sbert is None:
        sbert = np.zeros(sbert_dim, dtype=np.float32)
    else:
        sbert = sbert.astype(np.float32, copy=False)

    return LineFeatures(
        sbert=sbert,
        particles=particle_counts(spans, line_start, line_end),
        layout=layout_features(line_text, line_index, doc_lines),
        page_header_score=page_header_score(line_index, page_headers),
        section_header_flag=section_header_flag(
            spans, line_start, line_end,
        ),
    )
