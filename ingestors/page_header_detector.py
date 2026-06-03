"""Heuristic page-header detector for PDF-derived article plaintext.

Implements the 5-stage algorithm specified in
docs/page-header-detection.md and tracked as v4 plan §1.B (sub-rows
1.B.1 through 1.B.5):

    1.B.1 — collect_candidates       (digit-token discovery)
    1.B.2 — fit_sequence             (RANSAC + gap histogram)
    1.B.3 — partition_alternation    (recto/verso check)
    1.B.4 — cluster_header_text      (journal-name clustering)
    1.B.5 — recover_header_block     (two-pass block recovery)

The orchestrator ``detect_page_headers`` runs all five stages and
returns a JSON-serialisable dict that becomes the
``article.page-headers.json`` attachment.  The v4 feature assembler
(Step 2) reads ``per_line_confidence`` from this output to build the
``page_header_score[2]`` feature consumed by Pass-1's layout CRF.

Pure-numpy + stdlib regex/difflib — no scipy / sklearn / Levenshtein
dependency churn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# 1.B.1 — Candidate collection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageNumCandidate:
    """A digit token at the start or end of a line that *might* be a
    page number.  Final disposition (real page number vs. body-text
    coincidence) is decided downstream by 1.B.2's sequence fit.
    """
    line_index: int
    position: str       # 'start' or 'end'
    value: int
    raw_token: str      # preserves leading zeros if any
    prefix: str         # non-numeric text before the token (stripped)
    suffix: str         # non-numeric text after the token (stripped)


# A leading digit run that's either the whole line or followed by
# whitespace.  Captures all consecutive digits so 5+-digit tokens are
# detected (and rejected by the length check) rather than truncated to 4.
_START_DIGITS_RE = re.compile(r'^(\d+)(?=\s|$)')

# A trailing digit run, preceded by whitespace or start-of-line.
# Python's re module doesn't support variable-width lookbehind, so we
# match the boundary explicitly and use ``group(1)`` for the digits.
_END_DIGITS_RE = re.compile(r'(?:^|\s)(\d+)$')

# A 4-digit token that looks like a year (1900s or 2000s).  Per
# page-header-detection.md §Step 1, year-shape at line-end is a strong
# negative signal — years cluster rather than increment by 1 across a
# document, so they'd contaminate the sequence fit.
_YEAR_RE = re.compile(r'^(19|20)\d{2}$')


def collect_candidates(lines: List[str]) -> List[PageNumCandidate]:
    """Scan ``lines`` for page-number candidates.

    Rules (§Step 1 of page-header-detection.md):
    * Digit token of 1-4 digits at line-start OR line-end.
    * Token of 5+ digits is excluded (accession numbers, specimen IDs).
    * 4-digit year-shape (``(19|20)\\d{2}``) at line-end is excluded.
    * One candidate per line at most; if both ends match, the
      line-start candidate wins (running-header convention).
    """
    results: List[PageNumCandidate] = []
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            continue
        cand = _try_start(idx, stripped) or _try_end(idx, stripped)
        if cand is not None:
            results.append(cand)
    return results


def _try_start(line_index: int, stripped: str) -> PageNumCandidate | None:
    m = _START_DIGITS_RE.match(stripped)
    if m is None:
        return None
    token = m.group(1)
    if not 1 <= len(token) <= 4:
        return None
    # No year-shape rejection at start — see test_year_shape_accepted_at_start.
    return PageNumCandidate(
        line_index=line_index,
        position='start',
        value=int(token),
        raw_token=token,
        prefix='',
        suffix=stripped[m.end():].strip(),
    )


def _try_end(line_index: int, stripped: str) -> PageNumCandidate | None:
    m = _END_DIGITS_RE.search(stripped)
    if m is None:
        return None
    token = m.group(1)
    if not 1 <= len(token) <= 4:
        return None
    if _YEAR_RE.match(token):
        return None
    # m.start() includes the leading boundary character (whitespace
    # or start-of-line, captured by ``(?:^|\s)`` in the regex).  Use
    # m.start(1) to point at the digit token itself, so ``prefix``
    # excludes the leading-boundary whitespace.
    return PageNumCandidate(
        line_index=line_index,
        position='end',
        value=int(token),
        raw_token=token,
        prefix=stripped[:m.start(1)].strip(),
        suffix='',
    )
