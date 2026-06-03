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

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


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


# ---------------------------------------------------------------------------
# 1.B.2 — Sequence fitting via RANSAC
# ---------------------------------------------------------------------------


# RANSAC hyperparameters.  Chosen for journal-article docs which
# typically have 1-50 pages of body text with ~1 header per page.
_RANSAC_TRIALS = 50
_RANSAC_INLIER_THRESHOLD = 1.0  # value units; |predicted - actual| <= 1.0
_MIN_INLIERS = 4                # below this we don't trust the fit


@dataclass(frozen=True)
class SequenceFit:
    """Result of fitting a monotonic sequence to page-number candidates.

    ``value ≈ slope × line_index + intercept`` over the inliers.  The
    gap histogram captures the first-difference distribution across
    the inliers' values (after sorting by line_index); a sharp peak at
    1 or 2 implies a real page-number sequence rather than noise.
    """
    slope: float
    intercept: float
    inlier_line_indices: Tuple[int, ...]
    gap_histogram: Dict[int, int]
    quality_score: float


def fit_sequence(
    candidates: List[PageNumCandidate],
    *,
    seed: Optional[int] = None,
) -> Optional[SequenceFit]:
    """Fit a monotonic page-number sequence via simple RANSAC.

    Returns ``None`` when there aren't enough candidates to draw a
    meaningful conclusion (fewer than ``_MIN_INLIERS``) or when the
    best fit fails to gather that many inliers.  The ``seed`` kwarg
    makes the RANSAC sampler deterministic for tests.
    """
    if len(candidates) < _MIN_INLIERS:
        return None

    xs = np.array([c.line_index for c in candidates], dtype=np.float64)
    ys = np.array([c.value for c in candidates], dtype=np.float64)

    rng = np.random.default_rng(seed)
    n = len(candidates)

    best_inlier_mask: Optional[np.ndarray] = None
    best_slope = 0.0
    best_intercept = 0.0

    for _ in range(_RANSAC_TRIALS):
        i, j = rng.choice(n, size=2, replace=False)
        if xs[i] == xs[j]:
            continue
        slope = (ys[j] - ys[i]) / (xs[j] - xs[i])
        intercept = ys[i] - slope * xs[i]
        predicted = slope * xs + intercept
        inliers = np.abs(predicted - ys) <= _RANSAC_INLIER_THRESHOLD
        if best_inlier_mask is None or inliers.sum() > best_inlier_mask.sum():
            best_inlier_mask = inliers
            best_slope = float(slope)
            best_intercept = float(intercept)

    if best_inlier_mask is None or best_inlier_mask.sum() < _MIN_INLIERS:
        return None

    # Sort the surviving candidates by line_index for the gap histogram.
    inlier_pairs = sorted(
        (int(xs[k]), int(ys[k]))
        for k in range(n) if bool(best_inlier_mask[k])
    )
    inlier_indices = tuple(li for li, _ in inlier_pairs)
    inlier_values = [v for _, v in inlier_pairs]

    diffs = [inlier_values[k + 1] - inlier_values[k]
             for k in range(len(inlier_values) - 1)]
    # Keep only positive gaps in the histogram — negative gaps would
    # indicate a wrong line ordering (shouldn't happen after sorting
    # by line_index for a real sequence).
    positive_diffs = [d for d in diffs if d > 0]
    histogram: Dict[int, int] = dict(Counter(positive_diffs))

    if not positive_diffs:
        return None

    # Quality score: dominance of the most-frequent gap, but only
    # rewarded if the dominant gap is 1 or 2 (per page-header-detection.md
    # §Step 2 — those are the expected gaps for real page numbers).
    max_gap, max_count = max(histogram.items(), key=lambda kv: kv[1])
    if max_gap not in (1, 2):
        return None
    quality = max_count / len(positive_diffs)

    return SequenceFit(
        slope=best_slope,
        intercept=best_intercept,
        inlier_line_indices=inlier_indices,
        gap_histogram=histogram,
        quality_score=quality,
    )


# ---------------------------------------------------------------------------
# 1.B.3 — Recto/verso alternation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlternationFit:
    """Result of the recto/verso alternation check (§Step 3).

    ``verso_fit`` covers the even-valued candidates (left / verso
    pages by convention), ``recto_fit`` covers the odd-valued ones
    (right / recto).  ``alternation_score`` is the fraction of
    adjacent candidate pairs (sorted by ``line_index``) whose values
    differ in parity — high values imply a clean recto/verso layout.
    """
    verso_fit: Optional[SequenceFit]
    recto_fit: Optional[SequenceFit]
    alternation_score: float


def partition_alternation(
    candidates: List[PageNumCandidate],
    *,
    seed: Optional[int] = None,
) -> AlternationFit:
    """Fit even / odd candidate subsets independently and score how
    cleanly the parities interleave by ``line_index``.

    Always returns an ``AlternationFit`` — never None — so the
    orchestrator can fold an empty result into the per-line confidence
    array without special-casing.
    """
    even = [c for c in candidates if c.value % 2 == 0]
    odd = [c for c in candidates if c.value % 2 == 1]
    verso_fit = fit_sequence(even, seed=seed)
    recto_fit = fit_sequence(odd, seed=seed)

    if len(candidates) < 2:
        return AlternationFit(verso_fit, recto_fit, 0.0)

    sorted_cands = sorted(candidates, key=lambda c: c.line_index)
    alternations = 0
    pairs = 0
    for a, b in zip(sorted_cands, sorted_cands[1:]):
        pairs += 1
        if (a.value % 2) != (b.value % 2):
            alternations += 1

    score = alternations / pairs if pairs else 0.0
    return AlternationFit(verso_fit, recto_fit, score)


# ---------------------------------------------------------------------------
# 1.B.4 — Journal-name clustering
# ---------------------------------------------------------------------------


_CLUSTER_SIMILARITY_THRESHOLD = 0.75
_JOURNAL_CANONICAL_MAX_LEN = 30


@dataclass(frozen=True)
class HeaderTextCluster:
    """A cluster of candidate header lines whose non-numeric text
    matches under approximate string similarity (§Step 4).

    ``canonical`` is the cluster's representative text (first member's
    raw form).  ``members`` lists the ``line_index`` values that
    belong to the cluster.  ``cluster_kind`` is a coarse classifier
    that future stages can use to label runs (`journal`, `title`,
    or `other`) — purely heuristic.
    """
    canonical: str
    members: Tuple[int, ...]
    cluster_kind: str


def _normalize_header_text(text: str) -> str:
    """Lowercase + drop non-alphanumeric for similarity comparison.

    The drop step folds whitespace, punctuation, and accents-as-marks
    into a single normalised form so 'Smith et al. 2023' and
    'smithetal.2023' compare as equal under SequenceMatcher.
    """
    return ''.join(ch.lower() for ch in text if ch.isalnum())


def _classify_cluster_kind(canonical: str) -> str:
    """Heuristic kind label for a cluster's representative text."""
    stripped = canonical.strip()
    if not stripped:
        return 'other'
    alpha = ''.join(ch for ch in stripped if ch.isalpha())
    if alpha and alpha.isupper() and len(stripped) <= _JOURNAL_CANONICAL_MAX_LEN:
        return 'journal'
    if len(stripped) > _JOURNAL_CANONICAL_MAX_LEN:
        return 'title'
    return 'other'


def cluster_header_text(
    confirmed: List[PageNumCandidate],
) -> List[HeaderTextCluster]:
    """Cluster the non-numeric portions of confirmed candidates.

    Pulls ``suffix`` for start-position candidates and ``prefix`` for
    end-position candidates, normalises for comparison, then greedy
    agglomerates clusters whose representative texts have
    ``difflib.SequenceMatcher.ratio() >= 0.75``.  Only clusters with
    at least two members survive — singletons can't confirm a
    repeated-header pattern.
    """
    items: List[Tuple[int, str]] = []
    for c in confirmed:
        text = (c.suffix if c.position == 'start' else c.prefix).strip()
        if text:
            items.append((c.line_index, text))

    if len(items) < 2:
        return []

    # Start with one-element clusters; merge greedily.
    clusters: List[List[Tuple[int, str]]] = [[it] for it in items]

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a = _normalize_header_text(clusters[i][0][1])
                b = _normalize_header_text(clusters[j][0][1])
                if not a or not b:
                    continue
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                if ratio >= _CLUSTER_SIMILARITY_THRESHOLD:
                    clusters[i].extend(clusters[j])
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break

    results: List[HeaderTextCluster] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        canonical = cluster[0][1]
        members = tuple(li for li, _ in cluster)
        results.append(HeaderTextCluster(
            canonical=canonical,
            members=members,
            cluster_kind=_classify_cluster_kind(canonical),
        ))
    return results
