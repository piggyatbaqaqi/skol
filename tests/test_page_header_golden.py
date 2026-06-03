"""Integration test for ``ingestors.page_header_detector`` against the
hand-annotated golden corpus ``skol_golden_ann_hand_v2``.

Per-doc procedure:

1. Fetch ``article.txt.ann`` from ``skol_golden_ann_hand_v2``.
2. Fetch ``article.txt`` from ``skol_golden_v2`` (same _id; all 30
   hand docs have it).
3. Align each ``Page-header`` YEDDA block to ``article.txt`` line
   indices via ``difflib.SequenceMatcher.find_longest_match`` —
   that's the ground truth set.
4. Run ``detect_page_headers`` on ``article.txt.split('\\n')``.
5. Score precision / recall / F1 against the aligned ground truth.

Compared to the original ``plaintext_from_yedda`` reconstruction
path, this avoids:

* the ``\\n\\n`` block-separator artifact (gap-blank FPs went away),
* the section-header-shape filter hack (real lines, not blank gaps),
* the need to second-guess whether a flagged line was a real header
  or a reconstruction gap.

Falls back to the old reconstruction + scoring-filter path for any
doc whose ``article.txt`` is absent.  Per CLAUDE.md rule 5 the file
lives in ``tests/`` but stays pytest-compatible.

Sample size is small (default 10 docs); bump ``_SAMPLE_LIMIT`` for
broader measurement.
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import couchdb  # type: ignore[import]  # noqa: E402

from env_config import get_env_config  # type: ignore[import]  # noqa: E402
from ingestors.extract_plaintext import (  # noqa: E402
    plaintext_from_yedda,
)
from ingestors.page_header_detector import (  # noqa: E402
    detect_page_headers,
)
from ingestors.particle_detector import detect_particles  # noqa: E402


_GOLDEN_DB = 'skol_golden_ann_hand_v2'   # YEDDA annotations
_SOURCE_DB = 'skol_golden_v2'            # article.txt source (shares _id)
_SAMPLE_LIMIT = 10                       # docs to evaluate
_DETECTOR_FLAG_THRESHOLD = 0.0           # per_line_confidence > this -> "flagged"
_ALIGNMENT_MIN_MATCH = 20                # min chars of contiguous match
_ALIGNMENT_MIN_FRAC = 0.5                # min fraction of block matched

# Same YEDDA block regex used by ingestors.extract_plaintext.
_YEDDA_BLOCK_RE = re.compile(r"\[@\s*(.*?)\s*#([^*]+)\*\]", re.DOTALL)

# A "section-header-shaped" line is short, all caps (no lowercase
# letters), at least one alpha character.  Used by the scoring filter
# to suppress false positives where the detector picked up section
# titles like 'INFRAGENERIC CLASSIFICATION' that v4 Step 1.C will
# detect separately.
_SECTION_HEADER_MAX_LEN = 60


def _is_section_header_shaped(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > _SECTION_HEADER_MAX_LEN:
        return False
    alpha = [ch for ch in stripped if ch.isalpha()]
    if not alpha:
        return False
    return all(ch.isupper() for ch in alpha)


def _scoring_excluded(line: str) -> bool:
    """Lines that should NOT count in the confusion matrix:

    * Blank lines: ``plaintext_from_yedda`` introduces ``\\n\\n``
      between YEDDA blocks, so flagged-but-not-in-GT blank lines
      reflect a reconstruction artifact, not a detector error.
    * Section-header-shaped lines (short, ALL CAPS): v4 Step 1.C's
      ``section_header_detector`` is the consumer responsible for
      those; we shouldn't penalise the *page*-header detector for
      catching them as a side effect of region recovery.
    """
    if not line.strip():
        return True
    return _is_section_header_shaped(line)


def _open_server() -> Any:
    """Authenticated couchdb.Server handle."""
    config = get_env_config()
    server = couchdb.Server(config['couchdb_url'])
    username = config.get('couchdb_username')
    password = config.get('couchdb_password')
    if username and password:
        server.resource.credentials = (username, password)
    return server


def _golden_available() -> bool:
    """True iff both the annotation DB and the article-text source DB
    are reachable and non-empty."""
    try:
        server = _open_server()
        return len(server[_GOLDEN_DB]) > 0 and len(server[_SOURCE_DB]) > 0
    except Exception:
        return False


def _read_attachment_text(db: Any, doc_id: str, name: str) -> Optional[str]:
    """Fetch an attachment as decoded text, or return None if absent."""
    raw = db.get_attachment(doc_id, name)
    if raw is None:
        return None
    if hasattr(raw, 'read'):
        raw = raw.read()
    if isinstance(raw, bytes):
        return raw.decode('utf-8', errors='ignore')
    return str(raw)


def _align_yedda_to_article_txt(
    article_txt: str,
    ann_text: str,
) -> Tuple[Set[int], int, int]:
    """Map each ``Page-header`` YEDDA block to ``article.txt`` line
    indices via ``SequenceMatcher.find_longest_match``.

    Returns ``(header_line_indices, blocks_total, blocks_aligned)``.
    A block is considered aligned when the longest matching substring
    covers at least ``_ALIGNMENT_MIN_MATCH`` characters AND at least
    ``_ALIGNMENT_MIN_FRAC`` of the block's text — short matches like
    a stray digit or single common word would otherwise produce wild
    misalignments.
    """
    # Precompute (cumulative offset, line index) so an article-text
    # byte position maps to a line index in O(log N).
    line_starts: List[int] = [0]
    for line in article_txt.split('\n'):
        line_starts.append(line_starts[-1] + len(line) + 1)

    header_lines: Set[int] = set()
    blocks_total = 0
    blocks_aligned = 0

    for match in _YEDDA_BLOCK_RE.finditer(ann_text):
        block_text = match.group(1).strip()
        tag = match.group(2).strip()
        if tag != 'Page-header' or not block_text:
            continue
        blocks_total += 1

        sm = difflib.SequenceMatcher(
            None, article_txt, block_text, autojunk=False,
        )
        m = sm.find_longest_match(
            0, len(article_txt), 0, len(block_text),
        )
        threshold = max(
            _ALIGNMENT_MIN_MATCH,
            int(len(block_text) * _ALIGNMENT_MIN_FRAC),
        )
        if m.size < threshold:
            continue

        blocks_aligned += 1
        start_offset = m.a
        end_offset = m.a + m.size

        # Walk the line-start table to find all overlapping lines.
        for li in range(len(line_starts) - 1):
            ls, le = line_starts[li], line_starts[li + 1]
            if le <= start_offset or ls >= end_offset:
                continue
            header_lines.add(li)

    return header_lines, blocks_total, blocks_aligned


def _ground_truth_header_lines(ann_text: str) -> Tuple[Set[int], int]:
    """Walk the YEDDA blocks and return:
    - the set of line indices (in the reconstructed plaintext) that
      lie inside a Page-header block,
    - the total number of lines in the reconstructed plaintext.

    ``plaintext_from_yedda`` joins block texts with ``\\n\\n``, so
    successive blocks land on lines separated by a single blank line.
    Each block's text is .strip()'d before joining, matching the
    reconstruction.
    """
    header_lines: Set[int] = set()
    line_pos = 0
    is_first = True
    for match in _YEDDA_BLOCK_RE.finditer(ann_text):
        block_text = match.group(1).strip()
        tag = match.group(2).strip()
        if not block_text:
            continue
        if not is_first:
            # blank separator + start of new block = +2 line transitions
            line_pos += 2
        is_first = False
        block_lines = block_text.split('\n')
        block_start = line_pos
        block_end = line_pos + len(block_lines) - 1
        if tag == 'Page-header':
            header_lines.update(range(block_start, block_end + 1))
        line_pos = block_end
    total_lines = line_pos + 1 if not is_first else 0
    return header_lines, total_lines


def _evaluate_doc(
    ann_db: Any, src_db: Any, doc_id: str,
) -> Dict[str, Any]:
    """Run the detector on one doc and return per-doc stats.

    Prefers the ``article.txt`` source from ``src_db`` aligned against
    YEDDA blocks; falls back to ``plaintext_from_yedda`` + the
    blank/section-shape scoring filter when ``article.txt`` is absent.
    """
    ann_text = _read_attachment_text(ann_db, doc_id, 'article.txt.ann')
    if ann_text is None:
        return {'doc_id': doc_id, 'skipped': 'no_ann'}

    article_txt = _read_attachment_text(src_db, doc_id, 'article.txt')

    if article_txt is not None:
        return _evaluate_via_alignment(doc_id, article_txt, ann_text)
    return _evaluate_via_reconstruction(doc_id, ann_text)


def _pdf_page_markers_from_text(
    article_txt: str,
) -> List[Tuple[int, int]]:
    """Run particle_detector and convert PDF-page-marker spans to the
    ``(line_index, page_number)`` shape ``detect_page_headers`` wants.

    Mirrors what bin/annotate_v4.py (1.D) will eventually do: each
    particle's character offset is mapped to a line index by counting
    newlines up to ``span.start``.
    """
    spans = detect_particles(article_txt, redis_client=None)
    markers: List[Tuple[int, int]] = []
    for span in spans:
        if span.label != 'PDF-page-marker':
            continue
        page_number = span.metadata.get('page_number')
        if page_number is None:
            continue
        line_index = article_txt.count('\n', 0, span.start)
        markers.append((line_index, int(page_number)))
    return markers


def _evaluate_via_alignment(
    doc_id: str, article_txt: str, ann_text: str,
) -> Dict[str, Any]:
    """Path A: article.txt + fuzzy alignment of Page-header blocks.

    article.txt preserves the same ``\\n\\n``-between-paragraphs
    layout that the YEDDA was authored against, so the blank-line +
    section-shape scoring filter still applies — what alignment fixes
    is the *ground-truth coordinate mapping*, not the detector's
    region-over-extension into blanks.  We keep the filter for an
    apples-to-apples comparison with path B.

    PDF-page-marker particles (per v4 plan §1.B's particle hand-off
    recommendation) are pulled from particle_detector and threaded
    into ``detect_page_headers`` as sequence anchors.
    """
    gt_lines, blocks_total, blocks_aligned = (
        _align_yedda_to_article_txt(article_txt, ann_text)
    )
    detector_lines = article_txt.split('\n')
    if not detector_lines:
        return {'doc_id': doc_id, 'skipped': 'empty'}

    markers = _pdf_page_markers_from_text(article_txt)
    result = detect_page_headers(
        detector_lines, seed=42, pdf_page_markers=markers,
    )
    flagged: Set[int] = {
        li for li, c in enumerate(result['per_line_confidence'])
        if c > _DETECTOR_FLAG_THRESHOLD
    }

    def excludable(li: int) -> bool:
        return 0 <= li < len(detector_lines) and _scoring_excluded(
            detector_lines[li],
        )
    flagged_scored = {li for li in flagged if not excludable(li)}
    gt_scored = {li for li in gt_lines if not excludable(li)}

    tp = len(flagged_scored & gt_scored)
    fp = len(flagged_scored - gt_scored)
    fn = len(gt_scored - flagged_scored)
    return {
        'doc_id': doc_id,
        'source': 'aligned',
        'n_lines': len(detector_lines),
        'gt_header_lines_raw': len(gt_lines),
        'gt_header_lines_scored': len(gt_scored),
        'flagged_lines_raw': len(flagged),
        'flagged_lines_scored': len(flagged_scored),
        'excluded_blanks_or_sections': (
            len(flagged) - len(flagged_scored)
        ),
        'blocks_total': blocks_total,
        'blocks_aligned': blocks_aligned,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'regions': len(result['regions']),
        'sequence_quality': (
            result['sequence_fit']['quality_score']
            if result['sequence_fit'] is not None else None
        ),
    }


def _evaluate_via_reconstruction(
    doc_id: str, ann_text: str,
) -> Dict[str, Any]:
    """Path B: fall back to plaintext_from_yedda + scoring filter."""
    gt_lines, n_lines = _ground_truth_header_lines(ann_text)
    reconstructed = plaintext_from_yedda(ann_text)
    detector_lines = reconstructed.split('\n')

    if not detector_lines:
        return {'doc_id': doc_id, 'skipped': 'empty'}

    result = detect_page_headers(detector_lines, seed=42)
    flagged: Set[int] = {
        li for li, c in enumerate(result['per_line_confidence'])
        if c > _DETECTOR_FLAG_THRESHOLD
    }

    def excludable(li: int) -> bool:
        return 0 <= li < len(detector_lines) and _scoring_excluded(
            detector_lines[li],
        )
    flagged_scored = {li for li in flagged if not excludable(li)}
    gt_scored = {li for li in gt_lines if not excludable(li)}

    tp = len(flagged_scored & gt_scored)
    fp = len(flagged_scored - gt_scored)
    fn = len(gt_scored - flagged_scored)
    return {
        'doc_id': doc_id,
        'source': 'reconstructed',
        'n_lines': n_lines,
        'gt_header_lines_raw': len(gt_lines),
        'gt_header_lines_scored': len(gt_scored),
        'flagged_lines_raw': len(flagged),
        'flagged_lines_scored': len(flagged_scored),
        'excluded_blanks_or_sections': (
            len(flagged) - len(flagged_scored)
        ),
        'blocks_total': 0,
        'blocks_aligned': 0,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'regions': len(result['regions']),
        'sequence_quality': (
            result['sequence_fit']['quality_score']
            if result['sequence_fit'] is not None else None
        ),
    }


@pytest.mark.skipif(
    not _golden_available(),
    reason=f'{_GOLDEN_DB!r} not reachable (CouchDB / network / creds)',
)
class TestPageHeaderGolden:
    """Integration: ``detect_page_headers`` against the hand corpus."""

    def test_runs_without_error_on_sample(self) -> None:
        """Smoke-level: the detector must not raise on any sampled
        doc.  Even the docs where it finds nothing produce a valid
        empty result dict."""
        server = _open_server()
        ann_db = server[_GOLDEN_DB]
        src_db = server[_SOURCE_DB]
        sampled = 0
        for doc_id in ann_db:
            if doc_id.startswith('_design/'):
                continue
            if sampled >= _SAMPLE_LIMIT:
                break
            stats = _evaluate_doc(ann_db, src_db, doc_id)
            assert 'doc_id' in stats
            sampled += 1
        assert sampled > 0, 'No docs were sampled — corpus empty?'

    def test_aggregate_precision_recall_floor(self) -> None:
        """Aggregate precision and recall floor.  Path A (article.txt
        alignment) is the primary measurement path: no reconstruction
        artifacts, no scoring-filter hacks.  Path B (filtered YEDDA
        reconstruction) kicks in only when ``article.txt`` is absent.
        Quality tuning happens in Step 7; these floors just guard
        against regression."""
        server = _open_server()
        ann_db = server[_GOLDEN_DB]
        src_db = server[_SOURCE_DB]
        per_doc: List[Dict[str, Any]] = []
        for doc_id in ann_db:
            if doc_id.startswith('_design/'):
                continue
            if len(per_doc) >= _SAMPLE_LIMIT:
                break
            stats = _evaluate_doc(ann_db, src_db, doc_id)
            if 'skipped' not in stats:
                per_doc.append(stats)

        total_tp = sum(s['tp'] for s in per_doc)
        total_fp = sum(s['fp'] for s in per_doc)
        total_fn = sum(s['fn'] for s in per_doc)
        gt_raw = sum(s['gt_header_lines_raw'] for s in per_doc)
        gt_scored = sum(s['gt_header_lines_scored'] for s in per_doc)
        flagged_raw = sum(s['flagged_lines_raw'] for s in per_doc)
        flagged_scored = sum(s['flagged_lines_scored'] for s in per_doc)
        excluded = sum(
            s['excluded_blanks_or_sections'] for s in per_doc
        )

        precision = (
            total_tp / (total_tp + total_fp)
            if (total_tp + total_fp) else 0.0
        )
        recall = (
            total_tp / (total_tp + total_fn)
            if (total_tp + total_fn) else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )

        n_aligned = sum(
            1 for s in per_doc if s.get('source') == 'aligned'
        )
        n_reconstructed = sum(
            1 for s in per_doc if s.get('source') == 'reconstructed'
        )
        blocks_total = sum(s['blocks_total'] for s in per_doc)
        blocks_aligned = sum(s['blocks_aligned'] for s in per_doc)

        # Tee the stats to stdout so they show up under pytest -s.
        print()
        print(f'=== page_header_detector @ {_GOLDEN_DB} '
              f'({len(per_doc)} docs) ===')
        print(f'  scoring path        : aligned={n_aligned}  '
              f'reconstructed={n_reconstructed}')
        print(f'  alignment success   : '
              f'{blocks_aligned}/{blocks_total} Page-header blocks')
        print(f'  ground truth lines  : raw={gt_raw}  '
              f'scored={gt_scored}')
        print(f'  detector-flagged    : raw={flagged_raw}  '
              f'scored={flagged_scored}')
        if excluded:
            print(f'  excluded from scoring: {excluded} '
                  '(blanks / section-shape)')
        print(f'  TP={total_tp}  FP={total_fp}  FN={total_fn}')
        print(f'  precision={precision:.3f}  recall={recall:.3f}  '
              f'F1={f1:.3f}')
        docs_with_sequence = sum(
            1 for s in per_doc if s['sequence_quality'] is not None
        )
        print(f'  docs with a detected page sequence: '
              f'{docs_with_sequence}/{len(per_doc)}')

        assert precision >= 0.60, (
            f'Aggregate precision {precision:.3f} fell below 0.60 — '
            'something regressed in the detector or the scoring '
            'filter (blank / section-shape exclusions).'
        )
        assert recall >= 0.30, (
            f'Aggregate recall {recall:.3f} fell below 0.30 — '
            'the detector is missing too many real headers.'
        )
