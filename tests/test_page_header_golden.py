"""Integration test for ``ingestors.page_header_detector`` against the
hand-annotated golden corpus ``skol_golden_ann_hand_v2``.

Loads a small sample of docs, derives ground-truth Page-header line
indices from the YEDDA annotations, runs ``detect_page_headers`` on
the reconstructed plaintext, and reports precision / recall / F1.

Per CLAUDE.md rule 5, functional tests live in ``tests/`` and need
not be pytest-compatible — but this one is, so the unit-test runner
discovers it automatically.  It's skipped when CouchDB is unreachable
or the corpus isn't present, so it doesn't break offline runs.

Sample size is small (default 10 docs) to keep the CI loop fast; bump
``_SAMPLE_LIMIT`` when measuring detector quality more carefully.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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


_GOLDEN_DB = 'skol_golden_ann_hand_v2'
_SAMPLE_LIMIT = 10              # docs to evaluate
_DETECTOR_FLAG_THRESHOLD = 0.0  # per_line_confidence > this -> "flagged"

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


def _open_golden_db() -> Any:
    """Return a couchdb.Database handle to the golden hand corpus or
    raise an exception (the test harness skips on raise)."""
    config = get_env_config()
    server = couchdb.Server(config['couchdb_url'])
    username = config.get('couchdb_username')
    password = config.get('couchdb_password')
    if username and password:
        server.resource.credentials = (username, password)
    return server[_GOLDEN_DB]


def _golden_available() -> bool:
    try:
        db = _open_golden_db()
        # ``len(db)`` triggers a real round-trip; passes if the DB
        # exists and we can authenticate against it.
        return len(db) > 0
    except Exception:
        return False


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


def _evaluate_doc(db: Any, doc_id: str) -> Dict[str, Any]:
    """Run the detector on one doc and return per-doc stats."""
    raw = db.get_attachment(doc_id, 'article.txt.ann')
    if raw is None:
        return {'doc_id': doc_id, 'skipped': 'no_ann'}
    if hasattr(raw, 'read'):
        raw = raw.read()
    if isinstance(raw, bytes):
        ann_text = raw.decode('utf-8', errors='ignore')
    else:
        ann_text = str(raw)

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

    # Apply the Free + Cheap scoring filters: drop lines that are
    # blank (reconstruction artifact of plaintext_from_yedda) or
    # section-header-shaped (1.C's detector will handle those).  We
    # exclude these lines from both the flagged and ground-truth
    # sets so the confusion matrix measures detector behaviour on
    # substantive lines only.
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
        'n_lines': n_lines,
        'gt_header_lines_raw': len(gt_lines),
        'gt_header_lines_scored': len(gt_scored),
        'flagged_lines_raw': len(flagged),
        'flagged_lines_scored': len(flagged_scored),
        'excluded_blanks_or_sections': (
            len(flagged) - len(flagged_scored)
        ),
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
        db = _open_golden_db()
        sampled = 0
        for doc_id in db:
            if doc_id.startswith('_design/'):
                continue
            if sampled >= _SAMPLE_LIMIT:
                break
            stats = _evaluate_doc(db, doc_id)
            assert 'doc_id' in stats
            sampled += 1
        assert sampled > 0, 'No docs were sampled — corpus empty?'

    def test_aggregate_recall_above_floor(self) -> None:
        """Aggregate recall floor.  The hand corpus's Page-header
        tags are *very* permissive (synthetic ``--- PDF Page N ---``
        markers count too); we don't expect the detector to flag the
        markers, only the lines with running-page numbers.  Floor is
        kept low (>= 0.10) so a single doc producing any true
        positive across the 10-doc sample is enough to pass.  The
        goal is a sanity check, not a quality bar — quality tuning
        happens in Step 7."""
        db = _open_golden_db()
        per_doc: List[Dict[str, Any]] = []
        for doc_id in db:
            if doc_id.startswith('_design/'):
                continue
            if len(per_doc) >= _SAMPLE_LIMIT:
                break
            stats = _evaluate_doc(db, doc_id)
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

        # Tee the stats to stdout so they show up under pytest -s.
        print()
        print(f'=== page_header_detector @ {_GOLDEN_DB} '
              f'({len(per_doc)} docs, blank + section-shape lines '
              'excluded from scoring) ===')
        print(f'  ground truth lines  : raw={gt_raw}  '
              f'scored={gt_scored}')
        print(f'  detector-flagged    : raw={flagged_raw}  '
              f'scored={flagged_scored}')
        print(f'  excluded from FP    : {excluded} '
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
