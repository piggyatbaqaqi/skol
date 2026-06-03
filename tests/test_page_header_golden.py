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
    tp = len(flagged & gt_lines)
    fp = len(flagged - gt_lines)
    fn = len(gt_lines - flagged)
    return {
        'doc_id': doc_id,
        'n_lines': n_lines,
        'gt_header_lines': len(gt_lines),
        'flagged_lines': len(flagged),
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
        gt_total = sum(s['gt_header_lines'] for s in per_doc)
        flagged_total = sum(s['flagged_lines'] for s in per_doc)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )

        # Tee the stats to stdout so they show up under pytest -s.
        print()
        print(f'=== page_header_detector @ {_GOLDEN_DB} '
              f'({len(per_doc)} docs) ===')
        print(f'  ground truth header lines : {gt_total}')
        print(f'  detector-flagged lines    : {flagged_total}')
        print(f'  TP={total_tp}  FP={total_fp}  FN={total_fn}')
        print(f'  precision={precision:.3f}  recall={recall:.3f}  '
              f'F1={f1:.3f}')
        docs_with_sequence = sum(
            1 for s in per_doc if s['sequence_quality'] is not None
        )
        print(f'  docs with a detected page sequence: '
              f'{docs_with_sequence}/{len(per_doc)}')

        assert recall >= 0.10, (
            f'Aggregate recall {recall:.3f} fell below 0.10 — the '
            'detector found essentially nothing across the sample.'
        )
