"""Tests for bin/annotate_v4.py — the v4 Step-1 detector orchestrator."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestors.spans import Span  # noqa: E402

from annotate_v4 import (  # type: ignore[import]  # noqa: E402
    _PAGE_HEADERS_ATTACHMENT,
    _SPANS_ATTACHMENT,
    _markers_from_spans,
    _save_attachment,
    annotate_document_v4,
    process_documents_v4,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeDb:
    """Minimal stand-in for ``couchdb.Database``.  Records every
    ``put_attachment`` and ``get_attachment`` call."""

    def __init__(self, docs: Dict[str, Dict[str, Any]]) -> None:
        self.docs = docs
        self.put_calls: List[Dict[str, Any]] = []
        self.get_calls: List[Dict[str, Any]] = []

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.docs

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        return self.docs[doc_id]

    def get_attachment(self, doc_id: str, name: str) -> Optional[bytes]:
        self.get_calls.append({'doc_id': doc_id, 'name': name})
        atts = self.docs.get(doc_id, {}).get('_attachments') or {}
        if name in atts:
            return atts[name].get('data') or b''
        return None

    def put_attachment(
        self, doc: Dict[str, Any], content: bytes,
        filename: str, content_type: str,
    ) -> None:
        self.put_calls.append({
            'doc_id': doc.get('_id'),
            'filename': filename,
            'content_type': content_type,
            'size': len(content),
        })


# ---------------------------------------------------------------------------
# _markers_from_spans
# ---------------------------------------------------------------------------


class TestMarkersFromSpans(unittest.TestCase):
    """Pull (line_index, page_number) tuples out of PDF-page-marker
    particle spans.  This is the bridge between particle_detector
    (commit 237bd7f) and page_header_detector's pdf_page_markers
    parameter (commit c5d8b2c)."""

    def test_extracts_page_number_and_line_index(self):
        # Two markers at known character offsets; expect the
        # converted line indices.
        plaintext = (
            'body line 0\n'                              # line 0
            'body line 1\n'                              # line 1
            '--- PDF Page 2 Label 2 ---\n'               # line 2
            'body line 3\n'                              # line 3
            '--- PDF Page 3 Label 3 ---\n'               # line 4
        )
        # Compute char offsets where the markers start.
        m2_start = plaintext.index('--- PDF Page 2')
        m3_start = plaintext.index('--- PDF Page 3')
        spans = [
            Span(
                start=m2_start, end=m2_start + 26,
                label='PDF-page-marker',
                text='--- PDF Page 2 Label 2 ---',
                source='regex',
                metadata={'page_number': 2, 'label_number': 2},
            ),
            Span(
                start=m3_start, end=m3_start + 26,
                label='PDF-page-marker',
                text='--- PDF Page 3 Label 3 ---',
                source='regex',
                metadata={'page_number': 3, 'label_number': 3},
            ),
        ]
        result = _markers_from_spans(spans, plaintext)
        self.assertEqual(result, [(2, 2), (4, 3)])

    def test_ignores_non_marker_spans(self):
        """Mixed: TaxonName, DOI particle, PDF-page-marker — only
        the marker comes through."""
        plaintext = (
            'Boletus edulis Bull. on page\n'                  # line 0
            '--- PDF Page 2 Label 2 ---\n'                    # line 1
            'doi 10.3897/mycokeys.123\n'                      # line 2
        )
        marker_start = plaintext.index('--- PDF Page 2')
        spans = [
            Span(
                start=0, end=14, label='TaxonName',
                text='Boletus edulis', source='gnfinder',
            ),
            Span(
                start=marker_start, end=marker_start + 26,
                label='PDF-page-marker',
                text='--- PDF Page 2 Label 2 ---',
                source='regex',
                metadata={'page_number': 2},
            ),
            Span(
                start=plaintext.index('10.3897'),
                end=plaintext.index('10.3897') + len('10.3897/mycokeys.123'),
                label='DOI', text='10.3897/mycokeys.123',
                source='regex',
            ),
        ]
        result = _markers_from_spans(spans, plaintext)
        self.assertEqual(result, [(1, 2)])

    def test_handles_missing_page_number_metadata(self):
        """Defensive: a PDF-page-marker span without
        ``page_number`` in metadata is silently skipped (no crash)."""
        plaintext = '--- PDF Page 2 ---\n'
        spans = [
            Span(
                start=0, end=18, label='PDF-page-marker',
                text='--- PDF Page 2 ---', source='regex',
                metadata={},  # no page_number
            ),
        ]
        self.assertEqual(_markers_from_spans(spans, plaintext), [])


# ---------------------------------------------------------------------------
# annotate_document_v4
# ---------------------------------------------------------------------------


class TestAnnotateDocumentV4(unittest.TestCase):
    """The orchestrator returns ``(spans, page_headers_dict)`` after
    running all four Step-1 detectors and threading the
    PDF-page-marker anchors into page_header_detector."""

    def _patch_detectors(
        self,
        find_names_returns: Any = None,
        particle_spans: Optional[List[Span]] = None,
        section_spans: Optional[List[Span]] = None,
        page_headers: Optional[Dict[str, Any]] = None,
    ):
        """Set up a single mock.patch.multiple context.  Each patched
        callable's return value defaults to "empty" so individual
        tests only specify the parts they care about."""
        return mock.patch.multiple(
            'annotate_v4',
            find_names=mock.MagicMock(
                return_value=find_names_returns or [],
            ),
            parse_authorship_after_name=mock.MagicMock(
                return_value=None,
            ),
            detect_particles=mock.MagicMock(
                return_value=particle_spans or [],
            ),
            detect_section_headers=mock.MagicMock(
                return_value=section_spans or [],
            ),
            detect_page_headers=mock.MagicMock(
                return_value=page_headers or {
                    'schema_version': '1', 'n_lines': 0,
                    'regions': [], 'per_line_confidence': [],
                    'sequence_fit': None, 'alternation_score': 0.0,
                },
            ),
        )

    def test_returns_spans_and_page_headers_tuple(self):
        with self._patch_detectors():
            result = annotate_document_v4(
                'plaintext body', None, 'doc-1',
                gnfinder_url='http://x', gnparser_url='http://y',
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        spans, page_headers = result
        self.assertIsInstance(spans, list)
        self.assertIsInstance(page_headers, dict)
        self.assertEqual(page_headers.get('schema_version'), '1')

    def test_marker_handoff_invokes_detector_with_anchors(self):
        """When particle_detector yielded a PDF-page-marker, the
        orchestrator must thread it into ``detect_page_headers`` as
        a ``pdf_page_markers`` anchor — the whole reason 1.D exists."""
        plaintext = (
            'body line 0\n'
            '--- PDF Page 2 Label 2 ---\n'
            'body line 2\n'
        )
        marker_span = Span(
            start=plaintext.index('--- PDF'),
            end=plaintext.index('--- PDF') + 26,
            label='PDF-page-marker',
            text='--- PDF Page 2 Label 2 ---',
            source='regex',
            metadata={'page_number': 2, 'label_number': 2},
        )
        # Capture detect_page_headers separately so we can inspect
        # its call_args after the orchestrator runs (mock.patch.multiple
        # doesn't expose the individual mocks via the context object).
        detect_page_headers_mock = mock.MagicMock(return_value={
            'schema_version': '1', 'n_lines': 3,
            'regions': [], 'per_line_confidence': [0.0, 0.0, 0.0],
            'sequence_fit': None, 'alternation_score': 0.0,
        })
        with mock.patch.multiple(
            'annotate_v4',
            find_names=mock.MagicMock(return_value=[]),
            parse_authorship_after_name=mock.MagicMock(return_value=None),
            detect_particles=mock.MagicMock(return_value=[marker_span]),
            detect_section_headers=mock.MagicMock(return_value=[]),
            detect_page_headers=detect_page_headers_mock,
        ):
            annotate_document_v4(
                plaintext, None, 'doc-1',
                gnfinder_url='http://x', gnparser_url='http://y',
            )
        self.assertEqual(detect_page_headers_mock.call_count, 1)
        call_kwargs = detect_page_headers_mock.call_args.kwargs
        self.assertEqual(call_kwargs.get('pdf_page_markers'), [(1, 2)])

    def test_section_header_spans_join_combined_list(self):
        """A Section-header span from section_header_detector must
        appear in the returned spans list alongside particle / taxon
        spans."""
        section_span = Span(
            start=0, end=12, label='section-header',
            text='Introduction', source='regex',
            metadata={'canonical': 'introduction',
                      'yedda_hint': 'Misc-exposition'},
        )
        with self._patch_detectors(section_spans=[section_span]):
            spans, _ = annotate_document_v4(
                'Introduction\n', None, 'doc-1',
                gnfinder_url='http://x', gnparser_url='http://y',
            )
        labels = [s.label for s in spans]
        self.assertIn('section-header', labels)


# ---------------------------------------------------------------------------
# _save_attachment
# ---------------------------------------------------------------------------


class TestSaveAttachment(unittest.TestCase):
    """JSON attachments must always go in with the v4-specific name
    and ``application/json`` content_type — anything else would mean
    consumers can't reliably find them."""

    def test_put_attachment_called_with_right_name_and_content_type(self):
        db = FakeDb({'doc-1': {'_id': 'doc-1', '_rev': '1-aaa'}})
        body = b'{"hello":"world"}'
        _save_attachment(db, 'doc-1', _SPANS_ATTACHMENT, body)
        self.assertEqual(len(db.put_calls), 1)
        call = db.put_calls[0]
        self.assertEqual(call['filename'], 'article.spans.v4.json')
        self.assertEqual(call['content_type'], 'application/json')
        self.assertEqual(call['size'], len(body))


# ---------------------------------------------------------------------------
# process_documents_v4
# ---------------------------------------------------------------------------


def _doc_with(*attachment_names: str) -> Dict[str, Any]:
    """Helper: build a doc dict with the named attachments
    populated (so ``_spans_attachment_exists`` style checks against
    ``_attachments`` work)."""
    return {
        '_id': 'd1', '_rev': '5-xxxx',
        '_attachments': {
            name: {'data': b'placeholder',
                   'content_type': 'application/json'}
            for name in attachment_names
        },
    }


class TestProcessDocumentsV4(unittest.TestCase):
    """Per-doc loop semantics: skip-when-both-attachments-present,
    force override, dry-run safety."""

    def _patch_annotate(
        self, spans: Optional[List[Span]] = None,
        page_headers: Optional[Dict[str, Any]] = None,
    ):
        return mock.patch(
            'annotate_v4.annotate_document_v4',
            return_value=(spans or [], page_headers or {
                'schema_version': '1', 'n_lines': 0,
                'regions': [], 'per_line_confidence': [],
                'sequence_fit': None, 'alternation_score': 0.0,
            }),
        )

    def test_skip_only_when_both_attachments_present(self):
        """One of the two attachments missing → not skipped."""
        db = FakeDb({
            'd1': _doc_with(_SPANS_ATTACHMENT),  # missing page-headers
        })
        with self._patch_annotate(), \
                mock.patch('annotate_v4._read_attachment_text',
                           return_value='body text'):
            counts = process_documents_v4(
                db, ['d1'],
                skip_existing=True, force=False, dry_run=False,
                gnfinder_url='http://x', gnparser_url='http://y',
            )
        self.assertEqual(counts['skipped'], 0)
        self.assertEqual(counts['processed'], 1)

    def test_skip_when_both_attachments_present(self):
        """Both attachments present → skipped."""
        db = FakeDb({
            'd1': _doc_with(_SPANS_ATTACHMENT, _PAGE_HEADERS_ATTACHMENT),
        })
        with self._patch_annotate():
            counts = process_documents_v4(
                db, ['d1'],
                skip_existing=True, force=False, dry_run=False,
                gnfinder_url='http://x', gnparser_url='http://y',
            )
        self.assertEqual(counts['skipped'], 1)
        self.assertEqual(counts['processed'], 0)
        self.assertEqual(len(db.put_calls), 0)

    def test_force_overrides_skip(self):
        """Both attachments present but force=True → not skipped."""
        db = FakeDb({
            'd1': _doc_with(_SPANS_ATTACHMENT, _PAGE_HEADERS_ATTACHMENT),
        })
        with self._patch_annotate(), \
                mock.patch('annotate_v4._read_attachment_text',
                           return_value='body text'):
            counts = process_documents_v4(
                db, ['d1'],
                skip_existing=True, force=True, dry_run=False,
                gnfinder_url='http://x', gnparser_url='http://y',
            )
        self.assertEqual(counts['skipped'], 0)
        self.assertEqual(counts['processed'], 1)
        # Both attachments rewritten on force.
        filenames = sorted(c['filename'] for c in db.put_calls)
        self.assertEqual(
            filenames,
            sorted([_SPANS_ATTACHMENT, _PAGE_HEADERS_ATTACHMENT]),
        )

    def test_dry_run_no_db_writes(self):
        db = FakeDb({'d1': _doc_with()})
        with self._patch_annotate(), \
                mock.patch('annotate_v4._read_attachment_text',
                           return_value='body text'):
            counts = process_documents_v4(
                db, ['d1'],
                skip_existing=False, force=False, dry_run=True,
                gnfinder_url='http://x', gnparser_url='http://y',
            )
        self.assertEqual(counts['processed'], 1)
        self.assertEqual(len(db.put_calls), 0)


if __name__ == '__main__':
    unittest.main()
