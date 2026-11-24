"""Tests for fixes/backfill_bhl_ia_provenance.py."""
import base64
import hashlib
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backfill_bhl_ia_provenance import (  # type: ignore[import]  # noqa: E402
    attachment_md5_hex, backfill, bhl_docs, needs_provenance)

PDF = b'%PDF-1.5 a volume'
PDF_MD5 = hashlib.md5(PDF).hexdigest()


def _digest(data: bytes) -> str:
    return 'md5-' + base64.b64encode(hashlib.md5(data).digest()).decode()


def _doc(doc_id: str, **kw: Any) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        '_id': doc_id, 'source': 'bhl',
        'ia_identifier': f'ia-{doc_id}', 'withheld_pages': [],
        '_attachments': {'article.pdf': {'digest': _digest(PDF)}},
    }
    doc.update(kw)
    return doc


META = {'ia_contributor': 'University of California Libraries',
        'ia_scan_date': '2006-11-22T19:44:42', 'ia_pdf_md5': PDF_MD5}


class _Db:
    """Enough of couchdb.Database: find() and save()."""

    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        self.saved: List[Dict[str, Any]] = []
        self.selectors: List[Any] = []

    def find(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Like CouchDB's _find: 25 documents unless 'limit' says
        # otherwise, after 'skip'; couchdb-python drops the bookmark.
        self.selectors.append(query)
        hits = [d for d in self.docs
                if d.get('source') == query['selector'].get('source')]
        skip = query.get('skip', 0)
        return hits[skip:skip + query.get('limit', 25)]

    def save(self, doc: Dict[str, Any]) -> None:
        self.saved.append(dict(doc))


class TestNeedsProvenance(unittest.TestCase):
    def test_bhl_volume_without_fields(self) -> None:
        self.assertTrue(needs_provenance(_doc('a')))

    def test_any_ia_field_means_done(self) -> None:
        self.assertFalse(needs_provenance(_doc('a', ia_scan_date='x')))

    def test_other_sources_and_no_identifier(self) -> None:
        self.assertFalse(needs_provenance(_doc('a', source='pmc')))
        self.assertFalse(needs_provenance(_doc('a', ia_identifier='')))
        doc = _doc('a')
        del doc['ia_identifier']
        self.assertFalse(needs_provenance(doc))


class TestAttachmentMd5(unittest.TestCase):
    def test_couchdb_digest_to_hex(self) -> None:
        self.assertEqual(attachment_md5_hex(_doc('a')), PDF_MD5)

    def test_missing_or_other_digest(self) -> None:
        self.assertIsNone(attachment_md5_hex({'_id': 'a'}))
        self.assertIsNone(attachment_md5_hex(
            {'_attachments': {'article.pdf': {'digest': 'sha-xyz'}}}))
        self.assertIsNone(attachment_md5_hex(
            {'_attachments': {'other.pdf': {'digest': _digest(PDF)}}}))


class TestBhlDocs(unittest.TestCase):
    def test_mango_query_on_source(self) -> None:
        db = _Db([_doc('a'), _doc('b', source='pmc')])
        self.assertEqual([d['_id'] for d in bhl_docs(db)], ['a'])
        self.assertEqual(db.selectors[0]['selector'], {'source': 'bhl'})


class TestBhlDocsPaging(unittest.TestCase):
    """_find returns 25 documents by default and couchdb-python drops
    the bookmark, so bhl_docs must page with limit/skip."""

    @unittest.expectedFailure
    def test_every_document_is_returned_past_one_page(self) -> None:
        db = _Db([_doc(f'd{i:03d}') for i in range(130)])
        self.assertEqual(len(list(bhl_docs(db))), 130)

    @unittest.expectedFailure
    def test_queries_set_an_explicit_limit(self) -> None:
        db = _Db([_doc(f'd{i:03d}') for i in range(3)])
        list(bhl_docs(db))
        self.assertTrue(all('limit' in q for q in db.selectors))

    def test_empty_database(self) -> None:
        self.assertEqual(list(bhl_docs(_Db([]))), [])


class TestBackfill(unittest.TestCase):
    def _run(self, docs: List[Dict[str, Any]], meta: Any = None,
             **kw: Any) -> Any:
        self.db = _Db(docs)
        self.fetch = MagicMock(return_value=META if meta is None else meta)
        return backfill(self.db, self.fetch, verbosity=0, **kw)

    def test_fills_fields_on_eligible_docs_only(self) -> None:
        counts = self._run([_doc('a'), _doc('b', ia_contributor='done'),
                            _doc('c', source='pmc')])
        self.assertEqual(counts['scanned'], 2)
        self.assertEqual(counts['eligible'], 1)
        self.assertEqual(counts['written'], 1)
        self.fetch.assert_called_once_with('ia-a')
        saved = self.db.saved[0]
        self.assertEqual(saved['ia_contributor'],
                         'University of California Libraries')
        self.assertIn('modification_time', saved)

    def test_dry_run_saves_nothing(self) -> None:
        counts = self._run([_doc('a')], dry_run=True)
        self.assertEqual(counts['written'], 1)
        self.assertEqual(self.db.saved, [])

    def test_limit(self) -> None:
        counts = self._run([_doc('a'), _doc('b'), _doc('c')], limit=2)
        self.assertEqual(counts['written'], 2)
        self.assertEqual(len(self.db.saved), 2)

    def test_failed_lookup_leaves_doc_alone(self) -> None:
        counts = self._run([_doc('a')], meta={})
        self.assertEqual(counts['lookup_failed'], 1)
        self.assertEqual(counts['written'], 0)
        self.assertEqual(self.db.saved, [])

    def test_md5_mismatch_is_counted_and_fields_still_written(self) -> None:
        counts = self._run([_doc('a')], meta=dict(META, ia_pdf_md5='0' * 32))
        self.assertEqual(counts['md5_mismatch'], 1)
        self.assertEqual(counts['written'], 1)

    def test_redacted_volume_is_not_compared(self) -> None:
        doc = _doc('a', withheld_pages=[[3, 5]])
        counts = self._run([doc], meta=dict(META, ia_pdf_md5='0' * 32))
        self.assertEqual(counts['md5_mismatch'], 0)
        self.assertEqual(counts['written'], 1)

    def test_second_run_is_a_no_op(self) -> None:
        self._run([_doc('a')])
        rerun = backfill(_Db(self.db.saved), self.fetch, verbosity=0)
        self.assertEqual(rerun['eligible'], 0)


if __name__ == '__main__':
    unittest.main()
