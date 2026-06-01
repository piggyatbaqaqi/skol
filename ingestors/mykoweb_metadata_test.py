"""Tests for ingestors/mykoweb_metadata.py.

Pure helpers consumed by ``LocalMykowebLiteratureIngestor`` to look
up curated bibliographic metadata (journal / volume / issue / pages
/ author / year / human-edited title) by the relative path of the
ingested PDF.

The JSON file these helpers read is produced by
``bin/extract_mykoweb_pdf_metadata.py`` — see
docs/mykoweb_pdf_metadata_extraction.md.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.mykoweb_metadata import (  # type: ignore[import]
    load_metadata_index,
    lookup_pdf_metadata,
    metadata_to_doc_fields,
)


# ---------------------------------------------------------------------------
# load_metadata_index — JSON → dict, missing-file tolerant
# ---------------------------------------------------------------------------


class TestLoadMetadataIndex(unittest.TestCase):
    """The ingestor calls ``load_metadata_index(path)`` at init time
    to get a dict it can hand to ``lookup_pdf_metadata``.  A missing
    file must return an empty dict so the ingestor can fall back to
    its pre-integration behaviour (filename-as-title / itemtype=book)
    rather than crashing."""

    def test_existing_file_loaded(self):
        with self.subTest('load returns the dict'):
            tmp = Path('/tmp/mykoweb_metadata_test.json')
            tmp.write_text(json.dumps({
                'systematics/literature/foo.pdf': {'kind': 'book'},
            }))
            try:
                index = load_metadata_index(tmp)
                self.assertEqual(index['systematics/literature/foo.pdf'],
                                 {'kind': 'book'})
            finally:
                tmp.unlink()

    def test_missing_file_returns_empty_dict(self):
        index = load_metadata_index(Path('/tmp/does_not_exist_xyz.json'))
        self.assertEqual(index, {})

    def test_none_path_returns_empty_dict(self):
        """Caller may pass None to signal 'no metadata file
        configured' — must yield an empty index, not crash."""
        self.assertEqual(load_metadata_index(None), {})


# ---------------------------------------------------------------------------
# lookup_pdf_metadata — filesystem path → metadata record
# ---------------------------------------------------------------------------


class TestLookupPdfMetadata(unittest.TestCase):
    """``lookup_pdf_metadata(full_path, index, site_root)`` converts a
    walked filesystem path into the relative-to-site-root key used
    in the JSON index, then looks it up."""

    def _index(self) -> Dict[str, Dict[str, Any]]:
        return {
            'systematics/literature/clavaria.pdf':       {'kind': 'journal_article'},
            'systematics/literature/a_book.pdf':         {'kind': 'book'},
            'systematics/journals/Mycotaxon/v001n1.pdf': {'kind': 'journal'},
        }

    def test_literature_pdf_found(self):
        result = lookup_pdf_metadata(
            '/data/skol/www/mykoweb.com/systematics/literature/clavaria.pdf',
            self._index(),
            site_root='/data/skol/www/mykoweb.com',
        )
        assert result is not None
        self.assertEqual(result['kind'], 'journal_article')

    def test_pdf_not_in_index_returns_none(self):
        """A PDF that exists on disk but isn't in the JSON (e.g.
        added after the last extraction run) returns None — the
        caller falls back to its pre-integration behaviour."""
        result = lookup_pdf_metadata(
            '/data/skol/www/mykoweb.com/systematics/literature/new.pdf',
            self._index(),
            site_root='/data/skol/www/mykoweb.com',
        )
        self.assertIsNone(result)

    def test_path_outside_site_root_returns_none(self):
        """Defensive: a path that doesn't live under ``site_root``
        can't have a sensible lookup key."""
        result = lookup_pdf_metadata(
            '/somewhere/else/foo.pdf',
            self._index(),
            site_root='/data/skol/www/mykoweb.com',
        )
        self.assertIsNone(result)

    def test_trailing_slash_in_site_root_tolerated(self):
        """``site_root`` may or may not carry a trailing slash; both
        forms must yield the same lookup key."""
        result = lookup_pdf_metadata(
            '/data/skol/www/mykoweb.com/systematics/literature/clavaria.pdf',
            self._index(),
            site_root='/data/skol/www/mykoweb.com/',
        )
        assert result is not None
        self.assertEqual(result['kind'], 'journal_article')

    def test_empty_index_returns_none(self):
        result = lookup_pdf_metadata(
            '/data/skol/www/mykoweb.com/systematics/literature/x.pdf',
            {},
            site_root='/data/skol/www/mykoweb.com',
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# metadata_to_doc_fields — record → fields to merge into doc_dict
# ---------------------------------------------------------------------------


class TestMetadataToDocFields(unittest.TestCase):
    """Translate a metadata record (one value from the JSON index)
    into a dict of fields the ingestor merges into its outgoing
    ``doc_dict``.  Only the fields that *have* a value are included
    so we don't pollute the doc with null keys."""

    def test_journal_article_full(self):
        """Most enriched case: every bibliographic field populated."""
        record = {
            'kind':            'journal_article',
            'title':           'The North American Species of Clavaria',
            'container_title': 'Annals of the Missouri Botanical Garden',
            'volume':          '9',
            'issue':           '1',
            'pages':           '1-78',
            'author':          'Burt, E.A.',
            'year':            1922,
            'source_html':     'systematics/literature.html',
        }
        fields = metadata_to_doc_fields(record)
        self.assertEqual(fields['itemtype'], 'article')
        self.assertEqual(fields['journal'],
                         'Annals of the Missouri Botanical Garden')
        self.assertEqual(fields['volume'], '9')
        self.assertEqual(fields['issue'], '1')
        self.assertEqual(fields['pages'], '1-78')
        self.assertEqual(fields['title'],
                         'The North American Species of Clavaria')
        self.assertEqual(fields['author'], 'Burt, E.A.')
        self.assertEqual(fields['year'], 1922)

    def test_journal_article_without_issue(self):
        """Many real mykoweb citations have vol+pages but no issue
        — the issue field must not be set when None."""
        record = {
            'kind':            'journal_article',
            'title':           'Some Article',
            'container_title': 'Some Journal',
            'volume':          '1',
            'issue':           None,
            'pages':           '1-10',
        }
        fields = metadata_to_doc_fields(record)
        self.assertNotIn('issue', fields)
        self.assertEqual(fields['volume'], '1')

    def test_book_record(self):
        """Books: itemtype='book', title and author/year retained,
        but no journal/volume/issue/pages on the doc."""
        record = {
            'kind':            'book',
            'title':           'A Monograph of Favolaschia',
            'container_title': None,
            'author':          'Singer, R.',
            'year':            1974,
        }
        fields = metadata_to_doc_fields(record)
        self.assertEqual(fields['itemtype'], 'book')
        self.assertEqual(fields['title'], 'A Monograph of Favolaschia')
        self.assertEqual(fields['author'], 'Singer, R.')
        self.assertEqual(fields['year'], 1974)
        for k in ('journal', 'volume', 'issue', 'pages'):
            self.assertNotIn(k, fields)

    def test_book_with_no_author_year(self):
        """A book whose row had no <strong> author and no (YYYY)
        — itemtype + title only."""
        record = {
            'kind':  'book',
            'title': 'Untitled Monograph',
        }
        fields = metadata_to_doc_fields(record)
        self.assertEqual(fields['itemtype'], 'book')
        self.assertEqual(fields['title'], 'Untitled Monograph')
        self.assertNotIn('author', fields)
        self.assertNotIn('year', fields)

    def test_journal_kind_returns_empty(self):
        """``kind=journal`` (from journals.html) is handled by the
        journals ingestor, not the literature ingestor — return
        empty so this lookup path doesn't accidentally rewrite a
        journal-volume doc."""
        self.assertEqual(
            metadata_to_doc_fields(
                {'kind': 'journal', 'title': 'Mycotaxon Vol. 1'},
            ),
            {},
        )

    def test_misc_kind_emits_itemtype_and_journal(self):
        """``kind=misc`` records (CAF protologue, misc/ loose PDFs)
        used to return empty — leaving 307+ CAF/protologue docs in
        the Unknown bucket on the Sources page.  Now they emit
        ``itemtype=misc`` and propagate ``container_title`` →
        ``journal`` so they roll up under their collection name."""
        record = {
            'kind':            'misc',
            'title':           'Agaricus albissimus',
            'container_title': 'California Fungi',
        }
        fields = metadata_to_doc_fields(record)
        self.assertEqual(fields['itemtype'], 'misc')
        self.assertEqual(fields['journal'], 'California Fungi')
        self.assertEqual(fields['title'], 'Agaricus albissimus')

    def test_key_kind_emits_itemtype(self):
        """``kind=key`` (taxonomic-key PDFs) emits ``itemtype=key``
        and a journal field when container_title is set."""
        record = {
            'kind':            'key',
            'title':           'Arcangeliella key',
            'container_title': 'California Fungi',
        }
        fields = metadata_to_doc_fields(record)
        self.assertEqual(fields['itemtype'], 'key')
        self.assertEqual(fields['journal'], 'California Fungi')

    def test_book_with_container_now_emits_journal(self):
        """A book record with a container_title (e.g. a chapter of
        the *Funga Nordica* reference book) used to drop the
        container — now it propagates as ``journal`` so the book
        rolls up under its parent collection on the Sources page."""
        record = {
            'kind':            'book',
            'title':           'Russula',
            'container_title': 'Funga Nordica',
        }
        fields = metadata_to_doc_fields(record)
        self.assertEqual(fields['itemtype'], 'book')
        self.assertEqual(fields['journal'], 'Funga Nordica')
        self.assertEqual(fields['title'], 'Russula')

    def test_unknown_kind_returns_empty(self):
        """Defensive: a future ``kind`` value we don't know about
        produces no fields, not a crash."""
        self.assertEqual(
            metadata_to_doc_fields({'kind': 'mystery', 'title': 'X'}),
            {},
        )

    def test_empty_author_string_not_emitted(self):
        """Authors come from the page's ``<strong>`` / ``<b>``; if
        the row had no bold (empty string), don't emit a blank
        author field on the doc."""
        record = {
            'kind':            'journal_article',
            'title':           'X',
            'container_title': 'J',
            'volume':          '1',
            'pages':           '1-2',
            'author':          '',
        }
        fields = metadata_to_doc_fields(record)
        self.assertNotIn('author', fields)

    def test_journal_article_without_container_no_journal_field(self):
        """Defensive: if container_title is None on a journal_article
        record (shouldn't happen post-extractor, but if upstream ever
        emits one) we must not set ``journal`` to None on the doc."""
        record = {
            'kind':            'journal_article',
            'title':           'X',
            'container_title': None,
            'volume':          '1',
            'pages':           '1-2',
        }
        fields = metadata_to_doc_fields(record)
        self.assertNotIn('journal', fields)
        self.assertEqual(fields['itemtype'], 'article')


if __name__ == '__main__':
    unittest.main()
