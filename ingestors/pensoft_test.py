"""
Unit tests for the PensoftIngestor class.

Tests XML URL construction, JATS format detection, and the
download_pdf / download_xml flag behavior.
"""

import unittest
from io import BytesIO
from typing import Dict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
from uuid import uuid5, NAMESPACE_URL

from bs4 import BeautifulSoup

from .pensoft import PensoftIngestor


def _make_ingestor(**overrides):
    """Create a PensoftIngestor with mocked dependencies."""
    db = overrides.pop('db', MagicMock())
    robot_parser = overrides.pop('robot_parser', MagicMock())
    robot_parser.can_fetch.return_value = True
    defaults = dict(
        db=db,
        user_agent='test-agent',
        robot_parser=robot_parser,
        verbosity=0,
        journal_name='mycokeys',
        journal_id='11',
    )
    defaults.update(overrides)
    return PensoftIngestor(**defaults)


# Minimal article div HTML used by _extract_articles_from_issue_page.
_ARTICLE_HTML = """
<div class="article">
  <div class="articleHeadline">
    <a href="/article/12345/">A new species of Foo</a>
  </div>
  <div class="ArtDoi">
    <a href="https://doi.org/10.3897/mycokeys.100.12345">10.3897/mycokeys.100.12345</a>
  </div>
  <div class="DoiRow">Published: 15-06-2025</div>
  <div class="DownLink">
    <a href="/article/12345/download/pdf/67890">Download PDF</a>
  </div>
</div>
"""


class TestXmlUrlConstruction(unittest.TestCase):
    """Test that XML URLs are constructed from the article ID."""

    def test_xml_url_from_article(self):
        ing = _make_ingestor()
        soup = BeautifulSoup(_ARTICLE_HTML, 'html.parser')
        articles = ing._extract_articles_from_issue_page(soup)
        self.assertEqual(len(articles), 1)
        art = articles[0]
        self.assertEqual(
            art['xml_url'],
            'https://mycokeys.pensoft.net/article/12345/download/xml/',
        )
        # PDF URL should also be present
        self.assertIn('pdf_url', art)
        self.assertIn('/download/pdf/67890', art['pdf_url'])

    def test_xml_url_with_different_journal(self):
        ing = _make_ingestor(journal_name='imafungus')
        soup = BeautifulSoup(_ARTICLE_HTML, 'html.parser')
        articles = ing._extract_articles_from_issue_page(soup)
        self.assertEqual(
            articles[0]['xml_url'],
            'https://imafungus.pensoft.net/article/12345/download/xml/',
        )


class TestJatsDetection(unittest.TestCase):
    """Test _detect_xml_format for various XML headers."""

    def setUp(self):
        self.ing = _make_ingestor()

    def test_jats_keyword_in_doctype(self):
        xml = b'<?xml version="1.0"?>\n<!DOCTYPE article PUBLIC "-//NLM//DTD JATS v1.2//EN">\n<article>...'
        self.assertEqual(self.ing._detect_xml_format(xml), 'jats')

    def test_journalpublishing_dtd(self):
        xml = b'<?xml version="1.0"?>\n<!DOCTYPE article PUBLIC "-//NLM//DTD JournalPublishing v3.0//EN">\n<article>...'
        self.assertEqual(self.ing._detect_xml_format(xml), 'jats')

    def test_article_with_xmlns(self):
        xml = b'<?xml version="1.0"?>\n<article xmlns="http://example.org/schema">\n<front>...'
        self.assertEqual(self.ing._detect_xml_format(xml), 'jats')

    def test_article_with_dtd(self):
        xml = b'<?xml version="1.0"?>\n<!DOCTYPE article SYSTEM "article.dtd">\n<article>...'
        self.assertEqual(self.ing._detect_xml_format(xml), 'jats')

    def test_non_jats_xml(self):
        xml = b'<?xml version="1.0"?>\n<root><data>hello</data></root>'
        self.assertIsNone(self.ing._detect_xml_format(xml))

    def test_empty_xml(self):
        self.assertIsNone(self.ing._detect_xml_format(b''))

    def test_pensoft_taxpub_xml(self):
        """Real-world Pensoft TaxPub XML header."""
        xml = (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<!DOCTYPE article PUBLIC "-//TaxPub//DTD Taxonomic Treatment '
            b'Publishing DTD v1.0 20180101//EN" '
            b'"https://raw.githubusercontent.com/plazi/TaxPub/TaxPubJATS/tax-treatment-NS0-v1.dtd">\n'
            b'<article article-type="research-article" '
            b'dtd-version="3.0" xml:lang="en" '
            b'xmlns:mml="http://www.w3.org/1998/Math/MathML" '
            b'xmlns:xlink="http://www.w3.org/1999/xlink" '
            b'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            b'xmlns:tp="http://www.plazi.org/taxpub">\n'
        )
        self.assertEqual(self.ing._detect_xml_format(xml), 'jats')


class TestDownloadFlags(unittest.TestCase):
    """Test that download_pdf / download_xml flags control downloads."""

    def _make_response(self, content, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.content = content
        return resp

    def _run_ingest(self, download_pdf=True, download_xml=True):
        """Run _ingest_documents with mocked HTTP and DB, return the ingestor."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        # Track saved docs so db[doc_id] returns a real dict
        saved_state: Dict[str, dict] = {}

        def fake_save(doc):
            saved_state[doc['_id']] = dict(doc)

        def fake_getitem(doc_id):
            return dict(saved_state.get(doc_id, {'_id': doc_id}))

        db.save = MagicMock(side_effect=fake_save)
        db.__getitem__ = MagicMock(side_effect=fake_getitem)
        # Expose saved_state for assertions
        db._saved_state = saved_state

        ing = _make_ingestor(
            db=db,
            download_pdf=download_pdf,
            download_xml=download_xml,
        )

        pdf_content = b'%PDF-1.4 fake pdf content'
        xml_content = (
            b'<?xml version="1.0"?>\n'
            b'<!DOCTYPE article PUBLIC "-//NLM//DTD JATS v1.2//EN">\n'
            b'<article>content</article>'
        )

        def side_effect(url, **kwargs):
            if '/pdf/' in url:
                return self._make_response(pdf_content)
            if '/xml/' in url:
                return self._make_response(xml_content)
            return self._make_response(b'', status_code=404)

        ing._get_with_rate_limit = MagicMock(side_effect=side_effect)

        documents = [{
            'pdf_url': 'https://mycokeys.pensoft.net/article/12345/download/pdf/67890',
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]
        meta = {'source': 'pensoft', 'journal': 'mycokeys'}

        ing._ingest_documents(documents, meta, bibtex_link='https://example.com')
        return ing

    def test_both_downloads(self):
        """Both PDF and XML downloaded when both flags are True."""
        ing = self._run_ingest(download_pdf=True, download_xml=True)
        urls_fetched = [
            c.args[0] for c in ing._get_with_rate_limit.call_args_list
        ]
        self.assertTrue(
            any('/pdf/' in u for u in urls_fetched),
            f'Expected a PDF URL in {urls_fetched}',
        )
        self.assertTrue(
            any('/xml/' in u for u in urls_fetched),
            f'Expected an XML URL in {urls_fetched}',
        )
        # Two put_attachment calls: one for PDF, one for XML
        put_calls = ing.db.put_attachment.call_args_list
        names = [c.args[2] for c in put_calls]
        self.assertIn('article.pdf', names)
        self.assertIn('article.xml', names)

    def test_pdf_only(self):
        """Only PDF downloaded when download_xml=False."""
        ing = self._run_ingest(download_pdf=True, download_xml=False)
        urls_fetched = [
            c.args[0] for c in ing._get_with_rate_limit.call_args_list
        ]
        self.assertTrue(any('/pdf/' in u for u in urls_fetched))
        self.assertFalse(any('/xml/' in u for u in urls_fetched))

    def test_xml_only(self):
        """Only XML downloaded when download_pdf=False."""
        ing = self._run_ingest(download_pdf=False, download_xml=True)
        urls_fetched = [
            c.args[0] for c in ing._get_with_rate_limit.call_args_list
        ]
        self.assertFalse(any('/pdf/' in u for u in urls_fetched))
        self.assertTrue(any('/xml/' in u for u in urls_fetched))

    def test_neither_download(self):
        """No downloads when both flags are False (metadata-only doc)."""
        ing = self._run_ingest(download_pdf=False, download_xml=False)
        ing._get_with_rate_limit.assert_not_called()

    def test_xml_format_set_for_jats(self):
        """xml_format field is set when JATS XML is detected."""
        ing = self._run_ingest(download_pdf=False, download_xml=True)
        # Check that xml_format='jats' was saved to the doc
        saved_state = ing.db._saved_state
        jats_saved = any(
            doc.get('xml_format') == 'jats'
            for doc in saved_state.values()
        )
        self.assertTrue(
            jats_saved,
            'Expected xml_format="jats" to be saved to CouchDB',
        )


class TestSkipExisting(unittest.TestCase):
    """Test that existing attachments are not re-downloaded."""

    def test_skip_complete_document(self):
        """Skip document that already has both PDF and XML attachments."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=True)
        existing = {
            '_id': 'fake',
            '_attachments': {
                'article.pdf': {'content_type': 'application/pdf'},
                'article.xml': {'content_type': 'application/xml'},
            },
        }
        db.__getitem__ = MagicMock(return_value=existing)

        ing = _make_ingestor(db=db)
        ing._get_with_rate_limit = MagicMock()

        doc_id_url = 'https://mycokeys.pensoft.net/article/12345/download/pdf/67890'
        documents = [{
            'pdf_url': doc_id_url,
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')
        ing._get_with_rate_limit.assert_not_called()

    def test_add_missing_xml_to_existing(self):
        """Download XML for a doc that already has PDF but not XML."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=True)
        existing = {
            '_id': 'fake',
            '_attachments': {
                'article.pdf': {'content_type': 'application/pdf'},
            },
        }
        db.__getitem__ = MagicMock(return_value=existing)

        ing = _make_ingestor(db=db)

        xml_content = b'<?xml version="1.0"?>\n<root>data</root>'
        resp = MagicMock()
        resp.status_code = 200
        resp.content = xml_content
        ing._get_with_rate_limit = MagicMock(return_value=resp)

        documents = [{
            'pdf_url': 'https://mycokeys.pensoft.net/article/12345/download/pdf/67890',
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')
        # Should have fetched the XML URL
        urls_fetched = [
            c.args[0] for c in ing._get_with_rate_limit.call_args_list
        ]
        self.assertTrue(any('/xml/' in u for u in urls_fetched))
        self.assertFalse(any('/pdf/' in u for u in urls_fetched))


class TestXmlAvailability(unittest.TestCase):
    """Test that xml_available field is recorded and respected."""

    def _make_response(self, content, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.content = content
        return resp

    def test_skip_xml_when_marked_unavailable(self):
        """Don't attempt XML download if xml_available is False."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=True)
        existing = {
            '_id': 'fake',
            'xml_available': False,
            '_attachments': {
                'article.pdf': {'content_type': 'application/pdf'},
            },
        }
        db.__getitem__ = MagicMock(return_value=existing)

        ing = _make_ingestor(db=db)
        ing._get_with_rate_limit = MagicMock()

        documents = [{
            'pdf_url': 'https://mycokeys.pensoft.net/article/12345/download/pdf/67890',
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')
        # Should not have fetched anything (PDF exists, XML marked unavailable)
        ing._get_with_rate_limit.assert_not_called()

    def test_recheck_xml_ignores_unavailable(self):
        """With recheck_xml=True, retry even if xml_available is False."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=True)
        existing = {
            '_id': 'fake',
            'xml_available': False,
            '_attachments': {
                'article.pdf': {'content_type': 'application/pdf'},
            },
        }
        db.__getitem__ = MagicMock(return_value=existing)

        ing = _make_ingestor(db=db, recheck_xml=True)

        xml_content = b'<?xml version="1.0"?>\n<root>data</root>'
        ing._get_with_rate_limit = MagicMock(
            return_value=self._make_response(xml_content)
        )

        documents = [{
            'pdf_url': 'https://mycokeys.pensoft.net/article/12345/download/pdf/67890',
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')
        # Should have fetched the XML URL despite xml_available=False
        urls_fetched = [
            c.args[0]
            for c in ing._get_with_rate_limit.call_args_list
        ]
        self.assertTrue(any('/xml/' in u for u in urls_fetched))

    def test_xml_available_set_false_on_http_error(self):
        """xml_available set to False when XML download returns HTTP error."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        saved_state: Dict[str, dict] = {}

        def fake_save(doc):
            saved_state[doc['_id']] = dict(doc)

        def fake_getitem(doc_id):
            return dict(saved_state.get(doc_id, {'_id': doc_id}))

        db.save = MagicMock(side_effect=fake_save)
        db.__getitem__ = MagicMock(side_effect=fake_getitem)

        ing = _make_ingestor(db=db, download_pdf=False, download_xml=True)
        ing._get_with_rate_limit = MagicMock(
            return_value=self._make_response(b'', status_code=404)
        )

        documents = [{
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')

        # Find the saved doc and check xml_available
        self.assertTrue(len(saved_state) > 0)
        doc = list(saved_state.values())[-1]
        self.assertIs(doc['xml_available'], False)

    def test_xml_available_set_false_on_invalid_content(self):
        """xml_available set to False when response is not XML."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        saved_state: Dict[str, dict] = {}

        def fake_save(doc):
            saved_state[doc['_id']] = dict(doc)

        def fake_getitem(doc_id):
            return dict(saved_state.get(doc_id, {'_id': doc_id}))

        db.save = MagicMock(side_effect=fake_save)
        db.__getitem__ = MagicMock(side_effect=fake_getitem)

        ing = _make_ingestor(db=db, download_pdf=False, download_xml=True)
        # Return HTML instead of XML
        ing._get_with_rate_limit = MagicMock(
            return_value=self._make_response(
                b'This is not XML content at all'
            )
        )

        documents = [{
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')

        doc = list(saved_state.values())[-1]
        self.assertIs(doc['xml_available'], False)

    def test_xml_available_set_true_on_success(self):
        """xml_available set to True when XML download succeeds."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        saved_state: Dict[str, dict] = {}

        def fake_save(doc):
            saved_state[doc['_id']] = dict(doc)

        def fake_getitem(doc_id):
            return dict(saved_state.get(doc_id, {'_id': doc_id}))

        db.save = MagicMock(side_effect=fake_save)
        db.__getitem__ = MagicMock(side_effect=fake_getitem)

        ing = _make_ingestor(db=db, download_pdf=False, download_xml=True)
        xml_content = b'<?xml version="1.0"?>\n<root>data</root>'
        ing._get_with_rate_limit = MagicMock(
            return_value=self._make_response(xml_content)
        )

        documents = [{
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')

        doc = list(saved_state.values())[-1]
        self.assertIs(doc['xml_available'], True)


class TestDocumentIdStability(unittest.TestCase):
    """Test that document IDs are stable and based on pdf_url."""

    def test_doc_id_from_pdf_url(self):
        """Document ID is derived from pdf_url when available."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        ing = _make_ingestor(db=db, download_pdf=False, download_xml=False)

        pdf_url = 'https://mycokeys.pensoft.net/article/12345/download/pdf/67890'
        documents = [{
            'pdf_url': pdf_url,
            'xml_url': 'https://mycokeys.pensoft.net/article/12345/download/xml/',
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')

        expected_id = uuid5(NAMESPACE_URL, pdf_url).hex
        saved = ing.db.save.call_args.args[0]
        self.assertEqual(saved['_id'], expected_id)

    def test_doc_id_from_xml_url_when_no_pdf(self):
        """Document ID falls back to xml_url when no pdf_url."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        ing = _make_ingestor(db=db, download_pdf=False, download_xml=False)

        xml_url = 'https://mycokeys.pensoft.net/article/12345/download/xml/'
        documents = [{
            'xml_url': xml_url,
            'url': 'https://mycokeys.pensoft.net/article/12345/',
        }]

        ing._ingest_documents(documents, {}, bibtex_link='')

        expected_id = uuid5(NAMESPACE_URL, xml_url).hex
        saved = ing.db.save.call_args.args[0]
        self.assertEqual(saved['_id'], expected_id)


class TestArticleFilter(unittest.TestCase):
    """Test that the article eligibility filter in ingest_from_issues works."""

    def test_download_pdf_true_filters_without_pdf_url(self):
        """When download_pdf=True, articles without pdf_url are filtered."""
        ing = _make_ingestor(download_pdf=True)
        ing._ingest_documents = MagicMock()

        # Mock _fetch_page to return a page with two articles, one without PDF
        html = """
        <div class="article">
          <div class="articleHeadline">
            <a href="/article/111/">Article with PDF</a>
          </div>
          <div class="DownLink">
            <a href="/article/111/download/pdf/222">Download PDF</a>
          </div>
        </div>
        <div class="article">
          <div class="articleHeadline">
            <a href="/article/333/">Article without PDF</a>
          </div>
        </div>
        """
        # Simulate: issues page returns one issue, issue page returns articles
        issue_page = BeautifulSoup(
            '<a href="/issue/99/">Issue 99</a>', 'html.parser'
        )
        article_page = BeautifulSoup(html, 'html.parser')

        call_count = [0]
        def fake_fetch(url):
            call_count[0] += 1
            if call_count[0] == 1:
                return issue_page
            return article_page

        ing._fetch_page = MagicMock(side_effect=fake_fetch)

        ing.ingest_from_issues(max_issues=1)

        # _ingest_documents should have been called with only the article
        # that has a pdf_url
        docs = ing._ingest_documents.call_args.kwargs.get(
            'documents',
            ing._ingest_documents.call_args.args[0]
            if ing._ingest_documents.call_args.args else []
        )
        self.assertEqual(len(docs), 1)
        self.assertIn('pdf_url', docs[0])

    def test_download_pdf_false_keeps_all(self):
        """When download_pdf=False, articles without pdf_url are kept."""
        ing = _make_ingestor(download_pdf=False)
        ing._ingest_documents = MagicMock()

        html = """
        <div class="article">
          <div class="articleHeadline">
            <a href="/article/111/">Article with PDF</a>
          </div>
          <div class="DownLink">
            <a href="/article/111/download/pdf/222">Download PDF</a>
          </div>
        </div>
        <div class="article">
          <div class="articleHeadline">
            <a href="/article/333/">Article without PDF</a>
          </div>
        </div>
        """
        issue_page = BeautifulSoup(
            '<a href="/issue/99/">Issue 99</a>', 'html.parser'
        )
        article_page = BeautifulSoup(html, 'html.parser')

        call_count = [0]
        def fake_fetch(url):
            call_count[0] += 1
            if call_count[0] == 1:
                return issue_page
            return article_page

        ing._fetch_page = MagicMock(side_effect=fake_fetch)

        ing.ingest_from_issues(max_issues=1)

        docs = ing._ingest_documents.call_args.kwargs.get(
            'documents',
            ing._ingest_documents.call_args.args[0]
            if ing._ingest_documents.call_args.args else []
        )
        self.assertEqual(len(docs), 2)


class TestParseDate(unittest.TestCase):
    """Test the _parse_date helper."""

    def setUp(self):
        self.ing = _make_ingestor()

    def test_dd_mm_yyyy(self):
        self.assertEqual(self.ing._parse_date('27-11-2025'), '2025-11-27')

    def test_dd_month_yyyy(self):
        self.assertEqual(self.ing._parse_date('27 November 2025'), '2025-11-27')

    def test_iso(self):
        self.assertEqual(self.ing._parse_date('2025-11-27'), '2025-11-27')

    def test_none(self):
        self.assertIsNone(self.ing._parse_date(None))

    def test_empty(self):
        self.assertIsNone(self.ing._parse_date(''))


class TestConstructor(unittest.TestCase):
    """Test constructor parameter handling."""

    def test_default_flags(self):
        ing = _make_ingestor()
        self.assertTrue(ing.download_pdf)
        self.assertTrue(ing.download_xml)

    def test_override_flags(self):
        ing = _make_ingestor(download_pdf=False, download_xml=False)
        self.assertFalse(ing.download_pdf)
        self.assertFalse(ing.download_xml)

    def test_base_url(self):
        ing = _make_ingestor(journal_name='imafungus')
        self.assertEqual(ing.base_url, 'https://imafungus.pensoft.net')

    def test_issues_url_default(self):
        ing = _make_ingestor(journal_name='mycokeys')
        self.assertEqual(
            ing.issues_url, 'https://mycokeys.pensoft.net/issues'
        )

    def test_issues_url_override(self):
        ing = _make_ingestor(issues_url='https://custom.example.com/issues')
        self.assertEqual(ing.issues_url, 'https://custom.example.com/issues')


if __name__ == '__main__':
    unittest.main()
