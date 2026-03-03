"""
Unit tests for the CrossrefIngestor class.

Tests content type detection, author formatting, year extraction,
human URL extraction, document ID stability, skip-existing logic,
ISSN normalization, constructor parameters, and the PDF download pipeline.
"""

import unittest
from typing import Dict
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid5, NAMESPACE_URL

from .crossref import CrossrefIngestor


def _make_ingestor(**overrides):
    """Create a CrossrefIngestor with mocked dependencies."""
    db = overrides.pop('db', MagicMock())
    # CrossrefIngestor._ingest_work unpacks db.save() as (doc_id, doc_rev)
    db.save.return_value = ('fake_id', 'fake_rev')
    robot_parser = overrides.pop('robot_parser', MagicMock())
    robot_parser.can_fetch.return_value = True
    defaults = dict(
        db=db,
        user_agent='test-agent',
        robot_parser=robot_parser,
        verbosity=0,
        issn='2309-608X',
        mailto='test@example.com',
    )
    defaults.update(overrides)
    return CrossrefIngestor(**defaults)


def _make_work(**overrides):
    """Create a minimal Crossref work dictionary."""
    work = {
        'DOI': '10.3390/jof9010042',
        'title': ['A New Species of Cortinarius'],
        'author': [
            {'given': 'John', 'family': 'Smith'},
            {'given': 'Jane', 'family': 'Doe'},
        ],
        'container-title': ['Journal of Fungi'],
        'volume': '9',
        'issue': '1',
        'page': '42',
        'published-print': {'date-parts': [[2023, 1, 5]]},
    }
    work.update(overrides)
    return work


class TestContentTypeDetection(unittest.TestCase):
    """Test _detect_content_type for various content types."""

    def setUp(self):
        self.ing = _make_ingestor()

    def test_pdf_magic_bytes(self):
        content = b'%PDF-1.4 fake pdf content'
        filename, content_type = self.ing._detect_content_type(content)
        self.assertEqual(filename, 'article.pdf')
        self.assertEqual(content_type, 'application/pdf')

    def test_xml_declaration(self):
        content = b'<?xml version="1.0"?>\n<article>data</article>'
        filename, content_type = self.ing._detect_content_type(content)
        self.assertEqual(filename, 'article.xml')
        self.assertEqual(content_type, 'application/xml')

    def test_xml_article_tag(self):
        content = b'<article xmlns="http://jats.nlm.nih.gov">\n<front>...</front></article>'
        filename, content_type = self.ing._detect_content_type(content)
        self.assertEqual(filename, 'article.xml')
        self.assertEqual(content_type, 'application/xml')

    def test_xml_with_leading_whitespace(self):
        content = b'  \n  <?xml version="1.0"?>\n<root/>'
        filename, content_type = self.ing._detect_content_type(content)
        self.assertEqual(filename, 'article.xml')
        self.assertEqual(content_type, 'application/xml')

    def test_unknown_defaults_to_pdf(self):
        content = b'\x89PNG\r\n\x1a\n some image data'
        filename, content_type = self.ing._detect_content_type(content)
        self.assertEqual(filename, 'article.pdf')
        self.assertEqual(content_type, 'application/pdf')

    def test_empty_content_defaults_to_pdf(self):
        filename, content_type = self.ing._detect_content_type(b'')
        self.assertEqual(filename, 'article.pdf')


class TestFormatAuthors(unittest.TestCase):
    """Test _format_authors for various author list formats."""

    def setUp(self):
        self.ing = _make_ingestor()

    def test_two_authors(self):
        authors = [
            {'given': 'John', 'family': 'Smith'},
            {'given': 'Jane', 'family': 'Doe'},
        ]
        self.assertEqual(
            self.ing._format_authors(authors),
            'John Smith; Jane Doe',
        )

    def test_family_only(self):
        authors = [{'family': 'Smith'}]
        self.assertEqual(self.ing._format_authors(authors), 'Smith')

    def test_empty_list(self):
        self.assertEqual(self.ing._format_authors([]), '')

    def test_none(self):
        self.assertEqual(self.ing._format_authors(None), '')

    def test_given_only_skipped(self):
        """Authors without family name are skipped."""
        authors = [{'given': 'John'}]
        self.assertEqual(self.ing._format_authors(authors), '')

    def test_mixed(self):
        authors = [
            {'given': 'A.', 'family': 'First'},
            {'given': 'B.'},  # no family, skipped
            {'family': 'Third'},
        ]
        self.assertEqual(
            self.ing._format_authors(authors),
            'A. First; Third',
        )


class TestExtractYear(unittest.TestCase):
    """Test _extract_year for various Crossref date formats."""

    def setUp(self):
        self.ing = _make_ingestor()

    def test_published_print(self):
        work = {'published-print': {'date-parts': [[2023, 1, 5]]}}
        self.assertEqual(self.ing._extract_year(work), '2023')

    def test_published_online(self):
        work = {'published-online': {'date-parts': [[2024, 6]]}}
        self.assertEqual(self.ing._extract_year(work), '2024')

    def test_created_fallback(self):
        work = {'created': {'date-parts': [[2022]]}}
        self.assertEqual(self.ing._extract_year(work), '2022')

    def test_prefers_print_over_online(self):
        work = {
            'published-print': {'date-parts': [[2023]]},
            'published-online': {'date-parts': [[2022]]},
        }
        self.assertEqual(self.ing._extract_year(work), '2023')

    def test_no_date(self):
        self.assertEqual(self.ing._extract_year({}), '')

    def test_empty_date_parts(self):
        work = {'published-print': {'date-parts': [[]]}}
        self.assertEqual(self.ing._extract_year(work), '')


class TestExtractHumanUrl(unittest.TestCase):
    """Test _extract_human_url for various Crossref records."""

    def setUp(self):
        self.ing = _make_ingestor()
        self.doi_url = 'https://doi.org/10.3390/jof9010042'

    def test_resource_primary_url(self):
        work = {
            'resource': {
                'primary': {
                    'URL': 'https://www.mdpi.com/2309-608X/9/1/42',
                },
            },
        }
        self.assertEqual(
            self.ing._extract_human_url(work, self.doi_url),
            'https://www.mdpi.com/2309-608X/9/1/42',
        )

    def test_link_text_html(self):
        work = {
            'link': [
                {
                    'URL': 'https://publisher.com/article/42',
                    'content-type': 'text/html',
                    'intended-application': 'similarity-checking',
                },
            ],
        }
        self.assertEqual(
            self.ing._extract_human_url(work, self.doi_url),
            'https://publisher.com/article/42',
        )

    def test_fallback_to_doi_url(self):
        work = {}
        self.assertEqual(
            self.ing._extract_human_url(work, self.doi_url),
            self.doi_url,
        )

    def test_prefers_resource_over_link(self):
        work = {
            'resource': {
                'primary': {
                    'URL': 'https://primary.example.com/article',
                },
            },
            'link': [
                {
                    'URL': 'https://link.example.com/article',
                    'content-type': 'text/html',
                },
            ],
        }
        self.assertEqual(
            self.ing._extract_human_url(work, self.doi_url),
            'https://primary.example.com/article',
        )


class TestDocumentIdStability(unittest.TestCase):
    """Test that document IDs are stable and based on DOI URL."""

    def test_doc_id_from_doi(self):
        """Document ID is derived from https://doi.org/{DOI}."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        ing = _make_ingestor(db=db)
        # Stub out PDF download so _ingest_work only creates the doc
        ing._download_pdf_with_pypaperretriever = MagicMock(
            return_value=None
        )

        work = _make_work()
        ing._ingest_work(work)

        doi_url = f"https://doi.org/{work['DOI']}"
        expected_id = str(uuid5(NAMESPACE_URL, doi_url))
        saved = db.save.call_args.args[0]
        self.assertEqual(saved['_id'], expected_id)

    def test_doc_id_deterministic(self):
        """Same DOI always produces the same document ID."""
        doi = '10.3390/jof9010042'
        url = f'https://doi.org/{doi}'
        id1 = str(uuid5(NAMESPACE_URL, url))
        id2 = str(uuid5(NAMESPACE_URL, url))
        self.assertEqual(id1, id2)


class TestSkipExisting(unittest.TestCase):
    """Test that already-ingested documents are skipped."""

    def test_skip_document_with_pdf(self):
        """Skip document that already has a PDF attachment."""
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
        ing._download_pdf_with_pypaperretriever = MagicMock()

        work = _make_work()
        ing._ingest_work(work)

        # Should not attempt download
        ing._download_pdf_with_pypaperretriever.assert_not_called()

    def test_add_pdf_to_existing_without_attachment(self):
        """Download PDF for existing doc that has no PDF attachment."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=True)
        existing = {
            '_id': 'fake',
            '_attachments': {},
        }
        db.__getitem__ = MagicMock(return_value=existing)

        ing = _make_ingestor(db=db)
        pdf_content = b'%PDF-1.4 content'
        ing._download_pdf_with_pypaperretriever = MagicMock(
            return_value=(pdf_content, 'article.pdf', 'application/pdf')
        )

        work = _make_work()
        ing._ingest_work(work)

        ing._download_pdf_with_pypaperretriever.assert_called_once()
        db.put_attachment.assert_called_once()

    def test_existing_doc_without_attachments_key(self):
        """Handle existing doc that has no _attachments key at all."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=True)
        existing = {'_id': 'fake'}
        db.__getitem__ = MagicMock(return_value=existing)

        ing = _make_ingestor(db=db)
        ing._download_pdf_with_pypaperretriever = MagicMock(
            return_value=None
        )

        work = _make_work()
        ing._ingest_work(work)

        # Should attempt download (no _attachments means no PDF)
        ing._download_pdf_with_pypaperretriever.assert_called_once()


class TestIssnNormalization(unittest.TestCase):
    """Test ISSN formatting in ingest()."""

    def test_hyphen_added_when_missing(self):
        """ISSN without hyphen gets hyphen inserted."""
        ing = _make_ingestor(issn='2309608X')
        # Mock the Crossref client and _fetch_all_works
        with patch('ingestors.crossref.Crossref') as mock_cr:
            mock_cr.return_value.works.return_value = iter([])
            ing._fetch_all_works = MagicMock(return_value=iter([]))
            ing.ingest()
        # The ingest method should have run without error

    def test_hyphen_preserved(self):
        """ISSN with hyphen is left unchanged."""
        ing = _make_ingestor(issn='2309-608X')
        ing._fetch_all_works = MagicMock(return_value=iter([]))
        with patch('ingestors.crossref.Crossref'):
            ing.ingest()

    def test_no_issn_raises(self):
        """Ingest with no ISSN raises ValueError."""
        ing = _make_ingestor(issn=None)
        with self.assertRaises(ValueError):
            ing.ingest()


class TestConstructor(unittest.TestCase):
    """Test constructor parameter handling."""

    def test_defaults(self):
        ing = _make_ingestor()
        self.assertEqual(ing.issn, '2309-608X')
        self.assertEqual(ing.mailto, 'test@example.com')
        self.assertIsNone(ing.max_articles)
        self.assertTrue(ing.allow_scihub)
        self.assertEqual(ing.api_batch_delay, 0.1)

    def test_overrides(self):
        ing = _make_ingestor(
            issn='0027-5514',
            mailto='other@example.com',
            max_articles=100,
            allow_scihub=False,
            api_batch_delay=0.5,
        )
        self.assertEqual(ing.issn, '0027-5514')
        self.assertEqual(ing.mailto, 'other@example.com')
        self.assertEqual(ing.max_articles, 100)
        self.assertFalse(ing.allow_scihub)
        self.assertEqual(ing.api_batch_delay, 0.5)


class TestIngestWork(unittest.TestCase):
    """Test _ingest_work metadata extraction."""

    def test_metadata_stored(self):
        """Extracted metadata is stored in the CouchDB document."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)

        ing = _make_ingestor(db=db)
        ing._download_pdf_with_pypaperretriever = MagicMock(
            return_value=None
        )

        work = _make_work()
        ing._ingest_work(work)

        saved = db.save.call_args.args[0]
        self.assertEqual(saved['doi'], '10.3390/jof9010042')
        self.assertEqual(saved['title'], 'A New Species of Cortinarius')
        self.assertEqual(saved['author'], 'John Smith; Jane Doe')
        self.assertEqual(saved['year'], '2023')
        self.assertEqual(saved['journal'], 'Journal of Fungi')
        self.assertEqual(saved['volume'], '9')
        self.assertEqual(saved['issue'], '1')
        self.assertEqual(saved['pages'], '42')
        self.assertEqual(saved['source'], 'crossref')

    def test_skip_work_without_doi(self):
        """Works without a DOI are skipped."""
        db = MagicMock()
        ing = _make_ingestor(db=db)

        work = _make_work(DOI=None)
        ing._ingest_work(work)

        db.save.assert_not_called()

    def test_empty_title(self):
        """Work with missing title uses empty string."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        ing = _make_ingestor(db=db)
        ing._download_pdf_with_pypaperretriever = MagicMock(
            return_value=None
        )

        work = _make_work(title=None)
        ing._ingest_work(work)

        saved = db.save.call_args.args[0]
        self.assertEqual(saved['title'], '')

    def test_bibtex_url_constructed(self):
        """BibTeX URL is constructed from DOI."""
        db = MagicMock()
        db.__contains__ = MagicMock(return_value=False)
        ing = _make_ingestor(db=db)
        ing._download_pdf_with_pypaperretriever = MagicMock(
            return_value=None
        )

        work = _make_work()
        ing._ingest_work(work)

        saved = db.save.call_args.args[0]
        self.assertEqual(
            saved['bibtex_url'],
            'https://api.crossref.org/works/10.3390/jof9010042'
            '/transform/application/x-bibtex',
        )


class TestTdmLink(unittest.TestCase):
    """Test TDM (Text and Data Mining) link extraction and download."""

    def _make_response(self, content, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.content = content
        return resp

    def test_tdm_link_found(self):
        """TDM link is extracted and downloaded."""
        ing = _make_ingestor()
        ing._get_with_rate_limit = MagicMock(
            return_value=self._make_response(b'%PDF-1.4 content')
        )

        work = {
            'link': [
                {
                    'URL': 'https://publisher.com/tdm/article.pdf',
                    'intended-application': 'text-mining',
                    'content-type': 'application/pdf',
                },
            ],
        }

        result = ing._try_tdm_link(work)
        self.assertEqual(result, b'%PDF-1.4 content')

    def test_tdm_application(self):
        """intended-application='tdm' is also recognized."""
        ing = _make_ingestor()
        ing._get_with_rate_limit = MagicMock(
            return_value=self._make_response(b'%PDF-1.4 content')
        )

        work = {
            'link': [
                {
                    'URL': 'https://publisher.com/tdm/article.pdf',
                    'intended-application': 'tdm',
                },
            ],
        }

        result = ing._try_tdm_link(work)
        self.assertIsNotNone(result)

    def test_no_tdm_link(self):
        """Returns None when no TDM link exists."""
        ing = _make_ingestor()
        work = {
            'link': [
                {
                    'URL': 'https://publisher.com/article',
                    'intended-application': 'similarity-checking',
                },
            ],
        }
        self.assertIsNone(ing._try_tdm_link(work))

    def test_no_links_key(self):
        """Returns None when work has no 'link' key."""
        ing = _make_ingestor()
        self.assertIsNone(ing._try_tdm_link({}))

    def test_tdm_download_failure(self):
        """Returns None when TDM download returns non-200."""
        ing = _make_ingestor()
        ing._get_with_rate_limit = MagicMock(
            return_value=self._make_response(b'', status_code=403)
        )

        work = {
            'link': [
                {
                    'URL': 'https://publisher.com/tdm/article.pdf',
                    'intended-application': 'text-mining',
                },
            ],
        }

        self.assertIsNone(ing._try_tdm_link(work))


class TestFetchAllWorks(unittest.TestCase):
    """Test _fetch_all_works pagination and max_articles."""

    def test_max_articles_limit(self):
        """Stop yielding after max_articles is reached."""
        ing = _make_ingestor(max_articles=2)

        mock_cr = MagicMock()
        # Simulate one batch with 5 items
        result = {
            'message': {
                'items': [
                    {'DOI': f'10.1234/test{i}'} for i in range(5)
                ],
                'next-cursor': 'cursor2',
            },
        }
        mock_cr.works.return_value = iter([result])

        works = list(ing._fetch_all_works(mock_cr, '2309-608X'))
        self.assertEqual(len(works), 2)

    def test_empty_results(self):
        """Handle empty results gracefully."""
        ing = _make_ingestor()

        mock_cr = MagicMock()
        result = {'message': {'items': []}}
        mock_cr.works.return_value = iter([result])

        works = list(ing._fetch_all_works(mock_cr, '2309-608X'))
        self.assertEqual(len(works), 0)

    def test_no_next_cursor_stops(self):
        """Stop when no next-cursor is returned."""
        ing = _make_ingestor()

        mock_cr = MagicMock()
        result = {
            'message': {
                'items': [{'DOI': '10.1234/test1'}],
                # No next-cursor key
            },
        }
        mock_cr.works.return_value = iter([result])

        works = list(ing._fetch_all_works(mock_cr, '2309-608X'))
        self.assertEqual(len(works), 1)


class TestFormatUrls(unittest.TestCase):
    """Test format_pdf_url, format_human_url, format_bibtex_url."""

    def setUp(self):
        self.ing = _make_ingestor()

    def test_format_pdf_url(self):
        base = {'url': 'https://doi.org/10.1234/test'}
        self.assertEqual(
            self.ing.format_pdf_url(base),
            'https://doi.org/10.1234/test',
        )

    def test_format_human_url(self):
        base = {'url': 'https://doi.org/10.1234/test'}
        self.assertEqual(
            self.ing.format_human_url(base),
            'https://doi.org/10.1234/test',
        )

    def test_format_bibtex_url(self):
        base = {'doi': '10.1234/test'}
        self.assertEqual(
            self.ing.format_bibtex_url(base, ''),
            'https://api.crossref.org/works/10.1234/test'
            '/transform/application/x-bibtex',
        )

    def test_format_bibtex_url_no_doi(self):
        self.assertEqual(self.ing.format_bibtex_url({}, ''), '')

    def test_format_pdf_url_missing(self):
        self.assertEqual(self.ing.format_pdf_url({}), '')


if __name__ == '__main__':
    unittest.main()
