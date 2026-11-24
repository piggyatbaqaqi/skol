"""
Tests for the attachment allowlist on GET /api/pdf/<db>/<doc_id>/<name>/.

PDFAttachmentView is public (DEFAULT_PERMISSION_CLASSES is AllowAny) and
used to pass any attachment name straight to CouchDB, so any attachment
on any document was downloadable.  Every caller in the site asks for
``article.pdf`` (the default; skolweb/urls.py pdf_viewer, index.html,
taxon-result-widget.js), which is also the only attachment the pipeline
redacts for licence (docs/approved-licenses.md).  Other names now get the
same 404 as a missing attachment, without CouchDB being asked.

Run with: python manage.py test search.tests.test_pdf_attachment_allowlist
"""
from typing import Any, List
from unittest import mock

from django.test import Client, TestCase


def _pdf_response() -> mock.MagicMock:
    response = mock.MagicMock(status_code=200, content=b'%PDF-1.5 x')
    response.headers = {'Content-Type': 'application/pdf'}
    return response


class TestPdfAttachmentAllowlist(TestCase):

    def setUp(self) -> None:
        self.client = Client()
        patcher = mock.patch('search.views.requests.get',
                             return_value=_pdf_response())
        self.get = patcher.start()
        self.addCleanup(patcher.stop)

    def requested_urls(self) -> List[Any]:
        return [c.args[0] for c in self.get.call_args_list]

    def test_default_serves_article_pdf(self) -> None:
        resp = self.client.get('/api/pdf/skol_dev/abc123/')
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(self.requested_urls()[0].endswith(
            '/skol_dev/abc123/article.pdf'))

    def test_article_pdf_by_name_is_served(self) -> None:
        resp = self.client.get('/api/pdf/skol_dev/abc123/article.pdf/')
        self.assertEqual(resp.status_code, 200)

    def test_other_names_are_not_found_and_couchdb_is_not_asked(
            self) -> None:
        for name in ('volume_complete.pdf', 'article.txt', 'article.xml',
                     'article.txt.ann', 'article.pdf.ann', 'other.pdf',
                     'Article.pdf'):
            with self.subTest(name=name):
                resp = self.client.get(f'/api/pdf/skol_dev/abc123/{name}/')
                self.assertEqual(resp.status_code, 404)
        self.assertEqual(self.requested_urls(), [])

    def test_doc_id_is_escaped_in_the_couchdb_url(self) -> None:
        # Without escaping, '?' would end the CouchDB path and turn the
        # rest of the id into a query string.
        self.client.get('/api/pdf/skol_dev/abc%3Frev%3D1/')
        self.assertTrue(self.requested_urls()[0].endswith(
            '/skol_dev/abc%3Frev%3D1/article.pdf'))
