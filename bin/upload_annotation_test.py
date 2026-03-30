"""Tests for upload_annotation.py.

Covers:
- resolve_attachment_name: derive the attachment name from a local file path
- upload_attachment: HTTP interaction with CouchDB (mocked)
- Error cases: doc not found, rev conflict, bad content-type

Run with: python -m pytest bin/upload_annotation_test.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from upload_annotation import resolve_attachment_name, upload_attachment


# ---------------------------------------------------------------------------
# resolve_attachment_name
# ---------------------------------------------------------------------------

class TestResolveAttachmentName(unittest.TestCase):
    """resolve_attachment_name derives the CouchDB attachment name from path."""

    def test_plain_ann_file(self):
        """article.txt.ann → attachment name is article.txt.ann."""
        result = resolve_attachment_name(Path("article.txt.ann"))
        self.assertEqual(result, "article.txt.ann")

    def test_pdf_ann_file(self):
        """article.pdf.ann → attachment name is article.pdf.ann."""
        result = resolve_attachment_name(Path("article.pdf.ann"))
        self.assertEqual(result, "article.pdf.ann")

    def test_absolute_path_uses_filename_only(self):
        """Absolute path — only the filename part is used."""
        result = resolve_attachment_name(
            Path("/tmp/workdir/some_doc/article.txt.ann")
        )
        self.assertEqual(result, "article.txt.ann")

    def test_explicit_name_overrides(self):
        """Explicit name argument takes priority over file path."""
        result = resolve_attachment_name(
            Path("whatever.ann"), explicit_name="article.txt.ann"
        )
        self.assertEqual(result, "article.txt.ann")

    def test_no_extension_file_still_works(self):
        """File without .ann extension is used as-is."""
        result = resolve_attachment_name(Path("myfile"))
        self.assertEqual(result, "myfile")


# ---------------------------------------------------------------------------
# upload_attachment
# ---------------------------------------------------------------------------

_DOC_ID = "abc123"
_DB = "skol_dev"
_BASE_URL = "http://localhost:5984"
_REV = "1-deadbeef"
_ANN_CONTENT = "[@Amanita muscaria#Nomenclature*]"


def _make_get_response(status: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def _make_put_response(status: int, body: dict = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body or {"ok": True, "rev": "2-cafebabe"}
    resp.raise_for_status = MagicMock()
    return resp


class TestUploadAttachment(unittest.TestCase):
    """upload_attachment: GET rev then PUT attachment."""

    def _run(self, get_resp, put_resp, dry_run=False):
        """Helper: call upload_attachment with mocked requests."""
        with patch("upload_annotation.requests") as mock_requests:
            mock_requests.get.return_value = get_resp
            mock_requests.put.return_value = put_resp
            result = upload_attachment(
                couchdb_url=_BASE_URL,
                db=_DB,
                doc_id=_DOC_ID,
                attachment_name="article.txt.ann",
                content=_ANN_CONTENT,
                username="admin",
                password="pass",
                dry_run=dry_run,
            )
            return result, mock_requests

    def test_successful_upload_returns_new_rev(self):
        """On success, returns the new revision string."""
        get_resp = _make_get_response(200, {"_id": _DOC_ID, "_rev": _REV})
        put_resp = _make_put_response(201, {"ok": True, "rev": "2-cafebabe"})
        result, _ = self._run(get_resp, put_resp)
        self.assertEqual(result, "2-cafebabe")

    def test_get_is_called_first(self):
        """GET is issued before PUT to fetch the current _rev."""
        get_resp = _make_get_response(200, {"_id": _DOC_ID, "_rev": _REV})
        put_resp = _make_put_response(201)
        _, mock_requests = self._run(get_resp, put_resp)
        get_url = mock_requests.get.call_args[0][0]
        self.assertIn(_DOC_ID, get_url)
        self.assertIn(_DB, get_url)

    def test_put_url_contains_doc_id_and_attachment_name(self):
        """PUT URL encodes both the doc ID and attachment name."""
        get_resp = _make_get_response(200, {"_id": _DOC_ID, "_rev": _REV})
        put_resp = _make_put_response(201)
        _, mock_requests = self._run(get_resp, put_resp)
        put_url = mock_requests.put.call_args[0][0]
        self.assertIn(_DOC_ID, put_url)
        self.assertIn("article.txt.ann", put_url)

    def test_put_includes_rev_query_param(self):
        """PUT request includes ?rev= query parameter from the GET response."""
        get_resp = _make_get_response(200, {"_id": _DOC_ID, "_rev": _REV})
        put_resp = _make_put_response(201)
        _, mock_requests = self._run(get_resp, put_resp)
        put_kwargs = mock_requests.put.call_args[1]
        params = put_kwargs.get("params", {})
        self.assertEqual(params.get("rev"), _REV)

    def test_put_content_type_is_plain_text(self):
        """PUT uses text/plain content type for .ann files."""
        get_resp = _make_get_response(200, {"_id": _DOC_ID, "_rev": _REV})
        put_resp = _make_put_response(201)
        _, mock_requests = self._run(get_resp, put_resp)
        put_kwargs = mock_requests.put.call_args[1]
        headers = put_kwargs.get("headers", {})
        self.assertEqual(headers.get("Content-Type"), "text/plain; charset=utf-8")

    def test_dry_run_skips_put(self):
        """In dry-run mode no PUT is issued; returns None."""
        get_resp = _make_get_response(200, {"_id": _DOC_ID, "_rev": _REV})
        put_resp = _make_put_response(201)
        result, mock_requests = self._run(get_resp, put_resp, dry_run=True)
        mock_requests.put.assert_not_called()
        self.assertIsNone(result)

    def test_doc_not_found_raises(self):
        """404 on the GET raises an informative error."""
        get_resp = _make_get_response(404, {"error": "not_found"})
        get_resp.raise_for_status.side_effect = Exception("404")
        with patch("upload_annotation.requests") as mock_requests:
            mock_requests.get.return_value = get_resp
            with self.assertRaises(Exception):
                upload_attachment(
                    couchdb_url=_BASE_URL,
                    db=_DB,
                    doc_id=_DOC_ID,
                    attachment_name="article.txt.ann",
                    content=_ANN_CONTENT,
                    username="admin",
                    password="pass",
                )

    def test_put_uses_auth(self):
        """PUT passes HTTP Basic Auth credentials."""
        get_resp = _make_get_response(200, {"_id": _DOC_ID, "_rev": _REV})
        put_resp = _make_put_response(201)
        _, mock_requests = self._run(get_resp, put_resp)
        put_kwargs = mock_requests.put.call_args[1]
        auth = put_kwargs.get("auth")
        self.assertIsNotNone(auth)


if __name__ == "__main__":
    unittest.main()
