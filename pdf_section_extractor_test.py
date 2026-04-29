"""Tests for PDFSectionExtractor.pdf_to_text()."""
import io
from unittest.mock import MagicMock, patch

import pytest

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed"
)


def _make_pdf(page_text: str = "Hello world") -> bytes:
    """Return bytes of a minimal single-page PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), page_text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture()
def extractor():
    """PDFSectionExtractor with CouchDB connection mocked out."""
    with patch("couchdb.Server", return_value=MagicMock()):
        from pdf_section_extractor import PDFSectionExtractor
        return PDFSectionExtractor(verbosity=0)


class TestPdfToText:
    def test_returns_string(self, extractor):
        pdf = _make_pdf("Sample text")
        result = extractor.pdf_to_text(pdf)
        assert isinstance(result, str)

    def test_contains_page_marker(self, extractor):
        pdf = _make_pdf("Content here")
        result = extractor.pdf_to_text(pdf)
        assert "--- PDF Page 1" in result

    def test_contains_extracted_text(self, extractor):
        pdf = _make_pdf("Distinctive phrase xyz")
        result = extractor.pdf_to_text(pdf)
        assert "Distinctive phrase xyz" in result

    def test_get_label_returns_empty_falls_back_to_page_number(self, extractor):
        pdf = _make_pdf("Page label test")
        with patch("fitz.Page.get_label", return_value=""):
            result = extractor.pdf_to_text(pdf)
        # Should fall back to "1" when get_label returns ""
        assert "Label 1" in result

    def test_get_label_assertion_error_falls_back(self, extractor):
        pdf = _make_pdf("Broken label test")
        with patch("fitz.Page.get_label", side_effect=AssertionError("m_internal")):
            result = extractor.pdf_to_text(pdf)
        # Should not crash; page marker should use numeric fallback
        assert "--- PDF Page 1 Label 1 ---" in result

    def test_get_label_assertion_error_prints_warning_at_verbosity_1(self):
        with patch("couchdb.Server", return_value=MagicMock()):
            from pdf_section_extractor import PDFSectionExtractor
            ext = PDFSectionExtractor(verbosity=1)
        pdf = _make_pdf("Warning test")
        with patch("fitz.Page.get_label", side_effect=AssertionError("m_internal")):
            import builtins
            printed: list = []
            orig_print = builtins.print
            with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(str(a))):
                ext.pdf_to_text(pdf)
            assert any("Warning" in s or "get_label" in s for s in printed)

    def test_get_label_exception_falls_back(self, extractor):
        """Any unexpected exception from get_label() is also caught."""
        pdf = _make_pdf("Exception test")
        with patch("fitz.Page.get_label", side_effect=RuntimeError("unexpected")):
            result = extractor.pdf_to_text(pdf)
        assert "--- PDF Page 1 Label 1 ---" in result
