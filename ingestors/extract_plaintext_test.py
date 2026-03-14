"""Tests for centralized plaintext extraction module.

Tests for:
- plaintext_from_pdf: PDF bytes → text via PDFSectionExtractor
- plaintext_from_jats: JATS XML string → body text
- plaintext_from_efetch: PMCID → text via NCBI E-utilities
"""

import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

TP_NS = "http://www.plazi.org/taxpub"


def _jats_article(body_xml: str) -> str:
    """Build a minimal JATS article with the given body content."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<article xmlns:tp="{TP_NS}">'
        "<front><article-meta>"
        "<abstract><p>Abstract text.</p></abstract>"
        "</article-meta></front>"
        f"<body>{body_xml}</body>"
        "</article>"
    )


# ---------------------------------------------------------------------------
# Tests: plaintext_from_pdf
# ---------------------------------------------------------------------------

class TestPlaintextFromPdf(unittest.TestCase):
    """Test PDF-to-plaintext extraction."""

    @patch("ingestors.extract_plaintext.PDFSectionExtractor")
    def test_returns_extracted_text(self, MockExtractor):
        """plaintext_from_pdf delegates to PDFSectionExtractor.pdf_to_text."""
        from ingestors.extract_plaintext import plaintext_from_pdf

        mock_instance = MockExtractor.return_value
        mock_instance.pdf_to_text.return_value = (
            "\n--- PDF Page 1 Label 1 ---\nHello world\n"
            "\n--- PDF Page 2 Label 2 ---\nPage two text\n"
        )

        pdf_bytes = b"%PDF-1.4 fake content"
        result = plaintext_from_pdf(pdf_bytes)

        mock_instance.pdf_to_text.assert_called_once_with(pdf_bytes)
        self.assertIn("Hello world", result)
        self.assertIn("Page two text", result)
        # Page markers should be preserved for downstream section parsing
        self.assertIn("--- PDF Page 1 Label 1 ---", result)

    @patch("ingestors.extract_plaintext.PDFSectionExtractor")
    def test_empty_pdf_returns_empty_string(self, MockExtractor):
        """An empty/blank PDF should return empty text."""
        from ingestors.extract_plaintext import plaintext_from_pdf

        mock_instance = MockExtractor.return_value
        mock_instance.pdf_to_text.return_value = ""

        result = plaintext_from_pdf(b"%PDF-1.4")
        self.assertEqual(result, "")

    @patch("ingestors.extract_plaintext.PDFSectionExtractor")
    def test_propagates_import_error(self, MockExtractor):
        """If PyMuPDF is missing, ImportError propagates."""
        from ingestors.extract_plaintext import plaintext_from_pdf

        mock_instance = MockExtractor.return_value
        mock_instance.pdf_to_text.side_effect = ImportError("No PyMuPDF")

        with self.assertRaises(ImportError):
            plaintext_from_pdf(b"%PDF-1.4")


# ---------------------------------------------------------------------------
# Tests: plaintext_from_jats
# ---------------------------------------------------------------------------

class TestPlaintextFromJats(unittest.TestCase):
    """Test JATS XML to plaintext extraction."""

    def test_extracts_body_text(self):
        """Extracts text from JATS <body> element."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article(
            "<sec><title>Introduction</title>"
            "<p>Fungi are organisms.</p>"
            "</sec>"
        )
        result = plaintext_from_jats(xml)
        self.assertIn("Introduction", result)
        self.assertIn("Fungi are organisms.", result)

    def test_excludes_abstract(self):
        """Abstract text should NOT appear in plaintext (body only)."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article("<sec><p>Body text.</p></sec>")
        result = plaintext_from_jats(xml)
        self.assertNotIn("Abstract text", result)
        self.assertIn("Body text.", result)

    def test_multiple_sections_separated(self):
        """Multiple sections produce text separated by blank lines."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article(
            "<sec><title>Description</title>"
            "<p>First section text.</p></sec>"
            "<sec><title>Etymology</title>"
            "<p>Second section text.</p></sec>"
        )
        result = plaintext_from_jats(xml)
        self.assertIn("First section text.", result)
        self.assertIn("Second section text.", result)
        # Sections should be separated
        self.assertIn("Description", result)
        self.assertIn("Etymology", result)

    def test_strips_inline_markup(self):
        """Inline XML elements like <italic> are stripped, text preserved."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article(
            "<sec><p>The species <italic>Fungus maximus</italic> is new.</p></sec>"
        )
        result = plaintext_from_jats(xml)
        self.assertIn("Fungus maximus", result)
        self.assertNotIn("<italic>", result)

    def test_skips_object_id_elements(self):
        """object-id elements (UUIDs, MycoBank IDs) are excluded."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article(
            "<sec><p>"
            "<object-id>UUID-1234</object-id>"
            "Visible text."
            "</p></sec>"
        )
        result = plaintext_from_jats(xml)
        self.assertIn("Visible text.", result)
        self.assertNotIn("UUID-1234", result)

    def test_no_body_raises_value_error(self):
        """XML without a <body> element raises ValueError."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<article><front></front></article>"
        )
        with self.assertRaises(ValueError):
            plaintext_from_jats(xml)

    def test_taxon_name_elements_preserved(self):
        """tp:taxon-name parts are included in extracted text."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article(
            f'<sec xmlns:tp="{TP_NS}"><p>'
            "<tp:taxon-name>"
            '<tp:taxon-name-part taxon-name-part-type="genus">Sidera</tp:taxon-name-part> '
            '<tp:taxon-name-part taxon-name-part-type="species">parallela</tp:taxon-name-part>'
            "</tp:taxon-name>"
            " is described."
            "</p></sec>"
        )
        result = plaintext_from_jats(xml)
        self.assertIn("Sidera", result)
        self.assertIn("parallela", result)
        self.assertIn("is described.", result)

    def test_figure_text_included(self):
        """Figure labels and captions appear in plaintext."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article(
            "<sec><p>Description text.</p>"
            '<fig id="F1"><label>Figure 1.</label>'
            "<caption><p>Microscopic view.</p></caption></fig>"
            "</sec>"
        )
        result = plaintext_from_jats(xml)
        self.assertIn("Description text.", result)
        self.assertIn("Figure 1.", result)
        self.assertIn("Microscopic view.", result)

    def test_empty_body_returns_empty(self):
        """An empty body element produces empty/whitespace-only text."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article("")
        result = plaintext_from_jats(xml)
        self.assertEqual(result.strip(), "")


# ---------------------------------------------------------------------------
# Tests: plaintext_from_efetch
# ---------------------------------------------------------------------------

class TestPlaintextFromEfetch(unittest.TestCase):
    """Test NCBI E-utilities efetch XML download and plaintext extraction."""

    # Minimal JATS XML returned by efetch (wrapped in pmc-articleset).
    _EFETCH_XML = (
        '<pmc-articleset>'
        f'<article xmlns:tp="{TP_NS}">'
        '<front><article-meta></article-meta></front>'
        '<body><sec><p>Efetch body text.</p></sec></body>'
        '</article>'
        '</pmc-articleset>'
    )

    def _mock_efetch(self, MockClient, xml=None, status=200):
        """Set up a mocked RateLimitedHttpClient returning XML."""
        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = status
        mock_response.text = xml if xml is not None else self._EFETCH_XML
        mock_instance.get.return_value = mock_response
        return mock_instance

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_fetches_and_parses_xml_for_pmcid(self, MockClient):
        """Downloads JATS XML from efetch and extracts body text."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        self._mock_efetch(MockClient)

        result = plaintext_from_efetch("PMC10858444")

        self.assertIn("Efetch body text.", result)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_url_contains_pmcid_and_retmode_xml(self, MockClient):
        """Efetch URL contains the PMCID and retmode=xml."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = self._mock_efetch(MockClient)
        plaintext_from_efetch("PMC10858444")

        url = mock_instance.get.call_args[0][0]
        self.assertIn("PMC10858444", url)
        self.assertIn("retmode=xml", url)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_strips_pmc_prefix(self, MockClient):
        """PMCID with or without 'PMC' prefix works."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = self._mock_efetch(MockClient)
        plaintext_from_efetch("10858444")

        url = mock_instance.get.call_args[0][0]
        self.assertIn("10858444", url)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_includes_api_key_when_provided(self, MockClient):
        """API key is appended to the efetch URL when provided."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = self._mock_efetch(MockClient)
        plaintext_from_efetch("PMC10858444", api_key="MYKEY123")

        url = mock_instance.get.call_args[0][0]
        self.assertIn("api_key=MYKEY123", url)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_no_api_key_omits_parameter(self, MockClient):
        """Without an API key, no api_key parameter appears in URL."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = self._mock_efetch(MockClient)
        plaintext_from_efetch("PMC10858444")

        url = mock_instance.get.call_args[0][0]
        self.assertNotIn("api_key=", url)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_http_error_raises_value_error(self, MockClient):
        """Non-200 HTTP response raises ValueError."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        self._mock_efetch(MockClient, xml="Not found", status=404)

        with self.assertRaises(ValueError) as ctx:
            plaintext_from_efetch("PMC99999999")
        self.assertIn("404", str(ctx.exception))

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_empty_response_raises_value_error(self, MockClient):
        """Empty response text raises ValueError."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        self._mock_efetch(MockClient, xml="")

        with self.assertRaises(ValueError) as ctx:
            plaintext_from_efetch("PMC10858444")
        self.assertIn("empty", str(ctx.exception).lower())

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_rate_limit_defaults_without_api_key(self, MockClient):
        """Without API key, rate limit should be 3 rps (NCBI default)."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        self._mock_efetch(MockClient)
        plaintext_from_efetch("PMC10858444")

        call_kwargs = MockClient.call_args
        rate_min = call_kwargs[1].get("rate_limit_min_ms")
        self.assertIsNotNone(rate_min)
        self.assertGreaterEqual(rate_min, 300)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_rate_limit_faster_with_api_key(self, MockClient):
        """With API key, rate limit should be 10 rps (NCBI with key)."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        self._mock_efetch(MockClient)
        plaintext_from_efetch("PMC10858444", api_key="KEY")

        call_kwargs = MockClient.call_args
        rate_min = call_kwargs[1].get("rate_limit_min_ms")
        self.assertIsNotNone(rate_min)
        self.assertLessEqual(rate_min, 200)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_uses_db_pmc(self, MockClient):
        """Efetch URL uses db=pmc for PMC articles."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = self._mock_efetch(MockClient)
        plaintext_from_efetch("PMC10858444")

        url = mock_instance.get.call_args[0][0]
        self.assertIn("db=pmc", url)


# ---------------------------------------------------------------------------
# Tests: edge cases and integration
# ---------------------------------------------------------------------------

class TestPlaintextEdgeCases(unittest.TestCase):
    """Edge cases and cross-function concerns."""

    def test_jats_with_treatment_sections(self):
        """JATS with TaxPub treatment sections extracts all text."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article(
            f'<tp:taxon-treatment xmlns:tp="{TP_NS}">'
            "<tp:nomenclature>"
            '<tp:taxon-name><tp:taxon-name-part taxon-name-part-type="genus">'
            "Mycena</tp:taxon-name-part></tp:taxon-name>"
            "</tp:nomenclature>"
            '<tp:treatment-sec sec-type="description">'
            "<title>Description.</title>"
            "<p>Pileus 5-15 mm diam.</p>"
            "</tp:treatment-sec>"
            "</tp:taxon-treatment>"
        )
        result = plaintext_from_jats(xml)
        self.assertIn("Mycena", result)
        self.assertIn("Pileus 5-15 mm diam.", result)

    def test_jats_unicode_preserved(self):
        """Unicode characters (accents, special chars) are preserved."""
        from ingestors.extract_plaintext import plaintext_from_jats

        xml = _jats_article(
            "<sec><p>Typus: Höhnel 1909, München. "
            "Described by Réblová et al.</p></sec>"
        )
        result = plaintext_from_jats(xml)
        self.assertIn("Höhnel", result)
        self.assertIn("München", result)
        self.assertIn("Réblová", result)

# ---------------------------------------------------------------------------
# Tests: plaintext_from_yedda
# ---------------------------------------------------------------------------

class TestPlaintextFromYedda(unittest.TestCase):
    """Tests for plaintext_from_yedda (tag stripping)."""

    def test_strips_single_block(self):
        """Strips tags from a single YEDDA block."""
        from ingestors.extract_plaintext import plaintext_from_yedda

        yedda = "[@Glomus hoi S.M. Berch & Trappe#Nomenclature*]"
        result = plaintext_from_yedda(yedda)
        self.assertEqual(result, "Glomus hoi S.M. Berch & Trappe")

    def test_multiple_blocks_separated_by_blank_lines(self):
        """Multiple blocks are joined with blank lines."""
        from ingestors.extract_plaintext import plaintext_from_yedda

        yedda = (
            "[@First block text#Nomenclature*]\n\n"
            "[@Second block text#Description*]"
        )
        result = plaintext_from_yedda(yedda)
        self.assertEqual(result, "First block text\n\nSecond block text")

    def test_preserves_newlines_within_block(self):
        """Newlines within a block are preserved."""
        from ingestors.extract_plaintext import plaintext_from_yedda

        yedda = "[@Line one\nLine two\nLine three#Description*]"
        result = plaintext_from_yedda(yedda)
        self.assertIn("Line one\nLine two\nLine three", result)

    def test_empty_string_returns_empty(self):
        """Empty input returns empty string."""
        from ingestors.extract_plaintext import plaintext_from_yedda

        self.assertEqual(plaintext_from_yedda(""), "")

    def test_no_yedda_blocks_returns_empty(self):
        """Non-YEDDA text returns empty string."""
        from ingestors.extract_plaintext import plaintext_from_yedda

        self.assertEqual(plaintext_from_yedda("Just plain text."), "")

    def test_skips_empty_blocks(self):
        """Blocks with only whitespace content are skipped."""
        from ingestors.extract_plaintext import plaintext_from_yedda

        yedda = "[@  #Misc-exposition*]\n\n[@Actual text#Description*]"
        result = plaintext_from_yedda(yedda)
        self.assertEqual(result, "Actual text")

    def test_unicode_preserved(self):
        """Unicode characters are preserved through stripping."""
        from ingestors.extract_plaintext import plaintext_from_yedda

        yedda = "[@Höhnel 1909, München#Nomenclature*]"
        result = plaintext_from_yedda(yedda)
        self.assertIn("Höhnel", result)
        self.assertIn("München", result)

    def test_realistic_annotation(self):
        """Realistic multi-block YEDDA annotation."""
        from ingestors.extract_plaintext import plaintext_from_yedda

        yedda = (
            "[@ISSN (print) 0093-4666\n"
            "© 2011. Mycotaxon, Ltd.#Misc-exposition*]\n\n"
            "[@Glomus hoi S.M. Berch & Trappe, Mycologia 77: 654. 1985.#Nomenclature*]\n\n"
            "[@Key characters: Spores formed singly.#Description*]"
        )
        result = plaintext_from_yedda(yedda)
        self.assertIn("ISSN (print) 0093-4666", result)
        self.assertIn("Glomus hoi", result)
        self.assertIn("Key characters:", result)
        # Should have blank-line separators
        self.assertIn("\n\n", result)


# ---------------------------------------------------------------------------
# Tests: count_yedda_tags
# ---------------------------------------------------------------------------

class TestCountYeddaTags(unittest.TestCase):
    """Tests for count_yedda_tags."""

    def test_counts_distinct_tags(self):
        """Counts distinct tag types."""
        from ingestors.extract_plaintext import count_yedda_tags

        yedda = (
            "[@Text one#Nomenclature*]\n\n"
            "[@Text two#Description*]\n\n"
            "[@Text three#Nomenclature*]"
        )
        tags, count = count_yedda_tags(yedda)
        self.assertEqual(tags, {"Nomenclature", "Description"})
        self.assertEqual(count, 3)

    def test_empty_string(self):
        """Empty string returns empty set and zero."""
        from ingestors.extract_plaintext import count_yedda_tags

        tags, count = count_yedda_tags("")
        self.assertEqual(tags, set())
        self.assertEqual(count, 0)

    def test_single_block(self):
        """Single block returns one tag."""
        from ingestors.extract_plaintext import count_yedda_tags

        tags, count = count_yedda_tags("[@Text#Etymology*]")
        self.assertEqual(tags, {"Etymology"})
        self.assertEqual(count, 1)

    def test_many_tags(self):
        """Multiple distinct tags are all counted."""
        from ingestors.extract_plaintext import count_yedda_tags

        yedda = "\n\n".join(
            f"[@Text {i}#{tag}*]"
            for i, tag in enumerate([
                "Nomenclature", "Description", "Etymology",
                "Key", "Misc-exposition", "Figure",
            ])
        )
        tags, count = count_yedda_tags(yedda)
        self.assertEqual(len(tags), 6)
        self.assertEqual(count, 6)


if __name__ == "__main__":
    unittest.main()
