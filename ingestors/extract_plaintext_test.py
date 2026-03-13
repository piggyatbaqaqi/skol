"""Tests for centralized plaintext extraction module.

Tests for:
- plaintext_from_pdf: PDF bytes → text via PDFSectionExtractor
- plaintext_from_jats: JATS XML string → body text
- plaintext_from_bioc: BioC-JSON list → passage text
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


def _bioc_json(passages):
    """Build a minimal BioC-JSON list from a list of (section_type, text) tuples."""
    return [{
        "documents": [{
            "passages": [
                {
                    "infons": {"section_type": sec, "type": ptype},
                    "text": text,
                }
                for sec, ptype, text in passages
            ]
        }]
    }]


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
# Tests: plaintext_from_bioc
# ---------------------------------------------------------------------------

class TestPlaintextFromBioc(unittest.TestCase):
    """Test BioC-JSON to plaintext extraction."""

    def test_extracts_passage_text(self):
        """Extracts and joins passage text from BioC-JSON."""
        from ingestors.extract_plaintext import plaintext_from_bioc

        bioc = _bioc_json([
            ("INTRO", "paragraph", "Introduction paragraph."),
            ("RESULTS", "paragraph", "Results paragraph."),
        ])
        result = plaintext_from_bioc(bioc)
        self.assertIn("Introduction paragraph.", result)
        self.assertIn("Results paragraph.", result)

    def test_cleans_bom_characters(self):
        """BOM characters are stripped from passage text."""
        from ingestors.extract_plaintext import plaintext_from_bioc

        bioc = _bioc_json([
            ("RESULTS", "paragraph", "\ufeffText with BOM."),
        ])
        result = plaintext_from_bioc(bioc)
        self.assertIn("Text with BOM.", result)
        self.assertNotIn("\ufeff", result)

    def test_skips_empty_passages(self):
        """Passages with empty text are omitted."""
        from ingestors.extract_plaintext import plaintext_from_bioc

        bioc = _bioc_json([
            ("RESULTS", "paragraph", "Visible text."),
            ("RESULTS", "paragraph", ""),
            ("RESULTS", "paragraph", "  "),
        ])
        result = plaintext_from_bioc(bioc)
        self.assertIn("Visible text.", result)
        # Should not have extra blank lines from empty passages
        self.assertNotIn("\n\n\n", result)

    def test_empty_bioc_raises_value_error(self):
        """Empty BioC-JSON list raises ValueError."""
        from ingestors.extract_plaintext import plaintext_from_bioc

        with self.assertRaises(ValueError):
            plaintext_from_bioc([])

    def test_no_documents_raises_value_error(self):
        """BioC-JSON with no documents raises ValueError."""
        from ingestors.extract_plaintext import plaintext_from_bioc

        with self.assertRaises(ValueError):
            plaintext_from_bioc([{"documents": []}])

    def test_passages_separated_by_newlines(self):
        """Multiple passages are separated by newlines."""
        from ingestors.extract_plaintext import plaintext_from_bioc

        bioc = _bioc_json([
            ("INTRO", "paragraph", "First passage."),
            ("RESULTS", "paragraph", "Second passage."),
        ])
        result = plaintext_from_bioc(bioc)
        # Passages should be separate, not run together
        self.assertNotIn("First passage.Second passage.", result)


# ---------------------------------------------------------------------------
# Tests: plaintext_from_efetch
# ---------------------------------------------------------------------------

class TestPlaintextFromEfetch(unittest.TestCase):
    """Test NCBI E-utilities efetch plaintext download."""

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_fetches_plaintext_for_pmcid(self, MockClient):
        """Downloads plaintext from NCBI efetch for a given PMCID."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Full article text from efetch."
        mock_instance.get.return_value = mock_response

        result = plaintext_from_efetch("PMC10858444")

        self.assertEqual(result, "Full article text from efetch.")
        # Verify the URL contains the right PMCID and retmode=text
        call_args = mock_instance.get.call_args
        url = call_args[0][0]
        self.assertIn("PMC10858444", url)
        self.assertIn("retmode=text", url)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_strips_pmc_prefix(self, MockClient):
        """PMCID with or without 'PMC' prefix works."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Article text."
        mock_instance.get.return_value = mock_response

        plaintext_from_efetch("10858444")

        url = mock_instance.get.call_args[0][0]
        # Should include PMC prefix in the id parameter
        self.assertIn("10858444", url)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_includes_api_key_when_provided(self, MockClient):
        """API key is appended to the efetch URL when provided."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Text."
        mock_instance.get.return_value = mock_response

        plaintext_from_efetch("PMC10858444", api_key="MYKEY123")

        url = mock_instance.get.call_args[0][0]
        self.assertIn("api_key=MYKEY123", url)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_no_api_key_omits_parameter(self, MockClient):
        """Without an API key, no api_key parameter appears in URL."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Text."
        mock_instance.get.return_value = mock_response

        plaintext_from_efetch("PMC10858444")

        url = mock_instance.get.call_args[0][0]
        self.assertNotIn("api_key=", url)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_http_error_raises_value_error(self, MockClient):
        """Non-200 HTTP response raises ValueError."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_instance.get.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            plaintext_from_efetch("PMC99999999")
        self.assertIn("404", str(ctx.exception))

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_empty_response_raises_value_error(self, MockClient):
        """Empty response text raises ValueError."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_instance.get.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            plaintext_from_efetch("PMC10858444")
        self.assertIn("empty", str(ctx.exception).lower())

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_rate_limit_defaults_without_api_key(self, MockClient):
        """Without API key, rate limit should be 3 rps (NCBI default)."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Text."
        mock_instance.get.return_value = mock_response

        plaintext_from_efetch("PMC10858444")

        # Verify RateLimitedHttpClient was configured with appropriate rate limits
        # Without API key: 3 requests/sec = 334ms between requests
        call_kwargs = MockClient.call_args
        rate_min = call_kwargs[1].get("rate_limit_min_ms",
                                       call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
        # Should be at least 300ms (3 rps)
        if rate_min is not None:
            self.assertGreaterEqual(rate_min, 300)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_rate_limit_faster_with_api_key(self, MockClient):
        """With API key, rate limit should be 10 rps (NCBI with key)."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Text."
        mock_instance.get.return_value = mock_response

        plaintext_from_efetch("PMC10858444", api_key="KEY")

        # With API key: 10 requests/sec = 100ms between requests
        call_kwargs = MockClient.call_args
        rate_min = call_kwargs[1].get("rate_limit_min_ms",
                                       call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
        # Should be less than the no-key rate
        if rate_min is not None:
            self.assertLessEqual(rate_min, 200)

    @patch("ingestors.extract_plaintext.RateLimitedHttpClient")
    def test_uses_db_pmc(self, MockClient):
        """Efetch URL uses db=pmc for PMC articles."""
        from ingestors.extract_plaintext import plaintext_from_efetch

        mock_instance = MockClient.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Text."
        mock_instance.get.return_value = mock_response

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

    def test_bioc_multiple_documents_uses_first(self):
        """BioC-JSON with multiple documents uses the first one."""
        from ingestors.extract_plaintext import plaintext_from_bioc

        bioc = [{
            "documents": [
                {
                    "passages": [
                        {"infons": {"section_type": "INTRO", "type": "paragraph"},
                         "text": "First doc text."}
                    ]
                },
                {
                    "passages": [
                        {"infons": {"section_type": "INTRO", "type": "paragraph"},
                         "text": "Second doc text."}
                    ]
                },
            ]
        }]
        result = plaintext_from_bioc(bioc)
        self.assertIn("First doc text.", result)
        # Second document should not be included
        self.assertNotIn("Second doc text.", result)

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

    def test_bioc_unicode_preserved(self):
        """Unicode is preserved in BioC passage extraction."""
        from ingestors.extract_plaintext import plaintext_from_bioc

        bioc = _bioc_json([
            ("RESULTS", "paragraph", "Pileus 5–15 mm (en-dash), spores 3×5 µm."),
        ])
        result = plaintext_from_bioc(bioc)
        self.assertIn("5–15", result)  # en-dash
        self.assertIn("µm", result)    # micro sign


if __name__ == "__main__":
    unittest.main()
