"""Tests for ingestors/particle_detector.py.

Covers static regex patterns (DOI, MB-number, Page-ref, GBIF-ID),
FungariumDetector with mocked Redis and personal_fungaria.json data,
and the detect_particles() entry point.

Run with: python -m pytest ingestors/particle_detector_test.py -v
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.particle_detector import (
    FungariumDetector,
    _build_fungarium_pattern,
    _load_fungaria_codes,
    detect_particles,
)
from ingestors.spans import Span


# ---------------------------------------------------------------------------
# _load_fungaria_codes
# ---------------------------------------------------------------------------


class TestLoadFungariaCodes(unittest.TestCase):

    def _make_redis(self, data: dict) -> MagicMock:
        client = MagicMock()
        client.get.return_value = json.dumps(data)
        return client

    def test_loads_codes_from_redis(self) -> None:
        data = {"institutions": {"NY": {}, "K": {}, "BPI": {}}}
        redis = self._make_redis(data)
        codes = _load_fungaria_codes(redis)
        self.assertIn("NY", codes)
        self.assertIn("K", codes)
        self.assertIn("BPI", codes)

    def test_sorted_longest_first(self) -> None:
        data = {"institutions": {"NY": {}, "DUKE": {}, "K": {}}}
        redis = self._make_redis(data)
        codes = _load_fungaria_codes(redis)
        lengths = [len(c) for c in codes]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_redis_unavailable_returns_empty_or_personal_only(self) -> None:
        client = MagicMock()
        client.get.side_effect = Exception("connection refused")
        codes = _load_fungaria_codes(client)
        # Personal fungaria may contribute; should not raise
        self.assertIsInstance(codes, list)

    def test_no_redis_falls_back_gracefully(self) -> None:
        codes = _load_fungaria_codes(None)
        self.assertIsInstance(codes, list)

    def test_personal_fungaria_codes_included(self) -> None:
        fake_personal = json.dumps([{"code": "YARROLL", "organization": "Test"}])
        with patch(
            "ingestors.particle_detector._PERSONAL_FUNGARIA_PATH"
        ) as mock_path:
            mock_path.exists.return_value = True
            mock_path.read_text.return_value = fake_personal
            codes = _load_fungaria_codes(None)
        self.assertIn("YARROLL", codes)


# ---------------------------------------------------------------------------
# _build_fungarium_pattern
# ---------------------------------------------------------------------------


class TestBuildFungariumPattern(unittest.TestCase):

    def test_returns_none_for_empty_codes(self) -> None:
        self.assertIsNone(_build_fungarium_pattern([]))

    def test_pattern_matches_code_followed_by_accession(self) -> None:
        pattern = _build_fungarium_pattern(["NY"])
        self.assertIsNotNone(pattern)
        m = pattern.search("NY 12345")
        self.assertIsNotNone(m)

    def test_longer_code_matches_before_shorter(self) -> None:
        pattern = _build_fungarium_pattern(["DUKE", "DU"])
        m = pattern.search("DUKE 1234")
        self.assertEqual(m.group(1), "DUKE")


# ---------------------------------------------------------------------------
# FungariumDetector.detect
# ---------------------------------------------------------------------------


class TestFungariumDetector(unittest.TestCase):

    def _detector(self, codes=None) -> FungariumDetector:
        if codes is None:
            codes = ["NY", "BPI", "DUKE"]
        with patch("ingestors.particle_detector._load_fungaria_codes", return_value=codes):
            return FungariumDetector(redis_client=None)

    def test_detects_basic_voucher(self) -> None:
        detector = self._detector()
        spans = detector.detect("Holotype: NY 1234.")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].label, "Fungarium-code")

    def test_metadata_has_code_and_accession(self) -> None:
        detector = self._detector()
        spans = detector.detect("NY 1234")
        self.assertEqual(spans[0].metadata["code"], "NY")
        self.assertEqual(spans[0].metadata["accession"], "1234")

    def test_materials_examined_confidence_is_high(self) -> None:
        detector = self._detector()
        spans = detector.detect("NY 1234", section_label="Materials-examined")
        self.assertEqual(spans[0].confidence, 0.9)

    def test_other_section_confidence_is_low(self) -> None:
        detector = self._detector()
        spans = detector.detect("NY 1234", section_label="Description")
        self.assertEqual(spans[0].confidence, 0.6)

    def test_no_codes_returns_empty(self) -> None:
        detector = self._detector(codes=[])
        spans = detector.detect("NY 1234")
        self.assertEqual(spans, [])

    def test_no_match_returns_empty(self) -> None:
        detector = self._detector()
        spans = detector.detect("Nothing interesting here.")
        self.assertEqual(spans, [])


# ---------------------------------------------------------------------------
# detect_particles — static patterns
# ---------------------------------------------------------------------------


class TestDetectParticlesStatic(unittest.TestCase):
    """detect_particles finds DOI, MB-number, Page-ref, GBIF-ID, ISSN."""

    def _detect(self, text: str) -> list:
        with patch("ingestors.particle_detector._load_fungaria_codes", return_value=[]):
            return detect_particles(text, redis_client=None)

    def test_detects_doi(self) -> None:
        spans = self._detect("See doi: 10.1000/xyz123 for details.")
        labels = [s.label for s in spans]
        self.assertIn("DOI", labels)

    def test_detects_mb_number_short_form(self) -> None:
        spans = self._detect("MB 123456 was registered.")
        labels = [s.label for s in spans]
        self.assertIn("MB-number", labels)

    def test_detects_mycobank_long_form(self) -> None:
        spans = self._detect("MycoBank # 123456 registered.")
        labels = [s.label for s in spans]
        self.assertIn("MB-number", labels)

    def test_detects_page_ref_p(self) -> None:
        spans = self._detect("See p. 123 for the key.")
        labels = [s.label for s in spans]
        self.assertIn("Page-ref", labels)

    def test_detects_page_ref_pp(self) -> None:
        spans = self._detect("Described on pp. 45-47.")
        labels = [s.label for s in spans]
        self.assertIn("Page-ref", labels)

    def test_detects_gbif_id(self) -> None:
        spans = self._detect("GBIF: 1234567 in the database.")
        labels = [s.label for s in spans]
        self.assertIn("GBIF-ID", labels)

    def test_detects_issn(self) -> None:
        spans = self._detect("Published in ISSN 0027-5514.")
        labels = [s.label for s in spans]
        self.assertIn("ISSN", labels)

    def test_detects_issn_with_colon(self) -> None:
        spans = self._detect("ISSN: 0027-5514")
        labels = [s.label for s in spans]
        self.assertIn("ISSN", labels)

    def test_detects_issn_check_digit_x(self) -> None:
        spans = self._detect("ISSN 1234-567X")
        labels = [s.label for s in spans]
        self.assertIn("ISSN", labels)

    def test_no_match_returns_empty(self) -> None:
        spans = self._detect("Nothing special in this text.")
        self.assertEqual(spans, [])

    def test_returns_span_objects(self) -> None:
        spans = self._detect("10.1000/abc")
        self.assertTrue(all(isinstance(s, Span) for s in spans))

    def test_source_is_regex(self) -> None:
        spans = self._detect("10.1000/abc")
        self.assertTrue(all(s.source == "regex" for s in spans))


class TestPDFPageMarker(unittest.TestCase):
    """The skol PDF extractor injects ``--- PDF Page N Label M ---``
    markers at every page break.  These are deterministic synthetic
    tokens (not heuristic running-header text), so they live with the
    other structured particles rather than in page_header_detector.

    See the v4 plan §1.B recommendation for the architectural split.
    """

    def _detect(self, text: str) -> list:
        with patch(
            "ingestors.particle_detector._load_fungaria_codes",
            return_value=[],
        ):
            return detect_particles(text, redis_client=None)

    def test_detects_marker_with_label(self) -> None:
        spans = self._detect("--- PDF Page 2 Label 2 ---")
        labels = [s.label for s in spans]
        self.assertIn("PDF-page-marker", labels)

    def test_detects_marker_without_label(self) -> None:
        """Some extractor configurations omit the ``Label N`` half."""
        spans = self._detect("--- PDF Page 12 ---")
        labels = [s.label for s in spans]
        self.assertIn("PDF-page-marker", labels)

    def test_metadata_carries_page_number(self) -> None:
        spans = self._detect("--- PDF Page 7 Label 7 ---")
        markers = [s for s in spans if s.label == "PDF-page-marker"]
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].metadata.get("page_number"), 7)
        self.assertEqual(markers[0].metadata.get("label_number"), 7)

    def test_metadata_omits_label_when_absent(self) -> None:
        spans = self._detect("--- PDF Page 12 ---")
        markers = [s for s in spans if s.label == "PDF-page-marker"]
        self.assertEqual(markers[0].metadata.get("page_number"), 12)
        self.assertNotIn("label_number", markers[0].metadata)

    def test_must_be_line_anchored(self) -> None:
        """Inline ``--- PDF Page 2 ---`` text inside a paragraph
        shouldn't trigger — markers always sit on their own line."""
        spans = self._detect(
            "Some body text --- PDF Page 2 --- inline reference."
        )
        labels = [s.label for s in spans]
        self.assertNotIn("PDF-page-marker", labels)

    def test_multiple_markers_in_document(self) -> None:
        text = (
            "Some body text.\n"
            "--- PDF Page 1 Label 1 ---\n"
            "More body text.\n"
            "--- PDF Page 2 Label 2 ---\n"
            "Even more body text.\n"
        )
        spans = self._detect(text)
        markers = [s for s in spans if s.label == "PDF-page-marker"]
        self.assertEqual(len(markers), 2)
        self.assertEqual(
            [m.metadata['page_number'] for m in markers],
            [1, 2],
        )


class TestIconographyHeader(unittest.TestCase):
    """Iconography-header pattern detects selected-icons section headings."""

    def _detect(self, text: str) -> list:
        with patch(
            "ingestors.particle_detector._load_fungaria_codes",
            return_value=[],
        ):
            return detect_particles(text, redis_client=None)

    def _labels(self, text: str) -> list:
        return [s.label for s in self._detect(text)]

    def test_selected_icons(self) -> None:
        self.assertIn("Iconography-header", self._labels("Selected icons."))

    def test_selected_icon_singular(self) -> None:
        self.assertIn("Iconography-header", self._labels("Selected icon:"))

    def test_selected_iconography(self) -> None:
        self.assertIn(
            "Iconography-header",
            self._labels("Selected iconography. Bres., Fung. Trid., t. 106."),
        )

    def test_selected_illustrations(self) -> None:
        self.assertIn(
            "Iconography-header",
            self._labels("Selected illustrations - Cooke, Ill. Brit. Fung."),
        )

    def test_iconography_standalone(self) -> None:
        self.assertIn("Iconography-header", self._labels("Iconography:"))

    def test_case_insensitive(self) -> None:
        self.assertIn("Iconography-header", self._labels("SELECTED ICONS."))

    def test_mid_sentence_not_matched(self) -> None:
        # "icons" without "selected" prefix is not an iconography header
        labels = self._labels("The icons used in this figure are standard.")
        self.assertNotIn("Iconography-header", labels)

    def test_span_covers_header_phrase(self) -> None:
        spans = self._detect("Selected icons. - Bres., Fung. Trid., t. 106.")
        icon_spans = [s for s in spans if s.label == "Iconography-header"]
        self.assertEqual(len(icon_spans), 1)
        self.assertIn("Selected icons", icon_spans[0].text)

    def test_in_bibliography_block(self) -> None:
        text = (
            "Selected icons. - Cooke, Ill. Brit. Fung. Pl. 476 (1881); "
            "Fries, Icon. t. 168, f. 2 (1867)."
        )
        self.assertIn("Iconography-header", self._labels(text))


class TestAuthorFootnote(unittest.TestCase):
    """Author-footnote pattern detects numbered contact/affiliation markers."""

    def _detect(self, text: str) -> list:
        with patch(
            "ingestors.particle_detector._load_fungaria_codes",
            return_value=[],
        ):
            return detect_particles(text, redis_client=None)

    def _labels(self, text: str) -> list:
        return [s.label for s in self._detect(text)]

    def test_single_footnote_at_line_start(self) -> None:
        self.assertIn("Author-footnote", self._labels("1) Department of Botany"))

    def test_multiple_footnotes(self) -> None:
        text = "1) Department of Botany\n2) E-mail: foo@example.com"
        labels = self._labels(text)
        self.assertEqual(labels.count("Author-footnote"), 2)

    def test_footnote_mid_sentence_not_matched(self) -> None:
        # "1)" not at line start — should not match
        labels = self._labels("See footnote 1) for details.")
        self.assertNotIn("Author-footnote", labels)

    def test_digit_zero_not_matched(self) -> None:
        # Pattern is [1-9], so "0)" is not a valid footnote marker
        self.assertNotIn("Author-footnote", self._labels("0) some text"))

    def test_multidigit_not_matched(self) -> None:
        # "12)" is not the pattern — only single digits
        self.assertNotIn("Author-footnote", self._labels("12) Department"))

    def test_span_covers_marker(self) -> None:
        spans = self._detect("1) Department of Botany")
        fn_spans = [s for s in spans if s.label == "Author-footnote"]
        self.assertEqual(len(fn_spans), 1)
        self.assertEqual(fn_spans[0].text, "1)")

    def test_after_newline_matches(self) -> None:
        text = "Some text above.\n2) Author address here."
        self.assertIn("Author-footnote", self._labels(text))


# ---------------------------------------------------------------------------
# CBS culture-collection numbers (Westerdijk Institute)
# ---------------------------------------------------------------------------


class TestCbsNumberDetection(unittest.TestCase):
    """CBS culture numbers — Westerdijk Fungal Biodiversity Institute
    (https://wi.knaw.nl/fungal_table).  External strain identifiers,
    same shape of label as DOI / MB-number / GBIF-ID.

    Two formats appear in our 24 681-hit corpus survey:
      - Old dotted: ``CBS 513.77``  (3 digits + period + 2-4 digits)
      - Modern:     ``CBS 136259``  (5-7 contiguous digits)
    Both may carry a trailing ``T`` / ``t`` marking the type strain.
    """

    def _detect(self, text: str):
        with patch("ingestors.particle_detector.FungariumDetector"):
            return detect_particles(text, redis_client=None)

    def _cbs(self, text: str):
        return [s for s in self._detect(text) if s.label == "CBS-number"]

    def test_dotted_format(self) -> None:
        """Older format: 3 digits + period + 2-4 digits."""
        cbs = self._cbs("Type culture CBS 513.77 was deposited.")
        self.assertEqual(len(cbs), 1)
        self.assertEqual(cbs[0].text, "CBS 513.77")

    def test_modern_numeric(self) -> None:
        """Modern format: 5-7 contiguous digits, no period."""
        cbs = self._cbs("Strain CBS 136259 was sequenced.")
        self.assertEqual(len(cbs), 1)
        self.assertEqual(cbs[0].text, "CBS 136259")

    def test_type_strain_suffix_no_space(self) -> None:
        """Type-strain marker 'T' immediately after the digits."""
        cbs = self._cbs("Holotype CBS 128831T described herein.")
        self.assertEqual(len(cbs), 1)
        self.assertIn("128831", cbs[0].text)

    def test_type_strain_suffix_with_space(self) -> None:
        """Type-strain marker 'T' with a space before it."""
        cbs = self._cbs("Holotype CBS 132036 T (ex-type).")
        self.assertEqual(len(cbs), 1)
        self.assertIn("132036", cbs[0].text)

    def test_case_insensitive(self) -> None:
        """``cbs`` lowercase still matches."""
        cbs = self._cbs("Listed as cbs 115.96 in catalog.")
        self.assertEqual(len(cbs), 1)

    def test_composite_with_slash_matches_cbs_only(self) -> None:
        """``CBS 144700/AP 6516`` — match the CBS portion only, leave
        AP 6516 alone (a different collection's range, picked up by
        FungariumDetector if registered)."""
        cbs = self._cbs("Strain CBS 144700/AP 6516 T ex-type.")
        self.assertEqual(len(cbs), 1)
        self.assertTrue(cbs[0].text.startswith("CBS"))
        self.assertIn("144700", cbs[0].text)
        self.assertNotIn("AP", cbs[0].text)

    def test_no_false_match_on_bare_acronym(self) -> None:
        """The string 'CBS' alone without a number is not a match."""
        cbs = self._cbs("The CBS Broadcasting Company logo.")
        self.assertEqual(cbs, [])

    def test_no_false_match_on_substring(self) -> None:
        """Word boundary: ``TCBS 12`` (a different acronym) doesn't
        match.  The 5-7-digit length requirement also rules out the
        2-digit '12'; this test keeps the boundary check honest."""
        cbs = self._cbs("Unrelated code TCBS 12 in dataset.")
        self.assertEqual(cbs, [])

    def test_metadata_carries_accession_modern(self) -> None:
        """Span metadata exposes the accession string for downstream
        linking to wi.knaw.nl/fungal_table."""
        cbs = self._cbs("Sample CBS 136259 type strain.")
        self.assertEqual(len(cbs), 1)
        self.assertEqual(cbs[0].metadata.get("accession"), "136259")

    def test_metadata_carries_accession_dotted(self) -> None:
        """Dotted format's accession captured verbatim as '513.77'."""
        cbs = self._cbs("Sample CBS 513.77 described.")
        self.assertEqual(len(cbs), 1)
        self.assertEqual(cbs[0].metadata.get("accession"), "513.77")


if __name__ == "__main__":
    unittest.main()
