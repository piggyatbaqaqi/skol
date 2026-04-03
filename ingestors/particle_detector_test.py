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


if __name__ == "__main__":
    unittest.main()
