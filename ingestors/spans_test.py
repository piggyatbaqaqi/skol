"""Tests for ingestors/spans.py.

Covers Span construction/validation, spans_to_json/spans_from_json
round-trip, spans_to_bio BIO tagging, and resolve_conflicts overlap
resolution.

Run with: python -m pytest ingestors/spans_test.py -v
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.spans import (
    Span,
    resolve_conflicts,
    spans_from_json,
    spans_to_bio,
    spans_to_json,
)


# ---------------------------------------------------------------------------
# Span construction and validation
# ---------------------------------------------------------------------------


class TestSpanConstruction(unittest.TestCase):
    """Span validates its fields on construction."""

    def test_basic_construction(self) -> None:
        s = Span(start=0, end=5, label="TaxonName", text="Hello", source="gnfinder")
        self.assertEqual(s.start, 0)
        self.assertEqual(s.end, 5)
        self.assertEqual(s.length, 5)

    def test_default_confidence_and_metadata(self) -> None:
        s = Span(start=0, end=1, label="DOI", text="x", source="regex")
        self.assertEqual(s.confidence, 1.0)
        self.assertEqual(s.metadata, {})

    def test_negative_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            Span(start=-1, end=5, label="X", text="hello", source="s")

    def test_end_equal_to_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            Span(start=5, end=5, label="X", text="", source="s")

    def test_end_less_than_start_raises(self) -> None:
        with self.assertRaises(ValueError):
            Span(start=10, end=5, label="X", text="hello", source="s")

    def test_confidence_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            Span(start=0, end=5, label="X", text="hello", source="s", confidence=1.1)

    def test_confidence_below_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            Span(start=0, end=5, label="X", text="hello", source="s", confidence=-0.1)

    def test_overlaps_true(self) -> None:
        a = Span(start=0, end=10, label="X", text="0123456789", source="s")
        b = Span(start=5, end=15, label="Y", text="5678901234", source="s")
        self.assertTrue(a.overlaps(b))
        self.assertTrue(b.overlaps(a))

    def test_overlaps_false_adjacent(self) -> None:
        a = Span(start=0, end=5, label="X", text="01234", source="s")
        b = Span(start=5, end=10, label="Y", text="56789", source="s")
        self.assertFalse(a.overlaps(b))
        self.assertFalse(b.overlaps(a))

    def test_overlaps_false_disjoint(self) -> None:
        a = Span(start=0, end=3, label="X", text="012", source="s")
        b = Span(start=10, end=15, label="Y", text="01234", source="s")
        self.assertFalse(a.overlaps(b))


# ---------------------------------------------------------------------------
# spans_to_json / spans_from_json
# ---------------------------------------------------------------------------


class TestSpansJsonRoundTrip(unittest.TestCase):
    """spans_to_json + spans_from_json are inverse operations."""

    def _make_spans(self) -> list:
        return [
            Span(start=0, end=14, label="TaxonName", text="Pardosa moesta",
                 source="gnfinder", confidence=0.99,
                 metadata={"canonical": "Pardosa moesta", "cardinality": 2}),
            Span(start=100, end=112, label="DOI", text="10.1234/ab12",
                 source="regex"),
        ]

    def test_round_trip_preserves_fields(self) -> None:
        spans = self._make_spans()
        json_str = spans_to_json(spans, doc_id="abc123", source_attachment="article.txt")
        recovered = spans_from_json(json_str)
        self.assertEqual(len(recovered), 2)
        self.assertEqual(recovered[0].start, 0)
        self.assertEqual(recovered[0].end, 14)
        self.assertEqual(recovered[0].label, "TaxonName")
        self.assertEqual(recovered[0].confidence, 0.99)
        self.assertEqual(recovered[0].metadata["cardinality"], 2)

    def test_json_envelope_has_version(self) -> None:
        json_str = spans_to_json([], doc_id="x", source_attachment="article.txt")
        envelope = json.loads(json_str)
        self.assertEqual(envelope["version"], "1")

    def test_json_envelope_has_doc_id_and_source(self) -> None:
        json_str = spans_to_json([], doc_id="my_doc", source_attachment="article.txt")
        envelope = json.loads(json_str)
        self.assertEqual(envelope["doc_id"], "my_doc")
        self.assertEqual(envelope["source_attachment"], "article.txt")

    def test_empty_spans_round_trip(self) -> None:
        json_str = spans_to_json([], doc_id="x", source_attachment="f.txt")
        recovered = spans_from_json(json_str)
        self.assertEqual(recovered, [])

    def test_default_confidence_restored(self) -> None:
        spans = [Span(start=0, end=5, label="X", text="hello", source="regex")]
        json_str = spans_to_json(spans, doc_id="x", source_attachment="f.txt")
        # manually drop confidence from the JSON
        env = json.loads(json_str)
        del env["spans"][0]["confidence"]
        recovered = spans_from_json(json.dumps(env))
        self.assertEqual(recovered[0].confidence, 1.0)

    def test_missing_metadata_restored_as_empty_dict(self) -> None:
        spans = [Span(start=0, end=5, label="X", text="hello", source="regex")]
        json_str = spans_to_json(spans, doc_id="x", source_attachment="f.txt")
        env = json.loads(json_str)
        del env["spans"][0]["metadata"]
        recovered = spans_from_json(json.dumps(env))
        self.assertEqual(recovered[0].metadata, {})


# ---------------------------------------------------------------------------
# spans_to_bio
# ---------------------------------------------------------------------------


class TestSpansToBio(unittest.TestCase):
    """spans_to_bio produces correct BIO tags."""

    def test_no_spans_all_O(self) -> None:
        result = spans_to_bio("hello world", [])
        self.assertEqual(result, [("hello", "O"), ("world", "O")])

    def test_b_tag_on_first_token(self) -> None:
        text = "Amanita muscaria grows"
        span = Span(start=0, end=16, label="TaxonName", text="Amanita muscaria", source="gnfinder")
        result = spans_to_bio(text, [span])
        self.assertEqual(result[0], ("Amanita", "B-TaxonName"))
        self.assertEqual(result[1], ("muscaria", "I-TaxonName"))
        self.assertEqual(result[2], ("grows", "O"))

    def test_inner_span_gets_i_tag(self) -> None:
        text = "The Amanita muscaria grows"
        # span starts at "Amanita" (offset 4)
        span = Span(start=4, end=20, label="TaxonName", text="Amanita muscaria", source="gnfinder")
        result = spans_to_bio(text, [span])
        self.assertEqual(result[0], ("The", "O"))
        self.assertEqual(result[1], ("Amanita", "B-TaxonName"))
        self.assertEqual(result[2], ("muscaria", "I-TaxonName"))
        self.assertEqual(result[3], ("grows", "O"))

    def test_empty_text(self) -> None:
        result = spans_to_bio("", [])
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# resolve_conflicts
# ---------------------------------------------------------------------------


class TestResolveConflicts(unittest.TestCase):
    """resolve_conflicts keeps shorter spans when overlaps occur."""

    def test_no_overlaps_unchanged(self) -> None:
        spans = [
            Span(start=0, end=5, label="A", text="hello", source="s"),
            Span(start=10, end=15, label="B", text="world", source="s"),
        ]
        result = resolve_conflicts(spans)
        self.assertEqual(len(result), 2)

    def test_overlapping_keeps_shorter(self) -> None:
        long_span = Span(start=0, end=20, label="A", text="x" * 20, source="s")
        short_span = Span(start=5, end=10, label="B", text="xxxxx", source="s")
        result = resolve_conflicts([long_span, short_span])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "B")

    def test_equal_length_overlap_keeps_higher_confidence(self) -> None:
        low = Span(start=0, end=5, label="A", text="hello", source="s", confidence=0.5)
        high = Span(start=0, end=5, label="B", text="hello", source="s", confidence=0.9)
        result = resolve_conflicts([low, high])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, "B")

    def test_result_is_sorted_by_start(self) -> None:
        spans = [
            Span(start=20, end=25, label="C", text="xxxxx", source="s"),
            Span(start=0, end=5, label="A", text="hello", source="s"),
            Span(start=10, end=15, label="B", text="world", source="s"),
        ]
        result = resolve_conflicts(spans)
        starts = [s.start for s in result]
        self.assertEqual(starts, sorted(starts))

    def test_adjacent_spans_both_kept(self) -> None:
        """Adjacent (non-overlapping) spans are both preserved."""
        a = Span(start=0, end=5, label="A", text="hello", source="s")
        b = Span(start=5, end=10, label="B", text="world", source="s")
        result = resolve_conflicts([a, b])
        self.assertEqual(len(result), 2)

    def test_empty_input(self) -> None:
        self.assertEqual(resolve_conflicts([]), [])


if __name__ == "__main__":
    unittest.main()
