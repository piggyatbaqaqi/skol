"""Tests for :class:`PipelineState`."""

from __future__ import annotations

import io
from unittest import TestCase

from ingestors.spans import Span
from ingestors.yedda_tags import Tag, TaggedBlock

from .state import LabelContribution, PipelineState, SpanContribution


def _block(text: str, tag: Tag = Tag.MISC_EXPOSITION) -> TaggedBlock:
    return TaggedBlock(text=text, tag=tag)


def _span(start: int, end: int, label: str, src: str = "test") -> Span:
    return Span(
        start=start,
        end=end,
        label=label,
        text="x" * (end - start),
        source=src,
    )


class _FakeAttachment:
    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def read(self) -> bytes:
        return self._buf.read()


class _FakeDb:
    """Minimal couchdb-python-like stub used by attachment tests."""

    def __init__(self, attachments=None):
        self._attachments = attachments or {}

    def get_attachment(self, doc_id, name):
        data = self._attachments.get(doc_id, {}).get(name)
        return _FakeAttachment(data) if data is not None else None


class TestAttachmentLookup(TestCase):
    """``get_attachment`` checks three sources in priority order:
    cache → in-doc bytes → live couchdb_db."""

    def test_from_doc_attachments_dict_bytes(self) -> None:
        state = PipelineState(
            doc={"_id": "x", "_attachments": {"f.txt": b"hello"}},
        )
        self.assertEqual(state.get_attachment("f.txt"), b"hello")

    def test_from_doc_attachments_dict_str_encodes_utf8(self) -> None:
        state = PipelineState(
            doc={"_id": "x", "_attachments": {"f.txt": "héllo"}},
        )
        self.assertEqual(
            state.get_attachment("f.txt"), "héllo".encode("utf-8"),
        )

    def test_from_couchdb_db(self) -> None:
        state = PipelineState(
            doc={"_id": "x"},
            couchdb_db=_FakeDb({"x": {"f.txt": b"from-db"}}),
        )
        self.assertEqual(state.get_attachment("f.txt"), b"from-db")

    def test_cache_hit_does_not_re_fetch(self) -> None:
        """Second call returns the cached value even if the source
        disappears."""
        state = PipelineState(
            doc={"_id": "x", "_attachments": {"f.txt": b"hello"}},
        )
        state.get_attachment("f.txt")  # populates cache
        state.doc["_attachments"]["f.txt"] = b"changed"
        self.assertEqual(state.get_attachment("f.txt"), b"hello")

    def test_missing_raises_filenotfound(self) -> None:
        state = PipelineState(doc={"_id": "x"})
        with self.assertRaises(FileNotFoundError):
            state.get_attachment("absent.txt")


class TestLabelContributions(TestCase):
    """Section labels merge by highest-priority-wins (Commit-1 rule)."""

    def test_no_labelers_returns_empty(self) -> None:
        state = PipelineState()
        self.assertEqual(state.merged_section_labels(), [])

    def test_single_labeler_returns_its_blocks(self) -> None:
        state = PipelineState()
        blocks = [_block("foo"), _block("bar", Tag.DESCRIPTION)]
        state.add_section_labels("only", blocks, priority=5)
        self.assertEqual(state.merged_section_labels(), blocks)

    def test_higher_priority_wins(self) -> None:
        state = PipelineState()
        low_blocks = [_block("low")]
        hi_blocks = [_block("hi", Tag.NOMENCLATURE)]
        state.add_section_labels("low_src", low_blocks, priority=4)
        state.add_section_labels("hi_src", hi_blocks, priority=10)
        merged = state.merged_section_labels()
        self.assertEqual(merged, hi_blocks)

    def test_label_sources_lists_all_contributors(self) -> None:
        state = PipelineState()
        state.add_section_labels("a", [_block("x")], priority=1)
        state.add_section_labels("b", [_block("y")], priority=2)
        self.assertEqual(state.label_sources(), ["a", "b"])


class TestSpanContributions(TestCase):
    """Spans concatenate across contributors (Commit-1 rule)."""

    def test_no_detectors_returns_empty(self) -> None:
        state = PipelineState()
        self.assertEqual(state.merged_spans(), [])

    def test_concatenated_in_contribution_order(self) -> None:
        state = PipelineState()
        s1 = _span(0, 5, "DOI", src="regex")
        s2 = _span(10, 20, "TaxonName", src="gnfinder")
        state.add_spans("regex", [s1])
        state.add_spans("gnfinder", [s2])
        merged = state.merged_spans()
        self.assertEqual(merged, [s1, s2])

    def test_span_sources_lists_all_contributors(self) -> None:
        state = PipelineState()
        state.add_spans("a", [])
        state.add_spans("b", [])
        self.assertEqual(state.span_sources(), ["a", "b"])


class TestContributionDataclasses(TestCase):
    """LabelContribution / SpanContribution carry their fields."""

    def test_label_contribution_fields(self) -> None:
        lc = LabelContribution(source="x", blocks=[], priority=7)
        self.assertEqual(lc.source, "x")
        self.assertEqual(lc.blocks, [])
        self.assertEqual(lc.priority, 7)

    def test_span_contribution_fields(self) -> None:
        sc = SpanContribution(source="y", spans=[])
        self.assertEqual(sc.source, "y")
        self.assertEqual(sc.spans, [])
