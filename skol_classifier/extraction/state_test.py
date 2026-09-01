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

    def test_label_contribution_blocks_fields(self) -> None:
        lc = LabelContribution(source="x", blocks=[], priority=7)
        self.assertEqual(lc.source, "x")
        self.assertEqual(lc.blocks, [])
        self.assertIsNone(lc.ann_text)
        self.assertEqual(lc.priority, 7)

    def test_label_contribution_ann_text_fields(self) -> None:
        lc = LabelContribution(source="x", ann_text="[@foo#X*]", priority=4)
        self.assertEqual(lc.source, "x")
        self.assertIsNone(lc.blocks)
        self.assertEqual(lc.ann_text, "[@foo#X*]")

    def test_label_contribution_requires_exactly_one_shape(self) -> None:
        """Passing both ``blocks`` and ``ann_text`` is a programmer
        error — the contribution is ambiguous."""
        with self.assertRaises(ValueError):
            LabelContribution(
                source="x", blocks=[_block("y")], ann_text="z",
            )
        # Neither is also invalid.
        with self.assertRaises(ValueError):
            LabelContribution(source="x")

    def test_label_contribution_to_yedda_text_passthrough(self) -> None:
        lc = LabelContribution(source="x", ann_text="raw yedda")
        self.assertEqual(lc.to_yedda_text(), "raw yedda")

    def test_label_contribution_to_yedda_text_serialises_blocks(self) -> None:
        block = _block("hello", Tag.DESCRIPTION)
        lc = LabelContribution(source="x", blocks=[block])
        # Should match tagged_blocks_to_yedda's output exactly.
        from ingestors.yedda_tags import tagged_blocks_to_yedda
        self.assertEqual(lc.to_yedda_text(), tagged_blocks_to_yedda([block]))

    def test_span_contribution_fields(self) -> None:
        sc = SpanContribution(source="y", spans=[])
        self.assertEqual(sc.source, "y")
        self.assertEqual(sc.spans, [])


class TestAnnTextContributions(TestCase):
    """``add_ann_text`` carries YEDDA text contributions lossless
    through the merge."""

    def test_no_labelers_returns_empty_string(self) -> None:
        self.assertEqual(PipelineState().merged_ann_text(), "")

    def test_single_ann_text_returns_verbatim(self) -> None:
        state = PipelineState()
        state.add_ann_text("classifier", "[@some#Nomenclature*]", priority=4)
        self.assertEqual(
            state.merged_ann_text(), "[@some#Nomenclature*]",
        )

    def test_blocks_contribution_serialises_to_yedda(self) -> None:
        state = PipelineState()
        blocks = [_block("foo", Tag.DESCRIPTION)]
        state.add_section_labels("x", blocks, priority=10)
        from ingestors.yedda_tags import tagged_blocks_to_yedda
        self.assertEqual(
            state.merged_ann_text(), tagged_blocks_to_yedda(blocks),
        )

    def test_blocks_higher_priority_beats_text(self) -> None:
        """A taxpub-style blocks contribution at priority 10 wins over
        a classifier-style text contribution at priority 4."""
        state = PipelineState()
        blocks = [_block("from-blocks", Tag.NOMENCLATURE)]
        state.add_section_labels("taxpub", blocks, priority=10)
        state.add_ann_text("classifier", "[@from-text#FIX*]", priority=4)
        from ingestors.yedda_tags import tagged_blocks_to_yedda
        self.assertEqual(
            state.merged_ann_text(), tagged_blocks_to_yedda(blocks),
        )

    def test_text_higher_priority_beats_blocks(self) -> None:
        """Symmetric: a higher-priority text contribution wins over
        lower-priority blocks (exercises both directions of the
        polymorphic merge)."""
        state = PipelineState()
        state.add_section_labels(
            "low_blocks", [_block("z")], priority=2,
        )
        state.add_ann_text("hi_text", "[@won#Description*]", priority=10)
        self.assertEqual(
            state.merged_ann_text(), "[@won#Description*]",
        )

    def test_merged_section_labels_empty_when_winner_is_text(self) -> None:
        """``merged_section_labels`` can't structure-parse YEDDA text
        (yet); when the winning contribution is text-only it
        gracefully returns ``[]``.  The lossless path is
        ``merged_ann_text``."""
        state = PipelineState()
        state.add_ann_text("classifier", "[@x#Y*]", priority=4)
        self.assertEqual(state.merged_section_labels(), [])


class TestWinningLabelSource:
    """Treatments record which extractor produced them.

    Measured 2026-09-01: `treatment_assembler` hard-codes
    `attachment_name` to a constant, so all 8 622 treatments of a
    `production_v4_1` run claimed `article.txt.ann` — including the
    ones derived from `article.xml` by the G.1 taxpub sweep.  The two
    extraction paths were indistinguishable in stored data, which made
    a v4/v4_1 comparison uninterpretable (memo §12.3.42).
    """

    def test_reports_the_source_of_the_winning_contribution(self):
        st = PipelineState(doc={'_id': 'd'})
        st.contribute_ann_text(text='[@a#Notes*]', source='low', priority=4)
        st.contribute_ann_text(text='[@b#Notes*]', source='high', priority=10)
        assert st.winning_label_source() == 'high'

    def test_the_source_matches_the_text_actually_used(self):
        """**The property that makes the field trustworthy.**  If the
        recorded source were computed independently of
        `merged_ann_text`, the two could disagree and the provenance
        would be a lie rather than a gap.
        """
        st = PipelineState(doc={'_id': 'd'})
        st.contribute_ann_text(text='[@low#Notes*]', source='low',
                               priority=4)
        st.contribute_ann_text(text='[@high#Notes*]', source='high',
                               priority=10)
        assert 'high' in st.merged_ann_text()
        assert st.winning_label_source() == 'high'

    def test_ties_resolve_the_same_way_as_the_text(self):
        """Equal priorities must not let the source and the text pick
        different contributions.
        """
        st = PipelineState(doc={'_id': 'd'})
        st.contribute_ann_text(text='[@first#Notes*]', source='first',
                               priority=5)
        st.contribute_ann_text(text='[@second#Notes*]', source='second',
                               priority=5)
        won = st.winning_label_source()
        assert won in ('first', 'second')
        assert won in st.merged_ann_text()

    def test_no_contributions_gives_none(self):
        assert PipelineState(doc={'_id': 'd'}).winning_label_source() is None
