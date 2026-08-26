#!/usr/bin/env python3
"""Tests for ``treatments_to_structured.dossier``.

Every fixture below is the *shape* of a real pathology from the memo,
reduced to the smallest `.ann` that reproduces it.  The point of the
module is that these shapes are invisible in brat, so the tests are
written as "would a reviewer have seen this?"
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treatments_to_structured.dossier import (  # noqa: E402
    SpanRef,
    build_dossier,
    gaps,
    labels_for_span,
    parse_blocks,
    treatment_spans,
)

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason='T3e: implementation follows test confirmation',
)


def _ann(*pairs: tuple) -> str:
    """Build an `.ann` from (text, label) pairs, blank-line separated."""
    return '\n\n'.join(f'[@{t}#{lab}*]' for t, lab in pairs)


# ---------------------------------------------------------------------------
# parse_blocks — delegation only; the behaviour lives in
# ingestors/yedda_tags_test.py, where the format's reader now lives
# ---------------------------------------------------------------------------


class TestParseBlocksDelegates:
    def test_it_is_the_shared_parser(self) -> None:
        """Guard against the copy coming back.  The block regex was
        duplicated in bin/migrate_labels.py and fixes/merge_yedda.py;
        a third copy here would be how the dossier's offsets silently
        drift from the ones stored on treatments.
        """
        from ingestors.yedda_tags import parse_yedda_blocks
        ann = _ann(('Pileus brown.', 'Description'))
        assert parse_blocks(ann) == parse_yedda_blocks(ann)


# ---------------------------------------------------------------------------
# treatment_spans — field boundaries deliberately dissolved
# ---------------------------------------------------------------------------


class TestTreatmentSpans:
    def test_collects_every_spans_field_tagged_with_its_name(self) -> None:
        got = treatment_spans({
            'description_spans': [{'start_char': 100, 'end_char': 200}],
            'notes_spans': [{'start_char': 300, 'end_char': 400}],
        })
        assert {(s.field, s.start) for s in got} == {
            ('description', 100), ('notes', 300)}

    def test_sorted_by_offset_across_fields(self) -> None:
        """Sorting across fields is what makes gaps visible.

        `taxon_ecb0124d`'s severed Notes showed up only because a
        `Figure-caption` sat between spans of two *different* fields;
        per-field iteration would never have put them side by side.
        """
        got = treatment_spans({
            'notes_spans': [{'start_char': 900, 'end_char': 950}],
            'description_spans': [{'start_char': 100, 'end_char': 200},
                                  {'start_char': 500, 'end_char': 600}],
        })
        assert [s.start for s in got] == [100, 500, 900]

    def test_string_offsets_are_coerced(self) -> None:
        """taxon_09b97d5f stores its diagnosis_spans offsets as
        strings; span_resolver already coerces rather than trusts.
        """
        got = treatment_spans({
            'diagnosis_spans': [{'start_char': '100', 'end_char': '200'}],
        })
        assert (got[0].start, got[0].end) == (100, 200)

    def test_a_span_missing_offsets_is_skipped_not_fatal(self) -> None:
        """A dossier is a reading aid.  Refusing to render a whole
        treatment because one span is malformed defeats the purpose —
        and a malformed span is itself worth seeing.
        """
        got = treatment_spans({
            'description_spans': [{'start_char': 100, 'end_char': 200},
                                  {'paragraph_number': 7}],
        })
        assert len(got) == 1

    def test_paragraph_number_is_carried(self) -> None:
        got = treatment_spans({
            'description_spans': [{'start_char': 1, 'end_char': 2,
                                   'paragraph_number': 37}],
        })
        assert got[0].paragraph == 37

    def test_non_span_fields_are_ignored(self) -> None:
        got = treatment_spans({
            'description': 'Pileus brown.',
            'description_spans': [{'start_char': 1, 'end_char': 2}],
            'ingest': {'_id': 'abc'},
        })
        assert len(got) == 1


# ---------------------------------------------------------------------------
# labels_for_span — a span crossing a block boundary is a finding
# ---------------------------------------------------------------------------


class TestLabelsForSpan:
    def test_single_block(self) -> None:
        blocks = parse_blocks(_ann(('Pileus brown.', 'Description')))
        span = SpanRef('description', blocks[0].start, blocks[0].end)
        assert labels_for_span(blocks, span) == ['Description']

    def test_a_span_crossing_blocks_reports_both(self) -> None:
        """Not an edge case — it means the extractor joined material
        the layout pass had separated, which is §15's element-join
        artifact seen from the other side.
        """
        ann = _ann(('Pileus brown.', 'Description'),
                   ('Stipe white.', 'Misc-exposition'))
        blocks = parse_blocks(ann)
        span = SpanRef('description', blocks[0].start, blocks[1].end)
        assert labels_for_span(blocks, span) == ['Description',
                                                 'Misc-exposition']

    def test_a_span_matching_no_block_returns_empty(self) -> None:
        blocks = parse_blocks(_ann(('Pileus brown.', 'Description')))
        assert labels_for_span(blocks, SpanRef('x', 10_000, 10_100)) == []


# ---------------------------------------------------------------------------
# gaps — the centrepiece, per adjacent pair
#
# Measured 2026-08-25 over 300 treatments: 94.2 % have at least one
# unclaimed block between their first and last span, median 9 and max
# 8 356.  Pooling them is unusable.  Between CONSECUTIVE spans, with
# furniture excluded, the median run is 1 block and 92 % are <= 5 —
# which is the shape every pathology this was built to catch has.
# ---------------------------------------------------------------------------


class TestGaps:
    @staticmethod
    def _fixture():
        """The `taxon_fdbd1b53` shape: a species heading hidden in a
        `Table` block sitting between two claimed spans.
        """
        ann = _ann(('Conidiophora numerosa.', 'Description'),
                   ('33. cocculi Stigmina sp. nov. Maculae amphigenae.',
                    'Table'),
                   ('Leaf spots amphigenous.', 'Description'))
        blocks = parse_blocks(ann)
        spans = [SpanRef('description', blocks[0].start, blocks[0].end),
                 SpanRef('description', blocks[2].start, blocks[2].end)]
        return blocks, spans

    def test_finds_the_unclaimed_block_between_two_spans(self) -> None:
        blocks, spans = self._fixture()
        got = gaps(blocks, spans)
        assert len(got) == 1
        assert [b.label for b in got[0].blocks] == ['Table']
        assert 'cocculi' in got[0].blocks[0].text

    def test_the_gap_names_the_spans_it_sits_between(self) -> None:
        """Per-pair is the whole point: a reviewer needs to know WHICH
        boundary the block fell through, not that one exists somewhere.
        """
        blocks, spans = self._fixture()
        g = gaps(blocks, spans)[0]
        assert (g.after.start, g.before.start) == (spans[0].start,
                                                   spans[1].start)

    def test_claimed_blocks_are_not_gaps(self) -> None:
        blocks, spans = self._fixture()
        got = gaps(blocks, spans)
        assert all(b.label != 'Description'
                   for g in got for b in g.blocks)

    def test_furniture_is_counted_but_not_listed(self) -> None:
        """94.2 % of treatments have a gap block and the commonest are
        Page-header and Bibliography.  Listing them buries the finding;
        dropping them silently would misreport the gap as empty.
        """
        ann = _ann(('A.', 'Description'), ('p. 42', 'Page-header'),
                   ('Refs', 'Bibliography'), ('B.', 'Description'))
        blocks = parse_blocks(ann)
        spans = [SpanRef('description', blocks[0].start, blocks[0].end),
                 SpanRef('description', blocks[3].start, blocks[3].end)]
        g = gaps(blocks, spans)[0]
        assert g.blocks == []
        assert g.n_furniture == 2

    def test_furniture_can_be_asked_for(self) -> None:
        ann = _ann(('A.', 'Description'), ('p. 42', 'Page-header'),
                   ('B.', 'Description'))
        blocks = parse_blocks(ann)
        spans = [SpanRef('description', blocks[0].start, blocks[0].end),
                 SpanRef('description', blocks[2].start, blocks[2].end)]
        g = gaps(blocks, spans, exclude=frozenset())[0]
        assert [b.label for b in g.blocks] == ['Page-header']

    def test_long_runs_are_truncated_not_dropped(self) -> None:
        """The tail is real — p99 is 1 135 blocks and the max is 8 350.
        A silent cap would report a 900-block gap as a 5-block one.
        """
        pairs = [('A.', 'Description')]
        pairs += [(f'noise {i}', 'Misc-exposition') for i in range(12)]
        pairs += [('B.', 'Description')]
        blocks = parse_blocks(_ann(*pairs))
        spans = [SpanRef('description', blocks[0].start, blocks[0].end),
                 SpanRef('description', blocks[-1].start, blocks[-1].end)]
        g = gaps(blocks, spans, max_blocks=5)[0]
        assert len(g.blocks) == 5
        assert g.n_omitted == 7

    def test_one_gap_per_pair_not_one_pooled(self) -> None:
        ann = _ann(('A.', 'Description'), ('X', 'Table'),
                   ('B.', 'Description'), ('Y', 'Figure-caption'),
                   ('C.', 'Description'))
        blocks = parse_blocks(ann)
        spans = [SpanRef('description', blocks[i].start, blocks[i].end)
                 for i in (0, 2, 4)]
        got = gaps(blocks, spans)
        assert [[b.label for b in g.blocks] for g in got] == [
            ['Table'], ['Figure-caption']]

    def test_pairs_with_nothing_between_yield_no_gap(self) -> None:
        ann = _ann(('A.', 'Description'), ('B.', 'Description'))
        blocks = parse_blocks(ann)
        spans = [SpanRef('description', b.start, b.end) for b in blocks]
        assert gaps(blocks, spans) == []

    def test_blocks_outside_the_span_range_are_not_gaps(self) -> None:
        """Material before the first span and after the last belongs to
        neighbouring treatments.  Reporting it would bury the signal
        under every other treatment in the document.
        """
        ann = _ann(('Previous treatment.', 'Description'),
                   ('Pileus brown.', 'Description'),
                   ('Next treatment.', 'Description'))
        blocks = parse_blocks(ann)
        spans = [SpanRef('description', blocks[1].start, blocks[1].end)]
        assert gaps(blocks, spans) == []

    def test_a_partially_overlapped_block_is_not_a_gap(self) -> None:
        """Only wholly unclaimed blocks count.  A block the treatment
        partly covers is already visible through its span.
        """
        ann = _ann(('Pileus brown and broad.', 'Description'),
                   ('Stipe white.', 'Description'))
        blocks = parse_blocks(ann)
        spans = [SpanRef('description', blocks[0].start,
                         blocks[0].start + 6),
                 SpanRef('description', blocks[1].start, blocks[1].end)]
        assert gaps(blocks, spans) == []

    def test_a_nested_span_cannot_manufacture_a_gap(self) -> None:
        """Measured at 0 % of 300 treatments — `*_spans` do not nest,
        unlike the brat annotations of memo section 0.1.  Pinned anyway:
        "does not happen today" is not "cannot happen", and pairing
        unmerged spans would put a phantom gap inside the outer one.
        """
        ann = _ann(('Pileus brown and broad.', 'Description'),
                   ('Later block.', 'Table'),
                   ('Stipe white.', 'Description'))
        blocks = parse_blocks(ann)
        outer = SpanRef('description', blocks[0].start, blocks[0].end)
        inner = SpanRef('description', blocks[0].start + 2,
                        blocks[0].start + 8)
        last = SpanRef('description', blocks[2].start, blocks[2].end)
        got = gaps(blocks, [outer, inner, last])
        assert [[b.label for b in g.blocks] for g in got] == [['Table']]

    def test_no_spans_yields_no_gaps(self) -> None:
        blocks = parse_blocks(_ann(('Pileus brown.', 'Description')))
        assert gaps(blocks, []) == []


# ---------------------------------------------------------------------------
# build_dossier — the whole picture
# ---------------------------------------------------------------------------


class TestBuildDossier:
    def test_assembles_spans_blocks_gaps_and_labels(self) -> None:
        ann = _ann(('Pileus brown.', 'Description'),
                   ('Fig. 1. Type.', 'Figure-caption'),
                   ('Notes on this.', 'Notes'))
        blocks = parse_blocks(ann)
        treatment = {
            '_id': 'taxon_a',
            'description_spans': [{'start_char': blocks[0].start,
                                   'end_char': blocks[0].end}],
            'notes_spans': [{'start_char': blocks[2].start,
                             'end_char': blocks[2].end}],
        }
        d = build_dossier(treatment, ann)
        assert d.treatment_id == 'taxon_a'
        assert len(d.spans) == 2
        assert [b.label for b in d.gaps] == ['Figure-caption']
        assert d.labels['description'] == ['Description']

    def test_a_treatment_with_no_spans_still_builds(self) -> None:
        """p2b is 35 482 treatments with almost nothing in them, and
        they are exactly the ones worth looking at.
        """
        d = build_dossier({'_id': 'taxon_empty'}, _ann(('X.', 'Table')))
        assert d.treatment_id == 'taxon_empty'
        assert d.spans == [] and d.gaps == []


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
