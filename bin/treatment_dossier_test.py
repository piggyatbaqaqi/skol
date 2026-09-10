#!/usr/bin/env python3
"""Tests for bin/treatment_dossier.

The span/gap logic is tested in
``treatments_to_structured/dossier_test.py``.  This file covers what
the CLI adds: assembling the surrounding documents, and rendering —
where the corpus's own text is the adversary.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatment_dossier import (  # type: ignore[import]  # noqa: E402
    build_view,
    render_dossiers,
    render_html,
    render_text,
)


def _ann(*pairs: tuple) -> str:
    return '\n\n'.join(f'[@{t}#{lab}*]' for t, lab in pairs)


def _treatment(**over: object) -> dict:
    ann = _ann(('Pileus brown.', 'Description'),
               ('Fig. 1.', 'Figure-caption'),
               ('Notes here.', 'Notes'))
    from treatments_to_structured.dossier import parse_blocks
    b = parse_blocks(ann)
    doc = {
        '_id': 'taxon_a',
        'treatment': 'Amanita muscaria (L.) Lam.',
        'description': 'Pileus brown.',
        'description_spans': [{'start_char': b[0].start,
                               'end_char': b[0].end,
                               'paragraph_number': 11}],
        'notes': 'Notes here.',
        'notes_spans': [{'start_char': b[2].start, 'end_char': b[2].end,
                         'paragraph_number': 15}],
        'ingest': {'_id': 'src1'},
    }
    doc.update(over)
    return doc, ann


# ---------------------------------------------------------------------------
# build_view — everything optional, because p2b has almost nothing
# ---------------------------------------------------------------------------


class TestBuildView:
    def test_assembles_from_a_treatment_alone(self) -> None:
        doc, ann = _treatment()
        view = build_view(doc, ann)
        assert view.dossier.treatment_id == 'taxon_a'
        assert view.source == {} and view.status == {}

    def test_missing_status_is_not_an_error(self) -> None:
        """35 482 p2b treatments have no annotation status doc, and
        they are the ones most worth looking at.
        """
        doc, ann = _treatment()
        assert build_view(doc, ann, status=None).status == {}

    def test_merge_metric_is_computed_not_looked_up(self) -> None:
        """It is derived from description+diagnosis, so it is available
        even when no status doc exists to have cached it.
        """
        doc, ann = _treatment(
            description='Pileus. ' * 40 + 'Stipe. ' * 40)
        assert build_view(doc, ann).merge_metric is not None

    def test_siblings_are_carried_through(self) -> None:
        doc, ann = _treatment()
        sibs = [{'_id': 'taxon_b', 'treatment': 'Amanita phalloides'}]
        assert build_view(doc, ann, siblings=sibs).siblings == sibs


# ---------------------------------------------------------------------------
# Neighbours — §12.3.63.  A treatment's nomenclature can be stolen by the
# block that precedes it, and the theft is invisible from either treatment
# alone: the donor's trailing citation reads as a reference, and the victim
# reads as a paper that opened with a basionym.  1 102 treatments (2.2 %)
# open their nomenclature with a subordinate marker.  Putting the immediate
# neighbours' nomenclature lines in front of the reviewer is what makes the
# pair visible; the sibling list at the foot of the page did not, because it
# is sorted, long, and does not say which entries are adjacent.
# ---------------------------------------------------------------------------


class TestNeighbours:
    def test_next_is_the_nearest_sibling_after_this_one(self) -> None:
        doc, ann = _treatment()          # spans at paragraphs 11 and 15
        sibs = [{'_id': 'taxon_before', 'treatment': 'Boletus edulis',
                 'paragraph': 5},
                {'_id': 'taxon_next', 'treatment': 'Basionym. Phomopsis x.',
                 'paragraph': 19},
                {'_id': 'taxon_later', 'treatment': 'Russula emetica',
                 'paragraph': 31}]
        view = build_view(doc, ann, siblings=sibs)
        assert view.next_treatment is not None
        assert view.next_treatment['_id'] == 'taxon_next'

    def test_prev_is_the_nearest_sibling_before_this_one(self) -> None:
        """Symmetric, because a treatment can be the victim as easily
        as the donor and the reviewer needs to know which side."""
        doc, ann = _treatment()
        sibs = [{'_id': 'taxon_far', 'treatment': 'Boletus edulis',
                 'paragraph': 1},
                {'_id': 'taxon_prev', 'treatment': 'Diaporthe schini',
                 'paragraph': 7},
                {'_id': 'taxon_next', 'treatment': 'Russula emetica',
                 'paragraph': 19}]
        view = build_view(doc, ann, siblings=sibs)
        assert view.prev_treatment is not None
        assert view.prev_treatment['_id'] == 'taxon_prev'

    def test_no_siblings_means_no_neighbours(self) -> None:
        doc, ann = _treatment()
        view = build_view(doc, ann)
        assert view.prev_treatment is None
        assert view.next_treatment is None

    def test_a_sibling_with_no_paragraph_cannot_be_ordered(self) -> None:
        """`_sibling_index` yields ``paragraph: None`` for treatments
        whose spans carry no usable paragraph number.  Unordered is not
        the same as adjacent, so those are not offered as neighbours."""
        doc, ann = _treatment()
        sibs = [{'_id': 'taxon_x', 'treatment': 'Unplaceable',
                 'paragraph': None}]
        view = build_view(doc, ann, siblings=sibs)
        assert view.prev_treatment is None
        assert view.next_treatment is None

    def test_a_focal_treatment_with_no_paragraph_has_no_neighbours(
            self) -> None:
        doc, ann = _treatment(description_spans=[], notes_spans=[])
        sibs = [{'_id': 'taxon_next', 'treatment': 'X', 'paragraph': 19}]
        view = build_view(doc, ann, siblings=sibs)
        assert view.next_treatment is None

    def test_a_sibling_sharing_our_paragraph_is_not_a_neighbour(self) -> None:
        """Two treatments can report the same minimum paragraph.  That
        makes their order unknown, not adjacent."""
        doc, ann = _treatment()
        sibs = [{'_id': 'taxon_same', 'treatment': 'X', 'paragraph': 11}]
        view = build_view(doc, ann, siblings=sibs)
        assert view.prev_treatment is None
        assert view.next_treatment is None


class TestNeighboursAreRendered:
    """Asserting on the *neighbour section*, not on the page as a whole.

    The sibling list at the foot already prints every sibling's id and
    nomenclature, so `'Basionym.' in html` was true before any of this
    was written -- strict xfail caught five such tests passing against
    the old renderer.  A test that cannot fail against the thing being
    replaced is not testing the replacement.
    """

    NEIGH_TEXT = '--- neighbours in the source document ---'
    NEIGH_HTML = '<div class="neighbours">'

    def _sibs(self) -> list:
        return [{'_id': 'taxon_prev', 'treatment': 'Diaporthe schini '
                 'R.R. Gomes, C. Glienke & Crous, sp. nov.', 'paragraph': 7},
                {'_id': 'taxon_next', 'treatment': 'Basionym. Phomopsis '
                 'sclerotioides Kesteren, Neth. Jl. Pl. Path. 73: 115. 1967.',
                 'paragraph': 19}]

    def _section_text(self, text: str) -> str:
        i = text.index(self.NEIGH_TEXT)
        j = text.index('\n--- ', i + len(self.NEIGH_TEXT))
        return text[i:j]

    def _section_html(self, html: str) -> str:
        i = html.index(self.NEIGH_HTML)
        return html[i:html.index('</div>', i)]

    def test_text_section_names_both_neighbours(self) -> None:
        doc, ann = _treatment()
        sec = self._section_text(
            render_text(build_view(doc, ann, siblings=self._sibs())))
        assert 'Diaporthe schini' in sec
        assert 'Basionym. Phomopsis sclerotioides' in sec
        assert 'taxon_prev' in sec and 'taxon_next' in sec

    def test_html_section_names_both_neighbours(self) -> None:
        doc, ann = _treatment()
        sec = self._section_html(
            render_html(build_view(doc, ann, siblings=self._sibs())))
        assert 'Diaporthe schini' in sec
        assert 'Basionym. Phomopsis sclerotioides' in sec

    def test_the_subordinate_marker_survives_truncation(self) -> None:
        """The whole point is spotting `Basionym.` / `Type species.` /
        `Synonym.` at the START of a neighbour's nomenclature, so
        whatever truncation the renderer applies must keep the opening
        intact."""
        doc, ann = _treatment()
        sibs = [{'_id': 'taxon_next',
                 'treatment': 'Type species. ' + 'Xylaria longissima ' * 40,
                 'paragraph': 19}]
        view = build_view(doc, ann, siblings=sibs)
        assert 'Type species.' in self._section_text(render_text(view))
        assert 'Type species.' in self._section_html(render_html(view))

    def test_a_multiline_nomenclature_renders_on_one_line(self) -> None:
        """`treatment` is stored with the PDF's line breaks in it.  A
        neighbour line that wraps is not scannable."""
        doc, ann = _treatment()
        sibs = [{'_id': 'taxon_next',
                 'treatment': 'Basionym. Phomopsis\nsclerotioides Kesteren,',
                 'paragraph': 19}]
        sec = self._section_text(
            render_text(build_view(doc, ann, siblings=sibs)))
        line = next(x for x in sec.splitlines() if 'taxon_next' in x)
        assert 'Basionym. Phomopsis sclerotioides' in line

    def test_html_escapes_a_neighbour_name(self) -> None:
        """Same adversary as everywhere else in this file: the corpus
        contains `<`, `&` and `"` in real nomenclature strings."""
        doc, ann = _treatment()
        sibs = [{'_id': 'taxon_next',
                 'treatment': 'Basionym. <i>Phomopsis</i> & "x"',
                 'paragraph': 19}]
        sec = self._section_html(
            render_html(build_view(doc, ann, siblings=sibs)))
        assert '<i>Phomopsis</i>' not in sec
        assert '&lt;i&gt;' in sec

    def test_neighbours_appear_before_the_fields(self) -> None:
        """A reviewer reads top-down and stops when the treatment looks
        clean.  The sibling list at the foot is where this information
        already was, and it was not seen."""
        doc, ann = _treatment()
        text = render_text(build_view(doc, ann, siblings=self._sibs()))
        assert text.index(self.NEIGH_TEXT) < text.index('--- fields ---')

    def test_no_neighbour_section_when_there_are_none(self) -> None:
        doc, ann = _treatment()
        text = render_text(build_view(doc, ann))
        assert 'neighbour' not in text.lower()


# ---------------------------------------------------------------------------
# render_html — the corpus's own text is the adversary
# ---------------------------------------------------------------------------


class TestRenderHtml:
    def test_escapes_markup_in_treatment_text(self) -> None:
        """OCR'd text contains angle brackets and ampersands routinely,
        and `<` opens a tag.  Unescaped, a description reading
        `spores <3 um` silently swallows the rest of the page.
        """
        doc, ann = _treatment(description='spores <3 um & wide')
        html = render_html(build_view(doc, ann))
        assert '&lt;3 um &amp; wide' in html
        assert '<3 um' not in html

    def test_escapes_the_nomenclature_field_too(self) -> None:
        """Every field is OCR output, not just the description."""
        doc, ann = _treatment(treatment='Genus <sp> & Author')
        assert '<sp>' not in render_html(build_view(doc, ann))

    def test_is_self_contained(self) -> None:
        """Opened from file:// beside brat, where nothing external
        will load.
        """
        doc, ann = _treatment()
        html = render_html(build_view(doc, ann))
        for remote in ('http://', 'https://', '<script src',
                       '<link rel="stylesheet"'):
            assert remote not in html

    def test_names_the_treatment_and_its_fields(self) -> None:
        doc, ann = _treatment()
        html = render_html(build_view(doc, ann))
        assert 'taxon_a' in html
        assert 'Amanita muscaria' in html
        assert 'Pileus brown.' in html

    def test_shows_each_span_with_its_layout_label(self) -> None:
        """The single most useful thing brat cannot show."""
        doc, ann = _treatment()
        html = render_html(build_view(doc, ann))
        assert 'Description' in html and 'paragraph' in html.lower()

    def test_shows_the_gap_with_the_block_that_fell_through(self) -> None:
        doc, ann = _treatment()
        html = render_html(build_view(doc, ann))
        assert 'Figure-caption' in html and 'Fig. 1.' in html

    def test_gaps_are_collapsed_disclosures_in_the_span_flow(
        self,
    ) -> None:
        """Interleaved with the spans, not in a section of their own.

        A gap only means something next to the boundary it fell
        through, and 94.2 % of treatments have at least one -- so they
        default to closed or the page is unreadable on arrival.
        """
        doc, ann = _treatment()
        html = render_html(build_view(doc, ann))
        assert '<details' in html and '<summary' in html
        # no `open` attribute anywhere: every disclosure starts shut
        assert '<details open' not in html
        # the gap sits between the two spans it separates
        first = html.index('description')
        det = html.index('<details')
        last = html.rindex('notes')
        assert first < det < last

    def test_the_summary_says_what_is_inside_before_you_open_it(
        self,
    ) -> None:
        doc, ann = _treatment()
        html = render_html(build_view(doc, ann))
        summary = html[html.index('<summary'):html.index('</summary>')]
        assert 'Figure-caption' in summary or '1' in summary

    def test_each_span_carries_its_source_text_for_hover(self) -> None:
        """The span's own text, sliced from the .ann at its offsets --
        which is the thing a reviewer is checking the label against.
        """
        doc, ann = _treatment()
        html = render_html(build_view(doc, ann))
        assert 'Pileus brown.' in html and 'Notes here.' in html

    def test_hover_text_is_escaped_like_everything_else(self) -> None:
        """The hover panel is the one place raw .ann text reaches the
        page verbatim, so it is the likeliest escaping hole.
        """
        ann = _ann(('spores <3 um & wide', 'Description'))
        from treatments_to_structured.dossier import parse_blocks
        b = parse_blocks(ann)
        doc = {'_id': 'taxon_a',
               'description_spans': [{'start_char': b[0].start,
                                      'end_char': b[0].end}]}
        html = render_html(build_view(doc, ann))
        assert '&lt;3 um &amp; wide' in html
        assert '<3 um' not in html

    def test_hover_needs_no_javascript(self) -> None:
        """Self-contained means CSS-only: <details> is native HTML and
        the hover panel is a :hover rule.
        """
        doc, ann = _treatment()
        html = render_html(build_view(doc, ann))
        assert '<script' not in html
        assert ':hover' in html

    def test_reports_suppressed_furniture_and_omissions(self) -> None:
        """taxon_ecb0124d's first gap holds 279 blocks beyond the cap.
        A page that showed 5 and said nothing would misreport it as a
        small gap.
        """
        from treatments_to_structured.dossier import Gap
        doc, ann = _treatment()
        view = build_view(doc, ann)
        view.dossier.gaps = [Gap(
            after=view.dossier.spans[0],
            before=view.dossier.spans[1],
            blocks=[], n_furniture=9, n_omitted=279)]
        html = render_html(view)
        assert '279' in html and '9' in html

    def test_a_gap_not_anchored_to_a_span_is_still_shown(self) -> None:
        """Gaps are now positioned by the span they follow, so one
        whose `after` matches nothing would vanish from the page.

        A diagnostic tool that silently drops a finding is worse than
        one that shows it in the wrong place.
        """
        from treatments_to_structured.dossier import Gap, SpanRef
        doc, ann = _treatment()
        view = build_view(doc, ann)
        view.dossier.gaps = [Gap(
            after=SpanRef('nowhere', 99_000, 99_001),
            before=SpanRef('nowhere', 99_100, 99_101),
            blocks=[], n_furniture=0, n_omitted=42)]
        assert '42' in render_html(view)


# ---------------------------------------------------------------------------
# render_text — T3a's table is built from this, not from the HTML
# ---------------------------------------------------------------------------


class TestRenderText:
    def test_contains_no_markup(self) -> None:
        doc, ann = _treatment(description='spores <3 um')
        text = render_text(build_view(doc, ann))
        assert '<3 um' in text
        assert '&lt;' not in text and '<div' not in text

    def test_names_the_treatment_and_shows_gaps(self) -> None:
        doc, ann = _treatment()
        text = render_text(build_view(doc, ann))
        assert 'taxon_a' in text
        assert 'Figure-caption' in text

    def test_renders_a_treatment_with_no_spans(self) -> None:
        """p2b again — no spans, no gaps, and still worth a page."""
        text = render_text(build_view({'_id': 'taxon_empty'}, ''))
        assert 'taxon_empty' in text


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


class TestRenderDossiers:
    """The assembly loop, extracted so `brat_export --dossier-dir` can
    call it instead of copying it.

    The module docstring for `treatments_to_structured.dossier` says
    why: *"the same assembly backs the HTML page, T3a's merge-suspect
    table and T3d's cross-tab rather than each getting its own
    throwaway script."*  A second copy inside brat_export is exactly
    the drift that warns against.
    """

    def _server(self, treatments, ingest=None, status=None):
        return {
            'skol_experiments': {'production_v4': {'databases': {
                'treatments_prose': 'T', 'ingest': 'I',
                'features_status': 'S',
            }}},
            'T': treatments,
            'I': ingest or {},
            'S': status or {},
        }

    def test_writes_one_file_per_treatment(self, tmp_path) -> None:
        treatments = {'taxon_a': _treatment(_id='taxon_a')[0],
                      'taxon_b': _treatment(_id='taxon_b')[0]}
        n = render_dossiers(
            ['taxon_a', 'taxon_b'], server=self._server(treatments),
            experiment='production_v4', out_dir=tmp_path, siblings=False)
        assert n == 2
        assert sorted(p.name for p in tmp_path.glob('*.html')) == [
            'taxon_a.html', 'taxon_b.html']

    def test_a_missing_treatment_is_skipped_not_fatal(
            self, tmp_path) -> None:
        """A review set can name a treatment the DB no longer has; one
        bad id must not cost the other forty-four."""
        treatments = {'taxon_a': _treatment(_id='taxon_a')[0]}
        n = render_dossiers(
            ['taxon_a', 'taxon_gone'], server=self._server(treatments),
            experiment='production_v4', out_dir=tmp_path, siblings=False)
        assert n == 1
        assert [p.name for p in tmp_path.glob('*.html')] == ['taxon_a.html']

    def test_an_empty_id_list_writes_nothing(self, tmp_path) -> None:
        n = render_dossiers(
            [], server=self._server({}), experiment='production_v4',
            out_dir=tmp_path, siblings=False)
        assert n == 0
        assert list(tmp_path.glob('*')) == []

    def test_a_treatment_with_no_annotation_text_still_renders(
            self, tmp_path) -> None:
        """A dossier without the .ann still shows the fields and the
        metrics, which beats refusing to render -- and the treatments
        worth looking at are the ones missing the most."""
        treatments = {'taxon_a': _treatment(_id='taxon_a')[0]}
        n = render_dossiers(
            ['taxon_a'], server=self._server(treatments),
            experiment='production_v4', out_dir=tmp_path, siblings=False)
        assert n == 1
        assert (tmp_path / 'taxon_a.html').read_text()
