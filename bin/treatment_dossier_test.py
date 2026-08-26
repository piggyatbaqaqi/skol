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
    DossierView,
    build_view,
    render_html,
    render_text,
)

pytestmark = pytest.mark.xfail(
    raises=NotImplementedError, strict=True,
    reason='dossier CLI: implementation follows test confirmation',
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

    def test_reports_suppressed_furniture_and_omissions(self) -> None:
        """taxon_ecb0124d's first gap holds 279 blocks beyond the cap.
        A page that showed 5 and said nothing would misreport it as a
        small gap.
        """
        from treatments_to_structured.dossier import Gap, SpanRef
        view = DossierView(dossier=build_view(*_treatment()).dossier)
        view.dossier.gaps = [Gap(
            after=SpanRef('description', 0, 1),
            before=SpanRef('notes', 900, 950),
            blocks=[], n_furniture=9, n_omitted=279)]
        html = render_html(view)
        assert '279' in html and '9' in html


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
