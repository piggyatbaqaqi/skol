"""Tests for extract_mykoweb_pdf_metadata pure helpers.

The HTML fragments below are minimised representatives of the
actual rows in `/data/skol/www/mykoweb.com/systematics/*.html`, so a
green test run implies the extractor will handle the real files.

Network-touching code (filesystem walk, JSON emit) is exercised
end-to-end against the real site directory by hand; no fixture
here.
"""

import sys
import unittest
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_mykoweb_pdf_metadata import (  # type: ignore[import]
    classify_journals_row,
    classify_keys_row,
    classify_literature_row,
    classify_misc_row,
    extract_rows_from_html,
    parse_citation_tail,
)


# ---------------------------------------------------------------------------
# Stage 1 — raw-row extraction
# ---------------------------------------------------------------------------


class TestExtractRowsFromHtml(unittest.TestCase):
    """``extract_rows_from_html(html, source_html_name)`` walks the
    document and returns one dict per ``<a href>`` ending in
    ``.pdf``.  Off-site (http://...) PDF links are skipped because we
    can only map *local* PDFs to journals."""

    def test_simple_journals_li(self):
        """Single journals.html-shape row → one row dict with the
        leading journal name in li_text_before and the volume info
        in anchor_text."""
        html = """
        <ol>
          <li>Mycotaxon <a href="journals/Mycotaxon/Mycotaxon&#32;v001n1.pdf"
              target="_blank">Vol. 1 No. 1</a></li>
        </ol>
        """
        rows = extract_rows_from_html(html, 'systematics/journals.html')
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['source_html'], 'systematics/journals.html')
        self.assertEqual(row['href'], 'journals/Mycotaxon/Mycotaxon v001n1.pdf')
        self.assertEqual(row['anchor_text'], 'Vol. 1 No. 1')
        self.assertEqual(row['li_text_before'].strip(), 'Mycotaxon')
        self.assertEqual(row['li_text_after'].strip(), '')

    def test_literature_li_with_citation_tail(self):
        """literature.html-shape row → strong_text carries the
        author, li_text_after carries the post-link citation tail."""
        html = """
        <ol>
          <li><strong>Burt, E.A.</strong> (1922).
            <a href="literature/The&#32;North&#32;American&#32;Species&#32;of&#32;Clavaria.pdf">The
            North American Species of Clavaria with Illustrations of the
            Type Specimens</a>. Annals of the Missouri Botanical Garden
            9(1): 1-78.<br />
          </li>
        </ol>
        """
        rows = extract_rows_from_html(html, 'systematics/literature.html')
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(
            row['href'],
            'literature/The North American Species of Clavaria.pdf',
        )
        self.assertIn('North American Species of Clavaria',
                      row['anchor_text'])
        self.assertEqual(row['strong_text'], 'Burt, E.A.')
        self.assertEqual(row['year_match'], 1922)
        self.assertIn('Annals of the Missouri Botanical Garden 9(1): 1-78',
                      row['li_text_after'])

    def test_b_tag_treated_same_as_strong(self):
        """references.html uses ``<b>`` instead of ``<strong>`` for
        the bolded author / title — must be recognised."""
        html = """
        <li><a href="literature/foo.pdf"><b>C. H. Peck Species</b></a>
        <p>some description</p></li>
        """
        rows = extract_rows_from_html(html, 'systematics/references.html')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['strong_text'], 'C. H. Peck Species')

    def test_offsite_pdf_link_skipped(self):
        """``http(s)://`` PDF links are off-site (no local file to
        map); the extractor must skip them."""
        html = """
        <li><a href="http://example.com/some.pdf">External</a></li>
        <li><a href="literature/local.pdf">Local</a></li>
        """
        rows = extract_rows_from_html(html, 'systematics/keys.html')
        hrefs = [r['href'] for r in rows]
        self.assertEqual(hrefs, ['literature/local.pdf'])

    def test_non_pdf_link_skipped(self):
        """Non-PDF ``<a href>`` entries (CSS, other HTML pages,
        anchors) are ignored."""
        html = """
        <li><a href="keys.css">stylesheet</a></li>
        <li><a href="../index.html">home</a></li>
        <li><a href="literature/real.pdf">Real PDF</a></li>
        """
        rows = extract_rows_from_html(html, 'systematics/keys.html')
        self.assertEqual([r['href'] for r in rows], ['literature/real.pdf'])

    def test_two_pdfs_in_one_li(self):
        """references.html has a row where the description ``<p>``
        contains a second ``<a href>`` to a related PDF.  Both must
        be captured as separate rows."""
        html = """
        <li><a href="literature/CH&#32;Peck.pdf"><b>Peck Species</b></a>
            <p>...also an <a href="literature/Index&#32;to&#32;Peck.pdf">Index</a>
            by R.L. Gilbertson.</p>
        </li>
        """
        rows = extract_rows_from_html(html, 'systematics/references.html')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['href'], 'literature/CH Peck.pdf')
        self.assertEqual(rows[1]['href'], 'literature/Index to Peck.pdf')

    def test_html_entity_in_href_unescaped(self):
        """``&#32;`` (space) is the most common entity used in
        mykoweb hrefs; must be decoded so the path matches the
        filesystem."""
        html = '<li><a href="x/foo&#32;bar.pdf">x</a></li>'
        rows = extract_rows_from_html(html, 'systematics/literature.html')
        self.assertEqual(rows[0]['href'], 'x/foo bar.pdf')

    def test_year_regex_missing_doesnt_crash(self):
        """A literature.html-shape row without ``(YYYY)`` near the
        start (e.g. an undated reprint) returns year_match=None
        rather than crashing."""
        html = """
        <li><strong>Anon.</strong>
          <a href="literature/undated.pdf">Untitled</a> Some Journal.</li>
        """
        rows = extract_rows_from_html(html, 'systematics/literature.html')
        self.assertIsNone(rows[0]['year_match'])

    def test_em_strong_combo_for_author(self):
        """keys.html uses ``<em><strong>...</strong></em>`` (Armillaria
        row).  We only need the *text* of the bolded segment."""
        html = """
        <li><em><strong>Armillaria</strong></em>:
          <a href="../misc/armillaria.pdf">Field key</a></li>
        """
        rows = extract_rows_from_html(html, 'systematics/keys.html')
        self.assertEqual(rows[0]['strong_text'], 'Armillaria')

    def test_no_enclosing_li_still_yields_a_row(self):
        """A bare ``<a href>`` outside any ``<li>`` (rare but
        possible) yields a row with empty li_text_before /
        li_text_after."""
        html = '<p><a href="literature/loose.pdf">Loose</a></p>'
        rows = extract_rows_from_html(html, 'systematics/literature.html')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['href'], 'literature/loose.pdf')
        self.assertEqual(rows[0]['li_text_before'], '')
        self.assertEqual(rows[0]['li_text_after'], '')


# ---------------------------------------------------------------------------
# Stage 2a — citation-tail regex (literature.html the hard part)
# ---------------------------------------------------------------------------


class TestParseCitationTail(unittest.TestCase):
    """``parse_citation_tail(text)`` matches the bibliographic
    citation in the post-link tail of a literature.html ``<li>``
    against:

        CONTAINER VOL[(ISSUE)]: PAGES

    Returns a dict on match, None on miss (book / unstructured)."""

    def test_journal_vol_issue_pages(self):
        result = parse_citation_tail(
            '. Annals of the Missouri Botanical Garden 9(1): 1-78.'
        )
        assert result is not None
        self.assertEqual(result['container_title'],
                         'Annals of the Missouri Botanical Garden')
        self.assertEqual(result['volume'], '9')
        self.assertEqual(result['issue'], '1')
        self.assertEqual(result['pages'], '1-78')

    def test_journal_vol_no_issue(self):
        """No parenthetical issue is fine — vol + pages still
        identifies the article."""
        result = parse_citation_tail(' Ann. Missouri Bot. Gard. 1: 185-228.')
        assert result is not None
        self.assertEqual(result['container_title'], 'Ann. Missouri Bot. Gard.')
        self.assertEqual(result['volume'], '1')
        self.assertIsNone(result['issue'])
        self.assertEqual(result['pages'], '185-228')

    def test_en_dash_in_page_range(self):
        """Some citations use the en-dash ``–`` (U+2013) instead of
        a hyphen in the page range — must be accepted."""
        result = parse_citation_tail('Mycologia 67(3): 200–222.')
        assert result is not None
        self.assertEqual(result['pages'], '200–222')

    def test_trailing_megabyte_note_tolerated(self):
        """Many rows have a ``(17 MB)`` tail after the citation.
        The regex must anchor on the citation, not trip on the size
        annotation."""
        result = parse_citation_tail(
            ' Ann. Missouri Bot. Gard. 1: 185-228. (17 MB)'
        )
        assert result is not None
        self.assertEqual(result['container_title'], 'Ann. Missouri Bot. Gard.')
        self.assertEqual(result['volume'], '1')

    def test_empty_tail_returns_none(self):
        """A literature row with no post-link text is a book
        (anchor text IS the title); the regex must return None."""
        self.assertIsNone(parse_citation_tail(''))
        self.assertIsNone(parse_citation_tail('   '))

    def test_unstructured_tail_returns_none(self):
        """A trailing publisher blurb without vol/pages → None
        (caller treats as book)."""
        self.assertIsNone(parse_citation_tail(
            ' Published by the American Mycological Society.'
        ))


# ---------------------------------------------------------------------------
# Stage 2b — per-source classifiers
# ---------------------------------------------------------------------------


def _row(**fields: object) -> dict:
    """Build a raw row dict with sensible defaults for fields the
    classifier under test doesn't care about."""
    defaults: dict = {
        'source_html':    'systematics/literature.html',
        'href':           'literature/example.pdf',
        'anchor_text':    'Example Title',
        'li_text_before': '',
        'li_text_after':  '',
        'strong_text':    '',
        'year_match':     None,
    }
    defaults.update(fields)
    return defaults


class TestClassifyJournalsRow(unittest.TestCase):
    """journals.html rule: container = parent directory of href."""

    def test_mycotaxon_v001n1(self):
        row = _row(
            source_html='systematics/journals.html',
            href='journals/Mycotaxon/Mycotaxon v001n1.pdf',
            anchor_text='Vol. 1 No. 1',
            li_text_before='Mycotaxon',
        )
        out = classify_journals_row(row)
        self.assertEqual(out['kind'], 'journal')
        self.assertEqual(out['container_title'], 'Mycotaxon')
        self.assertEqual(out['volume_info'], 'Vol. 1 No. 1')
        self.assertEqual(out['title'], 'Mycotaxon Vol. 1 No. 1')
        self.assertEqual(out['source_html'], 'systematics/journals.html')

    def test_persoonia_subdir(self):
        row = _row(
            source_html='systematics/journals.html',
            href='journals/Persoonia/Persoonia v003.pdf',
            anchor_text='Vol. 3',
            li_text_before='Persoonia',
        )
        out = classify_journals_row(row)
        self.assertEqual(out['container_title'], 'Persoonia')


class TestClassifyLiteratureRow(unittest.TestCase):
    """literature.html rule: regex on li_text_after; fall back to
    'book' (no container) if it doesn't match."""

    def test_journal_article_full_citation(self):
        row = _row(
            source_html='systematics/literature.html',
            href='literature/clavaria.pdf',
            anchor_text='The North American Species of Clavaria',
            strong_text='Burt, E.A.',
            year_match=1922,
            li_text_after='. Annals of the Missouri Botanical Garden 9(1): 1-78.',
        )
        out = classify_literature_row(row)
        self.assertEqual(out['kind'], 'journal_article')
        self.assertEqual(out['title'],
                         'The North American Species of Clavaria')
        self.assertEqual(out['container_title'],
                         'Annals of the Missouri Botanical Garden')
        self.assertEqual(out['volume'], '9')
        self.assertEqual(out['issue'], '1')
        self.assertEqual(out['pages'], '1-78')
        self.assertEqual(out['author'], 'Burt, E.A.')
        self.assertEqual(out['year'], 1922)

    def test_book_no_citation_tail(self):
        """A literature row whose li_text_after doesn't match the
        citation regex is treated as a standalone book — title from
        anchor, container_title None."""
        row = _row(
            source_html='systematics/literature.html',
            href='literature/A Monograph of Favolaschia.pdf',
            anchor_text='A Monograph of Favolaschia',
            li_text_after='',
        )
        out = classify_literature_row(row)
        self.assertEqual(out['kind'], 'book')
        self.assertEqual(out['title'], 'A Monograph of Favolaschia')
        self.assertIsNone(out['container_title'])

    def test_book_with_publisher_blurb(self):
        """An unstructured publisher tail is still a book, not an
        article."""
        row = _row(
            source_html='systematics/literature.html',
            href='literature/some_book.pdf',
            anchor_text='Some Book',
            li_text_after=' Published by the American Mycological Society.',
        )
        out = classify_literature_row(row)
        self.assertEqual(out['kind'], 'book')


class TestClassifyKeysRow(unittest.TestCase):
    """keys.html rows are taxonomic keys, not citations — record
    title only, no container_title."""

    def test_basic_key_row(self):
        row = _row(
            source_html='systematics/keys.html',
            href='../misc/Agaricus_key.pdf',
            anchor_text='Trail Key to Common Agaricus Species',
            strong_text='Agaricus',
        )
        out = classify_keys_row(row)
        self.assertEqual(out['kind'], 'key')
        self.assertEqual(out['title'],
                         'Trail Key to Common Agaricus Species')
        self.assertNotIn('container_title', out)


class TestClassifyMiscRow(unittest.TestCase):
    """miscellanea.html / references.html rows: record anchor_text
    and li_text_after verbatim — small enough to hand-fix later."""

    def test_misc_row_keeps_raw_text(self):
        row = _row(
            source_html='systematics/miscellanea.html',
            href='../misc/TaxyBiblio.pdf',
            anchor_text='A Bibliography of Mushroom Books',
            li_text_after=' compiled by the Mycological Society.',
        )
        out = classify_misc_row(row)
        self.assertEqual(out['kind'], 'misc')
        self.assertEqual(out['title'], 'A Bibliography of Mushroom Books')
        self.assertEqual(out['li_text_after'],
                         ' compiled by the Mycological Society.')


if __name__ == '__main__':
    unittest.main()
