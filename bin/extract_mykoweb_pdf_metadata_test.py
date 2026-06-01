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
    _refine_container_title,
    classify_journals_row,
    classify_keys_row,
    classify_literature_row,
    classify_misc_row,
    extract_rows_from_html,
    merge_pdf_records,
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


# ---------------------------------------------------------------------------
# Round 2 — fixes from the live-data dry-run
# ---------------------------------------------------------------------------


class TestInlineTagSpacing(unittest.TestCase):
    """``get_text(strip=True)`` joins child text nodes with NO
    separator, so ``the <em>Ganodermataceae</em> Donk`` became
    ``theGanodermataceaeDonk`` in the JSON.  Use the separator-arg
    form so inline tags don't eat word boundaries."""

    def test_em_inside_anchor_preserves_word_boundary(self):
        html = (
            '<li><a href="literature/foo.pdf">A Nomenclatural Study of the '
            '<em>Ganodermataceae</em> Donk</a></li>'
        )
        rows = extract_rows_from_html(html, 'systematics/literature.html')
        self.assertEqual(
            rows[0]['anchor_text'],
            'A Nomenclatural Study of the Ganodermataceae Donk',
        )

    def test_em_in_li_text_after_preserves_boundary(self):
        """Same issue applies to the post-link tail — italics around
        a journal name shouldn't run into surrounding words."""
        html = (
            '<li><a href="literature/foo.pdf">A</a> <em>Persoonia</em> 3: 1-5.'
            '</li>'
        )
        rows = extract_rows_from_html(html, 'systematics/literature.html')
        self.assertIn('Persoonia 3: 1-5', rows[0]['li_text_after'])


class TestMergePdfRecords(unittest.TestCase):
    """When two HTML rows resolve to the same PDF relative path
    (e.g. the Melanogaster row has a "title" anchor and a "."
    anchor pointing at the same file, or the same PDF is referenced
    from both literature.html and references.html), merge into one
    record preferring the more-specific kind and the longer / more
    informative field values."""

    def test_two_anchors_same_pdf_merge_picks_better_title(self):
        """Melanogaster shape: one row has good title + book kind,
        the other has '.' title + journal_article kind (it carried
        the citation tail).  Merged: journal_article kind, the good
        title preserved."""
        first = {
            'kind':            'book',
            'title':           'Melanogaster',
            'container_title': None,
            'author':          'Zeller, S.M. & Dodge, C.W.',
            'year':            1936,
            'source_html':     'systematics/literature.html',
        }
        second = {
            'kind':            'journal_article',
            'title':           '.',
            'container_title': 'Ann. Missouri Bot. Gard.',
            'volume':          '23',
            'issue':           '636',
            'pages':           '655',
            'author':          'Zeller, S.M. & Dodge, C.W.',
            'year':            1936,
            'source_html':     'systematics/literature.html',
        }
        merged = merge_pdf_records(first, second)
        self.assertEqual(merged['kind'], 'journal_article')
        self.assertEqual(merged['title'], 'Melanogaster')
        self.assertEqual(merged['container_title'],
                         'Ann. Missouri Bot. Gard.')
        self.assertEqual(merged['volume'], '23')
        self.assertEqual(merged['issue'], '636')
        self.assertEqual(merged['pages'], '655')

    def test_book_then_journal_prefers_journal_kind(self):
        """journal_article is more specific than book — when in
        doubt, prefer it."""
        book = {'kind': 'book', 'title': 'X', 'source_html': 'a.html'}
        article = {
            'kind': 'journal_article',
            'title': 'X', 'container_title': 'Y',
            'volume': '1', 'pages': '1-2',
            'source_html': 'a.html',
        }
        self.assertEqual(merge_pdf_records(book, article)['kind'],
                         'journal_article')
        # Order shouldn't matter.
        self.assertEqual(merge_pdf_records(article, book)['kind'],
                         'journal_article')

    def test_merge_prefers_longer_title(self):
        a = {'kind': 'book', 'title': 'A', 'source_html': 'x.html'}
        b = {'kind': 'book', 'title': 'A Longer Title',
             'source_html': 'x.html'}
        merged = merge_pdf_records(a, b)
        self.assertEqual(merged['title'], 'A Longer Title')

    def test_merge_punctuation_only_title_loses(self):
        """Title='.' or whitespace-only is worse than any real
        anchor text — merge must drop it."""
        a = {'kind': 'book', 'title': '.', 'source_html': 'x.html'}
        b = {'kind': 'book', 'title': 'Real Title',
             'source_html': 'x.html'}
        merged = merge_pdf_records(a, b)
        self.assertEqual(merged['title'], 'Real Title')

    def test_merge_keeps_first_non_empty_for_unique_fields(self):
        """Fields present in only one record carry through to the
        merged result."""
        a = {'kind': 'book', 'title': 'X', 'author': 'Doe, J.',
             'source_html': 'x.html'}
        b = {'kind': 'book', 'title': 'X', 'year': 2000,
             'source_html': 'x.html'}
        merged = merge_pdf_records(a, b)
        self.assertEqual(merged['author'], 'Doe, J.')
        self.assertEqual(merged['year'], 2000)

    def test_merge_prefers_non_none_container_title(self):
        """A book record carries container_title=None; a journal
        record carries the real journal name.  Merged result must
        keep the real one."""
        book = {'kind': 'book', 'title': 'X', 'container_title': None,
                'source_html': 'x.html'}
        article = {'kind': 'journal_article', 'title': 'X',
                   'container_title': 'Some Journal',
                   'volume': '1', 'pages': '1-2',
                   'source_html': 'x.html'}
        merged = merge_pdf_records(book, article)
        self.assertEqual(merged['container_title'], 'Some Journal')


class TestRefineContainerTitle(unittest.TestCase):
    """When the citation regex matches a vol:pages inside a long
    interstitial-prose tail, the captured ``container_title`` runs
    on too long.  ``_refine_container_title`` clips on a clean
    sentence break (period + space + capitalised word) when the
    container is suspiciously long."""

    def test_short_container_unchanged(self):
        """Short, normal journal names pass through unchanged."""
        self.assertEqual(
            _refine_container_title('Ann. Missouri Bot. Gard.'),
            'Ann. Missouri Bot. Gard.',
        )
        self.assertEqual(
            _refine_container_title('Mycologia'),
            'Mycologia',
        )

    def test_abbreviation_periods_not_treated_as_sentence_breaks(self):
        """``Ann. Missouri Bot. Gard.`` has periods but they're
        abbreviation periods — the heuristic must not lop them off."""
        self.assertEqual(
            _refine_container_title(
                'Bull. Torrey bot. Club'
            ),
            'Bull. Torrey bot. Club',
        )

    def test_long_container_with_prose_clips_to_trailing_journal(self):
        """The Peck case: title's tail picked up interstitial prose
        before the real journal name."""
        result = _refine_container_title(
            'State Botanist, with bibliographic locations cited and '
            'some of the most obvious synonyms given. Report of the '
            'State Botanist'
        )
        self.assertEqual(result, 'Report of the State Botanist')

    def test_long_container_short_tail_unchanged(self):
        """If the chunk after the last sentence break is too short
        (e.g. 'Ann. Missouri Bot. Gard.' → tail 'Gard.' is only 5
        chars), don't clip — it's an abbreviation, not a sentence
        break."""
        self.assertEqual(
            _refine_container_title(
                'Some Long Container Name That Goes On A Bit. Gard.'
            ),
            'Some Long Container Name That Goes On A Bit. Gard.',
        )

    def test_none_passthrough(self):
        """Defensive: helper may be called with None / empty."""
        self.assertIsNone(_refine_container_title(None))
        self.assertEqual(_refine_container_title(''), '')


# ---------------------------------------------------------------------------
# Round 3 — disk-only PDF coverage (CAF, misc/Omphalina, OldBooks, etc.)
# ---------------------------------------------------------------------------


class TestRomanToInt(unittest.TestCase):
    """Roman-to-Arabic for the Omphalina-newsletter volume parser
    (filenames like ``O-VIII-2.pdf``).  Lenient — invalid input
    returns None rather than crashing."""

    def test_basic_numerals(self):
        from extract_mykoweb_pdf_metadata import _roman_to_int  # type: ignore[import]
        self.assertEqual(_roman_to_int('I'), 1)
        self.assertEqual(_roman_to_int('V'), 5)
        self.assertEqual(_roman_to_int('X'), 10)

    def test_compound_numerals(self):
        from extract_mykoweb_pdf_metadata import _roman_to_int
        self.assertEqual(_roman_to_int('VIII'), 8)
        self.assertEqual(_roman_to_int('IX'), 9)
        self.assertEqual(_roman_to_int('XII'), 12)

    def test_invalid_returns_none(self):
        from extract_mykoweb_pdf_metadata import _roman_to_int
        self.assertIsNone(_roman_to_int(''))
        self.assertIsNone(_roman_to_int('foo'))


class TestPrettyTitleFromStem(unittest.TestCase):
    """Filename-stem → human-readable title for disk-only PDFs.
    Replaces underscores with spaces, normalises whitespace."""

    def test_underscores_become_spaces(self):
        from extract_mykoweb_pdf_metadata import pretty_title_from_stem  # type: ignore[import]
        self.assertEqual(
            pretty_title_from_stem('Agaricus_albissimus'),
            'Agaricus albissimus',
        )

    def test_no_underscores_passthrough(self):
        from extract_mykoweb_pdf_metadata import pretty_title_from_stem
        self.assertEqual(
            pretty_title_from_stem(
                'A monographic study of the genera Crinipellis',
            ),
            'A monographic study of the genera Crinipellis',
        )

    def test_whitespace_normalised(self):
        from extract_mykoweb_pdf_metadata import pretty_title_from_stem
        self.assertEqual(
            pretty_title_from_stem('  trim_me  '),
            'trim me',
        )


class TestParseOmphalinaFilename(unittest.TestCase):
    """``misc/Omphalina/O-VIII-2.pdf`` → volume=8, issue=2."""

    def test_basic_parse(self):
        from extract_mykoweb_pdf_metadata import parse_omphalina_filename  # type: ignore[import]
        result = parse_omphalina_filename('O-VIII-2')
        assert result is not None
        self.assertEqual(result['volume'], '8')
        self.assertEqual(result['issue'], '2')
        self.assertEqual(result['title'], 'Omphalina Vol. 8 No. 2')

    def test_low_volume(self):
        from extract_mykoweb_pdf_metadata import parse_omphalina_filename
        result = parse_omphalina_filename('O-III-3')
        assert result is not None
        self.assertEqual(result['volume'], '3')
        self.assertEqual(result['issue'], '3')

    def test_non_omphalina_returns_none(self):
        from extract_mykoweb_pdf_metadata import parse_omphalina_filename
        self.assertIsNone(parse_omphalina_filename('something-else'))

    def test_malformed_returns_none(self):
        from extract_mykoweb_pdf_metadata import parse_omphalina_filename
        self.assertIsNone(parse_omphalina_filename('O-VIII'))


class TestApplyDiskRule(unittest.TestCase):
    """``apply_disk_rule(rule, file_relpath)`` produces records in
    the same shape as the HTML classifiers."""

    def test_caf_protologue(self):
        from extract_mykoweb_pdf_metadata import (  # type: ignore[import]
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES
                    if r['root'] == 'CAF/protologue')
        record = apply_disk_rule(
            rule, 'CAF/protologue/Agaricus_albissimus.pdf',
        )
        self.assertEqual(record['kind'], 'misc')
        self.assertEqual(record['container_title'], 'California Fungi')
        self.assertEqual(record['title'], 'Agaricus albissimus')

    def test_caf_funganordica_strips_prefix(self):
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES
                    if r['root'] == 'CAF/PDF/FungaNordica')
        record = apply_disk_rule(
            rule, 'CAF/PDF/FungaNordica/FungaNordica-Russula.pdf',
        )
        self.assertEqual(record['kind'], 'book')
        self.assertEqual(record['container_title'], 'Funga Nordica')
        self.assertEqual(record['title'], 'Russula')

    def test_caf_pdf_root_book(self):
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES if r['root'] == 'CAF/PDF')
        record = apply_disk_rule(
            rule,
            'CAF/PDF/A Systematic Study of Tricholoma in CA.pdf',
        )
        self.assertEqual(record['kind'], 'book')
        self.assertIsNone(record['container_title'])
        self.assertEqual(record['title'],
                         'A Systematic Study of Tricholoma in CA')

    def test_caf_keys(self):
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES if r['root'] == 'CAF/keys')
        record = apply_disk_rule(
            rule, 'CAF/keys/Arcangeliella_key.pdf',
        )
        self.assertEqual(record['kind'], 'key')
        self.assertEqual(record['title'], 'Arcangeliella key')

    def test_omphalina_with_vol_issue(self):
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES
                    if r['root'] == 'misc/Omphalina')
        record = apply_disk_rule(rule, 'misc/Omphalina/O-VIII-2.pdf')
        self.assertEqual(record['kind'], 'journal_article')
        self.assertEqual(record['container_title'], 'Omphalina')
        self.assertEqual(record['volume'], '8')
        self.assertEqual(record['issue'], '2')

    def test_omphalina_malformed_falls_back_to_misc(self):
        """Filename doesn't match O-<roman>-<num> — emit a misc
        record instead of fabricating vol/issue."""
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES
                    if r['root'] == 'misc/Omphalina')
        record = apply_disk_rule(rule, 'misc/Omphalina/special.pdf')
        self.assertEqual(record['kind'], 'misc')
        self.assertEqual(record['container_title'], 'Omphalina')

    def test_generic_book_directory(self):
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES if r['root'] == 'OldBooks')
        record = apply_disk_rule(
            rule, 'OldBooks/OurEdibleToadstoolsMushrooms.pdf',
        )
        self.assertEqual(record['kind'], 'book')
        self.assertIsNone(record['container_title'])
        self.assertEqual(record['title'],
                         'OurEdibleToadstoolsMushrooms')

    def test_record_carries_source_html_sentinel(self):
        """Records emitted from disk-extraction must carry a
        ``source_html`` value so downstream consumers can tell them
        apart from HTML-extracted ones.  Use the rule's root as the
        sentinel."""
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES if r['root'] == 'OldBooks')
        record = apply_disk_rule(rule, 'OldBooks/Foo.pdf')
        self.assertEqual(record['source_html'], 'OldBooks/')


class TestExtractDiskOnlyRecords(unittest.TestCase):
    """``extract_disk_only_records(site_root)`` walks ``_DISK_RULES``
    in order and emits the same ``{pdf_relpath: record}`` shape as
    the HTML pipeline."""

    def _build_fixture(self, tmp_root: Path) -> None:
        for relpath in [
            'CAF/protologue/Agaricus_albissimus.pdf',
            'CAF/PDF/FungaNordica/FungaNordica-Russula.pdf',
            'CAF/PDF/A monographic study of Tricholoma.pdf',
            'CAF/keys/Arcangeliella_key.pdf',
            'misc/Omphalina/O-VIII-2.pdf',
            'OldBooks/OurEdible.pdf',
        ]:
            p = tmp_root / relpath
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'%PDF-stub')

    def test_walks_every_disk_rule(self):
        from extract_mykoweb_pdf_metadata import extract_disk_only_records  # type: ignore[import]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            records = extract_disk_only_records(root)
        for expected in [
            'CAF/protologue/Agaricus_albissimus.pdf',
            'CAF/PDF/FungaNordica/FungaNordica-Russula.pdf',
            'CAF/PDF/A monographic study of Tricholoma.pdf',
            'CAF/keys/Arcangeliella_key.pdf',
            'misc/Omphalina/O-VIII-2.pdf',
            'OldBooks/OurEdible.pdf',
        ]:
            self.assertIn(expected, records)

    def test_caf_pdf_root_excludes_funganordica_subdir(self):
        from extract_mykoweb_pdf_metadata import extract_disk_only_records
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_fixture(root)
            records = extract_disk_only_records(root)
        rec = records[
            'CAF/PDF/FungaNordica/FungaNordica-Russula.pdf'
        ]
        self.assertEqual(rec['container_title'], 'Funga Nordica')
        loose = records[
            'CAF/PDF/A monographic study of Tricholoma.pdf'
        ]
        self.assertIsNone(loose['container_title'])

    def test_missing_directory_does_not_crash(self):
        from extract_mykoweb_pdf_metadata import extract_disk_only_records
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'OldBooks').mkdir()
            (root / 'OldBooks' / 'Foo.pdf').write_bytes(b'x')
            records = extract_disk_only_records(root)
        self.assertEqual(list(records.keys()), ['OldBooks/Foo.pdf'])


class TestApplyDiskRuleResidualRules(unittest.TestCase):
    """Round-3.1 — three additional rules covering the last 6 stragglers
    found in the live skol_dev data: loose PDFs in ``misc/`` (not
    ``misc/Omphalina/``), loose PDFs in a top-level ``literature/``
    directory (separate from ``systematics/literature/``), and disk
    PDFs in ``systematics/literature/`` not linked from
    literature.html."""

    def test_systematics_literature_disk_only(self):
        from extract_mykoweb_pdf_metadata import (  # type: ignore[import]
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES
                    if r['root'] == 'systematics/literature')
        record = apply_disk_rule(
            rule, 'systematics/literature/Polyporaceae of Wisconsin.pdf',
        )
        self.assertEqual(record['kind'], 'book')
        self.assertIsNone(record['container_title'])
        self.assertEqual(record['title'], 'Polyporaceae of Wisconsin')

    def test_toplevel_literature_dir(self):
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES if r['root'] == 'literature')
        record = apply_disk_rule(
            rule,
            'literature/The Agaricaceae of the Pacific Coast I.pdf',
        )
        self.assertEqual(record['kind'], 'book')
        self.assertIsNone(record['container_title'])

    def test_misc_loose_pdf(self):
        """``misc/Harry_D_Thiers.pdf`` (loose, NOT under
        misc/Omphalina/) → kind=misc, no container."""
        from extract_mykoweb_pdf_metadata import (
            apply_disk_rule, _DISK_RULES,
        )
        rule = next(r for r in _DISK_RULES
                    if r['root'] == 'misc' and r.get('recursive') is False)
        record = apply_disk_rule(rule, 'misc/Harry_D_Thiers.pdf')
        self.assertEqual(record['kind'], 'misc')
        self.assertIsNone(record['container_title'])
        self.assertEqual(record['title'], 'Harry D Thiers')
