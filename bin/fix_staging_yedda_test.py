#!/usr/bin/env python3
"""Tests for fix_staging_yedda.py."""

import sys
from pathlib import Path
from typing import Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_staging_yedda import (  # noqa: E402
    _YEDDA_BLOCK_RE,
    _infer_outer_tag,
    add_page_markers,
    fix_malformed_blocks,
    fix_yedda,
    parse_page_markers,
    strip_page_markers,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _page(
    page_num: int,
    page_label: str,
    header: str,
    lines_before: int = 100,
    ctx_before: str = "",
    ctx_after: str = "",
) -> Tuple[int, str, str, int, str, str]:
    """Build a 6-tuple page entry for add_page_markers / fix_yedda."""
    return (page_num, page_label, header, lines_before, ctx_before, ctx_after)


def _make_yedda(*blocks: tuple) -> str:
    """Build a YEDDA string from (text, tag) tuples."""
    return "\n\n".join(f"[@{t}#{g}*]" for t, g in blocks) + "\n"


# ---------------------------------------------------------------------------
# _infer_outer_tag
# ---------------------------------------------------------------------------

class TestInferOuterTag:
    def test_materials_examined(self):
        assert (
            _infer_outer_tag(
                "Material examined: TURKEY. ...", "Misc-exposition"
            ) == "Materials-examined"
        )

    def test_materials_examined_plural(self):
        assert (
            _infer_outer_tag(
                "Materials examined: TURKEY. ...", "Misc-exposition"
            ) == "Materials-examined"
        )

    def test_key_to(self):
        assert (
            _infer_outer_tag(
                "Key to the species\n1. ...", "Page-header"
            ) == "Key"
        )

    def test_notes(self):
        assert (
            _infer_outer_tag(
                "Notes on taxonomy.", "Misc-exposition"
            ) == "Notes"
        )

    def test_distribution(self):
        assert (
            _infer_outer_tag("Distribution: Widespread.", "Description")
            == "Distribution"
        )

    def test_fallback(self):
        assert (
            _infer_outer_tag("Some unknown text.", "Misc-exposition")
            == "Misc-exposition"
        )

    def test_empty_text_returns_fallback(self):
        assert _infer_outer_tag("", "Description") == "Description"


# ---------------------------------------------------------------------------
# fix_malformed_blocks
# ---------------------------------------------------------------------------

class TestFixMalformedBlocks:
    def test_clean_yedda_unchanged(self):
        yedda = (
            "[@Arthonia fuscopurpurea (Tul.) R. Sant.#Nomenclature*]\n\n"
            "[@A description.#Description*]\n"
        )
        fixed, n = fix_malformed_blocks(yedda)
        assert n == 0
        assert (
            "[@Arthonia fuscopurpurea (Tul.) R. Sant.#Nomenclature*]" in fixed
        )
        assert "[@A description.#Description*]" in fixed

    def test_single_malformed_block_splits_into_two(self):
        """A block missing *] before a nested [@ splits into two blocks."""
        yedda = (
            "[@Material examined: TURKEY. (ANES 14198).\n\n"
            "[@Stigmidium leucophlebiae Cl. Roux & Triebel"
            "#Misc-exposition*]\n\n"
            "[@A description.#Notes*]\n"
        )
        fixed, n = fix_malformed_blocks(yedda)
        assert n == 1
        assert (
            "[@Material examined: TURKEY. (ANES 14198).#Materials-examined*]"
            in fixed
        )
        assert (
            "[@Stigmidium leucophlebiae Cl. Roux & Triebel#Misc-exposition*]"
            in fixed
        )
        assert "[@A description.#Notes*]" in fixed

    def test_key_malformed_block_uses_key_tag(self):
        """A Key block missing *] before a nested [@ gets tag Key."""
        yedda = (
            "[@Key to the species\n1. Spores in asci ...\n\n"
            "[@284 ... Author#Page-header*]\n\n"
            "[@Next section.#Misc-exposition*]\n"
        )
        fixed, n = fix_malformed_blocks(yedda)
        assert n == 1
        assert "[@Key to the species\n1. Spores in asci ...#Key*]" in fixed
        assert "[@284 ... Author#Page-header*]" in fixed
        assert "[@Next section.#Misc-exposition*]" in fixed

    def test_empty_outer_block_discarded(self):
        """If nothing precedes the embedded [@, no outer block is emitted."""
        yedda = "[@\n\n[@Inner text#Notes*]\n\n[@After.#Description*]\n"
        fixed, n = fix_malformed_blocks(yedda)
        assert n == 1
        assert "[@Inner text#Notes*]" in fixed
        assert "[@After.#Description*]" in fixed

    def test_multiple_malformed_blocks(self):
        """Two malformed blocks in one document are both fixed."""
        yedda = (
            "[@Block A.\n\n[@Inner A#Misc-exposition*]\n\n"
            "[@Block B.\n\n[@Inner B#Notes*]\n\n"
            "[@Clean block.#Description*]\n"
        )
        fixed, n = fix_malformed_blocks(yedda)
        assert n == 2
        assert "[@Inner A#Misc-exposition*]" in fixed
        assert "[@Inner B#Notes*]" in fixed
        assert "[@Clean block.#Description*]" in fixed

    def test_multiline_clean_block_preserved(self):
        """Legitimate multi-line blocks (no embedded [@) are not changed."""
        yedda = "[@Line one\nLine two\nLine three.#Description*]\n"
        fixed, n = fix_malformed_blocks(yedda)
        assert n == 0
        assert "Line one" in fixed
        assert "Line three." in fixed

    def test_output_ends_with_newline(self):
        yedda = "[@Text.#Nomenclature*]\n"
        fixed, _ = fix_malformed_blocks(yedda)
        assert fixed.endswith("\n")

    def test_empty_input(self):
        fixed, n = fix_malformed_blocks("")
        assert fixed == ""
        assert n == 0


# ---------------------------------------------------------------------------
# parse_page_markers
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE_TXT = """\
--- PDF Page 1 Label 1 ---
ISSN (print) 0093-4666

Some front matter text.

--- PDF Page 2 Label 2 ---
278 ... Halici, Candan & Türk
Article body continues here.

--- PDF Page 3 Label 3 ---
Key to peltigericolous fungi in Turkey ... 279
More body.

--- PDF Page 13 Label roman_xiii ---

"""


class TestParsePageMarkers:
    def test_parses_three_pages(self):
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert len(pages) == 3  # page 13 has no content line

    def test_tuple_has_six_elements(self):
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert all(len(p) == 6 for p in pages)

    def test_page_numbers_correct(self):
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert [p[0] for p in pages] == [1, 2, 3]

    def test_page_labels_correct(self):
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert [p[1] for p in pages] == ["1", "2", "3"]

    def test_header_texts_correct(self):
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        headers = [p[2] for p in pages]
        assert headers[0] == "ISSN (print) 0093-4666"
        assert headers[1] == "278 ... Halici, Candan & Türk"
        assert headers[2] == "Key to peltigericolous fungi in Turkey ... 279"

    def test_lines_before_first_page_is_zero(self):
        """First page marker is at line 0 so lines_before == 0."""
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert pages[0][3] == 0

    def test_lines_before_subsequent_pages(self):
        """lines_before for page 2 equals the line distance between markers."""
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        # Page 1 marker is at line 0, page 2 marker is at line 5.
        assert pages[1][3] == 5

    def test_context_before_first_page_empty(self):
        """No lines before the first marker → context_before is empty."""
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert pages[0][4] == ""

    def test_context_before_second_page(self):
        """context_before for page 2 contains text from the previous page."""
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert "Some front matter text." in pages[1][4]

    def test_context_after_first_page(self):
        """context_after contains lines after the header line."""
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert "Some front matter text." in pages[0][5]

    def test_context_after_does_not_include_next_marker(self):
        """context_after stops at the next --- PDF Page --- line."""
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        assert "--- PDF Page" not in pages[0][5]

    def test_empty_string(self):
        assert parse_page_markers("") == []

    def test_no_markers(self):
        assert parse_page_markers("plain text\nno markers\n") == []

    def test_roman_label(self):
        txt = "--- PDF Page 5 Label xv ---\nRunning head\n"
        pages = parse_page_markers(txt)
        assert len(pages) == 1
        assert pages[0][0] == 5
        assert pages[0][1] == "xv"
        assert pages[0][2] == "Running head"


# ---------------------------------------------------------------------------
# add_page_markers — basic behaviour
# ---------------------------------------------------------------------------

class TestAddPageMarkers:
    def test_adds_marker_via_after_vote(self):
        """after_vote fires when the block's text begins with ctx_after."""
        yedda = _make_yedda(
            ("ISSN (print) 0093-4666", "Misc-exposition"),
            (
                "278 ... Halici, Candan & Turk\nArticle body.",
                "Misc-exposition",
            ),
        )
        # ctx_after starts with the header text (parse_page_markers behaviour).
        pages = [_page(
            2, "2", "278 ... Halici, Candan & Turk",
            ctx_before="ISSN (print) 0093-4666",
            ctx_after="278 ... Halici, Candan & Turk Article body.",
        )]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert (
            "--- PDF Page 2 Label 2 ---\n278 ... Halici, Candan & Turk"
            in marked
        )
        assert "#Page-header*]" in marked

    def test_relabels_to_page_header(self):
        """Matched block tag is changed to Page-header."""
        yedda = _make_yedda(("278 ... Author", "Misc-exposition"))
        pages = [_page(2, "2", "278 ... Author",
                       ctx_after="278 ... Author")]
        marked, _ = add_page_markers(yedda, pages)
        assert (
            "[@--- PDF Page 2 Label 2 ---\n278 ... Author#Page-header*]"
            in marked
        )

    def test_already_marked_block_not_doubled(self):
        """Blocks already carrying a marker are not marked again."""
        yedda = _make_yedda(
            ("--- PDF Page 2 Label 2 ---\n278 ... Author", "Page-header"),
        )
        pages = [_page(2, "2", "278 ... Author",
                       ctx_after="278 ... Author")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 0
        assert marked.count("--- PDF Page 2 Label 2 ---") == 1

    def test_no_match_returns_yedda_unchanged(self):
        """No block matches ctx_after or ctx_before → no marker added."""
        yedda = _make_yedda(("Completely different text.", "Misc-exposition"))
        pages = [_page(2, "2", "278 ... Author",
                       ctx_after="278 ... Author some more text")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 0
        assert "--- PDF Page" not in marked

    def test_page_skipped_if_no_context(self):
        """Page with empty ctx_before and ctx_after is silently skipped."""
        yedda = _make_yedda(("278 ... Author", "Misc-exposition"))
        pages = [_page(2, "2", "278 ... Author",
                       ctx_before="", ctx_after="")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 0
        assert "--- PDF Page" not in marked

    def test_multiple_pages_multiple_matches(self):
        """Two pages are each placed on the correct block."""
        yedda = _make_yedda(
            ("End of page 1.", "Misc-exposition"),
            ("Header page 2\nFirst line of page 2.", "Misc-exposition"),
            ("End of page 2.", "Description"),
            ("Header page 3\nFirst line of page 3.", "Misc-exposition"),
        )
        pages = [
            _page(2, "2", "Header page 2",
                  ctx_before="End of page 1.",
                  ctx_after="Header page 2 First line of page 2."),
            _page(3, "3", "Header page 3",
                  ctx_before="End of page 2.",
                  ctx_after="Header page 3 First line of page 3."),
        ]
        marked, n = add_page_markers(yedda, pages)
        assert n == 2
        assert "--- PDF Page 2 Label 2 ---" in marked
        assert "--- PDF Page 3 Label 3 ---" in marked

    def test_empty_pages_list(self):
        """Empty pages list leaves YEDDA unchanged."""
        yedda = _make_yedda(("Text.", "Description"))
        marked, n = add_page_markers(yedda, [])
        assert n == 0
        assert marked == yedda

    def test_monotonicity_enforced(self):
        """A page listed out of order cannot match a block before the
        previous match."""
        yedda = _make_yedda(
            ("Mycologia Vol. 102 2010\nbody text", "Misc-exposition"),
            ("Body text here.", "Description"),
            ("Persoonia European Flora\nbody text", "Misc-exposition"),
        )
        # Page 3 listed first → matches block 2; page 2 block is before
        # min_block so it is skipped.
        pages = [
            _page(3, "3", "Persoonia European Flora",
                  ctx_after="Persoonia European Flora body text"),
            _page(2, "2", "Mycologia Vol. 102 2010",
                  ctx_after="Mycologia Vol. 102 2010 body text"),
        ]
        marked, n = add_page_markers(yedda, pages)
        assert "--- PDF Page 3 Label 3 ---" in marked
        assert "--- PDF Page 2 Label 2 ---" not in marked
        assert n == 1


# ---------------------------------------------------------------------------
# add_page_markers — voting behaviour
# ---------------------------------------------------------------------------

class TestAddPageMarkersVoting:
    """Tests for the 3-vote scoring system."""

    def test_line_count_breaks_tie(self):
        """When two blocks both score after_vote, the one whose cumulative
        YEDDA line gap matches lines_before wins (2 votes vs 1)."""
        # Block 0: same header text as block 2, but too close to the start.
        # Block 1: 98 lines of body, pushing block 2 to the right distance.
        # Block 2: correct position — after_vote + line_count_vote = 2 votes.
        early_body = "\n".join(f"Line {i}" for i in range(98))
        yedda = _make_yedda(
            ("Running Head 279", "Misc-exposition"),  # block 0: too early
            (early_body, "Description"),              # block 1: 98 lines
            ("Running Head 279", "Misc-exposition"),  # block 2: correct
        )
        pages = [_page(5, "5", "Running Head 279",
                       lines_before=100,
                       ctx_after="Running Head 279")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        # Verify the marker landed on block 2, not block 0.  Block 0 should
        # be unchanged (no "--- PDF Page" prefix).
        block_texts = [m.group(1) for m in _YEDDA_BLOCK_RE.finditer(marked)]
        page_header_blocks = [
            t for t in block_texts if t.startswith("--- PDF Page")
        ]
        assert len(page_header_blocks) == 1
        # The preceding body block must appear between the two Running Head
        # blocks, confirming it's the second one that was marked.
        first_head_idx = next(
            i for i, t in enumerate(block_texts)
            if "Running Head 279" in t and not t.startswith("--- PDF Page")
        )
        marked_head_idx = next(
            i for i, t in enumerate(block_texts)
            if t.startswith("--- PDF Page")
        )
        assert first_head_idx < marked_head_idx

    def test_before_vote_fires(self):
        """ctx_before found in character window before block → before_vote."""
        yedda = _make_yedda(
            ("End of previous page content.", "Description"),
            ("Running Head 5", "Misc-exposition"),
        )
        pages = [_page(5, "5", "Running Head 5",
                       ctx_before="End of previous page content.")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 5 Label 5 ---" in marked

    def test_after_vote_anchored_fires(self):
        """ctx_after (starting with the header) matches the block start."""
        yedda = _make_yedda(
            ("Running Head 5\nFirst body line.", "Misc-exposition"),
        )
        # ctx_after starts with the header (parse_page_markers behaviour).
        pages = [_page(5, "5", "Running Head 5",
                       ctx_after="Running Head 5 First body line.")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 5 Label 5 ---" in marked

    def test_after_vote_anchored_no_false_positive(self):
        """Block BEFORE the correct one does not score after_vote because its
        after_window does not begin with ctx_after."""
        yedda = _make_yedda(
            ("End of previous page text.", "Description"),    # block 0
            ("Start of new page here.", "Misc-exposition"),   # block 1
        )
        pages = [_page(5, "5", "Start of new page here.",
                       ctx_after="Start of new page here.")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        # Marker must be on block 1, not block 0.
        block_texts = [m.group(1) for m in _YEDDA_BLOCK_RE.finditer(marked)]
        assert block_texts[0] == "End of previous page text."
        assert block_texts[1].startswith("--- PDF Page 5")

    def test_fallback_when_one_vote(self):
        """A single context vote is enough to place the marker via fallback."""
        yedda = _make_yedda(("Header X", "Misc-exposition"))
        # lines_before=999 → line-count vote fails; only after_vote fires.
        pages = [_page(7, "7", "Header X",
                       lines_before=999,
                       ctx_after="Header X")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 7 Label 7 ---" in marked

    def test_no_match_without_context_vote(self):
        """No context vote on any candidate → no marker placed."""
        yedda = _make_yedda(("Completely unrelated text.", "Description"))
        pages = [_page(7, "7", "Header X",
                       ctx_after="Header X some more text")]
        _, n = add_page_markers(yedda, pages)
        assert n == 0

    def test_orphan_block_spanned_by_window(self):
        """A short orphan block between context text and the target block does
        not prevent before_vote from firing (character window spans it)."""
        yedda = _make_yedda(
            ("Long page body ending here.", "Notes"),     # block 0
            ("8", "Misc-exposition"),                     # block 1: orphan
            ("New page content starts here.", "Misc-exposition"),  # block 2
        )
        pages = [_page(
            9, "9", "New page content starts here.",
            ctx_before="Long page body ending here.",
            ctx_after="New page content starts here.",
        )]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        # Marker is placed in the correct region (block 1 or 2), not on
        # block 0 (the long body content before the orphan).  The character
        # window lets before_vote for the target block span the orphan.
        block_texts = [m.group(1) for m in _YEDDA_BLOCK_RE.finditer(marked)]
        assert not block_texts[0].startswith("--- PDF Page 9")
        assert any(t.startswith("--- PDF Page 9") for t in block_texts[1:])

    def test_page_13_14_alternaria_scenario(self):
        """Integration test: page 13 goes before 'I have not seen...' and
        page 14 goes before 'many Alternaria species.' even though page 13
        has a bare-digit running head '7' and page 14 splits mid-sentence.

        Corresponds to doc 1303880f (Mycotaxon LV, 1995).  The old block-
        based approach misplaced page 14 because the stripped '7' and '8'
        orphan blocks disrupted context lookups; the character-window
        approach spans them correctly.
        """
        # Simplified YEDDA (orphan blocks already stripped by
        # strip_page_markers since their text is < 4 chars).
        yedda = _make_yedda(
            (
                "1827 Puccinia cheiri T. G. Lestiboudois, "
                "Botanogr. Belg., p. 132.",
                "Misc-exposition",
            ),
            (
                "I have not seen authentic Lestiboudois material. "
                "If the fungus is the same as "
                "Helminthosporium cheiranthi Lib.",
                "Notes",
            ),
            (
                "1831 Septosporium atrum Corda, in Sturm's Deutschl. Flora "
                "LXII. Abt., Bd. 3, p. 33-34. Fig. 17.",
                "Nomenclature",
            ),
            (
                "Type material of this taxon has not been found. "
                "The fungus was described as being phaeodictyosporic "
                "and occurring on dead herbaceous stems, as have so",
                "Notes",
            ),
            (
                "many Alternaria species. A variety was published later, S. "
                "atrum var. foliicolum Corda.",
                "Misc-exposition",
            ),
        )

        # ctx_after starts with the header text (parse_page_markers behaviour).
        # Page 13 header is bare "7"; ctx_after includes the first body line.
        # Page 14 header is mid-sentence "many Alternaria species.".
        pages = [
            _page(
                13, "13", "7",
                lines_before=10,
                ctx_before=(
                    "1827 Puccinia cheiri T. G. Lestiboudois, "
                    "Botanogr. Belg., p. 132."
                ),
                ctx_after=(
                    "7 I have not seen authentic Lestiboudois material."
                ),
            ),
            _page(
                14, "14", "many Alternaria species.",
                lines_before=12,
                ctx_before=(
                    "The fungus was described as being phaeodictyosporic "
                    "and occurring on dead herbaceous stems, as have so"
                ),
                ctx_after=(
                    "many Alternaria species. A variety was published later"
                ),
            ),
        ]
        marked, n = add_page_markers(yedda, pages)
        assert n == 2, f"Expected 2 markers placed, got {n}"

        block_texts = [m.group(1) for m in _YEDDA_BLOCK_RE.finditer(marked)]
        page_blocks = {
            t.split("\n")[0]: t
            for t in block_texts
            if t.startswith("--- PDF Page")
        }
        assert "--- PDF Page 13 Label 13 ---" in page_blocks, (
            "Page 13 marker not placed"
        )
        assert "--- PDF Page 14 Label 14 ---" in page_blocks, (
            "Page 14 marker not placed"
        )

        # Page 13 should be on the "I have not seen..." block.
        assert (
            "Lestiboudois material"
            in page_blocks["--- PDF Page 13 Label 13 ---"]
        )
        # Page 14 should be on the "many Alternaria species." block.
        assert (
            "many Alternaria species"
            in page_blocks["--- PDF Page 14 Label 14 ---"]
        )


# ---------------------------------------------------------------------------
# add_page_markers — fuzzy (OCR-error) matching
# ---------------------------------------------------------------------------

_OCR_PAIRS = [
    # clean text          OCR-noisy equivalent     substitution notes
    ("Mycologia 102(5) 2010",     "Mycolog1a 102(5) 2Ol0"),    # O->0, l->1
    ("TAXON 58 (4) August 2009",  "TAXON 58 (4) August 2OO9"),  # O->0 x2
    ("278 ... Halici & Turk",     "278 ... Ha1ici & Turk"),    # l->1
    ("Persoonia - Vol. 25, 2010", "Persoonia - Vol. 25, 2Ol0"),  # O->0
    ("Ann. bot. fenn. 47: 1-10",  "Ann. bot. fenn. 47: 1-10"),  # dash norm
]


class TestAddPageMarkersFuzzy:
    """Fuzzy matching: OCR-noisy context windows and block text are matched."""

    @pytest.mark.parametrize("clean,ocr", _OCR_PAIRS)
    def test_ocr_block_matched_via_before_vote(self, clean: str, ocr: str):
        """When a block has OCR noise (after_vote misses due to strict ratio),
        a clean preceding block still fires before_vote and places the marker
        via the fallback path."""
        yedda = _make_yedda(
            ("Previous page content.", "Description"),
            (ocr, "Misc-exposition"),
        )
        # ctx_after = clean header; ctx_before = text from preceding block.
        pages = [_page(3, "3", clean,
                       ctx_before="Previous page content.",
                       ctx_after=clean)]
        _, n = add_page_markers(yedda, pages)
        assert n == 1

    def test_ocr_header_in_longer_block(self):
        """OCR header at the start of a longer block: minor noise (1 sub) in a
        longer ctx_after string keeps ratio above _AFTER_THRESHOLD."""
        # One substitution: '0' → 'O' at the end ("201O" vs "2010")
        yedda = _make_yedda((
            "Mycologia 102(5) 201O\n"            # OCR: capital-O for zero
            "1105-1118 BIOECOLOGY OF TWO TAXA OF AGARICUS",
            "Misc-exposition",
        ))
        # ctx_after starts with clean header + body text (≥ 30 chars total
        # → 1 substitution in 35 chars keeps ratio well above 90 %).
        pages = [_page(5, "5", "Mycologia 102(5) 2010",
                       ctx_after=(
                           "Mycologia 102(5) 2010 "
                           "1105-1118 BIOECOLOGY OF TWO TAXA OF AGARICUS"
                       ))]
        _, n = add_page_markers(yedda, pages)
        assert n == 1

    def test_fuzzy_before_window_minor_noise(self):
        """OCR noise in before_window, clean ctx_before: partial_ratio >= 75.
        """
        yedda = _make_yedda(
            ("End 0f previous page c0ntent.", "Description"),  # 2 OCR subs
            ("Running Head 5", "Misc-exposition"),
        )
        pages = [_page(5, "5", "Running Head 5",
                       ctx_before="End of previous page content.")]
        _, n = add_page_markers(yedda, pages)
        assert n == 1

    def test_fuzzy_after_window_minor_noise(self):
        """Minor OCR noise in the after_window (1 sub in 45+ chars) keeps
        fuzz.ratio above _AFTER_THRESHOLD."""
        # "Start 0f next..." has one OCR substitution: 'o'→'0'.
        # ctx_after (from clean article.txt) starts with the header, so the
        # shared 45-char prefix has only 1 edit → ratio ≈ 95 %.
        yedda = _make_yedda(
            ("Running Head 5", "Misc-exposition"),
            ("Start 0f next page body text here.", "Description"),
        )
        pages = [_page(
            5, "5", "Running Head 5",
            ctx_after="Running Head 5 Start of next page body text here.",
        )]
        _, n = add_page_markers(yedda, pages)
        assert n == 1

    def test_completely_different_text_not_matched(self):
        """A block that shares almost no characters is not matched."""
        yedda = _make_yedda(("XXXXXX XXXXXX XXXXXX", "Misc-exposition"))
        pages = [_page(3, "3", "Mycologia 102(5) 2010",
                       ctx_after="Mycologia 102(5) 2010")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 0


# ---------------------------------------------------------------------------
# strip_page_markers
# ---------------------------------------------------------------------------

class TestStripPageMarkers:
    def test_strips_page_header_block(self):
        yedda = _make_yedda(
            ("--- PDF Page 3 Label 3 ---\n279 Running Head", "Page-header"),
        )
        stripped, n = strip_page_markers(yedda)
        assert n == 1
        assert "--- PDF Page" not in stripped
        assert "279 Running Head" in stripped
        assert "#Misc-exposition*]" in stripped

    def test_non_page_header_blocks_untouched(self):
        yedda = _make_yedda(
            ("Nomenclature text.", "Nomenclature"),
            ("Description text.", "Description"),
        )
        stripped, n = strip_page_markers(yedda)
        assert n == 0
        assert stripped == yedda

    def test_marker_only_block_discarded(self):
        """A Page-header with no content after the marker line is dropped."""
        yedda = _make_yedda(
            ("--- PDF Page 3 Label 3 ---", "Page-header"),
            ("Body text.", "Description"),
        )
        stripped, n = strip_page_markers(yedda)
        assert n == 1
        assert "--- PDF Page" not in stripped
        assert "Body text." in stripped

    def test_multiple_markers_stripped(self):
        yedda = _make_yedda(
            ("--- PDF Page 1 Label 1 ---\nFirst header", "Page-header"),
            ("Body.", "Description"),
            ("--- PDF Page 2 Label 2 ---\nSecond header", "Page-header"),
        )
        stripped, n = strip_page_markers(yedda)
        assert n == 2
        assert "--- PDF Page" not in stripped
        assert "First header" in stripped
        assert "Second header" in stripped
        assert "Body." in stripped

    def test_idempotent_when_no_markers(self):
        yedda = _make_yedda(
            ("Taxon name.", "Nomenclature"),
            ("Some description.", "Description"),
        )
        stripped, n = strip_page_markers(yedda)
        assert n == 0
        assert stripped == yedda


# ---------------------------------------------------------------------------
# fix_yedda (combined)
# ---------------------------------------------------------------------------

class TestFixYedda:
    def test_both_fixes_applied(self):
        """fix_yedda applies malformed-block fix then adds page markers."""
        yedda = (
            "[@Material examined: TURKEY. (ANES 1).\n\n"
            "[@Stigmidium sp.#Misc-exposition*]\n\n"
            "[@278 ... Author#Misc-exposition*]\n"
        )
        pages = [_page(2, "2", "278 ... Author",
                       ctx_after="278 ... Author")]
        fixed, stats = fix_yedda(yedda, pages)

        assert stats["n_malformed"] == 1
        assert stats["n_page_markers"] == 1
        assert (
            "[@Material examined: TURKEY. (ANES 1).#Materials-examined*]"
            in fixed
        )
        assert "[@Stigmidium sp.#Misc-exposition*]" in fixed
        assert "--- PDF Page 2 Label 2 ---" in fixed

    def test_stats_includes_n_stripped(self):
        _, stats = fix_yedda("[@Text.#Description*]\n", pages=None)
        assert "n_stripped" in stats

    def test_reentrant_strips_then_replaces(self):
        """Running fix_yedda twice re-places the marker at the same block."""
        yedda = "[@Running Head\nBody text.#Misc-exposition*]\n"
        # ctx_after starts with header; body text follows on same block.
        pages = [_page(3, "3", "Running Head",
                       ctx_after="Running Head Body text.")]

        first, stats1 = fix_yedda(yedda, pages)
        assert stats1["n_page_markers"] == 1
        assert "--- PDF Page 3 Label 3 ---" in first

        second, stats2 = fix_yedda(first, pages)
        assert stats2["n_stripped"] == 1
        assert stats2["n_page_markers"] == 1
        assert second.count("--- PDF Page 3 Label 3 ---") == 1

    def test_no_pages_skips_marker_step(self):
        fixed, stats = fix_yedda("[@Text.#Description*]\n", pages=None)
        assert stats["n_page_markers"] == 0
        assert "--- PDF Page" not in fixed

    def test_clean_doc_returns_unchanged(self):
        yedda = (
            "[@Taxon name.#Nomenclature*]\n\n"
            "[@Description text.#Description*]\n"
        )
        _, stats = fix_yedda(yedda, pages=None)
        assert stats["n_malformed"] == 0
        assert stats["n_page_markers"] == 0

    def test_three_level_nesting_fully_unwound(self):
        """Three levels of nesting require two passes; fix_yedda iterates."""
        yedda = (
            "[@Outer text.\n\n"
            "[@Middle text.\n\n"
            "[@Inner text.#Notes*]"
            "#Misc-exposition*]\n"
        )
        fixed, stats = fix_yedda(yedda, pages=None)
        assert stats["n_malformed"] >= 2
        texts_with_nested = [
            m.group(1)
            for m in _YEDDA_BLOCK_RE.finditer(fixed)
            if "[@" in m.group(1)
        ]
        assert not texts_with_nested
        assert "Outer text." in fixed
        assert "Middle text." in fixed
        assert "Inner text." in fixed
