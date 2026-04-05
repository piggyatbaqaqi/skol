#!/usr/bin/env python3
"""Tests for fix_staging_yedda.py."""

import sys
from pathlib import Path
from typing import Tuple

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
            _infer_outer_tag("Material examined: TURKEY. ...", "Misc-exposition")
            == "Materials-examined"
        )

    def test_materials_examined_plural(self):
        assert (
            _infer_outer_tag("Materials examined: TURKEY. ...", "Misc-exposition")
            == "Materials-examined"
        )

    def test_key_to(self):
        assert (
            _infer_outer_tag("Key to the species\n1. ...", "Page-header") == "Key"
        )

    def test_notes(self):
        assert (
            _infer_outer_tag("Notes on taxonomy.", "Misc-exposition") == "Notes"
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
        assert "[@Arthonia fuscopurpurea (Tul.) R. Sant.#Nomenclature*]" in fixed
        assert "[@A description.#Description*]" in fixed

    def test_single_malformed_block_splits_into_two(self):
        """A block missing *] before a nested [@ splits into two blocks."""
        yedda = (
            "[@Material examined: TURKEY. (ANES 14198).\n\n"
            "[@Stigmidium leucophlebiae Cl. Roux & Triebel#Misc-exposition*]\n\n"
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
    def test_adds_marker_to_matching_block(self):
        yedda = _make_yedda(
            ("ISSN (print) 0093-4666", "Misc-exposition"),
            ("278 ... Halici, Candan & Türk", "Misc-exposition"),
            ("Article body.", "Description"),
        )
        pages = [_page(2, "2", "278 ... Halici, Candan & Türk")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert (
            "--- PDF Page 2 Label 2 ---\n278 ... Halici, Candan & Türk" in marked
        )
        assert "#Page-header*]" in marked

    def test_relabels_to_page_header(self):
        yedda = _make_yedda(("278 ... Author", "Misc-exposition"))
        pages = [_page(2, "2", "278 ... Author")]
        marked, _ = add_page_markers(yedda, pages)
        assert (
            "[@--- PDF Page 2 Label 2 ---\n278 ... Author#Page-header*]" in marked
        )

    def test_already_marked_block_not_doubled(self):
        """Blocks already carrying a marker are not marked again."""
        yedda = _make_yedda(
            ("--- PDF Page 2 Label 2 ---\n278 ... Author", "Page-header"),
        )
        pages = [_page(2, "2", "278 ... Author")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 0
        assert marked.count("--- PDF Page 2 Label 2 ---") == 1

    def test_no_match_leaves_yedda_unchanged(self):
        yedda = _make_yedda(("Completely different text.", "Misc-exposition"))
        pages = [_page(2, "2", "278 ... Author")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 0
        assert "--- PDF Page" not in marked

    def test_partial_match_by_starts_with(self):
        """Block text that starts with the header text is matched."""
        yedda = _make_yedda(
            ("278 ... Author\nsome extra text on same block", "Misc-exposition"),
        )
        pages = [_page(2, "2", "278 ... Author")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 2 Label 2 ---" in marked

    def test_multiple_pages_multiple_matches(self):
        yedda = _make_yedda(
            ("Header page 2", "Misc-exposition"),
            ("Body.", "Description"),
            ("Header page 3", "Misc-exposition"),
        )
        pages = [
            _page(2, "2", "Header page 2"),
            _page(3, "3", "Header page 3"),
        ]
        marked, n = add_page_markers(yedda, pages)
        assert n == 2
        assert "--- PDF Page 2 Label 2 ---" in marked
        assert "--- PDF Page 3 Label 3 ---" in marked

    def test_empty_pages_list(self):
        yedda = _make_yedda(("Text.", "Description"))
        marked, n = add_page_markers(yedda, [])
        assert n == 0
        assert marked == yedda

    def test_monotonicity_enforced(self):
        """A page listed out of order cannot match a block before the
        previous match."""
        yedda = _make_yedda(
            ("Header page 2", "Misc-exposition"),
            ("Body.", "Description"),
            ("Header page 3", "Misc-exposition"),
        )
        # Page 3 listed first → matches block 2; page 2 block is before
        # min_block so it is skipped.
        pages = [
            _page(3, "3", "Header page 3"),
            _page(2, "2", "Header page 2"),
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

    def test_line_count_vote_breaks_tie(self):
        """Two blocks share the same header text.  The one whose cumulative
        YEDDA line gap matches lines_before wins."""
        # Block 0: 100 non-empty lines; block 1: correct header at gap≈100;
        # block 2: same header text but only 1 further line.
        early_body = "\n".join(f"Line {i}" for i in range(100))
        yedda = _make_yedda(
            (early_body, "Description"),               # 100 lines
            ("Running Head 279", "Misc-exposition"),   # correct position
            ("Running Head 279", "Misc-exposition"),   # false positive
        )
        pages = [_page(5, "5", "Running Head 279", lines_before=100)]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        marked_texts = [
            m.group(1)
            for m in _YEDDA_BLOCK_RE.finditer(marked)
            if m.group(1).startswith("--- PDF Page")
        ]
        assert len(marked_texts) == 1
        assert "Running Head 279" in marked_texts[0]

    def test_context_before_vote(self):
        """context_before appearing in the preceding block contributes a vote."""
        yedda = _make_yedda(
            ("End of previous page content.", "Description"),
            ("Running Head 5", "Misc-exposition"),
        )
        pages = [_page(
            5, "5", "Running Head 5",
            ctx_before="End of previous page content.",
        )]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 5 Label 5 ---" in marked

    def test_context_after_vote_in_next_block(self):
        """context_after appearing in the following block contributes a vote."""
        yedda = _make_yedda(
            ("Running Head 5", "Misc-exposition"),
            ("Start of next page body text.", "Description"),
        )
        pages = [_page(
            5, "5", "Running Head 5",
            ctx_after="Start of next page body text.",
        )]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 5 Label 5 ---" in marked

    def test_context_after_vote_in_block_tail(self):
        """context_after in the same block (after the header line) counts."""
        yedda = _make_yedda(
            ("Running Head 5\nFirst line of page body.", "Misc-exposition"),
        )
        pages = [_page(
            5, "5", "Running Head 5",
            ctx_after="First line of page body.",
        )]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1

    def test_fallback_when_no_two_vote_match(self):
        """When no candidate scores >= _MIN_VOTES the first header match wins."""
        yedda = _make_yedda(("Header X", "Misc-exposition"))
        # lines_before=999 → line-count vote fails; no context either
        pages = [_page(7, "7", "Header X", lines_before=999)]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 7 Label 7 ---" in marked

    def test_no_match_without_header_vote(self):
        """Without a header vote there is no fallback — no marker added."""
        yedda = _make_yedda(("Completely unrelated text.", "Description"))
        pages = [_page(7, "7", "Header X")]
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
        pages = [_page(2, "2", "278 ... Author")]
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
        pages = [_page(3, "3", "Running Head")]

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
