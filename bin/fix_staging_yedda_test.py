#!/usr/bin/env python3
"""Tests for fix_staging_yedda.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_staging_yedda import (
    _YEDDA_BLOCK_RE,
    _infer_outer_tag,
    add_page_markers,
    fix_malformed_blocks,
    fix_yedda,
    parse_page_markers,
)


# ---------------------------------------------------------------------------
# _infer_outer_tag
# ---------------------------------------------------------------------------

class TestInferOuterTag:
    def test_materials_examined(self):
        assert _infer_outer_tag("Material examined: TURKEY. ...", "Misc-exposition") == "Materials-examined"

    def test_materials_examined_plural(self):
        assert _infer_outer_tag("Materials examined: TURKEY. ...", "Misc-exposition") == "Materials-examined"

    def test_key_to(self):
        assert _infer_outer_tag("Key to the species\n1. ...", "Page-header") == "Key"

    def test_notes(self):
        assert _infer_outer_tag("Notes on taxonomy.", "Misc-exposition") == "Notes"

    def test_distribution(self):
        assert _infer_outer_tag("Distribution: Widespread.", "Description") == "Distribution"

    def test_fallback(self):
        assert _infer_outer_tag("Some unknown text.", "Misc-exposition") == "Misc-exposition"

    def test_empty_text_returns_fallback(self):
        assert _infer_outer_tag("", "Description") == "Description"


# ---------------------------------------------------------------------------
# fix_malformed_blocks
# ---------------------------------------------------------------------------

class TestFixMalformedBlocks:
    def test_clean_yedda_unchanged(self):
        yedda = "[@Arthonia fuscopurpurea (Tul.) R. Sant.#Nomenclature*]\n\n[@A description.#Description*]\n"
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
        # Outer block text truncated at [@
        assert "[@Material examined: TURKEY. (ANES 14198).#Materials-examined*]" in fixed
        # Recovered inner block
        assert "[@Stigmidium leucophlebiae Cl. Roux & Triebel#Misc-exposition*]" in fixed
        # Following valid block preserved
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

    def test_page_numbers_correct(self):
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        nums = [p[0] for p in pages]
        assert nums == [1, 2, 3]

    def test_page_labels_correct(self):
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        labels = [p[1] for p in pages]
        assert labels == ["1", "2", "3"]

    def test_header_texts_correct(self):
        pages = parse_page_markers(SAMPLE_ARTICLE_TXT)
        headers = [p[2] for p in pages]
        assert headers[0] == "ISSN (print) 0093-4666"
        assert headers[1] == "278 ... Halici, Candan & Türk"
        assert headers[2] == "Key to peltigericolous fungi in Turkey ... 279"

    def test_empty_string(self):
        assert parse_page_markers("") == []

    def test_no_markers(self):
        assert parse_page_markers("plain text\nno markers\n") == []

    def test_roman_label(self):
        txt = "--- PDF Page 5 Label xv ---\nRunning head\n"
        pages = parse_page_markers(txt)
        assert len(pages) == 1
        assert pages[0] == (5, "xv", "Running head")


# ---------------------------------------------------------------------------
# add_page_markers
# ---------------------------------------------------------------------------

class TestAddPageMarkers:
    def _make_yedda(self, *blocks: tuple) -> str:
        """Helper: build YEDDA string from (text, tag) tuples."""
        return "\n\n".join(f"[@{t}#{g}*]" for t, g in blocks) + "\n"

    def test_adds_marker_to_matching_block(self):
        yedda = self._make_yedda(
            ("ISSN (print) 0093-4666", "Misc-exposition"),
            ("278 ... Halici, Candan & Türk", "Misc-exposition"),
            ("Article body.", "Description"),
        )
        pages = [(2, "2", "278 ... Halici, Candan & Türk")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 2 Label 2 ---\n278 ... Halici, Candan & Türk" in marked
        assert "#Page-header*]" in marked

    def test_relabels_to_page_header(self):
        yedda = self._make_yedda(("278 ... Author", "Misc-exposition"))
        pages = [(2, "2", "278 ... Author")]
        marked, _ = add_page_markers(yedda, pages)
        assert "[@--- PDF Page 2 Label 2 ---\n278 ... Author#Page-header*]" in marked

    def test_already_marked_block_not_doubled(self):
        """Blocks already carrying a marker are not marked again."""
        yedda = self._make_yedda(
            ("--- PDF Page 2 Label 2 ---\n278 ... Author", "Page-header"),
        )
        pages = [(2, "2", "278 ... Author")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 0
        assert marked.count("--- PDF Page 2 Label 2 ---") == 1

    def test_no_match_leaves_yedda_unchanged(self):
        yedda = self._make_yedda(("Completely different text.", "Misc-exposition"))
        pages = [(2, "2", "278 ... Author")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 0
        assert "--- PDF Page" not in marked

    def test_partial_match_by_starts_with(self):
        """Block text that starts with the header text is matched."""
        yedda = self._make_yedda(
            ("278 ... Author\nsome extra text on same block", "Misc-exposition"),
        )
        pages = [(2, "2", "278 ... Author")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 1
        assert "--- PDF Page 2 Label 2 ---" in marked

    def test_multiple_pages_multiple_matches(self):
        yedda = self._make_yedda(
            ("Header page 2", "Misc-exposition"),
            ("Body.", "Description"),
            ("Header page 3", "Misc-exposition"),
        )
        pages = [(2, "2", "Header page 2"), (3, "3", "Header page 3")]
        marked, n = add_page_markers(yedda, pages)
        assert n == 2
        assert "--- PDF Page 2 Label 2 ---" in marked
        assert "--- PDF Page 3 Label 3 ---" in marked

    def test_empty_pages_list(self):
        yedda = self._make_yedda(("Text.", "Description"))
        marked, n = add_page_markers(yedda, [])
        assert n == 0
        assert marked == yedda


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
        pages = [(2, "2", "278 ... Author")]
        fixed, stats = fix_yedda(yedda, pages)

        assert stats["n_malformed"] == 1
        assert stats["n_page_markers"] == 1
        assert "[@Material examined: TURKEY. (ANES 1).#Materials-examined*]" in fixed
        assert "[@Stigmidium sp.#Misc-exposition*]" in fixed
        assert "--- PDF Page 2 Label 2 ---" in fixed

    def test_no_pages_skips_marker_step(self):
        yedda = "[@Text.#Description*]\n"
        fixed, stats = fix_yedda(yedda, pages=None)
        assert stats["n_page_markers"] == 0
        assert "--- PDF Page" not in fixed

    def test_clean_doc_returns_unchanged(self):
        """A clean doc with no pages returns stats of all zeros."""
        yedda = (
            "[@Taxon name.#Nomenclature*]\n\n"
            "[@Description text.#Description*]\n"
        )
        fixed, stats = fix_yedda(yedda, pages=None)
        assert stats["n_malformed"] == 0
        assert stats["n_page_markers"] == 0

    def test_three_level_nesting_fully_unwound(self):
        """Three levels of nesting require two passes; fix_yedda iterates."""
        # Level 1 block swallows levels 2 and 3.
        # After pass 1: level-1 outer text + level-2 block (which has [@) + level-3 block.
        # After pass 2: all three blocks are clean.
        yedda = (
            "[@Outer text.\n\n"
            "[@Middle text.\n\n"
            "[@Inner text.#Notes*]"
            "#Misc-exposition*]\n"
        )
        fixed, stats = fix_yedda(yedda, pages=None)
        assert stats["n_malformed"] >= 2
        # No block text should still contain [@
        texts_with_nested = [
            m.group(1)
            for m in _YEDDA_BLOCK_RE.finditer(fixed)
            if "[@" in m.group(1)
        ]
        assert not texts_with_nested
        assert "Outer text." in fixed
        assert "Middle text." in fixed
        assert "Inner text." in fixed
