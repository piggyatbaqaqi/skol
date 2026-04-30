"""Tests for merge_yedda three-way merge."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_yedda import (  # type: ignore[import]
    build_pos_map,
    find_in_text,
    format_conflict,
    strip_yedda,
    three_way_merge_yedda,
)

# ---------------------------------------------------------------------------
# find_in_text — page-marker stripping (strategy 2)
# ---------------------------------------------------------------------------


class TestFindInTextPageMarker:
    def test_page_marker_prefix_stripped_exact(self) -> None:
        """Synthetic PDF-Page prefix is stripped; remainder found."""
        block = "--- PDF Page 2 Label 2 ---\n110 Vitoria & al."
        hay = "110 Vitoria & al."
        start, end = find_in_text(block, hay)
        assert start == 0
        assert end == len(hay)

    def test_page_marker_prefix_stripped_within_haystack(self) -> None:
        """Remainder found at correct offset inside larger haystack."""
        block = "--- PDF Page 3 Label 3 ---\nRunning Header Text"
        hay = "some preamble\n\nRunning Header Text\n\nmore text"
        start, end = find_in_text(block, hay)
        assert start == 15  # after "some preamble\n\n"
        assert hay[start:end] == "Running Header Text"

    def test_page_marker_prefix_no_remainder_returns_minus_one(self) -> None:
        """Block has no text after marker line → not found."""
        block = "--- PDF Page 1 Label 1 ---"
        hay = "some text here"
        start, end = find_in_text(block, hay)
        assert start == -1
        assert end == -1

    def test_page_marker_prefix_remainder_not_in_haystack(self) -> None:
        """Remainder not in haystack → -1."""
        block = "--- PDF Page 4 Label 4 ---\nMissing Header"
        hay = "completely different content"
        start, end = find_in_text(block, hay)
        assert start == -1
        assert end == -1

    def test_non_marker_block_unaffected(self) -> None:
        """Block without page-marker prefix still works normally."""
        block = "plain text block"
        hay = "before plain text block after"
        start, end = find_in_text(block, hay)
        assert start == 7
        assert hay[start:end] == "plain text block"


# ---------------------------------------------------------------------------
# find_in_text — search_from parameter
# ---------------------------------------------------------------------------


class TestFindInTextSearchFrom:
    def test_search_from_zero_finds_first(self) -> None:
        hay = "abc abc"
        start, end = find_in_text("abc", hay, search_from=0)
        assert start == 0

    def test_search_from_past_first_finds_second(self) -> None:
        hay = "abc abc"
        start, end = find_in_text("abc", hay, search_from=1)
        assert start == 4

    def test_search_from_past_all_returns_minus_one(self) -> None:
        hay = "abc"
        start, end = find_in_text("abc", hay, search_from=1)
        assert start == -1

    def test_search_from_default_is_zero(self) -> None:
        hay = "hello world"
        s1, _ = find_in_text("hello", hay)
        s2, _ = find_in_text("hello", hay, search_from=0)
        assert s1 == s2 == 0

    def test_search_from_with_page_marker_strip(self) -> None:
        """search_from is respected when page-marker stripping fires."""
        hay = "Header\n\nHeader"
        block = "--- PDF Page 1 Label 1 ---\nHeader"
        s1, e1 = find_in_text(block, hay, search_from=0)
        assert s1 == 0
        s2, e2 = find_in_text(block, hay, search_from=1)
        assert s2 == 8


# ---------------------------------------------------------------------------
# three_way_merge_yedda — page-header integration
# ---------------------------------------------------------------------------


class TestThreeWayMergePageHeader:
    def test_synthetic_page_header_placed_not_orphaned(self) -> None:
        """Page-header with synthetic PDF-Page prefix is placed inline."""
        orig_ann = (
            "[@first block#Nomenclature*]\n\n"
            "[@110 Running Header#Page-header*]\n\n"
            "[@second block#Description*]"
        )
        # reviewed_ann has a synthetic prefix on the Page-header block
        reviewed_ann = (
            "[@first block#Nomenclature*]\n\n"
            "[@--- PDF Page 2 Label 2 ---\n"
            "110 Running Header#Page-header*]\n\n"
            "[@second block#Description*]"
        )
        new_text = "first block\n\n110 Running Header\n\nsecond block"
        result = three_way_merge_yedda(orig_ann, reviewed_ann, new_text)
        # Page-header should be placed inline, not as a conflict marker
        assert "Page-header" in result
        assert "<<<<<<< annotation" not in result


# ---------------------------------------------------------------------------
# strip_yedda
# ---------------------------------------------------------------------------

class TestStripYedda:
    def test_single_block(self) -> None:
        assert strip_yedda("[@hello world#Description*]") == "hello world"

    def test_two_blocks_joined_by_double_newline(self) -> None:
        yedda = "[@first#Nomenclature*]\n\n[@second#Description*]"
        assert strip_yedda(yedda) == "first\n\nsecond"

    def test_empty_returns_empty(self) -> None:
        assert strip_yedda("") == ""

    def test_surrounding_whitespace_stripped_from_blocks(self) -> None:
        # The YEDDA regex strips leading/trailing whitespace from block text.
        result = strip_yedda("[@  hello  #Description*]")
        assert result == "hello"


# ---------------------------------------------------------------------------
# build_pos_map
# ---------------------------------------------------------------------------
#
# build_pos_map takes the output of SequenceMatcher.get_matching_blocks() and
# returns a callable pos_map(orig_pos) -> (new_pos, certain: bool).
#   certain=True  means orig_pos falls inside a matching run.
#   certain=False means orig_pos falls in a gap (deleted region); new_pos is
#                 the interpolated boundary of the nearest preceding match.

class TestBuildPosMap:
    def _matching(self, orig: str, new: str):
        import difflib
        sm = difflib.SequenceMatcher(None, orig, new, autojunk=False)
        return sm.get_matching_blocks()

    def test_identical_texts_all_certain(self) -> None:
        text = "hello world"
        mb = self._matching(text, text)
        pos_map = build_pos_map(mb)
        for i in range(len(text)):
            new_pos, certain = pos_map(i)
            assert certain, f"pos {i} should be certain"
            assert new_pos == i

    def test_insertion_in_new_maps_correctly(self) -> None:
        # orig: "hello world"
        # new:  "hello EXTRA world"
        orig = "hello world"
        new = "hello EXTRA world"
        mb = self._matching(orig, new)
        pos_map = build_pos_map(mb)
        # "hello " (0-5) maps to "hello " (0-5) — certain
        new_pos, certain = pos_map(0)
        assert certain
        assert new_pos == 0
        # "world" in orig starts at 6 — maps to "world" in new at 12 — certain
        new_pos, certain = pos_map(6)
        assert certain
        assert new_pos == 12

    def test_deletion_from_orig_gives_uncertain(self) -> None:
        # orig: "hello DELETED world"
        # new:  "hello world"
        orig = "hello DELETED world"
        new = "hello world"
        mb = self._matching(orig, new)
        pos_map = build_pos_map(mb)
        # Position inside "DELETED" in orig should be uncertain.
        new_pos, certain = pos_map(7)   # inside "DELETED"
        assert not certain
        # Interpolated pos should be at boundary of preceding match ("hello ").
        assert new_pos == 6  # end of "hello " in new

    def test_position_at_end_of_orig(self) -> None:
        orig = "hello"
        new = "hello"
        mb = self._matching(orig, new)
        pos_map = build_pos_map(mb)
        new_pos, certain = pos_map(5)  # one past end
        assert new_pos == 5

    def test_substitution_uncertain(self) -> None:
        # orig: "abc XYZ def"
        # new:  "abc def"  (XYZ deleted)
        orig = "abc XYZ def"
        new = "abc def"
        mb = self._matching(orig, new)
        pos_map = build_pos_map(mb)
        new_pos, certain = pos_map(4)  # inside "XYZ"
        assert not certain


# ---------------------------------------------------------------------------
# format_conflict
# ---------------------------------------------------------------------------

class TestFormatConflict:
    def test_contains_markers(self) -> None:
        result = format_conflict("old text", "OldLabel", "new text")
        assert "<<<<<<< annotation" in result
        assert "=======" in result
        assert ">>>>>>> new_ocr" in result

    def test_annotation_side_contains_yedda_block(self) -> None:
        result = format_conflict("old text", "OldLabel", "new text")
        assert "[@old text#OldLabel*]" in result

    def test_new_side_contains_new_text(self) -> None:
        result = format_conflict("old text", "OldLabel", "new text")
        assert "new text" in result

    def test_empty_new_context(self) -> None:
        """Block deleted in new OCR — new side is empty."""
        result = format_conflict("old text", "OldLabel", "")
        assert "<<<<<<< annotation" in result
        assert ">>>>>>> new_ocr" in result


# ---------------------------------------------------------------------------
# three_way_merge_yedda — integration
# ---------------------------------------------------------------------------

class TestThreeWayMergeYedda:
    def test_identical_texts_clean_placement(self) -> None:
        """When orig == new, every block should be placed cleanly."""
        orig_ann = (
            "[@The quick brown fox#Description*]\n\n"
            "[@jumps over the lazy dog#Notes*]"
        )
        new_text = "The quick brown fox\n\njumps over the lazy dog"
        result = three_way_merge_yedda(orig_ann, orig_ann, new_text)
        assert "[@The quick brown fox#Description*]" in result
        assert "[@jumps over the lazy dog#Notes*]" in result
        assert "<<<<<<< annotation" not in result

    def test_minor_ocr_change_clean_placement(self) -> None:
        """Single-char difference resolved by fuzzy matching → no conflict."""
        orig_ann = "[@The quick brown fox#Description*]"
        reviewed_ann = orig_ann  # same label
        # New OCR has minor difference (ligature)
        new_text = "The quick brown fox"
        result = three_way_merge_yedda(orig_ann, reviewed_ann, new_text)
        assert "<<<<<<< annotation" not in result
        assert "Description" in result

    def test_deleted_block_generates_conflict(self) -> None:
        """Block text present in orig but absent from new → conflict marker."""
        orig_ann = (
            "[@present text#Description*]\n\n"
            "[@deleted text#Nomenclature*]"
        )
        reviewed_ann = orig_ann
        # New OCR omits the second paragraph entirely.
        new_text = "present text"
        result = three_way_merge_yedda(orig_ann, reviewed_ann, new_text)
        assert "<<<<<<< annotation" in result
        assert "[@deleted text#Nomenclature*]" in result

    def test_reviewed_label_preserved_over_original(self) -> None:
        """Label from reviewed_ann overrides orig_ann label."""
        orig_ann = "[@some text#Misc-exposition*]"
        reviewed_ann = "[@some text#Nomenclature*]"  # relabeled
        new_text = "some text"
        result = three_way_merge_yedda(orig_ann, reviewed_ann, new_text)
        assert "Nomenclature" in result
        assert "Misc-exposition" not in result

    def test_gap_text_becomes_to_review(self) -> None:
        """New OCR text between placed blocks becomes To-review."""
        orig_ann = (
            "[@first block#Nomenclature*]\n\n"
            "[@second block#Description*]"
        )
        reviewed_ann = orig_ann
        # New OCR has extra text between the blocks.
        new_text = "first block\n\nextra new paragraph\n\nsecond block"
        result = three_way_merge_yedda(orig_ann, reviewed_ann, new_text)
        assert "extra new paragraph" in result
        assert "To-review" in result

    def test_orphan_block_generates_conflict(self) -> None:
        """Block in reviewed_ann not found in orig_ann → conflict marker."""
        orig_ann = "[@known block#Description*]"
        # reviewed_ann has an extra block that was never in orig
        reviewed_ann = (
            "[@known block#Description*]\n\n"
            "[@orphan block text#Nomenclature*]"
        )
        new_text = "known block"
        result = three_way_merge_yedda(orig_ann, reviewed_ann, new_text)
        assert "<<<<<<< annotation" in result
        assert "[@orphan block text#Nomenclature*]" in result

    def test_output_is_valid_where_uncontested(self) -> None:
        """Clean blocks parse as valid YEDDA; conflict markers are outside blocks."""
        import re
        orig_ann = "[@clean block#Description*]"
        reviewed_ann = orig_ann
        new_text = "clean block"
        result = three_way_merge_yedda(orig_ann, reviewed_ann, new_text)
        # Valid YEDDA block present.
        assert re.search(r"\[@.*?#Description\*\]", result, re.DOTALL)
        # No stray conflict markers inside YEDDA blocks.
        block_texts = re.findall(r"\[@(.*?)#\w.*?\*\]", result, re.DOTALL)
        for bt in block_texts:
            assert "<<<<<<<" not in bt
