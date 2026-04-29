"""Tests for fix_missing_yedda core helpers."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fix_missing_yedda import (  # type: ignore[import]
    find_block_in_text,
    parse_yedda,
    blocks_to_yedda,
    split_gap,
)

# ---------------------------------------------------------------------------
# parse_yedda
# ---------------------------------------------------------------------------

class TestParseYedda:
    def test_single_block(self) -> None:
        result = parse_yedda("[@hello world#Description*]")
        assert result == [("hello world", "Description")]

    def test_multiple_blocks(self) -> None:
        result = parse_yedda(
            "[@first#Nomenclature*]\n\n[@second#Description*]"
        )
        assert result == [("first", "Nomenclature"), ("second", "Description")]

    def test_empty_string(self) -> None:
        assert parse_yedda("") == []


# ---------------------------------------------------------------------------
# find_block_in_text — strategies 1–3 regression
# ---------------------------------------------------------------------------

class TestFindBlockInTextBasic:
    def test_exact_match(self) -> None:
        hay = "some text here and more"
        start, end, eff = find_block_in_text("text here", hay, 0)
        assert start == 5
        assert end == 14
        assert eff == "text here"

    def test_search_from_respected(self) -> None:
        hay = "abc abc"
        start, end, eff = find_block_in_text("abc", hay, 1)
        assert start == 4

    def test_not_found_returns_minus_one(self) -> None:
        start, end, eff = find_block_in_text("xyz", "hello world", 0)
        assert start == -1
        assert end == -1
        assert eff == ""

    def test_page_marker_stripped(self) -> None:
        block = "--- PDF Page 1 Label 1 ---\nsome content here"
        hay = "some content here"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start == 0
        assert eff == "some content here"

    def test_whitespace_normalised(self) -> None:
        block = "hello   world"
        hay = "hello world"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start == 0


# ---------------------------------------------------------------------------
# find_block_in_text — strategy 4: NFKC normalisation
# ---------------------------------------------------------------------------

class TestFindBlockInTextNFKC:
    def test_fi_ligature_in_block_matches_fi_in_haystack(self) -> None:
        """ﬁ (U+FB01) in the annotated block matches 'fi' in the new OCR text."""
        block = "the ﬁrst time"
        hay = "start text the first time end"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start >= 0, "NFKC strategy should find ligature → two-char match"
        assert eff == "the first time"

    def test_fi_in_block_matches_ligature_in_haystack(self) -> None:
        """fi in the block also matches ﬁ in the haystack (reverse direction)."""
        block = "the first time"
        hay = "start text the ﬁrst time end"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start >= 0
        assert eff == "the ﬁrst time"

    def test_fl_ligature(self) -> None:
        block = "reﬂected in the data"
        hay = "reﬂected in the data and more"
        start, end, eff = find_block_in_text(block, hay, 0)
        # Exact match succeeds first — just confirm it works
        assert start == 0

    def test_fl_ligature_in_block_vs_fl_in_haystack(self) -> None:
        block = "reﬂected light"
        hay = "reflected light"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start >= 0
        assert eff == "reflected light"

    def test_nfkc_offsets_are_original_positions(self) -> None:
        """start/end index into the original haystack, not a normalised copy."""
        block = "ﬁrst"
        hay = "aaa first bbb"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start == 4
        assert end == 9
        assert hay[start:end] == "first"

    def test_nfkc_respects_search_from(self) -> None:
        hay = "first first"  # two occurrences
        block = "ﬁrst"
        start1, _, _ = find_block_in_text(block, hay, 0)
        start2, _, _ = find_block_in_text(block, hay, start1 + 1)
        assert start1 == 0
        assert start2 == 6


# ---------------------------------------------------------------------------
# find_block_in_text — strategy 5: difflib fuzzy matching
# ---------------------------------------------------------------------------

class TestFindBlockInTextDifflib:
    def test_single_char_substitution(self) -> None:
        """One OCR substitution (not a ligature) is found via difflib."""
        block = "the fungus Endocalyx melanoxanthus var. melanoxanthus"
        # 'x' → 'X' in one position
        hay = "prefix " + "the fungus Endocalyx melanoxanthus var. melanoxantHus" + " suffix"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start >= 0, "difflib should find near-match"

    def test_very_different_text_not_matched(self) -> None:
        """A block that shares few characters is not accepted as a match."""
        block = "Amanita muscaria found in northern Europe forests"
        hay = "Completely different text about something else entirely here now"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start == -1

    def test_difflib_returns_original_positions(self) -> None:
        """Returned start/end index into the original (unnormalised) haystack."""
        block = "the fungus grows in loamy soiI"   # I (capital i) for l
        correct = "the fungus grows in loamy soil"
        hay = "before " + correct + " after"
        start, end, eff = find_block_in_text(block, hay, 0)
        assert start >= 0
        # eff should be drawn from the original haystack
        assert eff == hay[start:end]

    def test_difflib_respects_search_from(self) -> None:
        correct = "the fungus grows in loamy soil"
        hay = correct + " extra " + correct
        block = "the fungus grows in loamy soiI"  # 1-char diff
        start1, end1, _ = find_block_in_text(block, hay, 0)
        start2, end2, _ = find_block_in_text(block, hay, end1 + 1)
        assert start1 < start2


# ---------------------------------------------------------------------------
# split_gap
# ---------------------------------------------------------------------------

class TestSplitGap:
    def test_plain_text_is_to_review(self) -> None:
        blocks = split_gap("some plain text")
        assert blocks == [("some plain text", "To-review")]

    def test_page_marker_becomes_page_header(self) -> None:
        blocks = split_gap("--- PDF Page 3 Label 3 ---")
        assert blocks == [("--- PDF Page 3 Label 3 ---", "Page-header")]

    def test_text_before_marker(self) -> None:
        blocks = split_gap("intro text\n--- PDF Page 2 Label 2 ---")
        assert blocks[0] == ("intro text", "To-review")
        assert blocks[1] == ("--- PDF Page 2 Label 2 ---", "Page-header")

    def test_empty_gap_returns_empty(self) -> None:
        assert split_gap("") == []
        assert split_gap("   \n  \n  ") == []
