"""Tests for yedda_to_brat conversion functions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from yedda_to_brat import (  # type: ignore[import]
    _strip_line_leading_ws,
    add_notes,
    yedda_to_brat,
)


# ---------------------------------------------------------------------------
# _strip_line_leading_ws
# ---------------------------------------------------------------------------

class TestStripLineLeadingWs:
    def test_no_leading_ws(self) -> None:
        assert _strip_line_leading_ws("Hello\nworld") == "Hello\nworld"

    def test_strips_leading_spaces(self) -> None:
        assert _strip_line_leading_ws("Hello\n  world") == "Hello\nworld"

    def test_strips_leading_tabs(self) -> None:
        assert _strip_line_leading_ws("Hello\n\tworld") == "Hello\nworld"

    def test_single_line_unchanged(self) -> None:
        assert _strip_line_leading_ws("  hello") == "hello"


# ---------------------------------------------------------------------------
# yedda_to_brat — basic behaviour
# ---------------------------------------------------------------------------

class TestYeddaToBratBasic:
    def test_empty_input(self) -> None:
        plaintext, ann = yedda_to_brat("")
        assert plaintext == ""
        assert ann == ""

    def test_single_block(self) -> None:
        yedda = "[@Hello world#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        assert plaintext == "Hello world"
        assert ann == "T1\tDescription 0 11\tHello world"

    def test_two_blocks_offsets(self) -> None:
        yedda = "[@Taxon name#Nomenclature*]\n\n[@Some description#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        assert plaintext == "Taxon name\n\nSome description"
        lines = ann.splitlines()
        assert lines[0] == "T1\tNomenclature 0 10\tTaxon name"
        # Second block starts at offset 12 (10 chars + "\n\n" separator)
        assert lines[1] == "T2\tDescription 12 28\tSome description"

    def test_actual_newline_in_block_escaped_in_ann(self) -> None:
        """Actual newlines within a block are escaped as \\n in the .ann file."""
        yedda = "[@First line\nSecond line#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        # plaintext retains actual newline
        assert "First line\nSecond line" in plaintext
        # ann uses \\n escape
        assert r"First line\nSecond line" in ann

    def test_empty_block_skipped(self) -> None:
        yedda = "[@  #Description*]\n\n[@content#Notes*]"
        plaintext, ann = yedda_to_brat(yedda)
        assert plaintext == "content"
        assert ann == "T1\tNotes 0 7\tcontent"


# ---------------------------------------------------------------------------
# OCR artefact normalisation — the key fix
# ---------------------------------------------------------------------------

class TestOcrArtifactNormalisation:
    """Verify that literal \\n (backslash + n, OCR artefact) in YEDDA block
    text is normalised to an actual newline before generating brat output,
    so that the .ann \\n escape round-trips correctly against the .txt file.
    """

    def test_ocr_backslash_n_normalised_in_plaintext(self) -> None:
        """Literal \\n (0x5C+0x6E) in block text becomes actual newline in .txt."""
        # Construct YEDDA with a literal backslash+n (OCR artefact) in the text.
        yedda = "[@Hello\\nworld#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        # plaintext must contain an actual newline, not the two-char sequence
        assert "\n" in plaintext
        assert "\\n" not in plaintext

    def test_ocr_backslash_n_escaped_correctly_in_ann(self) -> None:
        """Normalised newline is re-escaped to \\n in .ann so brat parses it."""
        yedda = "[@Hello\\nworld#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        # The ann line should contain the \\n escape (for brat to unescape)
        assert r"\n" in ann

    def test_ocr_artifact_round_trip(self) -> None:
        """Simulate brat's unescape: ann text field unescapes to match plaintext."""
        yedda = "[@Hello\\nworld#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        # brat's unescape: replace \\n with actual newline (single-pass str.replace)
        ann_text_field = ann.split("\t")[2]
        unescaped = ann_text_field.replace("\\n", "\n")
        assert unescaped == plaintext

    def test_backslash_then_ocr_backslash_n_round_trip(self) -> None:
        """Source had \\ + OCR-artefact \\n  (i.e. three bytes: 0x5C 0x5C 0x6E)."""
        # In Python source: '\\\\n' is the 3-char string backslash+backslash+n.
        # This mimics a source YEDDA where the text contained \ followed by an
        # OCR-artefact literal \n.
        yedda = "[@Hello\\\\nworld#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        # After normalisation the source \\n → newline, so we have \ + newline.
        assert "\\\n" in plaintext or "\n" in plaintext  # actual newline present
        # Round-trip: brat unescape of .ann == plaintext
        ann_text_field = ann.split("\t")[2]
        unescaped = ann_text_field.replace("\\n", "\n")
        assert unescaped == plaintext

    def test_offset_computed_after_normalisation(self) -> None:
        """Character offsets in .ann reflect the normalised (post-OCR) text."""
        yedda = "[@A\\nB#Nomenclature*]\n\n[@C#Description*]"
        plaintext, ann = yedda_to_brat(yedda)
        lines = ann.splitlines()
        # First block: "A\nB" = 3 chars (A, newline, B) → end=3
        assert lines[0].startswith("T1\tNomenclature 0 3\t")
        # Second block starts at 3 + 2 (separator) = 5
        assert lines[1].startswith("T2\tDescription 5 6\t")


# ---------------------------------------------------------------------------
# add_notes
# ---------------------------------------------------------------------------

class TestAddNotes:
    def test_no_changes_returns_ann_unchanged(self) -> None:
        ann = "T1\tDescription 0 5\thello"
        assert add_notes(ann, []) == ann

    def test_single_change_appended(self) -> None:
        ann = "T1\tDescription 0 5\thello"
        changes = [{"block_index": 0, "old_tag": "Misc-exposition"}]
        result = add_notes(ann, changes)
        assert "#1\tAnnotatorNotes T1\twas: Misc-exposition" in result

    def test_multiple_changes(self) -> None:
        ann = "T1\tDescription 0 5\thello\nT2\tNotes 7 12\tworld"
        changes = [
            {"block_index": 0, "old_tag": "Misc-exposition"},
            {"block_index": 1, "old_tag": "Description"},
        ]
        result = add_notes(ann, changes)
        assert "#1\tAnnotatorNotes T1\twas: Misc-exposition" in result
        assert "#2\tAnnotatorNotes T2\twas: Description" in result
