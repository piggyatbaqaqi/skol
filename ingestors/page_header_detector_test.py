"""Tests for ingestors/page_header_detector.py.

Pytest-compatible.  Each test class targets one sub-step of the
algorithm described in docs/page-header-detection.md (v4 plan §1.B
sub-rows 1.B.1 through 1.B.5):

    1.B.1 — TestCollectCandidates        — digit-token discovery
    1.B.2 — TestFitSequence              — RANSAC sequence fit
    1.B.3 — TestPartitionAlternation     — recto/verso check
    1.B.4 — TestClusterHeaderText        — journal-name clustering
    1.B.5 — TestRecoverHeaderBlock,      — block recovery + orchestrator
            TestDetectPageHeaders
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.page_header_detector import (  # noqa: E402
    PageNumCandidate,
    collect_candidates,
)


class TestCollectCandidates:
    """Per page-header-detection.md §Step 1: digit tokens of 1-4 digits
    at the start or end of ``line.strip()``, year-shaped tokens at
    line-end excluded, 5+ digit tokens excluded outright."""

    def test_digit_at_start(self) -> None:
        """Page number at line-start (left-side / verso convention)."""
        result = collect_candidates(['42  MYCOLOGIA'])
        assert len(result) == 1
        c = result[0]
        assert c.line_index == 0
        assert c.position == 'start'
        assert c.value == 42
        assert c.raw_token == '42'

    def test_digit_at_end(self) -> None:
        """Page number at line-end (right-side / recto convention)."""
        result = collect_candidates(['Smith et al. 2023   17'])
        # '2023' is year-shaped at-end and excluded; '17' is the page.
        positions = sorted((c.position, c.value) for c in result)
        assert ('end', 17) in positions
        assert ('end', 2023) not in positions

    def test_year_shape_rejected_at_end(self) -> None:
        """`(19|20)\\d{2}` at line-end is a year, not a page number."""
        assert collect_candidates(['Some Title 2024']) == []
        assert collect_candidates(['Title 1999']) == []

    def test_year_shape_accepted_at_start(self) -> None:
        """Per §Step 1 we only reject year-shape at line-end.  A
        4-digit page at line-start is rare but accepted; the sequence
        fit in 1.B.2 weeds out false positives anyway."""
        result = collect_candidates(['2024 Random Header'])
        assert len(result) == 1
        assert result[0].position == 'start'
        assert result[0].value == 2024

    def test_five_digit_rejected(self) -> None:
        """Specimen IDs, accession numbers, and other 5+ digit
        tokens are excluded outright (§Step 1)."""
        assert collect_candidates(['12345 Some Text']) == []
        assert collect_candidates(['Some Text 12345']) == []

    def test_embedded_digit_rejected(self) -> None:
        """Digits in the middle of a line are body text, not page
        markers."""
        assert collect_candidates(['Found 42 specimens in study']) == []

    def test_empty_and_whitespace_lines(self) -> None:
        """Blank lines yield no candidates."""
        assert collect_candidates(['']) == []
        assert collect_candidates(['   ']) == []
        assert collect_candidates(['\t\n']) == []

    def test_prefix_and_suffix_captured(self) -> None:
        """The non-numeric portion before/after the digit token must
        be captured for 1.B.4 clustering."""
        result = collect_candidates(['12  MYCOLOGIA volume 5'])
        assert len(result) == 1
        c = result[0]
        assert c.value == 12
        assert c.prefix == ''
        # Strip whitespace from suffix so clustering can use it directly.
        assert c.suffix.strip() == 'MYCOLOGIA volume 5'

    def test_multiple_lines_with_indices(self) -> None:
        """line_index preserves position in the input list."""
        lines = [
            'body text',
            '42  HEADER',
            'more body',
            'HEADER  43',
        ]
        result = collect_candidates(lines)
        indices = sorted(c.line_index for c in result)
        assert indices == [1, 3]
