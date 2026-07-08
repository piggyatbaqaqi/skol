"""Tests for ``constants`` — currently just the ``pdf_page_pattern`` regex.

The wrapped-form cases are the Trello #399 regression: ``annotate_v4``
emits page markers as ``[@--- PDF Page N Label N ---#Page-header*]``
inside .ann attachments, and the pre-fix pattern's ``^---`` /
``---\\s*$`` anchors rejected the wrapping, leaving ``pdf_page`` at 0
for every treatment extracted through ``fileobj.FileObject.read_line``
and ``pdf_section_extractor._get_pdf_page_marker``.
"""
import re

import pytest

from constants import pdf_page_pattern


_YEDDA_XFAIL = pytest.mark.xfail(
    reason="Trello #399 — no YEDDA-wrapped page marker support yet",
    strict=True,
)


class TestPdfPagePattern:
    def test_plain_marker_matches_with_label(self) -> None:
        m = re.match(pdf_page_pattern, '--- PDF Page 1 Label 1 ---')
        assert m is not None
        assert int(m.group(1)) == 1

    def test_plain_marker_matches_page_only(self) -> None:
        m = re.match(pdf_page_pattern, '--- PDF Page 7 ---')
        assert m is not None
        assert int(m.group(1)) == 7

    @_YEDDA_XFAIL
    def test_yedda_wrapped_marker_matches_with_label(self) -> None:
        m = re.match(
            pdf_page_pattern,
            '[@--- PDF Page 77 Label 77 ---#Page-header*]',
        )
        assert m is not None
        assert int(m.group(1)) == 77

    @_YEDDA_XFAIL
    def test_yedda_wrapped_marker_matches_page_only(self) -> None:
        m = re.match(pdf_page_pattern, '[@--- PDF Page 3 ---#Page-header*]')
        assert m is not None
        assert int(m.group(1)) == 3

    @_YEDDA_XFAIL
    def test_yedda_wrapped_non_numeric_label_matches(self) -> None:
        m = re.match(
            pdf_page_pattern,
            '[@--- PDF Page 5 Label v ---#Page-header*]',
        )
        assert m is not None
        assert int(m.group(1)) == 5
        assert m.group(3) == 'v'

    def test_non_marker_lines_do_not_match(self) -> None:
        for line in [
            '',
            'Some text',
            '[@Ascomata elongate...#Description*]',
            '--- PDF something else ---',
            '--- PDF Page abc ---',
        ]:
            assert re.match(pdf_page_pattern, line) is None, line
