"""Tests for span.Span."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from span import Span  # noqa: E402


def _span(**kw) -> Span:
    base = dict(
        paragraph_number=1, start_line=10, end_line=12,
        start_char=100, end_char=200, pdf_page=3,
        pdf_label='105', empirical_page='7',
    )
    base.update(kw)
    return Span(**base)


class TestConstruction:
    def test_stores_all_positions(self) -> None:
        s = _span()
        assert (s.paragraph_number, s.start_line, s.end_line) == (1, 10, 12)
        assert (s.start_char, s.end_char) == (100, 200)
        assert (s.pdf_page, s.pdf_label, s.empirical_page) == (3, '105', '7')

    def test_optional_fields_accept_none(self) -> None:
        s = _span(pdf_label=None, empirical_page=None)
        assert s.pdf_label is None and s.empirical_page is None


class TestGaps:
    """Gap arithmetic compares character offsets, which is why it is
    unaffected by which attachment the offsets index — both spans
    live in the same coordinate space."""

    def test_gap_detected(self) -> None:
        assert _span(start_char=300).has_gap_before(_span(end_char=200))

    def test_adjacent_spans_have_no_gap(self) -> None:
        assert not _span(start_char=200).has_gap_before(_span(end_char=200))

    def test_overlapping_spans_have_no_gap(self) -> None:
        assert not _span(start_char=150).has_gap_before(_span(end_char=200))

    def test_gap_size(self) -> None:
        assert _span(start_char=300).gap_size(_span(end_char=200)) == 100

    def test_gap_size_is_zero_when_adjacent_or_overlapping(self) -> None:
        assert _span(start_char=200).gap_size(_span(end_char=200)) == 0
        assert _span(start_char=150).gap_size(_span(end_char=200)) == 0


class TestSerialisation:
    def test_round_trips(self) -> None:
        s = _span()
        assert Span.from_dict(s.as_dict()) == s

    def test_head_omitted_from_dict_when_absent(self) -> None:
        """Legacy spans stay byte-identical; no churn on 81k docs.
        Passes today because `head` does not exist yet, and must keep
        passing once it does -- hence outside the xfail block."""
        assert 'head' not in _span().as_dict()

    def test_from_dict_tolerates_absent_optional_keys(self) -> None:
        d = _span().as_dict()
        del d['pdf_label'], d['empirical_page']
        restored = Span.from_dict(d)
        assert restored.pdf_label is None
        assert restored.empirical_page is None


class TestEquality:
    def test_equal_positions_are_equal(self) -> None:
        assert _span() == _span()

    def test_differing_position_is_unequal(self) -> None:
        assert _span() != _span(start_char=999)

    def test_other_types_are_not_equal(self) -> None:
        assert _span().__eq__('not a span') is NotImplemented


class TestRepr:
    def test_mentions_the_positions(self) -> None:
        text = repr(_span())
        assert '10-12' in text and '100-200' in text


@pytest.mark.xfail(
    reason="2026-08-21: Span.head not implemented yet", strict=True,
)
class TestSpanHead:
    """``head`` is the fingerprint that makes a wrong resolution loud.

    Without it, reading a span's offsets against the wrong attachment
    returns plausible prose rather than an error — see §16 in
    ``docs/data_quality_production_v4_model.md``.
    """

    def test_head_defaults_to_none(self) -> None:
        assert _span().head is None

    def test_head_round_trips_through_as_dict(self) -> None:
        assert _span(head='Stroma thin, whitish').as_dict()['head'] == \
            'Stroma thin, whitish'

    def test_from_dict_reads_head(self) -> None:
        assert Span.from_dict(_span(head='abc').as_dict()).head == 'abc'

    def test_from_dict_tolerates_missing_head(self) -> None:
        assert Span.from_dict(_span().as_dict()).head is None

    def test_equality_ignores_head(self) -> None:
        """head is derived from the text, not part of the position:
        two spans at the same offsets are the same span."""
        assert _span(head='abc') == _span()
