"""Tests for bin/extract_latin_wordlist.

The parser is exercised against small HTML fragments in the exact
shape Latdict's word-list pages use; the fixture-free helpers are
tested directly.
"""

import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_latin_wordlist import (  # noqa: E402
    expand_variants,
    fold_form,
    parse_entries,
    wordlist_from_html,
)


_XFAIL = pytest.mark.xfail(
    reason="2026-08-21: extractor not implemented yet",
    strict=True,
)


def _li(word: str, pos: str = 'n') -> str:
    return (
        f'<li id="word-0" class="word">'
        f'<a href="https://latin-dictionary.net/definition/1/x">'
        f'{word}</a> {pos}</li>'
    )


@_XFAIL
class TestParseEntries:
    """Each <li class="word"> yields one entry; the anchor text is
    a comma-separated list of principal parts."""

    def test_single_entry(self) -> None:
        assert parse_entries(_li('kaput, kapitis')) == [
            ('kaput, kapitis', 'n'),
        ]

    def test_multiple_entries(self) -> None:
        html = _li('albus, alba, album', 'adj') + _li('caro', 'n')
        assert parse_entries(html) == [
            ('albus, alba, album', 'adj'),
            ('caro', 'n'),
        ]

    def test_ignores_non_word_markup(self) -> None:
        html = '<li class="alpha-page"><a href="a.html">A</a></li>'
        assert parse_entries(html) == []

    def test_empty_html(self) -> None:
        assert parse_entries('') == []


@_XFAIL
class TestExpandVariants:
    """Latdict compresses orthographic alternatives two ways."""

    def test_plain_form_passes_through(self) -> None:
        assert expand_variants('kaput') == ['kaput']

    def test_parenthetical_suffix_yields_both(self) -> None:
        """``kalendari(i)`` is kalendari OR kalendarii."""
        assert sorted(expand_variants('kalendari(i)')) == [
            'kalendari', 'kalendarii',
        ]

    def test_slash_alternation_replaces_equal_suffix(self) -> None:
        """``Babylonos/is`` is Babylonos OR Babylonis — the chunk
        after the slash replaces an equal-length suffix."""
        assert sorted(expand_variants('Babylonos/is')) == [
            'Babylonis', 'Babylonos',
        ]

    def test_marker_tokens_are_dropped(self) -> None:
        for junk in ('(gen.)', '-', 'abb.', 'Aug.', ''):
            assert expand_variants(junk) == [], junk


@_XFAIL
class TestFoldForm:
    """Folding happens in the WORD LIST, never in treatment text."""

    def test_lowercases(self) -> None:
        assert fold_form('Karthago') == 'karthago'

    def test_strips_macrons(self) -> None:
        assert fold_form('abscīdō') == 'abscido'

    def test_folds_ae_ligature(self) -> None:
        assert fold_form('Cæsar') == 'caesar'

    def test_rejects_non_alphabetic(self) -> None:
        assert fold_form('a1b') is None
        assert fold_form('') is None


@_XFAIL
class TestWordlistFromHtml:
    """End-to-end over a page fragment."""

    def test_splits_principal_parts_and_folds(self) -> None:
        html = (
            _li('albus, alba, album', 'adj')
            + _li('kalendarium, kalendari(i)')
            + _li('K., abb.')
        )
        assert wordlist_from_html(html) == sorted({
            'albus', 'alba', 'album',
            'kalendarium', 'kalendari', 'kalendarii',
        })

    def test_deduplicates_across_entries(self) -> None:
        html = _li('caro, carnis') + _li('Caro')
        assert wordlist_from_html(html) == ['carnis', 'caro']

    def test_drops_single_character_forms(self) -> None:
        """One-letter 'words' are abbreviation residue and would
        make the D8 rejoin metric fire on ordinary text."""
        assert 'a' not in wordlist_from_html(_li('a, ab'))
