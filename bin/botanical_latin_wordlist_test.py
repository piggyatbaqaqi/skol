"""Tests for bin/botanical_latin_wordlist."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from botanical_latin_wordlist import (  # noqa: E402
    attested_forms,
    build_wordlist,
    expand_headword,
    parse_headwords,
)


_XFAIL = pytest.mark.xfail(
    reason="2026-08-21: botanical_latin_wordlist not implemented yet",
    strict=True,
)

_PAGE = '''
<b><i>bacca, -ae</i></b> - noun, 1st declension, feminine: berry<br>
&nbsp;&nbsp;&nbsp;<i>baccis</i> - ablative plural [p. 173.
(<a href="x"><i>Lonicera nigra</i></a>)]<br>
<b><i>baccatus, -a, -um</i></b> - adjective, group A: berry-like<br>
&nbsp;&nbsp;&nbsp;<i>baccatis</i> - ablative plural masculine [p. 365.]<br>
<b><i>acaulis, -is, -e</i></b> - adjective, group B: stemless<br>
<b><i>abrupte</i></b> - adverb: abruptly<br>
'''


@_XFAIL
class TestParseHeadwords:
    def test_finds_bold_italic_headwords(self) -> None:
        assert parse_headwords(_PAGE) == [
            'bacca, -ae', 'baccatus, -a, -um',
            'acaulis, -is, -e', 'abrupte',
        ]

    def test_ignores_taxon_names_in_citations(self) -> None:
        """Citations are full of <i>Genus species</i> links; only
        <b><i>...</i></b> is a headword."""
        assert 'Lonicera nigra' not in ' '.join(parse_headwords(_PAGE))


@_XFAIL
class TestExpandHeadword:
    """A -SUF part replaces the lemma's nominative ending."""

    def test_adjective_three_genders(self) -> None:
        assert expand_headword('baccatus, -a, -um') == {
            'baccatus', 'baccata', 'baccatum',
        }

    def test_noun_genitive(self) -> None:
        assert expand_headword('bacca, -ae') == {'bacca', 'baccae'}

    def test_third_declension_adjective(self) -> None:
        assert expand_headword('acaulis, -is, -e') == {
            'acaulis', 'acaule',
        }

    def test_bare_lemma(self) -> None:
        assert expand_headword('abrupte') == {'abrupte'}

    def test_spelled_out_alternatives_kept_whole(self) -> None:
        assert expand_headword('acaulos, acaulos, acaulon') == {
            'acaulos', 'acaulon',
        }


@_XFAIL
class TestAttestedForms:
    """The indented lines are real inflected forms from Linnaeus —
    the part no lemma list can supply."""

    def test_collects_indented_italic_forms(self) -> None:
        assert attested_forms(_PAGE) == {'baccis', 'baccatis'}

    def test_excludes_headwords_and_citations(self) -> None:
        got = attested_forms(_PAGE)
        assert 'bacca' not in got and 'Lonicera' not in got


@_XFAIL
class TestBuildWordlist:
    def test_union_sorted_and_folded(self) -> None:
        words = build_wordlist(_PAGE)
        assert words == sorted(set(words))
        assert {'bacca', 'baccae', 'baccis', 'baccatum', 'acaule'} <= set(words)
