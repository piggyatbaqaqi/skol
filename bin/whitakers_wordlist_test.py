"""Tests for bin/whitakers_wordlist.

Fragments are in the exact fixed-width / token shapes that
DICTLINE.GEN and INFLECTS.LAT use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from whitakers_wordlist import (  # noqa: E402
    build_wordlist,
    endings_for,
    fold_form,
    forms_for_entry,
    parse_dictline,
    parse_inflections,
)


def _dl(stems, pos, codes, age='X', defn='white;'):
    """Build one fixed-width DICTLINE record: 4x19-char stems,
    6-char POS, 18-char code block, then AGE AREA GEO FREQ SOURCE."""
    padded = [s.ljust(19) for s in (stems + [''] * 4)[:4]]
    return (''.join(padded) + pos.ljust(6) + codes.ljust(18)
            + f'{age} X X A O ' + defn)


_ADJ_INFLECTS = """
ADJ   1 1 NOM S M POS   1 2 us    X A
ADJ   1 0 NOM S F POS   1 1 a     X A
ADJ   1 0 ACC S M POS   2 2 um    X A
ADJ   0 0 NOM S C COMP  3 2 or    X A
N     1 1 NOM S C  1 1 a         X A
N     1 1 GEN S C  2 2 ae        X A
V     1 1 PRES ACTIVE IND 1 S 1 1 o   X A
-- a comment line, ignored
"""


class TestParseInflections:
    def test_keys_by_pos_decl_variant(self) -> None:
        infl = parse_inflections(_ADJ_INFLECTS)
        assert ('ADJ', '1', '1') in infl
        assert ('N', '1', '1') in infl
        assert ('V', '1', '1') in infl

    def test_records_carry_stem_key_and_ending(self) -> None:
        infl = parse_inflections(_ADJ_INFLECTS)
        assert (1, 'us') in [(sk, e) for sk, e, _ in infl[('ADJ', '1', '1')]]

    def test_comments_and_blanks_ignored(self) -> None:
        infl = parse_inflections(_ADJ_INFLECTS)
        assert all(k[0] in ('ADJ', 'N', 'V') for k in infl)


class TestEndingsFor:
    """A 0 in the INFLECTS declension or variant column is a
    wildcard: ADJ 1 0 applies to every variant of declension 1,
    and ADJ 0 0 (comparative/superlative) to every adjective."""

    def test_exact_match(self) -> None:
        infl = parse_inflections(_ADJ_INFLECTS)
        assert (1, 'us', 'M') in endings_for(infl, 'ADJ', '1', '1')

    def test_variant_wildcard_included(self) -> None:
        infl = parse_inflections(_ADJ_INFLECTS)
        got = endings_for(infl, 'ADJ', '1', '1')
        assert (1, 'a', 'F') in got, 'ADJ 1 0 must apply to variant 1'

    def test_declension_wildcard_included(self) -> None:
        infl = parse_inflections(_ADJ_INFLECTS)
        got = endings_for(infl, 'ADJ', '1', '1')
        assert (3, 'or', 'C') in got, 'ADJ 0 0 COMP must apply'

    def test_unknown_key_is_empty(self) -> None:
        assert endings_for(parse_inflections(_ADJ_INFLECTS),
                           'ADV', '9', '9') == []


class TestParseDictline:
    def test_extracts_stems_pos_codes_age(self) -> None:
        line = _dl(['alb', 'alb', 'albi', 'albissi'], 'ADJ', ' 1 1 X')
        entry = parse_dictline(line)
        assert entry is not None
        assert entry.stems == ['alb', 'alb', 'albi', 'albissi']
        assert entry.pos == 'ADJ'
        assert entry.codes[:2] == ['1', '1']
        assert entry.age == 'X'

    def test_reads_the_age_column(self) -> None:
        entry = parse_dictline(_dl(['fung'], 'N', ' 2 1 M T', age='G'))
        assert entry is not None and entry.age == 'G'

    def test_blank_line_returns_none(self) -> None:
        assert parse_dictline('') is None
        assert parse_dictline('   ') is None


class TestFormsForEntry:
    def test_generates_inflected_forms(self) -> None:
        infl = parse_inflections(_ADJ_INFLECTS)
        entry = parse_dictline(
            _dl(['alb', 'alb', 'albi', 'albissi'], 'ADJ', ' 1 1 X'))
        forms = forms_for_entry(entry, infl)
        assert {'albus', 'alba', 'album', 'albior'} <= forms

    def test_skips_empty_stem_slots(self) -> None:
        """Stem 3/4 are blank for most nouns; no bare-ending forms."""
        infl = parse_inflections(_ADJ_INFLECTS)
        entry = parse_dictline(_dl(['ros'], 'N', ' 1 1 F T'))
        assert all(f.startswith('ros') for f in forms_for_entry(entry, infl))

    def test_non_inflecting_pos_contributes_stems_verbatim(self) -> None:
        """An adverb's DICTLINE stems ARE full words -- WORDS
        stores `bene / melius / optime` as the three stems of one
        ADV entry, so all three belong in the list."""
        infl = parse_inflections(_ADJ_INFLECTS)
        entry = parse_dictline(
            _dl(['bene', 'melius', 'optime'], 'ADV', ' X'))
        assert forms_for_entry(entry, infl) == {
            'bene', 'melius', 'optime',
        }

    def test_inflecting_pos_without_codes_yields_nothing(self) -> None:
        infl = parse_inflections(_ADJ_INFLECTS)
        entry = parse_dictline(_dl(['xyz'], 'ADJ', ''))
        assert forms_for_entry(entry, infl) == set()


class TestFoldForm:
    """Folding applies to the WORD LIST only — never to treatment
    text, where it would invalidate stored span offsets."""

    def test_lowercases(self) -> None:
        assert fold_form('Karthago') == 'karthago'

    def test_strips_macrons_and_ligatures(self) -> None:
        assert fold_form('abscīdō') == 'abscido'
        assert fold_form('Cæsar') == 'caesar'

    def test_rejects_non_alphabetic_and_single_chars(self) -> None:
        assert fold_form('a1b') is None
        assert fold_form('a') is None
        assert fold_form('') is None


class TestBuildWordlist:
    def test_end_to_end(self) -> None:
        dictline = '\n'.join([
            _dl(['alb', 'alb', 'albi', 'albissi'], 'ADJ', ' 1 1 X'),
            _dl(['ros', 'ros'], 'N', ' 1 1 F T'),
        ])
        words = build_wordlist(dictline, _ADJ_INFLECTS)
        assert 'albus' in words and 'rosa' in words
        assert words == sorted(set(words))

    def test_age_filter_restricts(self) -> None:
        """--age G,H is how the post-15th-century scientific and
        modern vocabulary is isolated."""
        dictline = '\n'.join([
            _dl(['alb', 'alb', 'albi', 'albissi'], 'ADJ', ' 1 1 X',
                age='X'),
            _dl(['fung', 'fung'], 'N', ' 2 1 M T', age='G'),
        ])
        assert 'albus' in build_wordlist(dictline, _ADJ_INFLECTS)
        restricted = build_wordlist(dictline, _ADJ_INFLECTS, ages={'G', 'H'})
        assert 'albus' not in restricted

    def test_empty_age_set_means_no_filter(self) -> None:
        dictline = _dl(['alb', 'alb', 'albi', 'albissi'], 'ADJ', ' 1 1 X')
        assert build_wordlist(dictline, _ADJ_INFLECTS, ages=None)
