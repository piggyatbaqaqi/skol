"""Tests for bin/systematic_names_wordlist."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from systematic_names_wordlist import (  # noqa: E402
    build_wordlist,
    cell_text,
    greek_forms,
    latin_forms,
    parse_rows,
)


_PAGE = '''
<table class="wikitable">
<tbody><tr><th>Latin/Greek</th><th>Language</th><th>English</th></tr>
<tr><td><span lang="la">acanthus</span> etc.</td>
<td>G <span lang="grc"><a href="x">ἄκανθος</a></span></td>
<td>thorny, spiny</td></tr>
<tr><td>actin-, actino-</td>
<td>G <span lang="grc">ἀκτίς</span></td>
<td>ray, radial</td></tr>
<tr><td>acaulis</td><td>L</td><td>stemless</td></tr>
</tbody></table>
'''


class TestCellText:
    def test_strips_markup_and_unescapes(self) -> None:
        assert cell_text('<i>a</i>&amp;b ') == 'a&b'


class TestParseRows:
    def test_header_rows_dropped(self) -> None:
        assert len(parse_rows(_PAGE)) == 3

    def test_greek_originals_captured(self) -> None:
        rows = parse_rows(_PAGE)
        assert rows[0][1] == ['ἄκανθος']
        assert rows[2][1] == []

    def test_no_tables_yields_nothing(self) -> None:
        assert parse_rows('<p>no tables here</p>') == []


class TestLatinForms:
    def test_splits_on_commas_and_strips_hyphens(self) -> None:
        assert latin_forms('actin-, actino-') == {'actin', 'actino'}

    def test_drops_etc_marker(self) -> None:
        assert latin_forms('acanthus etc.') == {'acanthus'}

    def test_drops_short_forms(self) -> None:
        assert latin_forms('a, ab') == set()


class TestGreekForms:
    def test_yields_bare_nominative_and_combining(self) -> None:
        got = greek_forms(['ἄκανθος'])
        assert {'acanthos', 'acanthus', 'acantho'} <= got


class TestBuildWordlist:
    def test_union_is_sorted_and_deduplicated(self) -> None:
        words = build_wordlist(_PAGE)
        assert words == sorted(set(words))
        assert {'acanthus', 'acantho', 'actino', 'acaulis'} <= set(words)
