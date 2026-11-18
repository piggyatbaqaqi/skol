"""Tests for bin/build_vocab_spellings.py.

Proposal 2a of docs/plans/vocab-tree-narrowing.md (Trello AZeJMCR3).

Run with: python -m pytest bin/build_vocab_spellings_test.py -v
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_vocab_spellings import (  # type: ignore[import]  # noqa: E402
    DEFAULT_MIN_SUPPORT,
    build_parser,
    choose_spellings,
    count_spellings,
    table_text,
)
from treatments_to_structured.vocab_canonical import (  # noqa: E402
    load_spellings,
)


def test_default_min_support() -> None:
    assert DEFAULT_MIN_SUPPORT == 2


class TestCountSpellings:
    def test_once_per_document(self) -> None:
        docs: List[Dict[str, Any]] = [
            {'json_annotated': {'Conidia': {}, 'conidia': {}}},
            {'json_annotated': {'Conidia': {}}}]
        assert count_spellings(docs)['conidium'] == Counter(
            {'Conidia': 2, 'conidia': 1})

    def test_every_depth(self) -> None:
        docs = [{'json_annotated': {'Pileus': {'Colour': ['brown']}}}]
        counts = count_spellings(docs)
        assert counts['pileus'] == Counter({'Pileus': 1})
        assert counts['color'] == Counter({'Colour': 1})

    def test_candidates_are_cleaned(self) -> None:
        docs: List[Dict[str, Any]] = [
            {'json_annotated': {'`Culture_characteristics`': {}}}]
        assert count_spellings(docs)['culture characteristic'] == Counter(
            {'Culture characteristics': 1})

    def test_values_are_not_counted(self) -> None:
        docs = [{'json_annotated': {'Pileus': {'Color': ['Conidia']}}}]
        assert 'conidium' not in count_spellings(docs)

    def test_string_annotations_and_unusable_documents(self) -> None:
        docs = [{'json_annotated': '{"Asci": {}}'}, {'json_annotated': 'x'},
                {'_id': 'no annotation'}]
        assert count_spellings(docs) == {'ascus': Counter({'Asci': 1})}


class TestChooseSpellings:
    def test_most_frequent(self) -> None:
        counts = {'conidium': Counter({'Conidia': 5716, 'Conidium': 339})}
        assert choose_spellings(counts) == {'conidium': 'Conidia'}

    def test_tie_is_alphabetical(self) -> None:
        counts = {'pileus': Counter({'pileus': 3, 'Pileus': 3})}
        assert choose_spellings(counts) == {'pileus': 'Pileus'}

    def test_support_is_the_group_total(self) -> None:
        """Two documents, two spellings: the group is supported."""
        counts = {'ascus': Counter({'Asci': 1, 'asci': 1}),
                  'rare': Counter({'Rare': 1})}
        assert choose_spellings(counts, min_support=2) == {'ascus': 'Asci'}


class TestTableText:
    def test_round_trips_through_the_loader(self, tmp_path: Path) -> None:
        path = tmp_path / 'sp.json'
        path.write_text(table_text({'pileus': 'Pileus',
                                    'conidium': 'Conidia'},
                                   {'experiment': 'production_v4_1'}))
        assert load_spellings(path) == {'pileus': 'Pileus',
                                        'conidium': 'Conidia'}

    def test_deterministic(self) -> None:
        a = table_text({'b': 'B', 'a': 'A'}, {'x': 1})
        b = table_text({'a': 'A', 'b': 'B'}, {'x': 1})
        assert a == b
        assert a.endswith('\n')

    def test_records_how_it_was_generated(self) -> None:
        text = table_text({}, {'experiment': 'production_v4_1',
                               'min_support': 2})
        assert json.loads(text)['_generated'] == {
            'experiment': 'production_v4_1', 'min_support': 2}

    def test_unicode_is_written_as_is(self) -> None:
        assert 'Ä' in table_text({'äb': 'Äb'}, {})


class TestParser:
    def test_flags(self, tmp_path: Path) -> None:
        args = build_parser().parse_args([
            '--experiment', 'x', '--min-support', '3',
            '--output', str(tmp_path / 'o.json')])
        assert args.min_support == 3
        assert args.output == str(tmp_path / 'o.json')

    def test_strict(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--min-suport', '3'])
