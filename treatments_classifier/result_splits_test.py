"""Tests for result_splits (plan F1).

Approved by the user 2026-11-23; they ran as strict xfail against the
skeleton in commit 9c0ce57.
"""

from typing import Any, Dict, List

import pytest

from treatments_classifier.result_splits import (
    feature_score, score_structured, score_text, split_score,
    structured_features, text_ngrams)


def colour_docs() -> List[Dict[str, Any]]:
    """The plan's worked example: 200 results, Brown 100, Yellow 80,
    Blue 10, colour not mentioned 20.  Ten results list two colours."""
    docs: List[Dict[str, Any]] = []
    docs += [{'Ascomata': {'Color': ['Brown', 'Blue']}}] * 10
    docs += [{'Ascomata': {'Color': ['Brown']}}] * 90
    docs += [{'Ascomata': {'Color': ['Yellow']}}] * 80
    docs += [{'Asci': {'Ring': ['amyloid']}}] * 20
    return docs


def find(rows: List[Dict[str, Any]], path: List[str]) -> Dict[str, Any]:
    """The row whose path is ``path``."""
    matches = [r for r in rows if r['path'] == path]
    assert len(matches) == 1, f'{path} in {[r["path"] for r in rows]}'
    return matches[0]


def count_attributes(parts: List[Dict[str, Any]]) -> int:
    """Attributes in a part tree, at every depth."""
    return sum(len(p['attributes']) + count_attributes(p['parts'])
               for p in parts)


class TestSplitScore:

    def test_worked_colour_example(self) -> None:
        # Four answers summing to 210 > 200: weights use their sum.
        assert split_score([100, 80, 10, 20], 200) == pytest.approx(
            125.5 / 210)

    def test_one_answer_for_everyone_scores_zero(self) -> None:
        assert split_score([200], 200) == 0.0

    def test_no_results_scores_zero(self) -> None:
        assert split_score([], 0) == 0.0
        assert split_score([0, 0], 10) == 0.0

    def test_every_result_giving_every_answer_scores_zero(self) -> None:
        assert split_score([2, 2], 2) == 0.0


class TestFeatureScore:

    def test_even_split_is_half(self) -> None:
        assert feature_score(100, 200) == pytest.approx(0.5)

    def test_shared_or_absent_feature_scores_zero(self) -> None:
        assert feature_score(200, 200) == 0.0
        assert feature_score(0, 200) == 0.0

    def test_matches_yes_no_split(self) -> None:
        assert feature_score(30, 200) == pytest.approx(
            split_score([30, 170], 200))
        assert feature_score(30, 200) == pytest.approx(2 * 30 * 170 / 200**2)


class TestStructuredFeatures:

    def test_every_level_of_the_path(self) -> None:
        parts, attributes = structured_features({
            'Ascomata': {'Color': ['Brown', 'Black'],
                         'Peridium': {'Texture': 'smooth'}}})
        assert parts == {('Ascomata',), ('Ascomata', 'Peridium')}
        assert attributes == {
            ('Ascomata', 'Color'): {'Brown', 'Black'},
            ('Ascomata', 'Peridium', 'Texture'): {'smooth'},
        }

    def test_list_indices_are_dropped(self) -> None:
        parts, attributes = structured_features({
            'Asci': [{'Ring': ['amyloid']}, {'Ring': ['inamyloid']}]})
        assert parts == {('Asci',)}
        assert attributes == {('Asci', 'Ring'): {'amyloid', 'inamyloid'}}

    def test_keys_are_kept_as_written(self) -> None:
        # The old tokens joined paths with '_', which keys also contain.
        parts, attributes = structured_features(
            {'Ascomata_Color': ['Brown']})
        assert parts == set()
        assert attributes == {('Ascomata_Color',): {'Brown'}}

    def test_a_key_can_be_part_and_attribute(self) -> None:
        parts, attributes = structured_features(
            {'Stipe': ['central', {'Color': ['white']}]})
        assert parts == {('Stipe',)}
        assert attributes == {('Stipe',): {'central'},
                              ('Stipe', 'Color'): {'white'}}

    def test_empty_values_are_ignored(self) -> None:
        parts, attributes = structured_features(
            {'Pileus': {'Color': ['', None], 'Shape': 'convex'}})
        assert parts == {('Pileus',)}
        assert attributes == {('Pileus', 'Shape'): {'convex'}}

    def test_repeated_values_count_once(self) -> None:
        _, attributes = structured_features(
            {'Pileus': [{'Color': ['brown']}, {'Color': ['brown']}]})
        assert attributes == {('Pileus', 'Color'): {'brown'}}


class TestScoreStructured:

    def test_worked_colour_example(self) -> None:
        result = score_structured(colour_docs())
        assert result['n_results'] == 200
        ascomata = find(result['parts'], ['Ascomata'])
        assert ascomata['described'] == 180
        colour = find(ascomata['attributes'], ['Ascomata', 'Color'])
        assert colour['score'] == pytest.approx(0.598, abs=1e-3)
        assert colour['not_mentioned'] == 20
        assert [(v['value'], v['count']) for v in colour['values']] == [
            ('Brown', 100), ('Yellow', 80), ('Blue', 10)]
        assert [v['others'] for v in colour['values']] == pytest.approx(
            [0.5, 0.6, 0.95])
        assert colour['values'][0]['display_text'] == 'Ascomata color brown'

    def test_part_score_is_its_best_feature(self) -> None:
        result = score_structured(colour_docs())
        ascomata = find(result['parts'], ['Ascomata'])
        asci = find(result['parts'], ['Asci'])
        # Ascomata: described 180/200 scores 0.18; its colour, 0.598.
        assert ascomata['score'] == pytest.approx(0.598, abs=1e-3)
        # Asci: described 20/200 and its ring both score 0.18.
        assert asci['described'] == 20
        assert asci['score'] == pytest.approx(0.18)
        assert [p['path'] for p in result['parts']] == [
            ['Ascomata'], ['Asci']]

    def test_nested_part_lifts_its_parent(self) -> None:
        # A's own attribute is shared by all; its nested B/Texture splits
        # 2/2 (0.5).  C's colour splits 3/1 (0.375).  A comes first.
        docs = [{'A': {'Shape': ['x'], 'B': {'Texture': [t]}},
                 'C': {'Color': [c]}}
                for t, c in [('smooth', 'red'), ('smooth', 'red'),
                             ('rough', 'red'), ('rough', 'blue')]]
        result = score_structured(docs)
        assert [p['path'] for p in result['parts']] == [['A'], ['C']]
        a = result['parts'][0]
        assert a['score'] == pytest.approx(0.5)
        assert find(a['attributes'], ['A', 'Shape'])['score'] == 0.0
        b = find(a['parts'], ['A', 'B'])
        assert find(b['attributes'], ['A', 'B', 'Texture'])['score'] == (
            pytest.approx(0.5))
        assert result['parts'][1]['score'] == pytest.approx(0.375)

    def test_values_differing_only_in_case_are_one_value(self) -> None:
        docs = [{'P': {'Color': [c]}}
                for c in ['Brown', 'Brown', 'brown', 'Black']]
        colour = find(find(score_structured(docs)['parts'], ['P'])
                      ['attributes'], ['P', 'Color'])
        # Shown in its most common written form.
        assert [(v['value'], v['count']) for v in colour['values']] == [
            ('Brown', 3), ('Black', 1)]

    def test_results_without_a_description_count_in_n(self) -> None:
        result = score_structured([{'P': {'C': ['x']}}, {}])
        assert result['n_results'] == 2
        p = find(result['parts'], ['P'])
        assert p['described'] == 1
        assert find(p['attributes'], ['P', 'C'])['not_mentioned'] == 1

    def test_top_n_caps_attributes(self) -> None:
        docs = [{'A': {'Shape': ['x'], 'B': {'Texture': [t]}},
                 'C': {'Color': [c]}}
                for t, c in [('smooth', 'red'), ('smooth', 'red'),
                             ('rough', 'red'), ('rough', 'blue')]]
        result = score_structured(docs, top_n=1)
        assert count_attributes(result['parts']) == 1
        # Parts left with no attribute beneath them are dropped.
        assert [p['path'] for p in result['parts']] == [['A']]
        a = result['parts'][0]
        assert a['attributes'] == []
        assert find(a['parts'], ['A', 'B'])['attributes'][0]['path'] == [
            'A', 'B', 'Texture']

    def test_no_results(self) -> None:
        assert score_structured([]) == {'n_results': 0, 'parts': []}


class TestTextNgrams:

    def test_words_and_word_pairs_folded(self) -> None:
        assert text_ngrams('Clamp connections présent') == {
            'clamp', 'connections', 'present',
            'clamp connections', 'connections present'}

    def test_one_letter_words_are_dropped(self) -> None:
        # As today's TF-IDF tokenizer does (two or more word characters).
        assert text_ngrams('a b spores') == {'spores'}


class TestScoreText:

    def test_ngrams_in_the_same_results_collapse_to_the_longest(
            self) -> None:
        texts = ['spores clamp connections', 'clamp connections and spores',
                 'spores', 'spores']
        result = score_text(texts)
        assert result['n_results'] == 4
        rows = {r['text']: r for r in result['features']}
        assert 'clamp' not in rows and 'connections' not in rows
        assert rows['clamp connections']['count'] == 2
        assert rows['clamp connections']['score'] == pytest.approx(0.5)
        assert rows['spores']['count'] == 4
        assert rows['spores']['score'] == 0.0
        assert result['features'][0]['text'] == 'clamp connections'

    def test_ties_break_on_text(self) -> None:
        result = score_text(['ab cd', 'zz'])
        assert [(r['text'], r['count']) for r in result['features']] == [
            ('ab cd', 1), ('zz', 1)]

    def test_accented_and_plain_spellings_count_together(self) -> None:
        result = score_text(['présent', 'present', 'absent'])
        rows = {r['text']: r for r in result['features']}
        assert rows['present']['count'] == 2

    def test_top_n_caps_features(self) -> None:
        result = score_text(['ab', 'cd', 'ef'], top_n=2)
        assert len(result['features']) == 2

    def test_no_results(self) -> None:
        assert score_text([]) == {'n_results': 0, 'features': []}
