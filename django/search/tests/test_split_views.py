"""Tests for the result-splitting endpoints (plan F2).

docs/plans/result-splitting-features.md: Structured Splits
(POST /api/features/structured/) and Text Splits
(POST /api/features/text/).  The scoring is F1's
(treatments_classifier/result_splits.py), used for real here; only the
experiment lookup and the CouchDB fetch are patched.

Run with: pytest search/tests/test_split_views.py -v
"""
import json
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from django.conf import settings
from django.test import TestCase
from django.urls import reverse

# The views put the repo root on the path when they first need it.
if settings.SKOL_ROOT_PATH not in sys.path:
    sys.path.insert(0, settings.SKOL_ROOT_PATH)
from treatments_to_structured.vocab_canonical import (  # noqa: E402
    source_hash)

PENDING = pytest.mark.xfail(strict=True, reason='F2 not implemented yet')

PIN = {'synonyms': 'a' * 40, 'spellings': 'b' * 40, 'rules': '1'}
EXP = {
    'databases': {'treatments_structured': 'struct_db',
                  'treatments_prose': 'prose_db'},
    'vocab_canonical': PIN,
}


def structured(doc_id: str, annotated: Any,
               canonical: Optional[Dict[str, Any]] = None,
               version: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """A structured-treatments document; with ``canonical``, its
    json_canonical, fresh against ``version`` (default PIN)."""
    doc: Dict[str, Any] = {'_id': doc_id, 'json_annotated': annotated}
    if canonical is not None:
        parsed = (json.loads(annotated) if isinstance(annotated, str)
                  else annotated)
        doc.update({'json_canonical': canonical,
                    'canonical_map_version': dict(version or PIN),
                    'canonical_source': source_hash(parsed)})
    return doc


def prose(doc_id: str, description: str) -> Dict[str, Any]:
    return {'_id': doc_id, 'description': description}


def find(rows: List[Dict[str, Any]], path: List[str]) -> Dict[str, Any]:
    matches = [r for r in rows if r['path'] == path]
    assert len(matches) == 1, f'{path} in {[r["path"] for r in rows]}'
    return matches[0]


class SplitViewCase(TestCase):
    """Posts with the experiment and the fetch patched."""

    url_name = ''

    def post(self, body: Any, docs: Optional[List[Dict[str, Any]]] = None,
             exp: Optional[Dict[str, Any]] = EXP) -> Any:
        self.fetch = MagicMock(return_value=list(docs or []))
        with patch('search.views.get_user_experiment',
                   return_value=('production', exp)), \
                patch('search.views._fetch_result_docs', self.fetch):
            return self.client.post(reverse(self.url_name),
                                    data=json.dumps(body),
                                    content_type='application/json')


class ValidationMixin:
    """The same request checks for both endpoints."""

    def test_get_not_allowed(self) -> None:
        assert self.client.get(reverse(self.url_name)).status_code == 405

    @PENDING
    def test_treatment_ids_required(self) -> None:
        for body in ({}, {'treatment_ids': []},
                     {'treatment_ids': 'id1'}, {'treatment_ids': None}):
            response = self.post(body)
            assert response.status_code == 400, body
            assert 'treatment_ids' in response.json()['error']

    @PENDING
    def test_top_n_must_be_a_positive_integer(self) -> None:
        for top_n in (0, -1, 'ten', 2.5, True):
            response = self.post({'treatment_ids': ['a'], 'top_n': top_n})
            assert response.status_code == 400, top_n
            assert 'top_n' in response.json()['error']

    @PENDING
    def test_a_failure_is_a_500_with_its_error(self) -> None:
        with patch('search.views.get_user_experiment',
                   return_value=('production', EXP)), \
                patch('search.views._fetch_result_docs',
                      side_effect=RuntimeError('couchdb down')):
            response = self.client.post(
                reverse(self.url_name),
                data=json.dumps({'treatment_ids': ['a']}),
                content_type='application/json')
        assert response.status_code == 500
        assert 'couchdb down' in response.json()['error']


class TestStructuredSplits(ValidationMixin, SplitViewCase):
    url_name = 'search:features-structured'

    @PENDING
    def test_reads_the_experiments_structured_database(self) -> None:
        self.post({'treatment_ids': ['a', 'b']})
        self.fetch.assert_called_once_with('struct_db', ['a', 'b'])

    @PENDING
    def test_scores_the_results(self) -> None:
        docs = [structured(f'd{i}', {'Pileus': {'Color': [c]}},
                           canonical={'Pileus': {'Color': [c]}})
                for i, c in enumerate(['Brown', 'Brown', 'Yellow', 'Yellow'])]
        data = self.post({'treatment_ids': [d['_id'] for d in docs]},
                         docs).json()
        assert data['n_results'] == 4
        colour = find(find(data['parts'], ['Pileus'])['attributes'],
                      ['Pileus', 'Color'])
        assert colour['score'] == pytest.approx(0.5)
        assert [(v['value'], v['count']) for v in colour['values']] == [
            ('Brown', 2), ('Yellow', 2)]

    @PENDING
    def test_reports_the_experiments_pin(self) -> None:
        data = self.post({'treatment_ids': ['a']},
                         [structured('a', {'P': {'C': ['x']}})]).json()
        assert data['canonical_version'] == PIN

    @PENDING
    def test_uses_json_canonical_when_fresh(self) -> None:
        doc = structured('a', {'cap': {'colour': ['brown']}},
                         canonical={'Pileus': {'Color': ['brown']}})
        data = self.post({'treatment_ids': ['a']}, [doc]).json()
        assert [p['path'] for p in data['parts']] == [['Pileus']]
        assert data['n_fallback'] == 0

    @PENDING
    def test_falls_back_to_json_annotated_when_missing_or_stale(
            self) -> None:
        missing = structured('a', {'cap': {'colour': ['brown']}})
        stale = structured('b', {'cap': {'colour': ['white']}},
                           canonical={'Pileus': {'Color': ['white']}},
                           version={**PIN, 'rules': '0'})
        data = self.post({'treatment_ids': ['a', 'b']},
                         [missing, stale]).json()
        assert [p['path'] for p in data['parts']] == [['cap']]
        assert data['n_fallback'] == 2

    @PENDING
    def test_json_annotated_stored_as_a_string(self) -> None:
        doc = structured('a', json.dumps({'cap': {'colour': ['brown']}}))
        data = self.post({'treatment_ids': ['a']}, [doc]).json()
        assert [p['path'] for p in data['parts']] == [['cap']]

    @PENDING
    def test_without_an_experiment(self) -> None:
        """No pin to check freshness against: every document falls
        back, and the default database is read."""
        doc = structured('a', {'cap': {'colour': ['brown']}},
                         canonical={'Pileus': {'Color': ['brown']}})
        with self.settings(VOCAB_TREE_DB='default_struct_db'):
            data = self.post({'treatment_ids': ['a']}, [doc],
                             exp=None).json()
        self.fetch.assert_called_once_with('default_struct_db', ['a'])
        assert data['canonical_version'] is None
        assert data['n_fallback'] == 1
        assert [p['path'] for p in data['parts']] == [['cap']]

    @PENDING
    def test_ids_not_found_are_counted_not_scored(self) -> None:
        data = self.post({'treatment_ids': ['a', 'gone']},
                         [structured('a', {'P': {'C': ['x']}})]).json()
        assert data['n_results'] == 1
        assert data['n_missing'] == 1

    @PENDING
    def test_a_result_without_a_structure_still_counts(self) -> None:
        docs = [structured('a', {'P': {'C': ['x']}}), {'_id': 'b'}]
        data = self.post({'treatment_ids': ['a', 'b']}, docs).json()
        assert data['n_results'] == 2
        assert find(data['parts'], ['P'])['described'] == 1

    @PENDING
    def test_top_n_caps_attributes(self) -> None:
        docs = [structured(f'd{i}', {'P': {'A': [a], 'B': [b], 'C': [c]}})
                for i, (a, b, c) in enumerate(
                    [('x', 'y', 'z'), ('x', 'w', 'z'), ('v', 'w', 'u')])]
        data = self.post({'treatment_ids': ['d0', 'd1', 'd2'], 'top_n': 1},
                         docs).json()
        assert sum(len(p['attributes']) for p in data['parts']) == 1


class TestTextSplits(ValidationMixin, SplitViewCase):
    url_name = 'search:features-text'

    @PENDING
    def test_reads_the_experiments_prose_database(self) -> None:
        self.post({'treatment_ids': ['a', 'b']})
        self.fetch.assert_called_once_with('prose_db', ['a', 'b'])

    @PENDING
    def test_scores_the_descriptions(self) -> None:
        docs = [prose('a', 'spores clamp connections'),
                prose('b', 'clamp connections and spores'),
                prose('c', 'spores'), prose('d', 'spores')]
        data = self.post({'treatment_ids': ['a', 'b', 'c', 'd']},
                         docs).json()
        assert data['n_results'] == 4
        assert data['features'][0] == {
            'text': 'clamp connections', 'count': 2, 'score': 0.5}

    @PENDING
    def test_ids_not_found_are_counted_not_scored(self) -> None:
        data = self.post({'treatment_ids': ['a', 'gone']},
                         [prose('a', 'spores')]).json()
        assert data['n_results'] == 1
        assert data['n_missing'] == 1

    @PENDING
    def test_a_result_without_a_description_still_counts(self) -> None:
        data = self.post({'treatment_ids': ['a', 'b']},
                         [prose('a', 'spores'), {'_id': 'b'}]).json()
        assert data['n_results'] == 2
        assert data['features'][0]['count'] == 1

    @PENDING
    def test_top_n_caps_features(self) -> None:
        docs = [prose('a', 'ab'), prose('b', 'cd'), prose('c', 'ef')]
        data = self.post({'treatment_ids': ['a', 'b', 'c'], 'top_n': 2},
                         docs).json()
        assert len(data['features']) == 2


class TestFetchResultDocs(TestCase):
    """One ``_all_docs`` request; documents in the order asked; ids
    CouchDB does not have are left out."""

    @PENDING
    def test_bulk_fetch(self) -> None:
        from search.views import _fetch_result_docs
        reply = MagicMock(status_code=200)
        reply.json.return_value = {'rows': [
            {'key': 'b', 'doc': {'_id': 'b'}},
            {'key': 'gone', 'error': 'not_found'},
            {'key': 'old', 'value': {'deleted': True}, 'doc': None},
            {'key': 'a', 'doc': {'_id': 'a'}},
        ]}
        with patch('search.views.requests.post',
                   return_value=reply) as post, \
                self.settings(COUCHDB_URL='http://couch:5984'):
            docs = _fetch_result_docs('struct_db',
                                      ['b', 'gone', 'old', 'a'])
        assert [d['_id'] for d in docs] == ['b', 'a']
        args, kwargs = post.call_args
        assert args[0] == ('http://couch:5984/struct_db/_all_docs'
                           '?include_docs=true')
        assert kwargs['json'] == {'keys': ['b', 'gone', 'old', 'a']}


class TestRouting(TestCase):
    def test_urls(self) -> None:
        assert reverse('search:features-structured') == (
            '/api/features/structured/')
        assert reverse('search:features-text') == '/api/features/text/'
