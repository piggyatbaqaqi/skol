"""
Tests for SearchView nomenclature_pattern pre-filter and project_slugs filter.

Run with: python manage.py test search.tests.test_search_views
"""
import json
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from django.contrib.auth.models import User
from django.test import TestCase, Client
from django.urls import reverse

from search.models import Collection, CollectionProject, Project


def _make_mock_embeddings():
    """Build a small mock embeddings DataFrame with 3 taxa."""
    data = {
        'treatment': [
            'Agaricus bisporus',
            'Agaricus campestris',
            'Boletus edulis',
        ],
        'description': [
            'White cap, pink gills',
            'Field mushroom description',
            'Brown cap with pores',
        ],
        'source': ['SKOL_TAXA'] * 3,
        'filename': ['f1', 'f2', 'f3'],
    }
    # Add 4 fake embedding columns
    for i in range(4):
        data[f'F{i}'] = np.random.rand(3)
    return pd.DataFrame(data)


def _make_collection_embeddings(collection_ids):
    """Build mock embeddings with collection-type taxon_ids."""
    n = len(collection_ids)
    data = {
        'treatment': [f'Collection taxon {cid}' for cid in collection_ids],
        'description': [f'Description for {cid}' for cid in collection_ids],
        'source': ['SKOL_COLLECTIONS'] * n,
        'filename': [f'f{cid}' for cid in collection_ids],
        'taxon_id': [f'collection_{cid}' for cid in collection_ids],
    }
    for i in range(4):
        data[f'F{i}'] = np.random.rand(n)
    return pd.DataFrame(data)


def _make_mock_nearest(embeddings):
    """Build a mock nearest_neighbors DataFrame sorted by fake similarity."""
    similarities = [0.9, 0.8, 0.7][:len(embeddings)]
    return pd.DataFrame(
        {'similarity': similarities},
        index=embeddings.index,
    )


class TestSearchViewNomenclaturePattern(TestCase):
    """Tests for the nomenclature_pattern parameter on POST /api/search/."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('search:search')

    def test_invalid_regex_returns_400(self):
        """Invalid nomenclature_pattern regex returns 400."""
        response = self.client.post(
            self.url,
            data=json.dumps({
                'prompt': 'test',
                'embedding_name': 'skol:embedding:v1.1',
                'nomenclature_pattern': '[invalid',
            }),
            content_type='application/json',
        )
        assert response.status_code == 400
        assert 'Invalid nomenclature_pattern regex' in response.json()['error']

    @patch('search.views.settings')
    def test_pattern_matching_nothing_returns_empty(self, mock_settings):
        """nomenclature_pattern that matches nothing returns empty results."""
        mock_settings.REDIS_URL = 'redis://localhost:6379'
        mock_embeddings = _make_mock_embeddings()

        with patch(
            'dr_drafts_mycosearch.sota_search.read_narrative_embeddings_from_redis',
            return_value=mock_embeddings,
        ):
            response = self.client.post(
                self.url,
                data=json.dumps({
                    'prompt': 'test',
                    'embedding_name': 'skol:embedding:v1.1',
                    'nomenclature_pattern': '^Zzzznothing',
                }),
                content_type='application/json',
            )
        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 0
        assert data['results'] == []
        assert data['nomenclature_pattern'] == '^Zzzznothing'

    @patch('search.views.settings')
    def test_pattern_filters_results(self, mock_settings):
        """nomenclature_pattern restricts results to matching taxa."""
        mock_settings.REDIS_URL = 'redis://localhost:6379'
        mock_embeddings = _make_mock_embeddings()

        # Only Agaricus rows (indices 0, 1) should pass the filter
        filtered = mock_embeddings.iloc[:2]
        mock_nearest = _make_mock_nearest(filtered)

        with patch(
            'dr_drafts_mycosearch.sota_search.read_narrative_embeddings_from_redis',
            return_value=mock_embeddings,
        ), patch(
            'dr_drafts_mycosearch.sota_search.sort_by_similarity_to_prompt',
            return_value=mock_nearest,
        ) as mock_sort:
            response = self.client.post(
                self.url,
                data=json.dumps({
                    'prompt': 'edible mushroom',
                    'embedding_name': 'skol:embedding:v1.1',
                    'k': 10,
                    'nomenclature_pattern': '^Agaricus',
                }),
                content_type='application/json',
            )

        assert response.status_code == 200
        data = response.json()
        # Should only have 2 Agaricus results, not the Boletus
        assert data['count'] == 2
        titles = [r['Title'] for r in data['results']]
        assert all('Agaricus' in t for t in titles)
        assert not any('Boletus' in t for t in titles)

        # Verify sort_by_similarity_to_prompt was called with filtered DF
        called_df = mock_sort.call_args[0][1]
        assert len(called_df) == 2

    def test_without_pattern_does_not_include_field(self):
        """Response without nomenclature_pattern omits the field."""
        # This test just checks validation — the actual search would need
        # ML dependencies, so we only go as far as the missing embedding error
        response = self.client.post(
            self.url,
            data=json.dumps({
                'prompt': 'test',
                'embedding_name': 'nonexistent:embedding',
            }),
            content_type='application/json',
        )
        # Will fail with embedding not found — but the response should
        # NOT contain nomenclature_pattern key
        data = response.json()
        assert 'nomenclature_pattern' not in data


class TestSearchViewProjectFilter(TestCase):
    """Tests for the project_slugs post-filter on POST /api/search/."""

    def setUp(self) -> None:
        self.client = Client()
        self.url = reverse('search:search')
        self.user = User.objects.create_user('jsmith', 'j@example.com', 'pw')
        self.project = Project.objects.create(name='Field Guide', creator=self.user)
        # Two collections: coll_a in the project, coll_b not
        self.coll_a = Collection.objects.create(owner=self.user, name='Coll A')
        self.coll_b = Collection.objects.create(owner=self.user, name='Coll B')
        CollectionProject.objects.create(
            collection=self.coll_a, project=self.project, added_by=self.user
        )

    def _post_search(self, project_slugs):
        """Helper: POST /api/search/ with mocked Experiment and project_slugs."""
        coll_ids = [self.coll_a.collection_id, self.coll_b.collection_id]
        mock_embeddings = _make_collection_embeddings(coll_ids)
        mock_nearest = _make_mock_nearest(mock_embeddings)

        mock_exp = MagicMock()
        mock_exp.nearest_neighbors = mock_nearest
        mock_exp.embeddings = mock_embeddings

        with patch('search.views.settings') as mock_settings, \
             patch('dr_drafts_mycosearch.sota_search.Experiment', return_value=mock_exp) as _mock_cls:
            mock_settings.REDIS_URL = 'redis://localhost:6379'
            body = {
                'prompt': 'brown cap',
                'embedding_name': 'skol:embedding:v1.1',
                'k': 10,
            }
            if project_slugs is not None:
                body['project_slugs'] = project_slugs
            return self.client.post(
                self.url,
                data=json.dumps(body),
                content_type='application/json',
            )

    def test_no_project_filter_returns_both_collections(self) -> None:
        """Without project_slugs, all collection results are returned."""
        response = self._post_search(None)
        assert response.status_code == 200
        data = response.json()
        collection_ids = [r['CollectionId'] for r in data['results'] if 'CollectionId' in r]
        assert self.coll_a.collection_id in collection_ids
        assert self.coll_b.collection_id in collection_ids

    def test_project_filter_keeps_member_collection(self) -> None:
        """project_slugs retains collections that are in the project."""
        slug = f'jsmith/{self.project.slug}'
        response = self._post_search([slug])
        assert response.status_code == 200
        data = response.json()
        collection_ids = [r['CollectionId'] for r in data['results'] if 'CollectionId' in r]
        assert self.coll_a.collection_id in collection_ids

    def test_project_filter_removes_nonmember_collection(self) -> None:
        """project_slugs drops collection results not in any selected project."""
        slug = f'jsmith/{self.project.slug}'
        response = self._post_search([slug])
        assert response.status_code == 200
        data = response.json()
        collection_ids = [r['CollectionId'] for r in data['results'] if 'CollectionId' in r]
        assert self.coll_b.collection_id not in collection_ids

    def test_unknown_project_slug_drops_all_collection_results(self) -> None:
        """An unrecognised slug resolves to zero collection IDs — all filtered out."""
        response = self._post_search(['jsmith/nonexistent'])
        assert response.status_code == 200
        data = response.json()
        collection_ids = [r['CollectionId'] for r in data['results'] if 'CollectionId' in r]
        assert not collection_ids

    def test_empty_project_slugs_list_applies_no_filter(self) -> None:
        """An empty project_slugs list is treated as 'no filter'."""
        response = self._post_search([])
        assert response.status_code == 200
        data = response.json()
        collection_ids = [r['CollectionId'] for r in data['results'] if 'CollectionId' in r]
        assert self.coll_a.collection_id in collection_ids
        assert self.coll_b.collection_id in collection_ids
