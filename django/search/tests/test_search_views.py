"""
Tests for SearchView nomenclature_pattern pre-filter.

Run with: python manage.py test search.tests.test_search_views
"""
import json
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from django.test import TestCase, Client
from django.urls import reverse


def _make_mock_embeddings():
    """Build a small mock embeddings DataFrame with 3 taxa."""
    data = {
        'taxon': [
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
