"""
Tests for SourceContextView.

Covers:
- Valid field names (all 10 treatment section labels)
- Invalid field name returns 400
- <mark> tags carry section CSS class
- Missing spans returns 404
- Missing ingest._id returns 400

Run with: pytest search/tests/test_source_context_view.py -v
"""
import json
from unittest.mock import patch, MagicMock

import requests as req
from django.test import TestCase
from django.urls import reverse
from rest_framework import status


# All section-level field names accepted by SourceContextView.
_VALID_FIELDS = [
    'nomenclature',
    'description',
    'diagnosis',
    'etymology',
    'distribution',
    'materials_examined',
    'type_designation',
    'biology',
    'notes',
    'figure_caption',
]

# A minimal span dict (already parsed form).
_SPAN = {
    'paragraph_number': 1,
    'start_line': 0,
    'end_line': 0,
    'start_char': 10,
    'end_char': 30,
    'pdf_page': '5',
    'pdf_label': '5',
    'empirical_page': '127',
}

# Fake plaintext with a highlight-able region.
_ARTICLE_TEXT = 'A' * 10 + 'TARGET_TEXT_HERE____' + 'Z' * 50


def _make_mock_settings():
    mock = MagicMock()
    mock.COUCHDB_URL = 'http://localhost:5984'
    mock.COUCHDB_USERNAME = 'admin'
    mock.COUCHDB_PASSWORD = 'password'
    return mock


def _make_taxa_doc(field: str) -> dict:
    """Build a minimal taxa document with one span for *field*."""
    return {
        '_id': 'taxon_abc123',
        'ingest': {
            '_id': 'ingest_doc_001',
            'db_name': 'skol_dev',
            'url': 'http://example.com/article',
        },
        'attachment_name': 'article.txt.ann',
        f'{field}_spans': [_SPAN],
    }


class TestSourceContextViewValidFields(TestCase):
    """SourceContextView accepts every treatment section field name."""

    def _get_context(self, field: str):
        url = reverse('search:taxa-context', kwargs={'taxa_id': 'taxon_abc123'})
        taxa_doc = _make_taxa_doc(field)
        mock_settings = _make_mock_settings()

        # Build response mocks.
        taxa_resp = MagicMock()
        taxa_resp.status_code = 200
        taxa_resp.json.return_value = taxa_doc
        taxa_resp.raise_for_status = MagicMock()

        ann_resp = MagicMock()
        ann_resp.status_code = 200
        ann_resp.text = _ARTICLE_TEXT
        ann_resp.raise_for_status = MagicMock()

        def _mock_get(url_arg, **kwargs):
            if 'article.txt.ann' in url_arg or 'article.pdf.ann' in url_arg:
                return ann_resp
            return taxa_resp

        with patch('search.views.settings', mock_settings), \
                patch('search.views.requests.get', side_effect=_mock_get), \
                patch('search.views.get_user_experiment', return_value=(None, None)):
            return self.client.get(f'{url}?field={field}&taxa_db=skol_dev')

    def test_all_valid_fields_return_200(self):
        """Every section field name is accepted (not 400)."""
        for field in _VALID_FIELDS:
            with self.subTest(field=field):
                response = self._get_context(field)
                self.assertNotEqual(
                    response.status_code,
                    status.HTTP_400_BAD_REQUEST,
                    f'Field "{field}" was rejected as invalid',
                )
                self.assertEqual(
                    response.status_code,
                    status.HTTP_200_OK,
                    f'Field "{field}" returned {response.status_code}',
                )

    def test_invalid_field_returns_400(self):
        """Unrecognised field name returns 400."""
        url = reverse('search:taxa-context', kwargs={'taxa_id': 'taxon_abc123'})
        with patch('search.views.settings', _make_mock_settings()), \
                patch('search.views.get_user_experiment', return_value=(None, None)):
            response = self.client.get(f'{url}?field=banana&taxa_db=skol_dev')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('Invalid field', response.json()['error'])

    def test_response_contains_source_text(self):
        """Successful response has source_text key."""
        response = self._get_context('description')
        data = response.json()
        self.assertIn('source_text', data)

    def test_response_contains_total_spans(self):
        """Successful response includes total_spans count."""
        response = self._get_context('description')
        data = response.json()
        self.assertIn('total_spans', data)
        self.assertEqual(data['total_spans'], 1)


class TestSourceContextViewMarkCssClass(TestCase):
    """<mark> tags carry a section-specific CSS class."""

    def _get_highlighted(self, field: str) -> str:
        url = reverse('search:taxa-context', kwargs={'taxa_id': 'taxon_abc123'})
        taxa_doc = _make_taxa_doc(field)
        mock_settings = _make_mock_settings()

        taxa_resp = MagicMock()
        taxa_resp.status_code = 200
        taxa_resp.json.return_value = taxa_doc
        taxa_resp.raise_for_status = MagicMock()

        ann_resp = MagicMock()
        ann_resp.status_code = 200
        ann_resp.text = _ARTICLE_TEXT
        ann_resp.raise_for_status = MagicMock()

        def _mock_get(url_arg, **kwargs):
            if 'article.txt.ann' in url_arg or 'article.pdf.ann' in url_arg:
                return ann_resp
            return taxa_resp

        with patch('search.views.settings', mock_settings), \
                patch('search.views.requests.get', side_effect=_mock_get), \
                patch('search.views.get_user_experiment', return_value=(None, None)):
            response = self.client.get(f'{url}?field={field}&taxa_db=skol_dev')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        return response.json()['source_text']

    def test_description_mark_has_css_class(self):
        """description field produces <mark class="section-description"> tags."""
        highlighted = self._get_highlighted('description')
        self.assertIn('class="section-description"', highlighted)
        self.assertNotIn('<mark>', highlighted)

    def test_nomenclature_mark_has_css_class(self):
        """nomenclature field produces <mark class="section-nomenclature"> tags."""
        highlighted = self._get_highlighted('nomenclature')
        self.assertIn('class="section-nomenclature"', highlighted)

    def test_diagnosis_mark_has_css_class(self):
        """diagnosis field produces <mark class="section-diagnosis"> tags."""
        highlighted = self._get_highlighted('diagnosis')
        self.assertIn('class="section-diagnosis"', highlighted)

    def test_distribution_mark_has_css_class(self):
        """distribution field produces <mark class="section-distribution"> tags."""
        highlighted = self._get_highlighted('distribution')
        self.assertIn('class="section-distribution"', highlighted)


class TestSourceContextViewEdgeCases(TestCase):
    """Edge-case handling in SourceContextView."""

    def test_missing_spans_returns_404(self):
        """Returns 404 when the taxa doc has no spans for the requested field."""
        url = reverse('search:taxa-context', kwargs={'taxa_id': 'taxon_abc123'})
        taxa_doc = {
            '_id': 'taxon_abc123',
            'ingest': {'_id': 'ingest_doc_001', 'db_name': 'skol_dev'},
            # No description_spans key at all.
        }
        mock_settings = _make_mock_settings()

        taxa_resp = MagicMock()
        taxa_resp.status_code = 200
        taxa_resp.json.return_value = taxa_doc
        taxa_resp.raise_for_status = MagicMock()

        with patch('search.views.settings', mock_settings), \
                patch('search.views.requests.get', return_value=taxa_resp), \
                patch('search.views.get_user_experiment', return_value=(None, None)):
            response = self.client.get(f'{url}?field=description&taxa_db=skol_dev')

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_missing_ingest_id_returns_400(self):
        """Returns 400 when taxa doc lacks ingest._id."""
        url = reverse('search:taxa-context', kwargs={'taxa_id': 'taxon_abc123'})
        taxa_doc = {
            '_id': 'taxon_abc123',
            'ingest': {'db_name': 'skol_dev'},  # No _id
            'description_spans': [_SPAN],
        }
        mock_settings = _make_mock_settings()

        taxa_resp = MagicMock()
        taxa_resp.status_code = 200
        taxa_resp.json.return_value = taxa_doc
        taxa_resp.raise_for_status = MagicMock()

        with patch('search.views.settings', mock_settings), \
                patch('search.views.requests.get', return_value=taxa_resp), \
                patch('search.views.get_user_experiment', return_value=(None, None)):
            response = self.client.get(f'{url}?field=description&taxa_db=skol_dev')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
