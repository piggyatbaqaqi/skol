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
        url = reverse('search:treatments-context', kwargs={'treatment_id': 'taxon_abc123'})
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
            return self.client.get(f'{url}?field={field}&treatments_db=skol_dev')

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
        url = reverse('search:treatments-context', kwargs={'treatment_id': 'taxon_abc123'})
        with patch('search.views.settings', _make_mock_settings()), \
                patch('search.views.get_user_experiment', return_value=(None, None)):
            response = self.client.get(f'{url}?field=banana&treatments_db=skol_dev')
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
        url = reverse('search:treatments-context', kwargs={'treatment_id': 'taxon_abc123'})
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
            response = self.client.get(f'{url}?field={field}&treatments_db=skol_dev')

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
        url = reverse('search:treatments-context', kwargs={'treatment_id': 'taxon_abc123'})
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
            response = self.client.get(f'{url}?field=description&treatments_db=skol_dev')

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_missing_ingest_id_returns_400(self):
        """Returns 400 when taxa doc lacks ingest._id."""
        url = reverse('search:treatments-context', kwargs={'treatment_id': 'taxon_abc123'})
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
            response = self.client.get(f'{url}?field=description&treatments_db=skol_dev')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


# ---------------------------------------------------------------------------
# _collect_ann_db_candidates: priority-ordered list of DBs to probe for .ann
# ---------------------------------------------------------------------------


from search.views import _collect_ann_db_candidates  # noqa: E402


class TestCollectAnnDbCandidates(TestCase):
    """Priority order: explicit doc.annotations_db → ingest_db → experiment's
    databases.annotations.  Duplicates removed while preserving first-seen
    order."""

    def _mock_experiment_lookup(self, annotations_db=None):
        """Return a requests.post mock that fakes skol_experiments/_find."""
        body = {'docs': []}
        if annotations_db:
            body = {'docs': [{'databases': {'annotations': annotations_db}}]}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = body
        return MagicMock(return_value=resp)

    def test_explicit_annotations_db_on_doc_comes_first(self):
        taxa_doc = {'annotations_db': 'doc_ann_db'}
        with patch('search.views.requests.post',
                   self._mock_experiment_lookup('exp_ann_db')):
            result = _collect_ann_db_candidates(
                taxa_doc=taxa_doc,
                ingest_db='skol_dev',
                treatments_db='skol_treatments_dev',
                auth=None,
                couchdb_url='http://x',
            )
        self.assertEqual(result, ['doc_ann_db', 'skol_dev', 'exp_ann_db'])

    def test_no_explicit_then_ingest_db_first(self):
        with patch('search.views.requests.post',
                   self._mock_experiment_lookup('exp_ann_db')):
            result = _collect_ann_db_candidates(
                taxa_doc={},
                ingest_db='skol_dev',
                treatments_db='skol_treatments_dev',
                auth=None,
                couchdb_url='http://x',
            )
        self.assertEqual(result, ['skol_dev', 'exp_ann_db'])

    def test_no_experiment_match_returns_only_ingest(self):
        """When skol_experiments has no matching doc, candidate list is
        just the ingest DB."""
        with patch('search.views.requests.post',
                   self._mock_experiment_lookup(None)):
            result = _collect_ann_db_candidates(
                taxa_doc={},
                ingest_db='skol_dev',
                treatments_db='skol_treatments_dev',
                auth=None,
                couchdb_url='http://x',
            )
        self.assertEqual(result, ['skol_dev'])

    def test_duplicates_collapsed_preserving_order(self):
        """If experiment.databases.annotations happens to be the same DB as
        ingest_db (legacy single-DB setups), we don't probe it twice."""
        with patch('search.views.requests.post',
                   self._mock_experiment_lookup('skol_dev')):
            result = _collect_ann_db_candidates(
                taxa_doc={'annotations_db': 'skol_dev'},
                ingest_db='skol_dev',
                treatments_db='skol_treatments_dev',
                auth=None,
                couchdb_url='http://x',
            )
        self.assertEqual(result, ['skol_dev'])

    def test_experiment_lookup_failure_is_silent(self):
        """A transient CouchDB error during the experiment lookup must
        not break the .ann lookup — we just fall back to the legacy
        candidate list."""
        def boom(*a, **kw):
            raise req.ConnectionError('couchdb down')
        with patch('search.views.requests.post', side_effect=boom):
            result = _collect_ann_db_candidates(
                taxa_doc={},
                ingest_db='skol_dev',
                treatments_db='skol_treatments_dev',
                auth=None,
                couchdb_url='http://x',
            )
        self.assertEqual(result, ['skol_dev'])


# ---------------------------------------------------------------------------
# Integration: SourceContextView falls back to experiment's annotations DB
# ---------------------------------------------------------------------------


class TestSourceContextViewAnnDbFallback(TestCase):
    """When the .ann attachment isn't in the ingest DB, the view must
    consult the experiment's databases.annotations and try there."""

    def test_falls_back_to_experiment_annotations_db(self):
        url = reverse('search:treatments-context',
                      kwargs={'treatment_id': 'taxon_abc123'})
        taxa_doc = _make_taxa_doc('description')
        # No annotations_db field on the doc — must come from the experiment.

        # The taxa doc fetch succeeds; the first ann fetch (against
        # ingest_db=skol_dev) 404s; the second (against exp_ann_db) 200s.
        taxa_resp = MagicMock(status_code=200,
                              json=MagicMock(return_value=taxa_doc),
                              raise_for_status=MagicMock())
        ann_404 = MagicMock(status_code=404,
                            raise_for_status=MagicMock())
        ann_ok = MagicMock(status_code=200, text=_ARTICLE_TEXT,
                           raise_for_status=MagicMock())

        # GET calls in order: taxa doc, ann from skol_dev (404),
        # ann from exp_ann_db (200).  Use a URL-aware dispatcher.
        def _mock_get(url_arg, **kwargs):
            if '/skol_dev/' in url_arg and ('.ann' in url_arg):
                return ann_404
            if '/exp_ann_db/' in url_arg and ('.ann' in url_arg):
                return ann_ok
            return taxa_resp

        # POST to skol_experiments/_find returns the experiment with
        # databases.annotations = exp_ann_db.
        find_resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                'docs': [{'databases': {'annotations': 'exp_ann_db'}}],
            }),
        )

        with patch('search.views.settings', _make_mock_settings()), \
                patch('search.views.requests.get', side_effect=_mock_get), \
                patch('search.views.requests.post', return_value=find_resp), \
                patch('search.views.get_user_experiment',
                      return_value=(None, None)):
            response = self.client.get(
                f'{url}?field=description&treatments_db=skol_treatments_dev'
            )
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_error_message_lists_all_probed_dbs(self):
        """When every candidate DB 404s, the error names them all so
        the operator can see exactly where we looked."""
        url = reverse('search:treatments-context',
                      kwargs={'treatment_id': 'taxon_abc123'})
        taxa_doc = _make_taxa_doc('description')

        taxa_resp = MagicMock(status_code=200,
                              json=MagicMock(return_value=taxa_doc),
                              raise_for_status=MagicMock())
        ann_404 = MagicMock(status_code=404,
                            raise_for_status=MagicMock())

        def _mock_get(url_arg, **kwargs):
            if '.ann' in url_arg:
                return ann_404
            return taxa_resp

        find_resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                'docs': [{'databases': {'annotations': 'exp_ann_db'}}],
            }),
        )

        with patch('search.views.settings', _make_mock_settings()), \
                patch('search.views.requests.get', side_effect=_mock_get), \
                patch('search.views.requests.post', return_value=find_resp), \
                patch('search.views.get_user_experiment',
                      return_value=(None, None)):
            response = self.client.get(
                f'{url}?field=description&treatments_db=skol_treatments_dev'
            )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        err = response.json()['error']
        # Both ingest_db and exp_ann_db must be mentioned in the error.
        self.assertIn('skol_dev', err)
        self.assertIn('exp_ann_db', err)
