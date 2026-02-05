"""
Tests for Classifier API views (TextClassifierView, JsonClassifierView).

Run with: pytest search/tests/test_feature_views.py -v

Tests are organized into:
- Unit tests: Validation, HTTP methods, parameter handling
- Functional tests: Full request/response cycle with mocked classifiers
"""
import json
from unittest.mock import patch, MagicMock, call

from django.test import TestCase, Client
from django.urls import reverse


MOCK_STATS = {
    'n_samples': 10,
    'n_classes': 5,
    'n_features': 100,
    'tree_depth': 4,
    'tree_n_leaves': 8,
    'train_accuracy': 1.0,
}

MOCK_IMPORTANCES = [
    ('brown', 0.25),
    ('spores', 0.18),
    ('cap', 0.12),
]

MOCK_TREE_JSON = {
    'metadata': {'n_classes': 5, 'max_depth': 4},
    'tree': {'feature': 'brown', 'threshold': 0.5},
}


def _mock_settings():
    """Create a mock settings object with CouchDB configuration."""
    mock = MagicMock()
    mock.SKOL_ROOT_PATH = '/fake/path'
    mock.COUCHDB_URL = 'http://localhost:5984'
    mock.COUCHDB_USERNAME = 'admin'
    mock.COUCHDB_PASSWORD = 'password'
    return mock


def _mock_text_classifier(stats=None, importances=None, tree_json=None):
    """Create a mock TaxaDecisionTreeClassifier."""
    mock = MagicMock()
    mock.fit.return_value = stats or MOCK_STATS
    mock.get_feature_importances.return_value = importances or MOCK_IMPORTANCES
    mock.tree_to_json.return_value = tree_json or MOCK_TREE_JSON
    return mock


def _mock_json_classifier(stats=None, importances=None, tree_json=None):
    """Create a mock TaxaJsonClassifier."""
    mock = MagicMock()
    mock.fit.return_value = stats or MOCK_STATS
    mock.get_feature_importances.return_value = importances or MOCK_IMPORTANCES
    mock.tree_to_json.return_value = tree_json or MOCK_TREE_JSON
    return mock


class TestTextClassifierViewValidation(TestCase):
    """Unit tests for request validation on POST /api/classifier/text/."""

    def setUp(self) -> None:
        self.client = Client()
        self.url = reverse('search:classifier-text')

    def test_get_method_not_allowed(self) -> None:
        """GET requests should return 405 Method Not Allowed."""
        response = self.client.get(self.url)
        assert response.status_code == 405

    def test_put_method_not_allowed(self) -> None:
        """PUT requests should return 405 Method Not Allowed."""
        response = self.client.put(
            self.url,
            data=json.dumps({'taxa_ids': ['id1']}),
            content_type='application/json',
        )
        assert response.status_code == 405

    def test_delete_method_not_allowed(self) -> None:
        """DELETE requests should return 405 Method Not Allowed."""
        response = self.client.delete(self.url)
        assert response.status_code == 405

    def test_missing_taxa_ids_returns_400(self) -> None:
        """Empty body should return 400 with error message."""
        response = self.client.post(
            self.url,
            data=json.dumps({}),
            content_type='application/json',
        )
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
        assert 'taxa_ids' in data['error']

    def test_empty_taxa_ids_returns_400(self) -> None:
        """Empty taxa_ids list should return 400."""
        response = self.client.post(
            self.url,
            data=json.dumps({'taxa_ids': []}),
            content_type='application/json',
        )
        assert response.status_code == 400

    def test_taxa_ids_not_list_returns_400(self) -> None:
        """taxa_ids as a string should return 400."""
        response = self.client.post(
            self.url,
            data=json.dumps({'taxa_ids': 'not-a-list'}),
            content_type='application/json',
        )
        assert response.status_code == 400

    def test_taxa_ids_null_returns_400(self) -> None:
        """taxa_ids: null should return 400."""
        response = self.client.post(
            self.url,
            data=json.dumps({'taxa_ids': None}),
            content_type='application/json',
        )
        assert response.status_code == 400

    def test_taxa_ids_integer_returns_400(self) -> None:
        """taxa_ids as an integer should return 400."""
        response = self.client.post(
            self.url,
            data=json.dumps({'taxa_ids': 42}),
            content_type='application/json',
        )
        assert response.status_code == 400


class TestTextClassifierViewFunctional(TestCase):
    """Functional tests for POST /api/classifier/text/ with mocked classifiers."""

    def setUp(self) -> None:
        self.client = Client()
        self.url = reverse('search:classifier-text')

    @patch('search.views.settings')
    def test_successful_response_structure(self, mock_settings) -> None:
        """Verify the full response structure on success."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()
        mock_cls = MagicMock(return_value=mock_classifier)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=mock_cls
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({
                    'taxa_ids': ['id1', 'id2', 'id3'],
                    'top_n': 10,
                    'max_depth': 5,
                }),
                content_type='application/json',
            )

        assert response.status_code == 200
        data = response.json()

        # Top-level keys
        assert set(data.keys()) == {'features', 'metadata', 'tree_json'}

        # Feature structure
        assert len(data['features']) == 3
        for feature in data['features']:
            assert set(feature.keys()) == {'name', 'importance', 'display_text'}
            assert isinstance(feature['importance'], float)

        # Metadata structure
        meta = data['metadata']
        assert set(meta.keys()) == {'n_classes', 'n_features', 'tree_depth', 'taxa_count'}
        assert meta['n_classes'] == 5
        assert meta['taxa_count'] == 3

        # Tree JSON passed through
        assert data['tree_json'] == MOCK_TREE_JSON

    @patch('search.views.settings')
    def test_text_features_display_text_is_name(self, mock_settings) -> None:
        """Text classifier features: display_text should equal name."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        data = response.json()
        for feature in data['features']:
            assert feature['display_text'] == feature['name'], \
                f"Text feature display_text should equal name, got {feature['display_text']!r} != {feature['name']!r}"

    @patch('search.views.settings')
    def test_default_parameters(self, mock_settings) -> None:
        """Omitted top_n and max_depth should use defaults (30, 10)."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()
        mock_cls = MagicMock(return_value=mock_classifier)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=mock_cls
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        assert response.status_code == 200
        # Verify classifier was constructed with max_depth=10 (default)
        mock_cls.assert_called_once()
        ctor_kwargs = mock_cls.call_args[1]
        assert ctor_kwargs['max_depth'] == 10
        # Verify get_feature_importances was called with top_n=30 (default)
        mock_classifier.get_feature_importances.assert_called_once_with(top_n=30)
        # Verify tree_to_json was called with max_depth=10 (default)
        mock_classifier.tree_to_json.assert_called_once_with(max_depth=10)

    @patch('search.views.settings')
    def test_custom_parameters_passed_through(self, mock_settings) -> None:
        """Custom top_n, max_depth, min_df, max_df should be passed to classifier."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()
        mock_cls = MagicMock(return_value=mock_classifier)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=mock_cls
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({
                    'taxa_ids': ['id1', 'id2'],
                    'top_n': 15,
                    'max_depth': 7,
                    'min_df': 2,
                    'max_df': 0.8,
                }),
                content_type='application/json',
            )

        assert response.status_code == 200
        ctor_kwargs = mock_cls.call_args[1]
        assert ctor_kwargs['max_depth'] == 7
        assert ctor_kwargs['min_df'] == 2
        assert ctor_kwargs['max_df'] == 0.8
        mock_classifier.get_feature_importances.assert_called_once_with(top_n=15)
        mock_classifier.tree_to_json.assert_called_once_with(max_depth=7)

    @patch('search.views.settings')
    def test_default_min_df_max_df(self, mock_settings) -> None:
        """Default min_df should be 1 and max_df should be 1.0."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()
        mock_cls = MagicMock(return_value=mock_classifier)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=mock_cls
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        assert response.status_code == 200
        ctor_kwargs = mock_cls.call_args[1]
        assert ctor_kwargs['min_df'] == 1
        assert ctor_kwargs['max_df'] == 1.0

    @patch('search.views.settings')
    def test_classifier_uses_correct_database(self, mock_settings) -> None:
        """Text classifier should use 'skol_taxa_dev' database."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()
        mock_cls = MagicMock(return_value=mock_classifier)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=mock_cls
            ),
        }):
            self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        ctor_kwargs = mock_cls.call_args[1]
        assert ctor_kwargs['database'] == 'skol_taxa_dev'

    @patch('search.views.settings')
    def test_classifier_trains_with_test_size_zero(self, mock_settings) -> None:
        """Classifier should be trained with test_size=0.0 (no test split)."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        mock_classifier.fit.assert_called_once_with(
            taxa_ids=['id1', 'id2'], test_size=0.0
        )

    @patch('search.views.settings')
    def test_value_error_returns_400(self, mock_settings) -> None:
        """ValueError from classifier should return 400."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = MagicMock()
        mock_classifier.fit.side_effect = ValueError("Need at least 2 taxa documents")

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1']}),
                content_type='application/json',
            )

        assert response.status_code == 400
        assert 'Need at least 2' in response.json()['error']

    @patch('search.views.settings')
    def test_runtime_error_returns_500(self, mock_settings) -> None:
        """Unexpected exceptions from classifier should return 500."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = MagicMock()
        mock_classifier.fit.side_effect = ConnectionError("CouchDB unreachable")

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        assert response.status_code == 500
        assert 'CouchDB unreachable' in response.json()['error']

    @patch('search.views.settings')
    def test_feature_importance_values_are_floats(self, mock_settings) -> None:
        """Feature importance values should be JSON floats, not numpy types."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        data = response.json()
        for feature in data['features']:
            assert type(feature['importance']) is float, \
                f"Expected float, got {type(feature['importance'])}"

    @patch('search.views.settings')
    def test_feature_order_preserved(self, mock_settings) -> None:
        """Features should be returned in the same order as get_feature_importances."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_text_classifier()

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_decision_tree': MagicMock(
                TaxaDecisionTreeClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        data = response.json()
        names = [f['name'] for f in data['features']]
        assert names == ['brown', 'spores', 'cap']


class TestJsonClassifierViewValidation(TestCase):
    """Unit tests for request validation on POST /api/classifier/json/."""

    def setUp(self) -> None:
        self.client = Client()
        self.url = reverse('search:classifier-json')

    def test_get_method_not_allowed(self) -> None:
        response = self.client.get(self.url)
        assert response.status_code == 405

    def test_missing_taxa_ids_returns_400(self) -> None:
        response = self.client.post(
            self.url,
            data=json.dumps({}),
            content_type='application/json',
        )
        assert response.status_code == 400

    def test_empty_taxa_ids_returns_400(self) -> None:
        response = self.client.post(
            self.url,
            data=json.dumps({'taxa_ids': []}),
            content_type='application/json',
        )
        assert response.status_code == 400

    def test_taxa_ids_not_list_returns_400(self) -> None:
        response = self.client.post(
            self.url,
            data=json.dumps({'taxa_ids': {'key': 'value'}}),
            content_type='application/json',
        )
        assert response.status_code == 400


class TestJsonClassifierViewFunctional(TestCase):
    """Functional tests for POST /api/classifier/json/ with mocked classifiers."""

    def setUp(self) -> None:
        self.client = Client()
        self.url = reverse('search:classifier-json')

    @patch('search.views.settings')
    def test_display_text_conversion(self, mock_settings) -> None:
        """JSON features should convert key=value to 'key value' with _ as spaces."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        json_importances = [
            ('taxon_name_genus=Aspergillus', 0.30),
            ('morphology_spore_shape=globose', 0.20),
            ('habitat_substrate_0=wood', 0.10),
        ]

        mock_classifier = _mock_json_classifier(importances=json_importances)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_json_classifier': MagicMock(
                TaxaJsonClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data['features']) == 3
        assert data['features'][0]['display_text'] == 'taxon name genus Aspergillus'
        assert data['features'][1]['display_text'] == 'morphology spore shape globose'
        assert data['features'][2]['display_text'] == 'habitat substrate 0 wood'
        # Raw names should be preserved
        assert data['features'][0]['name'] == 'taxon_name_genus=Aspergillus'

    @patch('search.views.settings')
    def test_classifier_uses_correct_database(self, mock_settings) -> None:
        """JSON classifier should use 'skol_taxa_full_dev' database."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_json_classifier()
        mock_cls = MagicMock(return_value=mock_classifier)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_json_classifier': MagicMock(
                TaxaJsonClassifier=mock_cls
            ),
        }):
            self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        ctor_kwargs = mock_cls.call_args[1]
        assert ctor_kwargs['database'] == 'skol_taxa_full_dev'

    @patch('search.views.settings')
    def test_classifier_trains_with_test_size_zero(self, mock_settings) -> None:
        """JSON classifier should be trained with test_size=0.0."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = _mock_json_classifier()

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_json_classifier': MagicMock(
                TaxaJsonClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2', 'id3']}),
                content_type='application/json',
            )

        mock_classifier.fit.assert_called_once_with(
            taxa_ids=['id1', 'id2', 'id3'], test_size=0.0
        )

    @patch('search.views.settings')
    def test_value_error_returns_400(self, mock_settings) -> None:
        """ValueError from classifier should return 400."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = MagicMock()
        mock_classifier.fit.side_effect = ValueError("Need at least 2 documents")

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_json_classifier': MagicMock(
                TaxaJsonClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1']}),
                content_type='application/json',
            )

        assert response.status_code == 400
        assert 'Need at least 2 documents' in response.json()['error']

    @patch('search.views.settings')
    def test_runtime_error_returns_500(self, mock_settings) -> None:
        """Unexpected exceptions should return 500."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        mock_classifier = MagicMock()
        mock_classifier.fit.side_effect = RuntimeError("Unexpected failure")

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_json_classifier': MagicMock(
                TaxaJsonClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        assert response.status_code == 500
        assert 'Unexpected failure' in response.json()['error']

    @patch('search.views.settings')
    def test_couchdb_credentials_passed_to_classifier(self, mock_settings) -> None:
        """CouchDB credentials from settings should be passed to classifier."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://couch.example.com:5984'
        mock_settings.COUCHDB_USERNAME = 'myuser'
        mock_settings.COUCHDB_PASSWORD = 'mypass'

        mock_classifier = _mock_json_classifier()
        mock_cls = MagicMock(return_value=mock_classifier)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_json_classifier': MagicMock(
                TaxaJsonClassifier=mock_cls
            ),
        }):
            self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        ctor_kwargs = mock_cls.call_args[1]
        assert ctor_kwargs['couchdb_url'] == 'http://couch.example.com:5984'
        assert ctor_kwargs['username'] == 'myuser'
        assert ctor_kwargs['password'] == 'mypass'

    @patch('search.views.settings')
    def test_display_text_without_equals_sign(self, mock_settings) -> None:
        """Features without = should still convert _ to spaces."""
        mock_settings.SKOL_ROOT_PATH = '/fake/path'
        mock_settings.COUCHDB_URL = 'http://localhost:5984'
        mock_settings.COUCHDB_USERNAME = 'admin'
        mock_settings.COUCHDB_PASSWORD = 'password'

        importances = [
            ('simple_feature', 0.50),
            ('no_underscore', 0.25),
        ]
        mock_classifier = _mock_json_classifier(importances=importances)

        with patch.dict('sys.modules', {
            'taxa_classifier': MagicMock(),
            'taxa_classifier.taxa_json_classifier': MagicMock(
                TaxaJsonClassifier=MagicMock(return_value=mock_classifier)
            ),
        }):
            response = self.client.post(
                self.url,
                data=json.dumps({'taxa_ids': ['id1', 'id2']}),
                content_type='application/json',
            )

        data = response.json()
        assert data['features'][0]['display_text'] == 'simple feature'
        assert data['features'][1]['display_text'] == 'no underscore'


class TestClassifierURLRouting(TestCase):
    """Test that classifier URLs resolve correctly."""

    def test_text_classifier_url_resolves(self) -> None:
        url = reverse('search:classifier-text')
        assert url.endswith('/classifier/text/')

    def test_json_classifier_url_resolves(self) -> None:
        url = reverse('search:classifier-json')
        assert url.endswith('/classifier/json/')
