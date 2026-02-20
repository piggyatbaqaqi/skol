"""
Tests for the "Download My Data" export feature.

Run with: pytest search/tests/test_export_service.py -v

CouchDB interactions are mocked to avoid needing a live database.
"""
import io
import json
import zipfile
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from search.models import (
    Collection,
    SearchHistory,
    ExternalIdentifier,
    IdentifierType,
    UserSettings,
)


EXPORT_SERVICE = 'search.export_service'


class TestExportUserData(TestCase):
    """Tests for the export_user_data() service function."""

    def setUp(self):
        self.user = User.objects.create_user(
            'testuser', 'test@example.com', 'password',
        )
        self.id_type, _ = IdentifierType.objects.get_or_create(
            code='inat',
            defaults={
                'name': 'iNaturalist',
                'url_pattern': (
                    'https://www.inaturalist.org/'
                    'observations/{id}'
                ),
            },
        )

    def _export(self, user=None):
        """Run export and return (ZipFile, BytesIO)."""
        from search.export_service import export_user_data

        buf = export_user_data(user or self.user)
        zf = zipfile.ZipFile(buf, 'r')
        return zf, buf

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_contains_user_json(self, mock_server, mock_ids):
        zf, _ = self._export()
        assert 'user.json' in zf.namelist()
        data = json.loads(zf.read('user.json'))
        assert data['username'] == 'testuser'
        assert data['email'] == 'test@example.com'
        assert data['id'] == self.user.id

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_contains_settings(self, mock_server, mock_ids):
        UserSettings.objects.create(
            user=self.user,
            default_embargo_days=30,
            default_k=10,
        )
        zf, _ = self._export()
        data = json.loads(zf.read('user.json'))
        assert 'settings' in data
        assert data['settings']['default_embargo_days'] == 30
        assert data['settings']['default_k'] == 10

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_no_settings(self, mock_server, mock_ids):
        zf, _ = self._export()
        data = json.loads(zf.read('user.json'))
        assert 'settings' not in data

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_contains_collections(self, mock_server, mock_ids):
        c = Collection.objects.create(
            owner=self.user, name='My Collection',
        )
        zf, _ = self._export()
        expected = f'collections/{c.collection_id}.json'
        assert expected in zf.namelist()
        data = json.loads(zf.read(expected))
        assert data['name'] == 'My Collection'
        assert data['collection_id'] == c.collection_id

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_collections_include_search_history(
        self, mock_server, mock_ids,
    ):
        c = Collection.objects.create(
            owner=self.user, name='My Collection',
        )
        SearchHistory.objects.create(
            collection=c,
            event_type='search',
            prompt='red mushroom',
            k=5,
            result_count=3,
        )
        zf, _ = self._export()
        data = json.loads(
            zf.read(f'collections/{c.collection_id}.json')
        )
        assert len(data['search_history']) == 1
        assert data['search_history'][0]['prompt'] == 'red mushroom'

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_collections_include_identifiers(
        self, mock_server, mock_ids,
    ):
        c = Collection.objects.create(
            owner=self.user, name='My Collection',
        )
        ExternalIdentifier.objects.create(
            collection=c,
            identifier_type=self.id_type,
            value='12345',
        )
        zf, _ = self._export()
        data = json.loads(
            zf.read(f'collections/{c.collection_id}.json')
        )
        assert len(data['external_identifiers']) == 1
        assert data['external_identifiers'][0]['value'] == '12345'
        assert data['external_identifiers'][0]['identifier_type'] == 'inat'

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_couchdb_collections(self, mock_server, mock_ids):
        c = Collection.objects.create(
            owner=self.user, name='My Collection',
        )
        doc_id = f'collection_{c.collection_id}'
        couch_doc = {
            '_id': doc_id,
            '_rev': '1-abc',
            'type': 'collection',
            'taxon': 'Fungi',
        }

        # Mock CouchDB server and database
        mock_db = MagicMock()
        mock_db.__contains__ = MagicMock(
            side_effect=lambda k: k == doc_id
        )
        mock_db.__getitem__ = MagicMock(
            return_value=couch_doc
        )
        mock_srv = MagicMock()
        mock_srv.__contains__ = MagicMock(return_value=True)
        mock_srv.__getitem__ = MagicMock(return_value=mock_db)
        mock_server.return_value = mock_srv

        zf, _ = self._export()
        expected = f'couchdb_collections/{doc_id}.json'
        assert expected in zf.namelist()
        data = json.loads(zf.read(expected))
        assert data['taxon'] == 'Fungi'
        assert '_rev' not in data

    @patch(f'{EXPORT_SERVICE}.get_comments_for_collection')
    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author')
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_comment_threads(
        self, mock_server, mock_ids, mock_comments,
    ):
        c = Collection.objects.create(
            owner=self.user, name='My Collection',
        )
        mock_ids.return_value = {c.collection_id}
        mock_comments.return_value = [
            {
                '_id': 'comment_1',
                '_rev': '1-abc',
                'type': 'comment',
                'collection_id': c.collection_id,
                'body': 'Hello',
                'author': {
                    'user_id': self.user.id,
                    'username': 'testuser',
                },
            },
            {
                '_id': 'comment_2',
                '_rev': '2-def',
                'type': 'comment',
                'collection_id': c.collection_id,
                'body': 'Reply',
                'author': {
                    'user_id': 999,
                    'username': 'otheruser',
                },
            },
        ]

        zf, _ = self._export()
        expected = (
            f'comment_threads/'
            f'collection_{c.collection_id}_comments.json'
        )
        assert expected in zf.namelist()
        data = json.loads(zf.read(expected))
        assert data['collection_id'] == c.collection_id
        assert len(data['comments']) == 2
        # Full thread includes other users' comments
        assert data['comments'][1]['body'] == 'Reply'
        # _rev stripped
        assert '_rev' not in data['comments'][0]

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_empty_user(self, mock_server, mock_ids):
        """User with no data gets a valid ZIP with just user.json."""
        zf, _ = self._export()
        names = zf.namelist()
        assert names == ['user.json']

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_export_is_valid_zip(self, mock_server, mock_ids):
        _, buf = self._export()
        assert zipfile.is_zipfile(buf)


class TestExportMyDataAPIView(TestCase):
    """Tests for GET /api/export-my-data/."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            'testuser', 'test@example.com', 'password',
        )
        self.url = reverse('search:export-my-data')

    def test_unauthenticated_returns_403(self):
        response = self.client.get(self.url)
        assert response.status_code == 403

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_returns_zip(self, mock_server, mock_ids):
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert response['Content-Type'] == 'application/zip'
        assert 'skol-export-testuser.zip' in (
            response['Content-Disposition']
        )
        # Verify it's a valid ZIP
        buf = io.BytesIO(response.content)
        assert zipfile.is_zipfile(buf)


class TestExportMyDataAccountView(TestCase):
    """Tests for GET /accounts/export-my-data/."""

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            'testuser', 'test@example.com', 'password',
        )
        self.url = reverse('accounts:export_my_data')

    def test_unauthenticated_redirects_to_login(self):
        response = self.client.get(self.url)
        assert response.status_code == 302
        assert '/login/' in response.url

    @patch(f'{EXPORT_SERVICE}.get_collection_ids_for_author',
           return_value=set())
    @patch(f'{EXPORT_SERVICE}.get_couchdb_server')
    def test_returns_zip(self, mock_server, mock_ids):
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert response['Content-Type'] == 'application/zip'
        buf = io.BytesIO(response.content)
        assert zipfile.is_zipfile(buf)


class TestGetCollectionIdsForAuthor(TestCase):
    """Tests for get_collection_ids_for_author()."""

    @patch('search.comment_service.get_comments_db')
    def test_returns_collection_ids(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Mock design doc already exists
        mock_db.__contains__ = MagicMock(return_value=True)
        mock_db.__getitem__ = MagicMock(return_value={
            '_id': '_design/comments',
            'views': {},
            'language': 'javascript',
        })

        # Mock view result
        row1 = MagicMock()
        row1.value = 111111111
        row2 = MagicMock()
        row2.value = 222222222
        row3 = MagicMock()
        row3.value = 111111111  # duplicate
        mock_db.view.return_value = [row1, row2, row3]

        from search.comment_service import (
            get_collection_ids_for_author,
        )

        result = get_collection_ids_for_author(42)
        assert result == {111111111, 222222222}
        mock_db.view.assert_called_once_with(
            'comments/by_author', key=42,
        )
