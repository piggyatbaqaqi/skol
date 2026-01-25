"""
Tests for Collection API views.

Run with: pytest search/tests/test_collection_views.py -v
"""
import json
import pytest
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from search.models import (
    Collection,
    SearchHistory,
    ExternalIdentifier,
    IdentifierType,
)


class TestIdentifierTypeListView(TestCase):
    """Tests for GET /api/identifier-types/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.url = reverse('search:identifier-types')

    def test_unauthenticated_returns_403(self) -> None:
        """Test that unauthenticated requests return 403."""
        response = self.client.get(self.url)
        assert response.status_code == 403

    def test_authenticated_returns_identifier_types(self) -> None:
        """Test that authenticated requests return identifier types."""
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.url)
        assert response.status_code == 200
        data = response.json()
        assert 'identifier_types' in data
        assert 'count' in data
        # Should have seeded types
        assert data['count'] >= 5

    def test_response_contains_expected_fields(self) -> None:
        """Test that response contains expected fields for each type."""
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.url)
        data = response.json()
        for it in data['identifier_types']:
            assert 'id' in it
            assert 'code' in it
            assert 'name' in it
            assert 'url_pattern' in it


class TestCollectionListCreateView(TestCase):
    """Tests for GET/POST /api/collections/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.other_user = User.objects.create_user('otheruser', 'other@example.com', 'password')
        self.url = reverse('search:collection-list-create')

    def test_unauthenticated_returns_403(self) -> None:
        """Test that unauthenticated requests return 403."""
        response = self.client.get(self.url)
        assert response.status_code == 403

    def test_list_own_collections(self) -> None:
        """Test listing own collections."""
        Collection.objects.create(owner=self.user, name='My Collection')
        Collection.objects.create(owner=self.other_user, name='Other Collection')

        self.client.login(username='testuser', password='password')
        response = self.client.get(self.url)
        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 1
        assert data['collections'][0]['name'] == 'My Collection'

    def test_create_collection(self) -> None:
        """Test creating a new collection."""
        self.client.login(username='testuser', password='password')
        response = self.client.post(
            self.url,
            data=json.dumps({'name': 'New Collection', 'description': 'Test description'}),
            content_type='application/json'
        )
        assert response.status_code == 201
        data = response.json()
        assert data['name'] == 'New Collection'
        assert data['description'] == 'Test description'
        assert 'collection_id' in data
        assert 100000000 <= data['collection_id'] <= 999999999

    def test_create_collection_default_name(self) -> None:
        """Test creating a collection with default name."""
        self.client.login(username='testuser', password='password')
        response = self.client.post(
            self.url,
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 201
        data = response.json()
        assert data['name'] == 'Untitled Collection'


class TestCollectionDetailView(TestCase):
    """Tests for GET/PUT/DELETE /api/collections/<collection_id>/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.other_user = User.objects.create_user('otheruser', 'other@example.com', 'password')
        self.collection = Collection.objects.create(owner=self.user, name='Test Collection')

    def get_url(self, collection_id: int) -> str:
        return reverse('search:collection-detail', kwargs={'collection_id': collection_id})

    def test_get_collection_as_owner(self) -> None:
        """Test getting collection as owner."""
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.get_url(self.collection.collection_id))
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'Test Collection'
        assert 'search_history' in data
        assert 'external_identifiers' in data

    def test_get_collection_as_other_user(self) -> None:
        """Test getting collection as another authenticated user (allowed)."""
        self.client.login(username='otheruser', password='password')
        response = self.client.get(self.get_url(self.collection.collection_id))
        assert response.status_code == 200

    def test_get_nonexistent_collection(self) -> None:
        """Test getting nonexistent collection returns 404."""
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.get_url(999999999))
        assert response.status_code == 404

    def test_update_collection_as_owner(self) -> None:
        """Test updating collection as owner."""
        self.client.login(username='testuser', password='password')
        response = self.client.put(
            self.get_url(self.collection.collection_id),
            data=json.dumps({'name': 'Updated Name'}),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'Updated Name'

    def test_update_collection_as_other_user_forbidden(self) -> None:
        """Test updating collection as other user is forbidden."""
        self.client.login(username='otheruser', password='password')
        response = self.client.put(
            self.get_url(self.collection.collection_id),
            data=json.dumps({'name': 'Hacked Name'}),
            content_type='application/json'
        )
        assert response.status_code == 403

    def test_delete_collection_as_owner(self) -> None:
        """Test deleting collection as owner."""
        self.client.login(username='testuser', password='password')
        response = self.client.delete(self.get_url(self.collection.collection_id))
        assert response.status_code == 204
        assert not Collection.objects.filter(collection_id=self.collection.collection_id).exists()

    def test_delete_collection_as_other_user_forbidden(self) -> None:
        """Test deleting collection as other user is forbidden."""
        self.client.login(username='otheruser', password='password')
        response = self.client.delete(self.get_url(self.collection.collection_id))
        assert response.status_code == 403
        assert Collection.objects.filter(collection_id=self.collection.collection_id).exists()


class TestCollectionByUserView(TestCase):
    """Tests for GET /api/collections/user/<username>/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.other_user = User.objects.create_user('otheruser', 'other@example.com', 'password')
        Collection.objects.create(owner=self.user, name='Collection 1')
        Collection.objects.create(owner=self.user, name='Collection 2')
        Collection.objects.create(owner=self.other_user, name='Other Collection')

    def get_url(self, username: str) -> str:
        return reverse('search:collection-by-user', kwargs={'username': username})

    def test_view_user_collections(self) -> None:
        """Test viewing another user's collections."""
        self.client.login(username='otheruser', password='password')
        response = self.client.get(self.get_url('testuser'))
        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 2
        assert data['username'] == 'testuser'

    def test_view_own_collections(self) -> None:
        """Test viewing own collections via this endpoint."""
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.get_url('testuser'))
        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 2


class TestSearchHistoryListCreateView(TestCase):
    """Tests for GET/POST /api/collections/<id>/searches/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.other_user = User.objects.create_user('otheruser', 'other@example.com', 'password')
        self.collection = Collection.objects.create(owner=self.user, name='Test Collection')

    def get_url(self, collection_id: int) -> str:
        return reverse('search:search-history-list-create', kwargs={'collection_id': collection_id})

    def test_list_searches(self) -> None:
        """Test listing search history."""
        SearchHistory.objects.create(
            collection=self.collection,
            prompt='test search',
            embedding_name='skol:embedding:v1.1',
            k=3
        )
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.get_url(self.collection.collection_id))
        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 1
        assert data['searches'][0]['prompt'] == 'test search'

    def test_add_search_as_owner(self) -> None:
        """Test adding search as owner."""
        self.client.login(username='testuser', password='password')
        response = self.client.post(
            self.get_url(self.collection.collection_id),
            data=json.dumps({
                'prompt': 'new search',
                'embedding_name': 'skol:embedding:v1.1',
                'k': 5,
                'result_references': [{'title': 'Result 1', 'similarity': 0.95}]
            }),
            content_type='application/json'
        )
        assert response.status_code == 201
        data = response.json()
        assert data['prompt'] == 'new search'
        assert data['result_count'] == 1

    def test_add_search_as_other_user_forbidden(self) -> None:
        """Test adding search as other user is forbidden."""
        self.client.login(username='otheruser', password='password')
        response = self.client.post(
            self.get_url(self.collection.collection_id),
            data=json.dumps({
                'prompt': 'hacked search',
                'embedding_name': 'test',
                'k': 3
            }),
            content_type='application/json'
        )
        assert response.status_code == 403


class TestSearchHistoryDetailView(TestCase):
    """Tests for GET/DELETE /api/collections/<id>/searches/<sid>/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.other_user = User.objects.create_user('otheruser', 'other@example.com', 'password')
        self.collection = Collection.objects.create(owner=self.user, name='Test Collection')
        self.search = SearchHistory.objects.create(
            collection=self.collection,
            prompt='test search',
            embedding_name='skol:embedding:v1.1',
            k=3
        )

    def get_url(self, collection_id: int, search_id: int) -> str:
        return reverse('search:search-history-detail', kwargs={
            'collection_id': collection_id,
            'search_id': search_id
        })

    def test_get_search(self) -> None:
        """Test getting a specific search."""
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.get_url(self.collection.collection_id, self.search.id))
        assert response.status_code == 200
        data = response.json()
        assert data['prompt'] == 'test search'

    def test_delete_search_as_owner(self) -> None:
        """Test deleting search as owner."""
        self.client.login(username='testuser', password='password')
        response = self.client.delete(self.get_url(self.collection.collection_id, self.search.id))
        assert response.status_code == 204
        assert not SearchHistory.objects.filter(id=self.search.id).exists()

    def test_delete_search_as_other_user_forbidden(self) -> None:
        """Test deleting search as other user is forbidden."""
        self.client.login(username='otheruser', password='password')
        response = self.client.delete(self.get_url(self.collection.collection_id, self.search.id))
        assert response.status_code == 403


class TestExternalIdentifierListCreateView(TestCase):
    """Tests for GET/POST /api/collections/<id>/identifiers/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.other_user = User.objects.create_user('otheruser', 'other@example.com', 'password')
        self.collection = Collection.objects.create(owner=self.user, name='Test Collection')
        self.inat_type = IdentifierType.objects.get(code='inat')

    def get_url(self, collection_id: int) -> str:
        return reverse('search:identifier-list-create', kwargs={'collection_id': collection_id})

    def test_list_identifiers(self) -> None:
        """Test listing external identifiers."""
        ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.inat_type,
            value='12345'
        )
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.get_url(self.collection.collection_id))
        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 1
        assert data['identifiers'][0]['value'] == '12345'
        assert 'url' in data['identifiers'][0]

    def test_add_identifier_as_owner(self) -> None:
        """Test adding identifier as owner."""
        self.client.login(username='testuser', password='password')
        response = self.client.post(
            self.get_url(self.collection.collection_id),
            data=json.dumps({
                'identifier_type_code': 'inat',
                'value': '336010515',
                'notes': 'Main observation'
            }),
            content_type='application/json'
        )
        assert response.status_code == 201
        data = response.json()
        assert data['value'] == '336010515'
        assert data['identifier_type_code'] == 'inat'
        assert 'inaturalist.org' in data['url']

    def test_add_identifier_as_other_user_forbidden(self) -> None:
        """Test adding identifier as other user is forbidden."""
        self.client.login(username='otheruser', password='password')
        response = self.client.post(
            self.get_url(self.collection.collection_id),
            data=json.dumps({
                'identifier_type_code': 'inat',
                'value': '12345'
            }),
            content_type='application/json'
        )
        assert response.status_code == 403


class TestExternalIdentifierDetailView(TestCase):
    """Tests for GET/DELETE /api/collections/<id>/identifiers/<iid>/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.other_user = User.objects.create_user('otheruser', 'other@example.com', 'password')
        self.collection = Collection.objects.create(owner=self.user, name='Test Collection')
        self.inat_type = IdentifierType.objects.get(code='inat')
        self.identifier = ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.inat_type,
            value='12345'
        )

    def get_url(self, collection_id: int, identifier_id: int) -> str:
        return reverse('search:identifier-detail', kwargs={
            'collection_id': collection_id,
            'identifier_id': identifier_id
        })

    def test_get_identifier(self) -> None:
        """Test getting a specific identifier."""
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.get_url(self.collection.collection_id, self.identifier.id))
        assert response.status_code == 200
        data = response.json()
        assert data['value'] == '12345'

    def test_delete_identifier_as_owner(self) -> None:
        """Test deleting identifier as owner."""
        self.client.login(username='testuser', password='password')
        response = self.client.delete(self.get_url(self.collection.collection_id, self.identifier.id))
        assert response.status_code == 204
        assert not ExternalIdentifier.objects.filter(id=self.identifier.id).exists()

    def test_delete_identifier_as_other_user_forbidden(self) -> None:
        """Test deleting identifier as other user is forbidden."""
        self.client.login(username='otheruser', password='password')
        response = self.client.delete(self.get_url(self.collection.collection_id, self.identifier.id))
        assert response.status_code == 403
