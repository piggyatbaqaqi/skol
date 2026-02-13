"""
Tests for Comment/Discussion API views.

Run with: pytest search/tests/test_comment_views.py -v

These tests mock comment_service to avoid needing a live CouchDB.
"""
import json
from unittest.mock import patch, MagicMock

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from search.models import Collection


def _make_comment(
    doc_id='comment_123_1000_abcd',
    collection_id=None,
    user_id=1,
    username='testuser',
    body='Test comment',
    nomenclature='',
    path='/1/',
    parent_path='',
    deleted=False,
    hidden=False,
    flagged_by=None,
    edit_history=None,
):
    """Helper to build a comment dict matching CouchDB document shape."""
    return {
        '_id': doc_id,
        'type': 'comment',
        'collection_id': collection_id,
        'path': path,
        'depth': len(path.strip('/').split('/')) - 1,
        'parent_path': parent_path,
        'sort_key': path,
        'author': {'user_id': user_id, 'username': username},
        'body': body,
        'nomenclature': nomenclature,
        'created_at': '2026-02-13T19:00:00+00:00',
        'updated_at': '2026-02-13T19:00:00+00:00',
        'edit_history': edit_history or [],
        'deleted': deleted,
        'flagged_by': flagged_by or [],
        'hidden': hidden,
        'hidden_by': None,
        'hidden_at': None,
    }


SERVICE = 'search.comment_service'


class TestCommentListCreateView(TestCase):
    """Tests for GET/POST /api/collections/<id>/comments/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user(
            'testuser', 'test@example.com', 'password'
        )
        self.other_user = User.objects.create_user(
            'otheruser', 'other@example.com', 'password'
        )
        self.collection = Collection.objects.create(
            owner=self.user, name='Test Collection'
        )
        self.url = reverse(
            'search:comment-list-create',
            kwargs={'collection_id': self.collection.collection_id},
        )

    def test_unauthenticated_returns_403(self) -> None:
        response = self.client.get(self.url)
        assert response.status_code == 403

    @patch(f'{SERVICE}.get_comments_for_collection')
    def test_get_comments_as_owner(self, mock_get) -> None:
        cid = self.collection.collection_id
        mock_get.return_value = [
            _make_comment(collection_id=cid),
        ]
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.url)
        assert response.status_code == 200
        data = response.json()
        assert data['count'] == 1
        assert data['is_owner'] is True
        assert data['current_user_id'] == self.user.id
        mock_get.assert_called_once_with(cid, include_hidden=True)

    @patch(f'{SERVICE}.get_comments_for_collection')
    def test_get_comments_as_other_user(self, mock_get) -> None:
        cid = self.collection.collection_id
        mock_get.return_value = []
        self.client.login(username='otheruser', password='password')
        response = self.client.get(self.url)
        assert response.status_code == 200
        data = response.json()
        assert data['is_owner'] is False
        mock_get.assert_called_once_with(cid, include_hidden=False)

    @patch(f'{SERVICE}.get_comments_for_collection')
    def test_deleted_comments_sanitized(self, mock_get) -> None:
        cid = self.collection.collection_id
        mock_get.return_value = [
            _make_comment(
                collection_id=cid,
                deleted=True,
                body='Secret text',
                nomenclature='Secret name',
                edit_history=[{'body': 'old', 'nomenclature': '', 'edited_at': ''}],
            ),
        ]
        self.client.login(username='testuser', password='password')
        response = self.client.get(self.url)
        data = response.json()
        comment = data['comments'][0]
        assert comment['body'] == '[deleted]'
        assert comment['author']['user_id'] is None
        assert comment['nomenclature'] == ''
        assert comment['edit_history'] == []

    @patch(f'{SERVICE}.create_comment')
    def test_post_new_comment(self, mock_create) -> None:
        cid = self.collection.collection_id
        mock_create.return_value = _make_comment(
            collection_id=cid, body='New comment'
        )
        self.client.login(username='testuser', password='password')
        response = self.client.post(
            self.url,
            data=json.dumps({
                'body': 'New comment',
                'nomenclature': 'Agaricus',
            }),
            content_type='application/json',
        )
        assert response.status_code == 201
        mock_create.assert_called_once_with(
            collection_id=cid,
            user_id=self.user.id,
            username='testuser',
            body='New comment',
            nomenclature='Agaricus',
            parent_path='',
        )

    @patch(f'{SERVICE}.create_comment')
    def test_post_reply(self, mock_create) -> None:
        cid = self.collection.collection_id
        mock_create.return_value = _make_comment(
            collection_id=cid, path='/1/1/', parent_path='/1/'
        )
        self.client.login(username='testuser', password='password')
        response = self.client.post(
            self.url,
            data=json.dumps({
                'body': 'Reply text',
                'parent_path': '/1/',
            }),
            content_type='application/json',
        )
        assert response.status_code == 201
        mock_create.assert_called_once_with(
            collection_id=cid,
            user_id=self.user.id,
            username='testuser',
            body='Reply text',
            nomenclature='',
            parent_path='/1/',
        )

    def test_post_empty_body_returns_400(self) -> None:
        self.client.login(username='testuser', password='password')
        response = self.client.post(
            self.url,
            data=json.dumps({'body': '  '}),
            content_type='application/json',
        )
        assert response.status_code == 400


class TestCommentCountView(TestCase):
    """Tests for GET /api/collections/<id>/comments/count/."""

    def setUp(self) -> None:
        self.client = Client()
        self.collection = Collection.objects.create(
            owner=User.objects.create_user(
                'testuser', 'test@example.com', 'password'
            ),
            name='Test',
        )
        self.url = reverse(
            'search:comment-count',
            kwargs={'collection_id': self.collection.collection_id},
        )

    @patch(f'{SERVICE}.get_comment_count')
    def test_count_no_auth_required(self, mock_count) -> None:
        mock_count.return_value = 5
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert response.json()['count'] == 5


class TestCommentDetailView(TestCase):
    """Tests for PUT/DELETE /api/collections/<id>/comments/<cid>/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user(
            'testuser', 'test@example.com', 'password'
        )
        self.other_user = User.objects.create_user(
            'otheruser', 'other@example.com', 'password'
        )
        self.admin_user = User.objects.create_superuser(
            'admin', 'admin@example.com', 'password'
        )
        self.collection = Collection.objects.create(
            owner=self.user, name='Test Collection'
        )
        self.comment_id = 'comment_123_1000_abcd'

    def get_url(self):
        return reverse(
            'search:comment-detail',
            kwargs={
                'collection_id': self.collection.collection_id,
                'comment_id': self.comment_id,
            },
        )

    @patch(f'{SERVICE}.update_comment')
    @patch(f'{SERVICE}.get_comment')
    def test_put_as_author(self, mock_get, mock_update) -> None:
        mock_get.return_value = _make_comment(
            user_id=self.user.id
        )
        mock_update.return_value = _make_comment(
            user_id=self.user.id, body='Updated'
        )
        self.client.login(username='testuser', password='password')
        response = self.client.put(
            self.get_url(),
            data=json.dumps({'body': 'Updated', 'nomenclature': ''}),
            content_type='application/json',
        )
        assert response.status_code == 200
        mock_update.assert_called_once()

    @patch(f'{SERVICE}.get_comment')
    def test_put_as_non_author_returns_403(self, mock_get) -> None:
        mock_get.return_value = _make_comment(
            user_id=self.user.id
        )
        self.client.login(username='otheruser', password='password')
        response = self.client.put(
            self.get_url(),
            data=json.dumps({'body': 'Hacked'}),
            content_type='application/json',
        )
        assert response.status_code == 403

    @patch(f'{SERVICE}.soft_delete_comment')
    @patch(f'{SERVICE}.get_comment')
    def test_delete_as_author(self, mock_get, mock_delete) -> None:
        mock_get.return_value = _make_comment(
            user_id=self.user.id
        )
        mock_delete.return_value = _make_comment(
            user_id=self.user.id, deleted=True
        )
        self.client.login(username='testuser', password='password')
        response = self.client.delete(self.get_url())
        assert response.status_code == 200
        mock_delete.assert_called_once()

    @patch(f'{SERVICE}.soft_delete_comment')
    @patch(f'{SERVICE}.get_comment')
    def test_delete_as_admin(self, mock_get, mock_delete) -> None:
        mock_get.return_value = _make_comment(
            user_id=self.user.id
        )
        mock_delete.return_value = _make_comment(deleted=True)
        self.client.login(username='admin', password='password')
        response = self.client.delete(self.get_url())
        assert response.status_code == 200

    @patch(f'{SERVICE}.get_comment')
    def test_delete_as_non_author_non_owner_returns_403(
        self, mock_get
    ) -> None:
        mock_get.return_value = _make_comment(
            user_id=self.user.id
        )
        self.client.login(username='otheruser', password='password')
        response = self.client.delete(self.get_url())
        assert response.status_code == 403


class TestCommentFlagView(TestCase):
    """Tests for POST /api/collections/<id>/comments/<cid>/flag/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user(
            'testuser', 'test@example.com', 'password'
        )
        self.collection = Collection.objects.create(
            owner=self.user, name='Test'
        )
        self.comment_id = 'comment_123_1000_abcd'
        self.url = reverse(
            'search:comment-flag',
            kwargs={
                'collection_id': self.collection.collection_id,
                'comment_id': self.comment_id,
            },
        )

    @patch(f'{SERVICE}.flag_comment')
    def test_flag_comment(self, mock_flag) -> None:
        mock_flag.return_value = _make_comment(
            flagged_by=[self.user.id]
        )
        self.client.login(username='testuser', password='password')
        response = self.client.post(self.url)
        assert response.status_code == 200
        mock_flag.assert_called_once_with(
            self.comment_id, self.user.id
        )


class TestCommentHideView(TestCase):
    """Tests for POST/DELETE /api/collections/<id>/comments/<cid>/hide/."""

    def setUp(self) -> None:
        self.client = Client()
        self.owner = User.objects.create_user(
            'owner', 'owner@example.com', 'password'
        )
        self.other_user = User.objects.create_user(
            'other', 'other@example.com', 'password'
        )
        self.collection = Collection.objects.create(
            owner=self.owner, name='Test'
        )
        self.comment_id = 'comment_123_1000_abcd'
        self.url = reverse(
            'search:comment-hide',
            kwargs={
                'collection_id': self.collection.collection_id,
                'comment_id': self.comment_id,
            },
        )

    @patch(f'{SERVICE}.hide_comment')
    def test_owner_can_hide(self, mock_hide) -> None:
        mock_hide.return_value = _make_comment(hidden=True)
        self.client.login(username='owner', password='password')
        response = self.client.post(self.url)
        assert response.status_code == 200
        mock_hide.assert_called_once()

    def test_non_owner_cannot_hide(self) -> None:
        self.client.login(username='other', password='password')
        response = self.client.post(self.url)
        assert response.status_code == 403

    @patch(f'{SERVICE}.unhide_comment')
    def test_owner_can_unhide(self, mock_unhide) -> None:
        mock_unhide.return_value = _make_comment(hidden=False)
        self.client.login(username='owner', password='password')
        response = self.client.delete(self.url)
        assert response.status_code == 200
        mock_unhide.assert_called_once()


class TestCommentCopyNomenclatureView(TestCase):
    """Tests for POST .../copy-nomenclature/."""

    def setUp(self) -> None:
        self.client = Client()
        self.owner = User.objects.create_user(
            'owner', 'owner@example.com', 'password'
        )
        self.other_user = User.objects.create_user(
            'other', 'other@example.com', 'password'
        )
        self.collection = Collection.objects.create(
            owner=self.owner, name='Test', nomenclature='Old name'
        )
        self.comment_id = 'comment_123_1000_abcd'
        self.url = reverse(
            'search:comment-copy-nomenclature',
            kwargs={
                'collection_id': self.collection.collection_id,
                'comment_id': self.comment_id,
            },
        )

    @patch('search.couchdb_sync.sync_collection_to_couchdb')
    @patch(f'{SERVICE}.get_comment')
    def test_owner_can_copy_nomenclature(
        self, mock_get, mock_sync
    ) -> None:
        mock_get.return_value = _make_comment(
            nomenclature='Geastrum triplex'
        )
        mock_sync.return_value = True
        self.client.login(username='owner', password='password')
        response = self.client.post(self.url)
        assert response.status_code == 200
        data = response.json()
        assert data['nomenclature'] == 'Geastrum triplex'

        # Verify Django model updated
        self.collection.refresh_from_db()
        assert self.collection.nomenclature == 'Geastrum triplex'

    @patch(f'{SERVICE}.get_comment')
    def test_non_owner_cannot_copy(self, mock_get) -> None:
        self.client.login(username='other', password='password')
        response = self.client.post(self.url)
        assert response.status_code == 403

    @patch(f'{SERVICE}.get_comment')
    def test_empty_nomenclature_returns_400(self, mock_get) -> None:
        mock_get.return_value = _make_comment(nomenclature='')
        self.client.login(username='owner', password='password')
        response = self.client.post(self.url)
        assert response.status_code == 400
