"""Tests for PostInatCommentView."""

from unittest.mock import MagicMock, patch

import pytest
from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from search.models import Collection, ExternalIdentifier, IdentifierType

User = get_user_model()


def _create_social_account(user: "User") -> None:
    """Create an iNaturalist social account with token for testing."""
    from allauth.socialaccount.models import SocialAccount, SocialToken

    social_account = SocialAccount.objects.create(
        user=user,
        provider='inaturalist',
        uid='12345',
    )
    SocialToken.objects.create(
        account=social_account,
        token='fake-oauth-token',
    )


def _mock_jwt(mock_get: MagicMock) -> None:
    """Set up mock_get to return a JWT (no taxon lookup)."""
    mock_jwt_resp = MagicMock()
    mock_jwt_resp.json.return_value = {'api_token': 'fake-jwt'}
    mock_get.return_value = mock_jwt_resp


def _mock_jwt_and_taxon(
    mock_get: MagicMock, taxon_id: int = 48978,
    taxon_name: str = 'Russula emetica',
) -> None:
    """Set up mock_get to return JWT then taxon results."""
    mock_jwt_resp = MagicMock()
    mock_jwt_resp.json.return_value = {'api_token': 'fake-jwt'}

    mock_taxon_resp = MagicMock()
    mock_taxon_resp.json.return_value = {
        'results': [{'id': taxon_id, 'name': taxon_name}],
    }

    mock_get.side_effect = [mock_jwt_resp, mock_taxon_resp]


@pytest.mark.django_db
class TestPostInatCommentView(TestCase):
    """Tests for POST /api/collections/<id>/post-inat-comment/."""

    def setUp(self) -> None:
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser', password='testpass123'
        )
        self.other_user = User.objects.create_user(
            username='otheruser', password='testpass123'
        )
        self.client.login(username='testuser', password='testpass123')

        self.collection = Collection.objects.create(
            collection_id=100000001,
            owner=self.user,
            name='Test Collection',
            description='Pileus convex, 3-5 cm broad',
            nomenclature='Russula emetica',
        )

        self.inat_type, _ = IdentifierType.objects.get_or_create(
            code='inat',
            defaults={
                'name': 'iNaturalist',
                'url_pattern':
                    'https://www.inaturalist.org/observations/{id}',
            },
        )

        self.identifier = ExternalIdentifier.objects.create(
            collection=self.collection,
            identifier_type=self.inat_type,
            value='336010515',
        )

        self.url = (
            f'/api/collections/{self.collection.collection_id}'
            '/post-inat-comment/'
        )

    def test_unauthenticated_rejected(self) -> None:
        self.client.logout()
        resp = self.client.post(self.url)
        assert resp.status_code == 403

    def test_non_owner_rejected(self) -> None:
        self.client.login(username='otheruser', password='testpass123')
        resp = self.client.post(self.url)
        assert resp.status_code == 403

    def test_no_inat_identifier(self) -> None:
        self.identifier.delete()
        resp = self.client.post(self.url)
        assert resp.status_code == 400
        assert 'No iNaturalist identifier' in resp.json()['error']

    def test_empty_description(self) -> None:
        self.collection.description = ''
        self.collection.save()
        resp = self.client.post(self.url)
        assert resp.status_code == 400
        assert 'no description' in resp.json()['error'].lower()

    def test_no_social_account(self) -> None:
        resp = self.client.post(self.url)
        assert resp.status_code == 400
        assert 'No iNaturalist account' in resp.json()['error']

    def test_no_social_token(self) -> None:
        from allauth.socialaccount.models import SocialAccount

        SocialAccount.objects.create(
            user=self.user,
            provider='inaturalist',
            uid='12345',
        )
        resp = self.client.post(self.url)
        assert resp.status_code == 400
        assert 'No iNaturalist token' in resp.json()['error']

    @patch('requests.get')
    @patch('requests.post')
    def test_identification_with_description_as_body(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        """With nomenclature, posts a single identification with
        description as body (the 'Tell us why...' field)."""
        _create_social_account(self.user)
        _mock_jwt_and_taxon(mock_get)
        mock_post.return_value = MagicMock()

        resp = self.client.post(self.url)
        assert resp.status_code == 200
        data = resp.json()
        assert data['observation_id'] == '336010515'
        assert 'Identification posted' in data['message']
        assert data['identification']['taxon_id'] == 48978
        assert data['identification']['taxon_name'] == 'Russula emetica'

        # Only one post call — the identification (no separate comment)
        assert mock_post.call_count == 1
        id_json = mock_post.call_args.kwargs['json']['identification']
        assert id_json['observation_id'] == 336010515
        assert id_json['taxon_id'] == 48978
        assert id_json['body'] == 'Pileus convex, 3-5 cm broad'

    @patch('requests.get')
    @patch('requests.post')
    def test_comment_without_nomenclature(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        """Without nomenclature, posts a plain comment."""
        self.collection.nomenclature = ''
        self.collection.save()

        _create_social_account(self.user)
        _mock_jwt(mock_get)
        mock_post.return_value = MagicMock()

        resp = self.client.post(self.url)
        assert resp.status_code == 200
        data = resp.json()
        assert 'Comment posted' in data['message']
        assert 'identification' not in data

        # Only one post call (comment)
        assert mock_post.call_count == 1
        comment_body = mock_post.call_args.kwargs['json']['comment']['body']
        assert 'Pileus convex' in comment_body

    @patch('requests.get')
    @patch('requests.post')
    def test_unknown_nomenclature_posts_comment(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        """'Unknown' nomenclature is treated as absent."""
        self.collection.nomenclature = 'Unknown'
        self.collection.save()

        _create_social_account(self.user)
        _mock_jwt(mock_get)
        mock_post.return_value = MagicMock()

        resp = self.client.post(self.url)
        assert resp.status_code == 200
        assert 'Comment posted' in resp.json()['message']
        assert mock_post.call_count == 1

    @patch('requests.get')
    @patch('requests.post')
    def test_taxon_not_found_falls_back_to_comment(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        """When taxon lookup returns no results, falls back to comment."""
        _create_social_account(self.user)

        mock_jwt_resp = MagicMock()
        mock_jwt_resp.json.return_value = {'api_token': 'fake-jwt'}
        mock_taxon_resp = MagicMock()
        mock_taxon_resp.json.return_value = {'results': []}
        mock_get.side_effect = [mock_jwt_resp, mock_taxon_resp]

        mock_post.return_value = MagicMock()

        resp = self.client.post(self.url)
        assert resp.status_code == 200
        data = resp.json()
        assert 'Comment posted' in data['message']
        assert 'No taxon found' in data['identification']['warning']

        # Falls back to comment
        assert mock_post.call_count == 1
        comment_body = mock_post.call_args.kwargs['json']['comment']['body']
        assert 'Pileus convex' in comment_body
