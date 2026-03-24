"""Tests for PostInatCommentView."""

from unittest.mock import MagicMock, patch

import pytest
from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from search.models import Collection, ExternalIdentifier, IdentifierType

User = get_user_model()


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
    def test_successful_comment(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        from allauth.socialaccount.models import SocialAccount, SocialToken

        social_account = SocialAccount.objects.create(
            user=self.user,
            provider='inaturalist',
            uid='12345',
        )
        SocialToken.objects.create(
            account=social_account,
            token='fake-oauth-token',
        )

        # Mock JWT exchange
        mock_jwt_resp = MagicMock()
        mock_jwt_resp.json.return_value = {'api_token': 'fake-jwt'}
        mock_get.return_value = mock_jwt_resp

        # Mock comment post
        mock_comment_resp = MagicMock()
        mock_post.return_value = mock_comment_resp

        resp = self.client.post(self.url)
        assert resp.status_code == 200
        data = resp.json()
        assert data['observation_id'] == '336010515'
        assert 'Comment posted' in data['message']

        # Verify comment body includes nomenclature and description
        call_kwargs = mock_post.call_args
        comment_body = call_kwargs.kwargs['json']['comment']['body']
        assert 'Russula emetica' in comment_body
        assert 'Pileus convex' in comment_body

    @patch('requests.get')
    @patch('requests.post')
    def test_comment_without_nomenclature(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        from allauth.socialaccount.models import SocialAccount, SocialToken

        self.collection.nomenclature = ''
        self.collection.save()

        social_account = SocialAccount.objects.create(
            user=self.user,
            provider='inaturalist',
            uid='12345',
        )
        SocialToken.objects.create(
            account=social_account,
            token='fake-oauth-token',
        )

        mock_jwt_resp = MagicMock()
        mock_jwt_resp.json.return_value = {'api_token': 'fake-jwt'}
        mock_get.return_value = mock_jwt_resp

        mock_post.return_value = MagicMock()

        resp = self.client.post(self.url)
        assert resp.status_code == 200

        call_kwargs = mock_post.call_args
        comment_body = call_kwargs.kwargs['json']['comment']['body']
        assert 'Nomenclature' not in comment_body
        assert 'Pileus convex' in comment_body

    @patch('requests.get')
    @patch('requests.post')
    def test_unknown_nomenclature_excluded(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        from allauth.socialaccount.models import SocialAccount, SocialToken

        self.collection.nomenclature = 'Unknown'
        self.collection.save()

        social_account = SocialAccount.objects.create(
            user=self.user,
            provider='inaturalist',
            uid='12345',
        )
        SocialToken.objects.create(
            account=social_account,
            token='fake-oauth-token',
        )

        mock_jwt_resp = MagicMock()
        mock_jwt_resp.json.return_value = {'api_token': 'fake-jwt'}
        mock_get.return_value = mock_jwt_resp

        mock_post.return_value = MagicMock()

        resp = self.client.post(self.url)
        assert resp.status_code == 200

        call_kwargs = mock_post.call_args
        comment_body = call_kwargs.kwargs['json']['comment']['body']
        assert 'Nomenclature' not in comment_body
        assert 'Unknown' not in comment_body
