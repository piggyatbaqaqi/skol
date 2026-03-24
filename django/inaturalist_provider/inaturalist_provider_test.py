"""Tests for the iNaturalist OAuth2 provider."""

from unittest.mock import MagicMock

from inaturalist_provider.provider import (
    INaturalistAccount,
    INaturalistProvider,
)
from inaturalist_provider.views import INaturalistOAuth2Adapter


class TestINaturalistAccount:
    """Tests for INaturalistAccount."""

    def _make_account(self, extra_data: dict) -> INaturalistAccount:
        mock_account = MagicMock()
        mock_account.extra_data = extra_data
        return INaturalistAccount(mock_account)

    def test_get_profile_url_with_login(self) -> None:
        account = self._make_account({"login": "mycouser"})
        assert account.get_profile_url() == (
            "https://www.inaturalist.org/people/mycouser"
        )

    def test_get_profile_url_without_login(self) -> None:
        account = self._make_account({})
        assert account.get_profile_url() == ""

    def test_get_avatar_url(self) -> None:
        url = "https://static.inaturalist.org/photos/123.jpg"
        account = self._make_account({"icon_url": url})
        assert account.get_avatar_url() == url

    def test_get_avatar_url_missing(self) -> None:
        account = self._make_account({})
        assert account.get_avatar_url() == ""

    def test_to_str_with_login(self) -> None:
        account = self._make_account({"login": "fungifan"})
        assert account.to_str() == "fungifan"

    def test_to_str_without_login(self) -> None:
        mock_account = MagicMock()
        mock_account.extra_data = {}
        mock_account.get_provider.return_value.name = "iNaturalist"
        account = INaturalistAccount(mock_account)
        # Falls back to super().to_str() which returns provider name
        result = account.to_str()
        assert result == "iNaturalist"


class TestINaturalistProvider:
    """Tests for INaturalistProvider."""

    def test_provider_id(self) -> None:
        assert INaturalistProvider.id == "inaturalist"

    def test_provider_name(self) -> None:
        assert INaturalistProvider.name == "iNaturalist"

    def test_extract_uid(self) -> None:
        provider = INaturalistProvider.__new__(INaturalistProvider)
        assert provider.extract_uid({"id": 12345}) == "12345"

    def test_extract_common_fields(self) -> None:
        provider = INaturalistProvider.__new__(INaturalistProvider)
        data = {
            "id": 1,
            "login": "mycouser",
            "name": "Myco User",
            "email": "myco@example.com",
        }
        fields = provider.extract_common_fields(data)
        assert fields == {
            "username": "mycouser",
            "name": "Myco User",
            "email": "myco@example.com",
        }

    def test_extract_common_fields_missing(self) -> None:
        provider = INaturalistProvider.__new__(INaturalistProvider)
        fields = provider.extract_common_fields({"id": 1})
        assert fields == {
            "username": "",
            "name": "",
            "email": "",
        }

    def test_extract_email_addresses(self) -> None:
        provider = INaturalistProvider.__new__(INaturalistProvider)
        data = {"email": "myco@example.com"}
        addresses = provider.extract_email_addresses(data)
        assert len(addresses) == 1
        assert addresses[0].email == "myco@example.com"
        assert addresses[0].verified is True
        assert addresses[0].primary is True

    def test_extract_email_addresses_no_email(self) -> None:
        provider = INaturalistProvider.__new__(INaturalistProvider)
        addresses = provider.extract_email_addresses({})
        assert addresses == []

    def test_get_default_scope(self) -> None:
        provider = INaturalistProvider.__new__(INaturalistProvider)
        assert provider.get_default_scope() == ["login", "write"]


class TestINaturalistOAuth2Adapter:
    """Tests for INaturalistOAuth2Adapter."""

    def test_provider_id(self) -> None:
        assert INaturalistOAuth2Adapter.provider_id == "inaturalist"

    def test_authorize_url(self) -> None:
        assert INaturalistOAuth2Adapter.authorize_url == (
            "https://www.inaturalist.org/oauth/authorize"
        )

    def test_access_token_url(self) -> None:
        assert INaturalistOAuth2Adapter.access_token_url == (
            "https://www.inaturalist.org/oauth/token"
        )

    def test_profile_url(self) -> None:
        assert INaturalistOAuth2Adapter.profile_url == (
            "https://www.inaturalist.org/users/edit.json"
        )
