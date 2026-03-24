"""iNaturalist OAuth2 provider for django-allauth."""

from allauth.socialaccount.providers.base import ProviderAccount
from allauth.socialaccount.providers.oauth2.provider import OAuth2Provider


class INaturalistAccount(ProviderAccount):
    """Represent an iNaturalist account."""

    def get_profile_url(self) -> str:
        login: str = self.account.extra_data.get("login", "")
        if login:
            return f"https://www.inaturalist.org/people/{login}"
        return ""

    def get_avatar_url(self) -> str:
        url: str = self.account.extra_data.get("icon_url", "")
        return url

    def to_str(self) -> str:
        login: str = self.account.extra_data.get(
            "login", super().to_str()
        )
        return login


class INaturalistProvider(OAuth2Provider):
    """OAuth2 provider for iNaturalist."""

    id = "inaturalist"
    name = "iNaturalist"
    account_class = INaturalistAccount

    def get_default_scope(self) -> list[str]:
        return ["login"]

    def extract_uid(self, data: dict) -> str:
        return str(data["id"])

    def extract_common_fields(self, data: dict) -> dict:
        return {
            "username": data.get("login", ""),
            "name": data.get("name", ""),
            "email": data.get("email", ""),
        }

    def extract_email_addresses(self, data: dict) -> list:
        from allauth.account.models import EmailAddress

        addresses = []
        email = data.get("email")
        if email:
            addresses.append(
                EmailAddress(
                    email=email,
                    verified=True,
                    primary=True,
                )
            )
        return addresses


provider_classes = [INaturalistProvider]
