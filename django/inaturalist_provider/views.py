"""iNaturalist OAuth2 views for django-allauth."""

from allauth.socialaccount.adapter import get_adapter
from allauth.socialaccount.providers.oauth2.views import (
    OAuth2Adapter,
    OAuth2CallbackView,
    OAuth2LoginView,
)


class INaturalistOAuth2Adapter(OAuth2Adapter):
    """OAuth2 adapter for iNaturalist."""

    provider_id = "inaturalist"

    authorize_url = "https://www.inaturalist.org/oauth/authorize"
    access_token_url = "https://www.inaturalist.org/oauth/token"
    profile_url = "https://www.inaturalist.org/users/edit.json"

    def complete_login(
        self, request, app, token, **kwargs
    ):
        headers = {"Authorization": f"Bearer {token.token}"}
        resp = (
            get_adapter()
            .get_requests_session()
            .get(self.profile_url, headers=headers)
        )
        resp.raise_for_status()
        extra_data = resp.json()
        return self.get_provider().sociallogin_from_response(request, extra_data)


oauth2_login = OAuth2LoginView.adapter_view(INaturalistOAuth2Adapter)
oauth2_callback = OAuth2CallbackView.adapter_view(INaturalistOAuth2Adapter)
