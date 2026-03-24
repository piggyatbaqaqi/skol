"""URL patterns for iNaturalist OAuth2 provider."""

from django.urls import path

from . import views

urlpatterns = [
    path("inaturalist/login/", views.oauth2_login, name="inaturalist_login"),
    path(
        "inaturalist/login/callback/",
        views.oauth2_callback,
        name="inaturalist_callback",
    ),
]
