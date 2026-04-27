"""
Tests for UserSettings default_projects via the UserSettingsView API.

Run with: pytest search/tests/test_project_settings.py -v
"""
import json
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from search.models import Collection, CollectionProject, Project, UserSettings


class TestUserSettingsDefaultProjects(TestCase):
    """Tests for GET/PUT /api/user-settings/ default_project_slugs field."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.project = Project.objects.create(
            name="Field Guide", creator=self.user
        )
        self.url = reverse("search:user-settings")

    def test_get_includes_default_project_slugs(self) -> None:
        """GET returns default_project_slugs list."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert "default_project_slugs" in response.json()

    def test_default_project_slugs_empty_initially(self) -> None:
        """default_project_slugs is [] when no defaults are set."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.get(self.url)
        assert response.json()["default_project_slugs"] == []

    def test_put_sets_default_projects(self) -> None:
        """PUT with default_project_slugs updates the user's default projects."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.put(
            self.url,
            json.dumps({"default_project_slugs": ["jsmith/field-guide"]}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert "jsmith/field-guide" in response.json()["default_project_slugs"]

    def test_put_clears_default_projects(self) -> None:
        """PUT with empty list clears default projects."""
        settings, _ = UserSettings.objects.get_or_create(user=self.user)
        settings.default_projects.add(self.project)

        self.client.login(username="jsmith", password="pw")
        response = self.client.put(
            self.url,
            json.dumps({"default_project_slugs": []}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.json()["default_project_slugs"] == []

    def test_put_unknown_slug_ignored(self) -> None:
        """Unknown slugs in default_project_slugs are silently dropped."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.put(
            self.url,
            json.dumps({
                "default_project_slugs": [
                    "jsmith/field-guide",
                    "jsmith/does-not-exist",
                ]
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        slugs = response.json()["default_project_slugs"]
        assert "jsmith/field-guide" in slugs
        assert "jsmith/does-not-exist" not in slugs

    def test_put_without_default_project_slugs_preserves_existing(self) -> None:
        """PUT without default_project_slugs key leaves existing defaults unchanged."""
        settings, _ = UserSettings.objects.get_or_create(user=self.user)
        settings.default_projects.add(self.project)

        self.client.login(username="jsmith", password="pw")
        response = self.client.put(
            self.url,
            json.dumps({"default_k": 15}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert "jsmith/field-guide" in response.json()["default_project_slugs"]

    def test_new_collection_added_to_default_project(self) -> None:
        """Creating a collection auto-adds it to the user's default projects."""
        settings, _ = UserSettings.objects.get_or_create(user=self.user)
        settings.default_projects.add(self.project)

        collection = Collection.objects.create(
            owner=self.user, name="New Collection"
        )
        assert CollectionProject.objects.filter(
            collection=collection, project=self.project
        ).exists()
