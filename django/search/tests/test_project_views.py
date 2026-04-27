"""
Tests for Project API views.

Run with: pytest search/tests/test_project_views.py -v
"""
import json
import pytest
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from search.models import Collection, Project, CollectionProject, CollectionProjectRemoval


class TestProjectListCreateView(TestCase):
    """Tests for GET/POST /api/projects/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.other = User.objects.create_user("mjones", "m@example.com", "pw")
        self.url = reverse("search:project-list-create")

    def test_unauthenticated_list_returns_200(self) -> None:
        """Project list is public (no auth required)."""
        response = self.client.get(self.url)
        assert response.status_code == 200

    def test_list_returns_all_projects(self) -> None:
        """GET returns all projects site-wide."""
        Project.objects.create(name="Project A", creator=self.user)
        Project.objects.create(name="Project B", creator=self.other)
        response = self.client.get(self.url)
        data = response.json()
        assert data["count"] == 2

    def test_list_search_by_name(self) -> None:
        """?q= filters by name (case-insensitive)."""
        Project.objects.create(name="French Guiana Fungi", creator=self.user)
        Project.objects.create(name="Caribbean Guide", creator=self.other)
        response = self.client.get(self.url + "?q=french")
        data = response.json()
        assert data["count"] == 1
        assert data["projects"][0]["name"] == "French Guiana Fungi"

    def test_list_search_by_creator(self) -> None:
        """?q= also matches creator username."""
        Project.objects.create(name="Project A", creator=self.user)
        Project.objects.create(name="Project B", creator=self.other)
        response = self.client.get(self.url + "?q=jsmith")
        data = response.json()
        assert data["count"] == 1
        assert data["projects"][0]["creator_username"] == "jsmith"

    def test_create_requires_auth(self) -> None:
        """Creating a project requires authentication."""
        response = self.client.post(
            self.url,
            json.dumps({"name": "New Project"}),
            content_type="application/json",
        )
        assert response.status_code == 403

    def test_create_project(self) -> None:
        """Authenticated user can create a project."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.post(
            self.url,
            json.dumps({"name": "French Guiana Fungi"}),
            content_type="application/json",
        )
        assert response.status_code == 201
        data = response.json()
        assert data["slug"] == "french-guiana-fungi"
        assert data["creator_username"] == "jsmith"

    def test_create_project_with_description(self) -> None:
        """Project description is stored."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.post(
            self.url,
            json.dumps({"name": "My Guide", "description": "A nature guide"}),
            content_type="application/json",
        )
        assert response.status_code == 201
        assert response.json()["description"] == "A nature guide"

    def test_create_invalid_name_returns_400(self) -> None:
        """A name with no alphanumeric chars returns 400."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.post(
            self.url,
            json.dumps({"name": "---!!!"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_response_contains_expected_fields(self) -> None:
        """Project list response includes required fields."""
        Project.objects.create(name="Test", creator=self.user)
        response = self.client.get(self.url)
        project = response.json()["projects"][0]
        for field in ["name", "slug", "creator_username", "description",
                      "created_at", "collection_count", "namespaced_slug"]:
            assert field in project, f"Missing field: {field}"

    def test_list_filter_by_collection_id(self) -> None:
        """?collection_id= returns only projects containing that collection."""
        project_a = Project.objects.create(name="Project A", creator=self.user)
        project_b = Project.objects.create(name="Project B", creator=self.user)
        coll = Collection.objects.create(owner=self.user, name="Test Coll")
        CollectionProject.objects.create(
            collection=coll, project=project_a, added_by=self.user
        )

        response = self.client.get(
            self.url + f"?collection_id={coll.collection_id}"
        )
        data = response.json()
        slugs = [p["namespaced_slug"] for p in data["projects"]]
        assert project_a.slug in slugs[0]
        assert not any(project_b.slug in s for s in slugs)

    def test_namespaced_slug_format(self) -> None:
        """namespaced_slug is 'username/slug'."""
        Project.objects.create(name="French Guiana Fungi", creator=self.user)
        response = self.client.get(self.url)
        assert response.json()["projects"][0]["namespaced_slug"] == (
            "jsmith/french-guiana-fungi"
        )


class TestProjectDetailView(TestCase):
    """Tests for GET /api/projects/<username>/<slug>/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.project = Project.objects.create(
            name="French Guiana Fungi", creator=self.user
        )
        self.url = reverse(
            "search:project-detail",
            kwargs={"username": "jsmith", "slug": "french-guiana-fungi"},
        )

    def test_get_project_unauthenticated(self) -> None:
        """Anyone can retrieve a project."""
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert response.json()["name"] == "French Guiana Fungi"

    def test_get_nonexistent_returns_404(self) -> None:
        """Non-existent project returns 404."""
        url = reverse(
            "search:project-detail",
            kwargs={"username": "jsmith", "slug": "nonexistent"},
        )
        response = self.client.get(url)
        assert response.status_code == 404

    def test_get_includes_collection_count(self) -> None:
        """Detail response includes current collection count."""
        coll = Collection.objects.create(owner=self.user, name="C1")
        CollectionProject.objects.create(
            collection=coll, project=self.project, added_by=self.user
        )
        response = self.client.get(self.url)
        assert response.json()["collection_count"] == 1

    def test_patch_notes_authenticated(self) -> None:
        """Authenticated user can update project notes."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.patch(
            self.url,
            data=json.dumps({"notes": "Some field notes."}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.json()["notes"] == "Some field notes."
        self.project.refresh_from_db()
        assert self.project.notes == "Some field notes."

    def test_patch_description_authenticated(self) -> None:
        """Authenticated user can update project description."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.patch(
            self.url,
            data=json.dumps({"description": "A longer description."}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert response.json()["description"] == "A longer description."

    def test_patch_unauthenticated_returns_403(self) -> None:
        """Unauthenticated PATCH returns 403."""
        response = self.client.patch(
            self.url,
            data=json.dumps({"notes": "sneaky"}),
            content_type="application/json",
        )
        assert response.status_code == 403

    def test_patch_ignores_unknown_fields(self) -> None:
        """PATCH with unknown fields returns 200 and ignores unknown keys."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.patch(
            self.url,
            data=json.dumps({"notes": "ok", "creator": "hacker"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        self.project.refresh_from_db()
        assert self.project.creator == self.user  # unchanged


class TestProjectCollectionAddRemoveView(TestCase):
    """Tests for POST/DELETE /api/projects/<username>/<slug>/collections/<collection_id>/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.other = User.objects.create_user("mjones", "m@example.com", "pw")
        self.project = Project.objects.create(
            name="Field Guide", creator=self.user
        )
        self.collection = Collection.objects.create(
            owner=self.user, name="My Collection"
        )
        self.url = reverse(
            "search:project-collection-membership",
            kwargs={
                "username": "jsmith",
                "slug": "field-guide",
                "collection_id": self.collection.collection_id,
            },
        )

    # --- Add ---

    def test_add_requires_auth(self) -> None:
        """Adding a collection requires authentication."""
        response = self.client.post(self.url)
        assert response.status_code == 403

    def test_add_collection_to_project(self) -> None:
        """Authenticated user can add a collection to a project."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.post(self.url)
        assert response.status_code == 201
        assert CollectionProject.objects.filter(
            collection=self.collection, project=self.project
        ).exists()

    def test_any_user_can_add(self) -> None:
        """Any authenticated user can add any collection to any project."""
        self.client.login(username="mjones", password="pw")
        response = self.client.post(self.url)
        assert response.status_code == 201
        cp = CollectionProject.objects.get(
            collection=self.collection, project=self.project
        )
        assert cp.added_by == self.other

    def test_add_already_member_returns_200(self) -> None:
        """Adding a collection already in the project returns 200 (idempotent)."""
        CollectionProject.objects.create(
            collection=self.collection, project=self.project, added_by=self.user
        )
        self.client.login(username="jsmith", password="pw")
        response = self.client.post(self.url)
        assert response.status_code == 200

    def test_add_nonexistent_collection_returns_404(self) -> None:
        """Adding a non-existent collection returns 404."""
        self.client.login(username="jsmith", password="pw")
        url = reverse(
            "search:project-collection-membership",
            kwargs={
                "username": "jsmith",
                "slug": "field-guide",
                "collection_id": 999999999,
            },
        )
        response = self.client.post(url)
        assert response.status_code == 404

    # --- Remove ---

    def test_remove_requires_auth(self) -> None:
        """Removing a collection requires authentication."""
        response = self.client.delete(self.url)
        assert response.status_code == 403

    def test_remove_collection_from_project(self) -> None:
        """Authenticated user can remove a collection from a project."""
        CollectionProject.objects.create(
            collection=self.collection, project=self.project, added_by=self.user
        )
        self.client.login(username="jsmith", password="pw")
        response = self.client.delete(self.url)
        assert response.status_code == 200
        assert not CollectionProject.objects.filter(
            collection=self.collection, project=self.project
        ).exists()

    def test_remove_creates_audit_log(self) -> None:
        """Removing a collection creates a CollectionProjectRemoval record."""
        CollectionProject.objects.create(
            collection=self.collection, project=self.project, added_by=self.user
        )
        self.client.login(username="mjones", password="pw")
        self.client.delete(self.url)
        assert CollectionProjectRemoval.objects.filter(
            collection=self.collection,
            project=self.project,
            removed_by=self.other,
        ).exists()

    def test_remove_audit_log_records_original_adder(self) -> None:
        """Removal log captures who originally added the collection."""
        CollectionProject.objects.create(
            collection=self.collection, project=self.project, added_by=self.user
        )
        self.client.login(username="mjones", password="pw")
        self.client.delete(self.url)
        removal = CollectionProjectRemoval.objects.get(
            collection=self.collection, project=self.project
        )
        assert removal.original_added_by == self.user
        assert removal.removed_by == self.other

    def test_any_user_can_remove(self) -> None:
        """Any authenticated user can remove any collection from any project."""
        CollectionProject.objects.create(
            collection=self.collection, project=self.project, added_by=self.user
        )
        self.client.login(username="mjones", password="pw")
        response = self.client.delete(self.url)
        assert response.status_code == 200

    def test_remove_not_member_returns_404(self) -> None:
        """Removing a collection not in the project returns 404."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.delete(self.url)
        assert response.status_code == 404


class TestCollectionListProjectFilter(TestCase):
    """Tests for ?project= filter on GET /api/collections/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.other = User.objects.create_user("mjones", "m@example.com", "pw")
        self.project_a = Project.objects.create(name="Guide A", creator=self.user)
        self.project_b = Project.objects.create(name="Guide B", creator=self.user)
        self.c1 = Collection.objects.create(owner=self.user, name="C1")
        self.c2 = Collection.objects.create(owner=self.user, name="C2")
        self.c3 = Collection.objects.create(owner=self.other, name="C3")
        CollectionProject.objects.create(
            collection=self.c1, project=self.project_a, added_by=self.user
        )
        CollectionProject.objects.create(
            collection=self.c2, project=self.project_b, added_by=self.user
        )
        CollectionProject.objects.create(
            collection=self.c3, project=self.project_a, added_by=self.other
        )
        self.url = reverse("search:collection-list-create")

    def test_project_filter_returns_matching_collections(self) -> None:
        """?project=jsmith/guide-a returns only collections in that project."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.get(self.url + "?project=jsmith/guide-a")
        data = response.json()
        # c1 and c3 are in guide-a; but collection list only returns owner's collections
        # The filter is OR across project memberships
        names = {c["name"] for c in data["collections"]}
        assert "C1" in names

    def test_multiple_project_filter_or_semantics(self) -> None:
        """?project=A&project=B returns collections in A OR B."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.get(
            self.url + "?project=jsmith/guide-a&project=jsmith/guide-b"
        )
        data = response.json()
        names = {c["name"] for c in data["collections"]}
        assert "C1" in names
        assert "C2" in names

    def test_project_filter_unknown_project_returns_empty(self) -> None:
        """?project= with a non-existent project returns no collections."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.get(self.url + "?project=jsmith/does-not-exist")
        data = response.json()
        assert data["count"] == 0


class TestProjectExportView(TestCase):
    """Tests for GET /api/projects/<username>/<slug>/export/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.project = Project.objects.create(name="Field Guide", creator=self.user)
        self.collection = Collection.objects.create(
            owner=self.user, name="My Collection", description="desc text"
        )
        CollectionProject.objects.create(
            collection=self.collection, project=self.project, added_by=self.user
        )
        self.url = reverse(
            "search:project-export",
            kwargs={"username": "jsmith", "slug": "field-guide"},
        )

    def test_export_requires_auth(self) -> None:
        """Export requires authentication."""
        response = self.client.get(self.url)
        assert response.status_code == 403

    def test_export_returns_zip(self) -> None:
        """Export returns a ZIP file."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert response["Content-Type"] == "application/zip"

    def test_export_zip_contains_project_json(self) -> None:
        """Exported ZIP contains project.json with metadata."""
        import io
        import zipfile

        self.client.login(username="jsmith", password="pw")
        response = self.client.get(self.url)
        buf = io.BytesIO(response.content)
        with zipfile.ZipFile(buf) as zf:
            assert "project.json" in zf.namelist()
            project_data = json.loads(zf.read("project.json"))
        assert project_data["name"] == "Field Guide"
        assert project_data["slug"] == "field-guide"
        assert project_data["creator_username"] == "jsmith"

    def test_export_zip_contains_collection(self) -> None:
        """Exported ZIP contains collection JSON files."""
        import io
        import zipfile

        self.client.login(username="jsmith", password="pw")
        response = self.client.get(self.url)
        buf = io.BytesIO(response.content)
        with zipfile.ZipFile(buf) as zf:
            collection_files = [
                n for n in zf.namelist() if n.startswith("collections/")
            ]
        assert len(collection_files) == 1

    def test_export_nonexistent_project_returns_404(self) -> None:
        """Export of a non-existent project returns 404."""
        self.client.login(username="jsmith", password="pw")
        url = reverse(
            "search:project-export",
            kwargs={"username": "jsmith", "slug": "nonexistent"},
        )
        response = self.client.get(url)
        assert response.status_code == 404


class TestProjectDetailPageView(TestCase):
    """Tests for the project HTML detail page view."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        Project.objects.create(name="Test Project", creator=self.user)
        self.url = reverse(
            "project-detail-page",
            kwargs={"username": "jsmith", "slug": "test-project"},
        )

    def test_page_accessible_unauthenticated(self) -> None:
        """Project detail page is public."""
        response = self.client.get(self.url)
        assert response.status_code == 200
        assert b"project_username" in response.content or b"jsmith" in response.content

    def test_page_accessible_authenticated(self) -> None:
        """Project detail page works when logged in."""
        self.client.login(username="jsmith", password="pw")
        response = self.client.get(self.url)
        assert response.status_code == 200
