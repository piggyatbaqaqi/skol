"""
Tests for Project, CollectionProject, and CollectionProjectRemoval models.

Run with: pytest search/tests/test_project_models.py -v
"""
import pytest
from django.test import TestCase
from django.contrib.auth.models import User
from django.db import IntegrityError

from search.models import Collection, Project, CollectionProject, CollectionProjectRemoval


class TestProjectSlugGeneration(TestCase):
    """Tests for Project.generate_slug() utility."""

    def test_basic_slug(self) -> None:
        """Simple name produces lowercase hyphenated slug."""
        assert Project.generate_slug("French Guiana Fungi") == "french-guiana-fungi"

    def test_punctuation_collapsed(self) -> None:
        """Runs of non-alphanumeric characters collapse to a single hyphen."""
        assert Project.generate_slug("Fungi -- France!") == "fungi-france"

    def test_leading_trailing_stripped(self) -> None:
        """Leading and trailing hyphens are stripped."""
        assert Project.generate_slug("  -- My Project --  ") == "my-project"

    def test_unicode_letters_preserved(self) -> None:
        """Unicode letters are kept (they are alphanumeric)."""
        slug = Project.generate_slug("Champignons de Réunion")
        assert slug  # non-empty
        assert slug[0] != '-'
        assert slug[-1] != '-'

    def test_empty_after_stripping_raises(self) -> None:
        """A name with no alphanumeric content raises ValueError."""
        with self.assertRaises(ValueError):
            Project.generate_slug("---!!!")

    def test_all_punctuation_raises(self) -> None:
        """A name consisting entirely of symbols raises ValueError."""
        with self.assertRaises(ValueError):
            Project.generate_slug("!@#$%")


class TestProjectCreation(TestCase):
    """Tests for Project model creation and constraints."""

    def setUp(self) -> None:
        self.user = User.objects.create_user(
            "jsmith", "jsmith@example.com", "password"
        )
        self.other = User.objects.create_user(
            "mjones", "mjones@example.com", "password"
        )

    def test_create_project(self) -> None:
        """Creating a project sets slug, creator, and name."""
        p = Project.objects.create(
            name="French Guiana Fungi",
            creator=self.user,
        )
        assert p.slug == "french-guiana-fungi"
        assert p.creator == self.user
        assert p.name == "French Guiana Fungi"

    def test_slug_auto_generated(self) -> None:
        """Slug is generated from name if not explicitly provided."""
        p = Project.objects.create(name="Boletales of Europe", creator=self.user)
        assert p.slug == "boletales-of-europe"

    def test_different_users_same_slug_allowed(self) -> None:
        """Two different users can have the same slug."""
        p1 = Project.objects.create(name="My Project", creator=self.user)
        p2 = Project.objects.create(name="My Project", creator=self.other)
        assert p1.slug == p2.slug == "my-project"

    def test_same_user_same_slug_collision_appends_suffix(self) -> None:
        """Creating a duplicate slug for the same user appends -2."""
        p1 = Project.objects.create(name="My Project", creator=self.user)
        p2 = Project.objects.create(name="My Project", creator=self.user)
        assert p1.slug == "my-project"
        assert p2.slug == "my-project-2"

    def test_collision_increments(self) -> None:
        """Third collision appends -3."""
        Project.objects.create(name="My Project", creator=self.user)
        Project.objects.create(name="My Project", creator=self.user)
        p3 = Project.objects.create(name="My Project", creator=self.user)
        assert p3.slug == "my-project-3"

    def test_unique_constraint_on_creator_slug(self) -> None:
        """Direct duplicate (creator, slug) raises IntegrityError."""
        Project.objects.create(name="My Project", creator=self.user, slug="my-project")
        with self.assertRaises(IntegrityError):
            Project.objects.create(
                name="Another", creator=self.user, slug="my-project"
            )

    def test_str(self) -> None:
        """__str__ includes username and slug."""
        p = Project.objects.create(name="French Guiana Fungi", creator=self.user)
        assert "jsmith" in str(p)
        assert "french-guiana-fungi" in str(p)


class TestCollectionProject(TestCase):
    """Tests for CollectionProject through-table."""

    def setUp(self) -> None:
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.other = User.objects.create_user("mjones", "m@example.com", "pw")
        self.project = Project.objects.create(name="Test Project", creator=self.user)
        self.collection = Collection.objects.create(
            owner=self.user, name="My Collection"
        )

    def test_add_collection_to_project(self) -> None:
        """A collection can be added to a project."""
        cp = CollectionProject.objects.create(
            collection=self.collection,
            project=self.project,
            added_by=self.user,
        )
        assert cp.collection == self.collection
        assert cp.project == self.project
        assert cp.added_by == self.user

    def test_different_user_can_add(self) -> None:
        """Any authenticated user can add any collection to any project."""
        cp = CollectionProject.objects.create(
            collection=self.collection,
            project=self.project,
            added_by=self.other,
        )
        assert cp.added_by == self.other

    def test_duplicate_membership_raises(self) -> None:
        """Adding the same collection to the same project twice raises IntegrityError."""
        CollectionProject.objects.create(
            collection=self.collection,
            project=self.project,
            added_by=self.user,
        )
        with self.assertRaises(IntegrityError):
            CollectionProject.objects.create(
                collection=self.collection,
                project=self.project,
                added_by=self.other,
            )

    def test_collection_in_multiple_projects(self) -> None:
        """A collection can belong to multiple projects."""
        other_project = Project.objects.create(name="Other Project", creator=self.other)
        CollectionProject.objects.create(
            collection=self.collection, project=self.project, added_by=self.user
        )
        CollectionProject.objects.create(
            collection=self.collection, project=other_project, added_by=self.user
        )
        assert self.collection.projects.count() == 2

    def test_project_has_multiple_collections(self) -> None:
        """A project can contain multiple collections."""
        c2 = Collection.objects.create(owner=self.other, name="Other Collection")
        CollectionProject.objects.create(
            collection=self.collection, project=self.project, added_by=self.user
        )
        CollectionProject.objects.create(
            collection=c2, project=self.project, added_by=self.other
        )
        assert self.project.collections.count() == 2


class TestCollectionProjectRemoval(TestCase):
    """Tests for CollectionProjectRemoval audit log."""

    def setUp(self) -> None:
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.other = User.objects.create_user("mjones", "m@example.com", "pw")
        self.project = Project.objects.create(name="Test Project", creator=self.user)
        self.collection = Collection.objects.create(
            owner=self.user, name="My Collection"
        )

    def test_removal_log_created(self) -> None:
        """Removing a CollectionProject creates a CollectionProjectRemoval record."""
        cp = CollectionProject.objects.create(
            collection=self.collection,
            project=self.project,
            added_by=self.user,
        )
        # Record the audit log before deleting
        removal = CollectionProjectRemoval.objects.create(
            collection=self.collection,
            project=self.project,
            removed_by=self.other,
            original_added_by=cp.added_by,
            original_added_at=cp.added_at,
        )
        cp.delete()

        assert CollectionProject.objects.filter(
            collection=self.collection, project=self.project
        ).count() == 0
        assert removal.removed_by == self.other
        assert removal.original_added_by == self.user

    def test_removal_log_preserves_added_by(self) -> None:
        """The removal log records who originally added the collection."""
        cp = CollectionProject.objects.create(
            collection=self.collection,
            project=self.project,
            added_by=self.user,
        )
        removal = CollectionProjectRemoval.objects.create(
            collection=self.collection,
            project=self.project,
            removed_by=self.other,
            original_added_by=cp.added_by,
            original_added_at=cp.added_at,
        )
        assert removal.original_added_by == self.user
        assert removal.removed_by == self.other

    def test_multiple_removals_recorded(self) -> None:
        """Adding and removing the same collection multiple times is fully logged."""
        cp = CollectionProject.objects.create(
            collection=self.collection,
            project=self.project,
            added_by=self.user,
        )
        CollectionProjectRemoval.objects.create(
            collection=self.collection,
            project=self.project,
            removed_by=self.other,
            original_added_by=cp.added_by,
            original_added_at=cp.added_at,
        )
        cp.delete()

        # Re-add and remove again
        cp2 = CollectionProject.objects.create(
            collection=self.collection,
            project=self.project,
            added_by=self.other,
        )
        CollectionProjectRemoval.objects.create(
            collection=self.collection,
            project=self.project,
            removed_by=self.user,
            original_added_by=cp2.added_by,
            original_added_at=cp2.added_at,
        )
        cp2.delete()

        assert CollectionProjectRemoval.objects.filter(
            collection=self.collection, project=self.project
        ).count() == 2


class TestProjectDefaultProjects(TestCase):
    """Tests for UserSettings.default_projects M2M."""

    def setUp(self) -> None:
        self.user = User.objects.create_user("jsmith", "j@example.com", "pw")
        self.project = Project.objects.create(name="Field Guide", creator=self.user)

    def test_user_can_set_default_project(self) -> None:
        """UserSettings.default_projects can contain a project."""
        from search.models import UserSettings
        settings, _ = UserSettings.objects.get_or_create(user=self.user)
        settings.default_projects.add(self.project)
        assert settings.default_projects.filter(pk=self.project.pk).exists()

    def test_new_collection_auto_added_to_default_projects(self) -> None:
        """When a collection is created, it is added to the user's default projects."""
        from search.models import UserSettings
        settings, _ = UserSettings.objects.get_or_create(user=self.user)
        settings.default_projects.add(self.project)

        collection = Collection.objects.create(
            owner=self.user, name="New Collection"
        )

        assert CollectionProject.objects.filter(
            collection=collection, project=self.project
        ).exists()

    def test_existing_collection_not_retroactively_added(self) -> None:
        """Setting a default project does not affect existing collections."""
        collection = Collection.objects.create(
            owner=self.user, name="Pre-existing Collection"
        )
        from search.models import UserSettings
        settings, _ = UserSettings.objects.get_or_create(user=self.user)
        settings.default_projects.add(self.project)

        # The pre-existing collection should NOT be in the project
        assert not CollectionProject.objects.filter(
            collection=collection, project=self.project
        ).exists()
