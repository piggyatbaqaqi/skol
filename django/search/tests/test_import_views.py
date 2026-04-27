"""
Tests for POST /api/import/ (ZIP import endpoint).

Run with: pytest search/tests/test_import_views.py -v
"""
import io
import json
import zipfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from search.models import Collection, Project, CollectionProject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zip_bytes(files: dict) -> bytes:
    """Build an in-memory ZIP from {filename: content_str_or_bytes}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode()
            zf.writestr(name, content)
    return buf.getvalue()


def _project_zip(
    name: str = 'Test Project',
    slug: str = 'test-project',
    creator_username: str = 'origuser',
    description: str = '',
    notes: str = '',
    collections: list | None = None,
) -> bytes:
    """Minimal valid project export ZIP."""
    project_data = {
        'name': name,
        'slug': slug,
        'namespaced_slug': f'{creator_username}/{slug}',
        'creator_username': creator_username,
        'description': description,
        'notes': notes,
        'created_at': '2026-01-01T00:00:00Z',
        'current_memberships': [
            {
                'collection_id': c['collection_id'],
                'collection_name': c.get('name', ''),
                'added_by': creator_username,
                'added_at': '2026-01-02T00:00:00Z',
            }
            for c in (collections or [])
        ],
        'removal_log': [],
    }
    files = {'project.json': json.dumps(project_data)}
    for coll in (collections or []):
        cid = coll['collection_id']
        files[f'collections/{cid}.json'] = json.dumps(coll)
    return _zip_bytes(files)


def _uploaded(data: bytes, filename: str = 'export.zip') -> SimpleUploadedFile:
    return SimpleUploadedFile(filename, data, content_type='application/zip')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImportView(TestCase):
    """Tests for POST /api/import/."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('importer', 'i@example.com', 'pw')
        self.url = reverse('search:import')

    # -- authentication -------------------------------------------------------

    def test_requires_auth(self) -> None:
        """Unauthenticated POST returns 403."""
        response = self.client.post(
            self.url, {'file': _uploaded(_project_zip())}
        )
        assert response.status_code == 403

    # -- input validation -----------------------------------------------------

    def test_missing_file_returns_400(self) -> None:
        """POST with no file returns 400."""
        self.client.login(username='importer', password='pw')
        response = self.client.post(self.url, {})
        assert response.status_code == 400

    def test_non_zip_returns_400(self) -> None:
        """Uploading a non-ZIP file returns 400."""
        self.client.login(username='importer', password='pw')
        bad_file = SimpleUploadedFile('bad.zip', b'not a zip', 'application/zip')
        response = self.client.post(self.url, {'file': bad_file})
        assert response.status_code == 400

    def test_unknown_format_returns_400(self) -> None:
        """ZIP without project.json or user.json returns 400."""
        self.client.login(username='importer', password='pw')
        data = _zip_bytes({'README.txt': 'hello'})
        response = self.client.post(self.url, {'file': _uploaded(data)})
        assert response.status_code == 400
        assert 'type' in response.json() or 'detail' in response.json()

    # -- project import -------------------------------------------------------

    def test_import_project_returns_200_with_type(self) -> None:
        """Valid project ZIP returns 200 with type=project."""
        self.client.login(username='importer', password='pw')
        response = self.client.post(
            self.url, {'file': _uploaded(_project_zip())}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['type'] == 'project'
        assert 'project_name' in data

    def test_import_project_creates_project_record(self) -> None:
        """Importing creates a Project row."""
        self.client.login(username='importer', password='pw')
        self.client.post(self.url, {'file': _uploaded(_project_zip(
            name='My Import', slug='my-import', creator_username='nobody',
        ))})
        assert Project.objects.filter(name='My Import').exists()

    def test_import_project_importing_user_is_creator_when_original_absent(self) -> None:
        """Creator defaults to importing user when original username not found."""
        self.client.login(username='importer', password='pw')
        self.client.post(self.url, {'file': _uploaded(_project_zip(
            creator_username='ghost',
        ))})
        project = Project.objects.get(name='Test Project')
        assert project.creator == self.user

    def test_import_project_uses_original_creator_when_present(self) -> None:
        """If original creator exists on this instance, they become the creator."""
        orig = User.objects.create_user('origuser', 'o@example.com', 'pw')
        self.client.login(username='importer', password='pw')
        self.client.post(self.url, {'file': _uploaded(_project_zip(
            creator_username='origuser',
        ))})
        project = Project.objects.get(name='Test Project')
        assert project.creator == orig

    def test_import_project_preserves_description_and_notes(self) -> None:
        """description and notes are copied from project.json."""
        self.client.login(username='importer', password='pw')
        self.client.post(self.url, {'file': _uploaded(_project_zip(
            description='A field guide.', notes='Started 2026.',
        ))})
        project = Project.objects.get(name='Test Project')
        assert project.description == 'A field guide.'
        assert project.notes == 'Started 2026.'

    def test_reimport_updates_existing_project(self) -> None:
        """Re-importing the same creator/slug updates the existing project record."""
        self.client.login(username='importer', password='pw')
        self.client.post(self.url, {'file': _uploaded(_project_zip(
            description='Original desc.', notes='Original notes.',
        ))})
        project = Project.objects.get(name='Test Project', creator=self.user)
        original_pk = project.pk

        # Re-import with updated metadata.
        self.client.post(self.url, {'file': _uploaded(_project_zip(
            description='Updated desc.', notes='Updated notes.',
        ))})

        # Still only one project record.
        assert Project.objects.filter(creator=self.user, slug='test-project').count() == 1
        project.refresh_from_db()
        assert project.pk == original_pk
        assert project.description == 'Updated desc.'
        assert project.notes == 'Updated notes.'

    def test_import_project_creates_new_collection(self) -> None:
        """Collections from the ZIP that don't exist locally are created."""
        self.client.login(username='importer', password='pw')
        colls = [{'collection_id': 123456789, 'name': 'Amanita', 'description': 'd',
                   'notes': '', 'nomenclature': '', 'hidden': False,
                   'embargo_until': None, 'search_history': [],
                   'external_identifiers': []}]
        self.client.post(self.url, {'file': _uploaded(_project_zip(collections=colls))})
        assert Collection.objects.filter(collection_id=123456789).exists()
        project = Project.objects.get(name='Test Project')
        assert CollectionProject.objects.filter(
            project=project, collection__collection_id=123456789
        ).exists()

    def test_reimport_updates_existing_collection(self) -> None:
        """If a collection_id already exists, its data is updated from the ZIP."""
        existing = Collection.objects.create(
            owner=self.user, collection_id=111111111, name='Original Name',
            description='old desc', notes='old notes', nomenclature='Old taxon',
        )
        self.client.login(username='importer', password='pw')
        colls = [{'collection_id': 111111111, 'name': 'New Name',
                   'description': 'new desc', 'notes': 'new notes',
                   'nomenclature': 'New taxon', 'hidden': False,
                   'embargo_until': None, 'search_history': [],
                   'external_identifiers': []}]
        self.client.post(self.url, {'file': _uploaded(_project_zip(collections=colls))})
        existing.refresh_from_db()
        assert existing.name == 'New Name'
        assert existing.description == 'new desc'
        assert existing.notes == 'new notes'
        assert existing.nomenclature == 'New taxon'
        project = Project.objects.get(name='Test Project')
        assert CollectionProject.objects.filter(
            project=project, collection=existing
        ).exists()

    def test_import_response_includes_counts(self) -> None:
        """Response includes collections_created and collections_updated counts."""
        Collection.objects.create(
            owner=self.user, collection_id=222222222, name='Pre-existing',
        )
        self.client.login(username='importer', password='pw')
        colls = [
            {'collection_id': 222222222, 'name': 'Pre-existing', 'description': '',
             'notes': '', 'nomenclature': '', 'hidden': False,
             'embargo_until': None, 'search_history': [], 'external_identifiers': []},
            {'collection_id': 333333333, 'name': 'New One', 'description': '',
             'notes': '', 'nomenclature': '', 'hidden': False,
             'embargo_until': None, 'search_history': [], 'external_identifiers': []},
        ]
        response = self.client.post(
            self.url, {'file': _uploaded(_project_zip(collections=colls))}
        )
        data = response.json()
        assert data['collections_created'] == 1
        assert data['collections_updated'] == 1
