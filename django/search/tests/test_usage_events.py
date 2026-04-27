"""
Tests for UsageEvent model and related API endpoints.

Run with: pytest search/tests/test_usage_events.py -v
"""
import json
import zipfile
from unittest.mock import patch

from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse

from search.models import Collection, UsageEvent


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestUsageEventModel(TestCase):
    """Basic model creation and field validation."""

    def setUp(self) -> None:
        self.user = User.objects.create_user('alice', 'a@example.com', 'pw')
        self.collection = Collection.objects.create(
            owner=self.user, name='Test Collection',
        )

    def test_create_description_add_event(self) -> None:
        """Can create a description_add event tied to a collection."""
        evt = UsageEvent.objects.create(
            user=self.user,
            collection=self.collection,
            event_type='description_add',
            payload={'text': 'pileus convex;', 'source': 'vocabulary'},
        )
        assert evt.pk is not None
        assert evt.event_type == 'description_add'
        assert evt.payload['source'] == 'vocabulary'
        assert evt.created_at is not None

    def test_create_event_without_collection(self) -> None:
        """Events not tied to a collection (e.g. export) use null FK."""
        evt = UsageEvent.objects.create(
            user=self.user,
            event_type='user_data_exported',
            payload={},
        )
        assert evt.collection is None

    def test_create_event_without_user(self) -> None:
        """Anonymous events are allowed (user=None)."""
        evt = UsageEvent.objects.create(
            event_type='pdf_viewed',
            payload={'taxa_id': 'abc123'},
        )
        assert evt.user is None

    def test_ordering_most_recent_first(self) -> None:
        """Default ordering is newest first."""
        UsageEvent.objects.create(
            user=self.user, collection=self.collection,
            event_type='description_add', payload={'text': 'first'},
        )
        UsageEvent.objects.create(
            user=self.user, collection=self.collection,
            event_type='description_add', payload={'text': 'second'},
        )
        events = list(UsageEvent.objects.filter(collection=self.collection))
        assert events[0].payload['text'] == 'second'
        assert events[1].payload['text'] == 'first'

    def test_cascade_delete_with_collection(self) -> None:
        """Deleting a collection deletes its events."""
        evt = UsageEvent.objects.create(
            user=self.user, collection=self.collection,
            event_type='description_add', payload={},
        )
        self.collection.delete()
        assert not UsageEvent.objects.filter(pk=evt.pk).exists()

    def test_set_null_on_user_delete(self) -> None:
        """Deleting a user nullifies the user FK on events (no collection)."""
        other = User.objects.create_user('orphan', 'o@example.com', 'pw')
        evt = UsageEvent.objects.create(
            user=other,
            event_type='user_data_exported',
            payload={},
        )
        other.delete()
        evt.refresh_from_db()
        assert evt.user is None


# ---------------------------------------------------------------------------
# POST /api/collections/{id}/description-add/
# ---------------------------------------------------------------------------

class TestDescriptionAddView(TestCase):
    """Tests for the description-add logging endpoint."""

    def setUp(self) -> None:
        self.client = Client()
        self.owner = User.objects.create_user('owner', 'o@example.com', 'pw')
        self.other = User.objects.create_user('other', 'x@example.com', 'pw')
        self.collection = Collection.objects.create(
            owner=self.owner, name='Amanita study',
        )
        self.url = reverse(
            'search:description-add',
            kwargs={'collection_id': self.collection.collection_id},
        )

    def test_requires_auth(self) -> None:
        """Unauthenticated POST returns 403."""
        resp = self.client.post(
            self.url,
            data=json.dumps({'text': 'pileus convex;', 'source': 'vocab'}),
            content_type='application/json',
        )
        assert resp.status_code == 403

    def test_logs_event_for_owner(self) -> None:
        """Owner can log a description-add event."""
        self.client.login(username='owner', password='pw')
        resp = self.client.post(
            self.url,
            data=json.dumps(
                {'text': 'stipe white;', 'source': 'text-features'}
            ),
            content_type='application/json',
        )
        assert resp.status_code == 201
        evt = UsageEvent.objects.get(
            collection=self.collection,
            event_type='description_add',
        )
        assert evt.user == self.owner
        assert evt.payload['text'] == 'stipe white;'
        assert evt.payload['source'] == 'text-features'

    def test_logs_event_for_non_owner(self) -> None:
        """Authenticated non-owner can also log (viewing another's collection)."""
        self.client.login(username='other', password='pw')
        resp = self.client.post(
            self.url,
            data=json.dumps(
                {'text': 'spores globose;', 'source': 'json-features'}
            ),
            content_type='application/json',
        )
        assert resp.status_code == 201
        evt = UsageEvent.objects.get(
            collection=self.collection, event_type='description_add',
        )
        assert evt.user == self.other

    def test_source_defaults_to_unknown(self) -> None:
        """If source is omitted it is stored as 'unknown'."""
        self.client.login(username='owner', password='pw')
        resp = self.client.post(
            self.url,
            data=json.dumps({'text': 'pileus convex;'}),
            content_type='application/json',
        )
        assert resp.status_code == 201
        evt = UsageEvent.objects.get(
            collection=self.collection, event_type='description_add',
        )
        assert evt.payload['source'] == 'unknown'

    def test_missing_text_returns_400(self) -> None:
        """POST without 'text' returns 400."""
        self.client.login(username='owner', password='pw')
        resp = self.client.post(
            self.url,
            data=json.dumps({'source': 'vocabulary'}),
            content_type='application/json',
        )
        assert resp.status_code == 400

    def test_unknown_collection_returns_404(self) -> None:
        """POST to a non-existent collection returns 404."""
        self.client.login(username='owner', password='pw')
        url = reverse(
            'search:description-add',
            kwargs={'collection_id': 9999999},
        )
        resp = self.client.post(
            url,
            data=json.dumps({'text': 'x', 'source': 'vocabulary'}),
            content_type='application/json',
        )
        assert resp.status_code == 404

    def test_valid_sources_accepted(self) -> None:
        """All four tab source values are accepted."""
        self.client.login(username='owner', password='pw')
        sources = (
            'vocabulary', 'text-features', 'json-features', 'metrics',
        )
        for source in sources:
            resp = self.client.post(
                self.url,
                data=json.dumps({'text': 'x;', 'source': source}),
                content_type='application/json',
            )
            assert resp.status_code == 201, f"Failed for source={source}"


# ---------------------------------------------------------------------------
# Export integration
# ---------------------------------------------------------------------------

class TestUsageEventExportIntegration(TestCase):
    """Usage events appear in collection and user-data exports."""

    def setUp(self) -> None:
        self.user = User.objects.create_user('exporter', 'e@example.com', 'pw')
        self.collection = Collection.objects.create(
            owner=self.user, name='Export test',
        )

    _MOCK_SERVER = 'search.export_service.get_couchdb_server'
    _MOCK_COMMENTS = 'search.export_service.get_comments_for_collection'
    _MOCK_IDS = 'search.export_service.get_collection_ids_for_author'

    @patch(_MOCK_SERVER)
    @patch(_MOCK_COMMENTS, return_value=[])
    @patch(_MOCK_IDS, return_value=[])
    def test_description_add_events_in_collection_export(
        self, _mock_ids, _mock_comments, mock_server,
    ) -> None:
        """description_add events appear in the collection's export JSON."""
        mock_server.return_value = {}
        UsageEvent.objects.create(
            user=self.user,
            collection=self.collection,
            event_type='description_add',
            payload={'text': 'pileus convex;', 'source': 'vocabulary'},
        )
        from search.export_service import export_user_data
        buf = export_user_data(self.user)
        zf = zipfile.ZipFile(buf)
        cid = self.collection.collection_id
        coll_json = json.loads(
            zf.read(f'collections/{cid}.json')
        )
        events = coll_json.get('usage_events', [])
        assert len(events) == 1
        assert events[0]['event_type'] == 'description_add'
        assert events[0]['payload']['source'] == 'vocabulary'

    @patch(_MOCK_SERVER)
    @patch(_MOCK_COMMENTS, return_value=[])
    @patch(_MOCK_IDS, return_value=[])
    def test_non_collection_events_in_user_export(
        self, _mock_ids, _mock_comments, mock_server,
    ) -> None:
        """user_data_exported events appear in the top-level user.json."""
        mock_server.return_value = {}
        UsageEvent.objects.create(
            user=self.user,
            event_type='user_data_exported',
            payload={},
        )
        from search.export_service import export_user_data
        buf = export_user_data(self.user)
        zf = zipfile.ZipFile(buf)
        user_json = json.loads(zf.read('user.json'))
        events = user_json.get('usage_events', [])
        assert any(
            e['event_type'] == 'user_data_exported' for e in events
        )


# ---------------------------------------------------------------------------
# Server-side auto-logging
# ---------------------------------------------------------------------------

class TestServerSideEventLogging(TestCase):
    """Server-side views automatically log events without extra client calls."""

    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user('viewer', 'v@example.com', 'pw')

    @patch('search.export_service.get_couchdb_server')
    @patch(
        'search.export_service.get_collection_ids_for_author',
        return_value=[],
    )
    @patch(
        'search.export_service.get_comments_for_collection',
        return_value=[],
    )
    def test_export_my_data_logs_event(
        self, _mock_comments, _mock_ids, mock_server,
    ) -> None:
        """GET /api/export-my-data/ logs a user_data_exported event."""
        mock_server.return_value = {}
        self.client.login(username='viewer', password='pw')
        self.client.get(reverse('search:export-my-data'))
        assert UsageEvent.objects.filter(
            user=self.user, event_type='user_data_exported',
        ).exists()
