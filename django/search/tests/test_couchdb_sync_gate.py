"""
Tests for the COUCHDB_SYNC_ENABLED switch on the Collection -> CouchDB
sync.

Every test that saved a Collection synced it, through the post_save
signal, into the developer's real local CouchDB: 5,580 test documents
in skol_collections_dev by 2026-11-17 (Trello KosomUsu).  The sync now
honours settings.COUCHDB_SYNC_ENABLED (default True), and the test
runner turns it off.

Run with: python manage.py test search.tests.test_couchdb_sync_gate
"""
import unittest
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from search import couchdb_sync
from search.models import Collection

SERVER = 'search.couchdb_sync.get_couchdb_server'


class TestCouchdbSyncGate(TestCase):

    def setUp(self) -> None:
        self.user = User.objects.create_user('gate', 'g@example.com', 'pw')

    @unittest.expectedFailure
    @override_settings(COUCHDB_SYNC_ENABLED=False)
    def test_save_does_not_sync_when_disabled(self) -> None:
        with patch(SERVER) as server:
            Collection.objects.create(owner=self.user, name='Quiet')
        server.assert_not_called()

    @unittest.expectedFailure
    @override_settings(COUCHDB_SYNC_ENABLED=False)
    def test_delete_does_not_sync_when_disabled(self) -> None:
        with patch(SERVER) as server:
            collection = Collection.objects.create(owner=self.user,
                                                   name='Quiet')
            collection.delete()
        server.assert_not_called()

    @unittest.expectedFailure
    @override_settings(COUCHDB_SYNC_ENABLED=False)
    def test_direct_sync_reports_not_synced_when_disabled(self) -> None:
        with patch(SERVER) as server:
            collection = Collection.objects.create(owner=self.user,
                                                   name='Quiet')
            synced = couchdb_sync.sync_collection_to_couchdb(
                collection.collection_id)
        self.assertFalse(synced)
        server.assert_not_called()

    @override_settings(COUCHDB_SYNC_ENABLED=True)
    def test_save_syncs_when_enabled(self) -> None:
        """Already passes: enabled is today's behaviour."""
        with patch(SERVER, return_value=MagicMock()) as server:
            Collection.objects.create(owner=self.user, name='Synced')
        server.assert_called()

    @unittest.expectedFailure
    def test_test_runner_disables_the_sync(self) -> None:
        self.assertIs(settings.COUCHDB_SYNC_ENABLED, False)
