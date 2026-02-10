"""
Management command to sync all Django collections to CouchDB.

Usage:
    python manage.py sync_collections_to_couchdb
    python manage.py sync_collections_to_couchdb --dry-run
    python manage.py sync_collections_to_couchdb --collection-id 123456789
"""
from django.core.management.base import BaseCommand
from search.models import Collection
from search.couchdb_sync import sync_collection_to_couchdb, sync_all_collections


class Command(BaseCommand):
    help = 'Sync all existing collections to CouchDB for search embedding'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be synced without actually syncing'
        )
        parser.add_argument(
            '--collection-id',
            type=int,
            help='Sync a specific collection by its 9-digit ID'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        collection_id = options.get('collection_id')

        if collection_id:
            # Sync a specific collection
            try:
                collection = Collection.objects.get(collection_id=collection_id)
            except Collection.DoesNotExist:
                self.stderr.write(
                    self.style.ERROR(f'Collection {collection_id} not found')
                )
                return

            if dry_run:
                self.stdout.write(
                    f'[DRY RUN] Would sync collection: {collection_id} ({collection.name})'
                )
                return

            success = sync_collection_to_couchdb(collection_id)
            if success:
                self.stdout.write(
                    self.style.SUCCESS(f'Synced collection: {collection_id}')
                )
            else:
                self.stderr.write(
                    self.style.ERROR(f'Failed to sync collection: {collection_id}')
                )
            return

        # Sync all collections
        collections = Collection.objects.all()
        total = collections.count()

        if dry_run:
            self.stdout.write(f'[DRY RUN] Would sync {total} collections:')
            for collection in collections[:10]:
                self.stdout.write(
                    f'  - {collection.collection_id}: {collection.name}'
                )
            if total > 10:
                self.stdout.write(f'  ... and {total - 10} more')
            return

        self.stdout.write(f'Syncing {total} collections to CouchDB...')

        result = sync_all_collections()

        self.stdout.write('')
        self.stdout.write(
            self.style.SUCCESS(f"Synced: {result['synced']}")
        )
        if result['failed'] > 0:
            self.stdout.write(
                self.style.ERROR(f"Failed: {result['failed']}")
            )
        self.stdout.write(f"Total:  {result['total']}")
