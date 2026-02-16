"""
CouchDB synchronization for user collections.

Syncs Django Collection models to CouchDB for inclusion in semantic search.
Also maintains history records in a separate database.
"""
import logging
import os
from datetime import datetime
from typing import Optional

import couchdb

logger = logging.getLogger(__name__)


def get_couchdb_config():
    """Get CouchDB configuration from environment."""
    return {
        'url': os.environ.get('COUCHDB_URL', 'http://localhost:5984'),
        'username': os.environ.get('COUCHDB_USER', 'admin'),
        'password': os.environ.get('COUCHDB_PASSWORD', ''),
        'collections_db': os.environ.get(
            'COLLECTIONS_DB_NAME', 'skol_collections_dev'
        ),
        'history_db': os.environ.get(
            'COLLECTIONS_HISTORY_DB_NAME', 'skol_collections_history_dev'
        ),
        'comments_db': os.environ.get(
            'COMMENTS_DB_NAME', 'skol_comments_dev'
        ),
    }


def get_couchdb_server():
    """Get authenticated CouchDB server connection."""
    config = get_couchdb_config()
    server = couchdb.Server(config['url'])
    if config['username'] and config['password']:
        server.resource.credentials = (config['username'], config['password'])
    return server


def get_or_create_database(server, db_name: str):
    """Get or create a CouchDB database."""
    if db_name not in server:
        server.create(db_name)
        logger.info(f"Created CouchDB database: {db_name}")
    return server[db_name]


def sync_collection_to_couchdb(collection_id: int) -> bool:
    """
    Sync a Django collection to CouchDB (idempotent upsert).

    Creates or updates the collection document in skol_collections_dev.
    Also creates a history record if description or nomenclature changed.

    Args:
        collection_id: The collection_id (9-digit) to sync

    Returns:
        True if sync succeeded, False otherwise
    """
    # Import here to avoid circular imports
    from .models import Collection

    try:
        collection = Collection.objects.select_related('owner').get(
            collection_id=collection_id
        )
    except Collection.DoesNotExist:
        logger.warning(f"Collection {collection_id} not found for sync")
        return False

    config = get_couchdb_config()
    doc_id = f"collection_{collection_id}"

    try:
        server = get_couchdb_server()
        db = get_or_create_database(server, config['collections_db'])

        # Build the document
        now = datetime.utcnow().isoformat() + 'Z'
        doc = {
            '_id': doc_id,
            'type': 'collection',

            # Taxa-compatible fields for unified search
            'taxon': collection.nomenclature or '',
            'description': collection.description or '',

            # Collection metadata
            'collection': {
                'collection_id': collection.collection_id,
                'name': collection.name,
                'notes': collection.notes,
                'embargo_until': (
                    collection.embargo_until.isoformat()
                    if collection.embargo_until else None
                ),
                'hidden': collection.hidden,
            },

            # Source metadata (analogous to ingest in taxa)
            'ingest': {
                '_id': doc_id,
                'db_name': config['collections_db'],
                'source_type': 'user_collection',
            },

            # Owner info
            'owner': {
                'user_id': collection.owner.id,
                'username': collection.owner.username,
            },

            'created_at': collection.created_at.isoformat(),
            'updated_at': collection.updated_at.isoformat(),
            'django_synced_at': now,
        }

        # Check if document exists and track changes
        old_doc = None
        description_changed = False
        nomenclature_changed = False

        if doc_id in db:
            old_doc = db[doc_id]
            doc['_rev'] = old_doc['_rev']

            # Check for changes that should trigger history
            old_description = old_doc.get('description', '')
            old_nomenclature = old_doc.get('taxon', '')

            description_changed = old_description != doc['description']
            nomenclature_changed = old_nomenclature != doc['taxon']

        # Save the document
        db.save(doc)
        logger.info(f"Synced collection {collection_id} to CouchDB")

        # Create history record if content changed
        if old_doc is None:
            # New collection - create initial history
            create_history_record(collection, 'create')
        elif description_changed or nomenclature_changed:
            # Content changed - create update history
            create_history_record(collection, 'update')

        return True

    except Exception as e:
        logger.error(f"Failed to sync collection {collection_id}: {e}")
        return False


def create_history_record(collection, change_type: str = 'update') -> bool:
    """
    Create a history snapshot in skol_collections_history_dev.

    Args:
        collection: Django Collection model instance
        change_type: 'create' or 'update'

    Returns:
        True if history record created, False otherwise
    """
    config = get_couchdb_config()
    now = datetime.utcnow()
    timestamp = now.strftime('%Y%m%dT%H%M%S')
    doc_id = f"collection_{collection.collection_id}_{timestamp}"

    try:
        server = get_couchdb_server()
        db = get_or_create_database(server, config['history_db'])

        doc = {
            '_id': doc_id,
            'type': 'collection_history',

            'collection_id': collection.collection_id,
            'description': collection.description or '',
            'nomenclature': collection.nomenclature or '',
            'name': collection.name,

            'owner': {
                'user_id': collection.owner.id,
                'username': collection.owner.username,
            },

            'changed_at': now.isoformat() + 'Z',
            'change_type': change_type,
        }

        db.save(doc)
        logger.info(
            f"Created history record for collection {collection.collection_id}: "
            f"{change_type}"
        )
        return True

    except Exception as e:
        logger.error(
            f"Failed to create history for collection {collection.collection_id}: {e}"
        )
        return False


def delete_collection_from_couchdb(collection_id: int) -> bool:
    """
    Delete a collection from CouchDB.

    Args:
        collection_id: The collection_id (9-digit) to delete

    Returns:
        True if deletion succeeded or document didn't exist, False otherwise
    """
    config = get_couchdb_config()
    doc_id = f"collection_{collection_id}"

    try:
        server = get_couchdb_server()

        # Check if database exists
        if config['collections_db'] not in server:
            return True

        db = server[config['collections_db']]

        # Check if document exists
        if doc_id not in db:
            logger.info(f"Collection {collection_id} not in CouchDB, nothing to delete")
            return True

        # Delete the document
        doc = db[doc_id]
        db.delete(doc)
        logger.info(f"Deleted collection {collection_id} from CouchDB")
        return True

    except Exception as e:
        logger.error(f"Failed to delete collection {collection_id}: {e}")
        return False


def sync_all_collections() -> dict:
    """
    Sync all collections to CouchDB.

    Returns:
        Dict with 'synced', 'failed', and 'total' counts
    """
    from .models import Collection

    collections = Collection.objects.all()
    total = collections.count()
    synced = 0
    failed = 0

    for collection in collections:
        if sync_collection_to_couchdb(collection.collection_id):
            synced += 1
        else:
            failed += 1

    return {
        'synced': synced,
        'failed': failed,
        'total': total,
    }
