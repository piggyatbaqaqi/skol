"""
Data export service for GDPR-style "Download My Data" feature.

Gathers all data belonging to a user from Django ORM and CouchDB,
and packages it into an in-memory ZIP file of JSON records.
"""
import io
import json
import logging
import zipfile
from typing import Any, Dict, List, Optional

from django.contrib.auth.models import User

from .couchdb_sync import get_couchdb_config, get_couchdb_server
from .comment_service import (
    get_collection_ids_for_author,
    get_comments_for_collection,
)

logger = logging.getLogger(__name__)


def _serialize_user(user: User) -> Dict[str, Any]:
    """Serialize Django User model to a dict."""
    return {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'date_joined': user.date_joined.isoformat() if user.date_joined else None,
        'last_login': user.last_login.isoformat() if user.last_login else None,
    }


def _serialize_user_settings(user: User) -> Optional[Dict[str, Any]]:
    """Serialize UserSettings for the user, or None if not set."""
    try:
        settings = user.settings
    except Exception:
        return None

    return {
        'default_embargo_days': settings.default_embargo_days,
        'default_embedding': settings.default_embedding,
        'default_k': settings.default_k,
        'results_per_page': settings.results_per_page,
        'nomenclature_limit': settings.nomenclature_limit,
        'feature_taxa_count': settings.feature_taxa_count,
        'feature_top_n': settings.feature_top_n,
        'feature_max_tree_depth': settings.feature_max_tree_depth,
        'feature_min_df': settings.feature_min_df,
        'feature_max_df': settings.feature_max_df,
        'receive_admin_summary': settings.receive_admin_summary,
        'created_at': settings.created_at.isoformat() if settings.created_at else None,
        'updated_at': settings.updated_at.isoformat() if settings.updated_at else None,
    }


def _serialize_collections(user: User) -> List[Dict[str, Any]]:
    """Serialize all collections owned by the user with nested history and identifiers."""
    collections = user.collections.all().order_by('collection_id')
    result = []

    for coll in collections:
        # Search history
        searches = []
        for sh in coll.search_history.all().order_by('-created_at'):
            searches.append({
                'id': sh.id,
                'event_type': sh.event_type,
                'prompt': sh.prompt,
                'embedding_name': sh.embedding_name,
                'k': sh.k,
                'result_references': sh.result_references,
                'result_count': sh.result_count,
                'nomenclature': sh.nomenclature,
                'created_at': sh.created_at.isoformat() if sh.created_at else None,
            })

        # External identifiers
        identifiers = []
        for ei in coll.external_identifiers.all().select_related('identifier_type'):
            identifiers.append({
                'id': ei.id,
                'identifier_type': ei.identifier_type.code,
                'identifier_type_name': ei.identifier_type.name,
                'value': ei.value,
                'fungarium_code': ei.fungarium_code,
                'notes': ei.notes,
                'created_at': ei.created_at.isoformat() if ei.created_at else None,
            })

        result.append({
            'collection_id': coll.collection_id,
            'name': coll.name,
            'description': coll.description,
            'notes': coll.notes,
            'nomenclature': coll.nomenclature,
            'embargo_until': coll.embargo_until.isoformat() if coll.embargo_until else None,
            'hidden': coll.hidden,
            'created_at': coll.created_at.isoformat() if coll.created_at else None,
            'updated_at': coll.updated_at.isoformat() if coll.updated_at else None,
            'search_history': searches,
            'external_identifiers': identifiers,
        })

    return result


def _fetch_couchdb_collection_docs(user: User) -> List[Dict[str, Any]]:
    """Fetch CouchDB collection documents for the user's collections."""
    collection_ids = list(
        user.collections.values_list('collection_id', flat=True)
    )
    if not collection_ids:
        return []

    config = get_couchdb_config()
    try:
        server = get_couchdb_server()
        if config['collections_db'] not in server:
            return []
        db = server[config['collections_db']]
    except Exception:
        logger.warning("Could not connect to CouchDB collections database")
        return []

    docs = []
    for cid in collection_ids:
        doc_id = f"collection_{cid}"
        if doc_id in db:
            doc = dict(db[doc_id])
            doc.pop('_rev', None)
            docs.append(doc)

    return docs


def _fetch_comment_threads(user: User) -> List[Dict[str, Any]]:
    """Fetch full comment threads for collections the user has participated in."""
    try:
        collection_ids = get_collection_ids_for_author(user.id)
    except Exception:
        logger.warning("Could not query CouchDB for user's comment threads")
        return []

    if not collection_ids:
        return []

    threads = []
    for cid in sorted(collection_ids):
        try:
            comments = get_comments_for_collection(cid, include_hidden=False)
            # Strip CouchDB internal fields
            cleaned = []
            for c in comments:
                c.pop('_rev', None)
                cleaned.append(c)
            threads.append({
                'collection_id': cid,
                'comments': cleaned,
            })
        except Exception:
            logger.warning("Could not fetch comments for collection %s", cid)

    return threads


def export_user_data(user: User) -> io.BytesIO:
    """
    Export all user data as an in-memory ZIP file.

    Returns:
        io.BytesIO containing the ZIP archive.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. User record + settings
        user_data = _serialize_user(user)
        settings_data = _serialize_user_settings(user)
        if settings_data is not None:
            user_data['settings'] = settings_data
        zf.writestr('user.json', json.dumps(user_data, indent=2))

        # 2. Collections (one file per collection)
        collections = _serialize_collections(user)
        for coll in collections:
            filename = f"collections/{coll['collection_id']}.json"
            zf.writestr(filename, json.dumps(coll, indent=2))

        # 3. CouchDB collection documents
        couch_docs = _fetch_couchdb_collection_docs(user)
        for doc in couch_docs:
            filename = f"couchdb_collections/{doc['_id']}.json"
            zf.writestr(filename, json.dumps(doc, indent=2))

        # 4. Comment threads
        threads = _fetch_comment_threads(user)
        for thread in threads:
            filename = f"comment_threads/collection_{thread['collection_id']}_comments.json"
            zf.writestr(filename, json.dumps(thread, indent=2))

    buf.seek(0)
    return buf
