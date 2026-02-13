"""
CouchDB service layer for collection discussion/comment operations.

Stores threaded comments in a dedicated CouchDB database using
materialized paths for tree ordering.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import couchdb

from .couchdb_sync import get_couchdb_server, get_couchdb_config, get_or_create_database

logger = logging.getLogger(__name__)

# Design document with views for querying comments
DESIGN_DOC = {
    '_id': '_design/comments',
    'language': 'javascript',
    'views': {
        'by_collection_path': {
            'map': (
                'function(doc) {'
                '  if (doc.type === "comment" && doc.collection_id) {'
                '    emit([doc.collection_id, doc.sort_key], null);'
                '  }'
                '}'
            ),
        },
        'count_by_collection': {
            'map': (
                'function(doc) {'
                '  if (doc.type === "comment"'
                '      && doc.collection_id && !doc.deleted) {'
                '    emit(doc.collection_id, 1);'
                '  }'
                '}'
            ),
            'reduce': '_count',
        },
        'children_count': {
            'map': (
                'function(doc) {'
                '  if (doc.type === "comment" && doc.collection_id) {'
                '    emit([doc.collection_id, doc.parent_path], 1);'
                '  }'
                '}'
            ),
            'reduce': '_count',
        },
        'flagged': {
            'map': (
                'function(doc) {'
                '  if (doc.type === "comment" && doc.flagged_by'
                '      && doc.flagged_by.length > 0 && !doc.hidden) {'
                '    emit(doc.collection_id, doc.flagged_by.length);'
                '  }'
                '}'
            ),
        },
    },
}


def get_comments_db():
    """Get the comments CouchDB database, creating if needed."""
    config = get_couchdb_config()
    server = get_couchdb_server()
    return get_or_create_database(server, config['comments_db'])


def ensure_design_docs(db=None):
    """Create or update the _design/comments design document."""
    if db is None:
        db = get_comments_db()

    doc_id = DESIGN_DOC['_id']
    new_views = DESIGN_DOC['views']

    if doc_id in db:
        existing = db[doc_id]
        if existing.get('views') == new_views:
            logger.info("Design doc %s is up to date", doc_id)
            return
        existing['views'] = new_views
        existing['language'] = DESIGN_DOC['language']
        db.save(existing)
        logger.info("Updated design doc %s", doc_id)
    else:
        db.save(dict(DESIGN_DOC))
        logger.info("Created design doc %s", doc_id)


def generate_comment_id(collection_id: int) -> str:
    """Generate a unique comment document ID."""
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    rand = uuid.uuid4().hex[:4]
    return f"comment_{collection_id}_{ts}_{rand}"


def pad_path(path: str) -> str:
    """Convert path like /1/4/ to sort_key like 00001/00004/.

    Each numeric segment is zero-padded to 5 digits for correct
    lexicographic ordering.
    """
    if not path:
        return ''
    parts = path.strip('/').split('/')
    padded = '/'.join(p.zfill(5) for p in parts)
    return f"{padded}/"


def get_next_sibling_number(
    db, collection_id: int, parent_path: str
) -> int:
    """Query children_count view to get next sibling number."""
    result = db.view(
        'comments/children_count',
        key=[collection_id, parent_path],
        reduce=True,
    )
    rows = list(result)
    count = rows[0].value if rows else 0
    return count + 1


def create_comment(
    collection_id: int,
    user_id: int,
    username: str,
    body: str,
    nomenclature: str = '',
    parent_path: str = '',
) -> Dict[str, Any]:
    """Create a new comment document.

    Assigns the next available sibling number under parent_path.
    Retries once on conflict (if another comment claimed the same slot).
    """
    db = get_comments_db()
    max_retries = 2

    for attempt in range(max_retries):
        sibling_num = get_next_sibling_number(
            db, collection_id, parent_path
        )
        path = f"{parent_path}{sibling_num}/"
        sort_key = pad_path(path)
        depth = len(path.strip('/').split('/')) - 1

        now = datetime.now(timezone.utc).isoformat()
        doc_id = generate_comment_id(collection_id)

        doc = {
            '_id': doc_id,
            'type': 'comment',
            'collection_id': collection_id,
            'path': path,
            'depth': depth,
            'parent_path': parent_path,
            'sort_key': sort_key,
            'author': {
                'user_id': user_id,
                'username': username,
            },
            'body': body,
            'nomenclature': nomenclature,
            'created_at': now,
            'updated_at': now,
            'edit_history': [],
            'deleted': False,
            'flagged_by': [],
            'hidden': False,
            'hidden_by': None,
            'hidden_at': None,
        }

        try:
            db.save(doc)
            logger.info(
                "Created comment %s for collection %s",
                doc_id, collection_id,
            )
            return doc
        except couchdb.ResourceConflict:
            if attempt < max_retries - 1:
                logger.warning(
                    "Conflict creating comment, retrying (attempt %d)",
                    attempt + 1,
                )
                continue
            raise

    # Should not reach here, but satisfy type checker
    raise RuntimeError("Failed to create comment after retries")


def get_comments_for_collection(
    collection_id: int,
    include_hidden: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch all comments for a collection, ordered by path."""
    db = get_comments_db()
    result = db.view(
        'comments/by_collection_path',
        startkey=[collection_id, ''],
        endkey=[collection_id, '\ufff0'],
        include_docs=True,
    )

    comments = []
    for row in result:
        doc = dict(row.doc)
        if not include_hidden and doc.get('hidden'):
            continue
        comments.append(doc)

    return comments


def get_comment_count(collection_id: int) -> int:
    """Get count of non-deleted comments for a collection."""
    db = get_comments_db()
    result = db.view(
        'comments/count_by_collection',
        key=collection_id,
        reduce=True,
    )
    rows = list(result)
    return rows[0].value if rows else 0


def get_comment(doc_id: str) -> Dict[str, Any]:
    """Fetch a single comment by ID."""
    db = get_comments_db()
    return dict(db[doc_id])


def update_comment(
    doc_id: str,
    user_id: int,
    body: str,
    nomenclature: str,
) -> Dict[str, Any]:
    """Edit a comment, preserving previous version in edit_history.

    Only the original author can edit. Caller must verify this.
    """
    db = get_comments_db()
    doc = db[doc_id]

    if doc.get('deleted'):
        raise ValueError("Cannot edit a deleted comment")

    # Push current version to edit_history
    history_entry = {
        'body': doc['body'],
        'nomenclature': doc.get('nomenclature', ''),
        'edited_at': doc['updated_at'],
    }
    edit_history = doc.get('edit_history', [])
    edit_history.insert(0, history_entry)

    doc['body'] = body
    doc['nomenclature'] = nomenclature
    doc['updated_at'] = datetime.now(timezone.utc).isoformat()
    doc['edit_history'] = edit_history

    db.save(doc)
    logger.info("Updated comment %s", doc_id)
    return dict(doc)


def soft_delete_comment(doc_id: str, user_id: int) -> Dict[str, Any]:
    """Soft-delete a comment (set deleted=True).

    Caller must verify the user has permission (author, owner, or admin).
    """
    db = get_comments_db()
    doc = db[doc_id]
    doc['deleted'] = True
    doc['updated_at'] = datetime.now(timezone.utc).isoformat()
    db.save(doc)
    logger.info("Soft-deleted comment %s by user %d", doc_id, user_id)
    return dict(doc)


def flag_comment(doc_id: str, user_id: int) -> Dict[str, Any]:
    """Add user to flagged_by array (idempotent)."""
    db = get_comments_db()
    doc = db[doc_id]

    flagged_by = doc.get('flagged_by', [])
    if user_id not in flagged_by:
        flagged_by.append(user_id)
        doc['flagged_by'] = flagged_by
        db.save(doc)
        logger.info(
            "Comment %s flagged by user %d", doc_id, user_id
        )

    return dict(doc)


def hide_comment(doc_id: str, user_id: int) -> Dict[str, Any]:
    """Hide a comment (admin/owner moderation)."""
    db = get_comments_db()
    doc = db[doc_id]
    doc['hidden'] = True
    doc['hidden_by'] = user_id
    doc['hidden_at'] = datetime.now(timezone.utc).isoformat()
    db.save(doc)
    logger.info("Comment %s hidden by user %d", doc_id, user_id)
    return dict(doc)


def unhide_comment(doc_id: str, user_id: int) -> Dict[str, Any]:
    """Unhide a previously hidden comment."""
    db = get_comments_db()
    doc = db[doc_id]
    doc['hidden'] = False
    doc['hidden_by'] = None
    doc['hidden_at'] = None
    db.save(doc)
    logger.info("Comment %s unhidden by user %d", doc_id, user_id)
    return dict(doc)
