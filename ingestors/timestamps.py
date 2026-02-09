"""
Timestamp utilities for CouchDB documents.

Provides helper functions for adding create_time and modification_time
timestamps to documents before saving to CouchDB.
"""

from datetime import datetime, timezone
from typing import Any, Dict


def get_iso_timestamp() -> str:
    """Get current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def set_timestamps(
    doc: Dict[str, Any],
    is_new: bool = False
) -> Dict[str, Any]:
    """
    Set create_time and/or modification_time timestamps on a document.

    Args:
        doc: CouchDB document dictionary (modified in place)
        is_new: If True, always set create_time (for new documents).
                If False (default), only set create_time if missing.

    Behavior:
        - is_new=True: Sets both create_time and modification_time
        - is_new=False: Sets create_time only if missing, always sets
                        modification_time

    Returns:
        The same document dictionary (for chaining)
    """
    now = get_iso_timestamp()

    # Set create_time if this is a new document or if it's missing
    if is_new or 'create_time' not in doc:
        doc['create_time'] = now

    # Always update modification_time
    doc['modification_time'] = now

    return doc
