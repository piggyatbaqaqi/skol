#!/usr/bin/env python3
"""
Remove Bare Annotation Documents from an *_ann Database

Documents written by the old two-stage annotation pipeline contained only
_id, _rev, and _attachments — no metadata (title, doi, authors, etc.).
The current pipeline copies source metadata at write time, so these bare
docs are stale artifacts and safe to delete.

A document is considered "bare" if its only keys (ignoring CouchDB internals
that start with '_') are none — i.e. the document body has no application
fields beyond _id, _rev, and optionally _attachments.

Usage:
    python fixes/remove_bare_ann_docs.py --database skol_exp_taxpub_v1_ann \\
        --dry-run
    python fixes/remove_bare_ann_docs.py --database skol_exp_taxpub_v1_ann
    python fixes/remove_bare_ann_docs.py --experiment taxpub_v1 --dry-run
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config


def _is_bare(doc: Dict[str, Any]) -> bool:
    """Return True if the document has no application fields.

    Application fields are any keys that do not start with '_'.
    A bare doc looks like {"_id": "...", "_rev": "...", "_attachments": {...}}.
    """
    return not any(k for k in doc if not k.startswith('_'))


def remove_bare_docs(
    database: str,
    dry_run: bool = True,
    verbosity: int = 1,
) -> int:
    """Delete bare documents from *database*.

    Args:
        database: CouchDB database name to clean.
        dry_run: If True, report without deleting.
        verbosity: 0=quiet, 1=summary, 2=per-document.

    Returns:
        Number of documents deleted (or that would be deleted).
    """
    import couchdb

    config = get_env_config()
    couchdb_url = f"http://{config['couchdb_host']}"
    server = couchdb.Server(couchdb_url)
    if config['couchdb_username'] and config['couchdb_password']:
        server.resource.credentials = (
            config['couchdb_username'],
            config['couchdb_password'],
        )

    if database not in server:
        print(f"Database not found: {database}", file=sys.stderr)
        sys.exit(1)

    db = server[database]
    total_deleted = 0

    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue

        try:
            doc = db[doc_id]
        except Exception as exc:
            if verbosity >= 2:
                print(
                    f"  Warning: could not fetch {doc_id}: {exc}",
                    file=sys.stderr,
                )
            continue

        if not _is_bare(doc):
            continue

        n_attachments = len(doc.get('_attachments', {}))
        total_deleted += 1
        if verbosity >= 2:
            action = "Would delete" if dry_run else "Deleting"
            print(
                f"  {action} {doc_id} "
                f"({n_attachments} attachment(s), no metadata)"
            )
        if not dry_run:
            try:
                db.delete(doc)
            except Exception as exc:
                print(
                    f"  ERROR deleting {doc_id}: {exc}",
                    file=sys.stderr,
                )
                total_deleted -= 1

    if verbosity >= 1:
        if dry_run:
            print(
                f"DRY RUN: would delete {total_deleted} bare document(s) "
                f"from '{database}'"
            )
        else:
            print(
                f"Deleted {total_deleted} bare document(s) from '{database}'"
            )

    return total_deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete bare annotation docs (no metadata fields) "
            "from an *_ann database."
        )
    )
    parser.add_argument(
        '--database',
        default=None,
        help=(
            'CouchDB annotation database to clean '
            '(default: annotations_db_name from env/experiment config)'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview deletions without making changes',
    )
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        default=1,
        help='Verbosity level: 0=quiet, 1=summary, 2=per-document (default: 1)',
    )
    args = parser.parse_args()

    config = get_env_config()
    database = args.database or config.get('annotations_db_name')
    if not database:
        parser.error(
            "--database is required (or set via --experiment annotations_db)"
        )

    if args.dry_run:
        print(f"DRY RUN — scanning '{database}' for bare documents...")
    else:
        print(f"Scanning '{database}' for bare documents to delete...")

    remove_bare_docs(
        database=database,
        dry_run=args.dry_run,
        verbosity=args.verbosity,
    )


if __name__ == '__main__':
    main()
