#!/usr/bin/env python3
"""
Remove .ann Attachments from the Ingest Database

After the refactor that writes annotations directly to a separate output
database (--output-database), the ingest database (e.g. skol_dev) should
contain no *.ann attachments.  This script scans the ingest database and
removes any *.ann attachment it finds.

Usage:
    python fixes/remove_ann_from_ingest_db.py --dry-run
    python fixes/remove_ann_from_ingest_db.py
    python fixes/remove_ann_from_ingest_db.py --database skol_dev
"""

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config


def remove_ann_attachments(
    database: str,
    dry_run: bool = True,
    verbosity: int = 1,
) -> int:
    """Remove all *.ann attachments from *database*.

    Args:
        database: CouchDB database name to clean.
        dry_run: If True, report what would be removed without changing anything.
        verbosity: 0=quiet, 1=summary, 2=per-document.

    Returns:
        Number of attachments removed (or that would be removed in dry-run).
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
    total_docs = 0
    total_removed = 0

    for doc_id in db:
        try:
            doc = db[doc_id]
        except Exception as exc:
            if verbosity >= 2:
                print(f"  Warning: could not fetch {doc_id}: {exc}", file=sys.stderr)
            continue

        attachments: dict = doc.get('_attachments', {})
        ann_names: List[str] = [
            name for name in attachments if name.endswith('.ann')
        ]
        if not ann_names:
            continue

        total_docs += 1
        for att_name in ann_names:
            total_removed += 1
            if verbosity >= 2:
                action = "Would remove" if dry_run else "Removing"
                print(f"  {action} {doc_id}/{att_name}")
            if not dry_run:
                try:
                    # Re-fetch the doc each iteration — rev changes after deletion
                    current_doc = db[doc_id]
                    db.delete_attachment(current_doc, att_name)
                except Exception as exc:
                    print(
                        f"  ERROR deleting {doc_id}/{att_name}: {exc}",
                        file=sys.stderr,
                    )
                    total_removed -= 1

    if verbosity >= 1:
        if dry_run:
            print(
                f"DRY RUN: would remove {total_removed} .ann attachment(s) "
                f"from {total_docs} document(s) in '{database}'"
            )
        else:
            print(
                f"Removed {total_removed} .ann attachment(s) "
                f"from {total_docs} document(s) in '{database}'"
            )

    return total_removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove *.ann attachments from the ingest database."
    )
    parser.add_argument(
        '--database',
        default=None,
        help='CouchDB database to clean (default: ingest_db_name from env config)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview removals without making changes',
    )
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        default=1,
        help='Verbosity level: 0=quiet, 1=summary, 2=per-document (default: 1)',
    )
    args = parser.parse_args()

    config = get_env_config()
    database = args.database or config.get('ingest_db_name', 'skol_dev')

    if args.dry_run:
        print(f"DRY RUN — scanning '{database}' for .ann attachments...")
    else:
        print(f"Scanning '{database}' for .ann attachments to remove...")

    remove_ann_attachments(
        database=database,
        dry_run=args.dry_run,
        verbosity=args.verbosity,
    )


if __name__ == '__main__':
    main()
