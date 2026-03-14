#!/usr/bin/env python3
"""
Remove BioC-related fields from CouchDB documents.

Removes the 'bioc_json' and 'bioc_json_available' fields from all documents
in the specified database. These fields are no longer used after the migration
from BioC to JATS-only ingestion.

Usage:
    python remove_bioc_fields.py --database skol_dev --dry-run
    python remove_bioc_fields.py --database skol_dev
    python remove_bioc_fields.py --database skol_dev --doc-id SPECIFIC_DOC_ID
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config

BIOC_FIELDS = ("bioc_json", "bioc_json_available")


def process_document(db, doc_id, dry_run, verbosity):
    """Remove BioC fields from a single document.

    Returns True if the document was modified.
    """
    try:
        doc = db[doc_id]
    except Exception:
        if verbosity >= 1:
            print(f"  {doc_id}: not found, skipping", file=sys.stderr)
        return False

    fields_found = [f for f in BIOC_FIELDS if f in doc]
    if not fields_found:
        if verbosity >= 3:
            print(f"  {doc_id}: no BioC fields", file=sys.stderr)
        return False

    if dry_run:
        if verbosity >= 1:
            print(
                f"  {doc_id}: would remove {', '.join(fields_found)}",
                file=sys.stderr,
            )
        return True

    for field in fields_found:
        del doc[field]

    db.save(doc)

    if verbosity >= 2:
        print(
            f"  {doc_id}: removed {', '.join(fields_found)}",
            file=sys.stderr,
        )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove bioc_json and bioc_json_available fields "
                    "from CouchDB documents.",
    )
    parser.add_argument(
        "--database",
        type=str,
        required=True,
        help="CouchDB database name (e.g. skol_dev).",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Process only this specific document ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without modifying documents.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N documents.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output.",
    )

    args = parser.parse_args()
    verbosity = 0 if args.quiet else args.verbose
    config = get_env_config()

    import couchdb as couchdb_lib

    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    db = server[args.database]

    if verbosity >= 1:
        action = "DRY RUN: scanning" if args.dry_run else "Removing"
        print(
            f"{action} BioC fields from {args.database}...",
            file=sys.stderr,
        )

    # Collect document IDs
    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        doc_ids = [
            row.id for row in db.view("_all_docs", include_docs=False)
            if not row.id.startswith("_design/")
        ]
        if verbosity >= 1:
            print(
                f"Found {len(doc_ids)} documents in {args.database}",
                file=sys.stderr,
            )

    if args.limit is not None:
        doc_ids = doc_ids[:args.limit]

    modified = 0
    scanned = 0
    for doc_id in doc_ids:
        if process_document(db, doc_id, args.dry_run, verbosity):
            modified += 1
        scanned += 1

    if verbosity >= 1:
        prefix = "Would modify" if args.dry_run else "Modified"
        print(
            f"\n{prefix} {modified} of {scanned} documents.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
