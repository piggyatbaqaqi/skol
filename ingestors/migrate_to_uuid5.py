#!/usr/bin/env python3
"""
Migrate CouchDB documents from random UUIDs to UUID5-based IDs.

This script updates all documents in the skol_dev database to use deterministic
UUID5 IDs based on their pdf_url instead of random UUIDs. This makes the
database idempotent and prevents duplicate documents.

The script is idempotent - it can be run multiple times safely.
"""

import argparse
import os
import sys
from typing import Optional
from uuid import uuid5, NAMESPACE_URL

import couchdb


def is_uuid5_id(doc_id: str, pdf_url: str) -> bool:
    """
    Check if a document ID is the UUID5 hash of its pdf_url.

    Args:
        doc_id: The document's _id field
        pdf_url: The document's pdf_url field

    Returns:
        True if doc_id matches uuid5(NAMESPACE_URL, pdf_url).hex
    """
    expected_id = uuid5(NAMESPACE_URL, pdf_url).hex
    return doc_id == expected_id


def migrate_document(
    db: couchdb.Database,
    doc_id: str,
    verbosity: int = 2
) -> bool:
    """
    Migrate a single document from random UUID to UUID5-based ID.

    Args:
        db: CouchDB database instance
        doc_id: ID of the document to migrate
        verbosity: Verbosity level (0=silent, 1=warnings, 2=normal, 3=verbose)

    Returns:
        True if migration was performed, False if skipped
    """
    # Fetch the document
    doc = db[doc_id]

    # Check if document has pdf_url
    if 'pdf_url' not in doc:
        if verbosity >= 1:
            print(f"Warning: Document {doc_id} has no pdf_url, skipping")
        return False

    pdf_url = doc['pdf_url']
    new_id = uuid5(NAMESPACE_URL, pdf_url).hex

    # Check if already using UUID5
    if doc_id == new_id:
        if verbosity >= 3:
            print(f"Skipping {doc_id}: already using UUID5")
        return False

    # Check if target ID already exists
    if new_id in db:
        if verbosity >= 2:
            print(f"Deleting duplicate {doc_id}: target {new_id} exists")
        # Target already exists, just delete the old one
        db.delete(doc)
        return True

    # Create new document with UUID5 ID
    if verbosity >= 2:
        print(f"Migrating {doc_id} -> {new_id}")

    # Copy all fields except _id, _rev, and _attachments
    # _attachments must be added separately after the document is saved
    new_doc = {}
    for key, value in doc.items():
        if key not in ('_id', '_rev', '_attachments'):
            new_doc[key] = value

    new_doc['_id'] = new_id

    # Save the new document (without attachments)
    db.save(new_doc)

    # Copy attachments if any
    if '_attachments' in doc:
        if verbosity >= 3:
            print(f"  Copying {len(doc['_attachments'])} attachment(s)")

        for attachment_name in doc['_attachments']:
            # Fetch attachment from old document
            attachment = db.get_attachment(doc_id, attachment_name)
            if attachment is None:
                if verbosity >= 1:
                    print(f"  Warning: Could not fetch attachment {attachment_name}")
                continue

            # Get attachment metadata
            attachment_info = doc['_attachments'][attachment_name]
            content_type = attachment_info.get('content_type', 'application/octet-stream')

            # Attach to new document
            new_doc_current = db[new_id]
            db.put_attachment(
                new_doc_current,
                attachment,
                attachment_name,
                content_type
            )

            if verbosity >= 3:
                print(f"  Copied attachment: {attachment_name}")

    # Delete the old document
    db.delete(doc)

    if verbosity >= 2:
        print(f"  Migration complete")

    return True


def migrate_database(
    db: couchdb.Database,
    verbosity: int = 2,
    dry_run: bool = False
) -> tuple[int, int]:
    """
    Migrate all documents in the database to UUID5-based IDs.

    Args:
        db: CouchDB database instance
        verbosity: Verbosity level
        dry_run: If True, only report what would be done

    Returns:
        Tuple of (migrated_count, skipped_count)
    """
    # Get all document IDs
    all_docs = db.view('_all_docs', include_docs=False)
    doc_ids = [row.id for row in all_docs if not row.id.startswith('_design/')]

    total = len(doc_ids)
    migrated = 0
    skipped = 0

    if verbosity >= 1:
        print(f"Found {total} documents to process")
        if dry_run:
            print("DRY RUN MODE - no changes will be made")
        print()

    for idx, doc_id in enumerate(doc_ids, 1):
        if verbosity >= 2:
            print(f"[{idx}/{total}] Processing {doc_id}")

        if dry_run:
            # Just check what would happen
            try:
                doc = db[doc_id]
                if 'pdf_url' not in doc:
                    print(f"  Would skip: no pdf_url")
                    skipped += 1
                    continue

                pdf_url = doc['pdf_url']
                new_id = uuid5(NAMESPACE_URL, pdf_url).hex

                if doc_id == new_id:
                    print(f"  Would skip: already UUID5")
                    skipped += 1
                elif new_id in db:
                    print(f"  Would delete: duplicate of {new_id}")
                    migrated += 1
                else:
                    print(f"  Would migrate: {doc_id} -> {new_id}")
                    migrated += 1
            except Exception as e:
                print(f"  Error: {e}")
                skipped += 1
        else:
            # Actually perform migration
            try:
                if migrate_document(db, doc_id, verbosity):
                    migrated += 1
                else:
                    skipped += 1
            except Exception as e:
                if verbosity >= 1:
                    print(f"  Error migrating {doc_id}: {e}")
                skipped += 1

        if verbosity >= 2:
            print()

    return migrated, skipped


def main() -> int:
    """
    Main entry point for the migration script.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description='Migrate CouchDB documents from random UUIDs to UUID5-based IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be changed
  %(prog)s --dry-run

  # Migrate with default settings
  %(prog)s

  # Migrate with verbose output
  %(prog)s -v 3

  # Silent migration
  %(prog)s -v 0

  # Use custom CouchDB credentials
  export COUCHDB_USER=admin COUCHDB_PASSWORD=secret
  %(prog)s
        """
    )

    parser.add_argument(
        '--couchdb-url',
        type=str,
        default=os.environ.get('COUCHDB_URL', 'http://localhost:5984'),
        help='CouchDB server URL (default: $COUCHDB_URL or http://localhost:5984)'
    )
    parser.add_argument(
        '--couchdb-username',
        type=str,
        default=os.environ.get('COUCHDB_USER'),
        help='CouchDB username (default: $COUCHDB_USER)'
    )
    parser.add_argument(
        '--couchdb-password',
        type=str,
        default=os.environ.get('COUCHDB_PASSWORD'),
        help='CouchDB password (default: $COUCHDB_PASSWORD)'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='skol_dev',
        help='CouchDB database name (default: skol_dev)'
    )
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help='Verbosity level: 0=silent, 1=warnings, 2=normal, 3=verbose (default: 2)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    try:
        # Connect to CouchDB
        if args.verbosity >= 2:
            print(f"Connecting to CouchDB at {args.couchdb_url}...")

        if args.couchdb_username and args.couchdb_password:
            couch = couchdb.Server(args.couchdb_url)
            couch.resource.credentials = (args.couchdb_username, args.couchdb_password)
            if args.verbosity >= 3:
                print(f"Using credentials for user: {args.couchdb_username}")
        else:
            couch = couchdb.Server(args.couchdb_url)

        # Get database
        db = couch[args.database]
        if args.verbosity >= 2:
            print(f"Using database: {args.database}")
            print()

        # Perform migration
        migrated, skipped = migrate_database(
            db=db,
            verbosity=args.verbosity,
            dry_run=args.dry_run
        )

        # Print summary
        if args.verbosity >= 1:
            print("=" * 60)
            print("Migration Summary")
            print("=" * 60)
            print(f"Documents migrated: {migrated}")
            print(f"Documents skipped:  {skipped}")
            print(f"Total processed:    {migrated + skipped}")
            if args.dry_run:
                print("\nDRY RUN - no changes were made")

        return 0

    except couchdb.http.ResourceNotFound:
        print(f"Error: Database '{args.database}' not found", file=sys.stderr)
        return 1
    except couchdb.http.Unauthorized:
        print("Error: Unauthorized access to CouchDB", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbosity >= 3:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
