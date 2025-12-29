#!/usr/bin/env python3
"""
Cleanup script to fix records with session IDs in URLs.

This script identifies documents in CouchDB that have session IDs in their URLs
(e.g., ;jsessionid=...), creates corrected copies with session IDs removed,
updates the _id field to match the cleaned URL, and copies all attachments.

Usage:
    # Dry run (preview changes)
    ./cleanup_session_ids.py --dry-run

    # Execute cleanup
    ./cleanup_session_ids.py

    # Execute and delete old records
    ./cleanup_session_ids.py --delete-old
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple, Optional
from uuid import uuid5, NAMESPACE_URL

import couchdb


def clean_session_id(url: str) -> str:
    """
    Remove session IDs from URL.

    Args:
        url: URL that may contain session ID

    Returns:
        URL with session ID removed
    """
    # Remove ;jsessionid=... and everything up to ? or end of string
    return url.split(';')[0]


def needs_cleaning(doc: Dict) -> bool:
    """
    Check if a document has session IDs in any of its URLs.

    Args:
        doc: CouchDB document

    Returns:
        True if document needs cleaning
    """
    # Skip design documents
    if doc.get('_id', '').startswith('_design/'):
        return False

    # Check if any URL field contains session ID
    url_fields = ['url', 'pdf_url', 'human_url', 'bibtex_link']
    for field in url_fields:
        if field in doc:
            value = doc[field]
            if isinstance(value, str) and ';jsessionid=' in value:
                return True

    return False


def clean_document(doc: Dict) -> Tuple[Dict, Dict[str, str]]:
    """
    Create a cleaned copy of a document with session IDs removed.

    Args:
        doc: Original document

    Returns:
        Tuple of (cleaned_doc, url_changes) where url_changes maps field names to old URLs
    """
    cleaned = doc.copy()
    url_changes = {}

    # Remove CouchDB metadata from the copy
    for key in ['_id', '_rev', '_attachments']:
        cleaned.pop(key, None)

    # Clean all URL fields
    url_fields = ['url', 'pdf_url', 'human_url', 'bibtex_link']
    for field in url_fields:
        if field in cleaned and isinstance(cleaned[field], str):
            original = cleaned[field]
            cleaned_url = clean_session_id(original)
            if original != cleaned_url:
                url_changes[field] = original
                cleaned[field] = cleaned_url

    # Generate new _id based on cleaned URL
    if 'url' in cleaned:
        new_id = str(uuid5(NAMESPACE_URL, cleaned['url']))
        cleaned['_id'] = new_id

    return cleaned, url_changes


def copy_attachments(
    db: couchdb.Database,
    source_doc_id: str,
    target_doc_id: str,
    attachment_names: List[str],
    verbosity: int = 1
) -> int:
    """
    Copy attachments from source document to target document.

    Args:
        db: CouchDB database
        source_doc_id: Source document ID
        target_doc_id: Target document ID
        attachment_names: List of attachment names to copy
        verbosity: Verbosity level

    Returns:
        Number of attachments copied
    """
    copied = 0
    for att_name in attachment_names:
        try:
            # Get attachment from source
            attachment = db.get_attachment(source_doc_id, att_name)
            if attachment is None:
                if verbosity >= 2:
                    print(f"    Warning: Attachment '{att_name}' not found")
                continue

            # Get the target document to update it
            target_doc = db[target_doc_id]

            # Put attachment on target
            db.put_attachment(
                target_doc,
                attachment,
                filename=att_name,
                content_type='application/pdf'  # Assuming PDFs
            )
            copied += 1

            if verbosity >= 3:
                print(f"    Copied attachment: {att_name}")

        except Exception as e:
            if verbosity >= 1:
                print(f"    Error copying attachment '{att_name}': {e}")

    return copied


def find_affected_documents(db: couchdb.Database, verbosity: int = 1) -> List[Dict]:
    """
    Find all documents that need cleaning.

    Args:
        db: CouchDB database
        verbosity: Verbosity level

    Returns:
        List of documents that need cleaning
    """
    if verbosity >= 2:
        print("Scanning database for documents with session IDs...")

    affected = []
    total = 0

    for doc_id in db:
        total += 1
        try:
            doc = db[doc_id]
            if needs_cleaning(doc):
                affected.append(doc)

                if verbosity >= 3:
                    print(f"  Found: {doc_id}")

        except Exception as e:
            if verbosity >= 1:
                print(f"  Error reading document {doc_id}: {e}")

    if verbosity >= 2:
        print(f"Scanned {total} documents, found {len(affected)} needing cleanup")

    return affected


def preview_changes(affected_docs: List[Dict], verbosity: int = 1) -> None:
    """
    Preview the changes that would be made.

    Args:
        affected_docs: List of documents to be cleaned
        verbosity: Verbosity level
    """
    print("\n" + "=" * 80)
    print(f"PREVIEW: {len(affected_docs)} document(s) would be cleaned")
    print("=" * 80)

    for idx, doc in enumerate(affected_docs[:10], 1):  # Show first 10
        cleaned, url_changes = clean_document(doc)

        print(f"\n{idx}. Document: {doc['_id']}")
        print(f"   Title: {doc.get('title', 'N/A')[:60]}")
        print(f"   New ID: {cleaned['_id']}")

        if url_changes:
            print("   URL changes:")
            for field, old_url in url_changes.items():
                new_url = cleaned[field]
                print(f"     {field}:")
                print(f"       Old: {old_url}")
                print(f"       New: {new_url}")

        if '_attachments' in doc:
            att_count = len(doc['_attachments'])
            print(f"   Attachments: {att_count} to copy")

    if len(affected_docs) > 10:
        print(f"\n   ... and {len(affected_docs) - 10} more documents")

    print("\n" + "=" * 80)


def cleanup_documents(
    db: couchdb.Database,
    affected_docs: List[Dict],
    delete_old: bool = False,
    verbosity: int = 1
) -> Tuple[int, int, int]:
    """
    Clean up documents by creating corrected copies.

    Args:
        db: CouchDB database
        affected_docs: List of documents to clean
        delete_old: Whether to delete old documents after copying
        verbosity: Verbosity level

    Returns:
        Tuple of (created, attachments_copied, deleted)
    """
    created = 0
    attachments_copied = 0
    deleted = 0

    for idx, doc in enumerate(affected_docs, 1):
        old_id = doc['_id']
        title = doc.get('title', 'N/A')

        if verbosity >= 2:
            print(f"\n[{idx}/{len(affected_docs)}] Processing: {title[:50]}")
            print(f"  Old ID: {old_id}")

        try:
            # Create cleaned document
            cleaned, url_changes = clean_document(doc)
            new_id = cleaned['_id']

            if verbosity >= 2:
                print(f"  New ID: {new_id}")

            # Check if cleaned document already exists
            if new_id in db:
                if verbosity >= 2:
                    print(f"  Skipping: Cleaned version already exists")
                continue

            # Save cleaned document
            db.save(cleaned)
            created += 1

            if verbosity >= 2:
                print(f"  ✓ Created cleaned document")

            # Copy attachments if present
            if '_attachments' in doc:
                attachment_names = list(doc['_attachments'].keys())
                copied = copy_attachments(
                    db, old_id, new_id, attachment_names, verbosity
                )
                attachments_copied += copied

                if verbosity >= 2:
                    print(f"  ✓ Copied {copied} attachment(s)")

            # Delete old document if requested
            if delete_old:
                db.delete(doc)
                deleted += 1
                if verbosity >= 2:
                    print(f"  ✓ Deleted old document")

        except Exception as e:
            if verbosity >= 1:
                print(f"  ✗ Error processing document: {e}")

    return created, attachments_copied, deleted


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Clean up CouchDB documents with session IDs in URLs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes without making any modifications
  %(prog)s --dry-run

  # Execute cleanup, keeping old documents
  %(prog)s

  # Execute cleanup and delete old documents
  %(prog)s --delete-old

  # Use custom database
  %(prog)s --database skol_prod --dry-run
        """
    )

    # CouchDB connection arguments
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

    # Operation mode
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying the database'
    )
    parser.add_argument(
        '--delete-old',
        action='store_true',
        help='Delete old documents after creating cleaned copies'
    )

    # Verbosity
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help='Verbosity level (default: 2)'
    )

    args = parser.parse_args()

    try:
        # Connect to CouchDB
        if args.verbosity >= 2:
            print(f"Connecting to CouchDB at {args.couchdb_url}...")

        if args.couchdb_username and args.couchdb_password:
            couch = couchdb.Server(args.couchdb_url)
            couch.resource.credentials = (args.couchdb_username, args.couchdb_password)
        else:
            couch = couchdb.Server(args.couchdb_url)

        db = couch[args.database]
        if args.verbosity >= 2:
            print(f"Using database: {args.database}")

        # Find affected documents
        affected_docs = find_affected_documents(db, args.verbosity)

        if not affected_docs:
            print("\n✓ No documents need cleaning!")
            return 0

        # Preview changes
        preview_changes(affected_docs, args.verbosity)

        # Dry run mode - just show preview
        if args.dry_run:
            print("\nDRY RUN MODE: No changes made")
            return 0

        # Ask for confirmation
        print(f"\nThis will create {len(affected_docs)} cleaned document(s)")
        if args.delete_old:
            print(f"and DELETE the {len(affected_docs)} old document(s)")
        else:
            print("(old documents will be kept)")

        response = input("\nProceed? [y/N]: ").strip().lower()
        if response not in ('y', 'yes'):
            print("Cancelled.")
            return 0

        # Perform cleanup
        print("\nCleaning documents...")
        created, attachments_copied, deleted = cleanup_documents(
            db, affected_docs, args.delete_old, args.verbosity
        )

        # Summary
        print("\n" + "=" * 80)
        print("CLEANUP COMPLETE")
        print("=" * 80)
        print(f"Documents created: {created}")
        print(f"Attachments copied: {attachments_copied}")
        if args.delete_old:
            print(f"Old documents deleted: {deleted}")
        else:
            print(f"Old documents kept: {len(affected_docs)}")
        print("=" * 80)

        return 0

    except couchdb.http.ResourceNotFound:
        print(f"Error: Database '{args.database}' not found", file=sys.stderr)
        return 1
    except couchdb.http.Unauthorized:
        print("Error: Unauthorized access to CouchDB", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbosity >= 3:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
