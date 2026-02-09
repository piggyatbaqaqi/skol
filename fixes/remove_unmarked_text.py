#!/usr/bin/env python3
"""
Remove Text Attachments Without PDF Page Markers

This script scans CouchDB for documents with *.ann or *.txt attachments
and removes those that do not contain PDF page header markers
(e.g., "--- PDF Page 1 Label iv ---").

Text files without page markers were likely extracted from sources other
than PDFs or were extracted before the page marker feature was added.
These files may have incorrect character offsets that don't match the
source PDFs.

Usage:
    python remove_unmarked_text.py --database skol_dev --dry-run
    python remove_unmarked_text.py --database skol_dev
    python remove_unmarked_text.py --database skol_dev --doc-id SPECIFIC_DOC_ID
    python remove_unmarked_text.py --database skol_dev --limit 100

Options:
    --database      CouchDB database name (default: skol_dev)
    --doc-id        Process only this specific document ID
    --dry-run       Preview without deleting attachments
    --limit N       Process at most N documents
    --verbosity     Verbosity level (0=silent, 1=summary, 2=details)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import constants
from env_config import get_env_config
from ingestors.timestamps import set_timestamps


def has_pdf_page_markers(content: str) -> bool:
    """
    Check if content contains PDF page markers.

    Args:
        content: Text content to check

    Returns:
        True if at least one PDF page marker is found
    """
    pattern = re.compile(constants.pdf_page_pattern, re.MULTILINE)
    return bool(pattern.search(content))


def get_text_attachments(doc: Dict) -> List[str]:
    """
    Get list of .ann and .txt attachment names from a document.

    Args:
        doc: CouchDB document

    Returns:
        List of attachment names ending with .ann or .txt
    """
    attachments = doc.get('_attachments', {})
    return [
        name for name in attachments.keys()
        if name.endswith('.ann') or name.endswith('.txt')
    ]


def remove_unmarked_attachments(
    db,
    doc_id: Optional[str] = None,
    dry_run: bool = True,
    limit: Optional[int] = None,
    verbosity: int = 1
) -> Dict[str, int]:
    """
    Remove .ann and .txt attachments that lack PDF page markers.

    Args:
        db: CouchDB database connection
        doc_id: If specified, only process this document
        dry_run: If True, preview without deleting
        limit: Maximum number of documents to process
        verbosity: Output verbosity level

    Returns:
        Dictionary with counts of documents and attachments processed
    """
    results = {
        'docs_checked': 0,
        'docs_modified': 0,
        'attachments_checked': 0,
        'attachments_removed': 0,
        'attachments_kept': 0,
        'errors': 0,
    }

    # Get document IDs to process
    if doc_id:
        doc_ids = [doc_id] if doc_id in db else []
        if not doc_ids:
            print(f"Document not found: {doc_id}")
            return results
    else:
        # Query all documents with attachments
        doc_ids = [row.id for row in db.view('_all_docs')]

    if limit:
        doc_ids = doc_ids[:limit]

    if verbosity >= 1:
        print(f"Processing {len(doc_ids)} documents...")
        if dry_run:
            print("[DRY RUN - no changes will be made]")
        print()

    for doc_id in doc_ids:
        try:
            doc = db[doc_id]
            text_attachments = get_text_attachments(doc)

            if not text_attachments:
                continue

            results['docs_checked'] += 1
            attachments_to_remove: List[str] = []

            for att_name in text_attachments:
                results['attachments_checked'] += 1

                # Fetch attachment content
                try:
                    content = db.get_attachment(doc, att_name)
                    if content is None:
                        continue
                    # Read from file-like object if needed
                    if hasattr(content, 'read'):
                        content = content.read()
                    # Decode bytes to string
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                except Exception as e:
                    if verbosity >= 2:
                        print(f"  Error reading {doc_id}/{att_name}: {e}")
                    results['errors'] += 1
                    continue

                # Check for PDF page markers
                if has_pdf_page_markers(content):
                    results['attachments_kept'] += 1
                    if verbosity >= 2:
                        print(f"  KEEP: {doc_id}/{att_name} (has markers)")
                else:
                    attachments_to_remove.append(att_name)
                    if verbosity >= 2:
                        print(f"  REMOVE: {doc_id}/{att_name} (no markers)")

            # Remove attachments without markers
            if attachments_to_remove:
                if verbosity >= 1 and verbosity < 2:
                    print(f"{doc_id}: removing {len(attachments_to_remove)} attachment(s)")

                if not dry_run:
                    # Refresh document to get latest revision
                    fresh_doc = db[doc_id]

                    for att_name in attachments_to_remove:
                        try:
                            db.delete_attachment(fresh_doc, att_name)
                            # Refresh after each deletion to get new revision
                            fresh_doc = db[doc_id]
                            results['attachments_removed'] += 1
                        except Exception as e:
                            if verbosity >= 1:
                                print(f"  Error deleting {att_name}: {e}")
                            results['errors'] += 1

                    # Update modification timestamp
                    fresh_doc = db[doc_id]
                    set_timestamps(fresh_doc)
                    db.save(fresh_doc)

                    results['docs_modified'] += 1
                else:
                    # Dry run - just count
                    results['attachments_removed'] += len(attachments_to_remove)
                    results['docs_modified'] += 1

        except Exception as e:
            if verbosity >= 1:
                print(f"Error processing {doc_id}: {e}")
            results['errors'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Remove .ann and .txt attachments without PDF page markers'
    )
    parser.add_argument(
        '--database',
        default='skol_dev',
        help='CouchDB database name (default: skol_dev)'
    )
    parser.add_argument(
        '--doc-id',
        help='Process only this specific document ID'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without deleting attachments'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of documents to process'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        help='Verbosity level (0=silent, 1=summary, 2=details)'
    )

    args = parser.parse_args()

    # Get environment configuration
    env_config = get_env_config()
    couchdb_url = env_config['couchdb_url']
    couchdb_user = env_config['couchdb_username']
    couchdb_password = env_config['couchdb_password']

    # Connect to CouchDB
    import couchdb
    if couchdb_user and couchdb_password:
        server = couchdb.Server(couchdb_url)
        server.resource.credentials = (couchdb_user, couchdb_password)
    else:
        server = couchdb.Server(couchdb_url)

    if args.database not in server:
        print(f"Database not found: {args.database}")
        sys.exit(1)

    db = server[args.database]

    # Run the cleanup
    results = remove_unmarked_attachments(
        db=db,
        doc_id=args.doc_id,
        dry_run=args.dry_run,
        limit=args.limit,
        verbosity=args.verbosity
    )

    # Print summary
    if args.verbosity >= 1:
        print()
        print("=" * 60)
        print("Summary:")
        print(f"  Documents checked:     {results['docs_checked']}")
        print(f"  Documents modified:    {results['docs_modified']}")
        print(f"  Attachments checked:   {results['attachments_checked']}")
        print(f"  Attachments removed:   {results['attachments_removed']}")
        print(f"  Attachments kept:      {results['attachments_kept']}")
        if results['errors']:
            print(f"  Errors:                {results['errors']}")
        if args.dry_run:
            print()
            print("This was a dry run. Run without --dry-run to apply changes.")


if __name__ == '__main__':
    main()
