#!/usr/bin/env python3
"""
Remove Taxa Documents Without Spans

This script removes taxa documents from skol_taxa_dev that don't have
nomenclature_spans or description_spans fields. After removal, you can
use `bin/extract_taxa_to_couchdb --skip-existing` to regenerate them
with span data.

Usage:
    python fixes/remove_taxa_without_spans.py [--dry-run] [--verbosity LEVEL] [--db-name NAME]

Example:
    # Preview what would be removed
    python fixes/remove_taxa_without_spans.py --dry-run

    # Actually remove documents
    python fixes/remove_taxa_without_spans.py

    # Remove from a different database
    python fixes/remove_taxa_without_spans.py --db-name skol_taxa_full_dev
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config


def has_spans(doc: dict) -> bool:
    """
    Check if a document has span data.

    Args:
        doc: CouchDB document

    Returns:
        True if document has non-empty nomenclature_spans or description_spans
    """
    nom_spans = doc.get('nomenclature_spans')
    desc_spans = doc.get('description_spans')

    # Check if either spans field exists and has data
    has_nom = isinstance(nom_spans, list) and len(nom_spans) > 0
    has_desc = isinstance(desc_spans, list) and len(desc_spans) > 0

    return has_nom or has_desc


def remove_taxa_without_spans(
    config: dict,
    db_name: str = 'skol_taxa_dev',
    dry_run: bool = False,
    verbosity: int = 1,
    limit: Optional[int] = None
) -> dict:
    """
    Remove taxa documents that don't have span data.

    Args:
        config: Environment configuration
        db_name: Database name (default: skol_taxa_dev)
        dry_run: If True, show what would be done without making changes
        verbosity: Verbosity level (0=silent, 1=info, 2=debug)
        limit: Maximum number of documents to process (None for all)

    Returns:
        Dict with 'removed', 'kept', 'errors' counts
    """
    import couchdb

    # Build CouchDB URL
    couchdb_url = config['couchdb_url']
    username = config['couchdb_username']
    password = config['couchdb_password']

    print(f"\n{'='*70}")
    print(f"Remove Taxa Documents Without Spans")
    print(f"{'='*70}")
    print(f"CouchDB: {couchdb_url}")
    print(f"Database: {db_name}")
    if limit:
        print(f"Limit: {limit} documents")
    if dry_run:
        print(f"Mode: DRY RUN (no changes will be made)")
    print()

    # Connect to CouchDB
    if verbosity >= 1:
        print("Connecting to CouchDB...")
    server = couchdb.Server(couchdb_url)

    # Set credentials before any operations
    if username and password:
        server.resource.credentials = (username, password)

    if db_name not in server:
        print(f"ERROR: Database '{db_name}' not found")
        return {'removed': 0, 'kept': 0, 'errors': 1}

    db = server[db_name]

    # Get all documents
    results = {
        'removed': 0,
        'kept': 0,
        'errors': 0
    }

    if verbosity >= 1:
        print(f"Fetching documents from {db_name}...")

    # Query all documents
    all_docs = db.view('_all_docs', include_docs=True)
    total = len(all_docs)
    print(f"Found {total} documents")

    # Collect documents to delete (for bulk delete)
    docs_to_delete = []

    processed = 0
    for row in all_docs:
        if limit and processed >= limit:
            break

        doc = row.doc
        doc_id = doc.get('_id', 'unknown')

        # Skip design documents
        if doc_id.startswith('_design/'):
            continue

        processed += 1

        try:
            if has_spans(doc):
                results['kept'] += 1
                if verbosity >= 2:
                    print(f"  Keep: {doc_id} (has spans)")
            else:
                results['removed'] += 1
                if verbosity >= 1:
                    taxon_title = doc.get('taxon', '')[:50].strip()
                    print(f"  Remove: {doc_id} ({taxon_title}...)")

                if not dry_run:
                    # Mark for deletion
                    docs_to_delete.append({
                        '_id': doc_id,
                        '_rev': doc.get('_rev'),
                        '_deleted': True
                    })

                    # Bulk delete in batches of 100
                    if len(docs_to_delete) >= 100:
                        db.update(docs_to_delete)
                        docs_to_delete = []

        except Exception as e:
            results['errors'] += 1
            print(f"  ERROR: {doc_id}: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()

    # Delete remaining documents
    if not dry_run and docs_to_delete:
        try:
            db.update(docs_to_delete)
        except Exception as e:
            results['errors'] += 1
            print(f"  ERROR during final bulk delete: {e}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Processed: {processed}")
    print(f"Removed:   {results['removed']}")
    print(f"Kept:      {results['kept']}")
    print(f"Errors:    {results['errors']}")

    if dry_run:
        print(f"\nDRY RUN - no changes were made")
    else:
        print(f"\nDocuments removed. Run extract_taxa_to_couchdb --skip-existing to regenerate.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Remove taxa documents without span data'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--verbosity', '-v',
        type=int,
        default=1,
        help='Verbosity level (0=silent, 1=info, 2=debug)'
    )
    parser.add_argument(
        '--db-name',
        type=str,
        default='skol_taxa_dev',
        help='Database name (default: skol_taxa_dev)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of documents to process'
    )

    args = parser.parse_args()

    # Load configuration
    config = get_env_config()

    # Run the removal
    results = remove_taxa_without_spans(
        config=config,
        db_name=args.db_name,
        dry_run=args.dry_run,
        verbosity=args.verbosity,
        limit=args.limit
    )

    # Exit with error code if there were failures
    if results['errors'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
