#!/usr/bin/env python3
"""
Fix Span Format in Taxa Documents

This script converts span data from array format (legacy) to dictionary format.

The problem:
- PySpark's StructType schema converted span dictionaries to Row objects
- Row objects serialize as arrays/tuples instead of dictionaries
- Old format: [14853, 5992, 5995, 267767, 267991, 0, None, '24']
- New format: {'paragraph_number': 14853, 'start_line': 5992, ...}

The fix:
1. Reads all existing records from skol_taxa_dev
2. Identifies documents with array-format spans
3. Converts span arrays to dictionaries with proper field names
4. Updates the documents in place

Usage:
    python fixes/fix_span_format.py [--dry-run] [--verbosity LEVEL] [--db-name NAME]

Example:
    # Preview what would be fixed
    python fixes/fix_span_format.py --dry-run

    # Actually fix the spans
    python fixes/fix_span_format.py

    # Fix a different database
    python fixes/fix_span_format.py --db-name skol_taxa_full_dev
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add parent directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))
# Add parent directory to path for ingestors module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_config import get_env_config
from ingestors.timestamps import set_timestamps

# Span field names in order (matches StructType definition)
SPAN_FIELDS = [
    'paragraph_number',  # 0: int
    'start_line',        # 1: int
    'end_line',          # 2: int
    'start_char',        # 3: int
    'end_char',          # 4: int
    'pdf_page',          # 5: int
    'pdf_label',         # 6: str or None
    'empirical_page',    # 7: str or None
]


def convert_span_array_to_dict(span: Union[List, Dict]) -> Optional[Dict[str, Any]]:
    """
    Convert a span from array format to dictionary format.

    Args:
        span: Either an array [p, sl, el, sc, ec, pp, pl, ep] or already a dict

    Returns:
        Dictionary with proper field names, or None if input is invalid
    """
    if isinstance(span, dict):
        # Already in dict format, return as-is
        return span

    if not isinstance(span, (list, tuple)):
        return None

    if len(span) < 5:
        # Too short to be a valid span
        return None

    result = {}
    for i, field_name in enumerate(SPAN_FIELDS):
        if i < len(span):
            value = span[i]
            # Handle 'None' strings from MapType conversion
            if value == 'None':
                value = None
            result[field_name] = value
        else:
            result[field_name] = None

    return result


def needs_conversion(doc: Dict) -> bool:
    """
    Check if a document has spans that need conversion.

    Args:
        doc: CouchDB document

    Returns:
        True if document has array-format spans
    """
    for field in ['nomenclature_spans', 'description_spans']:
        spans = doc.get(field)
        if spans:
            for span in spans:
                if isinstance(span, (list, tuple)):
                    return True
    return False


def convert_document_spans(doc: Dict) -> Dict:
    """
    Convert all spans in a document from array to dict format.

    Args:
        doc: CouchDB document with array-format spans

    Returns:
        Document with dict-format spans
    """
    for field in ['nomenclature_spans', 'description_spans']:
        spans = doc.get(field)
        if spans:
            converted = []
            for span in spans:
                converted_span = convert_span_array_to_dict(span)
                if converted_span:
                    converted.append(converted_span)
            doc[field] = converted
    return doc


def fix_span_format(
    config: dict,
    db_name: str = 'skol_taxa_dev',
    dry_run: bool = False,
    verbosity: int = 1,
    limit: Optional[int] = None
) -> dict:
    """
    Fix span format in taxa documents.

    Args:
        config: Environment configuration
        db_name: Database name to fix (default: skol_taxa_dev)
        dry_run: If True, show what would be done without making changes
        verbosity: Verbosity level (0=silent, 1=info, 2=debug)
        limit: Maximum number of documents to process (None for all)

    Returns:
        Dict with 'fixed', 'skipped', 'already_ok', 'errors' counts
    """
    import couchdb

    # Build CouchDB URL
    couchdb_url = config['couchdb_url']
    username = config['couchdb_username']
    password = config['couchdb_password']

    print(f"\n{'='*70}")
    print(f"Fix Span Format in Taxa Documents")
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
        return {'fixed': 0, 'skipped': 0, 'already_ok': 0, 'errors': 1}

    db = server[db_name]

    # Get all documents
    results = {
        'fixed': 0,
        'skipped': 0,
        'already_ok': 0,
        'errors': 0
    }

    if verbosity >= 1:
        print(f"Fetching documents from {db_name}...")

    # Query all documents
    all_docs = db.view('_all_docs', include_docs=True)
    total = len(all_docs)
    print(f"Found {total} documents")

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
            # Check if this document has spans that need conversion
            if not needs_conversion(doc):
                # Check if it has any spans at all
                has_spans = bool(doc.get('nomenclature_spans') or doc.get('description_spans'))
                if has_spans:
                    results['already_ok'] += 1
                    if verbosity >= 2:
                        print(f"  Already OK: {doc_id}")
                else:
                    results['skipped'] += 1
                    if verbosity >= 2:
                        print(f"  No spans: {doc_id}")
                continue

            # Convert the spans
            if verbosity >= 1:
                nom_count = len(doc.get('nomenclature_spans', []))
                desc_count = len(doc.get('description_spans', []))
                print(f"  Converting: {doc_id} ({nom_count} nom, {desc_count} desc spans)")

            if not dry_run:
                # Convert spans in place
                convert_document_spans(doc)

                # Save the updated document
                set_timestamps(doc)  # is_new=False for existing doc
                db.save(doc)

            results['fixed'] += 1

        except Exception as e:
            results['errors'] += 1
            print(f"  ERROR: {doc_id}: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()

    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"Processed: {processed}")
    print(f"Fixed:     {results['fixed']}")
    print(f"Already OK:{results['already_ok']}")
    print(f"No spans:  {results['skipped']}")
    print(f"Errors:    {results['errors']}")

    if dry_run:
        print(f"\nDRY RUN - no changes were made")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fix span format in taxa documents (array -> dict)'
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
        help='Database name to fix (default: skol_taxa_dev)'
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

    # Run the fix
    results = fix_span_format(
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
