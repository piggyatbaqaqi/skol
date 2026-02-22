#!/usr/bin/env python3
"""
Fix skol_taxa_full_dev Document IDs

This script corrects document IDs in skol_taxa_full_dev that were created with
the wrong hash (using 'url' field which was None, instead of 'human_url').

The fix:
1. Reads all existing records from skol_taxa_full_dev
2. Recalculates the correct document ID using human_url
3. Creates a new document with the correct ID (preserving all data)
4. Deletes the old document with the wrong ID

Usage:
    python fixes/fix_taxa_full_ids.py [--dry-run] [--verbosity LEVEL]

Example:
    # Preview what would be fixed
    python fixes/fix_taxa_full_ids.py --dry-run

    # Actually fix the IDs
    python fixes/fix_taxa_full_ids.py

    # Verbose output
    python fixes/fix_taxa_full_ids.py --verbosity 2
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))
# Add parent directory to path for ingestors module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_config import get_env_config
from ingestors.timestamps import set_timestamps


def generate_taxon_doc_id(taxon_text: str, description_text: str) -> str:
    """
    Generate a content-based, deterministic document ID for a taxon.

    This must match the implementation in extract_taxa_to_couchdb.py.

    Args:
        taxon_text: The nomenclature/taxon text
        description_text: The description text

    Returns:
        Deterministic document ID in format 'taxon_<sha256_hex>'
    """
    content = (taxon_text or "").strip() + ":" + (description_text or "").strip()
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    return f"taxon_{hash_obj.hexdigest()}"


def fix_taxa_ids(
    config: dict,
    db_name: str = 'skol_taxa_full_dev',
    dry_run: bool = False,
    verbosity: int = 1
) -> dict:
    """
    Fix document IDs in skol_taxa_full_dev.

    Args:
        config: Environment configuration
        db_name: Database name to fix (default: skol_taxa_full_dev)
        dry_run: If True, show what would be done without making changes
        verbosity: Verbosity level (0=silent, 1=info, 2=debug)

    Returns:
        Dict with 'fixed', 'skipped', 'errors' counts
    """
    import couchdb

    # Build CouchDB URL
    couchdb_url = config['couchdb_url']
    username = config['couchdb_username']
    password = config['couchdb_password']

    print(f"\n{'='*70}")
    print(f"Fix Taxa Document IDs")
    print(f"{'='*70}")
    print(f"CouchDB: {couchdb_url}")
    print(f"Database: {db_name}")
    if dry_run:
        print(f"Mode: DRY RUN (no changes will be made)")
    print()

    # Connect to CouchDB
    if verbosity >= 1:
        print("Connecting to CouchDB...")

    server = couchdb.Server(couchdb_url)
    if username and password:
        server.resource.credentials = (username, password)

    if db_name not in server:
        print(f"Error: Database '{db_name}' not found")
        return {'fixed': 0, 'skipped': 0, 'errors': 1}

    db = server[db_name]

    if verbosity >= 1:
        print(f"Connected to {db_name}")

    # Collect all document IDs first (to avoid modifying while iterating)
    if verbosity >= 1:
        print("Collecting document IDs...")

    doc_ids = []
    for doc_id in db:
        if not doc_id.startswith('_design/'):
            doc_ids.append(doc_id)

    total = len(doc_ids)
    if verbosity >= 1:
        print(f"Found {total} documents to check")

    # Process each document
    fixed = 0
    skipped = 0
    errors = 0

    for i, old_doc_id in enumerate(doc_ids, 1):
        try:
            # Load the document
            doc = db[old_doc_id]

            # Get source info for ID calculation
            source = doc.get('source', {})
            source_doc_id = str(source.get('doc_id', 'unknown'))
            human_url = source.get('human_url')  # Correct field!
            line_number = doc.get('line_number', 0)

            # Calculate the correct document ID
            correct_doc_id = generate_taxon_doc_id(
                doc.get('taxon', ''),
                doc.get('description', '')
            )

            # Check if ID needs fixing
            if old_doc_id == correct_doc_id:
                skipped += 1
                if verbosity >= 2:
                    print(f"  [{i}/{total}] SKIP {old_doc_id[:50]}... (already correct)")
                continue

            # ID needs fixing
            if verbosity >= 1:
                print(f"  [{i}/{total}] FIX {old_doc_id[:40]}...")
                print(f"           -> {correct_doc_id[:40]}...")

            if dry_run:
                fixed += 1
                continue

            # Create new document with correct ID
            new_doc = dict(doc)
            new_doc['_id'] = correct_doc_id

            # Remove _rev from new doc (it's a new document)
            if '_rev' in new_doc:
                del new_doc['_rev']

            # Check if correct ID already exists (shouldn't happen, but be safe)
            is_new_doc = correct_doc_id not in db
            if not is_new_doc:
                # Update existing document
                existing = db[correct_doc_id]
                new_doc['_rev'] = existing['_rev']
                if verbosity >= 2:
                    print(f"           (updating existing document)")

            # Save new document
            set_timestamps(new_doc, is_new=is_new_doc)
            db.save(new_doc)

            # Delete old document
            db.delete(doc)

            fixed += 1

            if verbosity >= 2:
                print(f"           (saved and deleted old)")

        except Exception as e:
            errors += 1
            print(f"  [{i}/{total}] ERROR {old_doc_id[:50]}...: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("Fix Complete!")
    print(f"{'='*70}")
    print(f"Total documents: {total}")
    print(f"Fixed: {fixed}")
    print(f"Skipped (already correct): {skipped}")
    if errors > 0:
        print(f"Errors: {errors}")
    if dry_run:
        print(f"\n[DRY RUN] No changes were made")

    return {'fixed': fixed, 'skipped': skipped, 'errors': errors}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix document IDs in skol_taxa_full_dev',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script fixes document IDs that were incorrectly generated using
'url' (which was None) instead of 'human_url'.

The correct ID generation uses:
  - source.doc_id
  - source.human_url
  - line_number

Examples:
    # Preview what would be fixed
    python fixes/fix_taxa_full_ids.py --dry-run

    # Actually fix the IDs
    python fixes/fix_taxa_full_ids.py

    # Fix with verbose output
    python fixes/fix_taxa_full_ids.py --verbosity 2
"""
    )

    parser.add_argument(
        '--db-name',
        type=str,
        default='skol_taxa_full_dev',
        help='Database name to fix (default: skol_taxa_full_dev)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Verbosity level: 0=silent, 1=info, 2=debug (default: 1)'
    )

    args = parser.parse_args()

    # Get configuration
    config = get_env_config()

    # Run the fix
    try:
        results = fix_taxa_ids(
            config=config,
            db_name=args.db_name,
            dry_run=args.dry_run,
            verbosity=args.verbosity
        )

        if results['errors'] > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
