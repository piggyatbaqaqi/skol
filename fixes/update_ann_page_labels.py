#!/usr/bin/env python3
"""
Update PDF Page Markers in .ann Files to Include Label Syntax

This script scans the CouchDB database for documents where:
1. The .txt attachment has PDF page markers with Labels (format: --- PDF Page N Label X ---)
2. The .txt.ann attachment has page markers WITHOUT Labels (format: --- PDF Page N ---)

For each such document, it updates the markers in .txt.ann to include the
Label from .txt, preserving all YEDDA annotations.

Usage:
    python update_ann_page_labels.py [--database DATABASE] [--dry-run] [--doc-id DOC_ID]

Examples:
    # Scan entire database and show what would be fixed (dry run)
    python update_ann_page_labels.py --dry-run

    # Actually fix all documents
    python update_ann_page_labels.py

    # Fix a specific document
    python update_ann_page_labels.py --doc-id 7c7a503321655ca08dc47d6f9b7454c9
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import constants
from env_config import get_env_config


def extract_markers_with_labels(text: str) -> List[Tuple[int, int, Optional[str]]]:
    """
    Extract PDF page markers and their labels.

    Args:
        text: The text content

    Returns:
        List of tuples (line_number, page_number, label_or_None)
    """
    markers = []
    lines = text.split('\n')
    # Pattern that captures page number and optional label
    # Group 1: page number, Group 3: label (if present)
    pattern = re.compile(constants.pdf_page_pattern)

    for i, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            page_num = int(match.group(1))
            label = match.group(3)  # May be None if no label
            markers.append((i, page_num, label))

    return markers


def update_markers_with_labels(ann_content: str, txt_markers: List[Tuple[int, int, Optional[str]]]) -> str:
    """
    Update page markers in .ann content to include labels from .txt markers.

    Args:
        ann_content: Content of .txt.ann file
        txt_markers: List of (line_num, page_num, label) from .txt file

    Returns:
        Updated .txt.ann content with labels added to markers
    """
    # Build a map of page_number -> label from txt markers
    page_to_label: Dict[int, Optional[str]] = {}
    for _, page_num, label in txt_markers:
        if label:  # Only record if there's actually a label
            page_to_label[page_num] = label

    if not page_to_label:
        # No labels to add
        return ann_content

    # Pattern to match markers (with or without label)
    marker_pattern = re.compile(constants.pdf_page_pattern)

    lines = ann_content.split('\n')
    updated_lines = []

    for line in lines:
        match = marker_pattern.match(line.strip())
        if match:
            page_num = int(match.group(1))
            existing_label = match.group(3)

            # If this page has a label in txt and ann doesn't have it (or has different)
            if page_num in page_to_label and not existing_label:
                # Create new marker with label
                new_label = page_to_label[page_num]
                new_marker = f"--- PDF Page {page_num} Label {new_label} ---"
                updated_lines.append(new_marker)
            else:
                # Keep original line
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    return '\n'.join(updated_lines)


def find_documents_needing_label_update(db, doc_id: Optional[str] = None) -> List[dict]:
    """
    Find documents where .txt has labels but .txt.ann doesn't.

    Args:
        db: CouchDB database instance
        doc_id: Optional specific document ID to check

    Returns:
        List of dicts with document info
    """
    documents_to_fix = []

    if doc_id:
        doc_ids = [doc_id]
    else:
        doc_ids = [doc.id for doc in db.view('_all_docs')]

    total_docs = len(doc_ids)
    print(f"Scanning {total_docs} documents...")

    for i, doc_id in enumerate(doc_ids):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total_docs} documents scanned...")

        try:
            doc = db[doc_id]
            attachments = doc.get('_attachments', {})

            # Look for .txt and .txt.ann attachments
            txt_name = None
            ann_name = None

            for att_name in attachments.keys():
                if att_name.endswith('.txt') and not att_name.endswith('.txt.ann'):
                    txt_name = att_name
                elif att_name.endswith('.txt.ann'):
                    ann_name = att_name

            if not txt_name or not ann_name:
                continue

            # Read both files
            txt_content = db.get_attachment(doc_id, txt_name).read().decode('utf-8')
            ann_content = db.get_attachment(doc_id, ann_name).read().decode('utf-8')

            # Extract markers with label info
            txt_markers = extract_markers_with_labels(txt_content)
            ann_markers = extract_markers_with_labels(ann_content)

            # Count how many txt markers have labels
            txt_labels = [(p, l) for _, p, l in txt_markers if l]
            # Count how many ann markers are missing labels that txt has
            ann_pages_without_labels = {p for _, p, l in ann_markers if not l}

            # Find pages that need labels added
            pages_needing_labels = []
            for page_num, label in txt_labels:
                if page_num in ann_pages_without_labels:
                    pages_needing_labels.append((page_num, label))

            if pages_needing_labels:
                documents_to_fix.append({
                    'doc_id': doc_id,
                    'txt_attachment': txt_name,
                    'ann_attachment': ann_name,
                    'txt_content': txt_content,
                    'ann_content': ann_content,
                    'txt_markers': txt_markers,
                    'ann_markers': ann_markers,
                    'pages_needing_labels': pages_needing_labels,
                })

        except Exception as e:
            print(f"  Warning: Error processing document {doc_id}: {e}")
            continue

    return documents_to_fix


def fix_document(db, doc_info: dict, dry_run: bool = True) -> bool:
    """
    Fix a single document by adding labels to page markers.

    Args:
        db: CouchDB database instance
        doc_info: Document information dict
        dry_run: If True, don't actually update the database

    Returns:
        True if successful, False otherwise
    """
    doc_id = doc_info['doc_id']
    ann_name = doc_info['ann_attachment']

    try:
        # Create updated content
        updated_ann = update_markers_with_labels(
            doc_info['ann_content'],
            doc_info['txt_markers']
        )

        # Verify labels were added
        updated_markers = extract_markers_with_labels(updated_ann)
        pages_needing_labels = doc_info['pages_needing_labels']

        # Check that all pages now have labels
        updated_pages_with_labels = {p for _, p, l in updated_markers if l}
        missing = [p for p, _ in pages_needing_labels if p not in updated_pages_with_labels]

        if missing:
            print(f"    ERROR: Pages still missing labels: {missing}")
            return False

        if dry_run:
            labels_added = len(pages_needing_labels)
            print(f"    [DRY RUN] Would add labels to {labels_added} page marker(s)")
            for page_num, label in pages_needing_labels[:5]:
                print(f"      Page {page_num} -> Label {label}")
            if len(pages_needing_labels) > 5:
                print(f"      ... and {len(pages_needing_labels) - 5} more")
            return True

        # Actually update the document
        doc = db[doc_id]
        db.put_attachment(
            doc,
            updated_ann.encode('utf-8'),
            filename=ann_name,
            content_type='text/plain; charset=utf-8'
        )

        labels_added = len(pages_needing_labels)
        print(f"    Updated {ann_name}: added labels to {labels_added} page marker(s)")
        return True

    except Exception as e:
        print(f"    Error fixing document: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description='Update PDF page markers in .ann files to include Label syntax',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--database',
        default=None,
        help='Database name (default: from config or skol_dev)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    parser.add_argument(
        '--doc-id',
        help='Fix only this specific document ID'
    )

    args, _ = parser.parse_known_args()

    # Determine database
    database = args.database or config.get('ingest_database') or config.get('couchdb_database') or 'skol_dev'

    print(f"\n{'='*70}")
    print(f"Update PDF Page Markers with Labels")
    print(f"{'='*70}")
    print(f"Database: {database}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    if args.doc_id:
        print(f"Document: {args.doc_id}")
    print(f"{'='*70}\n")

    try:
        import couchdb

        couchdb_url = config['couchdb_url']
        server = couchdb.Server(couchdb_url)
        if config['couchdb_username'] and config['couchdb_password']:
            server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])

        if database not in server:
            print(f"Error: Database '{database}' not found")
            return 1

        db = server[database]

        # Find documents needing updates
        print("Step 1: Scanning for documents with missing page labels...\n")
        documents_to_fix = find_documents_needing_label_update(db, args.doc_id)

        if not documents_to_fix:
            print("\nNo documents found with missing page labels!")
            print("All .txt.ann files already have matching labels.\n")
            return 0

        print(f"\nFound {len(documents_to_fix)} documents with missing labels:\n")

        # Show summary
        total_pages_to_fix = 0
        for i, doc_info in enumerate(documents_to_fix, 1):
            doc_id = doc_info['doc_id']
            pages = doc_info['pages_needing_labels']
            total_pages_to_fix += len(pages)
            print(f"{i}. {doc_id}")
            print(f"   {len(pages)} page(s) need labels: {[p for p, _ in pages[:5]]}"
                  f"{'...' if len(pages) > 5 else ''}")

        print(f"\nTotal: {total_pages_to_fix} page markers need labels across {len(documents_to_fix)} documents\n")

        # Fix documents
        print(f"Step 2: {'[DRY RUN] Simulating fixes' if args.dry_run else 'Applying fixes'}...\n")

        success_count = 0
        fail_count = 0

        for i, doc_info in enumerate(documents_to_fix, 1):
            print(f"{i}/{len(documents_to_fix)}: {doc_info['doc_id']}")
            if fix_document(db, doc_info, dry_run=args.dry_run):
                success_count += 1
            else:
                fail_count += 1

        print()
        print(f"{'='*70}")
        print(f"Summary:")
        print(f"  Total documents processed: {len(documents_to_fix)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {fail_count}")
        if args.dry_run:
            print(f"\n  This was a DRY RUN - no changes were made.")
            print(f"  Run without --dry-run to actually update the database.")
        print(f"{'='*70}\n")

        return 0 if fail_count == 0 else 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
