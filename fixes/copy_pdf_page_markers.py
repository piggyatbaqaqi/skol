#!/usr/bin/env python3
"""
Copy PDF Page Markers from .txt to .txt.ann Files

This script scans the CouchDB database (default: skol_dev) for documents where:
1. The .txt attachment has PDF page markers (format: --- PDF Page N ---)
2. The .txt.ann attachment exists but is missing the PDF page markers

For each such document, it copies the PDF page markers from the .txt file
to the corresponding positions in the .txt.ann file.

This works because annotation does not add or remove lines - it only adds
YEDDA annotation markers around the text content.

Usage:
    python copy_pdf_page_markers.py [--database DATABASE] [--dry-run] [--doc-id DOC_ID]

Examples:
    # Scan entire skol_dev database and fix all issues (dry run first)
    python copy_pdf_page_markers.py --dry-run

    # Actually fix all issues
    python copy_pdf_page_markers.py

    # Fix a specific document
    python copy_pdf_page_markers.py --doc-id 7c7a503321655ca08dc47d6f9b7454c9

    # Scan a different database
    python copy_pdf_page_markers.py --database skol_prod --dry-run
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import constants
from env_config import get_env_config

def extract_page_markers_with_positions(text: str) -> List[Tuple[int, str]]:
    """
    Extract PDF page markers and their line positions.

    Args:
        text: The text content

    Returns:
        List of tuples (line_number, marker_text)
    """
    markers = []
    lines = text.split('\n')
    marker_pattern = re.compile(constants.pdf_page_pattern)

    for i, line in enumerate(lines):
        if marker_pattern.match(line):
            markers.append((i, line))

    return markers


def extract_content_without_markers(text: str) -> str:
    """
    Extract just the content, removing PDF page markers.

    Args:
        text: Text content potentially with page markers

    Returns:
        Text with page markers removed
    """
    lines = text.split('\n')
    marker_pattern = re.compile(constants.pdf_page_pattern)

    content_lines = [line for line in lines if not marker_pattern.match(line)]
    return '\n'.join(content_lines)


def extract_raw_content_from_yedda(ann_content: str) -> str:
    """
    Extract the raw text content from YEDDA-annotated content.

    This removes YEDDA annotation markers [@ ... #Label*] to get
    just the raw text content for matching.

    Args:
        ann_content: YEDDA-annotated content

    Returns:
        Raw text content without YEDDA markers
    """
    # Remove YEDDA opening markers [@
    text = re.sub(r'\[@\s*', '', ann_content)
    # Remove YEDDA closing markers #Label*]
    text = re.sub(r'\s*#[^\*]+\*\]', '', text)

    return text


def copy_markers_to_annotated(txt_content: str, ann_content: str) -> str:
    """
    Copy PDF page markers from txt to ann content using content matching.

    This handles cases where YEDDA annotation adds lines (coalescing),
    so line counts may differ between .txt and .txt.ann files.

    Strategy:
    1. First remove any existing page markers from .ann (in case of re-fix)
    2. Extract all non-marker content lines from .txt
    3. Strip YEDDA annotations from .ann to get raw content
    4. For each marker position in .txt, find what content comes
       immediately after it
    5. Search for that content in the stripped .ann to find where
       to insert the marker
    6. Insert markers at the found positions in the original .ann

    Args:
        txt_content: Content of .txt file with page markers
        ann_content: Content of .txt.ann file

    Returns:
        Updated .txt.ann content with page markers inserted
    """
    # First, strip any existing page markers from ann_content
    # (in case we're re-fixing a document with incorrect markers)
    ann_lines_orig = ann_content.split('\n')
    marker_pattern = re.compile(constants.pdf_page_pattern)
    ann_lines_clean = [line for line in ann_lines_orig if not marker_pattern.match(line)]
    ann_content = '\n'.join(ann_lines_clean)

    # Extract markers from txt file
    markers = extract_page_markers_with_positions(txt_content)

    if not markers:
        # No markers to copy
        return ann_content

    txt_lines = txt_content.split('\n')
    ann_lines = ann_content.split('\n')

    # For each marker, gather multiple lines of context after it
    # This helps with more reliable matching
    marker_insertions = []  # List of (marker_text, context_lines)

    for line_num, marker_text in markers:
        # Collect several substantive lines after this marker
        context_lines = []
        for j in range(line_num + 1, min(line_num + 20, len(txt_lines))):
            line = txt_lines[j].strip()
            # Skip empty lines, other markers
            if line and not re.match(constants.pdf_page_pattern, line):
                context_lines.append(line)
                if len(context_lines) >= 3:
                    break

        # If no content after (last page marker), look for content before
        if not context_lines and line_num > 0:
            # Look backwards for content
            for j in range(line_num - 1, max(0, line_num - 20), -1):
                line = txt_lines[j].strip()
                if line and not re.match(constants.pdf_page_pattern, line):
                    context_lines.insert(0, line)  # Insert at beginning since we're going backwards
                    if len(context_lines) >= 3:
                        break

        if context_lines:
            marker_insertions.append((marker_text, context_lines, line_num))
        else:
            # No context found - this is likely a blank page or part of consecutive blank pages
            # We'll handle these separately by inserting them at the beginning
            marker_insertions.append((marker_text, [], line_num))

    # Detect consecutive markers at document start (blank pages)
    # These need special handling even if they found context by looking ahead
    consecutive_start_markers = []
    for i, (marker_text, context_lines, line_num) in enumerate(marker_insertions):
        # Check if this is part of a consecutive group at the start
        if i == 0:
            consecutive_start_markers.append((marker_text, context_lines, line_num))
        else:
            prev_line_num = marker_insertions[i-1][2]
            # Check if there's non-marker content between this and previous marker
            lines_between = txt_lines[prev_line_num + 1:line_num]
            non_marker_content = [l for l in lines_between
                                 if l.strip() and not re.match(constants.pdf_page_pattern, l)]

            if not non_marker_content and len(consecutive_start_markers) > 0:
                # Still consecutive - add to group
                consecutive_start_markers.append((marker_text, context_lines, line_num))
            else:
                # No longer consecutive
                break

    # Separate markers: consecutive start group vs. others
    consecutive_start_line_nums = {l for _, _, l in consecutive_start_markers}
    markers_after_start = [m for m in marker_insertions
                          if m[2] not in consecutive_start_line_nums]

    # Further separate "markers after start" into those with/without content
    markers_with_content = [(m, c, l) for m, c, l in markers_after_start if c]
    markers_without_content = [(m, c, l) for m, c, l in markers_after_start if not c]

    # Now find where to insert each marker in the .ann file
    # Build a map: ann_line_idx -> marker_text
    insertions_map = {}

    # Strip YEDDA markers from all ann lines once for efficiency
    cleaned_ann_lines = []
    for ann_line in ann_lines:
        cleaned = re.sub(r'\[@\s*', '', ann_line)
        cleaned = re.sub(r'\s*#[^\*]+\*\]', '', cleaned)
        cleaned_ann_lines.append(cleaned.strip())

    # Process markers WITH content using fuzzy matching
    for marker_text, context_lines, orig_line_num in markers_with_content:
        # Try to find the context in the cleaned ann content
        # Use the first distinctive line as primary search
        search_text = context_lines[0][:60]  # First 60 chars

        # Find this text in the cleaned ann lines
        best_match_idx = None
        best_match_score = 0

        for i, cleaned_line in enumerate(cleaned_ann_lines):
            # Skip if we already plan to insert a marker here
            if i in insertions_map:
                continue

            # Check for exact match first
            if cleaned_line.startswith(search_text[:40]):
                best_match_idx = i
                break

            # Try fuzzy matching - check if significant portion matches
            # Useful when YEDDA annotation modifies whitespace
            match_score = 0
            if search_text[:20] in cleaned_line:
                match_score = 1

            # Check if next few lines also match context
            if match_score > 0 and len(context_lines) > 1:
                for j, context in enumerate(context_lines[1:3], start=1):
                    if i + j < len(cleaned_ann_lines):
                        next_line = cleaned_ann_lines[i + j]
                        if context[:20] in next_line:
                            match_score += 1

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_idx = i

        if best_match_idx is not None:
            insertions_map[best_match_idx] = marker_text
        else:
            # Marker couldn't be matched - check if it's the last marker
            # (i.e., appears at end of file with no following content)
            # In that case, we'll append it at the end
            if orig_line_num == len(txt_lines) - 1 or \
               all(not txt_lines[j].strip() for j in range(orig_line_num + 1, len(txt_lines))):
                # This is a trailing marker - mark it for appending at the end
                insertions_map['END'] = marker_text

    # Handle remaining blank page markers (those not in consecutive start group)
    # Place them using relative position in file
    for marker_text, _, orig_line_num in markers_without_content:
        # Calculate relative position (0.0 to 1.0) in txt file
        relative_pos = orig_line_num / len(txt_lines) if txt_lines else 0
        # Map to approximate line in ann file
        approx_ann_line = int(relative_pos * len(ann_lines))
        # Find nearest available position
        while approx_ann_line in insertions_map and approx_ann_line < len(ann_lines):
            approx_ann_line += 1
        if approx_ann_line < len(ann_lines):
            insertions_map[approx_ann_line] = marker_text

    # Build result by inserting markers at the appropriate positions
    result = []

    # First, add consecutive start markers in order (blank pages at document start)
    for marker_text, _, _ in sorted(consecutive_start_markers, key=lambda x: x[2]):
        result.append(marker_text)

    # Then add the rest of the content with markers inserted at matched positions
    for i, line in enumerate(ann_lines):
        if i in insertions_map:
            result.append(insertions_map[i])
        result.append(line)

    # Append any end markers
    if 'END' in insertions_map:
        result.append(insertions_map['END'])

    return '\n'.join(result)


def find_documents_with_missing_markers(db, doc_id: Optional[str] = None) -> List[dict]:
    """
    Find documents where .txt has page markers but .txt.ann doesn't.

    Args:
        db: CouchDB database instance
        doc_id: Optional specific document ID to check

    Returns:
        List of dicts with document info
    """
    problematic_docs = []

    if doc_id:
        # Check specific document
        doc_ids = [doc_id]
    else:
        # Scan all documents
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
                # Skip if missing either attachment
                continue

            # Read both files
            txt_content = db.get_attachment(doc_id, txt_name).read().decode('utf-8')
            ann_content = db.get_attachment(doc_id, ann_name).read().decode('utf-8')

            # Check for markers
            txt_markers = extract_page_markers_with_positions(txt_content)
            ann_markers = extract_page_markers_with_positions(ann_content)

            if txt_markers and not ann_markers:
                # Found a problematic document
                problematic_docs.append({
                    'doc_id': doc_id,
                    'txt_attachment': txt_name,
                    'ann_attachment': ann_name,
                    'marker_count': len(txt_markers),
                    'txt_content': txt_content,
                    'ann_content': ann_content,
                })
            elif txt_markers and ann_markers and len(txt_markers) != len(ann_markers):
                # Mismatched marker counts
                problematic_docs.append({
                    'doc_id': doc_id,
                    'txt_attachment': txt_name,
                    'ann_attachment': ann_name,
                    'marker_count': len(txt_markers),
                    'txt_content': txt_content,
                    'ann_content': ann_content,
                    'mismatch': True,
                    'txt_marker_count': len(txt_markers),
                    'ann_marker_count': len(ann_markers),
                })
            elif txt_markers and ann_markers and len(txt_markers) == len(ann_markers):
                # Check if markers are in correct order
                # txt_markers and ann_markers are lists of tuples (line_num, marker_text)
                txt_pages = [int(re.search(r'(\d+)', marker_text).group(1))
                             for _, marker_text in txt_markers]
                ann_pages = [int(re.search(r'(\d+)', marker_text).group(1))
                             for _, marker_text in ann_markers]

                if txt_pages != ann_pages:
                    # Markers exist but are out of order
                    problematic_docs.append({
                        'doc_id': doc_id,
                        'txt_attachment': txt_name,
                        'ann_attachment': ann_name,
                        'marker_count': len(txt_markers),
                        'txt_content': txt_content,
                        'ann_content': ann_content,
                        'order_mismatch': True,
                        'txt_marker_count': len(txt_markers),
                        'ann_marker_count': len(ann_markers),
                    })

        except Exception as e:
            print(f"  Warning: Error processing document {doc_id}: {e}")
            continue

    return problematic_docs


def fix_document(db, doc_info: dict, dry_run: bool = True) -> bool:
    """
    Fix a single document by copying page markers.

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
        updated_ann = copy_markers_to_annotated(
            doc_info['txt_content'],
            doc_info['ann_content']
        )

        # Verify markers were added
        updated_markers = extract_page_markers_with_positions(updated_ann)
        expected_count = doc_info['marker_count']

        if len(updated_markers) != expected_count:
            print(f"    ERROR: Expected {expected_count} markers, got {len(updated_markers)}")
            return False

        if dry_run:
            print(f"    [DRY RUN] Would update {ann_name} with {expected_count} page markers")
            return True

        # Actually update the document
        doc = db[doc_id]
        db.put_attachment(
            doc,
            updated_ann.encode('utf-8'),
            filename=ann_name,
            content_type='text/plain; charset=utf-8'
        )

        print(f"    ✓ Updated {ann_name} with {expected_count} page markers")
        return True

    except Exception as e:
        print(f"    ✗ Error fixing document: {e}")
        return False


def main():
    """Main entry point."""
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description='Copy PDF page markers from .txt to .txt.ann files',
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
    database = args.database or config['couchdb_database'] or 'skol_dev'

    print(f"\n{'='*70}")
    print(f"Copy PDF Page Markers - .txt to .txt.ann")
    print(f"{'='*70}")
    print(f"Database: {database}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    if args.doc_id:
        print(f"Document: {args.doc_id}")
    print(f"{'='*70}\n")

    try:
        # Connect to CouchDB
        import couchdb

        couchdb_url = f"http://{config['couchdb_host']}"
        server = couchdb.Server(couchdb_url)
        server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])

        if database not in server:
            print(f"Error: Database '{database}' not found")
            return 1

        db = server[database]

        # Find problematic documents
        print("Step 1: Scanning for documents with missing page markers...\n")
        problematic_docs = find_documents_with_missing_markers(db, args.doc_id)

        if not problematic_docs:
            print("\n✓ No documents found with missing page markers!")
            print("All .txt.ann files already have matching page markers.\n")
            return 0

        print(f"\nFound {len(problematic_docs)} documents with missing page markers:\n")

        # Show summary
        for i, doc_info in enumerate(problematic_docs, 1):
            doc_id = doc_info['doc_id']
            marker_count = doc_info['marker_count']
            if doc_info.get('order_mismatch'):
                print(f"{i}. {doc_id}")
                print(f"   ⚠ Marker order mismatch: {doc_info['txt_marker_count']} markers in wrong sequence")
            elif doc_info.get('mismatch'):
                print(f"{i}. {doc_id}")
                print(f"   ⚠ Marker count mismatch: .txt has {doc_info['txt_marker_count']}, "
                      f".txt.ann has {doc_info['ann_marker_count']}")
            else:
                print(f"{i}. {doc_id}")
                print(f"   Missing {marker_count} page markers in .txt.ann")

        print()

        # Fix documents
        print(f"Step 2: {'[DRY RUN] Simulating fixes' if args.dry_run else 'Applying fixes'}...\n")

        success_count = 0
        fail_count = 0

        for i, doc_info in enumerate(problematic_docs, 1):
            print(f"{i}/{len(problematic_docs)}: {doc_info['doc_id']}")
            if fix_document(db, doc_info, dry_run=args.dry_run):
                success_count += 1
            else:
                fail_count += 1

        print()
        print(f"{'='*70}")
        print(f"Summary:")
        print(f"  Total documents processed: {len(problematic_docs)}")
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
