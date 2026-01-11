#!/usr/bin/env python3
"""
Test PDF Page Marker Preservation

This script tests that PDF page markers are properly preserved through
the classification pipeline.

Usage:
    python test_page_marker_preservation.py --doc-id DOC_ID
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config


def test_page_markers(doc_id: str, database: str = "skol_dev"):
    """
    Test that page markers are preserved in .txt and .txt.ann files.

    Args:
        doc_id: Document ID to test
        database: Database name
    """
    import couchdb
    import re

    # Get environment configuration
    config = get_env_config()

    # Connect to CouchDB
    couchdb_url = f"http://{config['couchdb_host']}"
    server = couchdb.Server(couchdb_url)
    server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])

    if database not in server:
        print(f"Error: Database '{database}' not found")
        return False

    db = server[database]

    if doc_id not in db:
        print(f"Error: Document '{doc_id}' not found")
        return False

    doc = db[doc_id]
    attachments = doc.get('_attachments', {})

    print(f"\n{'='*70}")
    print(f"Testing Page Marker Preservation")
    print(f"{'='*70}")
    print(f"Document: {doc_id}")
    print(f"Database: {database}")
    print()

    # Test 1: Check .txt file has page markers
    print("Test 1: Checking .txt file for page markers...")
    if 'article.txt' not in attachments:
        print("  ✗ FAIL: article.txt not found")
        return False

    txt_content = db.get_attachment(doc_id, 'article.txt').read().decode('utf-8')
    txt_markers = re.findall(r'^---\s*PDF\s+Page\s+(\d+)\s*---\s*$', txt_content, re.MULTILINE)

    if not txt_markers:
        print("  ✗ FAIL: No page markers found in article.txt")
        print(f"  First 500 chars: {txt_content[:500]}")
        return False

    print(f"  ✓ PASS: Found {len(txt_markers)} page markers in article.txt")
    print(f"  Page numbers: {', '.join(txt_markers)}")
    print()

    # Test 2: Check .txt.ann file has page markers
    print("Test 2: Checking .txt.ann file for page markers...")
    if 'article.txt.ann' not in attachments:
        print("  ⚠ WARNING: article.txt.ann not found (not yet generated)")
        print("  Run prediction first: bin/predict_classifier.py")
        return True  # Not a failure, just not generated yet

    ann_content = db.get_attachment(doc_id, 'article.txt.ann').read().decode('utf-8')
    ann_markers = re.findall(r'^---\s*PDF\s+Page\s+(\d+)\s*---\s*$', ann_content, re.MULTILINE)

    if not ann_markers:
        print("  ✗ FAIL: No page markers found in article.txt.ann")
        print(f"  First 500 chars: {ann_content[:500]}")
        return False

    print(f"  ✓ PASS: Found {len(ann_markers)} page markers in article.txt.ann")
    print(f"  Page numbers: {', '.join(ann_markers)}")
    print()

    # Test 3: Verify page markers match
    print("Test 3: Verifying page markers match between .txt and .txt.ann...")
    if txt_markers == ann_markers:
        print(f"  ✓ PASS: Page markers match ({len(txt_markers)} markers)")
    else:
        print(f"  ✗ FAIL: Page markers don't match")
        print(f"  .txt markers: {txt_markers}")
        print(f"  .txt.ann markers: {ann_markers}")
        return False

    print()

    # Test 4: Check that page markers are NOT inside YEDDA annotations
    print("Test 4: Verifying page markers are not inside YEDDA annotations...")
    # Find all YEDDA blocks
    yedda_blocks = re.findall(r'\[@\s*.*?\s*#[^\*]+\*\]', ann_content, re.DOTALL)

    markers_in_blocks = 0
    for block in yedda_blocks:
        if re.search(r'---\s*PDF\s+Page\s+\d+\s*---', block):
            markers_in_blocks += 1
            print(f"  Found marker inside block: {block[:100]}...")

    if markers_in_blocks > 0:
        print(f"  ✗ FAIL: Found {markers_in_blocks} page markers inside YEDDA blocks")
        return False
    else:
        print(f"  ✓ PASS: No page markers found inside YEDDA blocks")

    print()

    # Test 5: Show sample output around page markers
    print("Test 5: Sample output around page markers...")
    lines = ann_content.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^---\s*PDF\s+Page\s+\d+\s*---\s*$', line):
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            print(f"\n  Context around line {i+1}:")
            for j in range(start, end):
                marker = ">>>" if j == i else "   "
                print(f"  {marker} {lines[j][:70]}")
            break  # Just show first marker

    print()
    print(f"{'='*70}")
    print("All Tests Passed! ✓")
    print(f"{'='*70}")
    print()

    return True


def main():
    """Main entry point."""
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description='Test PDF page marker preservation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific document
  %(prog)s --doc-id 0e4ec0213f3e540c9503efce61e58fe9

  # Test with custom database
  %(prog)s --doc-id 0e4ec0213f3e540c9503efce61e58fe9 --database skol_dev
"""
    )

    parser.add_argument(
        '--doc-id',
        required=True,
        help='Document ID to test'
    )

    parser.add_argument(
        '--database',
        default=None,
        help='Database name (default: from --couchdb-database or $COUCHDB_DATABASE or skol_dev)'
    )

    args, _ = parser.parse_known_args()

    # Use --database arg if provided, otherwise fall back to config
    database = args.database or config['couchdb_database'] or 'skol_dev'

    try:
        success = test_page_markers(args.doc_id, database)
        sys.exit(0 if success else 1)
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
