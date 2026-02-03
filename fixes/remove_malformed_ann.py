#!/usr/bin/env python3
"""
Remove Malformed Annotation Attachments

This script scans CouchDB for *.ann attachments that fail parsing due to
malformed annotation markup (e.g., "Label open not at start of line").

These malformed annotations typically occur when:
- The annotation markup [@...#Label*] spans multiple lines incorrectly
- OCR errors corrupted the annotation markers
- Manual editing introduced syntax errors

Usage:
    python remove_malformed_ann.py --database skol_dev --dry-run
    python remove_malformed_ann.py --database skol_dev
    python remove_malformed_ann.py --database skol_dev --doc-id SPECIFIC_DOC_ID

Options:
    --database      CouchDB database name
    --doc-id        Process only this specific document ID
    --dry-run       Preview without removing attachments
    --pattern       Attachment pattern to check (default: *.ann)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config


def validate_annotation(content: str, doc_id: str, att_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate annotation content by attempting to parse it.

    Args:
        content: The annotation file content
        doc_id: Document ID (for error messages)
        att_name: Attachment name (for error messages)

    Returns:
        Tuple of (is_valid, error_message_or_None)
    """
    from line import Line

    lines = content.split('\n')

    for line_num, line_text in enumerate(lines, 1):
        try:
            # Line constructor validates annotation markup and raises ValueError
            # if markers like [@ or #Label*] are malformed
            line_obj = Line(line_text)
        except ValueError as e:
            # Malformed annotation - return the error with line context
            return False, f"{doc_id}/{att_name}:{line_num}: {e}"

    return True, None


def process_document(
    db,
    doc_id: str,
    pattern: str = "*.ann",
    dry_run: bool = False,
    verbosity: int = 1
) -> Tuple[int, int, List[str]]:
    """
    Check and optionally remove malformed .ann attachments from a document.

    Returns:
        Tuple of (checked_count, removed_count, list_of_messages)
    """
    import fnmatch

    checked = 0
    removed = 0
    messages = []

    try:
        doc = db[doc_id]
        attachments = doc.get('_attachments', {})

        # Find matching attachments
        ann_attachments = []
        for att_name in attachments.keys():
            if fnmatch.fnmatch(att_name, pattern):
                ann_attachments.append(att_name)

        if not ann_attachments:
            return 0, 0, []

        for att_name in ann_attachments:
            checked += 1

            try:
                # Read attachment content
                content = db.get_attachment(doc_id, att_name).read().decode('utf-8')

                # Validate
                is_valid, error = validate_annotation(content, doc_id, att_name)

                if not is_valid:
                    if dry_run:
                        messages.append(f"[DRY RUN] Would remove {att_name}: {error}")
                        removed += 1
                    else:
                        # Remove the attachment
                        doc = db[doc_id]  # Refresh doc
                        db.delete_attachment(doc, att_name)
                        messages.append(f"Removed {att_name}: {error}")
                        removed += 1
                elif verbosity >= 2:
                    messages.append(f"OK: {att_name}")

            except Exception as e:
                messages.append(f"Error checking {att_name}: {e}")

        return checked, removed, messages

    except Exception as e:
        return 0, 0, [f"Error accessing document: {e}"]


def main():
    """Main entry point."""
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description='Remove malformed annotation attachments from CouchDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--database',
        default=None,
        help='CouchDB database name'
    )

    parser.add_argument(
        '--doc-id',
        help='Process only this specific document ID'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without removing attachments'
    )

    parser.add_argument(
        '--pattern',
        default='*.ann',
        help='Attachment pattern to check (default: *.ann)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all checked files, not just errors'
    )

    args, _ = parser.parse_known_args()

    database = args.database or config.get('ingest_database') or config.get('couchdb_database')
    if not database:
        parser.error("--database is required")

    verbosity = 2 if args.verbose else config.get('verbosity', 1)

    print(f"\n{'='*70}")
    print(f"Remove Malformed Annotation Attachments")
    print(f"{'='*70}")
    print(f"Database: {database}")
    print(f"CouchDB: {config['couchdb_url']}")
    print(f"Pattern: {args.pattern}")
    if args.doc_id:
        print(f"Document: {args.doc_id}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*70}\n")

    try:
        import couchdb

        server = couchdb.Server(config['couchdb_url'])
        if config['couchdb_username'] and config['couchdb_password']:
            server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])

        if database not in server:
            print(f"Error: Database '{database}' not found")
            return 1

        db = server[database]

        # Get documents to process
        if args.doc_id:
            doc_ids = [args.doc_id]
        else:
            doc_ids = []
            print(f"Scanning for documents with {args.pattern} attachments...")
            import fnmatch
            for doc_id in db:
                try:
                    doc = db[doc_id]
                    attachments = doc.get('_attachments', {})
                    for att_name in attachments.keys():
                        if fnmatch.fnmatch(att_name, args.pattern):
                            doc_ids.append(doc_id)
                            break
                except Exception:
                    continue

        if not doc_ids:
            print(f"No documents found with {args.pattern} attachments")
            return 0

        print(f"Found {len(doc_ids)} document(s) to check\n")

        total_checked = 0
        total_removed = 0
        total_errors = 0

        for idx, doc_id in enumerate(doc_ids, 1):
            checked, removed, messages = process_document(
                db=db,
                doc_id=doc_id,
                pattern=args.pattern,
                dry_run=args.dry_run,
                verbosity=verbosity
            )

            total_checked += checked
            total_removed += removed

            if messages:
                # Only show doc header if there are messages to show
                has_removals = any('Removed' in m or 'Would remove' in m for m in messages)
                has_errors = any('Error' in m for m in messages)

                if has_removals or has_errors or verbosity >= 2:
                    print(f"[{idx}/{len(doc_ids)}] {doc_id}")
                    for msg in messages:
                        print(f"  {msg}")
                    if has_errors:
                        total_errors += 1

        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"Documents scanned: {len(doc_ids)}")
        print(f"Attachments checked: {total_checked}")
        print(f"Malformed attachments {'found' if args.dry_run else 'removed'}: {total_removed}")
        if total_errors > 0:
            print(f"Errors: {total_errors}")
        if args.dry_run and total_removed > 0:
            print("\nThis was a DRY RUN - no attachments were removed.")
            print("Run without --dry-run to actually remove malformed attachments.")
        print()

        return 0 if total_errors == 0 else 1

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
