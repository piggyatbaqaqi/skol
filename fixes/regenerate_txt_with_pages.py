#!/usr/bin/env python3
"""
Regenerate .txt Attachments with PDF Page Markers

This script regenerates all .txt attachments in a CouchDB database by:
1. Extracting text from PDF attachments
2. Adding "--- PDF Page N ---" markers between pages
3. Saving as .txt attachments (replaces existing ones)

The page markers are essential for proper page tracking in the extraction pipeline.

Usage:
    python regenerate_txt_with_pages.py --database skol_dev
    python regenerate_txt_with_pages.py --database skol_dev --pattern "*.pdf" --verbosity 2
    python regenerate_txt_with_pages.py --database skol_dev --doc-id SPECIFIC_DOC_ID
    python regenerate_txt_with_pages.py --database skol_dev --dry-run  # Preview without saving

Example:
    # Regenerate all .txt files for PDFs in skol_dev
    python regenerate_txt_with_pages.py --database skol_dev

    # Regenerate only for specific document
    python regenerate_txt_with_pages.py --database skol_dev --doc-id 0e4ec0213f3e540c9503efce61e58fe9

    # Preview what would be regenerated
    python regenerate_txt_with_pages.py --database skol_dev --dry-run
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from pdf_section_extractor import PDFSectionExtractor
from env_config import get_env_config


def regenerate_txt_files(
    database: str,
    couchdb_url: str,
    username: str,
    password: str,
    pattern: str = "*.pdf",
    doc_id: Optional[str] = None,
    verbosity: int = 1,
    dry_run: bool = False
) -> None:
    """
    Regenerate .txt files with PDF page markers.

    Args:
        database: CouchDB database name
        couchdb_url: CouchDB server URL
        username: CouchDB username
        password: CouchDB password
        pattern: Pattern for PDF attachments (default: *.pdf)
        doc_id: Optional specific document ID to process
        verbosity: Verbosity level (0=silent, 1=info, 2=debug)
        dry_run: If True, preview without saving
    """
    # Create PDF extractor with 'eager' save mode to always replace existing .txt
    extractor = PDFSectionExtractor(
        couchdb_url=couchdb_url,
        username=username,
        password=password,
        verbosity=verbosity,
        save_text='eager' if not dry_run else None  # Don't save in dry-run mode
    )

    print(f"\n{'='*70}")
    print(f"Regenerating .txt files with PDF page markers")
    print(f"{'='*70}")
    print(f"Database: {database}")
    print(f"CouchDB: {couchdb_url}")
    if doc_id:
        print(f"Document ID: {doc_id}")
    else:
        print(f"Pattern: {pattern}")
    if dry_run:
        print(f"Mode: DRY RUN (preview only, no changes will be saved)")
    else:
        print(f"Mode: LIVE (will save/replace .txt attachments)")
    print()

    # Connect to CouchDB
    import couchdb
    server = couchdb.Server(couchdb_url)
    if username and password:
        server.resource.credentials = (username, password)

    if database not in server:
        print(f"Error: Database '{database}' not found")
        sys.exit(1)

    db = server[database]

    # Process specific document or all documents
    if doc_id:
        doc_ids = [doc_id]
    else:
        # Get all documents with PDF attachments matching pattern
        doc_ids = []
        for doc_id_iter in db:
            try:
                doc = db[doc_id_iter]
                attachments = doc.get('_attachments', {})

                # Check if document has PDF attachments
                for att_name in attachments.keys():
                    if pattern == "*.pdf" and att_name.endswith('.pdf'):
                        doc_ids.append(doc_id_iter)
                        break
                    elif pattern.startswith("*.") and att_name.endswith(pattern[1:]):
                        doc_ids.append(doc_id_iter)
                        break
            except Exception:
                # Skip documents we can't read
                continue

    if not doc_ids:
        print("No documents found matching criteria")
        return

    print(f"Found {len(doc_ids)} document(s) to process\n")

    # Process each document
    success_count = 0
    error_count = 0

    for idx, doc_id_to_process in enumerate(doc_ids, 1):
        try:
            if verbosity >= 1:
                print(f"[{idx}/{len(doc_ids)}] Processing {doc_id_to_process}...")

            doc = db[doc_id_to_process]
            attachments = doc.get('_attachments', {})

            # Find PDF attachment
            pdf_attachment = None
            for att_name, att_info in attachments.items():
                if att_name.endswith('.pdf'):
                    pdf_attachment = att_name
                    break

            if not pdf_attachment:
                if verbosity >= 1:
                    print(f"  Skipping: No PDF attachment found")
                continue

            # Get PDF data
            pdf_data = db.get_attachment(doc_id_to_process, pdf_attachment).read()

            # Extract text with page markers
            text = extractor.pdf_to_text(pdf_data)

            # Determine .txt attachment name
            txt_attachment = pdf_attachment.rsplit('.', 1)[0] + '.txt'

            if dry_run:
                # Just show what would be done
                page_count = text.count('--- PDF Page ')
                print(f"  Would create/replace: {txt_attachment} ({len(text):,} chars, {page_count} pages)")
            else:
                # Save as .txt attachment
                text_bytes = text.encode('utf-8')
                db.put_attachment(
                    doc,
                    text_bytes,
                    filename=txt_attachment,
                    content_type='text/plain'
                )

                page_count = text.count('--- PDF Page ')
                if verbosity >= 1:
                    print(f"  Saved: {txt_attachment} ({len(text_bytes):,} bytes, {page_count} pages)")

            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"  Error processing {doc_id_to_process}: {e}")
            if verbosity >= 2:
                import traceback
                traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total documents: {len(doc_ids)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    if dry_run:
        print("\nThis was a dry run. No changes were saved.")
        print("Remove --dry-run to actually save the .txt files.")
    print()


def main():
    """Main entry point."""
    # Get environment configuration
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description='Regenerate .txt attachments with PDF page markers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate all .txt files in skol_dev
  %(prog)s --database skol_dev

  # Regenerate for specific document
  %(prog)s --database skol_dev --doc-id 0e4ec0213f3e540c9503efce61e58fe9

  # Preview without saving
  %(prog)s --database skol_dev --dry-run

  # With custom CouchDB server
  export COUCHDB_URL=http://myserver:5984
  %(prog)s --database skol_dev

Environment Variables:
  COUCHDB_URL          CouchDB server URL (default: http://localhost:5984)
  COUCHDB_USER         CouchDB username (default: admin)
  COUCHDB_PASSWORD     CouchDB password
"""
    )

    parser.add_argument(
        '--database',
        required=True,
        help='CouchDB database name (e.g., skol_dev)'
    )

    parser.add_argument(
        '--couchdb-url',
        default=config['couchdb_url'],
        help='CouchDB server URL (default: $COUCHDB_URL or http://localhost:5984)'
    )

    parser.add_argument(
        '--couchdb-username',
        default=config['couchdb_username'],
        help='CouchDB username (default: $COUCHDB_USER)'
    )

    parser.add_argument(
        '--couchdb-password',
        default=config['couchdb_password'],
        help='CouchDB password (default: $COUCHDB_PASSWORD)'
    )

    parser.add_argument(
        '--pattern',
        default='*.pdf',
        help='Pattern for PDF attachments (default: *.pdf)'
    )

    parser.add_argument(
        '--doc-id',
        help='Process only this specific document ID'
    )

    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Verbosity level (0=silent, 1=info, 2=debug, default: 1)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be done without actually saving'
    )

    args = parser.parse_args()

    try:
        regenerate_txt_files(
            database=args.database,
            couchdb_url=args.couchdb_url,
            username=args.couchdb_username,
            password=args.couchdb_password,
            pattern=args.pattern,
            doc_id=args.doc_id,
            verbosity=args.verbosity,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbosity >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
