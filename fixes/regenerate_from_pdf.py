#!/usr/bin/env python3
"""
Regenerate Text and Update Annotations from PDF

This script performs a complete regeneration pipeline:
1. Extracts text from *.pdf attachments with "--- PDF Page N Label L ---" markers
2. Saves as *.txt attachments (replaces existing)
3. Updates *.txt.ann page markers to match *.txt (preserving YEDDA annotations)

This is useful when:
- PDFs were originally extracted without Label information
- You want to refresh the text extraction with updated page markers
- The *.ann files have markers but are missing Labels

Usage:
    python regenerate_from_pdf.py --database skol_dev --dry-run
    python regenerate_from_pdf.py --database skol_dev
    python regenerate_from_pdf.py --database skol_dev --doc-id SPECIFIC_DOC_ID

Options:
    --database      CouchDB database name
    --doc-id        Process only this specific document ID
    --dry-run       Preview without saving changes
    --skip-txt      Skip regenerating *.txt (only update *.ann from existing *.txt)
    --skip-ann      Skip updating *.ann (only regenerate *.txt)
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import constants
from pdf_section_extractor import PDFSectionExtractor
from env_config import get_env_config


def extract_markers_with_labels(text: str) -> List[Tuple[int, int, Optional[str]]]:
    """
    Extract PDF page markers and their labels.

    Returns:
        List of tuples (line_number, page_number, label_or_None)
    """
    markers = []
    lines = text.split('\n')
    pattern = re.compile(constants.pdf_page_pattern)

    for i, line in enumerate(lines):
        match = pattern.match(line.strip())
        if match:
            page_num = int(match.group(1))
            label = match.group(3)  # May be None if no label
            markers.append((i, page_num, label))

    return markers


def update_ann_markers_from_txt(ann_content: str, txt_content: str) -> Tuple[str, int]:
    """
    Update page markers in .ann content to match those in .txt content.

    This handles two cases:
    1. Markers exist in .ann but are missing Labels -> add Labels from .txt
    2. Markers missing from .ann entirely -> copy from .txt (using content matching)

    Args:
        ann_content: Content of .txt.ann file
        txt_content: Content of .txt file with correct markers

    Returns:
        Tuple of (updated_ann_content, number_of_changes)
    """
    txt_markers = extract_markers_with_labels(txt_content)
    ann_markers = extract_markers_with_labels(ann_content)

    # Build a map of page_number -> (label, full_marker_text) from txt
    txt_page_info: Dict[int, Tuple[Optional[str], str]] = {}
    txt_lines = txt_content.split('\n')
    pattern = re.compile(constants.pdf_page_pattern)

    for line_num, page_num, label in txt_markers:
        marker_line = txt_lines[line_num].strip()
        txt_page_info[page_num] = (label, marker_line)

    # Check what needs to be updated
    ann_pages = {p for _, p, _ in ann_markers}
    txt_pages = {p for _, p, _ in txt_markers}

    # Case 1: All markers present in ann, just need label updates
    if ann_pages == txt_pages:
        # Update existing markers to add labels
        ann_lines = ann_content.split('\n')
        updated_lines = []
        changes = 0

        for line in ann_lines:
            match = pattern.match(line.strip())
            if match:
                page_num = int(match.group(1))
                existing_label = match.group(3)

                if page_num in txt_page_info:
                    txt_label, txt_marker = txt_page_info[page_num]
                    # If txt has a label and ann doesn't, use txt marker
                    if txt_label and not existing_label:
                        updated_lines.append(txt_marker)
                        changes += 1
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        return '\n'.join(updated_lines), changes

    # Case 2: Markers missing from ann - need to copy them using content matching
    # This is more complex - use the copy_markers_to_annotated logic
    return copy_markers_to_annotated(txt_content, ann_content)


def copy_markers_to_annotated(txt_content: str, ann_content: str) -> Tuple[str, int]:
    """
    Copy PDF page markers from txt to ann content using content matching.
    Preserves YEDDA annotations.

    Returns:
        Tuple of (updated_content, number_of_markers_added)
    """
    # Strip any existing page markers from ann_content first
    ann_lines_orig = ann_content.split('\n')
    marker_pattern = re.compile(constants.pdf_page_pattern)
    ann_lines_clean = [line for line in ann_lines_orig if not marker_pattern.match(line.strip())]
    ann_content_clean = '\n'.join(ann_lines_clean)

    # Extract markers from txt file
    txt_lines = txt_content.split('\n')
    markers = []
    for i, line in enumerate(txt_lines):
        if marker_pattern.match(line.strip()):
            markers.append((i, line.strip()))

    if not markers:
        return ann_content, 0

    # For each marker, find context lines after it
    marker_insertions = []
    for line_num, marker_text in markers:
        context_lines = []
        for j in range(line_num + 1, min(line_num + 20, len(txt_lines))):
            line = txt_lines[j].strip()
            if line and not marker_pattern.match(line):
                context_lines.append(line)
                if len(context_lines) >= 3:
                    break

        if not context_lines and line_num > 0:
            for j in range(line_num - 1, max(0, line_num - 20), -1):
                line = txt_lines[j].strip()
                if line and not marker_pattern.match(line):
                    context_lines.insert(0, line)
                    if len(context_lines) >= 3:
                        break

        marker_insertions.append((marker_text, context_lines, line_num))

    # Detect consecutive markers at start
    consecutive_start_markers = []
    for i, (marker_text, context_lines, line_num) in enumerate(marker_insertions):
        if i == 0:
            consecutive_start_markers.append((marker_text, context_lines, line_num))
        else:
            prev_line_num = marker_insertions[i-1][2]
            lines_between = txt_lines[prev_line_num + 1:line_num]
            non_marker_content = [l for l in lines_between
                                 if l.strip() and not marker_pattern.match(l.strip())]
            if not non_marker_content and consecutive_start_markers:
                consecutive_start_markers.append((marker_text, context_lines, line_num))
            else:
                break

    consecutive_start_line_nums = {l for _, _, l in consecutive_start_markers}
    markers_after_start = [m for m in marker_insertions if m[2] not in consecutive_start_line_nums]

    markers_with_content = [(m, c, l) for m, c, l in markers_after_start if c]

    # Build insertions map
    insertions_map = {}
    ann_lines = ann_content_clean.split('\n')

    # Strip YEDDA markers for matching
    cleaned_ann_lines = []
    for ann_line in ann_lines:
        cleaned = re.sub(r'\[@\s*', '', ann_line)
        cleaned = re.sub(r'\s*#[^\*]+\*\]', '', cleaned)
        cleaned_ann_lines.append(cleaned.strip())

    for marker_text, context_lines, orig_line_num in markers_with_content:
        search_text = context_lines[0][:60] if context_lines else ""
        best_match_idx = None

        for i, cleaned_line in enumerate(cleaned_ann_lines):
            if i in insertions_map:
                continue
            if search_text and cleaned_line.startswith(search_text[:40]):
                best_match_idx = i
                break

        if best_match_idx is not None:
            insertions_map[best_match_idx] = marker_text

    # Build result
    result = []

    # Add consecutive start markers first
    for marker_text, _, _ in sorted(consecutive_start_markers, key=lambda x: x[2]):
        result.append(marker_text)

    # Add content with markers inserted
    for i, line in enumerate(ann_lines):
        if i in insertions_map:
            result.append(insertions_map[i])
        result.append(line)

    markers_added = len(consecutive_start_markers) + len(insertions_map)
    return '\n'.join(result), markers_added


def process_document(
    db,
    doc_id: str,
    extractor: PDFSectionExtractor,
    skip_txt: bool = False,
    skip_ann: bool = False,
    dry_run: bool = False,
    verbosity: int = 1
) -> Tuple[bool, str]:
    """
    Process a single document: regenerate txt from pdf, update ann.

    Returns:
        Tuple of (success, message)
    """
    try:
        doc = db[doc_id]
        attachments = doc.get('_attachments', {})

        # Find attachments
        pdf_name = None
        txt_name = None
        ann_name = None

        for att_name in attachments.keys():
            if att_name.endswith('.pdf'):
                pdf_name = att_name
            elif att_name.endswith('.txt') and not att_name.endswith('.txt.ann'):
                txt_name = att_name
            elif att_name.endswith('.txt.ann'):
                ann_name = att_name

        if not pdf_name:
            return False, "No PDF attachment found"

        results = []

        # Step 1: Regenerate .txt from .pdf
        if not skip_txt:
            pdf_data = db.get_attachment(doc_id, pdf_name).read()
            new_txt_content = extractor.pdf_to_text(pdf_data)

            txt_attachment_name = pdf_name.rsplit('.', 1)[0] + '.txt'
            page_count = new_txt_content.count('--- PDF Page ')

            if dry_run:
                results.append(f"[DRY RUN] Would create {txt_attachment_name} ({len(new_txt_content):,} chars, {page_count} pages)")
            else:
                # Need to refresh doc after potential changes
                doc = db[doc_id]
                db.put_attachment(
                    doc,
                    new_txt_content.encode('utf-8'),
                    filename=txt_attachment_name,
                    content_type='text/plain; charset=utf-8'
                )
                results.append(f"Created {txt_attachment_name} ({page_count} pages)")

            txt_name = txt_attachment_name
            txt_content = new_txt_content
        else:
            # Read existing txt
            if not txt_name:
                return False, "No .txt attachment and --skip-txt specified"
            txt_content = db.get_attachment(doc_id, txt_name).read().decode('utf-8')

        # Step 2: Update .ann markers
        if not skip_ann and ann_name:
            ann_content = db.get_attachment(doc_id, ann_name).read().decode('utf-8')

            updated_ann, changes = update_ann_markers_from_txt(ann_content, txt_content)

            if changes > 0:
                if dry_run:
                    results.append(f"[DRY RUN] Would update {ann_name} ({changes} marker changes)")
                else:
                    # Refresh doc again
                    doc = db[doc_id]
                    db.put_attachment(
                        doc,
                        updated_ann.encode('utf-8'),
                        filename=ann_name,
                        content_type='text/plain; charset=utf-8'
                    )
                    results.append(f"Updated {ann_name} ({changes} marker changes)")
            else:
                results.append(f"{ann_name} already up to date")
        elif not skip_ann and not ann_name:
            results.append("No .ann attachment to update")

        return True, "; ".join(results)

    except Exception as e:
        import traceback
        if verbosity >= 2:
            traceback.print_exc()
        return False, str(e)


def main():
    """Main entry point."""
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description='Regenerate text from PDF and update annotation markers',
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
        help='Preview without saving changes'
    )

    parser.add_argument(
        '--skip-txt',
        action='store_true',
        help='Skip regenerating .txt (only update .ann from existing .txt)'
    )

    parser.add_argument(
        '--skip-ann',
        action='store_true',
        help='Skip updating .ann (only regenerate .txt)'
    )

    args, _ = parser.parse_known_args()

    database = args.database or config.get('ingest_database') or config.get('couchdb_database')
    if not database:
        parser.error("--database is required")

    print(f"\n{'='*70}")
    print(f"Regenerate from PDF")
    print(f"{'='*70}")
    print(f"Database: {database}")
    print(f"CouchDB: {config['couchdb_url']}")
    if args.doc_id:
        print(f"Document: {args.doc_id}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Steps: {'txt' if not args.skip_txt else ''} {'ann' if not args.skip_ann else ''}")
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

        # Create PDF extractor
        extractor = PDFSectionExtractor(
            couchdb_url=config['couchdb_url'],
            username=config['couchdb_username'],
            password=config['couchdb_password'],
            verbosity=config['verbosity']
        )

        # Get documents to process
        if args.doc_id:
            doc_ids = [args.doc_id]
        else:
            doc_ids = []
            print("Scanning for documents with PDF attachments...")
            for doc_id in db:
                try:
                    doc = db[doc_id]
                    attachments = doc.get('_attachments', {})
                    for att_name in attachments.keys():
                        if att_name.endswith('.pdf'):
                            doc_ids.append(doc_id)
                            break
                except Exception:
                    continue

        if not doc_ids:
            print("No documents found with PDF attachments")
            return 0

        print(f"Found {len(doc_ids)} document(s) to process\n")

        success_count = 0
        error_count = 0

        for idx, doc_id in enumerate(doc_ids, 1):
            print(f"[{idx}/{len(doc_ids)}] {doc_id}")

            success, message = process_document(
                db=db,
                doc_id=doc_id,
                extractor=extractor,
                skip_txt=args.skip_txt,
                skip_ann=args.skip_ann,
                dry_run=args.dry_run,
                verbosity=config['verbosity']
            )

            if success:
                print(f"  {message}")
                success_count += 1
            else:
                print(f"  ERROR: {message}")
                error_count += 1

        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"Total: {len(doc_ids)}")
        print(f"Success: {success_count}")
        print(f"Errors: {error_count}")
        if args.dry_run:
            print("\nThis was a DRY RUN - no changes were saved.")
        print()

        return 0 if error_count == 0 else 1

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
