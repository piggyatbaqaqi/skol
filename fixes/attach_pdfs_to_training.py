#!/usr/bin/env python3
"""
Attach PDF files to skol_training documents that are missing them.

Reads each document's source_file field to locate the annotation file on disk,
then finds the corresponding PDF in the same directory and attaches it as
article.pdf.

PDF matching logic:
  1. Exact match: annotation stem + ".pdf" (e.g., s2.txt.ann -> s2.pdf)
  2. Substring match: PDF whose name contains the annotation stem
     (e.g., n3.txt.ann -> "Persoonia v16n3.pdf" contains "n3")
  3. Sole PDF: if the directory has exactly one PDF, use it
     (e.g., Mycotaxon volumes have one PDF shared by all articles)

Usage:
    python fixes/attach_pdfs_to_training.py --dry-run
    python fixes/attach_pdfs_to_training.py
    python fixes/attach_pdfs_to_training.py --doc-id SPECIFIC_DOC_ID
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config

# Root of the annotated data on disk
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "annotated"


def find_pdf_for_annotation(source_file: str, verbosity: int) -> Path | None:
    """Find the PDF file corresponding to an annotation's source_file path.

    Returns the absolute path to the PDF, or None if not found.
    """
    # source_file looks like "data/annotated/journals/Mycotaxon/Vol118/s6.txt.ann"
    # Strip leading "data/annotated/" to get relative path from DATA_ROOT
    rel = source_file
    prefix = "data/annotated/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]

    ann_path = DATA_ROOT / rel
    ann_dir = ann_path.parent
    # Stem: "s6" from "s6.txt.ann", "n3" from "n3.txt.ann"
    stem = ann_path.name.replace(".txt.ann", "")

    if not ann_dir.is_dir():
        if verbosity >= 2:
            print(f"  Directory not found: {ann_dir}", file=sys.stderr)
        return None

    pdfs = list(ann_dir.glob("*.pdf"))
    if not pdfs:
        if verbosity >= 2:
            print(f"  No PDFs in {ann_dir}", file=sys.stderr)
        return None

    # 1. Exact match
    exact = ann_dir / f"{stem}.pdf"
    if exact.exists():
        return exact

    # 2. Substring match
    matches = [p for p in pdfs if stem in p.name]
    if len(matches) == 1:
        return matches[0]

    # 3. Sole PDF in directory
    if len(pdfs) == 1:
        return pdfs[0]

    if verbosity >= 1:
        print(
            f"  Ambiguous: {len(pdfs)} PDFs in {ann_dir}, "
            f"none uniquely matching '{stem}'",
            file=sys.stderr,
        )
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Attach PDFs to skol_training documents missing article.pdf.",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="skol_training",
        help="CouchDB database name (default: skol_training).",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Process only this specific document ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without modifying documents.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output.",
    )

    args = parser.parse_args()
    verbosity = 0 if args.quiet else args.verbose

    config = get_env_config()

    import couchdb as couchdb_lib

    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    db = server[args.database]

    if verbosity >= 1:
        action = "DRY RUN: scanning" if args.dry_run else "Attaching PDFs to"
        print(f"{action} {args.database}...", file=sys.stderr)

    # Collect document IDs
    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        doc_ids = [
            row.id for row in db.view("_all_docs", include_docs=False)
            if not row.id.startswith("_design/")
        ]
        if verbosity >= 1:
            print(f"Found {len(doc_ids)} documents", file=sys.stderr)

    attached = 0
    skipped_has_pdf = 0
    skipped_no_source = 0
    skipped_no_pdf_found = 0
    errors = 0

    for doc_id in doc_ids:
        doc = db[doc_id]

        # Skip if already has article.pdf
        attachments = doc.get("_attachments", {})
        if "article.pdf" in attachments:
            skipped_has_pdf += 1
            if verbosity >= 3:
                print(f"  {doc_id}: already has article.pdf", file=sys.stderr)
            continue

        # Need source_file to find the PDF
        source_file = doc.get("source_file")
        if not source_file:
            skipped_no_source += 1
            if verbosity >= 2:
                print(f"  {doc_id}: no source_file field", file=sys.stderr)
            continue

        pdf_path = find_pdf_for_annotation(source_file, verbosity)
        if pdf_path is None:
            skipped_no_pdf_found += 1
            if verbosity >= 1:
                print(
                    f"  {doc_id}: no PDF found for {source_file}",
                    file=sys.stderr,
                )
            continue

        if args.dry_run:
            if verbosity >= 1:
                size_mb = pdf_path.stat().st_size / (1024 * 1024)
                print(
                    f"  {doc_id}: would attach {pdf_path.name} ({size_mb:.1f} MB)",
                    file=sys.stderr,
                )
            attached += 1
            continue

        # Attach the PDF
        try:
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            # Re-read doc to get latest _rev
            doc = db[doc_id]
            db.put_attachment(
                doc,
                pdf_content,
                filename='article.pdf',
                content_type='application/pdf',
            )
            attached += 1
            if verbosity >= 1:
                size_mb = len(pdf_content) / (1024 * 1024)
                print(
                    f"  {doc_id}: attached {pdf_path.name} ({size_mb:.1f} MB)",
                    file=sys.stderr,
                )
        except Exception as e:
            errors += 1
            print(f"  {doc_id}: ERROR attaching {pdf_path}: {e}", file=sys.stderr)

    if verbosity >= 1:
        prefix = "Would attach" if args.dry_run else "Attached"
        print(f"\n{prefix} PDFs to {attached} documents.", file=sys.stderr)
        print(f"Already had PDF: {skipped_has_pdf}", file=sys.stderr)
        print(f"No source_file: {skipped_no_source}", file=sys.stderr)
        print(f"No PDF found on disk: {skipped_no_pdf_found}", file=sys.stderr)
        if errors:
            print(f"Errors: {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
