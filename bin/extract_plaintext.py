#!/usr/bin/env python3
"""
Extract plaintext from articles in CouchDB and save as article.txt attachments.

Supports multiple plaintext sources: PDF, JATS XML, BioC JSON, and
NCBI E-utilities efetch. The 'auto' source tries each in priority order.

Examples:
    # Extract from PDFs for all documents
    python extract_plaintext.py --source pdf --skip-existing

    # Extract from JATS XML for one document
    python extract_plaintext.py --source jats --doc-id abc123

    # Download plaintext from NCBI for PMC articles
    python extract_plaintext.py --source efetch --limit 50

    # Auto-detect best source for each document
    python extract_plaintext.py --source auto --skip-existing
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script or as a module.
if __name__ == "__main__" and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    bin_dir = str(Path(__file__).resolve().parent)
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)

from env_config import get_env_config
from ingestors.extract_plaintext import (
    plaintext_from_bioc,
    plaintext_from_efetch,
    plaintext_from_jats,
    plaintext_from_pdf,
)

# Source priority for --source auto (highest to lowest).
_AUTO_SOURCES = ["efetch", "pdf", "jats", "bioc"]


def _connect_db(config: Dict[str, Any], database: str) -> Any:
    """Connect to a CouchDB database."""
    import couchdb as couchdb_lib

    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server[database]


def _has_plaintext(db: Any, doc_id: str) -> bool:
    """Check if a document already has an article.txt attachment."""
    try:
        doc = db[doc_id]
        return "article.txt" in doc.get("_attachments", {})
    except Exception:
        return False


def _extract_from_source(
    db: Any,
    doc: Dict[str, Any],
    doc_id: str,
    source: str,
    config: Dict[str, Any],
    verbosity: int,
) -> Optional[str]:
    """Try to extract plaintext from a single source.

    Returns the extracted text, or None if the source is unavailable.
    """
    if source == "pdf":
        attachment = db.get_attachment(doc_id, "article.pdf")
        if attachment is None:
            if verbosity >= 2:
                print(f"  {doc_id}: no article.pdf", file=sys.stderr)
            return None
        pdf_bytes = attachment.read()
        try:
            return plaintext_from_pdf(pdf_bytes)
        except ImportError:
            if verbosity >= 1:
                print(
                    f"  {doc_id}: PyMuPDF not installed",
                    file=sys.stderr,
                )
            return None

    if source == "jats":
        if not (doc.get("xml_available") and doc.get("xml_format") == "jats"):
            if verbosity >= 2:
                print(f"  {doc_id}: no JATS XML", file=sys.stderr)
            return None
        attachment = db.get_attachment(doc_id, "article.xml")
        if attachment is None:
            if verbosity >= 2:
                print(
                    f"  {doc_id}: no article.xml attachment",
                    file=sys.stderr,
                )
            return None
        xml_string = attachment.read().decode("utf-8")
        try:
            return plaintext_from_jats(xml_string)
        except ValueError as exc:
            if verbosity >= 1:
                print(
                    f"  {doc_id}: JATS extraction failed: {exc}",
                    file=sys.stderr,
                )
            return None

    if source == "bioc":
        bioc_json = doc.get("bioc_json")
        if not bioc_json:
            if verbosity >= 2:
                print(f"  {doc_id}: no bioc_json", file=sys.stderr)
            return None
        try:
            return plaintext_from_bioc(bioc_json)
        except ValueError as exc:
            if verbosity >= 1:
                print(
                    f"  {doc_id}: BioC extraction failed: {exc}",
                    file=sys.stderr,
                )
            return None

    if source == "efetch":
        pmcid = doc.get("pmcid")
        if not pmcid:
            if verbosity >= 2:
                print(f"  {doc_id}: no pmcid", file=sys.stderr)
            return None
        try:
            return plaintext_from_efetch(
                pmcid,
                api_key=config.get("ncbi_api_key"),
            )
        except ValueError as exc:
            if verbosity >= 1:
                print(
                    f"  {doc_id}: efetch failed: {exc}",
                    file=sys.stderr,
                )
            return None

    return None


def _extract_auto(
    db: Any,
    doc: Dict[str, Any],
    doc_id: str,
    config: Dict[str, Any],
    verbosity: int,
) -> Optional[str]:
    """Try each source in priority order, return first success."""
    for source in _AUTO_SOURCES:
        text = _extract_from_source(
            db, doc, doc_id, source, config, verbosity,
        )
        if text:
            if verbosity >= 2:
                print(
                    f"  {doc_id}: extracted from {source}",
                    file=sys.stderr,
                )
            return text
    return None


def _save_plaintext(
    db: Any,
    doc: Dict[str, Any],
    text: str,
    verbosity: int,
) -> None:
    """Save plaintext as article.txt attachment on a CouchDB document."""
    db.put_attachment(
        doc,
        text.encode("utf-8"),
        filename="article.txt",
        content_type="text/plain",
    )
    if verbosity >= 2:
        print(
            f"  Saved article.txt ({len(text)} chars)",
            file=sys.stderr,
        )


def _process_doc(
    db: Any,
    doc: Dict[str, Any],
    doc_id: str,
    source: str,
    config: Dict[str, Any],
    dry_run: bool,
    verbosity: int,
) -> bool:
    """Process a single document. Returns True on success."""
    if source == "auto":
        text = _extract_auto(db, doc, doc_id, config, verbosity)
    else:
        text = _extract_from_source(
            db, doc, doc_id, source, config, verbosity,
        )

    if not text:
        if verbosity >= 1:
            print(f"Skipping {doc_id}: no text from {source}",
                  file=sys.stderr)
        return False

    if dry_run:
        print(f"Would save article.txt for {doc_id} ({len(text)} chars)")
        return True

    _save_plaintext(db, doc, text, verbosity)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract plaintext and save as article.txt attachments.",
    )

    # Document selection (mutually exclusive).
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--doc-id",
        type=str,
        help="CouchDB document ID to process.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all documents in the database.",
    )

    # Source.
    parser.add_argument(
        "--source",
        choices=["pdf", "jats", "bioc", "efetch", "auto"],
        default="auto",
        help=(
            "Plaintext source (default: auto). "
            "auto tries: efetch > pdf > jats > bioc."
        ),
    )

    # Database.
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="CouchDB database (default: from env).",
    )

    # Work-skipping options.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without writing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have article.txt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing article.txt (overrides --skip-existing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N documents.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output.",
    )

    args = parser.parse_args()

    verbosity = 0 if args.quiet else args.verbose
    config = get_env_config()

    # Merge env_config defaults with explicit CLI flags.
    skip_existing = (
        args.skip_existing or config.get("skip_existing", False)
    )
    force = args.force or config.get("force", False)
    dry_run = args.dry_run or config.get("dry_run", False)
    limit = (
        args.limit if args.limit is not None
        else config.get("limit")
    )

    database = args.database or config["couchdb_database"]
    db = _connect_db(config, database)

    # Collect document IDs to process.
    doc_ids: List[str] = []
    if args.doc_id:
        doc_ids = [args.doc_id]
    elif args.all:
        for row in db.view("_all_docs", include_docs=False):
            if not row.id.startswith("_design/"):
                doc_ids.append(row.id)
        if verbosity >= 1:
            print(
                f"Found {len(doc_ids)} documents in {database}",
                file=sys.stderr,
            )

    if limit is not None:
        doc_ids = doc_ids[:limit]

    # Process.
    success = 0
    skipped = 0
    for doc_id in doc_ids:
        # Skip-existing check.
        if skip_existing and not force:
            if _has_plaintext(db, doc_id):
                if verbosity >= 2:
                    print(
                        f"Skipping {doc_id}: article.txt exists",
                        file=sys.stderr,
                    )
                skipped += 1
                continue

        try:
            doc = db[doc_id]
        except Exception:
            if verbosity >= 1:
                print(
                    f"Skipping {doc_id}: not found in {database}",
                    file=sys.stderr,
                )
            skipped += 1
            continue

        if _process_doc(
            db, doc, doc_id, args.source,
            config, dry_run, verbosity,
        ):
            success += 1
        else:
            skipped += 1

    if verbosity >= 1 and not dry_run and len(doc_ids) > 1:
        print(
            f"\nProcessed: {success}, Skipped: {skipped}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
