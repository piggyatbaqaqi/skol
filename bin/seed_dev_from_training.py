#!/usr/bin/env python3
"""Create skol_dev stubs for skol_training documents that lack skol_dev_id.

Some skol_training records were ingested before the skol_dev database existed
and were never linked to a skol_dev document.  Without a skol_dev record, these
treatments are invisible to the search index and end users.

For each skol_training document missing a skol_dev_id this script:

  1. Creates a new skol_dev document copying the available bibliographic
     metadata from skol_training.
  2. Copies the article.pdf attachment (required for PDF back-links).
  3. Extracts article.txt from the PDF using PDFSectionExtractor, which
     emits ``--- PDF Page N Label L ---`` page markers needed by the
     fix_staging_yedda page-marker recovery step.
  4. Writes both attachments to the new skol_dev document.
  5. Sets skol_training[doc_id].skol_dev_id to the new document's ID.

After this script runs, fix_staging_yedda.py will be able to recover page
markers for all 190 training documents symmetrically.

Usage::

    python bin/seed_dev_from_training.py [--dry-run] [-v]
"""

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Metadata fields to copy from skol_training into the new skol_dev document.
# Fields with None values are included only when non-null.
_COPY_FIELDS = (
    "pdf_url",
    "url",
    "journal",
    "volume",
    "number",
    "year",
    "author",
    "title",
    "pages",
    "issn",
    "eissn",
    "itemtype",
    "publishercode",
    "parent_itemid",
    "publication date",
)

_PDF_ATTACHMENT = "article.pdf"
_TXT_ATTACHMENT = "article.txt"


def _build_dev_doc(training_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Build the metadata dict for a new skol_dev document.

    Copies available bibliographic fields from a skol_training document.
    Fields absent from or null in *training_doc* are omitted.

    Args:
        training_doc: CouchDB document dict from skol_training.

    Returns:
        Dict suitable for saving as a new skol_dev document.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    doc: Dict[str, Any] = {
        "meta": {},
        "itemtype": "article",
        "create_time": now,
        "modification_time": now,
        "seeded_from_training": training_doc["_id"],
    }
    for field in _COPY_FIELDS:
        val = training_doc.get(field)
        if val is not None:
            doc[field] = val
    return doc


def _extract_plaintext(pdf_bytes: bytes) -> Optional[str]:
    """Extract article.txt text from PDF bytes using PDFSectionExtractor.

    Args:
        pdf_bytes: Raw PDF content.

    Returns:
        Extracted plaintext string with ``--- PDF Page N Label L ---``
        markers, or None if extraction fails.
    """
    try:
        from ingestors.extract_plaintext import plaintext_from_pdf
        return plaintext_from_pdf(pdf_bytes)
    except Exception as exc:
        print(f"    WARNING: PDF extraction failed: {exc}", file=sys.stderr)
        return None


def _find_docs_needing_seeding(training_db: Any) -> List[str]:
    """Return IDs of skol_training docs that lack skol_dev_id.

    Args:
        training_db: CouchDB database object for skol_training.

    Returns:
        List of document IDs.
    """
    missing: List[str] = []
    for row in training_db.view("_all_docs", include_docs=False):
        if row.id.startswith("_design/"):
            continue
        doc = training_db[row.id]
        if not doc.get("skol_dev_id"):
            missing.append(row.id)
    return missing


def seed_dev_from_training(
    training_db: Any,
    dev_db: Any,
    server: Any,
    dry_run: bool = False,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Seed skol_dev with stubs for all unlisted skol_training documents.

    Args:
        training_db: CouchDB database object for skol_training.
        dev_db: CouchDB database object for skol_dev.
        server: CouchDB server object (used to allocate UUIDs).
        dry_run: If True, report what would be done without writing.
        verbosity: 0 = quiet, 1 = one line per doc, 2 = full detail.

    Returns:
        Summary counts: docs_found, docs_seeded, docs_skipped_no_pdf,
        docs_skipped_no_text.
    """
    doc_ids = _find_docs_needing_seeding(training_db)
    totals: Dict[str, int] = {
        "docs_found": len(doc_ids),
        "docs_seeded": 0,
        "docs_skipped_no_pdf": 0,
        "docs_skipped_no_text": 0,
    }

    if verbosity >= 1:
        print(
            f"  Found {len(doc_ids)} skol_training doc(s)"
            " without skol_dev_id.",
            file=sys.stderr,
        )

    for doc_id in doc_ids:
        if verbosity >= 1:
            print(f"  Processing {doc_id} ...", file=sys.stderr)

        training_doc = training_db[doc_id]

        # Fetch PDF.
        pdf_att = training_db.get_attachment(doc_id, _PDF_ATTACHMENT)
        if pdf_att is None:
            if verbosity >= 1:
                print("    SKIP: no article.pdf attachment.", file=sys.stderr)
            totals["docs_skipped_no_pdf"] += 1
            continue

        pdf_bytes = pdf_att.read()

        # Extract plaintext.
        plaintext = _extract_plaintext(pdf_bytes)
        if plaintext is None:
            if verbosity >= 1:
                print(
                    "    SKIP: plaintext extraction failed.",
                    file=sys.stderr,
                )
            totals["docs_skipped_no_text"] += 1
            continue

        if verbosity >= 2:
            n_pages = plaintext.count("--- PDF Page ")
            print(
                f"    Extracted {len(plaintext)} chars, {n_pages} page(s).",
                file=sys.stderr,
            )

        if dry_run:
            if verbosity >= 1:
                print(
                    "    DRY RUN: would create skol_dev stub"
                    " and update skol_dev_id.",
                    file=sys.stderr,
                )
            totals["docs_seeded"] += 1
            continue

        # Allocate a new CouchDB UUID for the skol_dev document.
        new_id = server.uuids(1)[0]

        # Create the skol_dev document.
        dev_meta = _build_dev_doc(training_doc)
        dev_db[new_id] = dev_meta

        # Attach PDF.
        dev_db.put_attachment(
            dev_db[new_id],
            pdf_bytes,
            filename=_PDF_ATTACHMENT,
            content_type="application/pdf",
        )

        # Attach plaintext.
        dev_db.put_attachment(
            dev_db[new_id],
            plaintext.encode("utf-8"),
            filename=_TXT_ATTACHMENT,
            content_type="text/plain",
        )

        # Update skol_training with the new skol_dev_id.
        training_doc["skol_dev_id"] = new_id
        training_db.save(training_doc)

        totals["docs_seeded"] += 1
        if verbosity >= 1:
            print(
                f"    Created skol_dev/{new_id};"
                f" updated skol_training/{doc_id}.skol_dev_id.",
                file=sys.stderr,
            )

    return totals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: seed skol_dev from unlisted skol_training docs."""
    parser = argparse.ArgumentParser(
        description=(
            "Create skol_dev stubs for skol_training documents that lack"
            " skol_dev_id, so all training records are visible to the"
            " search index."
        )
    )
    parser.add_argument(
        "--training-db",
        default="skol_training",
        help=(
            "CouchDB database to read training documents from"
            " (default: skol_training)."
        ),
    )
    parser.add_argument(
        "--dev-db",
        default="skol_dev",
        help=(
            "CouchDB database to create stub documents in"
            " (default: skol_dev)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing any changes.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v = per-doc, -vv = full detail).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output.",
    )

    args = parser.parse_args()
    verbosity = 0 if args.quiet else args.verbose

    from env_config import get_env_config
    config = get_env_config()

    import couchdb as couchdb_lib
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )

    training_db = server[args.training_db]
    dev_db = server[args.dev_db]

    mode = "DRY RUN — " if args.dry_run else ""
    if verbosity >= 1:
        print(
            f"{mode}Seeding '{args.dev_db}' from '{args.training_db}'",
            file=sys.stderr,
        )

    summary = seed_dev_from_training(
        training_db=training_db,
        dev_db=dev_db,
        server=server,
        dry_run=args.dry_run,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print(
            f"\nDone: {summary['docs_found']} docs found without skol_dev_id, "
            f"{summary['docs_seeded']} seeded, "
            f"{summary['docs_skipped_no_pdf']} skipped (no PDF), "
            f"{summary['docs_skipped_no_text']} skipped (text failed).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
