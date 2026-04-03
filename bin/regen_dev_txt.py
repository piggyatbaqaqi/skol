#!/usr/bin/env python3
"""Regenerate article.txt in skol_dev for all docs linked from skol_training.

skol_dev documents that were ingested before PDFSectionExtractor matured may
carry article.txt files extracted with an older (often worse) tool.  Poor OCR
quality in the embedded PDF text means running-head lines come out garbled,
so fix_staging_yedda.py cannot match them to YEDDA blocks.

This script re-extracts article.txt from the article.pdf attachment for every
skol_dev document that is referenced by at least one skol_training record, then
writes the fresh text back to skol_dev.

After running this script, re-run fix_staging_yedda.py to pick up newly
matched page markers, then re-run yedda_to_brat.py to refresh the brat files.

Usage::

    python bin/regen_dev_txt.py [--dry-run] [-v]
    python bin/regen_dev_txt.py --dev-id ID [--dry-run] [-v]
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_PDF_ATTACHMENT = "article.pdf"
_TXT_ATTACHMENT = "article.txt"


def _extract_plaintext(pdf_bytes: bytes) -> Optional[str]:
    """Extract article.txt from PDF bytes using PDFSectionExtractor.

    Args:
        pdf_bytes: Raw PDF content.

    Returns:
        Extracted plaintext with ``--- PDF Page N Label L ---`` markers,
        or None if extraction fails.
    """
    try:
        from ingestors.extract_plaintext import plaintext_from_pdf
        return plaintext_from_pdf(pdf_bytes)
    except Exception as exc:
        print(f"    WARNING: extraction failed: {exc}", file=sys.stderr)
        return None


def _linked_dev_ids(training_db: Any) -> Dict[str, str]:
    """Return {training_id: skol_dev_id} for all training docs with a link.

    Args:
        training_db: CouchDB database object for skol_training.

    Returns:
        Dict mapping training doc ID → skol_dev_id.
    """
    linked: Dict[str, str] = {}
    for row in training_db.view("_all_docs", include_docs=False):
        if row.id.startswith("_design/"):
            continue
        doc = training_db[row.id]
        skol_dev_id = doc.get("skol_dev_id")
        if skol_dev_id:
            linked[row.id] = skol_dev_id
    return linked


def regen_dev_txt(
    training_db: Any,
    dev_db: Any,
    dry_run: bool = False,
    verbosity: int = 1,
    dev_id_filter: Optional[str] = None,
) -> Dict[str, int]:
    """Regenerate article.txt for all skol_dev docs linked from skol_training.

    Args:
        training_db: CouchDB database object for skol_training.
        dev_db: CouchDB database object for skol_dev.
        dry_run: If True, report what would be done without writing.
        verbosity: 0 = quiet, 1 = one line per doc, 2 = full detail.
        dev_id_filter: If set, only process this skol_dev document ID.

    Returns:
        Summary counts: docs_found, docs_updated, docs_skipped_no_pdf,
        docs_skipped_no_txt, docs_skipped_filter.
    """
    linked = _linked_dev_ids(training_db)

    totals: Dict[str, int] = {
        "docs_found": 0,
        "docs_updated": 0,
        "docs_skipped_no_pdf": 0,
        "docs_skipped_no_txt": 0,
        "docs_skipped_filter": 0,
    }

    # Deduplicate: multiple training docs may share one skol_dev_id (issue PDFs).
    seen_dev_ids: set = set()

    for training_id, skol_dev_id in linked.items():
        if dev_id_filter and skol_dev_id != dev_id_filter:
            totals["docs_skipped_filter"] += 1
            continue
        if skol_dev_id in seen_dev_ids:
            continue
        seen_dev_ids.add(skol_dev_id)

        totals["docs_found"] += 1

        if verbosity >= 1:
            print(
                f"  {skol_dev_id}"
                f" (training: {training_id}) ...",
                file=sys.stderr,
            )

        # Get PDF from training (the training copy is authoritative; the
        # skol_dev copy should be identical but training is the source).
        pdf_att = training_db.get_attachment(training_id, _PDF_ATTACHMENT)
        if pdf_att is None:
            if verbosity >= 1:
                print(
                    "    SKIP: no article.pdf in skol_training.",
                    file=sys.stderr,
                )
            totals["docs_skipped_no_pdf"] += 1
            continue

        pdf_bytes = pdf_att.read()

        plaintext = _extract_plaintext(pdf_bytes)
        if plaintext is None:
            if verbosity >= 1:
                print(
                    "    SKIP: text extraction failed.",
                    file=sys.stderr,
                )
            totals["docs_skipped_no_txt"] += 1
            continue

        if verbosity >= 2:
            n_pages = plaintext.count("--- PDF Page ")
            print(
                f"    Extracted {len(plaintext):,} chars,"
                f" {n_pages} page marker(s).",
                file=sys.stderr,
            )

        if dry_run:
            if verbosity >= 1:
                print(
                    "    DRY RUN: would update article.txt.",
                    file=sys.stderr,
                )
            totals["docs_updated"] += 1
            continue

        dev_db.put_attachment(
            dev_db[skol_dev_id],
            plaintext.encode("utf-8"),
            filename=_TXT_ATTACHMENT,
            content_type="text/plain",
        )
        totals["docs_updated"] += 1
        if verbosity >= 1:
            print(
                f"    Updated skol_dev/{skol_dev_id} article.txt.",
                file=sys.stderr,
            )

    return totals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: regenerate article.txt for linked skol_dev docs."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate article.txt in skol_dev for all docs linked from"
            " skol_training, using the current PDF text extractor."
        )
    )
    parser.add_argument(
        "--training-db",
        default="skol_training",
        help=(
            "CouchDB database to read PDFs from"
            " (default: skol_training)."
        ),
    )
    parser.add_argument(
        "--dev-db",
        default="skol_dev",
        help=(
            "CouchDB database to update article.txt in"
            " (default: skol_dev)."
        ),
    )
    parser.add_argument(
        "--dev-id",
        help="Process only this skol_dev document ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v = per-doc, -vv = page counts).",
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
            f"{mode}Regenerating article.txt in '{args.dev_db}'"
            f" from '{args.training_db}'",
            file=sys.stderr,
        )

    summary = regen_dev_txt(
        training_db=training_db,
        dev_db=dev_db,
        dry_run=args.dry_run,
        verbosity=verbosity,
        dev_id_filter=args.dev_id,
    )

    if verbosity >= 1:
        print(
            f"\nDone: {summary['docs_found']} skol_dev docs found,"
            f" {summary['docs_updated']} updated,"
            f" {summary['docs_skipped_no_pdf']} skipped (no PDF),"
            f" {summary['docs_skipped_no_txt']} skipped (extraction failed).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
