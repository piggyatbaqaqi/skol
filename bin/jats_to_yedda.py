#!/usr/bin/env python3
"""
Convert JATS/TaxPub XML documents from CouchDB to YEDDA-annotated text.

Reads JATS XML stored as article.xml attachments by PensoftIngestor
and produces YEDDA .txt.ann files suitable for classifier training.

Examples:
    # Print YEDDA for one article to stdout
    python jats_to_yedda.py --doc-id 015a24f8e4df5fa5a55928b448287a1d

    # Write to a file
    python jats_to_yedda.py --doc-id 015a24f8e4df5fa5a55928b448287a1d \
        --output-to file --output-dir ./output

    # Save as attachment in skol_training CouchDB
    python jats_to_yedda.py --doc-id 015a24f8e4df5fa5a55928b448287a1d \
        --output-to couchdb

    # Process all JATS documents
    python jats_to_yedda.py --all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running as a script or as a module.
if __name__ == "__main__" and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    bin_dir = str(Path(__file__).resolve().parent)
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)

from env_config import get_env_config
from ingestors.jats_to_yedda import jats_xml_to_yedda


def _connect_db(config: Dict[str, Any], database: str) -> Any:
    """Connect to a CouchDB database."""
    import couchdb as couchdb_lib

    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server[database]


def _write_stdout(
    yedda_text: str, doc_id: str, verbosity: int,
) -> None:
    if verbosity >= 2:
        print(f"--- {doc_id} ---", file=sys.stderr)
    sys.stdout.write(yedda_text)


def _write_file(
    yedda_text: str,
    doc_id: str,
    output_dir: str,
    verbosity: int,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / f"{doc_id}.txt.ann"
    filepath.write_text(yedda_text, encoding="utf-8")
    if verbosity >= 1:
        print(f"Wrote {filepath}")


def _write_couchdb(
    yedda_text: str,
    source_doc: Dict[str, Any],
    doc_id: str,
    target_db: Any,
    verbosity: int,
) -> None:
    # Create or update document in training database.
    if doc_id in target_db:
        target_doc = target_db[doc_id]
    else:
        target_doc: Dict[str, Any] = {"_id": doc_id}

    # Copy metadata from source.
    for key in ("title", "doi", "authors", "year",
                "article_type", "publication_date"):
        if key in source_doc:
            target_doc[key] = source_doc[key]

    target_doc["source"] = "jats_to_yedda"
    target_doc["source_database"] = "skol_dev"
    target_db.save(target_doc)

    # Attach YEDDA text.
    target_db.put_attachment(
        target_doc,
        yedda_text.encode("utf-8"),
        filename="article.txt.ann",
        content_type="text/plain",
    )
    if verbosity >= 1:
        print(f"Saved {doc_id} to CouchDB")


def _output_exists(
    doc_id: str,
    output_to: str,
    output_dir: str,
    target_db: Any,
) -> bool:
    """Check whether output already exists for a document."""
    if output_to == "file":
        return (Path(output_dir) / f"{doc_id}.txt.ann").exists()
    if (output_to == "couchdb"
            and target_db is not None
            and doc_id in target_db):
        doc = target_db[doc_id]
        return "article.txt.ann" in doc.get("_attachments", {})
    return False


def _get_xml_attachment(
    source_db: Any, doc_id: str, verbosity: int,
) -> str:
    """Fetch the article.xml attachment from a CouchDB document."""
    attachment = source_db.get_attachment(doc_id, "article.xml")
    if attachment is None:
        raise ValueError(f"No article.xml attachment for {doc_id}")
    xml_bytes = attachment.read()
    return xml_bytes.decode("utf-8")


def _process_doc(
    source_db: Any,
    doc: Dict[str, Any],
    doc_id: str,
    output_to: str,
    output_dir: str,
    target_db: Any,
    verbosity: int,
) -> bool:
    """Process a single document. Returns True on success."""
    try:
        xml_string = _get_xml_attachment(
            source_db, doc_id, verbosity,
        )
    except ValueError as exc:
        if verbosity >= 1:
            print(f"Skipping {doc_id}: {exc}", file=sys.stderr)
        return False

    try:
        yedda_text = jats_xml_to_yedda(xml_string)
    except ValueError as exc:
        if verbosity >= 1:
            print(
                f"Skipping {doc_id}: {exc}", file=sys.stderr,
            )
        return False

    if output_to == "stdout":
        _write_stdout(yedda_text, doc_id, verbosity)
    elif output_to == "file":
        _write_file(yedda_text, doc_id, output_dir, verbosity)
    elif output_to == "couchdb":
        _write_couchdb(
            yedda_text, doc, doc_id, target_db, verbosity,
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JATS/TaxPub XML to YEDDA-annotated text."
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
        help="Process all JATS XML documents.",
    )

    # Database.
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Source CouchDB database (default: from env).",
    )

    # Output.
    parser.add_argument(
        "--output-to",
        choices=["stdout", "file", "couchdb"],
        default="stdout",
        help="Where to write output (default: stdout).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./yedda_output",
        help="Directory for file output (default: ./yedda_output).",
    )
    parser.add_argument(
        "--output-database",
        type=str,
        default=None,
        help=(
            "Target CouchDB database for couchdb output "
            "(default: from TRAINING_DATABASE env)."
        ),
    )

    # Work-skipping and partial computation options.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without writing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output (overrides --skip-existing).",
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
    source_db = _connect_db(config, database)

    # Target database for couchdb output.
    target_db = None
    if args.output_to == "couchdb":
        target_database = (
            args.output_database or config["training_database"]
        )
        target_db = _connect_db(config, target_database)

    # Collect document IDs to process.
    doc_ids: List[str] = []
    if args.doc_id:
        doc_ids = [args.doc_id]
    elif args.all:
        # Only include documents with JATS XML available.
        for row in source_db.view(
            "_all_docs", include_docs=True,
        ):
            doc = row.doc
            if (doc
                    and doc.get("xml_available", False)
                    and doc.get("xml_format") == "jats"):
                doc_ids.append(row.id)
        if verbosity >= 1:
            print(
                f"Found {len(doc_ids)} JATS XML documents "
                f"in {database}",
                file=sys.stderr,
            )

    if limit is not None:
        doc_ids = doc_ids[:limit]

    # Process.
    success = 0
    skipped = 0
    for doc_id in doc_ids:
        if dry_run:
            print(f"Would process: {doc_id}")
            continue

        # Skip-existing check.
        if skip_existing and not force:
            if _output_exists(
                doc_id, args.output_to,
                args.output_dir, target_db,
            ):
                if verbosity >= 2:
                    print(
                        f"Skipping {doc_id}: output exists",
                        file=sys.stderr,
                    )
                skipped += 1
                continue

        try:
            doc = source_db[doc_id]
        except Exception:
            if verbosity >= 1:
                print(
                    f"Skipping {doc_id}: not found in "
                    f"{database}",
                    file=sys.stderr,
                )
            skipped += 1
            continue

        if _process_doc(
            source_db, doc, doc_id,
            args.output_to, args.output_dir,
            target_db, verbosity,
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
