#!/usr/bin/env python3
"""
Convert BioC-JSON documents from CouchDB to YEDDA-annotated text.

Reads BioC-JSON stored by PmcBiocIngestor and produces YEDDA .txt.ann
files suitable for classifier training.

Examples:
    # Print YEDDA for one article to stdout
    python bioc_to_yedda.py --pmcid PMC10858444

    # Write to a file
    python bioc_to_yedda.py --pmcid PMC10858444 --output-to file --output-dir ./output

    # Save as attachment in skol_training CouchDB
    python bioc_to_yedda.py --pmcid PMC10858444 --output-to couchdb

    # Process by CouchDB document ID
    python bioc_to_yedda.py --doc-id 0b529be2e5625b1c909bc548831009b6
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid5, NAMESPACE_URL

# Allow running as a script or as a module.
if __name__ == "__main__" and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    bin_dir = str(Path(__file__).resolve().parent)
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)

from env_config import get_env_config
from ingestors.bioc_to_yedda import bioc_json_to_yedda

PMC_ARTICLE_URL_TEMPLATE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC{}/"


def pmcid_to_doc_id(pmcid: str) -> str:
    """Convert a PMCID to a deterministic CouchDB document ID.

    Mirrors PmcBiocIngestor._make_doc_id.
    """
    # Strip leading "PMC" if present, then add it back for the URL.
    numeric = pmcid.lstrip("PMC")
    url = PMC_ARTICLE_URL_TEMPLATE.format(numeric)
    return uuid5(NAMESPACE_URL, url).hex


def _connect_db(config: Dict[str, Any], database: str) -> Any:
    """Connect to a CouchDB database."""
    import couchdb as couchdb_lib

    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server[database]


def _write_stdout(yedda_text: str, doc_id: str, verbosity: int) -> None:
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
    source_database: str,
    verbosity: int,
) -> None:
    # Create or update document in training database.
    if doc_id in target_db:
        target_doc = target_db[doc_id]
    else:
        target_doc: Dict[str, Any] = {"_id": doc_id}

    # Copy metadata from source.
    for key in ("title", "doi", "pmcid", "pmid", "license", "authors"):
        if key in source_doc:
            target_doc[key] = source_doc[key]

    target_doc["source"] = "bioc_to_yedda"
    target_doc["source_database"] = source_database
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
    if output_to == "couchdb" and target_db is not None and doc_id in target_db:
        doc = target_db[doc_id]
        return "article.txt.ann" in doc.get("_attachments", {})
    return False


def _process_doc(
    doc: Dict[str, Any],
    doc_id: str,
    output_to: str,
    output_dir: str,
    target_db: Any,
    source_database: str,
    verbosity: int,
) -> bool:
    """Process a single document. Returns True on success."""
    bioc_json = doc.get("bioc_json")
    if not bioc_json:
        if verbosity >= 1:
            print(f"Skipping {doc_id}: no bioc_json", file=sys.stderr)
        return False

    try:
        yedda_text = bioc_json_to_yedda(bioc_json)
    except ValueError as exc:
        if verbosity >= 1:
            print(f"Skipping {doc_id}: {exc}", file=sys.stderr)
        return False

    if output_to == "stdout":
        _write_stdout(yedda_text, doc_id, verbosity)
    elif output_to == "file":
        _write_file(yedda_text, doc_id, output_dir, verbosity)
    elif output_to == "couchdb":
        _write_couchdb(
            yedda_text, doc, doc_id, target_db, source_database, verbosity,
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BioC-JSON to YEDDA-annotated text."
    )

    # Document selection (mutually exclusive).
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pmcid",
        type=str,
        help="PMC ID to process (e.g., PMC10858444).",
    )
    group.add_argument(
        "--doc-id",
        type=str,
        help="CouchDB document ID to process.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all documents with bioc_json.",
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
        help="Target CouchDB database for couchdb output "
        "(default: from TRAINING_DATABASE env).",
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
        help="Skip documents that already have output in the target.",
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
    skip_existing = args.skip_existing or config.get("skip_existing", False)
    force = args.force or config.get("force", False)
    dry_run = args.dry_run or config.get("dry_run", False)
    limit = args.limit if args.limit is not None else config.get("limit")

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
    if args.pmcid:
        doc_ids = [pmcid_to_doc_id(args.pmcid)]
    elif args.doc_id:
        doc_ids = [args.doc_id]
    elif args.all:
        # Only include documents that have bioc_json available.
        for row in source_db.view("_all_docs", include_docs=True):
            doc = row.doc
            if doc and doc.get("bioc_json_available", False):
                doc_ids.append(row.id)
        if verbosity >= 1:
            print(
                f"Found {len(doc_ids)} documents with bioc_json in {database}",
                file=sys.stderr,
            )

    if limit is not None:
        doc_ids = doc_ids[: limit]

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
                doc_id, args.output_to, args.output_dir, target_db
            ):
                if verbosity >= 2:
                    print(f"Skipping {doc_id}: output already exists",
                          file=sys.stderr)
                skipped += 1
                continue

        try:
            doc = source_db[doc_id]
        except Exception:
            if verbosity >= 1:
                print(
                    f"Skipping {doc_id}: not found in {database}",
                    file=sys.stderr,
                )
            skipped += 1
            continue

        if _process_doc(
            doc, doc_id, args.output_to, args.output_dir,
            target_db, database, verbosity,
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
