#!/usr/bin/env python3
"""Upload a YEDDA annotation file to CouchDB as a document attachment.

This is the final step in the brat annotation round-trip:

    article.txt.ann  (YEDDA, from CouchDB)
        ↓  bin/yedda_to_brat.py
    article.txt + article.ann  (brat standoff — annotate in brat)
        ↓  bin/brat_to_yedda.py
    article.txt.ann  (YEDDA, updated)
        ↓  THIS SCRIPT
    CouchDB attachment updated

Usage:
    python bin/upload_annotation.py DOC_ID FILE [options]

Example:
    python bin/upload_annotation.py abc123 article.txt.ann --database skol_dev
    python bin/upload_annotation.py abc123 /tmp/out/article.txt.ann --dry-run -v
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config


# ---------------------------------------------------------------------------
# Core functions (pure / mockable)
# ---------------------------------------------------------------------------

def resolve_attachment_name(
    file_path: Path,
    explicit_name: Optional[str] = None,
) -> str:
    """Return the CouchDB attachment name to use for *file_path*.

    Args:
        file_path: Local path to the annotation file.
        explicit_name: If given, use this name instead of the filename.

    Returns:
        Attachment name string (filename only, no directory component).
    """
    if explicit_name:
        return explicit_name
    return file_path.name


def upload_attachment(
    couchdb_url: str,
    db: str,
    doc_id: str,
    attachment_name: str,
    content: str,
    username: str,
    password: str,
    dry_run: bool = False,
    verbosity: int = 1,
) -> Optional[str]:
    """Upload *content* as a CouchDB attachment, replacing any existing version.

    Fetches the current document revision first (required by CouchDB), then
    PUTs the attachment at ``{couchdb_url}/{db}/{doc_id}/{attachment_name}``.

    Args:
        couchdb_url: Base URL of the CouchDB server (e.g. 'http://localhost:5984').
        db: Database name.
        doc_id: Document ``_id``.
        attachment_name: Name to store the attachment under.
        content: UTF-8 text content of the annotation file.
        username: CouchDB username.
        password: CouchDB password.
        dry_run: If True, skip the PUT and return None.
        verbosity: 0=silent, 1=info, 2=debug.

    Returns:
        New revision string on success, or None in dry-run mode.

    Raises:
        requests.HTTPError: If the GET or PUT request fails.
    """
    auth = (username, password)
    doc_url = f"{couchdb_url}/{db}/{doc_id}"

    # Fetch current revision.
    if verbosity >= 2:
        print(f"  GET {doc_url}")
    get_resp = requests.get(doc_url, auth=auth, timeout=30)
    get_resp.raise_for_status()
    rev = get_resp.json()["_rev"]
    if verbosity >= 2:
        print(f"  Current _rev: {rev}")

    attachment_url = f"{doc_url}/{attachment_name}"

    if dry_run:
        print(
            f"[DRY RUN] Would PUT {len(content)} chars → "
            f"{db}/{doc_id}/{attachment_name}  (rev={rev})"
        )
        return None

    if verbosity >= 2:
        print(f"  PUT {attachment_url}?rev={rev}")

    put_resp = requests.put(
        attachment_url,
        params={"rev": rev},
        data=content.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        auth=auth,
        timeout=30,
    )
    put_resp.raise_for_status()
    new_rev = put_resp.json().get("rev", "")
    if verbosity >= 1:
        print(f"✓ Uploaded {attachment_name} → {db}/{doc_id}  (new rev: {new_rev})")
    return new_rev


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: upload an annotation file to CouchDB."""
    parser = argparse.ArgumentParser(
        description="Upload a YEDDA annotation file to CouchDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Upload article.txt.ann to document abc123 in the default database:
    python upload_annotation.py abc123 article.txt.ann

  Specify database and preview without writing:
    python upload_annotation.py abc123 article.txt.ann --database skol_dev --dry-run

  Override the attachment name stored in CouchDB:
    python upload_annotation.py abc123 /tmp/work/article.txt.ann \\
        --attachment-name article.txt.ann
""",
    )
    parser.add_argument(
        "doc_id",
        help="CouchDB document _id of the ingest record.",
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Local path to the YEDDA annotation file to upload.",
    )
    parser.add_argument(
        "--database",
        metavar="DB",
        default=None,
        help="CouchDB database name (default: from env_config / COUCHDB_DATABASE).",
    )
    parser.add_argument(
        "--attachment-name",
        metavar="NAME",
        default=None,
        help=(
            "Name to store the attachment under in CouchDB "
            "(default: filename of FILE, e.g. 'article.txt.ann')."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without making any changes.",
    )
    parser.add_argument(
        "-v", "--verbosity",
        action="count",
        default=0,
        help="Increase verbosity (-v info, -v -v debug).",
    )
    args = parser.parse_args()

    # Validate input file.
    if not args.file.exists():
        print(f"✗ File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Load configuration.
    config = get_env_config()
    couchdb_url = f"http://{config['couchdb_host']}"
    username = config["couchdb_username"]
    password = config["couchdb_password"]
    db = args.database or config.get("couchdb_database", "skol_dev")

    attachment_name = resolve_attachment_name(args.file, args.attachment_name)
    content = args.file.read_text(encoding="utf-8")

    if args.verbosity >= 1:
        print(f"Document : {db}/{args.doc_id}")
        print(f"File     : {args.file}  ({len(content)} chars)")
        print(f"Attachment: {attachment_name}")
        if args.dry_run:
            print("Mode     : DRY RUN")
        print()

    try:
        upload_attachment(
            couchdb_url=couchdb_url,
            db=db,
            doc_id=args.doc_id,
            attachment_name=attachment_name,
            content=content,
            username=username,
            password=password,
            dry_run=args.dry_run,
            verbosity=args.verbosity,
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        print(f"✗ HTTP {status}: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"✗ Upload failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
