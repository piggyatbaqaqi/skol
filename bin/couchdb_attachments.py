#!/usr/bin/env python3
"""Bulk extract/insert CouchDB attachments to/from a directory tree.

Directory layout::

    DIR/
        <doc_id_1>/
            article.txt
            article.txt.ann
            article.pdf
        <doc_id_2>/
            ...

Usage::

    # Extract every document's attachments from a database.
    python bin/couchdb_attachments.py extract \\
        --database skol_training --dir /tmp/exports

    # Extract specific documents only.
    python bin/couchdb_attachments.py extract \\
        --database skol_training --dir /tmp/exports \\
        --doc-id abc --doc-id def

    # Insert (or refresh) attachments back into a database.  Each
    # subdirectory of --dir whose name is a CouchDB doc_id has its
    # files uploaded as attachments.  Documents are created if they
    # don't already exist; existing attachments are overwritten.
    python bin/couchdb_attachments.py insert \\
        --database skol_training --dir /tmp/exports

Semantics match ``cp``: an insert always overwrites; a missing target
document is created.  Design documents (``_design/...``) are skipped.

Environment variables (or ~/.skol_env):
    COUCHDB_URL         CouchDB server URL
    COUCHDB_USER        CouchDB username
    COUCHDB_PASSWORD    CouchDB password
"""

import argparse
import logging
import mimetypes
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402


# ---------------------------------------------------------------------------
# CouchDB connection
# ---------------------------------------------------------------------------

def _connect_db(database: str) -> Any:
    """Connect to a CouchDB database via the project's env config."""
    import couchdb as couchdb_lib

    config = get_env_config()
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server[database]


# ---------------------------------------------------------------------------
# Doc-id selection
# ---------------------------------------------------------------------------

def select_doc_ids(db: Any, doc_ids: Optional[List[str]]) -> List[str]:
    """Return the list of doc IDs to process.

    If doc_ids is given, return them verbatim (no existence check — insert
    creates missing docs, extract surfaces a clear per-doc error).  Otherwise
    enumerate every non-design document in the database.
    """
    if doc_ids:
        return list(doc_ids)
    return [
        row.id
        for row in db.view("_all_docs", include_docs=False)
        if not row.id.startswith("_design/")
    ]


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_doc(db: Any, doc_id: str, out_dir: Path) -> int:
    """Write all attachments of doc_id into out_dir/<doc_id>/.

    Returns the number of attachments written.  No directory is created
    when the document has no attachments.
    """
    try:
        doc = db[doc_id]
    except Exception as exc:  # noqa: BLE001
        logging.warning("%s: cannot fetch document (%s)", doc_id, exc)
        return 0

    attachments = doc.get("_attachments", {}) or {}
    if not attachments:
        return 0

    target = out_dir / doc_id
    target.mkdir(parents=True, exist_ok=True)

    written = 0
    for name in attachments:
        att = db.get_attachment(doc_id, name)
        if att is None:
            logging.warning("%s/%s: attachment missing on read", doc_id, name)
            continue
        data = att.read()
        (target / name).write_bytes(data)
        written += 1
    return written


def cmd_extract(
    db: Any, dir_path: Path, doc_ids: Optional[List[str]], verbosity: int,
) -> int:
    """Extract attachments for all selected docs into dir_path."""
    dir_path.mkdir(parents=True, exist_ok=True)
    ids = select_doc_ids(db, doc_ids)
    total_files = 0
    for doc_id in ids:
        n = extract_doc(db, doc_id, dir_path)
        total_files += n
        if verbosity >= 1 and n > 0:
            print(f"  {doc_id}: {n} attachment(s) → {dir_path / doc_id}")
        elif verbosity >= 2:
            print(f"  {doc_id}: no attachments")
    print(
        f"Done: {len(ids)} document(s), {total_files} attachment(s) written"
    )
    return 0


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def _content_type_for(name: str) -> str:
    """Guess a content type from the filename extension."""
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


def _ensure_doc(db: Any, doc_id: str) -> Any:
    """Return doc_id's current document, creating a minimal one if missing."""
    try:
        return db[doc_id]
    except Exception:  # noqa: BLE001 — couchdb raises ResourceNotFound
        db.save({"_id": doc_id})
        return db[doc_id]


def insert_doc(db: Any, doc_id: str, source_dir: Path) -> int:
    """Upload every regular file under source_dir as a doc_id attachment.

    Hidden files (leading dot) and subdirectories are skipped.  Existing
    attachments with the same name are overwritten.  Returns the number
    of attachments written.
    """
    if not source_dir.is_dir():
        logging.warning(
            "%s: source directory %s not found — skipped", doc_id, source_dir
        )
        return 0

    files = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    if not files:
        return 0

    written = 0
    for path in files:
        # Always re-fetch the doc between attachments — CouchDB rev
        # changes after each put_attachment.
        doc = _ensure_doc(db, doc_id)
        db.put_attachment(
            doc,
            path.read_bytes(),
            filename=path.name,
            content_type=_content_type_for(path.name),
        )
        written += 1
    return written


def _iter_doc_dirs(
    dir_path: Path, doc_ids: Optional[List[str]],
) -> Iterable[Path]:
    """Yield each per-document subdirectory to process."""
    if doc_ids:
        for doc_id in doc_ids:
            yield dir_path / doc_id
        return
    for sub in sorted(dir_path.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            yield sub


def cmd_insert(
    db: Any, dir_path: Path, doc_ids: Optional[List[str]], verbosity: int,
) -> int:
    """Insert attachments from dir_path into the database."""
    if not dir_path.is_dir():
        print(f"Error: {dir_path} is not a directory.", file=sys.stderr)
        return 2
    total_files = 0
    total_docs = 0
    for sub in _iter_doc_dirs(dir_path, doc_ids):
        if not sub.is_dir():
            logging.warning("%s: not a directory — skipped", sub)
            continue
        doc_id = sub.name
        n = insert_doc(db, doc_id, sub)
        total_docs += 1
        total_files += n
        if verbosity >= 1:
            print(f"  {doc_id}: {n} attachment(s) uploaded")
    print(
        f"Done: {total_docs} document(s), {total_files} attachment(s) uploaded"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--database",
        required=True,
        metavar="DB",
        help="CouchDB database name.",
    )
    common.add_argument(
        "--dir",
        required=True,
        type=Path,
        metavar="DIR",
        dest="dir_path",
        help=(
            "Source directory (insert) or destination directory (extract). "
            "Subdirectories are named by CouchDB document ID."
        ),
    )
    common.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        metavar="ID",
        help="Process only this document ID (repeatable).",
    )
    common.add_argument(
        "-v", "--verbosity",
        action="count",
        default=1,
        help="Increase output verbosity (repeatable).",
    )

    sub.add_parser(
        "extract",
        parents=[common],
        help="Download attachments from CouchDB into DIR.",
    )
    sub.add_parser(
        "insert",
        parents=[common],
        help="Upload attachments from DIR into CouchDB.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.verbosity < 2 else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    db = _connect_db(args.database)
    if args.command == "extract":
        return cmd_extract(db, args.dir_path, args.doc_ids, args.verbosity)
    if args.command == "insert":
        return cmd_insert(db, args.dir_path, args.doc_ids, args.verbosity)
    return 2  # unreachable; argparse enforces required subcommand


if __name__ == "__main__":
    sys.exit(main())
