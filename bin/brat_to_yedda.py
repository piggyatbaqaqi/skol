#!/usr/bin/env python3
"""Convert brat standoff annotations back to YEDDA format.

Reads a brat ``{doc_id}.txt`` + ``{doc_id}.ann`` pair from a brat data
directory and reconstructs the ``[@text#Tag*]`` YEDDA blocks, sorted by
character offset.

Non-entity annotation lines (relations ``R…``, attributes ``A…``,
notes ``#…``, etc.) are silently ignored — only ``T…`` entity lines
are converted.

Core functions are pure (no I/O) for testability.

Usage:
    # Upload directly to CouchDB (most common):
    python bin/brat_to_yedda.py --doc-id ID --database DB [-v]

    # Write to a local file instead:
    python bin/brat_to_yedda.py --doc-id ID [--output PATH] [-v]

Configuration (priority: CLI > env var > .skol_env > default):
    --brat-data-dir / BRAT_DATA_DIR   brat data directory
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

_ANN_ATTACHMENT = "article.txt.ann"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# brat entity line parser
# ---------------------------------------------------------------------------

# Matches: T<n>\t<Label> <start> <end>[\t<text>]  (text field is optional)
_BRAT_ENTITY_RE = re.compile(
    r"^T\d+\t([A-Za-z][A-Za-z0-9_-]{0,49})\s+(\d+)\s+(\d+)(?:\t(.*))?$"
)


def brat_to_yedda(plaintext: str, ann: str) -> str:
    """Convert brat standoff annotation to a YEDDA-annotated string.

    Extracts all ``T{n}`` entity lines from *ann*, sorts them by start
    offset, and emits one ``[@text#Tag*]`` block per entity, joined by
    ``\\n\\n``.

    Non-entity lines (``R…``, ``A…``, ``#…``) are ignored.

    Args:
        plaintext: The plain text that *ann* annotates.  Block text is
            recovered via character offsets (``plaintext[start:end]``),
            so literal newlines are preserved correctly even when the
            ann text field uses ``\\n`` escape sequences.
        ann: brat standoff annotation string.

    Returns:
        YEDDA-annotated string.  Returns ``""`` if *ann* contains no
        entity lines.
    """
    entities: List[Tuple[int, int, str]] = []  # (start, end, label)

    for line in ann.splitlines():
        m = _BRAT_ENTITY_RE.match(line.strip())
        if m is None:
            continue
        label = m.group(1)
        start = int(m.group(2))
        end = int(m.group(3))
        entities.append((start, end, label))

    if not entities:
        return ""

    entities.sort(key=lambda e: e[0])

    blocks = [f"[@{plaintext[start:end]}#{label}*]"
              for start, end, label in entities]
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: convert brat standoff files to a YEDDA .ann file."""
    from env_config import get_env_config
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description="Convert brat standoff annotations to YEDDA format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--doc-id",
        required=True,
        metavar="ID",
        help="Document ID — used to locate {ID}.txt and {ID}.ann in brat-data-dir.",
    )
    parser.add_argument(
        "--database",
        default=None,
        metavar="DB",
        help=(
            "Upload the YEDDA output as an article.txt.ann attachment to "
            "this CouchDB database.  Mutually exclusive with --output."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write YEDDA output to this local file.  "
            "Default (when --database is not given): {brat-data-dir}/{doc-id}.txt.ann"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.database and args.output:
        parser.error("--database and --output are mutually exclusive.")

    brat_dir = config["brat_data_dir"]
    txt_file = brat_dir / f"{args.doc_id}.txt"
    ann_file = brat_dir / f"{args.doc_id}.ann"

    for path in (txt_file, ann_file):
        if not path.exists():
            print(f"✗ File not found: {path}", file=sys.stderr)
            sys.exit(1)

    plaintext = txt_file.read_text(encoding="utf-8")
    ann = ann_file.read_text(encoding="utf-8")
    yedda = brat_to_yedda(plaintext, ann)
    block_count = yedda.count("[@")

    if args.database:
        import couchdb
        server = couchdb.Server(config["couchdb_url"])
        server.resource.credentials = (
            config["couchdb_username"],
            config["couchdb_password"],
        )
        db = server[args.database]
        doc = db.get(args.doc_id)
        if doc is None:
            db[args.doc_id] = {}
            doc = db[args.doc_id]
        db.put_attachment(doc, yedda.encode("utf-8"), filename=_ANN_ATTACHMENT,
                          content_type="text/plain")
        if args.verbose:
            print(
                f"Uploaded {block_count} YEDDA blocks to "
                f"{args.database}/{args.doc_id}/{_ANN_ATTACHMENT}",
                file=sys.stderr,
            )
    else:
        output_path = args.output or (brat_dir / f"{args.doc_id}.txt.ann")
        output_path.write_text(yedda, encoding="utf-8")
        if args.verbose:
            print(
                f"Wrote {block_count} YEDDA blocks to {output_path}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
