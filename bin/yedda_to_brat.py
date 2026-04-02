#!/usr/bin/env python3
"""Convert YEDDA-annotated text to brat standoff format.

brat standoff uses two files:
  article.txt  — plain text, one token/sentence per line or free-form
  article.ann  — annotation lines, one entity per line:
                 T{n}\\t{Label} {start} {end}\\t{text}
                 #N\\tAnnotatorNotes T{n}\\twas: {old_tag}  (LLM change notes)

Character offsets are zero-based and span [start, end) (end is exclusive),
matching brat's convention.

Core functions are pure (no I/O) for testability.

Usage (file mode):
    python bin/yedda_to_brat.py INPUT.ann OUTPUT_DIR [--changes JSONL] [-v]

Usage (CouchDB batch mode):
    python bin/yedda_to_brat.py --staging-db NAME --output-dir DIR [-v]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_ANN_ATTACHMENT = "article.txt.ann"
_CHANGES_ATTACHMENT = "changes.json"

# ---------------------------------------------------------------------------
# YEDDA block parser
# ---------------------------------------------------------------------------

_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)


def add_notes(
    ann: str,
    changes: List[Dict[str, Any]],
) -> str:
    """Append AnnotatorNotes lines to a brat .ann string.

    Each change dict must have ``block_index`` (0-based) and ``old_tag``.
    The corresponding brat entity is ``T{block_index + 1}``.

    Args:
        ann: Existing brat annotation string (T-lines).
        changes: List of change dicts from llm_relabel diff_yedda.

    Returns:
        ann string with ``#N\\tAnnotatorNotes T{n}\\twas: {old_tag}`` lines
        appended, one per changed block.  Returns ann unchanged if changes
        is empty.
    """
    if not changes:
        return ann
    note_lines: List[str] = []
    for note_idx, change in enumerate(changes, start=1):
        entity_id = change["block_index"] + 1
        old_tag = change["old_tag"]
        note_lines.append(
            f"#{note_idx}\tAnnotatorNotes T{entity_id}\twas: {old_tag}"
        )
    separator = "\n" if ann else ""
    return ann + separator + "\n".join(note_lines)


def yedda_to_brat(
    yedda_text: str,
    changes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """Convert YEDDA-annotated text to brat plaintext + annotation strings.

    Args:
        yedda_text: YEDDA-annotated string (``[@text#Tag*]`` blocks).
        changes: Optional list of change dicts from llm_relabel.  When
            provided, AnnotatorNotes lines are appended to the ann string
            so reviewers see which blocks the LLM relabeled and what the
            original tag was.

    Returns:
        Tuple of (plaintext, ann) where:
          - plaintext is block texts joined by ``\\n\\n``
          - ann is a brat standoff annotation string (T-lines, then
            optional #-lines), empty string if no blocks
    """
    blocks: List[Tuple[str, str]] = []
    for match in _YEDDA_BLOCK_RE.finditer(yedda_text):
        text = match.group(1)
        tag = match.group(2).strip()
        if not text:  # skip empty blocks — they produce zero-length brat spans
            continue
        blocks.append((text, tag))

    if not blocks:
        return "", ""

    # Build plaintext: blocks joined by "\n\n" (two-char separator).
    plaintext = "\n\n".join(text for text, _ in blocks)

    # Compute character offsets for each block.
    ann_lines: List[str] = []
    offset = 0
    for i, (text, tag) in enumerate(blocks, start=1):
        start = offset
        end = offset + len(text)
        escaped = text.replace("\n", "\\n")
        ann_lines.append(f"T{i}\t{tag} {start} {end}\t{escaped}")
        offset = end + 2  # skip the "\n\n" separator

    ann = "\n".join(ann_lines)
    if changes:
        ann = add_notes(ann, changes)
    return plaintext, ann


# ---------------------------------------------------------------------------
# CouchDB batch conversion
# ---------------------------------------------------------------------------

def convert_staging_db(
    staging_db: Any,
    output_dir: Path,
    verbose: bool = False,
) -> int:
    """Convert all documents in a staging DB to brat standoff files.

    For each document, fetches ``article.txt.ann`` and (if present)
    ``changes.json``, then writes ``{doc_id}.txt`` and ``{doc_id}.ann``
    to output_dir.

    Args:
        staging_db: CouchDB database object.
        output_dir: Directory to write brat files into.
        verbose: Print progress to stderr.

    Returns:
        Number of documents converted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in staging_db.view("_all_docs", include_docs=False):
        doc_id = row.id
        if doc_id.startswith("_"):
            continue

        ann_att = staging_db.get_attachment(doc_id, _ANN_ATTACHMENT)
        if ann_att is None:
            continue
        yedda_text = ann_att.read().decode("utf-8")

        changes: Optional[List[Dict[str, Any]]] = None
        changes_att = staging_db.get_attachment(doc_id, _CHANGES_ATTACHMENT)
        if changes_att is not None:
            changes = json.loads(changes_att.read().decode("utf-8"))

        plaintext, ann = yedda_to_brat(yedda_text, changes)

        txt_path = output_dir / f"{doc_id}.txt"
        brat_ann_path = output_dir / f"{doc_id}.ann"
        txt_path.write_text(plaintext, encoding="utf-8")
        brat_ann_path.write_text(ann, encoding="utf-8")
        count += 1

        if verbose:
            n_notes = sum(1 for ln in ann.splitlines() if ln.startswith("#"))
            print(
                f"  {doc_id}: "
                f"{len(plaintext.splitlines())} lines, "
                f"{n_notes} note(s)",
                file=sys.stderr,
            )

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: convert YEDDA .ann file(s) to brat standoff files."""
    parser = argparse.ArgumentParser(
        description="Convert YEDDA-annotated text to brat standoff format."
    )

    # File mode (original positional interface)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input YEDDA .ann file (file mode).",
    )
    parser.add_argument(
        "output_dir_pos",
        nargs="?",
        type=Path,
        metavar="OUTPUT_DIR",
        help="Output directory for file mode.",
    )
    parser.add_argument(
        "--changes",
        type=Path,
        metavar="JSON",
        help=(
            "JSON file containing a list of change dicts "
            "(file mode only; CouchDB mode reads changes.json attachment)."
        ),
    )

    # CouchDB batch mode
    parser.add_argument(
        "--staging-db",
        metavar="NAME",
        help="CouchDB staging database to read from (batch mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Output directory (batch mode).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.staging_db:
        # CouchDB batch mode
        if args.output_dir is None:
            parser.error("--output-dir is required with --staging-db")
        import couchdb
        from env_config import get_env_config
        config = get_env_config()
        server = couchdb.Server(config["couchdb_url"])
        server.resource.credentials = (
            config["couchdb_username"],
            config["couchdb_password"],
        )
        staging_db = server[args.staging_db]
        n = convert_staging_db(staging_db, args.output_dir, args.verbose)
        if args.verbose:
            print(f"Converted {n} document(s) to {args.output_dir}",
                  file=sys.stderr)

    else:
        # File mode
        if args.input is None:
            parser.error("INPUT is required in file mode")
        output_dir = args.output_dir_pos or args.output_dir
        if output_dir is None:
            parser.error("OUTPUT_DIR is required in file mode")

        yedda_text = args.input.read_text(encoding="utf-8")
        changes: Optional[List[Dict[str, Any]]] = None
        if args.changes:
            changes = json.loads(args.changes.read_text(encoding="utf-8"))

        plaintext, ann = yedda_to_brat(yedda_text, changes)

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.input.stem  # e.g. "article.txt" from "article.txt.ann"
        txt_path = output_dir / stem
        ann_path = output_dir / (stem + ".ann")

        txt_path.write_text(plaintext, encoding="utf-8")
        ann_path.write_text(ann, encoding="utf-8")

        if args.verbose:
            block_count = sum(
                1 for ln in ann.splitlines() if ln.startswith("T")
            )
            note_count = sum(
                1 for ln in ann.splitlines() if ln.startswith("#")
            )
            print(
                f"Wrote {block_count} entities, {note_count} note(s) "
                f"to {ann_path}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
