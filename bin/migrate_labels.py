#!/usr/bin/env python3
"""Tier 1 automatic label migration for YEDDA annotations.

Applies rule-based relabeling to upgrade existing 8-tag YEDDA corpora
to the 12-tag scheme:

  - Holotype blocks split into Type-designation (short) or
    Materials-examined (long specimen lists).
  - Blocks starting with header keywords (Distribution, Diagnosis,
    Biology, etc.) relabeled from Misc-exposition or Description.

Core functions are pure (no CouchDB I/O) for testability.
The CLI handles database reads/writes.

Usage:
    python bin/migrate_labels.py --database skol_training [--dry-run] [-v]
    python bin/migrate_labels.py --experiment NAME [--dry-run] [-v]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestors.yedda_tags import Tag

# ---------------------------------------------------------------------------
# YEDDA parsing
# ---------------------------------------------------------------------------

_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)

# ---------------------------------------------------------------------------
# Header keyword → Tag table
# Checked only against the first sentence of a block.
# ---------------------------------------------------------------------------

_HEADER_RULES: List[Tuple[frozenset, Tag]] = [
    (frozenset({"distribution", "habitat", "range"}), Tag.DISTRIBUTION),
    (frozenset({"diagnosis"}), Tag.DIAGNOSIS),
    (frozenset({"biology", "ecology", "host"}), Tag.BIOLOGY),
]

# Tags eligible for header-keyword relabeling.
_KEYWORD_ELIGIBLE: frozenset = frozenset({
    Tag.MISC_EXPOSITION.value,
    Tag.DESCRIPTION.value,
})

# First-sentence extractor: up to (and including) the first .!?: or end-of-line.
_FIRST_SENTENCE_RE = re.compile(r'^([^.!?:]+[.!?:]?)')


# ---------------------------------------------------------------------------
# Core relabeling functions
# ---------------------------------------------------------------------------

def relabel_holotype(block_text: str) -> Tag:
    """Classify a Holotype block as Type-designation or Materials-examined.

    Counts non-empty lines:
      ≤ 2 non-empty lines → Type-designation (the "Holotype: NY 123" line).
      > 2 non-empty lines → Materials-examined (full specimen list).

    Args:
        block_text: Text content of the block.

    Returns:
        Tag.TYPE_DESIGNATION or Tag.MATERIALS_EXAMINED.
    """
    non_empty = [ln for ln in block_text.splitlines() if ln.strip()]
    return Tag.TYPE_DESIGNATION if len(non_empty) <= 2 else Tag.MATERIALS_EXAMINED


def relabel_by_header(block_text: str) -> Optional[Tag]:
    """Return a Tag if the block's first sentence matches a header keyword.

    Only the first sentence (text up to the first sentence-ending
    punctuation or colon, or the whole first line if none) is examined,
    so keywords that appear mid-paragraph are not matched.

    Args:
        block_text: Text content of the block.

    Returns:
        The matching Tag, or None if no keyword matches.
    """
    stripped = block_text.strip()
    if not stripped:
        return None

    first_line = stripped.splitlines()[0]
    m = _FIRST_SENTENCE_RE.match(first_line)
    first_sentence = m.group(1) if m else first_line

    words = frozenset(re.findall(r'\b\w+\b', first_sentence.lower()))

    for keywords, tag in _HEADER_RULES:
        if words & keywords:
            return tag
    return None


def migrate_yedda(yedda_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply all Tier 1 relabeling rules to YEDDA-annotated text.

    Rules applied (in order, first match wins per block):
      1. Holotype blocks → Type-designation (≤2 lines) or
         Materials-examined (>2 lines).
      2. Header keywords in first sentence → Distribution, Diagnosis,
         or Biology (only for Misc-exposition and Description blocks).

    Args:
        yedda_text: YEDDA-annotated string.

    Returns:
        Tuple of (new_yedda_text, changes) where changes is a list of
        dicts with keys: block_index, old_tag, new_tag, reason.
    """
    if not yedda_text.strip():
        return "", []

    changes: List[Dict[str, Any]] = []
    result_parts: List[str] = []

    for i, match in enumerate(_YEDDA_BLOCK_RE.finditer(yedda_text)):
        text = match.group(1).strip()
        tag = match.group(2).strip()
        new_tag = tag
        reason: Optional[str] = None

        if tag == Tag.HOLOTYPE.value:
            candidate = relabel_holotype(text)
            new_tag = candidate.value
            reason = "holotype-split"
        elif tag in _KEYWORD_ELIGIBLE:
            candidate = relabel_by_header(text)
            if candidate is not None and candidate.value != tag:
                new_tag = candidate.value
                reason = "header-keyword"

        result_parts.append(f"[@{text}#{new_tag}*]")

        if new_tag != tag:
            changes.append({
                "block_index": i,
                "old_tag": tag,
                "new_tag": new_tag,
                "reason": reason,
            })

    return "\n\n".join(result_parts) + ("\n" if result_parts else ""), changes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _migrate_database(
    db: Any,
    dry_run: bool,
    verbosity: int,
) -> Dict[str, int]:
    """Apply migrate_yedda to every article.txt.ann in a CouchDB database.

    Args:
        db: CouchDB database object.
        dry_run: If True, report changes without writing.
        verbosity: Logging level.

    Returns:
        Summary counts: docs_processed, docs_changed, blocks_changed.
    """
    docs_processed = 0
    docs_changed = 0
    blocks_changed = 0

    for row in db.view("_all_docs", include_docs=False):
        if row.id.startswith("_design/"):
            continue
        docs_processed += 1

        att = db.get_attachment(row.id, "article.txt.ann")
        if att is None:
            continue

        yedda_text = att.read().decode("utf-8")
        new_text, changes = migrate_yedda(yedda_text)

        if not changes:
            continue

        docs_changed += 1
        blocks_changed += len(changes)

        if verbosity >= 1:
            print(f"  {row.id}: {len(changes)} block(s) relabeled")
            for c in changes:
                print(
                    f"    block {c['block_index']}: "
                    f"{c['old_tag']} → {c['new_tag']} "
                    f"({c['reason']})"
                )

        if not dry_run:
            db.put_attachment(
                db[row.id],
                new_text.encode("utf-8"),
                filename="article.txt.ann",
                content_type="text/plain",
            )

    return {
        "docs_processed": docs_processed,
        "docs_changed": docs_changed,
        "blocks_changed": blocks_changed,
    }


def main() -> None:
    """Entry point for the migrate_labels CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Tier 1 automatic label migration: upgrade 8-tag YEDDA "
            "annotations to the 12-tag scheme."
        )
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--database",
        type=str,
        help="CouchDB database containing article.txt.ann attachments.",
    )
    src_group.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (resolves annotations database automatically).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing updated annotations.",
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

    from env_config import get_env_config
    config = get_env_config()

    import couchdb as couchdb_lib
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )

    if args.experiment:
        db_name = (
            config.get("annotations_db_name") or config["ingest_db_name"]
        )
    else:
        db_name = args.database

    try:
        db = server[db_name]
    except Exception:
        print(f"Error: database '{db_name}' not found.", file=sys.stderr)
        sys.exit(1)

    if verbosity >= 1:
        mode = "DRY RUN — " if args.dry_run else ""
        print(f"{mode}Migrating labels in '{db_name}'", file=sys.stderr)

    summary = _migrate_database(db, dry_run=args.dry_run, verbosity=verbosity)

    if verbosity >= 1:
        print(
            f"\nDone: {summary['docs_processed']} docs processed, "
            f"{summary['docs_changed']} changed, "
            f"{summary['blocks_changed']} blocks relabeled.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
