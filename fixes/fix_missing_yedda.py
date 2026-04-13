#!/usr/bin/env python3
"""Rebuild a YEDDA annotation file using a new complete plaintext source.

When a YEDDA file (e.g. from skol_ann_reviewed) is missing text blocks that
appear in a more-complete corresponding plaintext (e.g. skol_training/article.txt),
this script:

  1. Parses the old YEDDA into (text, label) blocks.
  2. Locates each block's text in the new plaintext using exact substring search.
     If the block starts with a ``--- PDF Page N Label L ---`` marker (which was
     embedded in the old YEDDA but is now a standalone line), the marker is
     stripped before searching and the search text is the block body only.
  3. Sorts located blocks by their position in the new text.
  4. Fills gaps between located blocks:
       - Lines matching ``--- PDF Page N Label L ---`` → Page-header blocks.
       - Other non-empty text segments → To-review blocks.
  5. Appends any unlocatable blocks as To-review blocks at the end.

Usage::

    python fixes/fix_missing_yedda.py OLD.ann NEW.txt OUTPUT.ann
    python fixes/fix_missing_yedda.py --dry-run OLD.ann NEW.txt

Options:
    --dry-run   Print summary without writing OUTPUT.ann.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Regex patterns (shared with fix_staging_yedda.py / migrate_labels.py)
# ---------------------------------------------------------------------------

_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)

_PAGE_MARKER_RE = re.compile(
    r"^---\s+PDF\s+Page\s+(\d+)\s+Label\s+(\S+)\s+---\s*$"
)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

Block = Tuple[str, str]  # (text, label)


def parse_yedda(text: str) -> List[Block]:
    """Return list of (block_text, label) from a YEDDA string."""
    return [(m.group(1), m.group(2)) for m in _YEDDA_BLOCK_RE.finditer(text)]


def _strip_leading_page_marker(text: str) -> Tuple[Optional[str], str]:
    """If text starts with a PDF page marker line, return (marker, rest).

    Returns (None, text) if no leading marker.
    """
    first_nl = text.find("\n")
    first_line = text[:first_nl] if first_nl >= 0 else text
    rest = text[first_nl + 1:].strip() if first_nl >= 0 else ""
    if _PAGE_MARKER_RE.match(first_line.strip()):
        return first_line.strip(), rest
    return None, text


def find_block_in_text(
    block_text: str,
    haystack: str,
    search_from: int,
) -> Tuple[int, int, str]:
    """Locate block_text in haystack starting at search_from.

    Handles blocks whose text begins with a PDF page marker (strips the marker
    before searching, since in the new text page markers are standalone lines).

    Returns:
        (start, end, effective_text) where effective_text is the portion of
        block_text that was actually found (may differ if a leading page marker
        was stripped).  Returns (-1, -1, '') if not found.
    """
    # First try exact match.
    pos = haystack.find(block_text, search_from)
    if pos >= 0:
        return pos, pos + len(block_text), block_text

    # Try stripping a leading PDF page marker.
    marker, body = _strip_leading_page_marker(block_text)
    if marker is not None and body:
        pos = haystack.find(body, search_from)
        if pos >= 0:
            return pos, pos + len(body), body

    # Try with normalised internal whitespace (collapse runs of whitespace).
    norm_needle = re.sub(r"\s+", " ", block_text).strip()
    # Build a regex that matches the normalised version.
    escaped = re.escape(norm_needle).replace(r"\ ", r"\s+")
    try:
        m = re.search(escaped, haystack[search_from:], re.DOTALL)
    except re.error:
        m = None
    if m is not None:
        start = search_from + m.start()
        end = search_from + m.end()
        return start, end, haystack[start:end]

    return -1, -1, ""


def split_gap(gap_text: str) -> List[Block]:
    """Convert a gap (unannotated text) into Page-header and To-review blocks.

    Lines matching the PDF page marker pattern become Page-header blocks.
    Remaining non-empty text segments become To-review blocks.
    """
    blocks: List[Block] = []
    current_lines: List[str] = []

    for line in gap_text.split("\n"):
        if _PAGE_MARKER_RE.match(line.rstrip()):
            body = "\n".join(current_lines).strip()
            if body:
                blocks.append((body, "To-review"))
            current_lines = []
            blocks.append((line.strip(), "Page-header"))
        else:
            current_lines.append(line)

    remainder = "\n".join(current_lines).strip()
    if remainder:
        blocks.append((remainder, "To-review"))

    return blocks


def blocks_to_yedda(blocks: List[Block]) -> str:
    """Serialise a list of (text, label) blocks to YEDDA format."""
    return "\n\n".join(f"[@{text}#{label}*]" for text, label in blocks) + "\n"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def rebuild_yedda(
    old_yedda: str,
    new_text: str,
    *,
    verbose: bool = False,
) -> Tuple[str, int, int]:
    """Rebuild YEDDA from old_yedda blocks anchored into new_text.

    Returns:
        (new_yedda_string, placed_count, unplaced_count)
    """
    old_blocks = parse_yedda(old_yedda)

    # Locate each old block in new_text, left-to-right.
    placed: List[Tuple[int, int, str, str]] = []  # (start, end, text, label)
    unplaced: List[Block] = []
    cursor = 0

    for i, (text, label) in enumerate(old_blocks):
        start, end, eff_text = find_block_in_text(text, new_text, cursor)
        if start >= 0:
            placed.append((start, end, eff_text, label))
            cursor = end
            if verbose:
                print(
                    f"  [{i}] {label!r}: found at [{start},{end})"
                    f"  {repr(eff_text[:40])}"
                )
        else:
            unplaced.append((text, label))
            print(
                f"  WARNING [{i}] {label!r}: not found — "
                f"{repr(text[:60])}",
                file=sys.stderr,
            )

    # Sort placed blocks by position (should already be ordered, but be safe).
    placed.sort(key=lambda t: t[0])

    # Assemble output blocks.
    out_blocks: List[Block] = []
    prev_end = 0

    for start, end, text, label in placed:
        if start > prev_end:
            gap = new_text[prev_end:start]
            gap_blocks = split_gap(gap)
            if verbose and gap_blocks:
                print(
                    f"  gap [{prev_end},{start}): "
                    f"{len(gap_blocks)} block(s) inserted"
                )
            out_blocks.extend(gap_blocks)
        out_blocks.append((text, label))
        prev_end = end

    # Trailing text after the last placed block.
    if prev_end < len(new_text):
        trailing = new_text[prev_end:]
        trailing_blocks = split_gap(trailing)
        if verbose and trailing_blocks:
            print(
                f"  trailing [{prev_end},{len(new_text)}): "
                f"{len(trailing_blocks)} block(s) inserted"
            )
        out_blocks.extend(trailing_blocks)

    # Append unplaced blocks at the end as To-review.
    for text, label in unplaced:
        out_blocks.append((text, "To-review"))

    return blocks_to_yedda(out_blocks), len(placed), len(unplaced)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a YEDDA annotation file from a new complete plaintext."
    )
    parser.add_argument(
        "old_ann",
        metavar="OLD.ann",
        help="Old YEDDA annotation file (with missing blocks).",
    )
    parser.add_argument(
        "new_txt",
        metavar="NEW.txt",
        help="New complete plaintext file (with PDF page markers).",
    )
    parser.add_argument(
        "output",
        metavar="OUTPUT.ann",
        nargs="?",
        help="Output YEDDA file. Required unless --dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing output.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-block placement details.",
    )
    args = parser.parse_args()

    if not args.dry_run and args.output is None:
        parser.error("OUTPUT.ann is required unless --dry-run is specified.")

    old_ann_path = Path(args.old_ann)
    new_txt_path = Path(args.new_txt)

    old_yedda = old_ann_path.read_text(encoding="utf-8")
    new_text = new_txt_path.read_text(encoding="utf-8")

    print(f"Old YEDDA: {old_ann_path.name}  ({len(old_yedda)} bytes)")
    print(f"New text:  {new_txt_path.name}  ({len(new_text)} bytes)")

    new_yedda, placed, unplaced = rebuild_yedda(
        old_yedda, new_text, verbose=args.verbose
    )

    total = placed + unplaced
    print(
        f"\nBlocks placed: {placed}/{total}  "
        f"unplaced: {unplaced}/{total}"
    )
    print(f"Output size: {len(new_yedda)} bytes")

    if args.dry_run:
        print("(dry run — no output written)")
        return

    out_path = Path(args.output)
    out_path.write_text(new_yedda, encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
