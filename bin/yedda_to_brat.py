#!/usr/bin/env python3
"""Convert YEDDA-annotated text to brat standoff format.

brat standoff uses two files:
  article.txt  — plain text, one token/sentence per line or free-form
  article.ann  — annotation lines, one entity per line:
                 T{n}\\t{Label} {start} {end}\\t{text}

Character offsets are zero-based and span [start, end) (end is exclusive),
matching brat's convention.

Core functions are pure (no I/O) for testability.

Usage:
    python bin/yedda_to_brat.py INPUT.ann OUTPUT_DIR [--verbose]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# YEDDA block parser
# ---------------------------------------------------------------------------

_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)


def yedda_to_brat(yedda_text: str) -> Tuple[str, str]:
    """Convert YEDDA-annotated text to brat plaintext + annotation strings.

    Args:
        yedda_text: YEDDA-annotated string (``[@text#Tag*]`` blocks).

    Returns:
        Tuple of (plaintext, ann) where:
          - plaintext is block texts joined by ``\\n\\n``
          - ann is a brat standoff annotation string (one ``T{n}\\t…`` line
            per block, ``\\n``-joined, empty string if no blocks)
    """
    blocks: List[Tuple[str, str]] = []
    for match in _YEDDA_BLOCK_RE.finditer(yedda_text):
        text = match.group(1)
        tag = match.group(2).strip()
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
        ann_lines.append(f"T{i}\t{tag} {start} {end}\t{text}")
        offset = end + 2  # skip the "\n\n" separator

    return plaintext, "\n".join(ann_lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: convert a YEDDA .ann file to brat standoff files."""
    parser = argparse.ArgumentParser(
        description="Convert YEDDA-annotated text to brat standoff format."
    )
    parser.add_argument("input", type=Path, help="Input YEDDA .ann file.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory; writes <stem>.txt and <stem>.ann.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    yedda_text = args.input.read_text(encoding="utf-8")
    plaintext, ann = yedda_to_brat(yedda_text)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem  # e.g. "article.txt" from "article.txt.ann"
    txt_path = args.output_dir / stem
    ann_path = args.output_dir / (stem + ".ann")

    txt_path.write_text(plaintext, encoding="utf-8")
    ann_path.write_text(ann, encoding="utf-8")

    if args.verbose:
        block_count = ann.count("\n") + 1 if ann else 0
        print(
            f"Wrote {block_count} entities to {ann_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
