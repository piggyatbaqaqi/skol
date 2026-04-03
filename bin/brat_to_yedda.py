#!/usr/bin/env python3
"""Convert brat standoff annotations back to YEDDA format.

Reads a brat ``article.txt`` + ``article.ann`` pair and reconstructs the
``[@text#Tag*]`` YEDDA blocks, sorted by character offset.

Non-entity annotation lines (relations ``R…``, attributes ``A…``,
notes ``#…``, etc.) are silently ignored — only ``T…`` entity lines
are converted.

Core functions are pure (no I/O) for testability.

Usage:
    python bin/brat_to_yedda.py TXT_FILE ANN_FILE [OUTPUT.ann] [--verbose]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    parser = argparse.ArgumentParser(
        description="Convert brat standoff annotations to YEDDA format."
    )
    parser.add_argument("txt_file", type=Path, help="brat plain-text file.")
    parser.add_argument("ann_file", type=Path, help="brat annotation (.ann) file.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Output YEDDA file (default: <txt_file>.ann beside the txt file)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    plaintext = args.txt_file.read_text(encoding="utf-8")
    ann = args.ann_file.read_text(encoding="utf-8")

    yedda = brat_to_yedda(plaintext, ann)

    if args.output is None:
        output_path = args.txt_file.with_suffix(
            args.txt_file.suffix + ".ann"
        )
    else:
        output_path = args.output

    output_path.write_text(yedda, encoding="utf-8")

    if args.verbose:
        block_count = yedda.count("[@")
        print(
            f"Wrote {block_count} YEDDA blocks to {output_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
