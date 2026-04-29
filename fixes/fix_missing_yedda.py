#!/usr/bin/env python3
"""Rebuild YEDDA annotation files using a new complete plaintext source.

For each document that has an ``article.txt.ann`` in ``--yedda-db`` and
an ``article.txt`` in ``--plaintext-db``, this script:

  1. Parses the old YEDDA into (text, label) blocks.
  2. Locates each block's text in the new plaintext using exact substring
     search, with fallbacks for blocks whose text begins with a PDF
     page-marker line or has minor whitespace differences.
  3. Sorts located blocks by their position in the new text.
  4. Fills gaps between located blocks:
       - Lines matching ``--- PDF Page N Label L ---`` → Page-header blocks.
       - Other non-empty text segments → To-review blocks.
  5. Appends any unlocatable blocks as To-review blocks at the end.
  6. Writes the rebuilt ``article.txt.ann`` to ``--output-db`` under the same
     ``_id``.

Usage::

    python fixes/fix_missing_yedda.py \\
        --plaintext-db skol_training \\
        --yedda-db skol_ann_reviewed \\
        --output-db skol_ann_fixed \\
        [--doc-id DOC_ID] [--dry-run] [-v]
"""

import argparse
import difflib
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)

_PAGE_MARKER_RE = re.compile(
    r"^---\s+PDF\s+Page\s+(\d+)\s+Label\s+(\S+)\s+---\s*$"
)

_TXT_ATTACHMENT = "article.txt"
_ANN_ATTACHMENT = "article.txt.ann"

# ---------------------------------------------------------------------------
# Core helpers (file-format logic; no CouchDB dependency)
# ---------------------------------------------------------------------------

Block = Tuple[str, str]  # (text, label)


def parse_yedda(text: str) -> List[Block]:
    """Return list of (block_text, label) from a YEDDA string."""
    return [(m.group(1), m.group(2)) for m in _YEDDA_BLOCK_RE.finditer(text)]


def _strip_leading_page_marker(text: str) -> Tuple[Optional[str], str]:
    """If text starts with a PDF page marker line return (marker, rest).

    Returns (None, text) if there is no leading marker.
    """
    first_nl = text.find("\n")
    first_line = text[:first_nl] if first_nl >= 0 else text
    rest = text[first_nl + 1:].strip() if first_nl >= 0 else ""
    if _PAGE_MARKER_RE.match(first_line.strip()):
        return first_line.strip(), rest
    return None, text


_FUZZY_THRESHOLD = 0.95  # minimum SequenceMatcher ratio to accept a match
_FUZZY_SLOP = 20         # extra chars on each side of the candidate window


def _nfkc_map(text: str) -> Tuple[str, List[int]]:
    """Return (nfkc_text, orig_offsets).

    ``orig_offsets[i]`` is the index in *text* of the original character
    whose NFKC expansion produced ``nfkc_text[i]``.  Useful for mapping a
    match position in normalised space back to the original string.
    """
    parts: List[str] = []
    orig: List[int] = []
    for i, ch in enumerate(text):
        n = unicodedata.normalize("NFKC", ch)
        parts.append(n)
        orig.extend([i] * len(n))
    return "".join(parts), orig


def find_block_in_text(
    block_text: str,
    haystack: str,
    search_from: int,
) -> Tuple[int, int, str]:
    """Locate block_text in haystack starting at search_from.

    Tries, in order:
      1. Exact substring match.
      2. Match after stripping a leading PDF page-marker line.
      3. Match with whitespace normalised (runs of whitespace → single space).
      4. NFKC-normalised exact match (handles typographic ligatures such as
         ﬁ → fi and fullwidth / halfwidth variants).
      5. difflib fuzzy match on original text (handles single-character OCR
         substitutions that are not resolved by NFKC).  Uses NFKC to locate
         a candidate region, then scores with SequenceMatcher against the
         original haystack so returned offsets require no remapping.

    Returns:
        (start, end, effective_text) on success, or (-1, -1, '') if not found.
    """
    # 1. Exact match.
    pos = haystack.find(block_text, search_from)
    if pos >= 0:
        return pos, pos + len(block_text), block_text

    # 2. Strip leading page marker and retry.
    marker, body = _strip_leading_page_marker(block_text)
    if marker is not None and body:
        pos = haystack.find(body, search_from)
        if pos >= 0:
            return pos, pos + len(body), body

    # 3. Normalised-whitespace regex match.
    norm = re.sub(r"\s+", " ", block_text).strip()
    escaped = re.escape(norm).replace(r"\ ", r"\s+")
    try:
        m = re.search(escaped, haystack[search_from:], re.DOTALL)
    except re.error:
        m = None
    if m is not None:
        start = search_from + m.start()
        end = search_from + m.end()
        return start, end, haystack[start:end]

    # 4. NFKC-normalised exact match.
    #    Build a position map for the search region so we can recover original
    #    offsets after matching in normalised space.
    norm_block = unicodedata.normalize("NFKC", block_text)
    norm_hay_region, hay_map = _nfkc_map(haystack[search_from:])
    npos = norm_hay_region.find(norm_block)
    if npos >= 0:
        orig_start = search_from + hay_map[npos]
        orig_end = search_from + hay_map[npos + len(norm_block) - 1] + 1
        return orig_start, orig_end, haystack[orig_start:orig_end]

    # 5. difflib fuzzy match.
    #    Try anchors at three positions (start, ⅓, ⅔) within the NFKC-
    #    normalised block to locate a candidate region even when the OCR error
    #    falls in the first part of the block.  Once a candidate region is
    #    found, score SequenceMatcher against the *original* haystack so
    #    returned offsets need no remapping.
    anchor_len = max(10, min(30, len(norm_block) // 3))
    anchor_offsets = [0]
    if len(norm_block) > anchor_len * 2:
        anchor_offsets += [len(norm_block) // 3, 2 * len(norm_block) // 3]

    orig_approx: Optional[int] = None
    for aoff in anchor_offsets:
        anchor = norm_block[aoff: aoff + anchor_len]
        npos_approx = norm_hay_region.find(anchor)
        if npos_approx >= 0:
            # Estimate where the start of the block falls in the original.
            norm_block_start = max(0, npos_approx - aoff)
            orig_approx = search_from + hay_map[norm_block_start]
            break

    if orig_approx is None:
        return -1, -1, ""

    # Slide a same-length window ± SLOP around the estimated position.
    # Comparing equal-length strings keeps SequenceMatcher.ratio() meaningful;
    # a wider window would dilute the score even with one-char differences.
    blen = len(block_text)
    best_ratio = 0.0
    best_start = -1
    for offset in range(-_FUZZY_SLOP, _FUZZY_SLOP + 1):
        cand_start = orig_approx + offset
        cand_end = cand_start + blen
        if cand_start < search_from or cand_end > len(haystack):
            continue
        cand = haystack[cand_start:cand_end]
        ratio = difflib.SequenceMatcher(
            None, block_text, cand, autojunk=False
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = cand_start

    if best_ratio < _FUZZY_THRESHOLD or best_start < 0:
        return -1, -1, ""
    best_end = best_start + blen
    return best_start, best_end, haystack[best_start:best_end]


def split_gap(gap_text: str) -> List[Block]:
    """Convert unannotated gap text into Page-header and To-review blocks.

    PDF page-marker lines become Page-header blocks; remaining non-empty
    text segments become To-review blocks.
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


def rebuild_yedda(
    old_yedda: str,
    new_text: str,
    *,
    verbose: bool = False,
) -> Tuple[str, int, int]:
    """Rebuild YEDDA by anchoring old blocks into new_text.

    Returns:
        (new_yedda_string, placed_count, unplaced_count)
    """
    old_blocks = parse_yedda(old_yedda)

    placed: List[Tuple[int, int, str, str]] = []  # (start, end, text, label)
    unplaced: List[Block] = []
    cursor = 0

    for i, (text, label) in enumerate(old_blocks):
        start, end, eff = find_block_in_text(text, new_text, cursor)
        if start >= 0:
            placed.append((start, end, eff, label))
            cursor = end
            if verbose:
                print(
                    f"    [{i}] {label!r}: [{start},{end})  "
                    f"{repr(eff[:40])}"
                )
        else:
            unplaced.append((text, label))
            print(
                f"    WARNING [{i}] {label!r}: not found — "
                f"{repr(text[:60])}",
                file=sys.stderr,
            )

    placed.sort(key=lambda t: t[0])

    out_blocks: List[Block] = []
    prev_end = 0

    for start, end, text, label in placed:
        if start > prev_end:
            gap_blocks = split_gap(new_text[prev_end:start])
            if verbose and gap_blocks:
                print(
                    f"    gap [{prev_end},{start}): "
                    f"{len(gap_blocks)} block(s) inserted"
                )
            out_blocks.extend(gap_blocks)
        out_blocks.append((text, label))
        prev_end = end

    if prev_end < len(new_text):
        trailing = split_gap(new_text[prev_end:])
        if verbose and trailing:
            print(
                f"    trailing [{prev_end},{len(new_text)}): "
                f"{len(trailing)} block(s) inserted"
            )
        out_blocks.extend(trailing)

    for text, label in unplaced:
        out_blocks.append((text, "To-review"))

    return blocks_to_yedda(out_blocks), len(placed), len(unplaced)


# ---------------------------------------------------------------------------
# CouchDB helpers
# ---------------------------------------------------------------------------

def _get_attachment_text(db, doc_id: str, filename: str) -> Optional[str]:
    """Return UTF-8 text of a CouchDB attachment, or None if absent."""
    try:
        data = db.get_attachment(doc_id, filename)
        if data is None:
            return None
        return data.read().decode("utf-8")  # type: ignore[return-value]
    except Exception:
        return None


def _put_attachment_text(db, doc_id: str, filename: str, text: str) -> None:
    """Write a UTF-8 text attachment, creating the document if necessary."""
    if doc_id not in db:
        db[doc_id] = {}
    doc = db[doc_id]
    db.put_attachment(
        doc,
        text.encode("utf-8"),
        filename=filename,
        content_type="text/plain; charset=utf-8",
    )


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

def process_document(
    doc_id: str,
    plaintext_db,
    yedda_db,
    output_db,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> Tuple[bool, int, int]:
    """Process one document.

    Returns:
        (success, placed_count, unplaced_count)
    """
    old_yedda = _get_attachment_text(yedda_db, doc_id, _ANN_ATTACHMENT)
    if old_yedda is None:
        print(f"  {doc_id}: no {_ANN_ATTACHMENT} in yedda-db — skipped",
              file=sys.stderr)
        return False, 0, 0

    new_text = _get_attachment_text(plaintext_db, doc_id, _TXT_ATTACHMENT)
    if new_text is None:
        print(f"  {doc_id}: no {_TXT_ATTACHMENT} in plaintext-db — skipped",
              file=sys.stderr)
        return False, 0, 0

    new_yedda, placed, unplaced = rebuild_yedda(
        old_yedda, new_text, verbose=verbose
    )

    if dry_run:
        print(
            f"  {doc_id}: placed={placed} unplaced={unplaced} "
            f"output={len(new_yedda)}B  (dry run)"
        )
    else:
        _put_attachment_text(output_db, doc_id, _ANN_ATTACHMENT, new_yedda)
        print(
            f"  {doc_id}: placed={placed} unplaced={unplaced} "
            f"→ {_ANN_ATTACHMENT} written to output-db"
        )

    return True, placed, unplaced


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild YEDDA annotation files by anchoring old blocks into a "
            "new complete plaintext."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--plaintext-db",
        required=True,
        metavar="DB",
        help="CouchDB database containing article.txt attachments.",
    )
    parser.add_argument(
        "--yedda-db",
        required=True,
        metavar="DB",
        help="CouchDB database containing old article.txt.ann attachments.",
    )
    parser.add_argument(
        "--output-db",
        required=True,
        metavar="DB",
        help=(
            "CouchDB database to receive rebuilt article.txt.ann attachments."
        ),
    )
    parser.add_argument(
        "--doc-id",
        metavar="ID",
        help="Process only this document ID (default: all docs in yedda-db).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing to output-db.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-block placement details.",
    )
    args = parser.parse_args()

    import couchdb  # type: ignore[import]

    server = couchdb.Server(config["couchdb_url"])
    if config.get("couchdb_username") and config.get("couchdb_password"):
        server.resource.credentials = (
            config["couchdb_username"],
            config["couchdb_password"],
        )

    plaintext_db = server[args.plaintext_db]
    yedda_db = server[args.yedda_db]
    if not args.dry_run:
        try:
            output_db = server[args.output_db]
        except couchdb.http.ResourceNotFound:
            output_db = server.create(args.output_db)
            if args.verbose:
                print(f"Created database: {args.output_db}")
    else:
        output_db = None

    # Collect doc IDs to process.
    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        doc_ids = [
            row.id
            for row in yedda_db.view("_all_docs")
            if not row.id.startswith("_")
        ]

    print(
        f"plaintext-db : {args.plaintext_db}\n"
        f"yedda-db     : {args.yedda_db}\n"
        f"output-db    : {args.output_db}\n"
        f"documents    : {len(doc_ids)}\n"
        f"mode         : {'dry run' if args.dry_run else 'apply'}\n"
    )

    total_placed = total_unplaced = success = errors = 0

    for doc_id in doc_ids:
        if args.verbose:
            print(f"{doc_id}:")
        ok, placed, unplaced = process_document(
            doc_id,
            plaintext_db=plaintext_db,
            yedda_db=yedda_db,
            output_db=output_db,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if ok:
            success += 1
            total_placed += placed
            total_unplaced += unplaced
        else:
            errors += 1

    print(
        f"\nDone: {success} processed, {errors} skipped\n"
        f"Total blocks placed={total_placed} unplaced={total_unplaced}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
