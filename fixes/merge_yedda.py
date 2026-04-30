#!/usr/bin/env python3
"""Three-way approximate merge of YEDDA annotations.

Analogous to ``git rebase`` / ``wiggle``: transfers human-reviewed YEDDA
block labels from an old OCR base onto a new OCR text, using the full-document
diff between old and new to establish character-offset correspondences.

Clean placements emit normal ``[@text#Label*]`` blocks.
Uncertain placements (block maps to a deleted region) emit inline
``<<<<<<< / ======= / >>>>>>>`` conflict markers that can be resolved by hand.

Usage::

    python fixes/merge_yedda.py \\
        --reviewed-db  skol_ann_reviewed \\
        --training-db  skol_training \\
        --output-db    skol_ann_merged \\
        [--doc-id DOC_ID] [--dry-run] [-v]
"""

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Block = Tuple[str, str]           # (text, label)
PosMap = Callable[[int], Tuple[int, bool]]  # orig_pos -> (new_pos, certain)

_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)
_PAGE_MARKER_RE = re.compile(
    r"^---\s+PDF\s+Page\s+(\d+)\s+Label\s+(\S+)\s+---\s*$"
)
_ANN_ATTACHMENT = "article.txt.ann"
_TXT_ATTACHMENT = "article.txt"

# Fuzzy-match threshold for locating reviewed blocks in orig text.
_FUZZY_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# YEDDA helpers
# ---------------------------------------------------------------------------


def strip_yedda(yedda_text: str) -> str:
    """Return plain text from a YEDDA string (blocks joined by ``\\n\\n``)."""
    blocks = [
        m.group(1).strip()
        for m in _YEDDA_BLOCK_RE.finditer(yedda_text)
    ]
    return "\n\n".join(blocks)


def _parse_yedda(text: str) -> List[Block]:
    return [
        (m.group(1).strip(), m.group(2))
        for m in _YEDDA_BLOCK_RE.finditer(text)
    ]


def _blocks_to_yedda(blocks: List[Block]) -> str:
    return "\n\n".join(f"[@{t}#{l}*]" for t, l in blocks) + "\n"


def _split_gap(gap_text: str) -> List[Block]:
    """Convert unannotated gap text to Page-header / To-review blocks."""
    blocks: List[Block] = []
    current: List[str] = []
    for line in gap_text.split("\n"):
        if _PAGE_MARKER_RE.match(line.rstrip()):
            body = "\n".join(current).strip()
            if body:
                blocks.append((body, "To-review"))
            current = []
            blocks.append((line.strip(), "Page-header"))
        else:
            current.append(line)
    remainder = "\n".join(current).strip()
    if remainder:
        blocks.append((remainder, "To-review"))
    return blocks


# ---------------------------------------------------------------------------
# Position mapping
# ---------------------------------------------------------------------------


def build_pos_map(
    matching_blocks: list,
) -> PosMap:
    """Build a position mapper from SequenceMatcher.get_matching_blocks().

    Returns a callable ``pos_map(orig_pos) -> (new_pos, certain)`` where
    ``certain`` is True when *orig_pos* falls inside a matching run (identical
    text) and False when it falls in a gap (deleted region).  Uncertain
    positions are interpolated to the end of the preceding matching run.
    """
    # Filter out the sentinel (0, 0, 0) entry that SequenceMatcher appends.
    runs = [(a, b, size) for a, b, size in matching_blocks if size > 0]
    runs.sort(key=lambda t: t[0])

    def pos_map(orig_pos: int) -> Tuple[int, bool]:
        # Binary search for the run that contains orig_pos.
        lo, hi = 0, len(runs)
        while lo < hi:
            mid = (lo + hi) // 2
            a, b, size = runs[mid]
            if orig_pos < a:
                hi = mid
            elif orig_pos < a + size:
                # Inside this run — certain mapping.
                return b + (orig_pos - a), True
            else:
                lo = mid + 1

        # orig_pos is in a gap.  Interpolate to the end of the preceding run.
        idx = lo - 1  # last run that starts before orig_pos
        if idx < 0:
            return 0, False
        a, b, size = runs[idx]
        return b + size, False  # end of the preceding run in new_text

    return pos_map


# ---------------------------------------------------------------------------
# Conflict formatting
# ---------------------------------------------------------------------------

_CONFLICT_ANNOTATION = "<<<<<<< annotation"
_CONFLICT_SEP = "======="
_CONFLICT_NEW = ">>>>>>> new_ocr"


def format_conflict(
    ann_block_text: str,
    ann_label: str,
    new_context: str,
) -> str:
    """Return a diff-u style conflict marker string.

    ``ann_block_text`` / ``ann_label`` are the annotation side; ``new_context``
    is the new OCR text at the interpolated position (may be empty when the
    block was deleted entirely from the new OCR).
    """
    yedda_block = f"[@{ann_block_text}#{ann_label}*]"
    return (
        f"{_CONFLICT_ANNOTATION}\n"
        f"{yedda_block}\n"
        f"{_CONFLICT_SEP}\n"
        f"{new_context}\n"
        f"{_CONFLICT_NEW}"
    )


# ---------------------------------------------------------------------------
# Block location helpers
# ---------------------------------------------------------------------------

def find_in_text(
    block_text: str,
    haystack: str,
) -> Tuple[int, int]:
    """Locate block_text anywhere in haystack.

    Returns (start, end) or (-1, -1).
    Tries: exact, page-marker strip, NFKC-normalised, difflib fuzzy
    (ratio ≥ _FUZZY_THRESHOLD).
    """
    import unicodedata

    # 1. Exact.
    pos = haystack.find(block_text)
    if pos >= 0:
        return pos, pos + len(block_text)

    # 2. Strip synthetic --- PDF Page N Label N --- prefix and retry.
    first_nl = block_text.find("\n")
    if first_nl > 0:
        first_line = block_text[:first_nl].strip()
        remainder = block_text[first_nl + 1:].strip()
        if _PAGE_MARKER_RE.match(first_line) and remainder:
            pos = haystack.find(remainder)
            if pos >= 0:
                return pos, pos + len(remainder)

    # 3. NFKC.
    def _nfkc_map(text: str) -> Tuple[str, List[int]]:
        parts: List[str] = []
        orig: List[int] = []
        for i, ch in enumerate(text):
            n = unicodedata.normalize("NFKC", ch)
            parts.append(n)
            orig.extend([i] * len(n))
        return "".join(parts), orig

    norm_block = unicodedata.normalize("NFKC", block_text)
    norm_hay, hay_map = _nfkc_map(haystack)
    npos = norm_hay.find(norm_block)
    if npos >= 0:
        orig_start = hay_map[npos]
        orig_end = hay_map[npos + len(norm_block) - 1] + 1
        return orig_start, orig_end

    # 4. Difflib fuzzy — try anchors at start, ⅓, ⅔ of block.
    slop = 20
    anchor_len = max(10, min(30, len(norm_block) // 3))
    anchor_offsets = [0]
    if len(norm_block) > anchor_len * 2:
        anchor_offsets += [len(norm_block) // 3, 2 * len(norm_block) // 3]

    for aoff in anchor_offsets:
        anchor = norm_block[aoff: aoff + anchor_len]
        npos_approx = norm_hay.find(anchor)
        if npos_approx < 0:
            continue
        norm_block_start = max(0, npos_approx - aoff)
        orig_approx = hay_map[norm_block_start]
        blen = len(block_text)
        best_ratio = 0.0
        best_start = -1
        for offset in range(-slop, slop + 1):
            cs = orig_approx + offset
            ce = cs + blen
            if cs < 0 or ce > len(haystack):
                continue
            ratio = difflib.SequenceMatcher(
                None, block_text, haystack[cs:ce], autojunk=False
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = cs
        if best_ratio >= _FUZZY_THRESHOLD and best_start >= 0:
            return best_start, best_start + blen

    return -1, -1


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------


def three_way_merge_yedda(
    orig_ann: str,
    reviewed_ann: str,
    new_text: str,
    *,
    verbose: bool = False,
) -> str:
    """Merge reviewed YEDDA labels onto new OCR text using three-way alignment.

    Args:
        orig_ann:     YEDDA annotation from the original OCR (source_file).
        reviewed_ann: Human-reviewed YEDDA (may have relabeled blocks).
        new_text:     New OCR plain text (article.txt from skol_training).
        verbose:      Print per-block progress to stderr.

    Returns:
        Rebuilt YEDDA with inline conflict markers for uncertain placements.
    """
    orig_text = strip_yedda(orig_ann)
    reviewed_blocks = _parse_yedda(reviewed_ann)

    # --- Step 1: locate each reviewed block in orig_text ---
    # (orig_start, orig_end, label, block_text)
    placed_in_orig: List[Tuple[int, int, str, str]] = []
    orphans: List[Block] = []

    for block_text, label in reviewed_blocks:
        start, end = find_in_text(block_text, orig_text)
        if start >= 0:
            placed_in_orig.append((start, end, label, block_text))
            if verbose:
                print(
                    f"  orig [{start},{end}) {label!r}: "
                    f"{repr(block_text[:40])}",
                    file=sys.stderr,
                )
        else:
            orphans.append((block_text, label))
            print(
                f"  WARNING orphan {label!r}: {repr(block_text[:60])}",
                file=sys.stderr,
            )

    placed_in_orig.sort(key=lambda t: t[0])

    # --- Step 2: build orig → new position map ---
    matcher = difflib.SequenceMatcher(
        None, orig_text, new_text, autojunk=False
    )
    pos_map = build_pos_map(matcher.get_matching_blocks())

    # --- Step 3: map each block's position to new_text ---
    #   Collect as (new_start, new_end, label, block_text, certain)
    Placed = Tuple[int, int, str, str, bool]
    mapped: List[Placed] = []

    for orig_start, orig_end, label, block_text in placed_in_orig:
        new_start, start_certain = pos_map(orig_start)
        # Map the last *inclusive* character of the block, then add 1 to get
        # the exclusive end.  Mapping orig_end (exclusive) directly fails when
        # orig_end falls exactly at the boundary of a matching run (it lies
        # outside the run and is reported as uncertain).
        if orig_end > orig_start:
            new_end_char, end_certain = pos_map(orig_end - 1)
            new_end = (new_end_char + 1) if end_certain else new_end_char
        else:
            new_end, end_certain = new_start, start_certain
        certain = start_certain and end_certain and new_start < new_end
        mapped.append((new_start, new_end, label, block_text, certain))

    mapped.sort(key=lambda t: t[0])

    # --- Step 4: assemble output ---
    # Build a flat list of output items (each is either a (text, label) block
    # or a raw conflict-marker string).
    output_items: List[str] = []
    prev_end = 0

    for new_start, new_end, label, orig_block_text, certain in mapped:
        # Fill gap before this block.
        gap_start = prev_end
        gap_end = new_start if certain else new_start
        if gap_start < gap_end:
            for gap_text, gap_label in _split_gap(new_text[gap_start:gap_end]):
                output_items.append(f"[@{gap_text}#{gap_label}*]")

        if certain:
            output_items.append(f"[@{new_text[new_start:new_end]}#{label}*]")
            prev_end = new_end
        else:
            # Uncertain placement — emit conflict marker.
            new_context = (
                new_text[new_start:new_end]
                if new_start < new_end
                else new_text[new_start: new_start + len(orig_block_text)]
            )
            output_items.append(
                format_conflict(orig_block_text, label, new_context)
            )
            prev_end = max(prev_end, new_end)

    # Trailing gap.
    if prev_end < len(new_text):
        for gap_text, gap_label in _split_gap(new_text[prev_end:]):
            output_items.append(f"[@{gap_text}#{gap_label}*]")

    # Orphan blocks at end as conflict markers (no new_context known).
    for block_text, label in orphans:
        output_items.append(format_conflict(block_text, label, ""))

    return "\n\n".join(output_items) + "\n"


# ---------------------------------------------------------------------------
# CouchDB helpers
# ---------------------------------------------------------------------------


def _get_attachment(db, doc_id: str, filename: str) -> Optional[str]:
    try:
        data = db.get_attachment(doc_id, filename)
        if data is None:
            return None
        return data.read().decode("utf-8")
    except Exception:
        return None


def _put_attachment(db, doc_id: str, filename: str, text: str) -> None:
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
    reviewed_db,
    training_db,
    output_db,
    data_root: Path,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """Process one document.  Returns True on success."""
    # Reviewed YEDDA.
    reviewed_ann = _get_attachment(reviewed_db, doc_id, _ANN_ATTACHMENT)
    if reviewed_ann is None:
        print(
            f"  {doc_id}: no {_ANN_ATTACHMENT} in reviewed-db — skipped",
            file=sys.stderr,
        )
        return False

    # New OCR plain text.
    new_text = _get_attachment(training_db, doc_id, _TXT_ATTACHMENT)
    if new_text is None:
        print(
            f"  {doc_id}: no {_TXT_ATTACHMENT} in training-db — skipped",
            file=sys.stderr,
        )
        return False

    # Original annotated file (positional anchor).
    training_doc = training_db.get(doc_id)
    if training_doc is None:
        print(
            f"  {doc_id}: not found in training-db — skipped",
            file=sys.stderr,
        )
        return False
    source_file = training_doc.get("source_file")
    if not source_file:
        print(
            f"  {doc_id}: no source_file field in training-db doc — skipped",
            file=sys.stderr,
        )
        return False

    orig_ann_path = data_root / source_file
    if not orig_ann_path.exists():
        print(
            f"  {doc_id}: source_file {orig_ann_path} not found — skipped",
            file=sys.stderr,
        )
        return False

    orig_ann = orig_ann_path.read_text(encoding="utf-8")

    if verbose:
        print(f"{doc_id}:", file=sys.stderr)

    merged = three_way_merge_yedda(
        orig_ann, reviewed_ann, new_text, verbose=verbose
    )
    conflicts = merged.count(_CONFLICT_ANNOTATION)

    if dry_run:
        print(
            f"  {doc_id}: conflicts={conflicts} "
            f"output={len(merged)}B  (dry run)",
            flush=True,
        )
    else:
        _put_attachment(output_db, doc_id, _ANN_ATTACHMENT, merged)
        print(
            f"  {doc_id}: conflicts={conflicts} → {_ANN_ATTACHMENT} written",
            flush=True,
        )

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    config = get_env_config()

    parser = argparse.ArgumentParser(
        description=(
            "Three-way approximate merge of YEDDA annotations "
            "onto new OCR text."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reviewed-db",
        required=True,
        metavar="DB",
        help="CouchDB database containing human-reviewed article.txt.ann.",
    )
    parser.add_argument(
        "--training-db",
        required=True,
        metavar="DB",
        help=(
            "CouchDB database containing new article.txt and "
            "source_file metadata."
        ),
    )
    parser.add_argument(
        "--output-db",
        required=True,
        metavar="DB",
        help="CouchDB database to receive merged article.txt.ann.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="DIR",
        help=(
            "Root directory for source_file paths "
            "(default: project root relative to this script)."
        ),
    )
    parser.add_argument(
        "--doc-id",
        metavar="ID",
        help="Process only this document ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing to output-db.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    data_root = (
        Path(args.data_root)
        if args.data_root
        else Path(__file__).resolve().parent.parent
    )

    import couchdb  # type: ignore[import]

    server = couchdb.Server(config["couchdb_url"])
    if config.get("couchdb_username") and config.get("couchdb_password"):
        server.resource.credentials = (
            config["couchdb_username"],
            config["couchdb_password"],
        )

    reviewed_db = server[args.reviewed_db]
    training_db = server[args.training_db]
    if not args.dry_run:
        try:
            output_db = server[args.output_db]
        except couchdb.http.ResourceNotFound:
            output_db = server.create(args.output_db)
            if args.verbose:
                print(f"Created database: {args.output_db}", file=sys.stderr)
    else:
        output_db = None

    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        doc_ids = [
            row.id
            for row in reviewed_db.view("_all_docs")
            if not row.id.startswith("_")
        ]

    print(
        f"reviewed-db  : {args.reviewed_db}\n"
        f"training-db  : {args.training_db}\n"
        f"output-db    : {args.output_db}\n"
        f"data-root    : {data_root}\n"
        f"documents    : {len(doc_ids)}\n"
        f"mode         : {'dry run' if args.dry_run else 'apply'}\n",
        flush=True,
    )

    success = errors = 0
    for doc_id in doc_ids:
        ok = process_document(
            doc_id,
            reviewed_db=reviewed_db,
            training_db=training_db,
            output_db=output_db,
            data_root=data_root,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if ok:
            success += 1
        else:
            errors += 1

    print(f"\nDone: {success} processed, {errors} skipped")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
