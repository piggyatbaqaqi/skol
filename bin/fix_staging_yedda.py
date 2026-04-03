#!/usr/bin/env python3
"""Fix malformed YEDDA blocks in skol_staging and recover PDF page markers.

Two fixes applied to each document:

1. **Malformed blocks**: A block whose text contains an embedded ``[@`` is
   missing its ``*]`` closing delimiter.  The regex captured text from the
   outer ``[@`` all the way to the next ``#Tag*]`` it found, swallowing the
   embedded block.  Fix: split at the embedded ``[@``, produce two blocks.
   The outer block's tag is inferred by heuristic; the recovered inner block
   keeps the tag that was captured (i.e. the tag of the swallowed block).

2. **PDF page markers**: ``skol_dev`` stores ``article.txt`` with
   ``--- PDF Page N Label L ---`` separator lines.  These carry page-number
   metadata that enables linking back to the source PDF.  For each document
   in ``skol_staging`` that has a corresponding ``skol_training`` entry with
   a ``skol_dev_id``, the first content line after each ``--- PDF Page …``
   marker is compared against block texts in the (already fixed) YEDDA.
   Matching blocks receive the marker as a prefix line and are relabeled to
   ``Page-header``.

Usage::

    python bin/fix_staging_yedda.py [--staging-db NAME] [--dry-run] [-v]
    python bin/fix_staging_yedda.py --doc-id ID [--staging-db NAME] \
        [--dry-run] [-v]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# YEDDA block regex (same as yedda_to_brat.py and migrate_labels.py)
# ---------------------------------------------------------------------------

_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)

# Page marker line format used by PDFSectionExtractor / skol_dev article.txt
_PAGE_MARKER_RE = re.compile(
    r"^---\s+PDF\s+Page\s+(\d+)\s+Label\s+(\S+)\s+---\s*$"
)

_ANN_ATTACHMENT = "article.txt.ann"
_ARTICLE_TXT = "article.txt"

# ---------------------------------------------------------------------------
# Heuristic tag inference for truncated malformed outer blocks
# ---------------------------------------------------------------------------

_OUTER_TAG_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^materials?\s+examined\b", re.I), "Materials-examined"),
    (re.compile(r"^material\s+examined\b", re.I), "Materials-examined"),
    (re.compile(r"^key\s+to\b", re.I), "Key"),
    (re.compile(r"^notes?\b", re.I), "Notes"),
    (re.compile(r"^distribution\b", re.I), "Distribution"),
    (re.compile(r"^diagnosis\b", re.I), "Diagnosis"),
    (re.compile(r"^biology\b|^ecology\b|^host\b", re.I), "Biology"),
    (re.compile(r"^etymology\b", re.I), "Etymology"),
    (re.compile(r"^description\b|^morphology\b", re.I), "Description"),
    (re.compile(r"^nomenclature\b|^taxonomy\b", re.I), "Nomenclature"),
    (re.compile(r"^type\s+designation\b|^holotype\b", re.I),
     "Type-designation"),
]


def _infer_outer_tag(text: str, fallback: str) -> str:
    """Guess the correct YEDDA tag for a truncated malformed outer block.

    Checks the first line of *text* against known section-header patterns.
    Returns *fallback* if no pattern matches.

    Args:
        text: Truncated block text (everything before the embedded ``[@``).
        fallback: Tag to use when no heuristic matches.

    Returns:
        YEDDA tag string.
    """
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    for pattern, tag in _OUTER_TAG_RULES:
        if pattern.match(first_line):
            return tag
    return fallback


# ---------------------------------------------------------------------------
# Fix 1: malformed blocks
# ---------------------------------------------------------------------------

def fix_malformed_blocks(
    yedda: str,
) -> Tuple[str, int]:
    """Fix YEDDA blocks whose text contains an embedded ``[@``.

    When a block is missing its ``*]`` closing delimiter before a nested
    ``[@``, the regex captures text all the way to the next ``#Tag*]``.
    This produces a block whose ``text`` field includes the literal ``[@``
    from the nested block.

    Fix strategy:

    * Split the captured text at the first ``[@``.
    * The prefix becomes the outer block, with a tag inferred by heuristic.
    * The suffix (stripping the recovered ``[@`` prefix) becomes the inner
      block, keeping the tag that was captured (since the regex found that
      tag as the closer for the embedded content).

    Empty outer or inner blocks are discarded.

    Args:
        yedda: Raw YEDDA-annotated text.

    Returns:
        Tuple of (fixed_yedda, n_fixed_blocks).
    """
    out_parts: List[str] = []
    n_fixed = 0

    for match in _YEDDA_BLOCK_RE.finditer(yedda):
        text = match.group(1)
        tag = match.group(2)

        if "[@" not in text:
            # Normal block — emit as-is (preserving whitespace trimming done
            # by the regex's \s* on both sides).
            if text.strip():
                out_parts.append(f"[@{text}#{tag}*]")
            continue

        # Malformed: split at the first embedded [@
        idx = text.index("[@")
        outer_text = text[:idx].rstrip()
        inner_text = text[idx + 2:].lstrip()  # strip the recovered [@

        outer_tag = _infer_outer_tag(outer_text, tag)

        if outer_text.strip():
            out_parts.append(f"[@{outer_text}#{outer_tag}*]")
        if inner_text.strip():
            out_parts.append(f"[@{inner_text}#{tag}*]")
        n_fixed += 1

    return "\n\n".join(out_parts) + ("\n" if out_parts else ""), n_fixed


# ---------------------------------------------------------------------------
# Fix 2: PDF page markers
# ---------------------------------------------------------------------------

def parse_page_markers(article_txt: str) -> List[Tuple[int, str, str]]:
    """Parse PDF page boundary markers from a ``skol_dev`` ``article.txt``.

    ``PDFSectionExtractor`` inserts lines of the form::

        --- PDF Page N Label L ---

    The first non-empty content line after each marker is treated as the
    page header text (typically a running head from the journal).

    Args:
        article_txt: Full text of a ``skol_dev`` ``article.txt`` attachment.

    Returns:
        List of ``(page_num, page_label, header_text)`` tuples, one per page
        boundary found.  Pages whose first content line is empty are omitted.
    """
    pages: List[Tuple[int, str, str]] = []
    lines = article_txt.splitlines()
    i = 0
    while i < len(lines):
        m = _PAGE_MARKER_RE.match(lines[i])
        if m:
            page_num = int(m.group(1))
            page_label = m.group(2)
            # Find first non-empty content line after the marker.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                pages.append((page_num, page_label, lines[j].strip()))
        i += 1
    return pages


def _normalise(text: str) -> str:
    """Collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", text.strip())


def add_page_markers(
    yedda: str,
    pages: List[Tuple[int, str, str]],
) -> Tuple[str, int]:
    """Prefix matching blocks with ``--- PDF Page N Label L ---`` lines.

    For each ``(page_num, page_label, header_text)`` in *pages*:

    * Find the first YEDDA block whose normalised text matches
      *header_text* (exact match after whitespace normalisation), or whose
      normalised text *starts with* the normalised *header_text*.
    * Prefix the block text with ``--- PDF Page N Label L ---\\n``.
    * Relabel the block to ``Page-header``.

    Blocks that already start with a ``--- PDF Page`` marker are skipped.

    Args:
        yedda: YEDDA-annotated text (should be already fixed for malformed
            blocks before calling this function).
        pages: List from :func:`parse_page_markers`.

    Returns:
        Tuple of (updated_yedda, n_markers_added).
    """
    if not pages:
        return yedda, 0

    # Build mutable list of (text, tag) from all parsed blocks.
    blocks: List[List[str]] = []  # [text, tag]
    for match in _YEDDA_BLOCK_RE.finditer(yedda):
        text = match.group(1)
        tag = match.group(2)
        if text.strip():
            blocks.append([text, tag])

    n_added = 0
    for page_num, page_label, header_text in pages:
        norm_header = _normalise(header_text)
        if not norm_header:
            continue
        marker_line = f"--- PDF Page {page_num} Label {page_label} ---"

        for block in blocks:
            text, tag = block
            # Skip if already marked.
            if text.startswith("--- PDF Page"):
                continue
            norm_text = _normalise(text)
            # Match: exact equality or the block text starts with the header.
            if norm_text == norm_header or norm_text.startswith(norm_header):
                block[0] = marker_line + "\n" + text.lstrip()
                block[1] = "Page-header"
                n_added += 1
                break  # first match wins; move on to next page

    out_parts = [f"[@{text}#{tag}*]" for text, tag in blocks]
    return "\n\n".join(out_parts) + ("\n" if out_parts else ""), n_added


# ---------------------------------------------------------------------------
# Combined fix for one document
# ---------------------------------------------------------------------------

def fix_yedda(
    yedda: str,
    pages: Optional[List[Tuple[int, str, str]]] = None,
) -> Tuple[str, Dict[str, int]]:
    """Apply all fixes to a YEDDA string.

    Applies :func:`fix_malformed_blocks` repeatedly until no malformed blocks
    remain (handles multi-level nesting), then :func:`add_page_markers` (if
    *pages* is provided).

    Args:
        yedda: Raw YEDDA-annotated text.
        pages: Page boundary list from :func:`parse_page_markers`, or None
            to skip the page-marker step.

    Returns:
        Tuple of (fixed_yedda, stats) where *stats* contains:
        ``n_malformed``, ``n_page_markers``.
    """
    fixed = yedda
    n_malformed = 0
    for _ in range(20):  # safety limit; nested depth never approaches 20
        step, n = fix_malformed_blocks(fixed)
        n_malformed += n
        if n == 0:
            break
        fixed = step

    n_page_markers = 0
    if pages:
        fixed, n_page_markers = add_page_markers(fixed, pages)
    return fixed, {
        "n_malformed": n_malformed,
        "n_page_markers": n_page_markers,
    }


# ---------------------------------------------------------------------------
# CouchDB batch processing
# ---------------------------------------------------------------------------

def _get_pages_for_doc(
    doc_id: str,
    training_db: Any,
    dev_db: Any,
) -> Optional[List[Tuple[int, str, str]]]:
    """Look up skol_dev article.txt page markers for a skol_staging doc.

    Lookup chain:
      skol_training[doc_id].skol_dev_id → skol_dev[skol_dev_id].article.txt

    Args:
        doc_id: Document ID (same in skol_staging and skol_training).
        training_db: CouchDB database object for skol_training.
        dev_db: CouchDB database object for skol_dev.

    Returns:
        List of page marker tuples, or None if the lookup fails.
    """
    try:
        training_doc = training_db[doc_id]
    except Exception:
        return None

    skol_dev_id = training_doc.get("skol_dev_id")
    if not skol_dev_id:
        return None

    att = dev_db.get_attachment(skol_dev_id, _ARTICLE_TXT)
    if att is None:
        return None

    article_txt = att.read().decode("utf-8")
    return parse_page_markers(article_txt)


def fix_staging_database(
    staging_db: Any,
    training_db: Any,
    dev_db: Any,
    dry_run: bool = False,
    verbosity: int = 1,
    doc_id_filter: Optional[str] = None,
) -> Dict[str, int]:
    """Apply all fixes to every document in skol_staging.

    Args:
        staging_db: CouchDB database object for skol_staging.
        training_db: CouchDB database object for skol_training.
        dev_db: CouchDB database object for skol_dev.
        dry_run: If True, report changes without writing.
        verbosity: 0 = quiet, 1 = one line per changed doc, 2 = full detail.
        doc_id_filter: If set, only process this document ID.

    Returns:
        Summary counts: docs_processed, docs_changed, n_malformed,
        n_page_markers.
    """
    totals: Dict[str, int] = {
        "docs_processed": 0,
        "docs_changed": 0,
        "n_malformed": 0,
        "n_page_markers": 0,
    }

    for row in staging_db.view("_all_docs", include_docs=False):
        doc_id = row.id
        if doc_id.startswith("_design/"):
            continue
        if doc_id_filter and doc_id != doc_id_filter:
            continue

        totals["docs_processed"] += 1

        ann_att = staging_db.get_attachment(doc_id, _ANN_ATTACHMENT)
        if ann_att is None:
            continue

        yedda = ann_att.read().decode("utf-8")

        # Collect page markers from skol_dev (may be None if no skol_dev_id).
        pages = _get_pages_for_doc(doc_id, training_db, dev_db)

        new_yedda, stats = fix_yedda(yedda, pages)

        if stats["n_malformed"] == 0 and stats["n_page_markers"] == 0:
            continue  # nothing changed

        totals["docs_changed"] += 1
        totals["n_malformed"] += stats["n_malformed"]
        totals["n_page_markers"] += stats["n_page_markers"]

        if verbosity >= 1:
            print(
                f"  {doc_id}: "
                f"{stats['n_malformed']} malformed fixed, "
                f"{stats['n_page_markers']} page markers added",
                file=sys.stderr,
            )
        if verbosity >= 2:
            print(f"    pages available: {bool(pages)}", file=sys.stderr)

        if not dry_run:
            staging_db.put_attachment(
                staging_db[doc_id],
                new_yedda.encode("utf-8"),
                filename=_ANN_ATTACHMENT,
                content_type="text/plain",
            )

    return totals


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: fix malformed YEDDA blocks and add page markers."""
    parser = argparse.ArgumentParser(
        description=(
            "Fix malformed YEDDA blocks and recover PDF page markers "
            "in skol_staging."
        )
    )
    parser.add_argument(
        "--staging-db",
        default="skol_staging",
        help=(
            "CouchDB database with YEDDA annotations to fix"
            " (default: skol_staging)."
        ),
    )
    parser.add_argument(
        "--training-db",
        default="skol_training",
        help=(
            "CouchDB database used to resolve skol_dev_id"
            " (default: skol_training)."
        ),
    )
    parser.add_argument(
        "--dev-db",
        default="skol_dev",
        help=(
            "CouchDB database with article.txt page markers"
            " (default: skol_dev)."
        ),
    )
    parser.add_argument(
        "--doc-id",
        help="Process only this document ID (useful for spot-checking).",
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
        help="Increase verbosity (-v = per-doc, -vv = full detail).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output.",
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

    staging_db = server[args.staging_db]
    training_db = server[args.training_db]
    dev_db = server[args.dev_db]

    mode = "DRY RUN — " if args.dry_run else ""
    if verbosity >= 1:
        print(
            f"{mode}Fixing YEDDA in '{args.staging_db}'",
            file=sys.stderr,
        )

    summary = fix_staging_database(
        staging_db=staging_db,
        training_db=training_db,
        dev_db=dev_db,
        dry_run=args.dry_run,
        verbosity=verbosity,
        doc_id_filter=args.doc_id,
    )

    if verbosity >= 1:
        print(
            f"\nDone: {summary['docs_processed']} docs processed, "
            f"{summary['docs_changed']} changed, "
            f"{summary['n_malformed']} malformed blocks fixed, "
            f"{summary['n_page_markers']} page markers added.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
