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

from rapidfuzz import fuzz as _fuzz

# Number of context lines to capture on each side of a page marker
_CONTEXT_LINES = 3
# Fractional tolerance for line-count vote (e.g. 0.5 = within 50%)
_LINE_COUNT_TOLERANCE = 0.50
# Minimum votes required to accept a candidate without falling back
_MIN_VOTES = 2
# Fuzzy-match threshold (rapidfuzz scores 0-100) for context window votes.
# partial_ratio — best-window match of the context string inside the YEDDA
# character window.
_CONTEXT_THRESHOLD = 75
# Threshold for the after-context vote, which uses fuzz.ratio on an
# equal-length prefix of the after window (anchored start comparison).
# Higher than _CONTEXT_THRESHOLD because anchoring is stricter and because
# ctx_after always starts with the page's first-content line (the header).
_AFTER_THRESHOLD = 90
# Number of characters of stripped YEDDA text to extract before and after the
# candidate insertion point when computing context votes.  At ~70 chars per
# line this covers roughly 5-6 lines on each side, comfortably spanning
# single orphan blocks left by a previous strip_page_markers pass.
_CTX_CHARS = 400
# Hard forward-window multiplier.  The search stops once the cumulative YEDDA
# line gap from prev_match exceeds accum_txt * _MAX_JUMP_FACTOR.  YEDDA
# non-empty lines run ~1.35× txt lines, so a factor of 4 allows ~3× the
# expected page distance before giving up.  Prevents one bad 2-vote match
# from consuming thousands of blocks and making subsequent pages unreachable.
_MAX_JUMP_FACTOR = 4

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

def parse_page_markers(
    article_txt: str,
) -> List[Tuple[int, str, str, int, str, str]]:
    """Parse PDF page boundary markers from a ``skol_dev`` ``article.txt``.

    ``PDFSectionExtractor`` inserts lines of the form::

        --- PDF Page N Label L ---

    The first non-empty content line after each marker is treated as the
    page header text (typically a running head from the journal).

    In addition to the header text, up to ``_CONTEXT_LINES`` non-empty
    content lines are collected on each side of the marker to provide
    voting signals for :func:`add_page_markers`.

    Args:
        article_txt: Full text of a ``skol_dev`` ``article.txt`` attachment.

    Returns:
        List of ``(page_num, page_label, header_text, lines_before,
        context_before, context_after)`` tuples, one per page boundary
        found.  Pages whose first content line is empty are omitted.

        - *lines_before*: number of ``article.txt`` lines between the
          previous page marker and this one (0 for the first marker).
        - *context_before*: up to ``_CONTEXT_LINES`` non-empty lines
          immediately before the marker, joined by spaces.
        - *context_after*: up to ``_CONTEXT_LINES`` non-empty lines
          after the header line, joined by spaces.
    """
    pages: List[Tuple[int, str, str, int, str, str]] = []
    lines = article_txt.splitlines()
    prev_marker_idx = 0

    i = 0
    while i < len(lines):
        m = _PAGE_MARKER_RE.match(lines[i])
        if m:
            page_num = int(m.group(1))
            page_label = m.group(2)
            lines_before = i - prev_marker_idx

            # Context before: last _CONTEXT_LINES non-empty non-marker lines.
            before: List[str] = []
            for k in range(i - 1, -1, -1):
                ln = lines[k].strip()
                if ln and not _PAGE_MARKER_RE.match(lines[k]):
                    before.append(ln)
                    if len(before) >= _CONTEXT_LINES:
                        break
            context_before = " ".join(reversed(before))

            # Find first non-empty content line after the marker (header).
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1

            if j < len(lines):
                header_text = lines[j].strip()

                # Context after: the header line itself followed by the
                # next _CONTEXT_LINES non-empty lines.  Including the
                # header at the start lets add_page_markers use an
                # anchored prefix comparison — only the YEDDA block that
                # *begins* with the page's first content will score high.
                after: List[str] = [header_text]
                k = j + 1
                while k < len(lines) and len(after) <= _CONTEXT_LINES:
                    ln = lines[k].strip()
                    if ln and not _PAGE_MARKER_RE.match(lines[k]):
                        after.append(ln)
                    k += 1
                context_after = " ".join(after)

                pages.append((
                    page_num, page_label, header_text,
                    lines_before, context_before, context_after,
                ))
                prev_marker_idx = i
        i += 1
    return pages


def _normalise(text: str) -> str:
    """Collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", text.strip())


def strip_page_markers(yedda: str) -> Tuple[str, int]:
    """Remove previously-inserted PDF page markers from YEDDA blocks.

    Removes the ``--- PDF Page N Label L ---\\n`` prefix line from any
    ``Page-header`` block and resets the tag to ``Misc-exposition``.
    Blocks whose text is entirely the marker line (no content after it)
    are discarded.

    This makes :func:`add_page_markers` re-entrant: calling
    ``strip_page_markers`` before ``add_page_markers`` lets the voting
    algorithm re-place markers cleanly on a document that was already
    processed by a previous run.

    Args:
        yedda: YEDDA-annotated text.

    Returns:
        Tuple of (stripped_yedda, n_stripped).
    """
    out_parts: List[str] = []
    n_stripped = 0
    for match in _YEDDA_BLOCK_RE.finditer(yedda):
        text = match.group(1)
        tag = match.group(2)
        stripped = text.lstrip()
        if tag == "Page-header" and stripped.startswith("--- PDF Page"):
            n_stripped += 1
            nl = stripped.find("\n")
            remainder = stripped[nl + 1:].strip() if nl >= 0 else ""
            # Discard bare running-head page numbers (< 4 chars after
            # stripping).  These arise when a previous run placed a page
            # marker on a short OCR page number such as "7" or "8".
            # Keeping them creates orphan single-char blocks that sit
            # between context-bearing blocks and their natural successors,
            # disrupting character-window context matching.
            if remainder and len(remainder) >= 4:
                out_parts.append(f"[@{remainder}#Misc-exposition*]")
            # Blocks with no content (or too-short content) are discarded.
        elif text.strip():
            out_parts.append(f"[@{text}#{tag}*]")
    return "\n\n".join(out_parts) + ("\n" if out_parts else ""), n_stripped


def add_page_markers(
    yedda: str,
    pages: List[Tuple[int, str, str, int, str, str]],
) -> Tuple[str, int]:
    """Prefix matching blocks with ``--- PDF Page N Label L ---`` lines.

    For each page in *pages*, search forward from the previous match for a
    YEDDA block that scores at least ``_MIN_VOTES`` on three signals:

    * **Before-context vote**: the normalised *context_before* (last few lines
      before the page marker in ``article.txt``) scores above
      ``_CONTEXT_THRESHOLD`` against the ``_CTX_CHARS``-character window of
      YEDDA plaintext immediately *before* the candidate insertion point.
    * **After-context vote**: the normalised *context_after* (first few lines
      after the marker) scores above ``_CONTEXT_THRESHOLD`` against the
      ``_CTX_CHARS``-character window of YEDDA plaintext immediately *after*
      the candidate insertion point.
    * **Line-count vote**: cumulative non-empty lines in YEDDA blocks between
      the previous match and this candidate is within
      ``_LINE_COUNT_TOLERANCE`` of the accumulated *lines_before* distance
      from the source file (summed across any skipped pages since the last
      match).

    Using character windows rather than adjacent-block comparisons makes the
    votes robust to orphan blocks (stripped bare-digit running heads),
    sentence fragments split across page boundaries, and variable block
    granularity.  The header text from ``article.txt`` need not correspond to
    a YEDDA block boundary at all.

    The first candidate reaching ``_MIN_VOTES`` wins.  If none does, the
    first candidate with any context vote is used as a fallback.  Searching
    always moves forward from the previous match to guarantee monotonicity.

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

    # Build mutable list of [text, tag] from all parsed blocks.
    blocks: List[List[str]] = []
    for match in _YEDDA_BLOCK_RE.finditer(yedda):
        text = match.group(1)
        tag = match.group(2)
        if text.strip():
            blocks.append([text, tag])

    # Precompute cumulative non-empty line counts for line-count voting.
    # cumulative[i] = total non-empty lines in blocks 0..i-1
    cumulative: List[int] = [0] * (len(blocks) + 1)
    for idx, blk in enumerate(blocks):
        cumulative[idx + 1] = cumulative[idx] + sum(
            bool(ln.strip()) for ln in blk[0].splitlines()
        )

    # Precompute a stripped-plaintext string and per-block character offsets
    # for the character-window context votes.  Each block contributes its
    # normalised text followed by a single space separator.
    plain_parts: List[str] = []
    block_char_starts: List[int] = []
    pos = 0
    for blk_text, _blk_tag in blocks:
        block_char_starts.append(pos)
        norm_part = _normalise(blk_text)
        plain_parts.append(norm_part)
        pos += len(norm_part) + 1   # +1 for the space separator
    plain_yedda = " ".join(plain_parts)
    block_char_starts.append(pos)  # sentinel: end of string

    n_added = 0
    min_block = 0       # forward-only: each match must be >= this index
    prev_match = 0      # block index of the last matched page (for line count)
    # Accumulated txt lines since the last successful match.  Sums
    # lines_before across skipped pages so the line-count vote and window cap
    # stay proportional to the true distance from the last anchor.
    accum_txt: int = 0

    for page_num, page_label, _header_text, lines_before, ctx_before, ctx_after \
            in pages:
        accum_txt += lines_before

        norm_ctx_before = _normalise(ctx_before)
        norm_ctx_after = _normalise(ctx_after)
        # Skip pages with no context signals at all (cannot place without any
        # text evidence — e.g. a page whose content section is empty).
        if not norm_ctx_before and not norm_ctx_after:
            continue

        marker_line = f"--- PDF Page {page_num} Label {page_label} ---"

        # Hard window cap: stop searching once the cumulative YEDDA line gap
        # from prev_match exceeds the accumulated txt lines * _MAX_JUMP_FACTOR.
        max_gap = accum_txt * _MAX_JUMP_FACTOR if accum_txt > 0 else None

        best_fallback: Optional[int] = None   # first block with any context vote

        for i in range(min_block, len(blocks)):
            text, _tag = blocks[i]
            if text.startswith("--- PDF Page"):
                continue

            # Hard cap: exit early once the window is exhausted.
            if max_gap is not None:
                if cumulative[i] - cumulative[prev_match] > max_gap:
                    break

            # --- Vote A: before-context (character window) ---
            # Extract _CTX_CHARS of plaintext immediately before block i's
            # start.  Spans across block boundaries and orphan blocks.
            char_pos = block_char_starts[i]
            before_window = plain_yedda[max(0, char_pos - _CTX_CHARS):char_pos]
            before_vote = (
                bool(norm_ctx_before)
                and _fuzz.partial_ratio(norm_ctx_before, before_window)
                >= _CONTEXT_THRESHOLD
            )

            # --- Vote B: after-context (anchored prefix) ---
            # ctx_after starts with the page's first-content line (header),
            # so comparing it against an equal-length prefix of after_window
            # checks whether block i *begins* with the new page's text.
            # fuzz.ratio (not partial_ratio) is used to anchor the comparison
            # to the start of the window, preventing preceding blocks whose
            # after_window merely *contains* the next page's text from scoring
            # high.  _AFTER_THRESHOLD is stricter than _CONTEXT_THRESHOLD to
            # account for this tighter matching criterion.
            after_window = plain_yedda[char_pos:char_pos + _CTX_CHARS]
            if norm_ctx_after:
                after_prefix = after_window[:len(norm_ctx_after)]
                after_vote = (
                    _fuzz.ratio(norm_ctx_after, after_prefix)
                    >= _AFTER_THRESHOLD
                )
            else:
                after_vote = False

            if (before_vote or after_vote) and best_fallback is None:
                best_fallback = i

            # --- Vote C: line-count proximity ---
            # Compare the YEDDA line gap from the last match against the
            # accumulated article.txt line distance (accum_txt).
            yedda_gap = cumulative[i] - cumulative[prev_match]
            if accum_txt > 0:
                line_count_vote = (
                    abs(yedda_gap - accum_txt) / accum_txt
                    <= _LINE_COUNT_TOLERANCE
                )
            else:
                line_count_vote = yedda_gap == 0

            votes = int(before_vote) + int(after_vote) + int(line_count_vote)

            if votes >= _MIN_VOTES:
                blocks[i][0] = marker_line + "\n" + text.lstrip()
                blocks[i][1] = "Page-header"
                n_added += 1
                prev_match = i
                min_block = i + 1
                accum_txt = 0
                best_fallback = None
                break
        else:
            # No candidate reached _MIN_VOTES — fall back to first context match.
            if best_fallback is not None:
                i = best_fallback
                blocks[i][0] = marker_line + "\n" + blocks[i][0].lstrip()
                blocks[i][1] = "Page-header"
                n_added += 1
                prev_match = i
                min_block = i + 1
                accum_txt = 0

    out_parts = [f"[@{text}#{tag}*]" for text, tag in blocks]
    return "\n\n".join(out_parts) + ("\n" if out_parts else ""), n_added


# ---------------------------------------------------------------------------
# Combined fix for one document
# ---------------------------------------------------------------------------

def fix_yedda(
    yedda: str,
    pages: Optional[List[Tuple[int, str, str, int, str, str]]] = None,
) -> Tuple[str, Dict[str, int]]:
    """Apply all fixes to a YEDDA string.

    Applies :func:`fix_malformed_blocks` repeatedly until no malformed blocks
    remain (handles multi-level nesting), then (if *pages* is provided)
    calls :func:`strip_page_markers` to remove any previously-placed markers
    before calling :func:`add_page_markers`.  Stripping first makes the
    function re-entrant: re-running on a document that was already processed
    re-places markers using the current algorithm rather than leaving old
    placements in place.

    Args:
        yedda: Raw YEDDA-annotated text.
        pages: Page boundary list from :func:`parse_page_markers`, or None
            to skip the page-marker step.

    Returns:
        Tuple of (fixed_yedda, stats) where *stats* contains:
        ``n_malformed``, ``n_stripped``, ``n_page_markers``.
    """
    fixed = yedda
    n_malformed = 0
    for _ in range(20):  # safety limit; nested depth never approaches 20
        step, n = fix_malformed_blocks(fixed)
        n_malformed += n
        if n == 0:
            break
        fixed = step

    n_stripped = 0
    n_page_markers = 0
    if pages:
        fixed, n_stripped = strip_page_markers(fixed)
        fixed, n_page_markers = add_page_markers(fixed, pages)
    return fixed, {
        "n_malformed": n_malformed,
        "n_stripped": n_stripped,
        "n_page_markers": n_page_markers,
    }


# ---------------------------------------------------------------------------
# CouchDB batch processing
# ---------------------------------------------------------------------------

def _get_pages_for_doc(
    doc_id: str,
    training_db: Any,
    dev_db: Any,
) -> Optional[List[Tuple[int, str, str, int, str, str]]]:
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
        n_stripped, n_page_markers.
    """
    totals: Dict[str, int] = {
        "docs_processed": 0,
        "docs_changed": 0,
        "n_malformed": 0,
        "n_stripped": 0,
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

        if (stats["n_malformed"] == 0
                and stats["n_stripped"] == 0
                and stats["n_page_markers"] == 0):
            continue  # nothing changed

        totals["docs_changed"] += 1
        totals["n_malformed"] += stats["n_malformed"]
        totals["n_stripped"] += stats["n_stripped"]
        totals["n_page_markers"] += stats["n_page_markers"]

        if verbosity >= 1:
            print(
                f"  {doc_id}: "
                f"{stats['n_malformed']} malformed fixed, "
                f"{stats['n_stripped']} markers stripped, "
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
            f"{summary['n_stripped']} markers stripped, "
            f"{summary['n_page_markers']} page markers added.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
