#!/usr/bin/env python3
"""LLM-assisted YEDDA merge for documents with many conflict markers.

The deterministic three-way merge (merge_yedda.py) leaves conflict markers
when the reviewed annotation text diverges too far from the new OCR text.
This script resolves those hard cases by sending the reviewed annotation and
new OCR text to Claude and asking it to produce a clean YEDDA-annotated
version of the new text.

Typical workflow::

    # 1. Run the deterministic merge first
    python fixes/merge_yedda.py --reviewed-db skol_ann_reviewed \\
        --training-db skol_training --output-db skol_ann_merged

    # 2. Estimate token cost for hard documents (>=5 conflicts)
    python fixes/llm_merge_yedda.py --merged-db skol_ann_merged \\
        --reviewed-db skol_ann_reviewed --training-db skol_training \\
        --estimate

    # 3. Run LLM merge for hard documents
    python fixes/llm_merge_yedda.py --merged-db skol_ann_merged \\
        --reviewed-db skol_ann_reviewed --training-db skol_training

Environment variables (or ~/.skol_env):
    ANTHROPIC_API_KEY   Claude API key (required)
    COUCHDB_URL         CouchDB server URL
    COUCHDB_USER        CouchDB username
    COUCHDB_PASSWORD    CouchDB password
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "bin"))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402

# Re-use shared helpers from llm_relabel rather than duplicating them.
from llm_relabel import (  # type: ignore[import]  # noqa: E402
    _BACKOFF_BASE,
    _MAX_RETRIES,
    _PRICING,
    _TAG_DEFINITIONS,
    _YEDDA_BLOCK_RE,
    _lcs_align_blocks,
    _reconstruct_ann,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_ANN_ATTACHMENT = "article.txt.ann"
_TXT_ATTACHMENT = "article.txt"
_DEFAULT_CONFLICT_THRESHOLD = 5
_DEFAULT_MAX_DROP_FRACTION = 0.25

_SYSTEM_PROMPT = (
    "You are a precise taxonomic text annotator. "
    "Output only YEDDA-annotated text with no explanation, "
    "no preamble, and no markdown fences."
)

# ---------------------------------------------------------------------------
# count_conflicts
# ---------------------------------------------------------------------------


def count_conflicts(text: str) -> int:
    """Count the number of conflict markers in a merged YEDDA text.

    Args:
        text: Content of a merged article.txt.ann file.

    Returns:
        Number of ``<<<<<<< annotation`` markers found.
    """
    return text.count("<<<<<<< annotation")


# ---------------------------------------------------------------------------
# build_merge_prompt
# ---------------------------------------------------------------------------

def build_merge_prompt(reviewed_ann: str, new_text: str) -> str:
    """Build the user-turn prompt asking Claude to re-annotate new_text.

    The reviewed annotation provides the authoritative label set; new_text
    is the placement target (raw OCR).  Claude is asked to produce a YEDDA
    .ann file that covers new_text using labels from reviewed_ann, matching
    blocks by semantic content rather than exact string equality.

    Args:
        reviewed_ann: Human-reviewed YEDDA annotation (the label authority).
        new_text: New OCR plaintext to be annotated.

    Returns:
        User-turn prompt string.
    """
    tag_lines = "\n".join(
        f"  {tag.value}: {defn}" for tag, defn in _TAG_DEFINITIONS
    )
    return (
        "You are given two inputs:\n"
        "1. REVIEWED ANNOTATION — a YEDDA-annotated version of a taxonomic "
        "document.  The labels in this file are authoritative.\n"
        "2. NEW OCR TEXT — a new plaintext scan of the same document.  "
        "The text may differ from the annotation (OCR noise, formatting "
        "differences, ligatures, etc.).\n\n"
        "Your task: produce a YEDDA-annotated version of the NEW OCR TEXT "
        "that uses the same labels as the REVIEWED ANNOTATION.  Match blocks "
        "by semantic content — a block in the reviewed annotation corresponds "
        "to the closest semantically equivalent passage in the new OCR"
        " text.\n\n"
        "TAG DEFINITIONS:\n"
        f"{tag_lines}\n\n"
        "RULES:\n"
        "1. Every line of NEW OCR TEXT must appear in exactly one output block "
        "— do not omit any text.\n"
        "2. Use the text from NEW OCR TEXT verbatim as block content — "
        "do not use the reviewed annotation text.\n"
        "3. Choose labels by matching each passage in the new text to the "
        "semantically closest block in the reviewed annotation.\n"
        "4. Page-header blocks (--- PDF Page N Label L --- lines) must "
        "always keep their Page-header tag.\n"
        "5. Holotype blocks: preserve the Holotype tag if it is present in "
        "the reviewed annotation — do not reclassify it.\n"
        "6. Return the complete YEDDA annotation as [@text#Tag*] blocks "
        "separated by blank lines, with no explanation, no markdown fences, "
        "and no other text outside the blocks.\n\n"
        "REVIEWED ANNOTATION:\n"
        f"{reviewed_ann.strip()}\n\n"
        "NEW OCR TEXT:\n"
        f"{new_text.strip()}"
    )


# ---------------------------------------------------------------------------
# parse_llm_response
# ---------------------------------------------------------------------------

def parse_llm_response(response: str, reviewed_ann: str) -> str:
    """Parse and validate the model's YEDDA output.

    Strips markdown fences and any preamble text before the first ``[@``
    block.  If the model returns a different number of blocks than
    reviewed_ann, attempts LCS-based recovery (preserving original block
    text with old tags for any blocks the model dropped).

    Args:
        response: Raw text returned by the Claude API.
        reviewed_ann: The reviewed annotation passed to the model (used as
            the reference block list for LCS recovery).

    Returns:
        Cleaned YEDDA annotation string.

    Raises:
        ValueError: If no YEDDA blocks are found in the response.
    """
    # Strip markdown fences.
    text = re.sub(r"^```[^\n]*\n?", "", response.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()

    # Strip any preamble before the first YEDDA block.
    first_block = text.find("[@")
    if first_block > 0:
        text = text[first_block:]

    new_blocks = _YEDDA_BLOCK_RE.findall(text)
    if not new_blocks:
        raise ValueError("no YEDDA blocks found in model response")

    old_blocks = _YEDDA_BLOCK_RE.findall(reviewed_ann)
    if len(new_blocks) == len(old_blocks):
        return text + ("\n" if not text.endswith("\n") else "")

    # Block count mismatch — attempt LCS recovery.
    # Use reviewed_ann blocks as the reference (authoritative text+tag pairs).
    # The model's output provides updated tags; for any dropped blocks the
    # original reviewed_ann tag is preserved.
    aligned, n_unmatched = _lcs_align_blocks(old_blocks, new_blocks)
    if n_unmatched > 0:
        logging.warning(
            "LCS recovery: %d/%d blocks unmatched in model response; "
            "original tags preserved",
            n_unmatched, len(old_blocks),
        )
    return _reconstruct_ann(aligned)


# ---------------------------------------------------------------------------
# merge_via_llm
# ---------------------------------------------------------------------------

def merge_via_llm(
    client: Any,
    reviewed_ann: str,
    new_text: str,
    doc_id: str,
    model: str = _DEFAULT_MODEL,
    max_drop_fraction: float = _DEFAULT_MAX_DROP_FRACTION,
) -> str:
    """Call the Claude API to merge reviewed_ann labels onto new_text.

    Retries up to _MAX_RETRIES times with exponential backoff on transient
    errors or empty responses.

    Args:
        client: anthropic.Anthropic client.
        reviewed_ann: Human-reviewed YEDDA annotation (label authority).
        new_text: New OCR plaintext to be annotated.
        doc_id: Document ID (used in error messages only).
        model: Claude model ID.
        max_drop_fraction: Maximum fraction of blocks the model may drop
            before the response is rejected and retried.

    Returns:
        YEDDA-annotated string covering new_text with labels from reviewed_ann.

    Raises:
        RuntimeError: If all retries fail.
    """
    user_prompt = build_merge_prompt(reviewed_ann, new_text)
    messages = [{"role": "user", "content": user_prompt}]
    old_blocks = _YEDDA_BLOCK_RE.findall(reviewed_ann)
    old_count = len(old_blocks)
    last_exc: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=_SYSTEM_PROMPT,
                messages=messages,
            )
            raw = response.content[0].text.strip()
            result = parse_llm_response(raw, reviewed_ann)

            # Check drop fraction (parse_llm_response may have done LCS
            # recovery; check the result block count against old_count).
            result_blocks = _YEDDA_BLOCK_RE.findall(result)
            result_count = len(result_blocks)
            if old_count > 0:
                mismatch = abs(old_count - result_count) / old_count
                if mismatch > max_drop_fraction:
                    raise ValueError(
                        f"Block count mismatch for {doc_id}: "
                        f"expected {old_count}, got {result_count} "
                        f"({mismatch:.0%} difference, "
                        f"limit {max_drop_fraction:.0%})"
                    )
            return result

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                logging.warning(
                    "%s: attempt %d failed (%s); retrying in %.0fs",
                    doc_id, attempt, exc, wait,
                )
                time.sleep(wait)

    raise RuntimeError(
        f"All {_MAX_RETRIES} attempts failed for {doc_id}: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _fetch_attachment(db: Any, doc_id: str, filename: str) -> Optional[str]:
    att = db.get_attachment(doc_id, filename)
    if att is None:
        return None
    return att.read().decode("utf-8")


def _write_ann(db: Any, doc_id: str, ann_text: str) -> None:
    doc = db[doc_id]
    db.put_attachment(
        doc,
        ann_text.encode("utf-8"),
        filename=_ANN_ATTACHMENT,
        content_type="text/plain",
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_documents(
    client: Any,
    merged_db: Any,
    reviewed_db: Any,
    training_db: Any,
    model: str,
    conflict_threshold: int,
    dry_run: bool,
    estimate: bool,
    verbosity: int,
    doc_ids: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Find hard documents and run LLM merge for each.

    Args:
        client: anthropic.Anthropic client.
        merged_db: CouchDB database containing merge_yedda output.
        reviewed_db: CouchDB database with human-reviewed annotations.
        training_db: CouchDB database with new OCR plaintext.
        model: Claude model ID.
        conflict_threshold: Minimum number of conflicts to trigger LLM merge.
        dry_run: If True, do not write results back.
        estimate: If True, only count tokens and exit.
        verbosity: Logging verbosity.
        doc_ids: If given, process only these document IDs.

    Returns:
        Summary count dict.
    """
    # Enumerate documents from merged_db.
    if doc_ids:
        ids = doc_ids
    else:
        ids = [
            row.id
            for row in merged_db.view("_all_docs", include_docs=False)
            if not row.id.startswith("_design/")
        ]

    # (doc_id, merged_ann, reviewed_ann, new_text)
    hard_docs: List[Tuple[str, str, str, str]] = []

    for doc_id in ids:
        merged_ann = _fetch_attachment(merged_db, doc_id, _ANN_ATTACHMENT)
        if merged_ann is None:
            continue
        n = count_conflicts(merged_ann)
        if n < conflict_threshold:
            if verbosity >= 2:
                print(
                    f"  skip {doc_id}: {n} conflict(s)"
                    f" < threshold {conflict_threshold}"
                )
            continue

        reviewed_ann = _fetch_attachment(reviewed_db, doc_id, _ANN_ATTACHMENT)
        new_text = _fetch_attachment(training_db, doc_id, _TXT_ATTACHMENT)
        if reviewed_ann is None or new_text is None:
            logging.warning(
                "%s: missing reviewed_ann or new_text — skipped", doc_id
            )
            continue

        hard_docs.append((doc_id, merged_ann, reviewed_ann, new_text))
        if verbosity >= 1:
            print(f"  {doc_id}: {n} conflict(s) → queued for LLM merge")

    if not hard_docs:
        print("No documents above threshold.")
        return {"processed": 0, "written": 0, "failed": 0}

    if estimate:
        total_tokens = 0
        for doc_id, _, reviewed_ann, new_text in hard_docs:
            prompt = build_merge_prompt(reviewed_ann, new_text)
            result = client.messages.count_tokens(
                model=model,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            total_tokens += result.input_tokens
            if verbosity >= 1:
                print(f"  {doc_id}: ~{result.input_tokens} input tokens")

        pricing = _PRICING.get(model, {"input": 3.00, "output": 15.00})
        est_output = total_tokens  # output ≈ input for re-annotation
        cost = (
            total_tokens * pricing["input"]
            + est_output * pricing["output"]
        ) / 1_000_000
        print(
            f"\n{len(hard_docs)} document(s) — "
            f"~{total_tokens + est_output:,} total tokens — "
            f"estimated cost: ${cost:.4f}"
        )
        return {"processed": 0, "written": 0, "failed": 0}

    processed = written = failed = 0
    for doc_id, merged_ann, reviewed_ann, new_text in hard_docs:
        processed += 1
        try:
            result = merge_via_llm(
                client, reviewed_ann, new_text, doc_id=doc_id, model=model
            )
            if verbosity >= 1:
                n_before = count_conflicts(merged_ann)
                n_after = count_conflicts(result)
                print(
                    f"  {doc_id}: conflicts {n_before} → {n_after}"
                    + (" [dry-run]" if dry_run else "")
                )
            if not dry_run:
                _write_ann(merged_db, doc_id, result)
                written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logging.error("%s: LLM merge failed: %s", doc_id, exc)

    return {"processed": processed, "written": written, "failed": failed}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for llm_merge_yedda."""
    parser = argparse.ArgumentParser(
        description=(
            "LLM-assisted YEDDA merge: use Claude to resolve conflict "
            "markers left by the deterministic merge_yedda pass."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--merged-db",
        default="skol_ann_merged",
        metavar="DB",
        help="CouchDB database with deterministic merge output "
             "(default: skol_ann_merged).",
    )
    parser.add_argument(
        "--reviewed-db",
        default="skol_ann_reviewed",
        metavar="DB",
        help="CouchDB database with human-reviewed annotations "
             "(default: skol_ann_reviewed).",
    )
    parser.add_argument(
        "--training-db",
        default="skol_training",
        metavar="DB",
        help="CouchDB database with new OCR plaintext "
             "(default: skol_training).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=_DEFAULT_CONFLICT_THRESHOLD,
        metavar="N",
        help=(
            "Minimum conflicts to trigger LLM merge "
            f"(default: {_DEFAULT_CONFLICT_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Claude model ID (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        metavar="ID",
        help="Process only this document ID (repeatable).",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Count tokens and estimate cost; do not call the generation API.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run LLM merge but do not write results back to CouchDB.",
    )
    parser.add_argument(
        "-v", "--verbosity",
        action="count",
        default=1,
        help="Increase output verbosity (repeatable).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.verbosity < 2 else logging.DEBUG,
        format="%(levelname)s %(message)s",
    )

    config = get_env_config()

    import couchdb  # type: ignore[import]
    server = couchdb.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )

    merged_db = server[args.merged_db]
    reviewed_db = server[args.reviewed_db]
    training_db = server[args.training_db]

    import anthropic  # type: ignore[import]
    api_key = os.environ.get("ANTHROPIC_API_KEY") or config.get(
        "ANTHROPIC_API_KEY", ""
    )
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY not set in environment or ~/.skol_env",
            file=sys.stderr,
        )
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    counts = process_documents(
        client=client,
        merged_db=merged_db,
        reviewed_db=reviewed_db,
        training_db=training_db,
        model=args.model,
        conflict_threshold=args.threshold,
        dry_run=args.dry_run,
        estimate=args.estimate,
        verbosity=args.verbosity,
        doc_ids=args.doc_ids,
    )

    if not args.estimate:
        print(
            f"\nDone: {counts['processed']} processed, "
            f"{counts['written']} written, "
            f"{counts['failed']} failed."
        )


if __name__ == "__main__":
    main()
