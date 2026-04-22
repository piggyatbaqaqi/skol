#!/usr/bin/env python3
"""Tier 2 LLM-assisted relabeling of YEDDA annotations.

Uses the Claude API to upgrade existing YEDDA .ann files to the 12-tag scheme.
Writes relabeled annotations to a staging database for human review in brat.

Workflow::

    # 1. Estimate token cost before running
    python bin/llm_relabel.py --database skol_training --estimate

    # 2. Run relabeling (writes to staging DB)
    python bin/llm_relabel.py --database skol_training [--staging-db skol_training_llm_stage]

    # 3. Convert staging DB to brat for review
    python bin/yedda_to_brat.py --database skol_training_llm_stage --output-dir brat/

    # 4. Review and correct in brat, then round-trip back
    python bin/brat_to_yedda.py ...
    python bin/upload_annotation.py ...

Staging DB documents have the same _id as the source, carry only the
relabeled article.txt.ann attachment, and store source_db and change_count
fields so yedda_to_brat.py can fetch plaintext from the original database.

Environment variables (or ~/.skol_env):
    ANTHROPIC_API_KEY   Claude API key (required)
    COUCHDB_URL         CouchDB server URL
    COUCHDB_USER        CouchDB username
    COUCHDB_PASSWORD    CouchDB password
"""

import argparse
import concurrent.futures
import difflib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import couchdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config
from ingestors.yedda_tags import Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_ANN_ATTACHMENT = "article.txt.ann"
_CHANGES_ATTACHMENT = "changes.json"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds
# Blocks per API call.  Chunks of this size fit comfortably within the 8192
# output-token limit even for verbose documents.
_DEFAULT_CHUNK_SIZE = 150
# Hard ceiling: documents above this block count are skipped even with
# chunking (extreme edge case — thousands of chunks would be impractical).
# Maximum fraction of blocks the model may drop before we give up and retry
# rather than accepting the partial result via LCS alignment.
_DEFAULT_MAX_DROP_FRACTION = 0.25

# Pricing per million tokens (as of 2026-03).
# Only used for --estimate output; not authoritative.
_PRICING: Dict[str, Dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}

_TAG_DEFINITIONS: List[Tuple[Tag, str]] = [
    (Tag.NOMENCLATURE,
     "Complete Taxonomic Citation: formal name + authorship, "
     "nomenclatural act (sp. nov., comb. nov., etc.), synonymy "
     "(nomenclature citations), and any immediately following "
     "iconography / figure-plate references (e.g. 'Pl. 5, Figs. 1–3'). "
     "Include the entire treatment header unit even if it spans several "
     "typographic lines."),
    (Tag.DESCRIPTION,
     "Morphological description: macro/microscopic features, dimensions, "
     "colour, spore ornamentation, etc."),
    (Tag.DIAGNOSIS,
     "Differential diagnosis distinguishing this taxon from related taxa"),
    (Tag.ETYMOLOGY,
     "Derivation or meaning of the scientific name"),
    (Tag.DISTRIBUTION,
     "Deprecated; use Biology instead. Geographic range, locality data,"
     " distributional statements"),
    (Tag.MATERIALS_EXAMINED,
     "List of specimens examined; herbarium/museum accessions; "
     "collector, date, location data (typically >2 lines)"),
    (Tag.MATERIALS_AND_METHODS,
     "Methods section describing techniques, protocols, or procedures "
     "used in the study — not a specimen list"),
    (Tag.TYPE_DESIGNATION,
     "Holotype/lectotype/neotype designation line — short (1-2 lines); "
     "longer specimen lists → Materials-examined instead"),
    (Tag.BIOLOGY,
     "Ecology, host, habitat, substrate, phenology, Geographic range,"
     " locality data, distributional statements"),
    (Tag.PHYLOGENY,
     "Phylogenetic analysis, cladistic placement, molecular phylogeny, "
     "or discussion of evolutionary relationships"),
    (Tag.NEW_COMBINATIONS,
     "Lists or prose of subtaxa being moved to a new taxon."),
    (Tag.NOTES,
     "Additional remarks, taxonomic notes, informal comments inside the Taxonomic part."),
    (Tag.KEY,
     "Identification key (dichotomous or otherwise)"),
    (Tag.FIGURE_CAPTION,
     "Caption for a figure or illustration"),
    (Tag.BIBLIOGRAPHY,
     "Bibliographic reference list entries (literature cited section)"),
    (Tag.TABLE,
     "Tabular data: specimen tables, comparison tables, key tables"),
    (Tag.INDEX,
     "Indexes that are not part of the taxonomic treatment. Inside"
     " a taxonomic treatment use Table."),
    (Tag.TOC,
     "Table of contents."),
    (Tag.MISC_EXPOSITION,
     "Everything else: introduction, discussion, acknowledgements, "
     "references, transitional text"),
    (Tag.PAGE_HEADER,
     "Running head or page header from a printed journal page — "
     "structural pagination artefact, not taxonomic content; "
     "preserve as-is whenever present"),
    (Tag.FIX,
     "The item requires human attention to label correctly."
     " Do not use this tag."),
]

_SYSTEM_PROMPT = (
    "You are a precise taxonomic text classifier. "
    "Output only the relabeled YEDDA annotation text with no explanation, "
    "no preamble, and no markdown fences."
)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)


def _build_user_prompt(ann_text: str) -> str:
    """Build the user-turn prompt for a single .ann document."""
    tag_lines = "\n".join(
        f"  {tag.value}: {defn}" for tag, defn in _TAG_DEFINITIONS
    )
    return (
        "Relabel the YEDDA-annotated text below using the 14-tag scheme.\n\n"
        "TAG DEFINITIONS:\n"
        f"{tag_lines}\n\n"
        "RULES:\n"
        "1. Do NOT change any text content — only the tag name after #.\n"
        "2. Return the complete YEDDA text as [@text#Tag*] blocks "
        "separated by blank lines.\n"
        "3. Holotype blocks → Type-designation (≤2 lines) or "
        "Materials-examined (>2 lines).\n"
        "4. Materials-and-methods describes techniques/protocols; "
        "Materials-examined lists specific specimens.\n"
        "5. Page-header blocks must always keep their Page-header tag — "
        "never reassign them.\n"
        "6. A section heading such as 'Description in vivo' or 'Etymology' "
        "belongs inside the block it introduces — keep it part of that block "
        "rather than splitting it off. This does not apply to the article "
        "title.\n"
        "7. If a block is already correctly labeled, keep it unchanged.\n"
        "8. Use only the 15 tags defined above.\n\n"
        "INPUT:\n"
        f"{ann_text}"
    )


# ---------------------------------------------------------------------------
# Diff: compute per-block changes between old and new .ann
# ---------------------------------------------------------------------------

def diff_yedda(
    old_text: str,
    new_text: str,
) -> List[Dict[str, Any]]:
    """Return a list of changed blocks between two .ann texts.

    Args:
        old_text: Original YEDDA text.
        new_text: Relabeled YEDDA text.

    Returns:
        List of dicts with keys: block_index, old_tag, new_tag, snippet
        (first 80 chars of block text).
    """
    old_blocks = _YEDDA_BLOCK_RE.findall(old_text)
    new_blocks = _YEDDA_BLOCK_RE.findall(new_text)

    changes: List[Dict[str, Any]] = []
    for i, (old, new) in enumerate(zip(old_blocks, new_blocks)):
        old_tag, new_tag = old[1].strip(), new[1].strip()
        if old_tag != new_tag:
            changes.append({
                "block_index": i,
                "old_tag": old_tag,
                "new_tag": new_tag,
                "snippet": old[0].strip()[:80],
            })
    return changes


# ---------------------------------------------------------------------------
# LCS-based partial-response recovery
# ---------------------------------------------------------------------------

Block = Tuple[str, str]  # (text, tag)


def _lcs_align_blocks(
    old_blocks: List[Block],
    new_blocks: List[Block],
) -> Tuple[List[Block], int]:
    """Align new_blocks onto old_blocks using LCS on block text.

    When the model drops or merges blocks the response has fewer entries than
    the original.  This function uses difflib.SequenceMatcher to find which
    old blocks survived (matched by text content) and substitutes their new
    tags, while keeping the original tag for any block the model omitted.

    Args:
        old_blocks: Original (text, tag) pairs from the source .ann.
        new_blocks: (text, tag) pairs from the model response.

    Returns:
        Tuple of (aligned_blocks, n_unmatched) where aligned_blocks has the
        same length as old_blocks, and n_unmatched is the count of old blocks
        whose text was not found in the response (original tags preserved).
    """
    old_texts = [b[0] for b in old_blocks]
    new_texts = [b[0] for b in new_blocks]

    matcher = difflib.SequenceMatcher(
        None, old_texts, new_texts, autojunk=False
    )
    old_to_new: Dict[int, int] = {}
    for old_start, new_start, length in matcher.get_matching_blocks():
        for k in range(length):
            old_to_new[old_start + k] = new_start + k

    aligned: List[Block] = []
    n_unmatched = 0
    for i, (text, old_tag) in enumerate(old_blocks):
        if i in old_to_new:
            aligned.append((text, new_blocks[old_to_new[i]][1]))
        else:
            aligned.append((text, old_tag))
            n_unmatched += 1

    return aligned, n_unmatched


def _reconstruct_ann(blocks: List[Block]) -> str:
    """Serialize (text, tag) pairs back to YEDDA .ann format."""
    return (
        "\n\n".join(f"[@{text}#{tag}*]" for text, tag in blocks) + "\n"
    )


# ---------------------------------------------------------------------------
# Single-document relabeling via Claude API
# ---------------------------------------------------------------------------

def relabel_ann(
    client: Any,
    ann_text: str,
    model: str,
    doc_id: str,
    max_drop_fraction: float = _DEFAULT_MAX_DROP_FRACTION,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Call the Claude API to relabel a single .ann file.

    When the model returns fewer blocks than the input (a common failure
    mode — the model merges adjacent same-tag blocks or silently drops
    short blocks), the response is recovered via LCS alignment if the
    fraction of dropped blocks is within max_drop_fraction.  Blocks the
    model omitted keep their original tags and are not reported as changes.

    Retries up to _MAX_RETRIES times with exponential backoff on transient
    errors and on above-threshold block-count mismatches.

    Args:
        client: anthropic.Anthropic client.
        ann_text: Original YEDDA-annotated text.
        model: Claude model ID.
        doc_id: Document ID (for error messages only).
        max_drop_fraction: Maximum fraction of blocks the model may omit
            before the response is rejected and retried.  Defaults to
            _DEFAULT_MAX_DROP_FRACTION (10 %).

    Returns:
        Tuple of (relabeled_ann_text, changes) where changes is the diff
        against the original.  The returned ann_text always has the same
        block count as the input.

    Raises:
        RuntimeError: If all retries fail or the model consistently drops
            more than max_drop_fraction of blocks.
    """
    user_prompt = _build_user_prompt(ann_text)
    messages = [{"role": "user", "content": user_prompt}]
    last_exc: Optional[Exception] = None
    old_blocks = _YEDDA_BLOCK_RE.findall(ann_text)
    old_count = len(old_blocks)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=_SYSTEM_PROMPT,
                messages=messages,
            )
            new_text = response.content[0].text.strip()
            new_blocks = _YEDDA_BLOCK_RE.findall(new_text)
            new_count = len(new_blocks)

            if new_count == 0:
                raise ValueError(
                    f"Response contains no YEDDA blocks for {doc_id}"
                )

            if new_count == old_count:
                # Perfect match — fast path.
                changes = diff_yedda(ann_text, new_text)
                return new_text, changes

            # Model returned a different number of blocks — attempt LCS
            # recovery in both directions.
            #
            # Drops (new_count < old_count): unmatched old blocks keep their
            # original tags.  Extra new blocks are simply ignored in both
            # cases; LCS discards them.
            #
            # Splits (new_count > old_count): the model split a block at a
            # perceived section boundary.  LCS matches each old block to the
            # best-fitting new block; the orphaned new blocks are discarded.
            # The net effect is that the split is silently merged back and the
            # old block receives the tag of whichever new block it matched.
            mismatch_fraction = abs(old_count - new_count) / old_count
            if mismatch_fraction > max_drop_fraction:
                # Too far from the original — retry.
                raise ValueError(
                    f"Block count mismatch for {doc_id}: "
                    f"expected {old_count}, got {new_count} "
                    f"({mismatch_fraction:.0%} difference, "
                    f"limit {max_drop_fraction:.0%})"
                )

            # Within acceptable range — recover via LCS alignment.
            aligned, n_unmatched = _lcs_align_blocks(old_blocks, new_blocks)
            direction = (
                "dropped" if new_count < old_count else "extra (split)"
            )
            logging.warning(
                "%s: LCS recovery — %d/%d blocks %s in model response;"
                " original tags preserved for unmatched blocks",
                doc_id, n_unmatched, old_count, direction,
            )
            recovered_text = _reconstruct_ann(aligned)
            changes = diff_yedda(ann_text, recovered_text)
            return recovered_text, changes

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                time.sleep(wait)

    raise RuntimeError(
        f"All {_MAX_RETRIES} attempts failed for {doc_id}: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Chunked relabeling for large documents
# ---------------------------------------------------------------------------

def chunk_ann(ann_text: str, chunk_size: int) -> List[str]:
    """Split a YEDDA .ann text into chunks of at most chunk_size blocks.

    Each chunk is a valid standalone .ann string.  The split is on block
    boundaries only — no block is ever broken in half.

    Args:
        ann_text: Full YEDDA-annotated text.
        chunk_size: Maximum number of YEDDA blocks per chunk.

    Returns:
        List of YEDDA strings, each containing at most chunk_size blocks.
        Returns a single-element list when the document is already small enough.
    """
    matches = list(_YEDDA_BLOCK_RE.finditer(ann_text))
    if not matches:
        return [ann_text]
    chunks: List[str] = []
    for i in range(0, len(matches), chunk_size):
        batch = matches[i:i + chunk_size]
        chunk = "\n\n".join(
            f"[@{m.group(1)}#{m.group(2)}*]" for m in batch
        ) + "\n"
        chunks.append(chunk)
    return chunks


def relabel_ann_chunked(
    client: Any,
    ann_text: str,
    model: str,
    doc_id: str,
    chunk_size: int,
    max_drop_fraction: float = _DEFAULT_MAX_DROP_FRACTION,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Relabel a .ann file, splitting into chunks if necessary.

    Documents with more blocks than chunk_size are split and each chunk is
    sent as a separate API call.  The results are reassembled into a single
    .ann string and the block indices in the change list are adjusted to
    refer to the full document.

    Blocks at chunk boundaries may lack context from adjacent chunks; the
    resulting label may be less accurate than for blocks in the middle of a
    chunk.  This is an accepted trade-off: partial labeling is more useful
    than skipping the document entirely.

    Args:
        client: anthropic.Anthropic client.
        ann_text: Original YEDDA-annotated text.
        model: Claude model ID.
        doc_id: Document ID (used in log messages and error strings).
        chunk_size: Maximum blocks per API call.
        max_drop_fraction: Passed through to relabel_ann for each chunk.

    Returns:
        Tuple of (relabeled_ann_text, changes) where changes covers all
        chunks with block indices relative to the full document.
    """
    chunks = chunk_ann(ann_text, chunk_size)
    if len(chunks) == 1:
        return relabel_ann(
            client, ann_text, model, doc_id, max_drop_fraction
        )

    relabeled_parts: List[str] = []
    all_changes: List[Dict[str, Any]] = []
    block_offset = 0

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id} [chunk {idx + 1}/{len(chunks)}]"
        new_chunk, changes = relabel_ann(
            client, chunk, model, chunk_id, max_drop_fraction
        )
        relabeled_parts.append(new_chunk.rstrip("\n"))
        for c in changes:
            adjusted = dict(c)
            adjusted["block_index"] += block_offset
            all_changes.append(adjusted)
        block_offset += len(_YEDDA_BLOCK_RE.findall(chunk))

    new_text = "\n\n".join(relabeled_parts) + "\n"
    return new_text, all_changes


# ---------------------------------------------------------------------------
# Token estimation (no generation)
# ---------------------------------------------------------------------------

def estimate_tokens(
    client: Any,
    ann_texts: Dict[str, str],
    model: str,
) -> Dict[str, Any]:
    """Count input tokens for all documents without generating any output.

    Uses the Anthropic count_tokens API.  Output tokens are estimated as
    equal to input tokens (relabeling does not change text length materially).

    Args:
        client: anthropic.Anthropic client.
        ann_texts: Mapping of doc_id → ann_text.
        model: Claude model ID.

    Returns:
        Dict with keys: doc_count, total_input_tokens, est_output_tokens,
        est_total_tokens, est_input_cost_usd, est_output_cost_usd,
        est_total_cost_usd.
    """
    total_input = 0
    for doc_id, ann_text in ann_texts.items():
        user_prompt = _build_user_prompt(ann_text)
        result = client.messages.count_tokens(
            model=model,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        total_input += result.input_tokens

    # Output: relabeling returns text of similar length to input
    est_output = total_input

    pricing = _PRICING.get(model, {"input": 3.00, "output": 15.00})
    input_cost = total_input * pricing["input"] / 1_000_000
    output_cost = est_output * pricing["output"] / 1_000_000

    return {
        "doc_count": len(ann_texts),
        "total_input_tokens": total_input,
        "est_output_tokens": est_output,
        "est_total_tokens": total_input + est_output,
        "est_input_cost_usd": round(input_cost, 4),
        "est_output_cost_usd": round(output_cost, 4),
        "est_total_cost_usd": round(input_cost + output_cost, 4),
    }


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _load_ann_texts(
    db: Any,
    doc_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, str]:
    """Fetch article.txt.ann attachments from a CouchDB database.

    Args:
        db: CouchDB database object.
        doc_ids: If given, only fetch these doc IDs.
        limit: Maximum number of documents to fetch.
        verbosity: Logging verbosity.

    Returns:
        Mapping of doc_id → ann_text for documents that have the attachment.
    """
    result: Dict[str, str] = {}

    if doc_ids:
        ids_to_fetch = doc_ids
    else:
        ids_to_fetch = [
            row.id
            for row in db.view("_all_docs", include_docs=False)
            if not row.id.startswith("_design/")
        ]

    if limit:
        ids_to_fetch = ids_to_fetch[:limit]

    for doc_id in ids_to_fetch:
        att = db.get_attachment(doc_id, _ANN_ATTACHMENT)
        if att is None:
            if verbosity >= 2:
                print(f"  skip {doc_id}: no {_ANN_ATTACHMENT}", file=sys.stderr)
            continue
        result[doc_id] = att.read().decode("utf-8")

    return result


def _ensure_staging_db(
    server: Any,
    staging_db_name: str,
) -> Any:
    """Get or create the staging CouchDB database."""
    if staging_db_name in server:
        return server[staging_db_name]
    return server.create(staging_db_name)


def _write_to_staging(
    staging_db: Any,
    source_db_name: str,
    doc_id: str,
    new_ann_text: str,
    changes: List[Dict[str, Any]],
) -> None:
    """Write a relabeled .ann attachment to the staging database.

    Creates a minimal document if one does not already exist.

    Args:
        staging_db: CouchDB staging database object.
        source_db_name: Name of the source database (stored for reference).
        doc_id: Document ID.
        new_ann_text: Relabeled YEDDA text.
        changes: Change list from diff_yedda.
    """
    if doc_id in staging_db:
        doc = staging_db[doc_id]
    else:
        doc = {
            "_id": doc_id,
            "source_db": source_db_name,
            "llm_relabeled": True,
            "relabeled_at": datetime.now(timezone.utc).isoformat(),
            "change_count": len(changes),
        }
        staging_db.save(doc)
        doc = staging_db[doc_id]

    staging_db.put_attachment(
        doc,
        new_ann_text.encode("utf-8"),
        filename=_ANN_ATTACHMENT,
        content_type="text/plain",
    )
    if changes:
        doc = staging_db[doc_id]  # refresh rev after ann attachment
        staging_db.put_attachment(
            doc,
            json.dumps(changes, ensure_ascii=False).encode("utf-8"),
            filename=_CHANGES_ATTACHMENT,
            content_type="application/json",
        )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_documents(
    client: Any,
    source_db: Any,
    source_db_name: str,
    staging_db: Any,
    ann_texts: Dict[str, str],
    model: str,
    workers: int,
    dry_run: bool,
    log_file: Optional[Path],
    verbosity: int,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    max_drop_fraction: float = _DEFAULT_MAX_DROP_FRACTION,
) -> Dict[str, int]:
    """Relabel all documents and write results to the staging database.

    Args:
        client: anthropic.Anthropic client.
        source_db: CouchDB source database object.
        source_db_name: Name of the source database.
        staging_db: CouchDB staging database object.
        ann_texts: Mapping of doc_id → ann_text.
        model: Claude model ID.
        workers: Number of parallel API workers.
        dry_run: If True, do not write to staging DB.
        log_file: Path to write JSONL change log, or None.
        verbosity: Logging verbosity.
        chunk_size: Maximum blocks per API call; larger documents are split.
        max_drop_fraction: Maximum fraction of blocks the model may drop
            before the response is rejected and retried.

    Returns:
        Summary counts.
    """
    docs_processed = 0
    docs_changed = 0
    docs_failed = 0
    docs_skipped = 0
    total_blocks_changed = 0
    log_entries: List[Dict[str, Any]] = []

    def _process_one(
        item: Tuple[str, str],
    ) -> Tuple[str, Optional[str], List[Dict[str, Any]], Optional[str]]:
        doc_id, ann_text = item
        try:
            new_text, changes = relabel_ann_chunked(
                client, ann_text, model, doc_id,
                chunk_size, max_drop_fraction,
            )
            return doc_id, new_text, changes, None
        except Exception as exc:  # noqa: BLE001
            return doc_id, None, [], str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, item): item[0]
            for item in ann_texts.items()
        }
        for future in concurrent.futures.as_completed(futures):
            doc_id, new_text, changes, error = future.result()
            docs_processed += 1

            if error:
                docs_failed += 1
                print(
                    f"  FAILED {doc_id}: {error}", file=sys.stderr
                )
                continue

            if changes:
                docs_changed += 1
                total_blocks_changed += len(changes)

            if verbosity >= 1:
                status = f"{len(changes)} block(s) changed" if changes else "unchanged"
                print(f"  {doc_id}: {status}")
            if verbosity >= 2:
                for c in changes:
                    print(
                        f"    block {c['block_index']}: "
                        f"{c['old_tag']} → {c['new_tag']}  "
                        f"{c['snippet']!r}"
                    )

            if not dry_run and new_text is not None:
                _write_to_staging(
                    staging_db, source_db_name, doc_id, new_text, changes
                )

            if log_file is not None and changes:
                log_entries.append({
                    "doc_id": doc_id,
                    "changes": changes,
                })

    if log_file is not None and log_entries:
        with log_file.open("w") as fh:
            for entry in log_entries:
                fh.write(json.dumps(entry) + "\n")
        if verbosity >= 1:
            print(f"\nChange log written to {log_file}", file=sys.stderr)

    return {
        "docs_processed": docs_processed,
        "docs_changed": docs_changed,
        "docs_failed": docs_failed,
        "docs_skipped": docs_skipped,
        "total_blocks_changed": total_blocks_changed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the llm_relabel CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Tier 2 LLM-assisted relabeling: use Claude to upgrade "
            "8-tag YEDDA annotations to the 12-tag scheme."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--database",
        metavar="DB",
        help="CouchDB source database containing article.txt.ann attachments.",
    )
    src_group.add_argument(
        "--experiment",
        metavar="NAME",
        help="Experiment name (resolves annotations database automatically).",
    )

    parser.add_argument(
        "--staging-db",
        metavar="DB",
        help=(
            "CouchDB database for LLM-relabeled output "
            "(default: <source_db>_llm_stage)."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=_DEFAULT_MODEL,
        metavar="MODEL",
        dest="model",
        help=f"Claude model ID (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="N",
        help="Parallel API workers (default: 5).",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help=(
            "Count input tokens and estimate cost without generating output. "
            "Use this before a full run to check token budget."
        ),
    )
    parser.add_argument(
        "--doc-id",
        metavar="ID[,ID,...]",
        help="Process only specific document IDs (comma-separated).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Process at most N documents.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_CHUNK_SIZE,
        metavar="N",
        help=(
            f"Maximum YEDDA blocks per API call (default: {_DEFAULT_CHUNK_SIZE}). "
            "Larger documents are split into chunks of this size."
        ),
    )
    parser.add_argument(
        "--max-drop-fraction",
        type=float,
        default=_DEFAULT_MAX_DROP_FRACTION,
        metavar="FRAC",
        help=(
            "Maximum fraction of blocks the model may drop before the "
            f"response is retried (default: {_DEFAULT_MAX_DROP_FRACTION}). "
            "Responses within this limit are recovered via LCS alignment; "
            "dropped blocks keep their original tags."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have a relabeled .ann in staging DB.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but do not write to the staging DB.",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        help=(
            "Write per-document change log as JSONL to FILE "
            "(default: llm_relabel_<timestamp>.jsonl)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v, -vv).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors.",
    )

    args = parser.parse_args()
    verbosity = 0 if args.quiet else args.verbose

    # ------------------------------------------------------------------ setup
    import anthropic as anthropic_lib
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY environment variable not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = anthropic_lib.Anthropic(api_key=api_key)

    config = get_env_config()
    server = couchdb.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )

    if args.experiment:
        source_db_name = (
            config.get("annotations_db_name") or config["ingest_db_name"]
        )
    else:
        source_db_name = args.database

    try:
        source_db = server[source_db_name]
    except Exception:
        print(
            f"Error: source database '{source_db_name}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    staging_db_name = args.staging_db or f"{source_db_name}_llm_stage"

    # ---------------------------------------------------------- load .ann files
    doc_ids: Optional[List[str]] = (
        [d.strip() for d in args.doc_id.split(",") if d.strip()]
        if args.doc_id else None
    )

    if verbosity >= 1:
        print(
            f"Loading .ann files from '{source_db_name}'…",
            file=sys.stderr,
        )

    ann_texts = _load_ann_texts(
        source_db,
        doc_ids=doc_ids,
        limit=args.limit,
        verbosity=verbosity,
    )

    if not ann_texts:
        print("No .ann files found. Nothing to do.", file=sys.stderr)
        sys.exit(0)

    # Skip documents already processed in staging DB
    if args.skip_existing:
        try:
            staging_db_existing = server[staging_db_name]
            before = len(ann_texts)
            ann_texts = {
                doc_id: text
                for doc_id, text in ann_texts.items()
                if doc_id not in staging_db_existing
                or staging_db_existing.get_attachment(doc_id, _ANN_ATTACHMENT) is None
            }
            skipped = before - len(ann_texts)
            if verbosity >= 1 and skipped:
                print(
                    f"Skipping {skipped} already-processed document(s).",
                    file=sys.stderr,
                )
        except Exception:
            pass  # staging DB doesn't exist yet; nothing to skip

    if verbosity >= 1:
        print(
            f"Documents to process: {len(ann_texts)}  model: {args.model}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------ estimate mode
    if args.estimate:
        if verbosity >= 1:
            print(
                "\nCounting tokens (no generation)…",
                file=sys.stderr,
            )
        stats = estimate_tokens(client, ann_texts, args.model)
        print(f"\nToken estimate for {stats['doc_count']} document(s):")
        print(f"  Input tokens (actual):    {stats['total_input_tokens']:>10,}")
        print(f"  Output tokens (estimate): {stats['est_output_tokens']:>10,}")
        print(f"  Total tokens (estimate):  {stats['est_total_tokens']:>10,}")
        print()
        print(f"  Model: {args.model}")
        pricing = _PRICING.get(args.model, {"input": 3.00, "output": 15.00})
        print(
            f"  Pricing: ${pricing['input']:.2f}/1M input, "
            f"${pricing['output']:.2f}/1M output"
        )
        print(f"  Est. input cost:  ${stats['est_input_cost_usd']:.4f}")
        print(f"  Est. output cost: ${stats['est_output_cost_usd']:.4f}")
        print(f"  Est. total cost:  ${stats['est_total_cost_usd']:.4f}")
        return

    # ---------------------------------------------------------- full run
    staging_db = _ensure_staging_db(server, staging_db_name)

    if args.dry_run:
        if verbosity >= 1:
            print("\nDRY RUN — no writes to staging DB.", file=sys.stderr)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_path = (
        Path(args.log_file) if args.log_file
        else Path(f"llm_relabel_{timestamp}.jsonl")
    )

    if verbosity >= 1:
        print(
            f"\nStaging DB: '{staging_db_name}'  "
            f"Workers: {args.workers}",
            file=sys.stderr,
        )

    summary = process_documents(
        client=client,
        source_db=source_db,
        source_db_name=source_db_name,
        staging_db=staging_db,
        ann_texts=ann_texts,
        model=args.model,
        workers=args.workers,
        dry_run=args.dry_run,
        log_file=log_path if not args.dry_run else None,
        verbosity=verbosity,
        chunk_size=args.chunk_size,
        max_drop_fraction=args.max_drop_fraction,
    )

    if verbosity >= 1:
        skipped_msg = (
            f", {summary['docs_skipped']} skipped (too large)"
            if summary['docs_skipped'] else ""
        )
        print(
            f"\nDone: {summary['docs_processed']} processed, "
            f"{summary['docs_changed']} changed "
            f"({summary['total_blocks_changed']} blocks), "
            f"{summary['docs_failed']} failed"
            f"{skipped_msg}.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
