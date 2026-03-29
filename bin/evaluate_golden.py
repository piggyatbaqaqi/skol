#!/usr/bin/env python3
"""Evaluate annotation quality against the golden dataset.

Compares predicted annotations (from a classifier experiment) against
golden-standard annotations. Reports per-tag precision/recall/F1,
token-level IoU, and a confusion matrix.

Examples:
    # Evaluate the production experiment against hand annotations
    python evaluate_golden.py --experiment production \\
        --golden-db skol_golden_ann_hand

    # Evaluate JATS annotations against hand annotations
    python evaluate_golden.py --predicted-db skol_golden_ann_jats \\
        --golden-db skol_golden_ann_hand

    # Full report with confusion matrix
    python evaluate_golden.py --experiment production \\
        --golden-db skol_golden_ann_hand -v -v
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow running as a script or as a module.
if __name__ == "__main__" and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    bin_dir = str(Path(__file__).resolve().parent)
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)

from env_config import get_env_config

# YEDDA block pattern: [@text#Tag*]
# Tag must be a short alphanumeric-with-hyphens name (no newlines, not too
# long) to avoid mismatching when passage text contains '#' characters.
_YEDDA_BLOCK_RE = re.compile(
    r"\[@\s*(.*?)\s*#([A-Za-z][A-Za-z0-9_-]{0,49})\*\]", re.DOTALL
)


# ---------------------------------------------------------------------------
# YEDDA parsing
# ---------------------------------------------------------------------------

def parse_yedda_blocks(
    yedda_text: str,
) -> List[Tuple[str, str]]:
    """Parse YEDDA into (text, tag) blocks.

    Args:
        yedda_text: YEDDA-annotated string.

    Returns:
        List of (text, tag) tuples.
    """
    blocks: List[Tuple[str, str]] = []
    for match in _YEDDA_BLOCK_RE.finditer(yedda_text):
        text = match.group(1).strip()
        tag = match.group(2).strip()
        if text:
            blocks.append((text, tag))
    return blocks


def blocks_to_char_tags(
    blocks: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Expand blocks into per-character (char, tag) pairs.

    Args:
        blocks: List of (text, tag) tuples.

    Returns:
        List of (character, tag) pairs.
    """
    result: List[Tuple[str, str]] = []
    for text, tag in blocks:
        for ch in text:
            result.append((ch, tag))
    return result


# ---------------------------------------------------------------------------
# Label collapsing
# ---------------------------------------------------------------------------

_KEEP_TAGS = frozenset({"Nomenclature", "Description"})


def collapse_tag(tag: str) -> str:
    """Collapse a fine-grained YEDDA tag to the 3-class scheme."""
    return tag if tag in _KEEP_TAGS else "Misc-exposition"


# ---------------------------------------------------------------------------
# Plaintext-anchored character-level evaluation
# ---------------------------------------------------------------------------

def _normalize_ws(text: str) -> str:
    """Collapse whitespace runs to single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def project_blocks_to_plaintext(
    plaintext: str,
    blocks: List[Tuple[str, str]],
    collapse: bool = True,
) -> List[Optional[str]]:
    """Project YEDDA blocks onto plaintext character positions.

    For each block, find its normalized text within the normalized
    plaintext (scanning forward) and assign the block's tag to
    those character positions in the *original* plaintext.

    Args:
        plaintext: The raw article.txt content.
        blocks: List of (text, tag) tuples from YEDDA.
        collapse: If True, collapse tags to 3-class scheme.

    Returns:
        List of length len(plaintext) where each element is
        a tag name or None (untagged).
    """
    tags: List[Optional[str]] = [None] * len(plaintext)
    norm_plain = _normalize_ws(plaintext)

    # Build a mapping from normalized-string positions back to
    # original-string positions.
    norm_to_orig: List[int] = []
    for i, ch in enumerate(plaintext):
        if ch.isspace():
            # In a whitespace run, only the first space maps to
            # the normalized single space.
            if i == 0 or not plaintext[i - 1].isspace():
                norm_to_orig.append(i)
            # Subsequent whitespace chars don't appear in norm
        else:
            norm_to_orig.append(i)
    # Strip leading/trailing: find where norm_plain starts/ends
    # in the norm_to_orig mapping.
    lstripped = len(re.match(r"\s*", plaintext).group())  # type: ignore[union-attr]
    # Rebuild mapping for stripped+collapsed version
    norm_to_orig = []
    orig_idx = lstripped
    for nch in norm_plain:
        # Advance orig_idx to match nch
        while orig_idx < len(plaintext):
            if plaintext[orig_idx].isspace() and nch == " ":
                norm_to_orig.append(orig_idx)
                # Skip remaining whitespace in original
                orig_idx += 1
                while (orig_idx < len(plaintext)
                       and plaintext[orig_idx].isspace()):
                    orig_idx += 1
                break
            elif plaintext[orig_idx] == nch:
                norm_to_orig.append(orig_idx)
                orig_idx += 1
                break
            else:
                orig_idx += 1

    scan_pos = 0
    for block_text, tag in blocks:
        if collapse:
            tag = collapse_tag(tag)
        norm_block = _normalize_ws(block_text)
        if not norm_block:
            continue

        idx = norm_plain.find(norm_block, scan_pos)
        if idx == -1:
            # Try from beginning as fallback (overlapping blocks)
            idx = norm_plain.find(norm_block)
        if idx == -1:
            continue

        # Map normalized range back to original positions
        end_idx = idx + len(norm_block)
        if idx < len(norm_to_orig) and end_idx <= len(norm_to_orig):
            orig_start = norm_to_orig[idx]
            orig_end = norm_to_orig[end_idx - 1] + 1
            # Tag all original chars in this range
            for j in range(orig_start, min(orig_end, len(plaintext))):
                tags[j] = tag

        scan_pos = end_idx

    return tags


def compute_char_metrics(
    pred_tags: List[Optional[str]],
    gold_tags: List[Optional[str]],
) -> Dict[str, Any]:
    """Compute character-level precision/recall/F1 per tag.

    Args:
        pred_tags: Predicted tag per character (None = untagged).
        gold_tags: Golden tag per character (None = untagged).

    Returns:
        Dict with per-tag metrics, macro average, and accuracy.
    """
    assert len(pred_tags) == len(gold_tags)

    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    correct = 0
    total = 0

    for p, g in zip(pred_tags, gold_tags):
        if p is None and g is None:
            continue
        total += 1
        if p == g:
            correct += 1
            if p is not None:
                tp[p] += 1
        else:
            if p is not None:
                fp[p] += 1
            if g is not None:
                fn[g] += 1

    all_tags = sorted(set(tp.keys()) | set(fp.keys()) | set(fn.keys()))

    metrics: Dict[str, Dict[str, float]] = {}
    for tag in all_tags:
        t = tp[tag]
        f_p = fp[tag]
        f_n = fn[tag]
        precision = t / (t + f_p) if (t + f_p) > 0 else 0.0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        iou = t / (t + f_p + f_n) if (t + f_p + f_n) > 0 else 0.0
        metrics[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "tp": t,
            "fp": f_p,
            "fn": f_n,
        }

    if all_tags:
        macro_p = sum(m["precision"] for m in metrics.values()) / len(all_tags)
        macro_r = sum(m["recall"] for m in metrics.values()) / len(all_tags)
        macro_f1 = sum(m["f1"] for m in metrics.values()) / len(all_tags)
        macro_iou = sum(m["iou"] for m in metrics.values()) / len(all_tags)
    else:
        macro_p = macro_r = macro_f1 = macro_iou = 0.0

    metrics["macro_avg"] = {
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
        "iou": macro_iou,
        "tp": sum(tp.values()),
        "fp": sum(fp.values()),
        "fn": sum(fn.values()),
    }

    return {
        "tag_metrics": metrics,
        "accuracy": correct / total if total > 0 else 0.0,
        "total_chars": total,
    }


# ---------------------------------------------------------------------------
# Alignment (block-level, legacy)
# ---------------------------------------------------------------------------

def align_blocks(
    predicted: List[Tuple[str, str]],
    golden: List[Tuple[str, str]],
) -> List[Tuple[Optional[str], Optional[str], str]]:
    """Align predicted and golden blocks by text content.

    Uses exact text matching. Blocks present in both are aligned;
    blocks in only one side are marked as unmatched.

    Args:
        predicted: Predicted (text, tag) blocks.
        golden: Golden (text, tag) blocks.

    Returns:
        List of (predicted_tag, golden_tag, text) tuples.
        predicted_tag is None for golden-only blocks;
        golden_tag is None for predicted-only blocks.
    """
    # Build lookup by text
    pred_by_text: Dict[str, List[str]] = defaultdict(list)
    for text, tag in predicted:
        pred_by_text[text].append(tag)

    gold_by_text: Dict[str, List[str]] = defaultdict(list)
    for text, tag in golden:
        gold_by_text[text].append(tag)

    all_texts = set(pred_by_text.keys()) | set(gold_by_text.keys())
    aligned: List[Tuple[Optional[str], Optional[str], str]] = []

    for text in all_texts:
        p_tags = pred_by_text.get(text, [])
        g_tags = gold_by_text.get(text, [])

        # Match tags in order
        max_len = max(len(p_tags), len(g_tags))
        for i in range(max_len):
            p_tag = p_tags[i] if i < len(p_tags) else None
            g_tag = g_tags[i] if i < len(g_tags) else None
            aligned.append((p_tag, g_tag, text))

    return aligned


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_tag_metrics(
    aligned: List[Tuple[Optional[str], Optional[str], str]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-tag precision, recall, F1 from aligned blocks.

    Args:
        aligned: List of (predicted_tag, golden_tag, text) tuples.

    Returns:
        Dict mapping tag name to {precision, recall, f1, tp, fp, fn}.
        Includes a "macro_avg" entry with macro-averaged metrics.
    """
    # Count TP, FP, FN per tag
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for pred_tag, gold_tag, text in aligned:
        if pred_tag == gold_tag and pred_tag is not None:
            tp[pred_tag] += 1
        else:
            if pred_tag is not None:
                fp[pred_tag] += 1
            if gold_tag is not None:
                fn[gold_tag] += 1

    all_tags = sorted(set(tp.keys()) | set(fp.keys()) | set(fn.keys()))

    metrics: Dict[str, Dict[str, float]] = {}
    for tag in all_tags:
        t = tp[tag]
        f_p = fp[tag]
        f_n = fn[tag]

        precision = t / (t + f_p) if (t + f_p) > 0 else 0.0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": t,
            "fp": f_p,
            "fn": f_n,
        }

    # Macro average
    if all_tags:
        macro_p = sum(m["precision"] for m in metrics.values()) / len(all_tags)
        macro_r = sum(m["recall"] for m in metrics.values()) / len(all_tags)
        macro_f1 = sum(m["f1"] for m in metrics.values()) / len(all_tags)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    metrics["macro_avg"] = {
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
        "tp": sum(tp.values()),
        "fp": sum(fp.values()),
        "fn": sum(fn.values()),
    }

    return metrics


def compute_token_iou(
    predicted_blocks: List[Tuple[str, str]],
    golden_blocks: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Compute token-level IoU (Intersection over Union) per tag.

    Builds a character-level stream from each set of blocks and
    compares tag assignments character by character.

    Args:
        predicted_blocks: Predicted (text, tag) blocks.
        golden_blocks: Golden (text, tag) blocks.

    Returns:
        Dict mapping tag name to IoU score. Includes "micro_avg".
    """
    pred_chars = blocks_to_char_tags(predicted_blocks)
    gold_chars = blocks_to_char_tags(golden_blocks)

    # Build tag-to-character-set mappings
    pred_tag_chars: Dict[str, Set[int]] = defaultdict(set)
    gold_tag_chars: Dict[str, Set[int]] = defaultdict(set)

    for i, (ch, tag) in enumerate(pred_chars):
        pred_tag_chars[tag].add(i)

    for i, (ch, tag) in enumerate(gold_chars):
        gold_tag_chars[tag].add(i)

    all_tags = sorted(
        set(pred_tag_chars.keys()) | set(gold_tag_chars.keys())
    )

    iou_scores: Dict[str, float] = {}
    total_intersection = 0
    total_union = 0

    for tag in all_tags:
        p_set = pred_tag_chars.get(tag, set())
        g_set = gold_tag_chars.get(tag, set())
        intersection = len(p_set & g_set)
        union = len(p_set | g_set)
        iou = intersection / union if union > 0 else 0.0
        iou_scores[tag] = iou
        total_intersection += intersection
        total_union += union

    iou_scores["micro_avg"] = (
        total_intersection / total_union if total_union > 0 else 0.0
    )

    return iou_scores


def compute_confusion_matrix(
    aligned: List[Tuple[Optional[str], Optional[str], str]],
) -> Dict[str, Dict[str, int]]:
    """Build a confusion matrix from aligned blocks.

    Args:
        aligned: List of (predicted_tag, golden_tag, text) tuples.

    Returns:
        Dict[golden_tag][predicted_tag] = count.
        Uses "(none)" for missing tags.
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())

    for pred_tag, gold_tag, text in aligned:
        p = pred_tag or "(none)"
        g = gold_tag or "(none)"
        matrix[g][p] += 1

    return {k: dict(v) for k, v in matrix.items()}


# ---------------------------------------------------------------------------
# Evaluation over documents
# ---------------------------------------------------------------------------

def evaluate_documents(
    predicted_db,
    golden_db,
    verbosity: int,
    plaintext_db=None,
) -> Dict[str, Any]:
    """Evaluate predicted annotations against golden annotations.

    When *plaintext_db* is provided, uses character-level evaluation
    anchored to the shared plaintext (handles different block
    granularity).  Otherwise falls back to block-level alignment.

    Args:
        predicted_db: CouchDB database with predicted article.txt.ann.
        golden_db: CouchDB database with golden article.txt.ann.
        verbosity: Logging verbosity.
        plaintext_db: CouchDB database with article.txt (optional).

    Returns:
        Evaluation result dict with per-tag and aggregate metrics.
    """
    all_aligned: List[Tuple[Optional[str], Optional[str], str]] = []
    all_pred_blocks: List[Tuple[str, str]] = []
    all_gold_blocks: List[Tuple[str, str]] = []
    # Character-level accumulators
    all_pred_chars: List[Optional[str]] = []
    all_gold_chars: List[Optional[str]] = []
    doc_count = 0
    matched = 0
    skipped = 0

    for row in golden_db.view("_all_docs", include_docs=False):
        if row.id.startswith("_design/"):
            continue
        doc_count += 1

        # Get golden annotation
        gold_att = golden_db.get_attachment(row.id, "article.txt.ann")
        if gold_att is None:
            if verbosity >= 2:
                print(
                    f"  {row.id}: no golden article.txt.ann",
                    file=sys.stderr,
                )
            skipped += 1
            continue

        gold_text = gold_att.read().decode("utf-8")
        gold_blocks = parse_yedda_blocks(gold_text)

        if not gold_blocks:
            skipped += 1
            continue

        # Get predicted annotation (try .txt.ann first, then .pdf.ann)
        pred_att = None
        for ann_name in ("article.txt.ann", "article.pdf.ann"):
            try:
                pred_att = predicted_db.get_attachment(row.id, ann_name)
            except Exception:
                pass
            if pred_att is not None:
                break

        if pred_att is None:
            if verbosity >= 2:
                print(
                    f"  {row.id}: no predicted annotation",
                    file=sys.stderr,
                )
            skipped += 1
            continue

        pred_text = pred_att.read().decode("utf-8")
        pred_blocks = parse_yedda_blocks(pred_text)

        # Character-level evaluation via shared plaintext
        if plaintext_db is not None:
            pt_att = plaintext_db.get_attachment(row.id, "article.txt")
            if pt_att is not None:
                plaintext = pt_att.read().decode("utf-8")
                p_chars = project_blocks_to_plaintext(
                    plaintext, pred_blocks, collapse=True,
                )
                g_chars = project_blocks_to_plaintext(
                    plaintext, gold_blocks, collapse=True,
                )
                all_pred_chars.extend(p_chars)
                all_gold_chars.extend(g_chars)

        # Block-level alignment (legacy)
        aligned = align_blocks(pred_blocks, gold_blocks)
        all_aligned.extend(aligned)
        all_pred_blocks.extend(pred_blocks)
        all_gold_blocks.extend(gold_blocks)
        matched += 1

        if verbosity >= 2:
            print(
                f"  {row.id}: {len(pred_blocks)} predicted, "
                f"{len(gold_blocks)} golden blocks",
                file=sys.stderr,
            )

    if verbosity >= 1:
        print(
            f"\nDocuments: {doc_count} total, {matched} matched, "
            f"{skipped} skipped",
            file=sys.stderr,
        )

    # Compute block-level metrics (legacy)
    tag_metrics = compute_tag_metrics(all_aligned)
    iou_scores = compute_token_iou(all_pred_blocks, all_gold_blocks)
    confusion = compute_confusion_matrix(all_aligned)

    result: Dict[str, Any] = {
        "documents_total": doc_count,
        "documents_matched": matched,
        "documents_skipped": skipped,
        "block_tag_metrics": tag_metrics,
        "token_iou": iou_scores,
        "confusion_matrix": confusion,
    }

    # Character-level metrics (primary when plaintext available)
    if all_pred_chars:
        char_result = compute_char_metrics(all_pred_chars, all_gold_chars)
        result["char_tag_metrics"] = char_result["tag_metrics"]
        result["char_accuracy"] = char_result["accuracy"]
        result["char_total"] = char_result["total_chars"]
        result["tag_metrics"] = char_result["tag_metrics"]
        result["macro_f1"] = (
            char_result["tag_metrics"]
            .get("macro_avg", {})
            .get("f1", 0.0)
        )
    else:
        result["tag_metrics"] = tag_metrics
        result["macro_f1"] = (
            tag_metrics.get("macro_avg", {}).get("f1", 0.0)
        )

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(
    evaluation: Dict[str, Any],
    predicted_name: str,
    golden_name: str,
    verbosity: int,
) -> str:
    """Format evaluation results as a markdown report.

    Args:
        evaluation: Result dict from evaluate_documents().
        predicted_name: Name of the predicted database.
        golden_name: Name of the golden database.
        verbosity: Logging verbosity (controls detail level).

    Returns:
        Markdown-formatted report string.
    """
    lines: List[str] = []
    lines.append(f"# Evaluation: {predicted_name} vs {golden_name}")
    lines.append("")
    lines.append(f"- Documents matched: {evaluation['documents_matched']}")
    lines.append(f"- Documents skipped: {evaluation['documents_skipped']}")
    lines.append(
        f"- **Macro F1: {evaluation['macro_f1']:.4f}**"
    )
    if "char_accuracy" in evaluation:
        lines.append(
            f"- Character accuracy: {evaluation['char_accuracy']:.4f}"
        )
        lines.append(
            f"- Total characters evaluated: "
            f"{evaluation['char_total']:,}"
        )
    lines.append("")

    # Character-level metrics (primary)
    char_metrics = evaluation.get("char_tag_metrics")
    if char_metrics:
        lines.append(
            "## Character-level Metrics (3-class, collapsed)"
        )
        lines.append("")
        lines.append(
            "| Tag | Precision | Recall | F1 | IoU "
            "| TP | FP | FN |"
        )
        lines.append(
            "|-----|-----------|--------|-----|-----"
            "|----|----|-----|"
        )
        for tag in sorted(char_metrics.keys()):
            if tag == "macro_avg":
                continue
            m = char_metrics[tag]
            lines.append(
                f"| {tag} | {m['precision']:.3f} | "
                f"{m['recall']:.3f} | {m['f1']:.3f} | "
                f"{m['iou']:.3f} | "
                f"{m['tp']:.0f} | {m['fp']:.0f} | "
                f"{m['fn']:.0f} |"
            )
        m = char_metrics.get("macro_avg", {})
        lines.append(
            f"| **Macro Avg** | "
            f"**{m.get('precision', 0):.3f}** | "
            f"**{m.get('recall', 0):.3f}** | "
            f"**{m.get('f1', 0):.3f}** | "
            f"**{m.get('iou', 0):.3f}** | "
            f"{m.get('tp', 0):.0f} | {m.get('fp', 0):.0f} | "
            f"{m.get('fn', 0):.0f} |"
        )
        lines.append("")

    # Block-level metrics (legacy)
    lines.append("## Block-level Metrics (legacy)")
    lines.append("")
    lines.append(
        "| Tag | Precision | Recall | F1 | TP | FP | FN |"
    )
    lines.append(
        "|-----|-----------|--------|-----|----|----|-----|"
    )

    tag_metrics = evaluation.get("block_tag_metrics", evaluation["tag_metrics"])
    for tag in sorted(tag_metrics.keys()):
        if tag == "macro_avg":
            continue
        m = tag_metrics[tag]
        lines.append(
            f"| {tag} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['tp']:.0f} | {m['fp']:.0f} | "
            f"{m['fn']:.0f} |"
        )

    # Macro average row
    m = tag_metrics.get("macro_avg", {})
    lines.append(
        f"| **Macro Avg** | **{m.get('precision', 0):.3f}** | "
        f"**{m.get('recall', 0):.3f}** | **{m.get('f1', 0):.3f}** | "
        f"{m.get('tp', 0):.0f} | {m.get('fp', 0):.0f} | "
        f"{m.get('fn', 0):.0f} |"
    )
    lines.append("")

    # Token-level IoU
    iou = evaluation.get("token_iou", {})
    if iou:
        lines.append("## Token-level IoU")
        lines.append("")
        lines.append("| Tag | IoU |")
        lines.append("|-----|-----|")
        for tag in sorted(iou.keys()):
            if tag == "micro_avg":
                continue
            lines.append(f"| {tag} | {iou[tag]:.3f} |")
        if "micro_avg" in iou:
            lines.append(
                f"| **Micro Avg** | **{iou['micro_avg']:.3f}** |"
            )
        lines.append("")

    # Confusion matrix (verbose only)
    if verbosity >= 2:
        confusion = evaluation.get("confusion_matrix", {})
        if confusion:
            lines.append("## Confusion Matrix")
            lines.append("")
            lines.append("Rows = golden, Columns = predicted")
            lines.append("")

            all_tags = sorted(
                set().union(
                    *[set(v.keys()) for v in confusion.values()],
                    confusion.keys(),
                )
            )
            header = "| | " + " | ".join(all_tags) + " |"
            sep = "|---" * (len(all_tags) + 1) + "|"
            lines.append(header)
            lines.append(sep)
            for g_tag in all_tags:
                row_data = confusion.get(g_tag, {})
                cells = [str(row_data.get(p_tag, 0)) for p_tag in all_tags]
                lines.append(f"| {g_tag} | " + " | ".join(cells) + " |")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate annotations against the golden dataset.",
    )

    # Source of predictions
    pred_group = parser.add_mutually_exclusive_group(required=True)
    pred_group.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (reads predictions from its ingest DB).",
    )
    pred_group.add_argument(
        "--predicted-db",
        type=str,
        help="Database containing predicted article.txt.ann.",
    )

    # Golden standard
    parser.add_argument(
        "--golden-db",
        type=str,
        required=True,
        help="Database with golden article.txt.ann annotations.",
    )

    # Plaintext source for character-level evaluation
    parser.add_argument(
        "--plaintext-db",
        type=str,
        default=None,
        help=(
            "Database with article.txt plaintext for "
            "character-level evaluation (default: skol_golden)."
        ),
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON results to this file.",
    )
    parser.add_argument(
        "--save-to-experiment",
        action="store_true",
        help="Save evaluation to the experiment's evaluation field.",
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
    config = get_env_config()

    import couchdb as couchdb_lib
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )

    # Resolve predicted database
    # Prefer annotations DB if set (from experiment), fall back to ingest
    if args.experiment:
        predicted_db_name = (
            config.get('annotations_db_name')
            or config['ingest_db_name']
        )
    else:
        predicted_db_name = args.predicted_db

    # Open databases
    try:
        predicted_db = server[predicted_db_name]
    except Exception:
        print(
            f"Error: predicted database '{predicted_db_name}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        golden_db = server[args.golden_db]
    except Exception:
        print(
            f"Error: golden database '{args.golden_db}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Open plaintext DB for character-level evaluation
    plaintext_db_name = args.plaintext_db or "skol_golden"
    plaintext_db = None
    try:
        plaintext_db = server[plaintext_db_name]
    except Exception:
        if verbosity >= 1:
            print(
                f"Warning: plaintext database "
                f"'{plaintext_db_name}' not found; "
                f"character-level evaluation disabled.",
                file=sys.stderr,
            )

    if verbosity >= 1:
        print(
            f"Evaluating: {predicted_db_name} vs {args.golden_db}",
            file=sys.stderr,
        )

    # Run evaluation
    evaluation = evaluate_documents(
        predicted_db, golden_db, verbosity,
        plaintext_db=plaintext_db,
    )

    # Format and print report
    report = format_report(
        evaluation, predicted_db_name, args.golden_db, verbosity,
    )
    print(report)

    # Save JSON output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(evaluation, f, indent=2, default=str)
        if verbosity >= 1:
            print(f"\nJSON results written to {args.output}", file=sys.stderr)

    # Save to experiment
    if args.save_to_experiment and args.experiment:
        from datetime import datetime, timezone
        exp_db_name = config.get("experiments_database", "skol_experiments")
        try:
            exp_db = server[exp_db_name]
            experiment_doc = exp_db[args.experiment]
        except Exception as exc:
            print(
                f"Error: could not load experiment '{args.experiment}' "
                f"for saving results: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        experiment_doc["evaluation"] = {
            "macro_f1": evaluation["macro_f1"],
            "per_tag": {
                tag: m["f1"]
                for tag, m in evaluation["tag_metrics"].items()
                if tag != "macro_avg"
            },
            "documents_matched": evaluation["documents_matched"],
            "golden_database": args.golden_db,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        experiment_doc["status"] = "evaluated"
        experiment_doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        exp_db.save(experiment_doc)
        if verbosity >= 1:
            print(
                f"\nSaved evaluation to experiment '{args.experiment}'",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
