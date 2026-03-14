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
_YEDDA_BLOCK_RE = re.compile(r"\[@\s*(.*?)\s*#([^*]+)\*\]", re.DOTALL)


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
# Alignment
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
) -> Dict[str, Any]:
    """Evaluate predicted annotations against golden annotations.

    Iterates over all documents in the golden database, looks for
    corresponding predicted annotations, and computes metrics.

    Args:
        predicted_db: CouchDB database with predicted article.txt.ann.
        golden_db: CouchDB database with golden article.txt.ann.
        verbosity: Logging verbosity.

    Returns:
        Evaluation result dict with per-tag and aggregate metrics.
    """
    all_aligned: List[Tuple[Optional[str], Optional[str], str]] = []
    all_pred_blocks: List[Tuple[str, str]] = []
    all_gold_blocks: List[Tuple[str, str]] = []
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

        # Align and accumulate
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

    # Compute metrics
    tag_metrics = compute_tag_metrics(all_aligned)
    iou_scores = compute_token_iou(all_pred_blocks, all_gold_blocks)
    confusion = compute_confusion_matrix(all_aligned)

    return {
        "documents_total": doc_count,
        "documents_matched": matched,
        "documents_skipped": skipped,
        "tag_metrics": tag_metrics,
        "token_iou": iou_scores,
        "confusion_matrix": confusion,
        "macro_f1": tag_metrics.get("macro_avg", {}).get("f1", 0.0),
    }


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
    lines.append("")

    # Per-tag metrics table
    lines.append("## Per-tag Metrics")
    lines.append("")
    lines.append(
        "| Tag | Precision | Recall | F1 | TP | FP | FN |"
    )
    lines.append(
        "|-----|-----------|--------|-----|----|----|-----|"
    )

    tag_metrics = evaluation["tag_metrics"]
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
    # When --experiment is used, get_env_config() already resolved ingest_db_name
    if args.experiment:
        predicted_db_name = config['ingest_db_name']
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

    if verbosity >= 1:
        print(
            f"Evaluating: {predicted_db_name} vs {args.golden_db}",
            file=sys.stderr,
        )

    # Run evaluation
    evaluation = evaluate_documents(predicted_db, golden_db, verbosity)

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
