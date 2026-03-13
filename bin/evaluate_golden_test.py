"""Tests for golden dataset evaluation metrics.

Tests the core evaluation functions with synthetic YEDDA fixtures.
"""

import unittest
import sys
from pathlib import Path

# Allow imports from parent directory.
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
bin_dir = str(Path(__file__).resolve().parent)
if bin_dir not in sys.path:
    sys.path.insert(0, bin_dir)

from evaluate_golden import (
    align_blocks,
    blocks_to_char_tags,
    compute_confusion_matrix,
    compute_tag_metrics,
    compute_token_iou,
    format_report,
    parse_yedda_blocks,
)


class TestParseYeddaBlocks(unittest.TestCase):
    """Tests for parse_yedda_blocks."""

    def test_single_block(self):
        result = parse_yedda_blocks("[@Some text#Nomenclature*]")
        self.assertEqual(result, [("Some text", "Nomenclature")])

    def test_multiple_blocks(self):
        yedda = (
            "[@Block one#Nomenclature*]\n\n"
            "[@Block two#Description*]"
        )
        result = parse_yedda_blocks(yedda)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ("Block one", "Nomenclature"))
        self.assertEqual(result[1], ("Block two", "Description"))

    def test_empty_string(self):
        self.assertEqual(parse_yedda_blocks(""), [])

    def test_multiline_block(self):
        yedda = "[@Line one\nLine two#Description*]"
        result = parse_yedda_blocks(yedda)
        self.assertEqual(len(result), 1)
        self.assertIn("Line one\nLine two", result[0][0])

    def test_skips_empty_blocks(self):
        yedda = "[@  #Misc-exposition*]\n\n[@Real text#Description*]"
        result = parse_yedda_blocks(yedda)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("Real text", "Description"))


class TestAlignBlocks(unittest.TestCase):
    """Tests for align_blocks."""

    def test_perfect_alignment(self):
        pred = [("text A", "Nomenclature"), ("text B", "Description")]
        gold = [("text A", "Nomenclature"), ("text B", "Description")]
        aligned = align_blocks(pred, gold)
        for p_tag, g_tag, text in aligned:
            self.assertEqual(p_tag, g_tag)

    def test_mismatched_tags(self):
        pred = [("text A", "Description")]
        gold = [("text A", "Nomenclature")]
        aligned = align_blocks(pred, gold)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0][0], "Description")
        self.assertEqual(aligned[0][1], "Nomenclature")

    def test_predicted_only(self):
        pred = [("text A", "Nomenclature")]
        gold = []
        aligned = align_blocks(pred, gold)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0][0], "Nomenclature")
        self.assertIsNone(aligned[0][1])

    def test_golden_only(self):
        pred = []
        gold = [("text A", "Nomenclature")]
        aligned = align_blocks(pred, gold)
        self.assertEqual(len(aligned), 1)
        self.assertIsNone(aligned[0][0])
        self.assertEqual(aligned[0][1], "Nomenclature")


class TestComputeTagMetrics(unittest.TestCase):
    """Tests for compute_tag_metrics."""

    def test_perfect_match(self):
        aligned = [
            ("Nomenclature", "Nomenclature", "text A"),
            ("Description", "Description", "text B"),
        ]
        metrics = compute_tag_metrics(aligned)
        self.assertEqual(metrics["Nomenclature"]["precision"], 1.0)
        self.assertEqual(metrics["Nomenclature"]["recall"], 1.0)
        self.assertEqual(metrics["Nomenclature"]["f1"], 1.0)
        self.assertEqual(metrics["macro_avg"]["f1"], 1.0)

    def test_no_match(self):
        aligned = [
            ("Description", "Nomenclature", "text A"),
        ]
        metrics = compute_tag_metrics(aligned)
        # Description: tp=0, fp=1 -> precision=0
        self.assertEqual(metrics["Description"]["precision"], 0.0)
        # Nomenclature: tp=0, fn=1 -> recall=0
        self.assertEqual(metrics["Nomenclature"]["recall"], 0.0)

    def test_partial_match(self):
        aligned = [
            ("Nomenclature", "Nomenclature", "text A"),  # TP
            ("Description", "Nomenclature", "text B"),    # FP for Desc, FN for Nom
            ("Description", "Description", "text C"),     # TP
        ]
        metrics = compute_tag_metrics(aligned)

        # Nomenclature: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F1=0.667
        self.assertAlmostEqual(metrics["Nomenclature"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["Nomenclature"]["recall"], 0.5)

        # Description: tp=1, fp=1, fn=0 -> P=0.5, R=1.0, F1=0.667
        self.assertAlmostEqual(metrics["Description"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["Description"]["recall"], 1.0)

    def test_empty_aligned(self):
        metrics = compute_tag_metrics([])
        self.assertEqual(metrics["macro_avg"]["f1"], 0.0)

    def test_predicted_only_blocks(self):
        aligned = [("Nomenclature", None, "text")]
        metrics = compute_tag_metrics(aligned)
        self.assertEqual(metrics["Nomenclature"]["tp"], 0)
        self.assertEqual(metrics["Nomenclature"]["fp"], 1)
        self.assertEqual(metrics["Nomenclature"]["precision"], 0.0)

    def test_golden_only_blocks(self):
        aligned = [(None, "Nomenclature", "text")]
        metrics = compute_tag_metrics(aligned)
        self.assertEqual(metrics["Nomenclature"]["tp"], 0)
        self.assertEqual(metrics["Nomenclature"]["fn"], 1)
        self.assertEqual(metrics["Nomenclature"]["recall"], 0.0)


class TestComputeTokenIoU(unittest.TestCase):
    """Tests for compute_token_iou."""

    def test_identical_blocks(self):
        blocks = [("abc", "Nomenclature"), ("def", "Description")]
        iou = compute_token_iou(blocks, blocks)
        self.assertEqual(iou["Nomenclature"], 1.0)
        self.assertEqual(iou["Description"], 1.0)
        self.assertEqual(iou["micro_avg"], 1.0)

    def test_no_overlap(self):
        pred = [("abc", "Nomenclature")]
        gold = [("xyz", "Description")]
        iou = compute_token_iou(pred, gold)
        self.assertEqual(iou["Nomenclature"], 0.0)
        self.assertEqual(iou["Description"], 0.0)
        self.assertEqual(iou["micro_avg"], 0.0)

    def test_empty_blocks(self):
        iou = compute_token_iou([], [])
        self.assertEqual(iou["micro_avg"], 0.0)


class TestComputeConfusionMatrix(unittest.TestCase):
    """Tests for compute_confusion_matrix."""

    def test_basic_confusion(self):
        aligned = [
            ("Nomenclature", "Nomenclature", "a"),
            ("Description", "Nomenclature", "b"),
            ("Description", "Description", "c"),
        ]
        matrix = compute_confusion_matrix(aligned)
        # Golden Nomenclature -> Predicted Nomenclature: 1
        # Golden Nomenclature -> Predicted Description: 1
        self.assertEqual(matrix["Nomenclature"]["Nomenclature"], 1)
        self.assertEqual(matrix["Nomenclature"]["Description"], 1)
        # Golden Description -> Predicted Description: 1
        self.assertEqual(matrix["Description"]["Description"], 1)

    def test_none_tags_use_placeholder(self):
        aligned = [
            (None, "Nomenclature", "a"),
            ("Description", None, "b"),
        ]
        matrix = compute_confusion_matrix(aligned)
        self.assertEqual(matrix["Nomenclature"]["(none)"], 1)
        self.assertEqual(matrix["(none)"]["Description"], 1)


class TestFormatReport(unittest.TestCase):
    """Tests for format_report."""

    def test_generates_markdown(self):
        evaluation = {
            "documents_matched": 10,
            "documents_skipped": 2,
            "macro_f1": 0.85,
            "tag_metrics": {
                "Nomenclature": {
                    "precision": 0.9, "recall": 0.8, "f1": 0.85,
                    "tp": 9, "fp": 1, "fn": 2,
                },
                "macro_avg": {
                    "precision": 0.9, "recall": 0.8, "f1": 0.85,
                    "tp": 9, "fp": 1, "fn": 2,
                },
            },
            "token_iou": {"Nomenclature": 0.75, "micro_avg": 0.75},
            "confusion_matrix": {},
        }
        report = format_report(evaluation, "pred_db", "gold_db", 1)
        self.assertIn("# Evaluation", report)
        self.assertIn("Macro F1: 0.8500", report)
        self.assertIn("Nomenclature", report)
        self.assertIn("| Tag |", report)


if __name__ == "__main__":
    unittest.main()
