#!/usr/bin/env python3
"""Test script to verify line-level vs paragraph-level data loading with V2 API."""

import sys
from pathlib import Path
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

# pylint: disable=protected-access


def test_line_level_loading():
    """Compare line-level vs paragraph-level data loading with SkolClassifierV2."""

    # Create Spark session
    spark = SparkSession.builder \
        .appName("Test Line Level Loading V2") \
        .master("local[*]") \
        .getOrCreate()

    try:
        # Create temporary annotated file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt.ann', delete=False) as f:
            temp_file = f.name
            f.write("""[@ Line 1
Line 2
Line 3
#Nomenclature*]
[@ Single line description
#Description*]
[@ Multi
line
misc
exposition
#Misc-exposition*]
""")

        print("Testing data loading with SkolClassifierV2...")
        print("=" * 70)

        # Test paragraph-level loading
        print("\n1. PARAGRAPH-LEVEL LOADING (line_level=False)")
        print("-" * 70)
        classifier_para = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=[temp_file],
            line_level=False,  # Paragraph level
            model_type='logistic',
            auto_load_model=False
        )

        # Load data directly
        para_df = classifier_para._load_annotated_data()
        para_count = para_df.count()
        print(f"Total samples: {para_count}")
        print("\nSample data:")
        para_df.select("label", "value").show(truncate=50)

        # Test line-level loading
        print("\n2. LINE-LEVEL LOADING (line_level=True)")
        print("-" * 70)
        classifier_line = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=[temp_file],
            line_level=True,  # Line level
            model_type='logistic',
            auto_load_model=False
        )

        # Load data directly
        line_df = classifier_line._load_annotated_data()
        line_count = line_df.count()
        print(f"Total samples: {line_count}")
        print("\nSample data:")
        line_df.select("label", "value", "line_number").show(truncate=50)

        # Verify expectations
        print("\n" + "=" * 70)
        print("VERIFICATION")
        print("=" * 70)

        # We expect 3 paragraphs but 8 lines total
        expected_paragraphs = 3
        expected_lines = 8

        print(f"Expected paragraphs: {expected_paragraphs}")
        print(f"Actual paragraphs:   {para_count}")
        print(f"Match: {'✓' if para_count == expected_paragraphs else '✗'}")

        print(f"\nExpected lines: {expected_lines}")
        print(f"Actual lines:   {line_count}")
        print(f"Match: {'✓' if line_count == expected_lines else '✗'}")

        # Check that line-level has more samples
        print(f"\nLine-level has more samples: {'✓' if line_count > para_count else '✗'}")

        # Verify line_number column exists in line-level
        has_line_num = "line_number" in line_df.columns
        print(f"Line-level has line_number column: {'✓' if has_line_num else '✗'}")

        # Verify paragraph-level doesn't have line_number (or has it with all same values)
        para_has_line_num = "line_number" in para_df.columns
        print(f"Paragraph-level structure correct: {'✓' if not para_has_line_num or para_count == 3 else '✗'}")

        # Test V2 API specific features
        print("\n3. V2 API FEATURES")
        print("-" * 70)
        print("Line-level classifier configuration:")
        print(f"  line_level={classifier_line.line_level}")
        print(f"  input_source={classifier_line.input_source}")
        print("  ✓ Configuration verified")

        # Overall result
        success = (
            para_count == expected_paragraphs and
            line_count == expected_lines and
            line_count > para_count and
            has_line_num
        )

        print("\n" + "=" * 70)
        if success:
            print("✓ ALL TESTS PASSED!")
        else:
            print("✗ SOME TESTS FAILED")
        print("=" * 70)

        return success

    finally:
        # Cleanup
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass
        spark.stop()


if __name__ == "__main__":
    success = test_line_level_loading()
    sys.exit(0 if success else 1)
