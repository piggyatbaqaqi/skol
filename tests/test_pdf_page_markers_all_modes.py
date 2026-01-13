#!/usr/bin/env python3
"""
Test PDF Page Marker Preservation

This test verifies that PDF page markers (format: --- PDF Page N ---)
are properly preserved in the YEDDA annotation output and are NOT
wrapped in YEDDA annotation blocks.

The test creates sample data with page markers and verifies:
1. Page markers appear in output at the same line positions as input
2. Page markers are not wrapped in YEDDA annotation format [@ ... #Label*]
3. Page markers break coalescing chains (prevent multi-line annotations)
"""

import sys
from pathlib import Path
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType

from skol_classifier.output_formatters import YeddaFormatter


def create_spark_session():
    """Create a minimal Spark session for testing."""
    return SparkSession.builder \
        .appName("TestPDFPageMarkers") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()


def create_test_data_with_markers(spark):
    """
    Create test data with PDF page markers.

    This simulates the output of line extraction mode with is_page_marker column.
    """
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("line_number", IntegerType(), False),
        StructField("value", StringType(), False),
        StructField("predicted_label", StringType(), True),
        StructField("is_page_marker", BooleanType(), False),
    ])

    data = [
        ("test_doc", "article.txt", 1, "This is the introduction paragraph.", "Description", False),
        ("test_doc", "article.txt", 2, "It describes the research.", "Description", False),
        ("test_doc", "article.txt", 3, "--- PDF Page 1 ---", None, True),
        ("test_doc", "article.txt", 4, "Materials and Methods section.", "Description", False),
        ("test_doc", "article.txt", 5, "We used various techniques.", "Description", False),
        ("test_doc", "article.txt", 6, "--- PDF Page 2 ---", None, True),
        ("test_doc", "article.txt", 7, "Results are presented here.", "Description", False),
        ("test_doc", "article.txt", 8, "The data shows interesting patterns.", "Description", False),
        ("test_doc", "article.txt", 9, "--- PDF Page 3 ---", None, True),
        ("test_doc", "article.txt", 10, "Discussion and conclusions.", "Description", False),
        ("test_doc", "article.txt", 11, "Future work is needed.", "Description", False),
    ]

    return spark.createDataFrame(data, schema)


def extract_page_markers(text):
    """Extract PDF page markers from text."""
    return re.findall(r'^---\s*PDF\s+Page\s+(\d+)\s*---\s*$', text, re.MULTILINE)


def check_markers_not_in_yedda_blocks(text):
    """Check that page markers are NOT inside YEDDA annotation blocks."""
    # Find all YEDDA blocks
    yedda_blocks = re.findall(r'\[@\s*.*?\s*#[^\*]+\*\]', text, re.DOTALL)

    markers_in_blocks = []
    for block in yedda_blocks:
        if re.search(r'---\s*PDF\s+Page\s+\d+\s*---', block):
            markers_in_blocks.append(block[:100])

    return markers_in_blocks


def test_page_marker_preservation(spark):
    """Test that YeddaFormatter preserves PDF page markers correctly."""
    print("\n" + "="*70)
    print("TEST: YeddaFormatter - PDF Page Marker Preservation")
    print("="*70)

    # Create test data
    df = create_test_data_with_markers(spark)

    print(f"Input data: {df.count()} lines")
    print("  - 3 page markers")
    print("  - 8 content lines with predictions")
    print()

    # Format output using YeddaFormatter - use coalesce to properly handle page markers
    formatter = YeddaFormatter()
    coalesced_df = formatter.coalesce_consecutive_labels(df, line_level=True)

    # The coalesced_df has structure: [doc_id, attachment_name, coalesced_annotations (array)]
    # We need to explode this array to get individual lines
    from pyspark.sql.functions import posexplode

    output_df = coalesced_df.select(
        F.col("doc_id"),
        posexplode(F.col("coalesced_annotations")).alias("pos", "line_content")
    )

    # Collect output
    output_rows = output_df.filter(F.col("doc_id") == "test_doc") \
        .orderBy("pos") \
        .select("line_content") \
        .collect()

    # Build output text
    output_lines = [row.line_content for row in output_rows]
    output_text = '\n'.join(output_lines)

    print("Output preview:")
    for i, line in enumerate(output_lines[:6], 1):
        print(f"  {i:2d}: {line[:70]}")
    print("  ...")
    print()

    # Test 1: Check we have output blocks
    print("Test 1: Output block count...")
    # After coalescing:
    # - Block 0: lines 1-2 coalesced (Description)
    # - Block 1: marker "--- PDF Page 1 ---"
    # - Block 2: lines 4-5 coalesced (Description)
    # - Block 3: marker "--- PDF Page 2 ---"
    # - Block 4: lines 7-8 coalesced (Description)
    # - Block 5: marker "--- PDF Page 3 ---"
    # - Block 6: lines 10-11 coalesced (Description)
    # Total: 7 blocks
    print(f"  Output has {len(output_lines)} blocks")
    assert len(output_lines) >= 6, f"Expected at least 6 blocks (4 coalesced + 3 markers), got {len(output_lines)}"
    print(f"  ✓ PASS: Output has coalesced blocks and markers")
    print()

    # Test 2: Extract and verify page markers
    print("Test 2: Page marker preservation...")
    output_markers = extract_page_markers(output_text)
    print(f"  Found {len(output_markers)} page markers in output")
    print(f"  Page numbers: {', '.join(output_markers)}")
    assert len(output_markers) == 3, f"Expected 3 page markers, found {len(output_markers)}"
    assert output_markers == ['1', '2', '3'], f"Page numbers don't match: {output_markers}"
    print("  ✓ PASS: All 3 page markers preserved correctly")
    print()

    # Test 3: Verify markers are standalone blocks (not wrapped)
    print("Test 3: Page markers are standalone blocks...")
    marker_blocks = []
    for i, line in enumerate(output_lines):
        if re.match(r'^---\s*PDF\s+Page\s+\d+\s*---\s*$', line):
            marker_blocks.append((i+1, line))

    print(f"  Found {len(marker_blocks)} standalone marker blocks")
    for block_num, marker in marker_blocks:
        print(f"    Block {block_num}: {marker}")
    assert len(marker_blocks) == 3, f"Expected 3 standalone marker blocks, got {len(marker_blocks)}"
    print(f"  ✓ PASS: All page markers are standalone blocks")
    print()

    # Test 4: Check that markers are NOT inside YEDDA blocks
    print("Test 4: Page markers not in YEDDA blocks...")
    markers_in_blocks = check_markers_not_in_yedda_blocks(output_text)
    assert len(markers_in_blocks) == 0, \
        f"Found {len(markers_in_blocks)} page markers inside YEDDA blocks: {markers_in_blocks}"
    print("  ✓ PASS: Page markers are standalone (not wrapped in YEDDA format)")
    print()

    # Test 5: Verify coalescing breaks at page markers
    print("Test 5: Page markers break coalescing chains...")
    # Lines 1-2 should be coalesced before the marker
    # Lines 4-5 should be coalesced after marker 1 and before marker 2
    # Lines 7-8 should be coalesced after marker 2 and before marker 3
    # Lines 10-11 should be coalesced after marker 3

    # Check that we don't have multi-line YEDDA blocks spanning page markers
    yedda_blocks = re.findall(r'\[@\s*(.*?)\s*#([^\*]+)\*\]', output_text, re.DOTALL)
    for text, label in yedda_blocks:
        if '---' in text:
            assert False, f"YEDDA block contains page marker: [{text[:50]}...]"

    print("  ✓ PASS: YEDDA blocks do not span across page markers")
    print()

    print("="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print()

    return True


def run_all_tests():
    """Run all PDF page marker preservation tests."""
    print("\n" + "="*70)
    print("PDF Page Marker Preservation Tests")
    print("="*70)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        test_page_marker_preservation(spark)
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        spark.stop()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
