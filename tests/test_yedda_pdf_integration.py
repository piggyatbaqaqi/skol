#!/usr/bin/env python3
"""
Test YEDDA annotation integration in PDFSectionExtractor.

This script tests the YEDDA annotation parsing functionality that was added
to pdf_section_extractor.py.
"""

from pyspark.sql import SparkSession
from pdf_section_extractor import PDFSectionExtractor


def test_yedda_parsing():
    """Test YEDDA annotation parsing with nested annotations."""

    # Create test text with YEDDA annotations
    test_text = """[@ This is the introduction section.
It has multiple lines.
#Introduction*]

[@ This is a nomenclature section.
[@ This is nested description text.
It has details.
#Description*]
Back to nomenclature.
#Nomenclature*]

This line has no annotation.

[@ Final section text.
#Conclusion*]"""

    # Initialize extractor
    spark = SparkSession.builder.appName("YEDDATest").getOrCreate()
    extractor = PDFSectionExtractor(spark=spark, verbosity=2)

    # Test the annotation parser
    line_to_label = extractor._parse_yedda_annotations(test_text)

    print("\n=== YEDDA Annotation Parsing Test ===\n")

    # Display results
    lines = test_text.split('\n')
    for i, line in enumerate(lines, 1):
        label = line_to_label.get(i, None)
        print(f"Line {i:2d}: [{label or 'None':15s}] {line[:60]}")

    # Verify nesting: innermost label should win
    print("\n=== Verification ===")

    # Lines 1-2 should be Introduction
    assert line_to_label.get(1) == 'Introduction', f"Line 1 should be Introduction, got {line_to_label.get(1)}"
    assert line_to_label.get(2) == 'Introduction', f"Line 2 should be Introduction, got {line_to_label.get(2)}"

    # Lines 5-6 should be Nomenclature (outer annotation)
    assert line_to_label.get(5) == 'Nomenclature', f"Line 5 should be Nomenclature, got {line_to_label.get(5)}"

    # Lines 6-8 should be Description (inner annotation wins!)
    assert line_to_label.get(6) == 'Description', f"Line 6 should be Description, got {line_to_label.get(6)}"
    assert line_to_label.get(7) == 'Description', f"Line 7 should be Description, got {line_to_label.get(7)}"

    # Line 9 should be back to Nomenclature
    assert line_to_label.get(9) == 'Nomenclature', f"Line 9 should be Nomenclature, got {line_to_label.get(9)}"

    # Line 12 should have no annotation
    assert line_to_label.get(12) is None, f"Line 12 should be None, got {line_to_label.get(12)}"

    # Lines 14-15 should be Conclusion
    assert line_to_label.get(14) == 'Conclusion', f"Line 14 should be Conclusion, got {line_to_label.get(14)}"

    print("\n✓ All assertions passed!")
    print("✓ Nested annotations work correctly - innermost label wins")

    spark.stop()
    return True


def test_yedda_in_dataframe():
    """Test that YEDDA labels appear in the DataFrame output."""

    # Create test text with YEDDA annotations and proper paragraph breaks
    # Note: Using correct page marker format
    test_text = """--- PDF Page 1 Label i ---

[@ This is the introduction section.
#Introduction*]

[@ This is the nomenclature section.
#Nomenclature*]

This is conclusion text without annotation."""

    # Initialize extractor
    spark = SparkSession.builder.appName("YEDDADataFrameTest").getOrCreate()
    extractor = PDFSectionExtractor(spark=spark, verbosity=2)

    # Parse into DataFrame
    df = extractor.parse_text_to_sections(
        text=test_text,
        doc_id='test_doc',
        attachment_name='test.pdf',
        min_paragraph_length=5
    )

    print("\n=== DataFrame with YEDDA Labels ===\n")
    df.select("value", "line_number", "label").show(truncate=False)

    # Verify labels are in DataFrame
    rows = df.select("value", "label").collect()

    print("\n=== Verification ===")
    for row in rows:
        print(f"Text: {row['value'][:50]:50s} | Label: {row['label'] or 'None'}")

    # Check that we have at least some labels
    labels_found = [row['label'] for row in rows if row['label'] is not None]
    print(f"\n✓ Found {len(labels_found)} sections with YEDDA labels")
    print(f"✓ Labels: {set(labels_found)}")

    assert len(labels_found) > 0, "Should have at least one YEDDA label in DataFrame"
    assert 'Introduction' in labels_found, "Should have Introduction label"
    assert 'Nomenclature' in labels_found, "Should have Nomenclature label"

    print("\n✓ All DataFrame tests passed!")

    spark.stop()
    return True


if __name__ == "__main__":
    print("Testing YEDDA annotation support in PDFSectionExtractor\n")
    print("=" * 70)

    try:
        # Test 1: Basic parsing
        test_yedda_parsing()

        print("\n" + "=" * 70)

        # Test 2: DataFrame integration
        test_yedda_in_dataframe()

        print("\n" + "=" * 70)
        print("\n✅ All tests passed successfully!\n")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
