#!/usr/bin/env python3
"""
Test text attachment support in PDFSectionExtractor.

Tests that .txt files with form feed characters are properly processed.
"""

from pyspark.sql import SparkSession
from pdf_section_extractor import PDFSectionExtractor


def test_txt_to_text_with_pages():
    """Test form feed replacement in text files."""

    # Create test text with form feeds
    test_data = "Page 1 content\nLine 2 of page 1\f" \
                "Page 2 content\nLine 2 of page 2\f" \
                "Page 3 content\nLine 2 of page 3"

    test_bytes = test_data.encode('utf-8')

    # Initialize extractor
    spark = SparkSession.builder.appName("TextTest").getOrCreate()
    extractor = PDFSectionExtractor(spark=spark, verbosity=2)

    # Process text
    result = extractor.txt_to_text_with_pages(test_bytes)

    print("\n=== Form Feed Replacement Test ===\n")
    print("Input (with \\f for form feeds):")
    print(repr(test_data))
    print("\nOutput:")
    print(result)

    # Verify page markers were added
    assert "--- PDF Page 1 Label i ---" in result
    assert "--- PDF Page 2 Label ii ---" in result
    assert "--- PDF Page 3 Label iii ---" in result

    # Verify content is preserved
    assert "Page 1 content" in result
    assert "Page 2 content" in result
    assert "Page 3 content" in result

    # Verify no form feeds remain
    assert '\f' not in result

    print("\n✓ Form feed replacement works correctly")
    print("✓ Page markers added")
    print("✓ Content preserved")

    spark.stop()
    return True


def test_txt_parsing_to_dataframe():
    """Test parsing text file into DataFrame with sections."""

    # Create test text with YEDDA annotations and form feeds
    test_text = """--- PDF Page 1 Label i ---

[@ Introduction text here.
This is the introduction section.
#Introduction*]

[@ Nomenclature section
#Nomenclature*]

--- PDF Page 2 Label ii ---

[@ Methods section here.
#Methods*]

Conclusion text without annotation."""

    # Initialize extractor
    spark = SparkSession.builder.appName("TextDFTest").getOrCreate()
    extractor = PDFSectionExtractor(spark=spark, verbosity=2)

    # Parse into DataFrame
    df = extractor.parse_text_to_sections(
        text=test_text,
        doc_id='test_txt_doc',
        attachment_name='test.txt',
        min_paragraph_length=5
    )

    print("\n=== Text File DataFrame Parsing Test ===\n")
    df.select("value", "page_number", "label").show(truncate=False)

    # Verify sections were created
    assert df.count() >= 3, "Should have at least 3 sections"

    # Verify page numbers are correct
    rows = df.collect()
    page_numbers = [row['page_number'] for row in rows]
    assert 1 in page_numbers, "Should have sections from page 1"
    assert 2 in page_numbers, "Should have sections from page 2"

    # Verify YEDDA labels work
    labels = [row['label'] for row in rows if row['label'] is not None]
    assert 'Introduction' in labels, "Should have Introduction label"
    assert 'Nomenclature' in labels, "Should have Nomenclature label"
    assert 'Methods' in labels, "Should have Methods label"

    print(f"\n✓ Created {df.count()} sections")
    print(f"✓ Found labels: {set(labels)}")
    print("✓ Page numbers correctly tracked")

    spark.stop()
    return True


def test_complete_txt_workflow():
    """Test complete workflow: form feeds -> page markers -> DataFrame."""

    # Create test text with form feeds and YEDDA annotations
    test_with_form_feeds = (
        "[@ Introduction section\n"
        "#Introduction*]\n"
        "\n"
        "Some regular text.\n"
        "\f"  # Form feed - should become page 2
        "[@ Methods section\n"
        "#Methods*]\n"
        "\n"
        "More text.\n"
        "\f"  # Form feed - should become page 3
        "Final page content."
    )

    test_bytes = test_with_form_feeds.encode('utf-8')

    # Initialize extractor
    spark = SparkSession.builder.appName("CompleteTest").getOrCreate()
    extractor = PDFSectionExtractor(spark=spark, verbosity=2)

    print("\n=== Complete Workflow Test ===\n")

    # Step 1: Replace form feeds with page markers
    text_with_markers = extractor.txt_to_text_with_pages(test_bytes)

    print("After form feed replacement:")
    print(text_with_markers[:200] + "...")

    # Verify page markers
    assert "--- PDF Page 1 Label i ---" in text_with_markers
    assert "--- PDF Page 2 Label ii ---" in text_with_markers
    assert "--- PDF Page 3 Label iii ---" in text_with_markers

    print("\n✓ Form feeds replaced with page markers")

    # Step 2: Parse into DataFrame
    df = extractor.parse_text_to_sections(
        text=text_with_markers,
        doc_id='test_doc',
        attachment_name='test.txt',
        min_paragraph_length=5
    )

    print("\nDataFrame created:")
    df.select("value", "page_number", "label").show(truncate=False)

    # Verify results
    rows = df.collect()

    # Check page distribution
    page_counts = {}
    for row in rows:
        page = row['page_number']
        page_counts[page] = page_counts.get(page, 0) + 1

    print(f"\n✓ Sections by page: {page_counts}")
    assert 1 in page_counts, "Should have content on page 1"
    assert 2 in page_counts, "Should have content on page 2"
    assert 3 in page_counts, "Should have content on page 3"

    # Check labels
    labeled_sections = [row for row in rows if row['label'] is not None]
    print(f"✓ Found {len(labeled_sections)} labeled sections")

    labels = {row['label'] for row in labeled_sections}
    assert 'Introduction' in labels, "Should have Introduction label"
    assert 'Methods' in labels, "Should have Methods label"

    print(f"✓ Labels found: {labels}")

    spark.stop()
    return True


if __name__ == "__main__":
    print("Testing text attachment support in PDFSectionExtractor\n")
    print("=" * 70)

    try:
        # Test 1: Form feed replacement
        test_txt_to_text_with_pages()

        print("\n" + "=" * 70)

        # Test 2: DataFrame parsing
        test_txt_parsing_to_dataframe()

        print("\n" + "=" * 70)

        # Test 3: Complete workflow
        test_complete_txt_workflow()

        print("\n" + "=" * 70)
        print("\n✅ All tests passed successfully!\n")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
