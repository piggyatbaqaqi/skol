#!/usr/bin/env python3
"""
Test that AnnotatedTextParser correctly adds section_name column.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from skol_classifier.preprocessing import AnnotatedTextParser


def main():
    """Test section name detection in annotated parser."""
    print("=" * 70)
    print("Testing AnnotatedTextParser Section Name Detection")
    print("=" * 70)

    # Create Spark session
    spark = SparkSession.builder \
        .appName("TestParserSectionNames") \
        .master("local[2]") \
        .getOrCreate()

    # Create test data with YEDDA annotations containing section headers
    test_data = [
        {
            'doc_id': 'test_doc_1',
            'human_url': 'http://example.com/doc1',
            'attachment_name': 'article.txt.ann',
            'value': '''[@ Introduction
This is the introduction section.
More introduction text.
#Misc-exposition*]

[@ Methods section
This describes the methods used.
#Misc-exposition*]

[@ Agaricus campestris (L.) Fr. 1821
This is a nomenclature entry.
#Nomenclature*]

[@ Description: Cap 5-10 cm wide.
Detailed description here.
#Description*]'''
        }
    ]

    # Create DataFrame
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("human_url", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("value", StringType(), False)
    ])

    df = spark.createDataFrame(test_data, schema)

    print("\n✓ Created test DataFrame")

    # Parse with AnnotatedTextParser
    parser = AnnotatedTextParser(extraction_mode='paragraph', collapse_labels=True)
    result_df = parser.parse(df)

    print(f"\n✓ Parsed {result_df.count()} paragraphs")

    # Check that section_name column exists
    assert 'section_name' in result_df.columns, "section_name column missing!"
    print("✓ section_name column exists")

    # Show results
    print("\nParsed paragraphs with section names:")
    result_df.select("label", "value", "section_name").show(truncate=50)

    # Check specific section names
    rows = result_df.collect()

    section_names_found = [row['section_name'] for row in rows if row['section_name'] is not None]

    print(f"\n✓ Found section names: {section_names_found}")

    # Verify expected sections
    if 'Introduction' in section_names_found:
        print("✓ Detected 'Introduction' section")
    else:
        print("⚠ Did not detect 'Introduction' section")

    if 'Methods' in section_names_found:
        print("✓ Detected 'Methods' section")
    else:
        print("⚠ Did not detect 'Methods' section")

    if 'Description' in section_names_found:
        print("✓ Detected 'Description' section")
    else:
        print("⚠ Did not detect 'Description' section")

    print("\n" + "=" * 70)
    print("✅ Section name detection test complete!")
    print("=" * 70)

    spark.stop()


if __name__ == '__main__':
    main()
