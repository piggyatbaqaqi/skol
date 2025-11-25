"""Test script for line-by-line classification with YEDA output."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from skol_classifier.classifier import SkolClassifier


def test_line_classification():
    """Test line-by-line classification and YEDA output."""

    # Create Spark session
    spark = SparkSession.builder \
        .appName("Test Line Classifier") \
        .master("local[*]") \
        .getOrCreate()

    try:
        # Initialize classifier
        classifier = SkolClassifier(spark=spark, auto_load=False)

        # Test the coalesce function
        print("Testing coalesce_consecutive_labels()...")

        test_data = [
            {'line': 'Line 1', 'label': 'Nomenclature'},
            {'line': 'Line 2', 'label': 'Nomenclature'},
            {'line': 'Line 3', 'label': 'Description'},
            {'line': 'Line 4', 'label': 'Description'},
            {'line': 'Line 5', 'label': 'Misc-exposition'},
        ]

        result = classifier.coalesce_consecutive_labels(test_data)
        print("\nInput:")
        for item in test_data:
            print(f"  {item['label']}: {item['line']}")

        print("\nYEDA Output:")
        print(result)

        # Verify the output
        expected_blocks = 3  # Should create 3 YEDA blocks
        actual_blocks = result.count('[@ ')

        assert actual_blocks == expected_blocks, \
            f"Expected {expected_blocks} blocks, got {actual_blocks}"

        print("\n✓ Test passed!")

        # Test with empty data
        print("\nTesting with empty data...")
        empty_result = classifier.coalesce_consecutive_labels([])
        assert empty_result == "", "Empty data should return empty string"
        print("✓ Empty data test passed!")

        # Test with single line
        print("\nTesting with single line...")
        single_result = classifier.coalesce_consecutive_labels([
            {'line': 'Single line', 'label': 'Nomenclature'}
        ])
        assert '[@ Single line\n#Nomenclature*]' in single_result
        print("✓ Single line test passed!")

        print("\n" + "="*50)
        print("All tests passed!")
        print("="*50)

    finally:
        spark.stop()


if __name__ == "__main__":
    test_line_classification()
