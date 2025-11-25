#!/usr/bin/env python3
"""
Example script demonstrating line-by-line classification with YEDA output.

This script shows how to:
1. Load a trained classifier
2. Classify text files line-by-line (instead of paragraph-by-paragraph)
3. Coalesce consecutive lines with the same label into YEDA blocks
4. Save the output in YEDA format
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from skol_classifier.classifier import SkolClassifier


def main():
    """Run line-by-line classification example."""

    print("="*70)
    print("SKOL Line-by-Line Classifier with YEDA Output")
    print("="*70)

    # Create Spark session
    print("\nInitializing Spark session...")
    spark = SparkSession.builder \
        .appName("SKOL Line Classifier Example") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    try:
        # Initialize classifier
        print("Initializing classifier...")
        classifier = SkolClassifier(spark=spark)

        # Check if model is loaded
        if classifier.pipeline_model is None or classifier.classifier_model is None:
            print("\n⚠ No trained model found.")
            print("You need to train a model first using:")
            print("  1. Load training data with load_annotated_data()")
            print("  2. Build and fit feature pipeline with fit_features()")
            print("  3. Train classifier with train_classifier()")
            print("\nFor this example, we'll just demonstrate the YEDA formatting.")
            print("\nShowing example YEDA coalescence:")

            # Demonstrate YEDA formatting
            example_lines = [
                {'line': 'Glomus mosseae Nicolson & Gerdemann, 1963.', 'label': 'Nomenclature'},
                {'line': '≡ Glomus mosseae (Nicolson & Gerdemann) C. Walker & A. Schüssler', 'label': 'Nomenclature'},
                {'line': '', 'label': 'Misc-exposition'},
                {'line': 'Key characters: Spores formed singly or in loose clusters.', 'label': 'Description'},
                {'line': 'Spore wall structure: mono- to multiple-layered.', 'label': 'Description'},
                {'line': '', 'label': 'Misc-exposition'},
                {'line': 'This species is commonly found in temperate regions.', 'label': 'Misc-exposition'},
                {'line': 'It forms arbuscular mycorrhizal associations.', 'label': 'Misc-exposition'},
            ]

            yeda_output = classifier.coalesce_consecutive_labels(example_lines)

            print("\nInput lines with labels:")
            for i, item in enumerate(example_lines, 1):
                print(f"  {i}. [{item['label']}] {item['line'][:60]}...")

            print("\nCoalesced YEDA output:")
            print("-" * 70)
            print(yeda_output)
            print("-" * 70)

        else:
            print("\n✓ Model loaded successfully!")
            print(f"  Labels: {classifier.labels}")

            # Example: Process files line-by-line
            print("\nTo process files line-by-line:")
            print("  predictions = classifier.predict_lines(['path/to/file.txt'])")
            print("  classifier.save_yeda_output(predictions, 'output_dir')")

            print("\nKey differences from paragraph-based classification:")
            print("  • Uses predict_lines() instead of predict_raw_text()")
            print("  • Each line is classified independently")
            print("  • Consecutive lines with same label are coalesced into blocks")
            print("  • Output is in YEDA format with proper block structure")

        print("\n" + "="*70)
        print("Example complete!")
        print("="*70)

    finally:
        print("\nStopping Spark session...")
        spark.stop()
        print("Done!")


if __name__ == "__main__":
    main()
