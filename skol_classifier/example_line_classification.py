#!/usr/bin/env python3
"""
Example script demonstrating line-by-line classification with YEDDA output.

This script shows how to:
1. Load a trained classifier
2. Classify text files line-by-line (instead of paragraph-by-paragraph)
3. Coalesce consecutive lines with the same label into YEDDA blocks
4. Save the output in YEDDA format
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2


def main():
    """Run line-by-line classification example."""

    print("="*70)
    print("SKOL Line-by-Line Classifier with YEDDA Output (V2 API)")
    print("="*70)

    # Create Spark session
    print("\nInitializing Spark session...")
    spark = SparkSession.builder \
        .appName("SKOL Line Classifier Example") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    try:
        # Get annotated training files
        data_dir = Path(__file__).parent.parent / "data" / "annotated"

        if not data_dir.exists():
            print(f"\n⚠ Data directory not found: {data_dir}")
            print("This example demonstrates the V2 API for line-level classification.")
            print("\nTo use SkolClassifierV2 with line-level classification:")
            print()
            print("  # Train a model")
            print("  classifier = SkolClassifierV2(")
            print("      spark=spark,")
            print("      input_source='files',")
            print("      file_paths=['data/*.ann'],")
            print("      line_level=True,           # Enable line-level processing")
            print("      coalesce_labels=True,      # Coalesce consecutive labels")
            print("      model_type='logistic'")
            print("  )")
            print("  results = classifier.fit()")
            print()
            print("  # Make predictions")
            print("  predictions = classifier.predict()")
            print()
            print("  # Save with coalescing")
            print("  classifier.save_annotated(predictions)")
            print()
            print("\nKey Features:")
            print("  • line_level=True: Process text line-by-line")
            print("  • coalesce_labels=True: Merge consecutive lines with same label")
            print("  • Output in YEDDA format: [@ text #Label*]")
            print("  • Supports files and CouchDB")

        else:
            # Find annotated files
            annotated_files = list(data_dir.glob("**/*.ann"))

            if len(annotated_files) == 0:
                print(f"\n⚠ No annotated files found in {data_dir}")
            else:
                print(f"\nFound {len(annotated_files)} annotated files")

                # Initialize and train classifier
                print("\nInitializing SkolClassifierV2...")
                classifier = SkolClassifierV2(
                    spark=spark,
                    input_source='files',
                    file_paths=[str(f) for f in annotated_files],
                    line_level=True,
                    use_suffixes=False,
                    model_type='logistic',
                    output_format='annotated',
                    coalesce_labels=True
                )

                # Train
                print("Training classifier...")
                results = classifier.fit()

                print(f"\n✓ Training complete!")
                print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
                print(f"  F1 Score: {results.get('f1_score', 0):.4f}")

                # Make predictions on first file
                print("\nMaking predictions...")
                predictions = classifier.predict()

                # Show sample predictions
                print("\nSample predictions (line-level):")
                predictions.select(
                    "value", "predicted_label", "annotated_value"
                ).show(10, truncate=50)

                print("\nLabel distribution:")
                predictions.groupBy("predicted_label").count().orderBy("count", ascending=False).show()

                print("\nNote: When saving with coalesce_labels=True,")
                print("consecutive lines with the same label are merged into blocks.")

        print("\n" + "="*70)
        print("Example complete!")
        print("="*70)

    finally:
        print("\nStopping Spark session...")
        spark.stop()
        print("Done!")


if __name__ == "__main__":
    main()
