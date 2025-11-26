"""Test script for line-by-line classification with YEDA output using V2 API."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2


def test_line_classification():
    """Test line-by-line classification with SkolClassifierV2."""

    # Create Spark session
    spark = SparkSession.builder \
        .appName("Test Line Classifier V2") \
        .master("local[*]") \
        .getOrCreate()

    try:
        print("Testing SkolClassifierV2 with line-level processing...")

        # Get test data
        data_dir = Path(__file__).parent.parent / "data" / "annotated"

        if not data_dir.exists():
            print(f"⚠ Test data directory not found: {data_dir}")
            print("Creating minimal test case...")

            # Create minimal test with mock data
            print("\nTest 1: Verify V2 API initialization")
            classifier = SkolClassifierV2(
                spark=spark,
                input_source='files',
                file_paths=[],  # Empty for now
                line_level=True,
                coalesce_labels=True,
                model_type='logistic'
            )
            print("✓ SkolClassifierV2 initialized successfully")

            print("\nTest 2: Verify configuration")
            assert classifier.line_level == True, "line_level should be True"
            assert classifier.coalesce_labels == True, "coalesce_labels should be True"
            print("✓ Configuration verified")

            print("\n⚠ Skipping functional tests (no training data)")
            print("To run full tests, add annotated files to:", data_dir)

        else:
            # Find annotated files
            annotated_files = list(data_dir.glob("**/*.ann"))

            if len(annotated_files) == 0:
                print(f"⚠ No annotated files found in {data_dir}")
            else:
                print(f"Found {len(annotated_files)} annotated files")

                # Test 1: Train with line-level processing
                print("\nTest 1: Training with line-level processing...")
                classifier = SkolClassifierV2(
                    spark=spark,
                    input_source='files',
                    file_paths=[str(f) for f in annotated_files[:2]],  # Use first 2 files
                    line_level=True,
                    use_suffixes=False,
                    model_type='logistic',
                    output_format='annotated'
                )

                results = classifier.fit()
                print(f"✓ Training complete: Accuracy={results.get('accuracy', 0):.4f}")

                # Test 2: Make predictions
                print("\nTest 2: Making predictions...")
                predictions = classifier.predict()

                # Verify structure
                assert 'value' in predictions.columns, "Missing 'value' column"
                assert 'predicted_label' in predictions.columns, "Missing 'predicted_label' column"
                assert 'annotated_value' in predictions.columns, "Missing 'annotated_value' column"
                assert 'line_number' in predictions.columns, "Missing 'line_number' column"

                count = predictions.count()
                print(f"✓ Predictions generated: {count} lines")

                # Test 3: Verify YEDA formatting
                print("\nTest 3: Verifying YEDA format...")
                sample = predictions.select("annotated_value").first()
                if sample:
                    annotated = sample['annotated_value']
                    assert '[@ ' in annotated, "Missing YEDA opening"
                    assert '*]' in annotated, "Missing YEDA closing"
                    assert '#' in annotated, "Missing label marker"
                    print(f"✓ YEDA format verified: {annotated[:50]}...")

                # Test 4: Verify label distribution
                print("\nTest 4: Checking label distribution...")
                label_counts = predictions.groupBy("predicted_label").count()
                num_labels = label_counts.count()
                print(f"✓ Found {num_labels} distinct labels")

                # Test 5: Test with coalescing disabled
                print("\nTest 5: Testing without coalescing...")
                classifier_no_coalesce = SkolClassifierV2(
                    spark=spark,
                    input_source='files',
                    file_paths=[str(annotated_files[0])],
                    line_level=True,
                    coalesce_labels=False,  # Disabled
                    model_type='logistic',
                    output_format='annotated',
                    auto_load_model=False
                )

                # Use existing model
                classifier_no_coalesce._model = classifier._model
                classifier_no_coalesce._feature_pipeline = classifier._feature_pipeline
                classifier_no_coalesce._label_mapping = classifier._label_mapping
                classifier_no_coalesce._reverse_label_mapping = classifier._reverse_label_mapping

                predictions_no_coalesce = classifier_no_coalesce.predict()

                # Should have same structure (coalescing happens at save time)
                assert 'predicted_label' in predictions_no_coalesce.columns
                print("✓ No-coalesce mode works correctly")

        print("\n" + "="*50)
        print("All tests passed!")
        print("="*50)

    finally:
        spark.stop()


if __name__ == "__main__":
    test_line_classification()
