"""
Basic usage example for SKOL text classifier using V2 API
"""

from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2
from skol_classifier import get_file_list


def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("SKOL Basic Usage Example") \
        .master("local[*]") \
        .getOrCreate()

    # Example 1: Train from files and save to disk
    print("=" * 60)
    print("Example 1: Training the classifier")
    print("=" * 60)

    annotated_files = get_file_list(
        "/path/to/annotated/data",
        pattern="**/*.txt.ann"
    )

    # Initialize classifier with all configuration in constructor
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=annotated_files,
        model_storage='disk',
        model_path='models/basic_model.pkl',
        line_level=False,
        use_suffixes=True,
        model_type='logistic',
        maxIter=10,
        regParam=0.01
    )

    # Train the model
    results = classifier.fit()

    # Save the model to disk
    classifier.save_model()

    print(f"\nTraining Results:")
    print(f"  Train size: {results.get('train_size', 'N/A')}")
    print(f"  Test size: {results.get('test_size', 'N/A')}")
    print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"  Precision: {results.get('precision', 0):.4f}")
    print(f"  Recall: {results.get('recall', 0):.4f}")
    print(f"  F1 Score: {results.get('f1_score', 0):.4f}")
    print(f"  Model saved to: models/basic_model.pkl")

    # Example 2: Predict on raw text files
    print("\n" + "=" * 60)
    print("Example 2: Predicting on raw text")
    print("=" * 60)

    raw_files = get_file_list(
        "/path/to/raw/data",
        pattern="**/*.txt"
    )

    # Create predictor with same model
    predictor = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=raw_files,
        output_dest='files',
        output_path='/path/to/output/annotated',
        model_storage='disk',
        model_path='models/basic_model.pkl',
        auto_load_model=True,
        line_level=False,
        output_format='annotated'
    )

    # Load, predict, and save
    raw_df = predictor.load_raw()
    print(f"Loaded {raw_df.count()} documents")

    predictions = predictor.predict(raw_df)

    # Show sample predictions
    print("\nSample predictions:")
    predictions.select(
        "filename", "predicted_label", "value"
    ).show(5, truncate=50)

    # Example 3: Save annotated output
    print("\n" + "=" * 60)
    print("Example 3: Saving annotated output")
    print("=" * 60)

    predictor.save_annotated(predictions)
    print(f"Annotated output saved to: {predictor.output_path}")

    # Example 4: Train and predict in one session
    print("\n" + "=" * 60)
    print("Example 4: Complete workflow")
    print("=" * 60)

    # Single classifier instance handles everything
    workflow_classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=annotated_files,
        line_level=True,  # Process at line level
        use_suffixes=True,
        model_type='random_forest',
        n_estimators=50
    )

    # Train
    results = workflow_classifier.fit()
    print(f"Trained with {results.get('train_size', 0)} samples")

    # Switch to prediction mode
    workflow_classifier.input_source = 'files'
    workflow_classifier.file_paths = raw_files
    workflow_classifier.output_dest = 'files'
    workflow_classifier.output_path = '/path/to/output'

    # Predict and save
    raw_df = workflow_classifier.load_raw()
    predictions = workflow_classifier.predict(raw_df)
    workflow_classifier.save_annotated(predictions)

    print(f"Complete workflow finished!")

    spark.stop()


if __name__ == "__main__":
    main()
