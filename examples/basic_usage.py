"""
Basic usage example for SKOL text classifier
"""

from skol_classifier import SkolClassifier, get_file_list


def main():
    # Initialize classifier
    classifier = SkolClassifier()

    # Example 1: Load and train on annotated data
    print("=" * 60)
    print("Example 1: Training the classifier")
    print("=" * 60)

    annotated_files = get_file_list(
        "/path/to/annotated/data",
        pattern="**/*.txt.ann"
    )

    # Train the model
    results = classifier.fit(
        annotated_file_paths=annotated_files,
        model_type="logistic",  # or "random_forest"
        use_suffixes=True,
        test_size=0.2,
        maxIter=10,
        regParam=0.01
    )

    print(f"\nTraining Results:")
    print(f"  Train size: {results['train_size']}")
    print(f"  Test size: {results['test_size']}")
    print(f"  Model: {results['model_type']}")
    print(f"  Features: {results['features_col']}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['f1_score']:.4f}")

    # Example 2: Predict on raw text
    print("\n" + "=" * 60)
    print("Example 2: Predicting on raw text")
    print("=" * 60)

    raw_files = get_file_list(
        "/path/to/raw/data",
        pattern="**/*.txt"
    )

    predictions = classifier.predict_raw_text(
        file_paths=raw_files,
        output_format="annotated"
    )

    # Show sample predictions
    print("\nSample predictions:")
    predictions.select(
        "filename", "row_number", "predicted_label", "value"
    ).show(5, truncate=50)

    # Example 3: Save annotated output
    print("\n" + "=" * 60)
    print("Example 3: Saving annotated output")
    print("=" * 60)

    output_path = "/path/to/output/annotated"
    classifier.save_annotated_output(predictions, output_path)
    print(f"Annotated output saved to: {output_path}")

    # Example 4: Manual pipeline (for more control)
    print("\n" + "=" * 60)
    print("Example 4: Manual pipeline")
    print("=" * 60)

    # Load annotated data
    annotated_df = classifier.load_annotated_data(annotated_files)
    print(f"Loaded {annotated_df.count()} annotated paragraphs")

    # Extract features
    features = classifier.fit_features(
        annotated_df,
        use_suffixes=True,
        min_doc_freq=10
    )
    print(f"Extracted features, labels: {classifier.labels}")

    # Split data
    train_data, test_data = features.randomSplit([0.8, 0.2], seed=42)

    # Train model
    classifier.train_classifier(
        train_data,
        model_type="logistic",
        features_col="combined_idf",
        maxIter=10,
        regParam=0.01
    )
    print("Model trained")

    # Predict and evaluate
    predictions = classifier.predict(test_data)
    stats = classifier.evaluate(predictions)
    print(f"\nEvaluation stats: {stats}")


if __name__ == "__main__":
    main()