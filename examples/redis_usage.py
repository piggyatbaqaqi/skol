"""
Example of using SKOL classifier with Redis for model persistence
"""

import redis
from skol_classifier import SkolClassifier, get_file_list


def example_save_to_redis():
    """Train a model and save it to Redis."""

    # Connect to Redis
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=False  # Important: keep as bytes for binary data
    )

    # Initialize classifier with Redis connection
    classifier = SkolClassifier(
        redis_client=redis_client,
        redis_key="skol_model_v1"
    )

    # Train the model
    print("Training model...")
    annotated_files = get_file_list(
        "/path/to/annotated/data",
        pattern="**/*.txt.ann"
    )

    results = classifier.fit(
        annotated_file_paths=annotated_files,
        model_type="logistic",
        use_suffixes=True,
        test_size=0.2
    )

    print(f"Training complete. F1 Score: {results['f1_score']:.4f}")

    # Save to Redis
    print("Saving model to Redis...")
    success = classifier.save_to_redis()

    if success:
        print("Model successfully saved to Redis!")
    else:
        print("Failed to save model to Redis")


def example_load_from_redis():
    """Load a pre-trained model from Redis and use it for predictions."""

    # Connect to Redis
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=False
    )

    # Initialize classifier
    classifier = SkolClassifier(
        redis_client=redis_client,
        redis_key="skol_model_v1"
    )

    # Load model from Redis
    print("Loading model from Redis...")
    success = classifier.load_from_redis()

    if not success:
        print("Failed to load model from Redis")
        return

    print("Model loaded successfully!")
    print(f"Model labels: {classifier.labels}")

    # Use the loaded model for predictions
    raw_files = get_file_list("/path/to/raw/data", pattern="**/*.txt")

    print("Making predictions...")
    predictions = classifier.predict_raw_text(raw_files)

    # Show results
    predictions.select(
        "filename", "predicted_label", "value"
    ).show(10, truncate=50)

    # Save annotated output
    classifier.save_annotated_output(predictions, "/output/annotated")
    print("Predictions saved!")


def example_with_custom_redis_config():
    """Example with custom Redis configuration."""

    # Connect to Redis with custom configuration
    redis_client = redis.Redis(
        host='redis.example.com',
        port=6379,
        db=0,
        password='your_password',
        ssl=True,
        decode_responses=False
    )

    # Initialize classifier
    classifier = SkolClassifier(
        redis_client=redis_client,
        redis_key="skol_production_model"
    )

    # Train and save
    annotated_files = get_file_list("/data/annotated")
    results = classifier.fit(annotated_files)

    # Save with explicit parameters
    classifier.save_to_redis(
        redis_client=redis_client,
        redis_key="skol_model_2024_01_15"
    )


def example_save_to_disk():
    """Example of saving/loading models to/from disk instead of Redis."""

    # Train model
    classifier = SkolClassifier()
    annotated_files = get_file_list("/path/to/annotated/data")
    results = classifier.fit(annotated_files)

    # Save to disk
    print("Saving model to disk...")
    classifier.save_to_disk("/models/skol_classifier")
    print("Model saved!")

    # Later, load from disk
    new_classifier = SkolClassifier()
    print("Loading model from disk...")
    new_classifier.load_from_disk("/models/skol_classifier")
    print(f"Model loaded! Labels: {new_classifier.labels}")

    # Use loaded model
    raw_files = get_file_list("/path/to/raw/data")
    predictions = new_classifier.predict_raw_text(raw_files)


def example_multiple_models():
    """Example of managing multiple models in Redis."""

    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Train and save model version 1 (logistic regression)
    classifier_v1 = SkolClassifier(redis_client=redis_client)
    annotated_files = get_file_list("/path/to/annotated/data")

    print("Training Logistic Regression model...")
    classifier_v1.fit(annotated_files, model_type="logistic")
    classifier_v1.save_to_redis(redis_key="skol_model_logistic")

    # Train and save model version 2 (random forest)
    classifier_v2 = SkolClassifier(redis_client=redis_client)
    print("Training Random Forest model...")
    classifier_v2.fit(annotated_files, model_type="random_forest")
    classifier_v2.save_to_redis(redis_key="skol_model_rf")

    # Later, compare models by loading them
    lr_model = SkolClassifier(redis_client=redis_client)
    lr_model.load_from_redis(redis_key="skol_model_logistic")

    rf_model = SkolClassifier(redis_client=redis_client)
    rf_model.load_from_redis(redis_key="skol_model_rf")

    # Use both for predictions and compare
    test_files = get_file_list("/path/to/test/data")

    lr_predictions = lr_model.predict_raw_text(test_files)
    rf_predictions = rf_model.predict_raw_text(test_files)


if __name__ == "__main__":
    # Run examples
    print("=" * 60)
    print("Example 1: Save model to Redis")
    print("=" * 60)
    example_save_to_redis()

    print("\n" + "=" * 60)
    print("Example 2: Load model from Redis")
    print("=" * 60)
    example_load_from_redis()

    print("\n" + "=" * 60)
    print("Example 3: Save/Load from disk")
    print("=" * 60)
    example_save_to_disk()
