"""
Example of using SKOL classifier V2 with Redis for model persistence

The V2 API simplifies Redis integration with automatic model saving/loading
controlled by constructor parameters.
"""

import redis
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2
from skol_classifier import get_file_list


def example_save_to_redis():
    """Train a model and save it to Redis."""
    spark = SparkSession.builder \
        .appName("SKOL Redis Save") \
        .master("local[*]") \
        .getOrCreate()

    # Connect to Redis
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=False  # Important: keep as bytes for binary data
    )

    # Train and save to Redis
    print("Training model...")
    annotated_files = get_file_list(
        "/path/to/annotated/data",
        pattern="**/*.txt.ann"
    )

    classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=annotated_files,
        model_storage='redis',
        redis_client=redis_client,
        redis_key="skol_model_v2",
        line_level=True,
        use_suffixes=True,
        model_type='logistic'
    )

    results = classifier.fit()

    # Save the model to Redis
    classifier.save_model()

    print(f"Training complete. F1 Score: {results.get('f1_score', 0):.4f}")
    print("✓ Model saved to Redis!")

    spark.stop()


def example_load_from_redis():
    """Load a pre-trained model from Redis and use it for predictions."""
    spark = SparkSession.builder \
        .appName("SKOL Redis Load") \
        .master("local[*]") \
        .getOrCreate()

    # Connect to Redis
    redis_client = redis.Redis(
        host='localhost',
        port=6379,
        db=0,
        decode_responses=False
    )

    # Initialize classifier with auto_load_model=True
    print("Loading model from Redis...")
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=get_file_list("/path/to/raw/data", pattern="**/*.txt"),
        output_dest='files',
        output_path='/output/annotated',
        model_storage='redis',
        redis_client=redis_client,
        redis_key="skol_model_v2",
        auto_load_model=True,
        line_level=True
    )

    print("✓ Model loaded from Redis!")

    # Use the loaded model
    print("Making predictions...")
    raw_df = classifier.load_raw()
    predictions = classifier.predict(raw_df)

    # Show results
    predictions.select(
        "filename", "predicted_label", "value"
    ).show(10, truncate=50)

    # Save annotated output
    classifier.save_annotated(predictions)
    print("Predictions saved!")

    spark.stop()


def example_with_custom_redis_config():
    """Example with custom Redis configuration."""
    spark = SparkSession.builder \
        .appName("SKOL Redis Custom") \
        .master("local[*]") \
        .getOrCreate()

    # Connect to Redis with custom configuration
    redis_client = redis.Redis(
        host='redis.example.com',
        port=6379,
        db=0,
        password='your_password',
        ssl=True,
        decode_responses=False
    )

    # Train and save with custom key
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=get_file_list("/data/annotated", pattern="**/*.ann"),
        model_storage='redis',
        redis_client=redis_client,
        redis_key="skol_production_model_2024_01_15",
        line_level=True,
        model_type='random_forest',
        n_estimators=100
    )

    results = classifier.fit()

    # Save the model to Redis
    classifier.save_model()

    print(f"Model trained and saved to Redis with custom key")
    print(f"F1 Score: {results.get('f1_score', 0):.4f}")

    spark.stop()


def example_disk_vs_redis():
    """Example comparing disk and Redis storage."""
    spark = SparkSession.builder \
        .appName("SKOL Storage Comparison") \
        .master("local[*]") \
        .getOrCreate()

    annotated_files = get_file_list("/path/to/annotated/data", pattern="**/*.ann")

    # Option 1: Save to disk
    print("Training and saving to disk...")
    disk_classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=annotated_files,
        model_storage='disk',
        model_path='/models/skol_classifier.pkl',
        line_level=True,
        model_type='logistic'
    )
    disk_classifier.fit()
    disk_classifier.save_model()
    print("✓ Model saved to disk at /models/skol_classifier.pkl")

    # Option 2: Save to Redis
    print("\nTraining and saving to Redis...")
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    redis_classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=annotated_files,
        model_storage='redis',
        redis_client=redis_client,
        redis_key='skol_model',
        line_level=True,
        model_type='logistic'
    )
    redis_classifier.fit()
    redis_classifier.save_model()
    print("✓ Model saved to Redis with key 'skol_model'")

    # Later, load from either storage
    print("\nLoading from disk...")
    disk_predictor = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=get_file_list("/path/to/raw/data", pattern="**/*.txt"),
        model_storage='disk',
        model_path='/models/skol_classifier.pkl',
        auto_load_model=True,
        line_level=True
    )
    print("✓ Loaded from disk")

    print("\nLoading from Redis...")
    redis_predictor = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=get_file_list("/path/to/raw/data", pattern="**/*.txt"),
        model_storage='redis',
        redis_client=redis_client,
        redis_key='skol_model',
        auto_load_model=True,
        line_level=True
    )
    print("✓ Loaded from Redis")

    spark.stop()


def example_train_or_load():
    """Example: Train if model doesn't exist, otherwise load from Redis."""
    spark = SparkSession.builder \
        .appName("SKOL Train or Load") \
        .master("local[*]") \
        .getOrCreate()

    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
    model_key = "my_production_model"

    # Check if model exists in Redis
    if redis_client.exists(model_key):
        print("Model found in Redis, loading...")
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=get_file_list("/path/to/raw/data", pattern="**/*.txt"),
            model_storage='redis',
            redis_client=redis_client,
            redis_key=model_key,
            auto_load_model=True,
            line_level=True
        )
        print("✓ Model loaded from Redis")
    else:
        print("No model found in Redis. Training new model...")
        annotated_files = get_file_list("/path/to/annotated/data", pattern="**/*.ann")

        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=annotated_files,
            model_storage='redis',
            redis_client=redis_client,
            redis_key=model_key,
            line_level=True,
            use_suffixes=True,
            model_type='logistic'
        )

        results = classifier.fit()
        classifier.save_model()
        print(f"✓ Model trained and saved to Redis. F1: {results.get('f1_score', 0):.4f}")

    # Now use the model (either freshly trained or loaded)
    classifier.input_source = 'files'
    classifier.file_paths = get_file_list("/path/to/raw/data", pattern="**/*.txt")

    raw_df = classifier.load_raw()
    predictions = classifier.predict(raw_df)
    print(f"Predictions made on {predictions.count()} items")

    spark.stop()


def example_multiple_models():
    """Example of managing multiple models in Redis."""
    spark = SparkSession.builder \
        .appName("SKOL Multiple Models") \
        .master("local[*]") \
        .getOrCreate()

    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
    annotated_files = get_file_list("/path/to/annotated/data", pattern="**/*.ann")

    # Train and save logistic regression model
    print("Training Logistic Regression model...")
    lr_classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=annotated_files,
        model_storage='redis',
        redis_client=redis_client,
        redis_key="skol_model_logistic_v2",
        line_level=True,
        model_type='logistic'
    )
    lr_results = lr_classifier.fit()
    lr_classifier.save_model()
    print(f"✓ LR Model saved. F1: {lr_results.get('f1_score', 0):.4f}")

    # Train and save random forest model
    print("\nTraining Random Forest model...")
    rf_classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=annotated_files,
        model_storage='redis',
        redis_client=redis_client,
        redis_key="skol_model_rf_v2",
        line_level=True,
        model_type='random_forest',
        n_estimators=100
    )
    rf_results = rf_classifier.fit()
    rf_classifier.save_model()
    print(f"✓ RF Model saved. F1: {rf_results.get('f1_score', 0):.4f}")

    # Later, load and compare models
    test_files = get_file_list("/path/to/test/data", pattern="**/*.txt")

    print("\nComparing models on test data...")

    # Load LR model
    lr_predictor = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=test_files,
        model_storage='redis',
        redis_client=redis_client,
        redis_key="skol_model_logistic_v2",
        auto_load_model=True,
        line_level=True
    )

    # Load RF model
    rf_predictor = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=test_files,
        model_storage='redis',
        redis_client=redis_client,
        redis_key="skol_model_rf_v2",
        auto_load_model=True,
        line_level=True
    )

    # Make predictions
    test_df = lr_predictor.load_raw()

    lr_predictions = lr_predictor.predict(test_df)
    rf_predictions = rf_predictor.predict(test_df)

    print(f"LR predictions: {lr_predictions.count()}")
    print(f"RF predictions: {rf_predictions.count()}")

    spark.stop()


def example_redis_with_couchdb():
    """Combine Redis model storage with CouchDB data source."""
    spark = SparkSession.builder \
        .appName("SKOL Redis + CouchDB") \
        .master("local[*]") \
        .getOrCreate()

    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Load model from Redis, process CouchDB data
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='skol_documents',
        couchdb_username='admin',
        couchdb_password='password',
        couchdb_pattern='*.txt',
        output_dest='couchdb',
        output_couchdb_suffix='.ann',
        model_storage='redis',
        redis_client=redis_client,
        redis_key='production_model',
        auto_load_model=True,
        line_level=True,
        coalesce_labels=True
    )

    print("Processing CouchDB documents with Redis-stored model...")
    raw_df = classifier.load_raw()
    predictions = classifier.predict(raw_df)
    classifier.save_annotated(predictions)

    print("✓ Complete! Model from Redis, data from/to CouchDB")

    spark.stop()


if __name__ == "__main__":
    print("=" * 80)
    print("SKOL Classifier V2 - Redis Integration Examples")
    print("=" * 80)

    print("\n" + "=" * 60)
    print("Example 1: Save model to Redis")
    print("=" * 60)
    example_save_to_redis()

    print("\n" + "=" * 60)
    print("Example 2: Load model from Redis")
    print("=" * 60)
    example_load_from_redis()

    print("\n" + "=" * 60)
    print("Example 3: Disk vs Redis storage")
    print("=" * 60)
    example_disk_vs_redis()

    print("\n" + "=" * 60)
    print("Example 4: Train or load from Redis")
    print("=" * 60)
    example_train_or_load()

    print("\n" + "=" * 60)
    print("Example 5: Multiple models in Redis")
    print("=" * 60)
    example_multiple_models()

    print("\n" + "=" * 60)
    print("Example 6: Redis with CouchDB")
    print("=" * 60)
    example_redis_with_couchdb()

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("\nKey benefits of V2 API with Redis:")
    print("- Save models to Redis with save_model() after fit()")
    print("- Automatic load with auto_load_model=True")
    print("- Model storage independent of data source")
    print("- Unified configuration in constructor")
    print("=" * 80)
