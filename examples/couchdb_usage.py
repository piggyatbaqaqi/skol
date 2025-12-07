"""
Example of using SKOL classifier V2 with CouchDB for input/output

This example demonstrates the SkolClassifierV2 unified API for:
- Loading raw documents from CouchDB
- Training/loading models
- Making predictions
- Saving annotated results back to CouchDB

The V2 API simplifies CouchDB workflows with configuration-driven behavior.
All CouchDB operations are distributed using PySpark's partitioning for scalability.
"""

import redis
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2
from skol_classifier import get_file_list


def example_couchdb_prediction():
    """
    Load model from Redis, predict from CouchDB, save back to CouchDB.

    This demonstrates the unified V2 API where all configuration
    is specified in the constructor.
    """
    spark = SparkSession.builder \
        .appName("SKOL CouchDB Prediction") \
        .master("local[*]") \
        .getOrCreate()

    # CouchDB settings
    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    username = "admin"
    password = "password"

    # Redis settings
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Initialize classifier with unified configuration
    print("Initializing classifier with V2 API...")
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url=couchdb_url,
        couchdb_database=database,
        couchdb_username=username,
        couchdb_password=password,
        couchdb_pattern='*.txt',
        output_dest='couchdb',
        output_couchdb_suffix='.ann',
        model_storage='redis',
        redis_client=redis_client,
        redis_key='production_model',
        auto_load_model=True,
        line_level=True,
        coalesce_labels=True,
        output_format='annotated'
    )

    print(f"Model loaded from Redis")

    # Load, predict, and save in streamlined workflow
    print("\nLoading documents from CouchDB...")
    raw_df = classifier.load_raw()
    print(f"Loaded {raw_df.count()} documents")

    print("\nMaking predictions...")
    predictions = classifier.predict(raw_df)

    # Show sample predictions
    print("\nSample predictions:")
    predictions.select(
        "doc_id", "attachment_name", "predicted_label"
    ).show(5, truncate=50)

    # Save back to CouchDB
    print("\nSaving annotations back to CouchDB...")
    classifier.save_annotated(predictions)

    print("✓ Complete! Predictions saved as .ann attachments")
    spark.stop()


def example_train_and_predict():
    """
    Train model from files, save to Redis, then predict from CouchDB.

    Demonstrates using file-based training with CouchDB-based prediction.
    """
    spark = SparkSession.builder \
        .appName("SKOL Train and Predict") \
        .master("local[*]") \
        .getOrCreate()

    # Get annotated training files
    annotated_files = get_file_list("/data/annotated", pattern="**/*.ann")

    # Redis client
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Step 1: Train model from files and save to Redis
    print("=" * 60)
    print("Step 1: Training model from files")
    print("=" * 60)

    trainer = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=annotated_files,
        model_storage='redis',
        redis_client=redis_client,
        redis_key='skol_production_model',
        line_level=True,
        use_suffixes=True,
        model_type='logistic'
    )

    results = trainer.fit()

    # Save the model to Redis
    trainer.save_model()

    print(f"Training complete!")
    print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"  F1 Score: {results.get('f1_score', 0):.4f}")
    print(f"  Model saved to Redis")

    # Step 2: Use trained model to process CouchDB documents
    print("\n" + "=" * 60)
    print("Step 2: Processing CouchDB documents")
    print("=" * 60)

    predictor = SkolClassifierV2(
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
        redis_key='skol_production_model',
        auto_load_model=True,
        line_level=True,
        coalesce_labels=True
    )

    raw_df = predictor.load_raw()
    predictions = predictor.predict(raw_df)
    predictor.save_annotated(predictions)

    print(f"✓ Processed {raw_df.count()} documents from CouchDB")
    spark.stop()


def example_disk_model_with_couchdb():
    """
    Use disk-based model storage with CouchDB input/output.

    Demonstrates that model storage is independent of data source.
    """
    spark = SparkSession.builder \
        .appName("SKOL Disk Model") \
        .master("local[*]") \
        .getOrCreate()

    # Train and save to disk
    print("Training model and saving to disk...")
    trainer = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=get_file_list("/data/annotated", pattern="**/*.ann"),
        model_storage='disk',
        model_path='models/couchdb_classifier.pkl',
        line_level=True,
        model_type='random_forest',
        n_estimators=100
    )

    trainer.fit()
    trainer.save_model()
    print("✓ Model saved to disk")

    # Load from disk and use with CouchDB
    print("\nLoading model from disk for CouchDB processing...")
    predictor = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='skol_documents',
        couchdb_username='admin',
        couchdb_password='password',
        output_dest='couchdb',
        model_storage='disk',
        model_path='models/couchdb_classifier.pkl',
        auto_load_model=True,
        line_level=True
    )

    raw_df = predictor.load_raw()
    predictions = predictor.predict(raw_df)
    predictor.save_annotated(predictions)

    print("✓ Complete!")
    spark.stop()


def example_couchdb_to_files():
    """
    Read from CouchDB, predict, save to local files.

    Demonstrates mixing input/output destinations.
    """
    spark = SparkSession.builder \
        .appName("SKOL CouchDB to Files") \
        .master("local[*]") \
        .getOrCreate()

    classifier = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='skol_documents',
        couchdb_username='admin',
        couchdb_password='password',
        couchdb_pattern='*.txt',
        output_dest='files',
        output_path='/data/output/annotated',
        model_storage='disk',
        model_path='models/classifier.pkl',
        auto_load_model=True,
        line_level=True
    )

    print("Loading from CouchDB...")
    raw_df = classifier.load_raw()

    print("Making predictions...")
    predictions = classifier.predict(raw_df)

    print("Saving to local files...")
    classifier.save_annotated(predictions)

    print(f"✓ Saved annotated files to {classifier.output_path}")
    spark.stop()


def example_partitioning_control():
    """
    Control parallelism by adjusting DataFrame partitioning.

    More partitions = more parallel CouchDB connections.
    Rule of thumb: 2-4x the number of CPU cores.
    """
    spark = SparkSession.builder \
        .appName("SKOL Partitioning") \
        .master("local[*]") \
        .getOrCreate()

    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    classifier = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='skol_documents',
        couchdb_username='admin',
        couchdb_password='password',
        output_dest='couchdb',
        model_storage='redis',
        redis_client=redis_client,
        redis_key='production_model',
        auto_load_model=True,
        line_level=True
    )

    # Load documents
    raw_df = classifier.load_raw()
    print(f"Loaded {raw_df.count()} documents")
    print(f"Default partitions: {raw_df.rdd.getNumPartitions()}")

    # Repartition for optimal parallelism
    # If you have 100 cores, use 200-400 partitions
    num_partitions = 100
    raw_df = raw_df.repartition(num_partitions)

    print(f"After repartitioning: {num_partitions} partitions")
    print(f"This means {num_partitions} parallel CouchDB connections")

    # Predict with optimized partitioning
    predictions = classifier.predict(raw_df)

    # Save (uses same partitioning)
    classifier.save_annotated(predictions)

    print(f"✓ Processed using {num_partitions} parallel workers")
    spark.stop()


if __name__ == "__main__":
    print("=" * 80)
    print("SKOL Classifier V2 - CouchDB Integration Examples")
    print("=" * 80)

    print("\n" + "=" * 60)
    print("Example 1: CouchDB Prediction with Redis Model")
    print("=" * 60)
    example_couchdb_prediction()

    print("\n" + "=" * 60)
    print("Example 2: Train from Files, Predict from CouchDB")
    print("=" * 60)
    example_train_and_predict()

    print("\n" + "=" * 60)
    print("Example 3: Disk Model with CouchDB")
    print("=" * 60)
    example_disk_model_with_couchdb()

    print("\n" + "=" * 60)
    print("Example 4: CouchDB to Local Files")
    print("=" * 60)
    example_couchdb_to_files()

    print("\n" + "=" * 60)
    print("Example 5: Partitioning Control")
    print("=" * 60)
    example_partitioning_control()

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("\nKey benefits of V2 API:")
    print("- Single constructor with all configuration")
    print("- Unified methods: load_raw(), predict(), save_annotated()")
    print("- Mix and match input sources and output destinations")
    print("- Model storage independent of data source")
    print("=" * 80)
