#!/usr/bin/env python3
"""
Example usage of SkolClassifierV2 with unified API.

This script demonstrates various usage patterns of the new SkolClassifierV2 class,
showing how the configuration-driven design simplifies common workflows.
"""

from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2


def example_1_train_from_files_save_to_disk():
    """
    Example 1: Train from annotated files, save model to disk.

    This is the simplest workflow - train on local files and save the model.
    """
    print("\n" + "="*70)
    print("Example 1: Train from files, save model to disk")
    print("="*70)

    spark = SparkSession.builder \
        .appName("SkolClassifierV2 Example 1") \
        .master("local[*]") \
        .getOrCreate()

    # Initialize classifier with training configuration
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=['data/training/*.txt.ann'],  # Glob pattern for annotated files
        model_storage='disk',
        model_path='models/taxon_classifier.pkl',
        line_level=True,           # Train on individual lines
        use_suffixes=True,         # Use word suffix features
        model_type='logistic',     # Logistic regression
        collapse_labels=True       # Collapse similar labels
    )

    # Train the model
    stats = classifier.fit()

    # Save the model to disk
    classifier.save_model()

    print(f"\nTraining complete!")
    print(f"  Training samples: {stats.get('train_size', 'N/A')}")
    print(f"  Test samples:     {stats.get('test_size', 'N/A')}")
    print(f"  Accuracy:         {stats.get('accuracy', 'N/A'):.4f}")
    print(f"  Model saved to:   {classifier.model_path}")

    spark.stop()


def example_2_predict_from_couchdb_save_to_couchdb():
    """
    Example 2: Load model, predict from CouchDB, save back to CouchDB.

    This workflow shows how to use a trained model to process documents
    stored in CouchDB and save the predictions back as annotations.
    """
    print("\n" + "="*70)
    print("Example 2: Predict from CouchDB, save to CouchDB")
    print("="*70)

    spark = SparkSession.builder \
        .appName("SkolClassifierV2 Example 2") \
        .master("local[*]") \
        .getOrCreate()

    # Initialize classifier with prediction configuration
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='taxonomic_articles',
        couchdb_username='admin',
        couchdb_password='password',
        couchdb_pattern='*.txt',           # Process all .txt attachments
        output_dest='couchdb',
        output_couchdb_suffix='.ann',      # Save as .txt.ann attachments
        model_storage='disk',
        model_path='models/taxon_classifier.pkl',
        auto_load_model=True,              # Load model on initialization
        line_level=True,
        coalesce_labels=True,              # Merge consecutive same-label lines
        output_format='annotated'
    )

    # Load raw documents from CouchDB
    print("\nLoading documents from CouchDB...")
    raw_df = classifier.load_raw()
    print(f"Loaded {raw_df.count()} documents")

    # Make predictions
    print("\nMaking predictions...")
    predictions_df = classifier.predict(raw_df)
    print(f"Generated predictions for {predictions_df.count()} items")

    # Save predictions back to CouchDB
    print("\nSaving predictions to CouchDB...")
    classifier.save_annotated(predictions_df)
    print("Predictions saved successfully!")

    spark.stop()


def example_3_train_and_predict_same_session():
    """
    Example 3: Train and predict in the same session with different configs.

    This shows how to use two separate classifier instances for training
    and prediction, demonstrating the flexibility of the unified API.
    """
    print("\n" + "="*70)
    print("Example 3: Train and predict in same session")
    print("="*70)

    spark = SparkSession.builder \
        .appName("SkolClassifierV2 Example 3") \
        .master("local[*]") \
        .getOrCreate()

    # Step 1: Train a model from files
    print("\nStep 1: Training model from annotated files...")
    trainer = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=['data/training/*.txt.ann'],
        model_storage='disk',
        model_path='models/quick_model.pkl',
        line_level=False,          # Train on paragraphs
        use_suffixes=True,
        model_type='random_forest',
        n_estimators=100,          # Model-specific parameter
        max_depth=10
    )

    stats = trainer.fit()

    # Save the model to disk
    trainer.save_model()

    print(f"Training complete - Accuracy: {stats.get('accuracy', 'N/A'):.4f}")

    # Step 2: Use the trained model to predict on new files
    print("\nStep 2: Making predictions on new files...")
    predictor = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=['data/unlabeled/*.txt'],
        output_dest='files',
        output_path='data/predictions/',
        model_storage='disk',
        model_path='models/quick_model.pkl',
        auto_load_model=True,      # Load the model we just trained
        line_level=False,
        output_format='annotated'
    )

    # Load, predict, and save
    raw_df = predictor.load_raw()
    predictions_df = predictor.predict(raw_df)
    predictor.save_annotated(predictions_df)

    print(f"Predictions saved to {predictor.output_path}")

    spark.stop()


def example_4_redis_model_storage():
    """
    Example 4: Use Redis for model storage.

    This demonstrates using Redis as the model storage backend,
    which is useful for shared model access across multiple processes.
    """
    print("\n" + "="*70)
    print("Example 4: Redis model storage")
    print("="*70)

    try:
        import redis
    except ImportError:
        print("Redis library not installed. Skipping this example.")
        print("Install with: pip install redis")
        return

    spark = SparkSession.builder \
        .appName("SkolClassifierV2 Example 4") \
        .master("local[*]") \
        .getOrCreate()

    # Connect to Redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0)

    # Train and save to Redis
    print("\nTraining model and saving to Redis...")
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=['data/training/*.txt.ann'],
        model_storage='redis',
        redis_client=redis_client,
        redis_key='skol:taxon_model:v1',
        line_level=True,
        use_suffixes=True,
        model_type='logistic'
    )

    stats = classifier.fit()

    # Save the model to Redis
    classifier.save_model()

    print(f"Model trained and saved to Redis key: {classifier.redis_key}")

    # Load from Redis and predict
    print("\nLoading model from Redis for prediction...")
    predictor = SkolClassifierV2(
        spark=spark,
        input_source='files',
        file_paths=['data/unlabeled/*.txt'],
        output_dest='files',
        output_path='data/predictions/',
        model_storage='redis',
        redis_client=redis_client,
        redis_key='skol:taxon_model:v1',
        auto_load_model=True,
        line_level=True
    )

    raw_df = predictor.load_raw()
    predictions_df = predictor.predict(raw_df)
    predictor.save_annotated(predictions_df)

    print("Predictions complete!")

    spark.stop()


def example_5_model_comparison():
    """
    Example 5: Compare different model configurations.

    This shows how easy it is to train and compare multiple models
    with the unified API.
    """
    print("\n" + "="*70)
    print("Example 5: Model comparison")
    print("="*70)

    spark = SparkSession.builder \
        .appName("SkolClassifierV2 Example 5") \
        .master("local[*]") \
        .getOrCreate()

    # Define model configurations to compare
    configs = [
        {
            'name': 'Logistic (line-level)',
            'model_type': 'logistic',
            'line_level': True,
            'use_suffixes': True
        },
        {
            'name': 'Logistic (paragraph-level)',
            'model_type': 'logistic',
            'line_level': False,
            'use_suffixes': True
        },
        {
            'name': 'Random Forest (line-level)',
            'model_type': 'random_forest',
            'line_level': True,
            'use_suffixes': True,
            'n_estimators': 50
        },
        {
            'name': 'Gradient Boosted (line-level)',
            'model_type': 'gradient_boosted',
            'line_level': True,
            'use_suffixes': True,
            'max_iter': 50
        }
    ]

    results = []

    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-" * 70)

        # Extract name and model_type
        name = config.pop('name')

        # Create classifier with this configuration
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['data/training/*.txt.ann'],
            **config  # Pass all other config parameters
        )

        # Train and collect stats
        stats = classifier.fit()

        results.append({
            'name': name,
            'accuracy': stats.get('accuracy', 0.0),
            'precision': stats.get('precision', 0.0),
            'recall': stats.get('recall', 0.0),
            'f1': stats.get('f1', 0.0)
        })

        print(f"  Accuracy:  {stats.get('accuracy', 0.0):.4f}")
        print(f"  Precision: {stats.get('precision', 0.0):.4f}")
        print(f"  Recall:    {stats.get('recall', 0.0):.4f}")
        print(f"  F1 Score:  {stats.get('f1', 0.0):.4f}")

    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<40} {'Accuracy':>10} {'F1 Score':>10}")
    print("-" * 70)
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['name']:<40} {result['accuracy']:>10.4f} {result['f1']:>10.4f}")

    spark.stop()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("SkolClassifierV2 Examples")
    print("="*70)
    print("\nThis script demonstrates various usage patterns of the new unified API.")
    print("Each example shows a different workflow:")
    print("  1. Train from files, save to disk")
    print("  2. Predict from CouchDB, save to CouchDB")
    print("  3. Train and predict in same session")
    print("  4. Use Redis for model storage")
    print("  5. Compare different model configurations")
    print("\nNote: Some examples require data files and/or services to be available.")

    # Run examples (comment out ones that require unavailable resources)
    try:
        example_1_train_from_files_save_to_disk()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")

    try:
        example_2_predict_from_couchdb_save_to_couchdb()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")

    try:
        example_3_train_and_predict_same_session()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")

    try:
        example_4_redis_model_storage()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")

    try:
        example_5_model_comparison()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")

    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
