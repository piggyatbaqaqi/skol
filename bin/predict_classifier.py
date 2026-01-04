#!/usr/bin/env python3
"""
Predict Labels Using SKOL Classifier from Redis

This standalone program loads a trained classifier from Redis, applies it to
documents in CouchDB, and saves the predictions back to CouchDB as .ann files.

Usage:
    python predict_classifier.py [--model MODEL_NAME] [--verbosity LEVEL]
                                [--read-text] [--save-text {eager,lazy}]
                                [--pattern PATTERN] [--batch-size SIZE]

Example:
    python predict_classifier.py --model logistic_sections --verbosity 2
    python predict_classifier.py --pattern "*.pdf" --batch-size 96
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import redis
from pyspark.sql import SparkSession

from skol_classifier.classifier_v2 import SkolClassifierV2


# ============================================================================
# Model Configurations
# ============================================================================
# This table maps model names to their base configurations
# Note: These should match the configurations in train_classifier.py

MODEL_CONFIGS = {
    "logistic_sections": {
        "name": "Logistic Regression (sections, words + suffixes + sections)",
        "extraction_mode": "section",
        "coalesce_labels": True,
        "output_format": "annotated",
    }
}


# ============================================================================
# Environment Configuration
# ============================================================================

def get_env_config() -> Dict[str, Any]:
    """
    Get environment configuration from environment variables or defaults.

    Returns:
        Dictionary of configuration values
    """
    return {
        # CouchDB settings
        'couchdb_host': os.environ.get('COUCHDB_HOST', '127.0.0.1:5984'),
        'couchdb_username': os.environ.get('COUCHDB_USER', 'admin'),
        'couchdb_password': os.environ.get('COUCHDB_PASSWORD', 'SU2orange!'),
        'ingest_db_name': os.environ.get('INGEST_DB_NAME', 'skol_dev'),

        # Redis settings
        'redis_host': os.environ.get('REDIS_HOST', 'localhost'),
        'redis_port': int(os.environ.get('REDIS_PORT', '6379')),

        # Model settings
        'model_version': os.environ.get('MODEL_VERSION', 'v2.0'),

        # Prediction settings
        'couchdb_pattern': os.environ.get('COUCHDB_PATTERN', '*.txt'),
        'prediction_batch_size': int(os.environ.get('PREDICTION_BATCH_SIZE', '96')),
        'num_workers': int(os.environ.get('NUM_WORKERS', '4')),

        # Spark settings
        'cores': int(os.environ.get('SPARK_CORES', '4')),
        'bahir_package': os.environ.get('BAHIR_PACKAGE', 'org.apache.bahir:spark-sql-cloudant_2.12:4.0.0'),
    }


def make_spark_session(config: Dict[str, Any]) -> SparkSession:
    """
    Create and configure a Spark session.

    Args:
        config: Environment configuration dictionary

    Returns:
        Configured SparkSession
    """
    return SparkSession \
        .builder \
        .appName("SKOL Classifier Prediction") \
        .master(f"local[{config['cores']}]") \
        .config("cloudant.protocol", "http") \
        .config("cloudant.host", config['couchdb_host']) \
        .config("cloudant.username", config['couchdb_username']) \
        .config("cloudant.password", config['couchdb_password']) \
        .config("spark.jars.packages", config['bahir_package']) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_and_save(
    model_name: str,
    model_config: Dict[str, Any],
    config: Dict[str, Any],
    redis_client: redis.Redis,
    verbosity_override: int = None,
    read_text_override: bool = None,
    save_text_override: str = None,
    pattern_override: str = None,
    batch_size_override: int = None
) -> None:
    """
    Load classifier from Redis, make predictions, and save to CouchDB.

    Args:
        model_name: Name of the model configuration to use
        model_config: Model configuration dictionary
        config: Environment configuration
        redis_client: Redis client instance
        verbosity_override: Optional verbosity level override
        read_text_override: Optional read_text parameter override
        save_text_override: Optional save_text parameter override
        pattern_override: Optional CouchDB pattern override
        batch_size_override: Optional batch size override
    """
    # Apply overrides if provided
    model_config = model_config.copy()
    if verbosity_override is not None:
        model_config['verbosity'] = verbosity_override
    else:
        model_config.setdefault('verbosity', 1)

    if read_text_override is not None:
        model_config['read_text'] = read_text_override
    if save_text_override is not None:
        model_config['save_text'] = save_text_override

    # Determine batch size and pattern
    batch_size = batch_size_override if batch_size_override is not None else config['prediction_batch_size']
    pattern = pattern_override if pattern_override is not None else config['couchdb_pattern']

    model_config['prediction_batch_size'] = batch_size
    model_config['num_workers'] = config['num_workers']

    # Build Redis key for model
    classifier_model_name = f"skol:classifier:model:{model_name}_{config['model_version']}"

    # Build CouchDB URL
    couchdb_url = f"http://{config['couchdb_host']}"

    print(f"\n{'='*70}")
    print(f"Predicting with Model: {model_config.get('name', model_name)}")
    print(f"{'='*70}")
    print(f"Redis key: {classifier_model_name}")
    print(f"CouchDB: {couchdb_url}")
    print(f"Database: {config['ingest_db_name']}")
    print(f"Pattern: {pattern}")
    print(f"Batch size: {batch_size}")
    print()

    # Check if model exists in Redis
    if not redis_client.exists(classifier_model_name):
        print(f"✗ Model not found in Redis: {classifier_model_name}")
        print(f"  Please train the model first using train_classifier.py")
        sys.exit(1)

    # Create Spark session
    print("Initializing Spark session...")
    spark = make_spark_session(config)

    try:
        # Create classifier instance
        print("Creating classifier with SkolClassifierV2...")
        classifier = SkolClassifierV2(
            spark=spark,

            # Input configuration
            input_source='couchdb',
            couchdb_url=couchdb_url,
            couchdb_database=config['ingest_db_name'],
            couchdb_username=config['couchdb_username'],
            couchdb_password=config['couchdb_password'],
            couchdb_pattern=pattern,

            # Output configuration
            output_dest='couchdb',
            output_couchdb_suffix='.ann',

            # Model I/O
            auto_load_model=True,
            model_storage='redis',
            redis_client=redis_client,
            redis_key=classifier_model_name,

            # Model configuration
            **model_config
        )

        print(f"✓ Model loaded from Redis: {classifier_model_name}")

        # Load raw data
        print("\nLoading documents from CouchDB...")
        raw_df = classifier.load_raw()
        doc_count = raw_df.count()
        print(f"✓ Loaded {doc_count} documents")

        if doc_count == 0:
            print("\n⚠ No documents found matching pattern. Nothing to predict.")
            return

        # Show sample
        if model_config.get('verbosity', 1) >= 2:
            print("\nSample documents:")
            raw_df.show(5, truncate=50)

        # Make predictions
        print("\nMaking predictions...")
        predictions = classifier.predict(raw_df)

        # Show sample predictions
        if model_config.get('verbosity', 1) >= 1:
            print("\nSample predictions:")
            predictions.select(
                "doc_id", "line_number", "attachment_name", "predicted_label", "value"
            ).show(5, truncate=50)

        # Save results back to CouchDB
        print("\nSaving predictions to CouchDB as .ann attachments...")
        classifier.save_annotated(predictions)

        print(f"\n{'='*70}")
        print("Prediction Complete!")
        print(f"{'='*70}")
        print(f"✓ Predictions saved to CouchDB")
        print(f"  Database: {config['ingest_db_name']}")
        print(f"  Documents processed: {doc_count}")

    finally:
        # Clean up Spark session
        spark.stop()
        print("\nSpark session stopped.")


# ============================================================================
# Main Program
# ============================================================================

def main():
    """Main entry point for the prediction program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Predict labels using SKOL classifier from Redis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  """ + '\n  '.join([f"{k}: {v.get('name', k)}" for k, v in MODEL_CONFIGS.items()]) + """

Environment Variables:
  COUCHDB_HOST          CouchDB host (default: 127.0.0.1:5984)
  COUCHDB_USER          CouchDB username (default: admin)
  COUCHDB_PASSWORD      CouchDB password (default: SU2orange!)
  INGEST_DB_NAME        Database to predict on (default: skol_dev)
  REDIS_HOST            Redis host (default: localhost)
  REDIS_PORT            Redis port (default: 6379)
  MODEL_VERSION         Model version tag (default: v2.0)
  COUCHDB_PATTERN       File pattern to match (default: *.txt)
  PREDICTION_BATCH_SIZE Batch size for predictions (default: 96)
  NUM_WORKERS           Number of workers (default: 4)
  SPARK_CORES           Number of Spark cores (default: 4)
"""
    )

    parser.add_argument(
        '--model',
        dest='model_name',
        default='logistic_sections',
        choices=list(MODEL_CONFIGS.keys()),
        help='Model configuration to use (default: logistic_sections)'
    )

    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=None,
        help='Override verbosity level (0=silent, 1=info, 2=debug)'
    )

    parser.add_argument(
        '--read-text',
        action='store_true',
        help='Read from .txt attachment instead of converting PDF'
    )

    parser.add_argument(
        '--save-text',
        choices=['eager', 'lazy'],
        default=None,
        help="Save text attachment: 'eager' (always save/replace), 'lazy' (save if missing)"
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        metavar='PATTERN',
        help='CouchDB file pattern to match (default: *.txt)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        metavar='SIZE',
        help='Prediction batch size (default: 96)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model configurations and exit'
    )

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("\nAvailable Model Configurations:")
        print("=" * 70)
        for name, config in MODEL_CONFIGS.items():
            print(f"\n{name}:")
            print(f"  Name: {config.get('name', 'N/A')}")
            print(f"  Extraction mode: {config.get('extraction_mode', 'N/A')}")
        print()
        return

    # Get configuration
    config = get_env_config()

    # Get model configuration
    model_config = MODEL_CONFIGS.get(args.model_name)
    if model_config is None:
        print(f"✗ Unknown model: {args.model_name}")
        print(f"  Available models: {', '.join(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    # Connect to Redis
    print(f"Connecting to Redis at {config['redis_host']}:{config['redis_port']}...")
    try:
        redis_client = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=False  # We need bytes for model serialization
        )
        # Test connection
        redis_client.ping()
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        sys.exit(1)

    # Run predictions
    try:
        predict_and_save(
            model_name=args.model_name,
            model_config=model_config,
            config=config,
            redis_client=redis_client,
            verbosity_override=args.verbosity,
            read_text_override=args.read_text or None,
            save_text_override=args.save_text,
            pattern_override=args.pattern,
            batch_size_override=args.batch_size
        )
    except KeyboardInterrupt:
        print("\n\n✗ Prediction interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
