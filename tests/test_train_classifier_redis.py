#!/usr/bin/env python3
"""
Test program for training a classifier and saving to Redis using SkolClassifierV2.

This script is extracted from the Jupyter notebook jupyter/ist769_skol.ipynb,
specifically the block: "Train classifier on annotated data and save to Redis using SkolClassifierV2"

Usage:
    python tests/test_train_classifier_redis.py [--model-type {rnn,logistic}] [--force]

Arguments:
    --model-type: Model type to train (rnn or logistic). Default: rnn
    --force: Force retraining even if model exists in Redis
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from pyspark.sql import SparkSession

from skol_classifier.classifier_v2 import SkolClassifierV2
from skol_classifier.utils import get_file_list


def make_spark_session(cores: int, couchdb_host: str, couchdb_username: str,
                       couchdb_password: str, bahir_package: str) -> SparkSession:
    """
    Create a Spark session with CouchDB configuration.

    Args:
        cores: Number of cores to use
        couchdb_host: CouchDB host (e.g., "127.0.0.1:5984")
        couchdb_username: CouchDB username
        couchdb_password: CouchDB password
        bahir_package: Bahir package for CouchDB integration

    Returns:
        SparkSession configured for CouchDB
    """
    return SparkSession \
        .builder \
        .appName("SKOL Classifier Training - Redis Storage") \
        .master(f"local[{cores}]") \
        .config("cloudant.protocol", "http") \
        .config("cloudant.host", couchdb_host) \
        .config("cloudant.username", couchdb_username) \
        .config("cloudant.password", couchdb_password) \
        .config("spark.jars.packages", bahir_package) \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "20g") \
        .config("spark.driver.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
                "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED") \
        .config("spark.executor.extraJavaOptions",
                "--add-opens=java.base/java.nio=ALL-UNNAMED "
                "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
                "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED") \
        .getOrCreate()


def get_rnn_config(cores: int, epochs: int = 4, verbosity: int = 1, **kwargs) -> dict:
    """
    Get RNN model configuration.

    Args:
        cores: Number of cores for num_workers

    Returns:
        Dictionary with RNN model configuration
    """
    return {
        "name": "RNN BiLSTM (line-level, advanced config)",
        "model_type": "rnn",
        "use_suffixes": True,
        "line_level": True,
        "input_size": 1000,
        "hidden_size": 128,
        "num_layers": 2,
        "num_classes": 3,
        "dropout": 0.3,
        "window_size": 20,
        "prediction_stride": 20,  # 0 overlap
        "prediction_batch_size": 32,
        "batch_size": 16384,  # 16730MiB footprint, 38s-40s per step
        "epochs": epochs,
        "num_workers": cores,
        "verbosity": verbosity,
        **kwargs
    }


def get_logistic_config(**kwargs) -> dict:
    """
    Get Logistic Regression model configuration.

    Returns:
        Dictionary with Logistic Regression configuration
    """
    return {
        "name": "Logistic Regression (line-level, words + suffixes)",
        "model_type": "logistic",
        "use_suffixes": True,
        "maxIter": 10,
        "regParam": 0.01,
        "line_level": True,
        **kwargs
    }


def train_classifier(
    model_to_use: str = "rnn", force_retrain: bool = False, save_model: bool = False, epochs: int = 4, verbosity: int = 1):
    """
    Train a SKOL classifier and save to Redis.

    Args:
        model_to_use: Model type ("rnn" or "logistic")
        force_retrain: Force retraining even if model exists
    """
    # Configuration
    cores = 2

    # CouchDB configuration
    couchdb_host = "127.0.0.1:5984"
    couchdb_username = "admin"
    couchdb_password = "SU2orange!"
    couchdb_url = f'http://{couchdb_host}'
    ingest_db_name = "skol_dev"

    # Redis configuration
    redis_host = "127.0.0.1"
    redis_port = 6379
    redis_db = 0
    classifier_model_name = f"skol:classifier:model:{model_to_use}-v1.0"
    classifier_model_expire = 60 * 60 * 24 * 2  # Expire after 2 days

    # Spark configuration
    bahir_package = 'org.apache.bahir:spark-sql-cloudant_2.12:2.4.0'

    # Connect to Redis
    print(f"Connecting to Redis at {redis_host}:{redis_port}...")
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=False
    )

    # Check if we should skip training
    create_classifier = force_retrain or not redis_client.exists(classifier_model_name)

    if not create_classifier:
        print(f"Model '{classifier_model_name}' already exists in Redis.")
        print("Use --force to retrain anyway.")
        return

    # Get model configuration
    print(f"\nConfiguring {model_to_use.upper()} model...")
    if model_to_use == "rnn":
        model_config = get_rnn_config(cores, epochs=epochs, verbosity=verbosity)
    elif model_to_use == "logistic":
        model_config = get_logistic_config(epochs=epochs, verbosity=verbosity)
    else:
        raise ValueError(f"Unrecognized model: {model_to_use}")

    print(f"Model configuration:")
    for key, value in model_config.items():
        if key not in ['name']:
            print(f"  {key}: {value}")

    # Get annotated training files
    annotated_path = Path(__file__).parent.parent / "data" / "annotated"
    print(f"\nLoading annotated files from: {annotated_path}")

    if not annotated_path.exists():
        print(f"ERROR: Directory does not exist: {annotated_path}")
        print("Please ensure annotated training data is available.")
        sys.exit(1)

    annotated_files = get_file_list(str(annotated_path), pattern="**/*.ann")

    if len(annotated_files) == 0:
        print(f"ERROR: No annotated files found in {annotated_path}")
        sys.exit(1)

    print(f"Found {len(annotated_files)} annotated files")

    # Create Spark session
    print("\nCreating Spark session...")
    spark = make_spark_session(
        cores=cores,
        couchdb_host=couchdb_host,
        couchdb_username=couchdb_username,
        couchdb_password=couchdb_password,
        bahir_package=bahir_package
    )

    # Train using SkolClassifierV2 with unified API
    print("\n" + "="*70)
    print("Training classifier with SkolClassifierV2...")
    print("="*70)

    classifier = SkolClassifierV2(
        spark=spark,

        # Input
        input_source='files',
        file_paths=annotated_files,

        # Model I/O
        auto_load_model=False,  # Fit a new model
        model_storage='redis',
        redis_client=redis_client,
        redis_key=classifier_model_name,
        redis_expire=classifier_model_expire,

        # Output options
        output_dest='couchdb',
        couchdb_url=couchdb_url,
        couchdb_database=ingest_db_name,
        output_couchdb_suffix='.ann',

        # Model and preprocessing options
        **model_config
    )

    # Train the model
    print("\nStarting training...")
    results = classifier.fit()

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)
    print(f"  Accuracy:  {results.get('accuracy', 0):.4f}")
    print(f"  Precision: {results.get('precision', 0):.4f}")
    print(f"  Recall:    {results.get('recall', 0):.4f}")
    print(f"  F1 Score:  {results.get('f1_score', 0):.4f}")
    print(f"  Frequencies: {results.get('class_frequencies', {})}")

    # Save model to Redis
    if save_model:
        print(f"\nSaving model to Redis...")
        classifier.save_model()
        print(f"✓ Model saved to Redis with key: {classifier_model_name}")
        print(f"  Expiration: {classifier_model_expire} seconds ({classifier_model_expire / 86400:.1f} days)")

    # Stop Spark
    spark.stop()
    print("\n✓ Training complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train SKOL classifier and save to Redis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model-type',
        choices=['rnn', 'logistic'],
        default='rnn',
        help='Model type to train (default: rnn)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if model exists in Redis'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the trained model to Redis (default: do not save)' # Default: do not save
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        help='Verbosity level for logging (default: 1)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of epochs to train the model (default: 4)'
    )

    args = parser.parse_args()

    try:
        train_classifier(
            model_to_use=args.model_type,
            force_retrain=args.force,
            save_model=args.save_model,
            epochs=args.epochs,
            verbosity=args.verbosity
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
