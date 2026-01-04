#!/usr/bin/env python3
"""
Train SKOL Classifier and Save to Redis

This standalone program trains a text classifier using annotated data
and saves it to Redis for later use.

Usage:
    python train_classifier.py [--model MODEL_NAME] [--verbosity LEVEL]
                              [--read-text] [--save-text {eager,lazy}]

Example:
    python train_classifier.py --model logistic_sections --verbosity 2
    python train_classifier.py --read-text --save-text lazy
"""

import argparse
from datetime import datetime
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import redis
from pyspark.sql import SparkSession

from skol_classifier.classifier_v2 import SkolClassifierV2
from skol_classifier.utils import get_file_list


# ============================================================================
# Model Configurations
# ============================================================================
# This table is meant to be edited by hand to add or modify model configurations

MODEL_CONFIGS = {
    "logistic_sections": {
        "name": "Logistic Regression (sections, words + suffixes + sections)",
        "model_type": "logistic",
        "verbosity": 2,
        "input_source": "couchdb",
        "couchdb_training_database": "skol_training",
        "use_suffixes": True,
        "maxIter": 100,
        "regParam": 0.01,
        "extraction_mode": "section",
        "class_weights": {
            "Nomenclature": 250.0,
            "Description": 20.0,
            "Misc-exposition": 20.0
        },
        "word_vocab_size": 3600,
        "suffix_vocab_size": 400,
        "section_name_vocab_size": 50,
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
        'classifier_model_expire': os.environ.get('MODEL_EXPIRE', None),  # %H:%M:%S; None = never expires

        # Data paths
        'annotated_path': Path(os.environ.get('ANNOTATED_PATH', Path.cwd().parent / "data" / "annotated")),

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
        .appName("SKOL Classifier Training") \
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
# Training Functions
# ============================================================================

def train_classifier(
    model_name: str,
    model_config: Dict[str, Any],
    config: Dict[str, Any],
    redis_client: redis.Redis,
    verbosity_override: Optional[int] = None,
    read_text_override: Optional[bool] = None,
    save_text_override: Optional[str] = None,
    expire_override: Optional[str] = None
) -> None:
    """
    Train a classifier model and save it to Redis.

    Args:
        model_name: Name of the model configuration to use
        model_config: Model configuration dictionary
        config: Environment configuration
        redis_client: Redis client instance
        verbosity_override: Optional verbosity level override
        read_text_override: Optional read_text parameter override
        save_text_override: Optional save_text parameter override ('eager', 'lazy', or None)
        expire_override: Optional Redis expiration time override in HH:MM:SS format (None = no expiration)
    """
    # Apply overrides if provided
    if verbosity_override is not None or read_text_override is not None or save_text_override is not None:
        model_config = model_config.copy()
        if verbosity_override is not None:
            model_config['verbosity'] = verbosity_override
        if read_text_override is not None:
            model_config['read_text'] = read_text_override
        if save_text_override is not None:
            model_config['save_text'] = save_text_override

    # Determine Redis expiration time
    expire_time = config.get('classifier_model_expire')
    if expire_override is not None:
        duration = datetime.strptime(expire_override, '%H:%M:%S')
        expire_time = duration.hour * 3600 + duration.minute * 60 + duration.second
    redis_expire = expire_time

    # Build Redis key for model
    classifier_model_name = f"skol:classifier:model:{model_name}_{config['model_version']}"

    # Build CouchDB URL
    couchdb_url = f"http://{config['couchdb_host']}"

    print(f"\n{'='*70}")
    print(f"Training Model: {model_config.get('name', model_name)}")
    print(f"{'='*70}")
    print(f"Redis key: {classifier_model_name}")
    print(f"CouchDB: {couchdb_url}")
    print(f"Training database: {model_config.get('couchdb_training_database')}")
    print()

    # Create Spark session
    print("Initializing Spark session...")
    spark = make_spark_session(config)

    try:
        # Create classifier instance
        print("Creating classifier with SkolClassifierV2...")
        classifier = SkolClassifierV2(
            spark=spark,

            # Model I/O
            auto_load_model=False,  # Fit a new model
            model_storage='redis',
            redis_client=redis_client,
            redis_key=classifier_model_name,
            redis_expire=redis_expire,

            # Output options
            output_dest='couchdb',
            couchdb_url=couchdb_url,
            couchdb_database=config['ingest_db_name'],
            couchdb_username=config['couchdb_username'],
            couchdb_password=config['couchdb_password'],
            output_couchdb_suffix='.ann',

            # Model and preprocessing options
            **model_config
        )

        # Train the model
        print("\nTraining model...")
        results = classifier.fit()

        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"  F1 Score: {results.get('f1_score', 0):.4f}")

        # Save model to Redis
        classifier.save_model()
        print(f"\n✓ Model saved to Redis with key: {classifier_model_name}")
        if redis_expire is not None:
            print(f"  Expiration: {redis_expire} seconds")
        else:
            print(f"  Expiration: None (never expires)")

    finally:
        # Clean up Spark session
        spark.stop()
        print("\nSpark session stopped.")


# ============================================================================
# Main Program
# ============================================================================

def main():
    """Main entry point for the training program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train SKOL classifier and save to Redis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  """ + '\n  '.join([f"{k}: {v.get('name', k)}" for k, v in MODEL_CONFIGS.items()]) + """

Environment Variables:
  COUCHDB_HOST          CouchDB host (default: 127.0.0.1:5984)
  COUCHDB_USER          CouchDB username (default: admin)
  COUCHDB_PASSWORD      CouchDB password (default: SU2orange!)
  INGEST_DB_NAME        Ingestion database name (default: skol_dev)
  REDIS_HOST            Redis host (default: localhost)
  REDIS_PORT            Redis port (default: 6379)
  MODEL_VERSION         Model version tag (default: v2.0)
  MODEL_EXPIRE          Model expiration in seconds (default: 172800)
  ANNOTATED_PATH        Path to annotated training data
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
        '--expire',
        type=int,
        default=None,
        metavar='SECONDS',
        help='Redis key expiration time in seconds (None = never expires, default: 172800)'
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
            print(f"  Type: {config.get('model_type', 'N/A')}")
            print(f"  Extraction mode: {config.get('extraction_mode', 'N/A')}")
            print(f"  Verbosity: {config.get('verbosity', 'N/A')}")
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

    # Train the model
    try:
        train_classifier(
            model_name=args.model_name,
            model_config=model_config,
            config=config,
            redis_client=redis_client,
            verbosity_override=args.verbosity,
            read_text_override=args.read_text or None,
            save_text_override=args.save_text,
            expire_override=args.expire
        )
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
