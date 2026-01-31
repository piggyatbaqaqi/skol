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

    # Skip if model already exists in Redis
    python train_classifier.py --skip-existing

    # Force retrain even if model exists
    python train_classifier.py --force

    # Preview what would be trained without saving
    python train_classifier.py --dry-run
"""

import argparse
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import redis
from pyspark.sql import SparkSession

# Add parent directory to path for skol modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skol_classifier.classifier_v2 import SkolClassifierV2
from skol_classifier.utils import get_file_list
from env_config import get_env_config, create_redis_client


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
        .config("spark.driver.memory", config['spark_driver_memory']) \
        .config("spark.executor.memory", config['spark_executor_memory']) \
        .getOrCreate()


# ============================================================================
# Training Functions
# ============================================================================

def train_classifier(
    model_name: str,
    model_config: Dict[str, Any],
    config: Dict[str, Any],
    redis_client: redis.Redis,
    read_text_override: Optional[bool] = None,
    save_text_override: Optional[str] = None,
    expire_override: Optional[str] = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    force: bool = False,
) -> None:
    """
    Train a classifier model and save it to Redis.

    Args:
        model_name: Name of the model configuration to use
        model_config: Model configuration dictionary
        config: Environment configuration
        redis_client: Redis client instance
        read_text_override: Optional read_text parameter override
        save_text_override: Optional save_text parameter override ('eager', 'lazy', or None)
        expire_override: Optional Redis expiration time override in HH:MM:SS format (None = no expiration)
        dry_run: If True, preview without training or saving
        skip_existing: If True, skip if model already exists in Redis
        force: If True, train even if model exists (overrides skip_existing)
    """
    # Apply overrides if provided and use config verbosity
    model_config = model_config.copy()

    # Override verbosity from config (command-line or environment)
    model_config['verbosity'] = config['verbosity']
    model_config['union_batch_size'] = config['union_batch_size']

    if read_text_override is not None:
        model_config['read_text'] = read_text_override
    if save_text_override is not None:
        model_config['save_text'] = save_text_override

    # Determine Redis expiration time
    expire_time = config.get('classifier_model_expire')
    if expire_override is not None:
        if expire_override.lower() == 'none':
            expire_time = None
        else:
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
    if dry_run:
        print(f"Mode: DRY RUN (no changes will be saved)")
    if skip_existing and not force:
        print(f"Mode: SKIP EXISTING (skip if model exists in Redis)")
    if force:
        print(f"Mode: FORCE (train even if model exists)")
    print()

    # Check if model already exists in Redis (unless --force)
    model_exists = redis_client.exists(classifier_model_name)
    if skip_existing and not force and model_exists:
        print(f"✓ Model already exists in Redis: {classifier_model_name}")
        print("  Use --force to retrain anyway")
        return

    if model_exists and not force:
        print(f"⚠ Warning: Model will overwrite existing key: {classifier_model_name}")

    # Handle dry-run mode
    if dry_run:
        print(f"\n[DRY RUN] Would train model with configuration:")
        print(f"  Model type: {model_config.get('model_type', 'N/A')}")
        print(f"  Extraction mode: {model_config.get('extraction_mode', 'N/A')}")
        print(f"  Word vocab size: {model_config.get('word_vocab_size', 'N/A')}")
        print(f"  Suffix vocab size: {model_config.get('suffix_vocab_size', 'N/A')}")
        print(f"  Max iterations: {model_config.get('maxIter', 'N/A')}")
        print(f"  Regularization: {model_config.get('regParam', 'N/A')}")
        print(f"\n[DRY RUN] Would save to Redis key: {classifier_model_name}")
        if redis_expire is not None:
            print(f"[DRY RUN] With expiration: {redis_expire} seconds")
        return

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

Work Control Options:
  --dry-run             Preview what would be trained without saving
  --skip-existing       Skip if model already exists in Redis
  --force               Train even if model exists (overrides --skip-existing)

Environment Variables:
  DRY_RUN=1             Same as --dry-run
  SKIP_EXISTING=1       Same as --skip-existing
  FORCE=1               Same as --force
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
  SPARK_DRIVER_MEMORY   Spark driver memory (default: 4g)
  SPARK_EXECUTOR_MEMORY Spark executor memory (default: 4g)
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
        type=str,
        default=None,
        metavar='HH:MM:SS',
        help='Redis key expiration time (None = never expires, default: 172800)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model configurations and exit'
    )

    args, _ = parser.parse_known_args()

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

    # Connect to Redis (respects REDIS_TLS and REDIS_PASSWORD settings)
    print(f"Connecting to Redis at {config['redis_host']}:{config['redis_port']}...")
    try:
        redis_client = create_redis_client(decode_responses=False)  # Bytes for model serialization
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
            read_text_override=args.read_text or None,
            save_text_override=args.save_text,
            expire_override=args.expire,
            dry_run=config.get('dry_run', False),
            skip_existing=config.get('skip_existing', False),
            force=config.get('force', False),
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
