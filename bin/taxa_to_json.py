#!/usr/bin/env python3
"""
Convert Taxa Descriptions to Structured JSON

This standalone program loads taxa descriptions from one CouchDB database,
translates them to structured JSON using a fine-tuned Mistral model, and
saves the enriched records to another CouchDB database.

Usage:
    python taxa_to_json.py [--source-db NAME] [--dest-db NAME] [--checkpoint PATH]
                           [--batch-size N] [--pattern PATTERN] [--limit N]
                           [--verbosity LEVEL] [--incremental]

Example:
    # Default: load from skol_taxa_dev, save to skol_taxa_full
    python taxa_to_json.py

    # Specify databases
    python taxa_to_json.py --source-db mycobank_taxa --dest-db mycobank_taxa_full

    # Process only specific taxa
    python taxa_to_json.py --pattern "taxon_abc*"

    # Process only first 5 records (useful for debugging)
    python taxa_to_json.py --limit 5

    # Use a specific model checkpoint
    python taxa_to_json.py --checkpoint ./mistral_checkpoints/checkpoint-100

    # Save each record as soon as it's translated (recommended for long jobs)
    python taxa_to_json.py --incremental

    # Skip records that already exist in destination (for cheap restarts)
    python taxa_to_json.py --skip-existing --incremental
"""

import argparse
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for skol modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Python 3.11+ compatibility: Apply formatargspec shim before importing ML libraries
import skol_compat  # noqa: F401 (imported for side effects)

from env_config import get_env_config


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_SOURCE_DB = 'skol_taxa_dev'
DEFAULT_DEST_DB = 'skol_taxa_full'
DEFAULT_BATCH_SIZE = 10
DEFAULT_PATTERN = '*'


# ============================================================================
# JSON Translation Functions
# ============================================================================

def translate_taxa_to_json(
    config: Dict[str, Any],
    source_db: str,
    dest_db: str,
    checkpoint_path: Optional[str] = None,
    pattern: str = '*',
    batch_size: int = 10,
    limit: Optional[int] = None,
    verbosity: int = 1,
    validate: bool = True,
    dry_run: bool = False,
    incremental: bool = False,
    skip_existing: bool = False
) -> None:
    """
    Load taxa from source database, translate to JSON, and save to destination.

    Args:
        config: Environment configuration
        source_db: Source CouchDB database name
        dest_db: Destination CouchDB database name
        checkpoint_path: Path to fine-tuned model checkpoint (optional)
        pattern: Pattern for document IDs to process (default: '*')
        batch_size: Number of descriptions to process per batch
        limit: Maximum number of records to process (None = all)
        verbosity: Verbosity level (0=silent, 1=info, 2=debug)
        validate: If True, validate JSON output
        dry_run: If True, show what would be done without actually saving
        incremental: If True, save each record as it's translated (recommended)
        skip_existing: If True, skip records that already exist in destination database
    """
    # Import here to avoid slow startup for --help
    from pyspark.sql import SparkSession
    from taxa_json_translator import TaxaJSONTranslator

    # Build CouchDB URL
    couchdb_url = config['couchdb_url']
    username = config['couchdb_username']
    password = config['couchdb_password']

    print(f"\n{'='*70}")
    print(f"Taxa to JSON Translation")
    print(f"{'='*70}")
    print(f"CouchDB: {couchdb_url}")
    print(f"Source database: {source_db}")
    print(f"Destination database: {dest_db}")
    print(f"Pattern: {pattern}")
    print(f"Batch size: {batch_size}")
    if limit:
        print(f"Limit: {limit} records")
    if checkpoint_path:
        print(f"Model checkpoint: {checkpoint_path}")
    else:
        print(f"Model checkpoint: None (using base Mistral model)")
    if dry_run:
        print(f"Mode: DRY RUN (no changes will be saved)")
    if incremental:
        print(f"Mode: INCREMENTAL (save each record immediately)")
    if skip_existing:
        print(f"Mode: SKIP EXISTING (skip records already in destination)")
    print()

    # Initialize Spark
    if verbosity >= 1:
        print("Initializing Spark...")

    spark = SparkSession.builder \
        .appName("Taxa JSON Translation") \
        .master(f"local[{config['cores']}]") \
        .config("spark.driver.memory", config['spark_driver_memory']) \
        .config("spark.executor.memory", config['spark_executor_memory']) \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.sql.broadcastTimeout", "600") \
        .getOrCreate()

    # Set log level
    if verbosity < 2:
        spark.sparkContext.setLogLevel("WARN")

    if verbosity >= 1:
        print("✓ Spark initialized")

    try:
        # Initialize translator
        if verbosity >= 1:
            print("\nInitializing TaxaJSONTranslator...")

        translator = TaxaJSONTranslator(
            spark=spark,
            couchdb_url=couchdb_url,
            username=username,
            password=password,
            checkpoint_path=checkpoint_path,
            device="cuda",
            load_in_4bit=True
        )

        # Load taxa from source database
        if verbosity >= 1:
            print(f"\nLoading taxa from {source_db}...")

        taxa_df = translator.load_taxa(db_name=source_db, pattern=pattern)
        loaded_count = taxa_df.count()

        if loaded_count == 0:
            print(f"\n⚠ No taxa found matching pattern '{pattern}' in {source_db}")
            return

        if verbosity >= 1:
            print(f"✓ Loaded {loaded_count} taxa")

        # Skip existing records if requested
        if skip_existing:
            if verbosity >= 1:
                print(f"\nChecking for existing records in {dest_db}...")

            import couchdb
            server = couchdb.Server(couchdb_url)
            if username and password:
                server.resource.credentials = (username, password)

            # Get existing doc IDs from destination database
            existing_ids = set()
            if dest_db in server:
                dest_db_conn = server[dest_db]
                for doc_id in dest_db_conn:
                    existing_ids.add(doc_id)

            if verbosity >= 1:
                print(f"  Found {len(existing_ids)} existing records in destination")

            # Filter out existing records using Spark
            from pyspark.sql.functions import col
            existing_ids_broadcast = spark.sparkContext.broadcast(existing_ids)

            def not_in_existing(doc_id):
                return doc_id not in existing_ids_broadcast.value

            from pyspark.sql.functions import udf
            from pyspark.sql.types import BooleanType
            not_existing_udf = udf(not_in_existing, BooleanType())

            taxa_df = taxa_df.filter(not_existing_udf(col("_id")))
            filtered_count = taxa_df.count()
            skipped_count = loaded_count - filtered_count

            if verbosity >= 1:
                print(f"  Skipping {skipped_count} existing records")
                print(f"  Processing {filtered_count} new records")

            loaded_count = filtered_count

            if loaded_count == 0:
                print(f"\n✓ All records already exist in {dest_db}, nothing to process")
                return

        # Apply limit if specified
        if limit and limit < loaded_count:
            if verbosity >= 1:
                print(f"  Limiting to {limit} records (from {loaded_count})")
            taxa_df = taxa_df.limit(limit)
            total_count = limit
        else:
            total_count = loaded_count

        # Show sample if verbose
        if verbosity >= 2:
            print("\nSample taxa:")
            sample_rows = taxa_df.select("_id", "taxon").limit(3).collect()
            for row in sample_rows:
                taxon_preview = row['taxon'][:80] + '...' if len(row['taxon']) > 80 else row['taxon']
                print(f"  {row['_id']}: {taxon_preview}")

        # Handle dry run
        if dry_run:
            print(f"\n[DRY RUN] Would process {total_count} taxa")
            print(f"[DRY RUN] Source: {source_db} -> Destination: {dest_db}")
            print(f"[DRY RUN] No changes have been made")
            return

        # Process based on mode
        if incremental:
            # Incremental mode: translate and save each record immediately
            if verbosity >= 1:
                print(f"\nTranslating and saving {total_count} descriptions incrementally...")

            results = translator.translate_and_save_streaming(
                taxa_df,
                db_name=dest_db,
                description_col="description",
                batch_size=batch_size,
                validate=validate,
                verbosity=verbosity
            )
            # Results summary is printed by translate_and_save_streaming

        else:
            # Batch mode: translate all, then save all
            if verbosity >= 1:
                print(f"\nTranslating {total_count} descriptions to JSON...")

            enriched_df = translator.translate_descriptions_batch(
                taxa_df,
                description_col="description",
                output_col="features_json",
                batch_size=batch_size
            )

            # Validate JSON if requested
            if validate:
                if verbosity >= 1:
                    print("\nValidating JSON output...")

                validated_df = translator.validate_json(enriched_df, json_col="features_json")

                # Show validation results
                valid_count = validated_df.filter("json_valid = 'true'").count()
                invalid_count = total_count - valid_count

                if invalid_count > 0:
                    print(f"⚠ {invalid_count} descriptions produced invalid JSON")
                    if verbosity >= 2:
                        print("\nInvalid entries:")
                        invalid_rows = validated_df.filter("json_valid = 'false'").select("_id", "taxon").limit(5).collect()
                        for row in invalid_rows:
                            print(f"  - {row['_id']}")
            else:
                validated_df = enriched_df

            # Save to destination database
            if verbosity >= 1:
                print(f"\nSaving enriched taxa to {dest_db}...")

            results_df = translator.save_taxa(
                validated_df,
                db_name=dest_db,
                json_annotated_col="features_json"
            )

            # Show save results
            success_count = results_df.filter("success = true").count()
            failure_count = total_count - success_count

            print(f"\n{'='*70}")
            print("Translation Complete!")
            print(f"{'='*70}")
            print(f"✓ Successfully processed: {success_count} taxa")
            if failure_count > 0:
                print(f"✗ Failed: {failure_count} taxa")

                if verbosity >= 2:
                    print("\nFailed entries:")
                    failed_rows = results_df.filter("success = false").select("doc_id", "error_message").limit(5).collect()
                    for row in failed_rows:
                        print(f"  - {row['doc_id']}: {row['error_message']}")

    except Exception as e:
        print(f"\n✗ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        if verbosity >= 1:
            print("\nStopping Spark...")
        spark.stop()
        if verbosity >= 1:
            print("✓ Spark stopped")


# ============================================================================
# Main Program
# ============================================================================

def main():
    """Main entry point for the taxa to JSON translation program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Convert taxa descriptions to structured JSON using a fine-tuned Mistral model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  COUCHDB_URL           CouchDB URL (default: http://localhost:5984)
  COUCHDB_HOST          CouchDB host (default: 127.0.0.1:5984)
  COUCHDB_USER          CouchDB username (default: admin)
  COUCHDB_PASSWORD      CouchDB password
  SOURCE_DB             Source taxa database (default: skol_taxa_dev)
  DEST_DB               Destination taxa database (default: skol_taxa_full)
  CHECKPOINT_PATH       Path to model checkpoint
  SPARK_CORES           Number of Spark cores (default: 4)
  SPARK_DRIVER_MEMORY   Spark driver memory (default: 4g)
  VERBOSITY             Verbosity level 0-2 (default: 1)

Examples:
  # Process all taxa with default settings
  python taxa_to_json.py

  # Process specific taxa pattern
  python taxa_to_json.py --pattern "taxon_abc*"

  # Use custom databases
  python taxa_to_json.py --source-db mycobank_taxa --dest-db mycobank_taxa_full

  # Dry run to see what would be processed
  python taxa_to_json.py --dry-run

  # Process only first 5 records for debugging
  python taxa_to_json.py --limit 5

  # Skip JSON validation
  python taxa_to_json.py --no-validate

  # Incremental mode: save each record immediately (recommended for long jobs)
  python taxa_to_json.py --incremental

  # Skip records already in destination (for cheap restarts)
  python taxa_to_json.py --skip-existing --incremental

  # Combination: debug with limit and incremental saving
  python taxa_to_json.py --limit 10 --incremental --verbosity 2
"""
    )

    parser.add_argument(
        '--source-db',
        type=str,
        default=None,
        metavar='NAME',
        help=f'Source CouchDB database name (default: {DEFAULT_SOURCE_DB})'
    )

    parser.add_argument(
        '--dest-db',
        type=str,
        default=None,
        metavar='NAME',
        help=f'Destination CouchDB database name (default: {DEFAULT_DEST_DB})'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to fine-tuned model checkpoint (default: use base model)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default=DEFAULT_PATTERN,
        metavar='PATTERN',
        help=f'Pattern for document IDs to process (default: {DEFAULT_PATTERN})'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar='N',
        help=f'Number of descriptions per batch (default: {DEFAULT_BATCH_SIZE})'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        metavar='N',
        help='Maximum number of records to process (default: all)'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip JSON validation step'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without saving changes'
    )

    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Save each record as it completes (recommended for long jobs, crash-resistant)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip records that already exist in destination database (for cheap restarts)'
    )

    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=None,
        metavar='LEVEL',
        help='Verbosity level: 0=silent, 1=info, 2=debug (default: 1)'
    )

    args, _ = parser.parse_known_args()

    # Get configuration from environment
    config = get_env_config()

    # Override config with command-line arguments
    source_db = args.source_db or config.get('source_db') or DEFAULT_SOURCE_DB
    dest_db = args.dest_db or config.get('dest_db') or DEFAULT_DEST_DB
    checkpoint_path = args.checkpoint or config.get('checkpoint_path')
    verbosity = args.verbosity if args.verbosity is not None else config['verbosity']

    # Run translation
    try:
        translate_taxa_to_json(
            config=config,
            source_db=source_db,
            dest_db=dest_db,
            checkpoint_path=checkpoint_path,
            pattern=args.pattern,
            batch_size=args.batch_size,
            limit=args.limit,
            verbosity=verbosity,
            validate=not args.no_validate,
            dry_run=args.dry_run,
            incremental=args.incremental,
            skip_existing=args.skip_existing
        )
    except KeyboardInterrupt:
        print("\n\n✗ Translation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Translation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
