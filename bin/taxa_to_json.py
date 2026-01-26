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

    # Recompute only records with invalid/missing JSON (after model update)
    python taxa_to_json.py --recompute-invalid --incremental
"""

import argparse
import os
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
    skip_existing: bool = False,
    doc_ids: Optional[list] = None,
    force: bool = False,
    recompute_invalid: bool = False,
    use_constrained_decoding: bool = False,
    schema_max_depth: int = 4,
    schema_min_depth: int = 2,
    use_ontology_context: bool = False,
    ontology_dir: Optional[str] = None,
    normalize_vocabulary: bool = False,
    normalization_threshold: float = 0.85,
    analyze_coverage: bool = False,
    graceful_degradation: bool = False,
    preserve_threshold: float = 0.5,
    include_annotations: bool = False,
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
        skip_existing: If True, skip records already in destination database
        doc_ids: If specified, only process these specific document IDs
        force: If True, process even if output exists (overrides skip_existing)
        recompute_invalid: If True, only process records with invalid JSON
        use_constrained_decoding: If True, use Outlines for guaranteed valid JSON
        schema_max_depth: Maximum nesting depth for constrained decoding (2-6)
        schema_min_depth: Minimum nesting depth for constrained decoding
        use_ontology_context: If True, inject PATO/FAO ontology context into prompts
        ontology_dir: Directory containing pato.obo and fao.obo files
        normalize_vocabulary: If True, apply threshold-gated vocabulary normalization
        normalization_threshold: Similarity threshold for normalization (0.0-1.0)
        analyze_coverage: If True, analyze and report vocabulary coverage
        graceful_degradation: If True, enable graceful degradation for novel terms
        preserve_threshold: Threshold below which to preserve original terms
        include_annotations: If True, include confidence annotations in output
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
    if skip_existing and not force:
        print(f"Mode: SKIP EXISTING (skip records already in destination)")
    if force:
        print(f"Mode: FORCE (process even if output exists)")
    if doc_ids:
        print(f"Processing specific doc_ids: {len(doc_ids)} documents")
    if recompute_invalid:
        print("Mode: RECOMPUTE INVALID (only process records with invalid JSON)")
    if use_constrained_decoding:
        print(f"Constrained decoding: ENABLED (depth {schema_min_depth}-{schema_max_depth})")
    if use_ontology_context:
        ont_dir = ontology_dir or os.environ.get('ONTOLOGY_DIR', '/opt/skol/data/ontologies')
        print(f"Ontology context: ENABLED (from {ont_dir})")
    if normalize_vocabulary:
        print(f"Vocabulary normalization: ENABLED (threshold={normalization_threshold})")
    if analyze_coverage:
        print(f"Coverage analysis: ENABLED")
    if graceful_degradation:
        print(f"Graceful degradation: ENABLED (preserve_threshold={preserve_threshold})")
        if include_annotations:
            print(f"  Include annotations: YES")
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
            load_in_4bit=True,
            use_constrained_decoding=use_constrained_decoding,
            schema_max_depth=schema_max_depth,
            schema_min_depth=schema_min_depth,
            use_ontology_context=use_ontology_context,
            ontology_dir=ontology_dir
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

        # Filter to specific doc_ids if provided
        if doc_ids:
            from pyspark.sql.functions import col
            taxa_df = taxa_df.filter(col("_id").isin(doc_ids))
            filtered_count = taxa_df.count()
            if verbosity >= 1:
                print(f"  Filtered to {len(doc_ids)} specified doc_ids: {filtered_count} found")
            loaded_count = filtered_count

            if loaded_count == 0:
                print(f"\n⚠ No taxa found for specified doc_ids")
                return

        # Filter to only invalid records if recompute_invalid is set
        if recompute_invalid:
            if verbosity >= 1:
                print(f"\nFinding records with invalid/missing JSON in {dest_db}...")

            import couchdb
            server = couchdb.Server(couchdb_url)
            if username and password:
                server.resource.credentials = (username, password)

            # Find doc_ids with invalid or missing json_annotated
            invalid_ids = set()
            if dest_db in server:
                dest_db_conn = server[dest_db]
                for doc_id in dest_db_conn:
                    if doc_id.startswith('_design/'):
                        continue
                    try:
                        doc = dest_db_conn[doc_id]
                        json_annotated = doc.get('json_annotated')
                        json_valid = doc.get('json_valid')
                        # Invalid if: no json_annotated, empty json_annotated, or explicitly marked invalid
                        if not json_annotated or json_valid == 'false':
                            invalid_ids.add(doc_id)
                    except Exception:
                        # If we can't read the doc, consider it invalid
                        invalid_ids.add(doc_id)

            if verbosity >= 1:
                print(f"  Found {len(invalid_ids)} records with invalid/missing JSON")

            if not invalid_ids:
                print(f"\n✓ No invalid records found in {dest_db}, nothing to recompute")
                return

            # Filter to only invalid records using Spark
            from pyspark.sql.functions import col
            invalid_ids_broadcast = spark.sparkContext.broadcast(invalid_ids)

            def in_invalid(doc_id):
                return doc_id in invalid_ids_broadcast.value

            from pyspark.sql.functions import udf
            from pyspark.sql.types import BooleanType
            in_invalid_udf = udf(in_invalid, BooleanType())

            taxa_df = taxa_df.filter(in_invalid_udf(col("_id")))
            filtered_count = taxa_df.count()

            if verbosity >= 1:
                print(f"  Filtered to {filtered_count} invalid records to recompute")

            loaded_count = filtered_count

            if loaded_count == 0:
                print(f"\n⚠ No matching taxa found in source for invalid records")
                return

        # Skip existing records if requested (unless --force)
        if skip_existing and not force:
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

    # Run coverage analysis if requested
    if analyze_coverage and not dry_run:
        run_coverage_analysis(
            couchdb_url=couchdb_url,
            username=username,
            password=password,
            dest_db=dest_db,
            ontology_dir=ontology_dir,
            threshold=normalization_threshold,
            limit=limit,
            verbosity=verbosity
        )

    # Run graceful degradation analysis if requested
    if graceful_degradation and not dry_run:
        run_graceful_degradation_analysis(
            couchdb_url=couchdb_url,
            username=username,
            password=password,
            dest_db=dest_db,
            ontology_dir=ontology_dir,
            normalization_threshold=normalization_threshold,
            preserve_threshold=preserve_threshold,
            limit=limit,
            verbosity=verbosity
        )


def run_coverage_analysis(
    couchdb_url: str,
    username: str,
    password: str,
    dest_db: str,
    ontology_dir: Optional[str] = None,
    threshold: float = 0.85,
    limit: Optional[int] = None,
    verbosity: int = 1
) -> None:
    """
    Analyze vocabulary coverage of extracted JSON features against ontologies.

    Args:
        couchdb_url: CouchDB server URL
        username: CouchDB username
        password: CouchDB password
        dest_db: Database containing extracted features
        ontology_dir: Directory containing ontology files
        threshold: Similarity threshold for "well-covered"
        limit: Maximum number of documents to analyze
        verbosity: Verbosity level
    """
    import json
    import couchdb

    from skol.ontology import (
        OntologyRegistry,
        VocabularyAnalyzer,
        print_coverage_report
    )

    print(f"\n{'='*70}")
    print("Vocabulary Coverage Analysis")
    print(f"{'='*70}")

    # Determine ontology directory
    if ontology_dir:
        ont_dir = Path(ontology_dir)
    elif os.environ.get('ONTOLOGY_DIR'):
        ont_dir = Path(os.environ['ONTOLOGY_DIR'])
    else:
        default_installed = Path('/opt/skol/data/ontologies')
        if default_installed.exists():
            ont_dir = default_installed
        else:
            ont_dir = Path(__file__).parent.parent / "data" / "ontologies"

    print(f"Ontology directory: {ont_dir}")

    # Check for ontology files
    pato_path = ont_dir / "pato.obo"
    fao_path = ont_dir / "fao.obo"

    if not pato_path.exists() and not fao_path.exists():
        print(f"✗ No ontology files found in {ont_dir}")
        print("  Coverage analysis requires at least one ontology file (pato.obo or fao.obo)")
        return

    # Load ontologies
    print("\nLoading ontologies...")
    registry = OntologyRegistry()

    if pato_path.exists():
        print(f"  Loading PATO from {pato_path}...")
        registry.register("pato", str(pato_path), category="base")
        print(f"  ✓ PATO: {len(registry.get('pato'))} terms")

    if fao_path.exists():
        print(f"  Loading FAO from {fao_path}...")
        registry.register("fao", str(fao_path), category="base")
        print(f"  ✓ FAO: {len(registry.get('fao'))} terms")

    # Create analyzer
    analyzer = VocabularyAnalyzer(registry, threshold=threshold)

    # Connect to CouchDB
    print(f"\nConnecting to {dest_db}...")
    server = couchdb.Server(couchdb_url)
    server.resource.credentials = (username, password)

    if dest_db not in server:
        print(f"✗ Database '{dest_db}' not found")
        return

    db = server[dest_db]

    # Fetch documents with json_annotated
    print("Fetching extracted features...")
    all_features = []
    count = 0

    for doc_id in db:
        if limit and count >= limit:
            break

        doc = db[doc_id]
        json_annotated = doc.get('json_annotated')

        if json_annotated and isinstance(json_annotated, dict):
            all_features.append(json_annotated)
            count += 1

        if count % 100 == 0 and verbosity >= 1:
            print(f"  Fetched {count} documents...")

    print(f"  ✓ Loaded {len(all_features)} documents with extracted features")

    if not all_features:
        print("✗ No documents with json_annotated found")
        return

    # Run analysis
    print("\nAnalyzing vocabulary coverage...")
    analysis = analyzer.analyze_batch(all_features)

    # Print report
    print()
    report = print_coverage_report(
        analysis,
        show_details=verbosity >= 1,
        max_details=30 if verbosity >= 2 else 15
    )
    print(report)

    # Recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS:")
    print("-" * 60)

    if analysis.coverage_ratio >= 0.9:
        print("✓ Coverage is excellent (≥90%). Vocabulary normalization optional.")
    elif analysis.coverage_ratio >= 0.7:
        print("~ Coverage is moderate (70-90%). Consider vocabulary normalization.")
        print("  Run with --normalize-vocabulary to apply threshold-gated normalization.")
    else:
        print("✗ Coverage is low (<70%). Vocabulary normalization recommended.")
        print("  Consider:")
        print("    1. Adding domain-specific ontologies (e.g., lichen anatomy)")
        print("    2. Running with --normalize-vocabulary --normalization-threshold 0.8")

    if analysis.novel > 0:
        print(f"\n  {analysis.novel} novel terms detected that may need:")
        print("    - Custom ontology additions")
        print("    - Manual review for domain-specific vocabulary")


def run_graceful_degradation_analysis(
    couchdb_url: str,
    username: str,
    password: str,
    dest_db: str,
    ontology_dir: Optional[str] = None,
    normalization_threshold: float = 0.85,
    preserve_threshold: float = 0.5,
    limit: Optional[int] = None,
    verbosity: int = 1
) -> None:
    """
    Analyze vocabulary with graceful degradation and generate detailed report.

    Args:
        couchdb_url: CouchDB server URL
        username: CouchDB username
        password: CouchDB password
        dest_db: Database containing extracted features
        ontology_dir: Directory containing ontology files
        normalization_threshold: Threshold for HIGH confidence
        preserve_threshold: Threshold below which to preserve originals
        limit: Maximum number of documents to analyze
        verbosity: Verbosity level
    """
    import json
    import couchdb

    from skol.ontology import (
        OntologyRegistry,
        RobustOntologyPipeline,
    )

    print(f"\n{'='*70}")
    print("Graceful Degradation Analysis")
    print(f"{'='*70}")

    # Determine ontology directory
    if ontology_dir:
        ont_dir = Path(ontology_dir)
    elif os.environ.get('ONTOLOGY_DIR'):
        ont_dir = Path(os.environ['ONTOLOGY_DIR'])
    else:
        default_installed = Path('/opt/skol/data/ontologies')
        if default_installed.exists():
            ont_dir = default_installed
        else:
            ont_dir = Path(__file__).parent.parent / "data" / "ontologies"

    print(f"Ontology directory: {ont_dir}")
    print(f"Normalization threshold: {normalization_threshold}")
    print(f"Preserve threshold: {preserve_threshold}")

    # Check for ontology files
    pato_path = ont_dir / "pato.obo"
    fao_path = ont_dir / "fao.obo"

    if not pato_path.exists() and not fao_path.exists():
        print(f"✗ No ontology files found in {ont_dir}")
        return

    # Load ontologies
    print("\nLoading ontologies...")
    registry = OntologyRegistry()

    if pato_path.exists():
        print(f"  Loading PATO from {pato_path}...")
        registry.register("pato", str(pato_path), category="base")
        print(f"  ✓ PATO: {len(registry.get('pato'))} terms")

    if fao_path.exists():
        print(f"  Loading FAO from {fao_path}...")
        registry.register("fao", str(fao_path), category="base")
        print(f"  ✓ FAO: {len(registry.get('fao'))} terms")

    # Create pipeline
    pipeline = RobustOntologyPipeline(
        registry,
        normalization_threshold=normalization_threshold,
        preserve_threshold=preserve_threshold
    )

    # Connect to CouchDB
    print(f"\nConnecting to {dest_db}...")
    server = couchdb.Server(couchdb_url)
    server.resource.credentials = (username, password)

    if dest_db not in server:
        print(f"✗ Database '{dest_db}' not found")
        return

    db = server[dest_db]

    # Fetch documents with json_annotated
    print("Fetching extracted features...")
    all_features = []
    count = 0

    for doc_id in db:
        if limit and count >= limit:
            break

        doc = db[doc_id]
        json_annotated = doc.get('json_annotated')

        if json_annotated and isinstance(json_annotated, dict):
            all_features.append(json_annotated)
            count += 1

    print(f"  ✓ Loaded {len(all_features)} documents with extracted features")

    if not all_features:
        print("✗ No documents with json_annotated found")
        return

    # Analyze each document and combine
    print("\nAnalyzing vocabulary with graceful degradation...")

    # Use the first document for detailed analysis
    if all_features:
        sample_report = pipeline.analyze_and_report(all_features[0])
        print()
        print(sample_report)

    # If multiple documents, show aggregate stats
    if len(all_features) > 1:
        print(f"\n{'='*60}")
        print(f"AGGREGATE ANALYSIS ({len(all_features)} documents)")
        print("=" * 60)

        from skol.ontology import VocabularyAnalyzer
        analyzer = VocabularyAnalyzer(registry)

        all_terms = []
        for features in all_features:
            all_terms.extend(analyzer.extract_terms(features))

        aggregate_summary = pipeline.degradation_handler.summarize(all_terms)
        print(f"Total terms across all documents: {aggregate_summary['total']}")
        print(f"  HIGH:   {aggregate_summary['high']['count']:5d} ({aggregate_summary['high']['percent']:.1f}%)")
        print(f"  MEDIUM: {aggregate_summary['medium']['count']:5d} ({aggregate_summary['medium']['percent']:.1f}%)")
        print(f"  LOW:    {aggregate_summary['low']['count']:5d} ({aggregate_summary['low']['percent']:.1f}%)")
        print(f"  NOVEL:  {aggregate_summary['novel']['count']:5d} ({aggregate_summary['novel']['percent']:.1f}%)")


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
Work Control Options (from env_config):
  --dry-run                  Preview what would be done without saving
  --skip-existing            Skip records already in destination database
  --force                    Process even if output exists
  --limit N                  Process at most N records
  --doc-id ID1,ID2,...       Process only specific document IDs
  --incremental              Save each record immediately (crash-resistant)
  --recompute-invalid        Only process records with invalid JSON
  --use-constrained-decoding Use Outlines for guaranteed valid JSON output

Environment Variables:
  DRY_RUN=1             Same as --dry-run
  SKIP_EXISTING=1       Same as --skip-existing
  FORCE=1               Same as --force
  LIMIT=N               Same as --limit N
  DOC_IDS=id1,id2,...   Same as --doc-id
  INCREMENTAL=1         Same as --incremental
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

  # Recompute only records with invalid/missing JSON (after model update)
  python taxa_to_json.py --recompute-invalid --incremental
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
        '--recompute-invalid',
        action='store_true',
        help='Only process records with invalid/missing JSON in destination'
    )

    parser.add_argument(
        '--use-constrained-decoding',
        action='store_true',
        help='Use Outlines for constrained JSON generation (guarantees valid JSON)'
    )

    parser.add_argument(
        '--schema-max-depth',
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 6],
        metavar='N',
        help='Maximum nesting depth for constrained decoding (2-6, default: 4)'
    )

    parser.add_argument(
        '--schema-min-depth',
        type=int,
        default=2,
        choices=[2, 3, 4, 5, 6],
        metavar='N',
        help='Minimum nesting depth for constrained decoding (2-6, default: 2)'
    )

    parser.add_argument(
        '--use-ontology-context',
        action='store_true',
        help='Inject PATO/FAO ontology context into prompts for guided vocabulary'
    )

    parser.add_argument(
        '--ontology-dir',
        type=str,
        default=None,
        metavar='PATH',
        help='Directory containing pato.obo and fao.obo (default: $ONTOLOGY_DIR or /opt/skol/data/ontologies/)'
    )

    parser.add_argument(
        '--normalize-vocabulary',
        action='store_true',
        help='Apply threshold-gated vocabulary normalization to extracted features'
    )

    parser.add_argument(
        '--normalization-threshold',
        type=float,
        default=0.85,
        metavar='THRESH',
        help='Similarity threshold for vocabulary normalization (0.0-1.0, default: 0.85)'
    )

    parser.add_argument(
        '--analyze-coverage',
        action='store_true',
        help='Analyze and report vocabulary coverage against ontologies (no normalization applied)'
    )

    parser.add_argument(
        '--graceful-degradation',
        action='store_true',
        help='Enable graceful degradation for novel terms (preserve originals with annotations)'
    )

    parser.add_argument(
        '--preserve-threshold',
        type=float,
        default=0.5,
        metavar='THRESH',
        help='Preserve original term if similarity below this threshold (0.0-1.0, default: 0.5)'
    )

    parser.add_argument(
        '--include-annotations',
        action='store_true',
        help='Include confidence annotations in output (requires --graceful-degradation)'
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

    # Merge work control options from command-line args and env_config
    # Command-line args take precedence over env_config
    dry_run = args.dry_run or config.get('dry_run', False)
    skip_existing = args.skip_existing or config.get('skip_existing', False)
    incremental = args.incremental or config.get('incremental', False)
    force = config.get('force', False)  # Only from env_config (no command-line arg)
    limit = args.limit if args.limit is not None else config.get('limit')
    doc_ids = config.get('doc_ids')  # Only from env_config (via --doc-id command line)
    recompute_invalid = args.recompute_invalid
    use_constrained_decoding = args.use_constrained_decoding
    schema_max_depth = args.schema_max_depth
    schema_min_depth = args.schema_min_depth
    use_ontology_context = args.use_ontology_context
    ontology_dir = args.ontology_dir
    normalize_vocabulary = args.normalize_vocabulary
    normalization_threshold = args.normalization_threshold
    analyze_coverage = args.analyze_coverage
    graceful_degradation = args.graceful_degradation
    preserve_threshold = args.preserve_threshold
    include_annotations = args.include_annotations

    # Validate schema depth parameters
    if schema_min_depth > schema_max_depth:
        print(f"Error: --schema-min-depth ({schema_min_depth}) cannot be greater "
              f"than --schema-max-depth ({schema_max_depth})")
        sys.exit(1)

    # Run translation
    try:
        translate_taxa_to_json(
            config=config,
            source_db=source_db,
            dest_db=dest_db,
            checkpoint_path=checkpoint_path,
            pattern=args.pattern,
            batch_size=args.batch_size,
            limit=limit,
            verbosity=verbosity,
            validate=not args.no_validate,
            dry_run=dry_run,
            incremental=incremental,
            skip_existing=skip_existing,
            doc_ids=doc_ids,
            force=force,
            recompute_invalid=recompute_invalid,
            use_constrained_decoding=use_constrained_decoding,
            schema_max_depth=schema_max_depth,
            schema_min_depth=schema_min_depth,
            use_ontology_context=use_ontology_context,
            ontology_dir=ontology_dir,
            normalize_vocabulary=normalize_vocabulary,
            normalization_threshold=normalization_threshold,
            analyze_coverage=analyze_coverage,
            graceful_degradation=graceful_degradation,
            preserve_threshold=preserve_threshold,
            include_annotations=include_annotations,
        )
    except KeyboardInterrupt:
        print("\n\n✗ Translation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Translation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
