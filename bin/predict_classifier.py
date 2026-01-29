#!/usr/bin/env python3
"""
Predict Labels Using SKOL Classifier from Redis

This standalone program loads a trained classifier from Redis, applies it to
documents in CouchDB, and saves the predictions back to CouchDB as .ann files.

Before applying the ML model, documents are pre-filtered by checking for common
taxonomy abbreviations (sp., var., gen., etc.). Documents are marked with a
'taxonomy' field in CouchDB to speed up future processing.

Usage:
    python predict_classifier.py [--model MODEL_NAME] [--verbosity LEVEL]
                                [--read-text] [--save-text {eager,lazy}]
                                [--couchdb-pattern PATTERN] [--prediction-batch-size SIZE]

Example:
    python predict_classifier.py --model logistic_sections --verbosity 2
    python predict_classifier.py --couchdb-pattern "*.pdf" --prediction-batch-size 96

    # Skip documents that already have .ann attachments
    python predict_classifier.py --skip-existing

    # Process only specific documents
    python predict_classifier.py --doc-id doc1,doc2,doc3

    # Preview what would be done without saving
    python predict_classifier.py --dry-run

    # Process at most 10 documents
    python predict_classifier.py --limit 10

Environment Variables:
    TAXONOMY_ABBREVS - Comma-separated list of taxonomy abbreviations to check
                       (default: comb.,fam.,gen.,ined.,var.,subg.,subsp.,sp.,f.,nov.,spec.,ssp.)
    SKIP_TAXONOMY_CHECK - Set to 'true' to disable taxonomy abbreviation checking entirely
                          (all documents will be treated as having taxonomy abbreviations)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import redis
from pyspark.sql import SparkSession

# Add parent directory to path for skol modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skol_classifier.classifier_v2 import SkolClassifierV2
from env_config import get_env_config
from ingestors import RateLimitedHttpClient

import couchdb
from io import BytesIO


# ============================================================================
# PDF Re-download Functions
# ============================================================================

def redownload_pdf_from_url(
    doc_id: str,
    db: couchdb.Database,
    verbosity: int = 1,
    http_client: Optional[RateLimitedHttpClient] = None
) -> bool:
    """
    Re-download the PDF for a document from its pdf_url field.

    This deletes any existing article.pdf and article.txt and article.txt.ann attachments,
    then downloads a fresh copy from the pdf_url.

    Args:
        doc_id: Document ID in CouchDB
        db: CouchDB database instance
        verbosity: Verbosity level
        http_client: Optional RateLimitedHttpClient for rate-limited downloads

    Returns:
        True if PDF was successfully re-downloaded, False otherwise
    """
    try:
        doc = db[doc_id]
    except couchdb.ResourceNotFound:
        if verbosity >= 1:
            print(f"    Document {doc_id} not found")
        return False

    pdf_url = doc.get('pdf_url')
    if not pdf_url:
        if verbosity >= 1:
            print(f"    Document {doc_id} has no pdf_url field")
        return False

    if verbosity >= 2:
        print(f"    Re-downloading PDF from: {pdf_url}")

    # Delete existing attachments
    attachments = doc.get('_attachments', {})
    attachments_to_delete = []
    for att_name in list(attachments.keys()):
        if att_name in ('article.pdf', 'article.txt', 'article.txt.ann'):
            attachments_to_delete.append(att_name)

    for att_name in attachments_to_delete:
        try:
            db.delete_attachment(doc, att_name)
            # Refresh doc after deletion
            doc = db[doc_id]
            if verbosity >= 2:
                print(f"    Deleted existing {att_name}")
        except Exception as e:
            if verbosity >= 2:
                print(f"    Failed to delete {att_name}: {e}")

    # Helper to save download error to document
    def save_download_error(error_msg: str):
        """Save download error to document's download_error field."""
        try:
            # Refresh doc to get latest revision
            fresh_doc = db[doc_id]
            fresh_doc['download_error'] = error_msg
            db.save(fresh_doc)
        except Exception:
            pass  # Best effort

    # Download fresh PDF
    try:
        # Use http_client if provided, otherwise create a simple one
        if http_client is not None:
            response = http_client.get(pdf_url, stream=False)
        else:
            # Create a simple client for one-off downloads
            client = RateLimitedHttpClient(
                user_agent='SKOL-Classifier/1.0',
                verbosity=verbosity,
                timeout=60
            )
            response = client.get(pdf_url, stream=False)

        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}"
            if verbosity >= 1:
                print(f"    Failed to download PDF: {error_msg}")
            save_download_error(error_msg)
            return False

        pdf_content = response.content
        if len(pdf_content) < 100:
            error_msg = f"PDF too small ({len(pdf_content)} bytes)"
            if verbosity >= 1:
                print(f"    Downloaded PDF too small ({len(pdf_content)} bytes)")
            save_download_error(error_msg)
            return False

        # Validate PDF magic bytes - must start with %PDF
        if not pdf_content.startswith(b'%PDF'):
            preview = pdf_content[:20].hex() if len(pdf_content) >= 20 else pdf_content.hex()
            error_msg = f"Invalid PDF (missing %PDF header, starts with: {preview[:40]})"
            if verbosity >= 1:
                print(f"    Invalid PDF (not %PDF): starts with {preview[:40]}")
            save_download_error(error_msg)
            return False

        # Save new PDF attachment
        db.put_attachment(
            doc,
            BytesIO(pdf_content),
            'article.pdf',
            'application/pdf'
        )

        # Clear any previous download error on success
        try:
            fresh_doc = db[doc_id]
            if 'download_error' in fresh_doc:
                del fresh_doc['download_error']
                db.save(fresh_doc)
        except Exception:
            pass  # Best effort

        if verbosity >= 1:
            print(f"    ✓ Re-downloaded PDF ({len(pdf_content)} bytes)")
        return True

    except Exception as e:
        error_msg = str(e)
        if verbosity >= 1:
            print(f"    Failed to download PDF: {error_msg}")
        save_download_error(error_msg)
        return False


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
        .config("spark.driver.memory", config['spark_driver_memory']) \
        .config("spark.executor.memory", config['spark_executor_memory']) \
        .getOrCreate()


def mark_taxonomy_documents(
    doc_ids: List[str],
    config: Dict[str, Any],
    verbosity: int = 1
) -> Dict[str, bool]:
    """
    Check documents for taxonomy abbreviations and mark them in CouchDB.

    This performs a fast pre-filter by searching for common taxonomy abbreviations
    (like 'sp.', 'var.', 'gen.', etc.) in document text. Documents are marked with
    a 'taxonomy' field (true/false) in CouchDB.

    Args:
        doc_ids: List of document IDs to check
        config: Environment configuration
        verbosity: Verbosity level

    Returns:
        Dictionary mapping doc_id to taxonomy flag (True if contains abbreviations)
    """
    import os
    import couchdb
    import re

    # Check if taxonomy checking is disabled
    if os.environ.get('SKIP_TAXONOMY_CHECK', '').lower() in ('1', 'true', 'yes'):
        if verbosity >= 1:
            print(f"\n⚠ Taxonomy abbreviation checking DISABLED (SKIP_TAXONOMY_CHECK=true)")
            print(f"  Treating all {len(doc_ids)} documents as having taxonomy abbreviations")
        # Return all documents as having taxonomy (so none are filtered out)
        return {doc_id: True for doc_id in doc_ids}

    abbrevs = config['taxonomy_abbrevs']

    # Compile regex pattern for efficient single-pass checking
    # Remove trailing periods and escape special regex characters
    abbrev_parts = [re.escape(abbrev) for abbrev in abbrevs]
    # Create pattern that matches any abbreviation followed by a period
    # Use word boundary at start and end to avoid partial matches
    abbrev_pattern = re.compile(r'\b(' + '|'.join(abbrev_parts) + r')\b')

    if verbosity >= 1:
        print(f"\nChecking {len(doc_ids)} documents for taxonomy abbreviations...")
        if verbosity >= 2:
            print(f"  Abbreviations: {', '.join(abbrevs)}")

    # Connect to CouchDB
    couch_server = couchdb.Server(config['couchdb_url'])
    if config['couchdb_username'] and config['couchdb_password']:
        couch_server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])
    db = couch_server[config['ingest_db_name']]

    taxonomy_flags = {}
    marked_count = 0

    for doc_id in doc_ids:
        try:
            doc = db[doc_id]

            # Get text content from attachments
            text_content = ""
            attachments = doc.get('_attachments', {})

            # Check .txt attachment first (if exists)
            for att_name in attachments.keys():
                if att_name.endswith('.txt'):
                    try:
                        att_content = db.get_attachment(doc_id, att_name)
                        if att_content:
                            text_content = att_content.decode('utf-8', errors='ignore')
                            break
                    except Exception:
                        continue

            # If no .txt, we'll skip for now (PDF extraction is expensive)
            # The taxonomy flag will be set during prediction if needed
            if not text_content:
                continue

            # Check for taxonomy abbreviations (single pass with regex)
            has_taxonomy = bool(abbrev_pattern.search(text_content))
            taxonomy_flags[doc_id] = has_taxonomy

            # Update document if flag changed
            if 'taxonomy' not in doc or doc['taxonomy'] != has_taxonomy:
                doc['taxonomy'] = has_taxonomy
                db.save(doc)
                marked_count += 1

        except Exception as e:
            if verbosity >= 2:
                print(f"  Error checking {doc_id}: {e}")
            continue

    if verbosity >= 1:
        taxonomy_count = sum(1 for v in taxonomy_flags.values() if v)
        print(f"  Found {taxonomy_count} documents with taxonomy abbreviations")
        print(f"  Updated {marked_count} documents with taxonomy flag")

    return taxonomy_flags


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_and_save(
    model_name: str,
    model_config: Dict[str, Any],
    config: Dict[str, Any],
    redis_client: redis.Redis,
    read_text_override: bool = False,
    save_text_override: str = None,
    dry_run: bool = False,
    skip_existing: bool = False,
    force: bool = False,
    incremental: bool = False,
    incremental_batch_size: int = 50,
    taxonomy_filter: bool = False,
    limit: Optional[int] = None,
    doc_ids: Optional[List[str]] = None,
    retry_failed_extraction: bool = False,
) -> None:
    """
    Load classifier from Redis, make predictions, and save to CouchDB.

    When incremental=True, processes documents in batches and saves after each
    batch completes. This prevents OOM errors and allows resumption after crashes.

    Args:
        model_name: Name of the model configuration to use
        model_config: Model configuration dictionary
        config: Environment configuration
        redis_client: Redis client instance
        read_text_override: Optional read_text parameter override
        save_text_override: Optional save_text parameter override
        dry_run: If True, preview without saving changes
        skip_existing: If True, skip documents that already have .ann attachments
        force: If True, process even if output exists (overrides skip_existing)
        incremental: If True, process in batches and save after each batch
        incremental_batch_size: Number of documents per batch when incremental=True
        taxonomy_filter: If True, filter to only documents with taxonomy abbreviations
        limit: If set, process at most this many documents
        doc_ids: If set, only process these specific document IDs
        retry_failed_extraction: If True, re-download PDF and retry on extraction failure
    """
    # Apply overrides and use config verbosity
    model_config = model_config.copy()

    # Use verbosity from config (command-line or environment)
    model_config['verbosity'] = config['verbosity']

    if read_text_override:
        model_config['read_text'] = read_text_override
    if save_text_override is not None:
        model_config['save_text'] = save_text_override

    # Get batch size and pattern from config (can be overridden via command-line)
    batch_size = config['prediction_batch_size']
    pattern = config['couchdb_pattern']

    model_config['prediction_batch_size'] = batch_size
    model_config['num_workers'] = config['num_workers']
    model_config['union_batch_size'] = config['union_batch_size']
    model_config['incremental'] = incremental

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
    if dry_run:
        print(f"Mode: DRY RUN (no changes will be saved)")
    if skip_existing and not force:
        print(f"Mode: SKIP EXISTING (skip documents with .ann attachments)")
    if force:
        print(f"Mode: FORCE (process all, ignore existing)")
    if incremental:
        print(f"Mode: INCREMENTAL (save each .ann immediately)")
    if taxonomy_filter:
        print(f"Mode: TAXONOMY FILTER (only docs with taxonomy abbreviations)")
    if limit:
        print(f"Limit: {limit} documents")
    if doc_ids:
        print(f"Document IDs: {', '.join(doc_ids[:5])}{'...' if len(doc_ids) > 5 else ''}")
    if retry_failed_extraction:
        print(f"Mode: RETRY FAILED (re-download PDF and retry on extraction failure)")
    print()

    # Check if model exists in Redis
    if not redis_client.exists(classifier_model_name):
        print(f"✗ Model not found in Redis: {classifier_model_name}")
        print("  Please train the model first using bin/train_classifier.")
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

        # For section mode, we need to filter doc_ids BEFORE loading to avoid extracting all PDFs
        # Discover and filter doc_ids if needed
        filtered_doc_ids = doc_ids  # Start with user-specified doc_ids (if any)

        if not filtered_doc_ids or skip_existing or limit:
            # Need to discover/filter doc_ids before loading
            import couchdb
            couch_server = couchdb.Server(couchdb_url)
            if config['couchdb_username'] and config['couchdb_password']:
                couch_server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])
            db = couch_server[config['ingest_db_name']]

            # If no doc_ids specified, discover all PDFs
            if not filtered_doc_ids:
                if model_config.get('verbosity', 1) >= 1:
                    print("\nDiscovering PDF documents in database...")
                all_doc_ids = []
                for doc_id in db:
                    try:
                        doc = db[doc_id]
                        attachments = doc.get('_attachments', {})
                        if any(att.endswith('.pdf') for att in attachments.keys()):
                            all_doc_ids.append(doc_id)
                    except Exception:
                        continue
                filtered_doc_ids = all_doc_ids
                if model_config.get('verbosity', 1) >= 1:
                    print(f"  Found {len(filtered_doc_ids)} documents with PDF attachments")

            # Skip existing documents with .ann attachments (unless --force)
            if skip_existing and not force:
                if model_config.get('verbosity', 1) >= 1:
                    print(f"\nFiltering out documents with existing .ann attachments...")
                existing_ann_docs = set()
                for doc_id in db:
                    try:
                        doc = db[doc_id]
                        attachments = doc.get('_attachments', {})
                        if any(att.endswith('.ann') for att in attachments.keys()):
                            existing_ann_docs.add(doc_id)
                    except Exception:
                        continue

                original_count = len(filtered_doc_ids)
                filtered_doc_ids = [d for d in filtered_doc_ids if d not in existing_ann_docs]

                if model_config.get('verbosity', 1) >= 1:
                    skipped = original_count - len(filtered_doc_ids)
                    print(f"  Skipping {skipped} documents with existing .ann attachments")
                    print(f"  Remaining: {len(filtered_doc_ids)} documents to process")

            # Apply limit if specified
            if limit and len(filtered_doc_ids) > limit:
                if model_config.get('verbosity', 1) >= 1:
                    print(f"\nLimiting to {limit} documents (from {len(filtered_doc_ids)})")
                filtered_doc_ids = filtered_doc_ids[:limit]

        # Check if we have any documents to process
        if not filtered_doc_ids:
            print("\n⚠ No documents found matching criteria. Nothing to predict.")
            return

        # Taxonomy filtering (only when --taxonomy-filter is set)
        taxonomy_flags = {}
        if taxonomy_filter and not force:
            # Mark documents with taxonomy flag based on abbreviation presence
            taxonomy_flags = mark_taxonomy_documents(
                filtered_doc_ids,
                config,
                verbosity=model_config.get('verbosity', 1)
            )

            # Filter out documents without taxonomy abbreviations
            original_count = len(filtered_doc_ids)
            # Keep only documents that have taxonomy abbreviations OR don't have .txt yet (will check after PDF extraction)
            filtered_doc_ids = [
                doc_id for doc_id in filtered_doc_ids
                if taxonomy_flags.get(doc_id, True)  # True means not checked yet (no .txt)
            ]

            if model_config.get('verbosity', 1) >= 1:
                skipped = original_count - len(filtered_doc_ids)
                if skipped > 0:
                    print(f"\nFiltering out documents without taxonomy abbreviations...")
                    print(f"  Skipping {skipped} documents without taxonomy markers")
                    print(f"  Remaining: {len(filtered_doc_ids)} documents to process")

            # Check if we have any documents to process after taxonomy filtering
            if not filtered_doc_ids:
                print("\n⚠ No documents with taxonomy abbreviations found. Nothing to predict.")
                print("  Remove --taxonomy-filter to process all documents.")
                return

        total_docs = len(filtered_doc_ids)
        verbosity = model_config.get('verbosity', 1)

        # Process in batches when incremental mode is enabled
        if incremental:
            print(f"\n{'='*70}")
            print(f"INCREMENTAL MODE: Processing {total_docs} documents in batches of {incremental_batch_size}")
            print(f"{'='*70}")

            total_processed = 0
            total_saved = 0
            total_errors = 0
            total_retried = 0
            batch_num = 0

            # Track documents that have already been retried (to prevent infinite loops)
            retried_doc_ids = set()

            # Connect to CouchDB and create HTTP client for retry operations
            couch_db = None
            http_client = None
            if retry_failed_extraction:
                couch_server = couchdb.Server(couchdb_url)
                if config['couchdb_username'] and config['couchdb_password']:
                    couch_server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])
                couch_db = couch_server[config['ingest_db_name']]
                # Create shared HTTP client for rate-limited downloads
                http_client = RateLimitedHttpClient(
                    user_agent='SKOL-Classifier/1.0',
                    verbosity=verbosity,
                    timeout=60
                )

            # Split filtered_doc_ids into batches
            for batch_start in range(0, total_docs, incremental_batch_size):
                batch_num += 1
                batch_doc_ids = filtered_doc_ids[batch_start:batch_start + incremental_batch_size]
                batch_size_actual = len(batch_doc_ids)

                print(f"\n--- Batch {batch_num}: documents {batch_start + 1}-{batch_start + batch_size_actual} of {total_docs} ---")

                try:
                    # Create classifier for this batch only
                    batch_classifier = SkolClassifierV2(
                        spark=spark,
                        input_source='couchdb',
                        couchdb_url=couchdb_url,
                        couchdb_database=config['ingest_db_name'],
                        couchdb_username=config['couchdb_username'],
                        couchdb_password=config['couchdb_password'],
                        couchdb_pattern=pattern,
                        couchdb_doc_ids=batch_doc_ids,
                        output_dest='couchdb',
                        output_couchdb_suffix='.ann',
                        auto_load_model=True,
                        model_storage='redis',
                        redis_client=redis_client,
                        redis_key=classifier_model_name,
                        **model_config
                    )

                    # Load raw data for this batch
                    if verbosity >= 1:
                        print(f"  Loading {batch_size_actual} documents...")
                    batch_raw_df = batch_classifier.load_raw()

                    # Check for failed extractions and retry if enabled
                    if retry_failed_extraction and couch_db is not None:
                        # Get list of successfully extracted doc_ids
                        extracted_doc_ids = set(
                            row.doc_id for row in batch_raw_df.select("doc_id").distinct().collect()
                        )
                        # Find documents that failed extraction
                        failed_doc_ids = [
                            doc_id for doc_id in batch_doc_ids
                            if doc_id not in extracted_doc_ids and doc_id not in retried_doc_ids
                        ]

                        if failed_doc_ids:
                            if verbosity >= 1:
                                print(f"  Found {len(failed_doc_ids)} documents with failed extraction, attempting retry...")

                            # Re-download PDFs for failed documents
                            redownloaded_doc_ids = []
                            for doc_id in failed_doc_ids:
                                retried_doc_ids.add(doc_id)  # Mark as retried
                                if verbosity >= 2:
                                    print(f"  Retrying {doc_id}...")
                                if redownload_pdf_from_url(doc_id, couch_db, verbosity, http_client):
                                    redownloaded_doc_ids.append(doc_id)

                            # Retry extraction for successfully re-downloaded documents
                            if redownloaded_doc_ids:
                                if verbosity >= 1:
                                    print(f"  Re-extracting {len(redownloaded_doc_ids)} documents...")

                                # Create a new classifier just for the retry documents
                                retry_classifier = SkolClassifierV2(
                                    spark=spark,
                                    input_source='couchdb',
                                    couchdb_url=couchdb_url,
                                    couchdb_database=config['ingest_db_name'],
                                    couchdb_username=config['couchdb_username'],
                                    couchdb_password=config['couchdb_password'],
                                    couchdb_pattern=pattern,
                                    couchdb_doc_ids=redownloaded_doc_ids,
                                    output_dest='couchdb',
                                    output_couchdb_suffix='.ann',
                                    auto_load_model=True,
                                    model_storage='redis',
                                    redis_client=redis_client,
                                    redis_key=classifier_model_name,
                                    **model_config
                                )

                                try:
                                    retry_raw_df = retry_classifier.load_raw()
                                    retry_count = retry_raw_df.select("doc_id").distinct().count()
                                    if retry_count > 0:
                                        if verbosity >= 1:
                                            print(f"  ✓ Successfully re-extracted {retry_count} documents")
                                        # Union retry results with main batch
                                        batch_raw_df = batch_raw_df.union(retry_raw_df)
                                        total_retried += retry_count
                                    else:
                                        if verbosity >= 1:
                                            print(f"  ✗ Retry extraction yielded no documents")
                                except Exception as retry_e:
                                    if verbosity >= 1:
                                        print(f"  ✗ Retry extraction failed: {retry_e}")

                    # Re-check taxonomy for documents that weren't checked before (only if taxonomy_filter)
                    if taxonomy_filter and not force:
                        docs_to_recheck = [doc_id for doc_id in batch_doc_ids if doc_id not in taxonomy_flags]
                        if docs_to_recheck:
                            new_taxonomy_flags = mark_taxonomy_documents(
                                docs_to_recheck, config, verbosity=0  # Quiet for batches
                            )
                            taxonomy_flags.update(new_taxonomy_flags)

                            # Filter out documents without taxonomy markers
                            docs_without_taxonomy = [
                                doc_id for doc_id in batch_doc_ids
                                if taxonomy_flags.get(doc_id, False) is False
                            ]
                            if docs_without_taxonomy:
                                from pyspark.sql.functions import col
                                batch_raw_df = batch_raw_df.filter(~col("doc_id").isin(docs_without_taxonomy))
                                if verbosity >= 2:
                                    print(f"  Filtered out {len(docs_without_taxonomy)} docs without taxonomy")

                    # Make predictions for this batch
                    if verbosity >= 1:
                        print(f"  Making predictions...")
                    batch_predictions = batch_classifier.predict(batch_raw_df)

                    # Save immediately (unless dry_run)
                    if dry_run:
                        if verbosity >= 1:
                            print(f"  [DRY RUN] Would save predictions")
                    else:
                        if verbosity >= 1:
                            print(f"  Saving to CouchDB...")
                        batch_classifier.save_annotated(batch_predictions)

                    total_processed += batch_size_actual
                    total_saved += batch_size_actual
                    print(f"  ✓ Batch {batch_num} complete ({total_processed}/{total_docs} total)")

                    # Explicitly unpersist to free memory
                    batch_raw_df.unpersist()
                    batch_predictions.unpersist()

                except Exception as e:
                    total_errors += batch_size_actual
                    print(f"  ✗ Batch {batch_num} failed: {e}")
                    if verbosity >= 2:
                        import traceback
                        traceback.print_exc()
                    continue

            # Final summary for incremental mode
            print(f"\n{'='*70}")
            print("Incremental Processing Complete!" + (" (DRY RUN)" if dry_run else ""))
            print(f"{'='*70}")
            print(f"  Total batches: {batch_num}")
            print(f"  Documents processed: {total_processed}")
            print(f"  Documents saved: {total_saved if not dry_run else 0}")
            if retry_failed_extraction:
                print(f"  Documents retried: {total_retried}")
            print(f"  Errors: {total_errors}")
            print(f"  Database: {config['ingest_db_name']}")

        else:
            # Non-incremental mode: process all at once (original behavior)
            if verbosity >= 1:
                print(f"\nProcessing {total_docs} documents (non-incremental mode)")
                print("  TIP: Use --incremental for crash-resistant batch processing")

            classifier = SkolClassifierV2(
                spark=spark,
                input_source='couchdb',
                couchdb_url=couchdb_url,
                couchdb_database=config['ingest_db_name'],
                couchdb_username=config['couchdb_username'],
                couchdb_password=config['couchdb_password'],
                couchdb_pattern=pattern,
                couchdb_doc_ids=filtered_doc_ids,
                output_dest='couchdb',
                output_couchdb_suffix='.ann',
                auto_load_model=True,
                model_storage='redis',
                redis_client=redis_client,
                redis_key=classifier_model_name,
                **model_config
            )

            # Load raw data
            print("\nLoading documents from CouchDB...")
            raw_df = classifier.load_raw()

            # Check for failed extractions and retry if enabled
            if retry_failed_extraction:
                # Connect to CouchDB for retry operations
                couch_server = couchdb.Server(couchdb_url)
                if config['couchdb_username'] and config['couchdb_password']:
                    couch_server.resource.credentials = (
                        config['couchdb_username'],
                        config['couchdb_password']
                    )
                couch_db = couch_server[config['ingest_db_name']]

                # Create shared HTTP client for rate-limited downloads
                http_client = RateLimitedHttpClient(
                    user_agent='SKOL-Classifier/1.0',
                    verbosity=verbosity,
                    timeout=60
                )

                # Get list of successfully extracted doc_ids
                extracted_doc_ids = set(
                    row.doc_id for row in
                    raw_df.select("doc_id").distinct().collect()
                )
                # Find documents that failed extraction
                failed_doc_ids = [
                    doc_id for doc_id in filtered_doc_ids
                    if doc_id not in extracted_doc_ids
                ]

                if failed_doc_ids:
                    if verbosity >= 1:
                        print(f"\nFound {len(failed_doc_ids)} documents "
                              "with failed extraction, attempting retry...")

                    # Re-download PDFs for failed documents
                    redownloaded_doc_ids = []
                    for doc_id in failed_doc_ids:
                        if verbosity >= 2:
                            print(f"  Retrying {doc_id}...")
                        if redownload_pdf_from_url(doc_id, couch_db, verbosity, http_client):
                            redownloaded_doc_ids.append(doc_id)

                    # Retry extraction for successfully re-downloaded documents
                    if redownloaded_doc_ids:
                        if verbosity >= 1:
                            print(f"  Re-extracting "
                                  f"{len(redownloaded_doc_ids)} documents...")

                        # Create a new classifier just for the retry documents
                        retry_classifier = SkolClassifierV2(
                            spark=spark,
                            input_source='couchdb',
                            couchdb_url=couchdb_url,
                            couchdb_database=config['ingest_db_name'],
                            couchdb_username=config['couchdb_username'],
                            couchdb_password=config['couchdb_password'],
                            couchdb_pattern=pattern,
                            couchdb_doc_ids=redownloaded_doc_ids,
                            output_dest='couchdb',
                            output_couchdb_suffix='.ann',
                            auto_load_model=True,
                            model_storage='redis',
                            redis_client=redis_client,
                            redis_key=classifier_model_name,
                            **model_config
                        )

                        try:
                            retry_raw_df = retry_classifier.load_raw()
                            retry_count = (
                                retry_raw_df.select("doc_id").distinct().count()
                            )
                            if retry_count > 0:
                                if verbosity >= 1:
                                    print(f"  ✓ Successfully re-extracted "
                                          f"{retry_count} documents")
                                # Union retry results with main batch
                                raw_df = raw_df.union(retry_raw_df)
                        except Exception as retry_e:
                            if verbosity >= 1:
                                print(f"  ✗ Retry extraction failed: {retry_e}")

            # Re-check taxonomy for documents that now have .txt (after PDF extraction)
            # Only if taxonomy_filter is enabled
            if taxonomy_filter and not force:
                docs_to_recheck = [doc_id for doc_id in filtered_doc_ids if doc_id not in taxonomy_flags]

                if docs_to_recheck and verbosity >= 1:
                    print(f"\nRe-checking {len(docs_to_recheck)} documents for taxonomy abbreviations...")

                if docs_to_recheck:
                    new_taxonomy_flags = mark_taxonomy_documents(
                        docs_to_recheck, config, verbosity=verbosity
                    )
                    taxonomy_flags.update(new_taxonomy_flags)

                    docs_without_taxonomy = [
                        doc_id for doc_id in filtered_doc_ids
                        if taxonomy_flags.get(doc_id, False) is False
                    ]

                    if docs_without_taxonomy:
                        if verbosity >= 1:
                            print(f"  Filtering out {len(docs_without_taxonomy)} documents without taxonomy markers")
                        from pyspark.sql.functions import col
                        raw_df = raw_df.filter(~col("doc_id").isin(docs_without_taxonomy))
                        filtered_doc_ids = [d for d in filtered_doc_ids if d not in docs_without_taxonomy]

            # Count if verbose
            if verbosity >= 2:
                doc_count = raw_df.count()
                print(f"✓ {doc_count} documents to process")
                if doc_count == 0:
                    print("\n⚠ No documents found matching criteria. Nothing to predict.")
                    return
            else:
                print("✓ Documents loaded (use --verbosity 2 to see count)")

            # Show sample
            if verbosity >= 2:
                print("\nSample documents:")
                raw_df.show(5, truncate=50)

            # Make predictions
            print("\nMaking predictions...")
            predictions = classifier.predict(raw_df)

            # Show sample predictions
            if verbosity >= 1:
                print("\nSample predictions:")
                predictions.select(
                    "doc_id", "line_number", "attachment_name", "predicted_label", "value"
                ).show(5, truncate=50)

            # Save results
            if dry_run:
                print("\n[DRY RUN] Would save predictions to CouchDB as .ann attachments")
                if verbosity >= 1:
                    print("\n[DRY RUN] Sample predictions that would be saved:")
                    predictions.select(
                        "doc_id", "line_number", "attachment_name", "predicted_label", "value"
                    ).show(10, truncate=50)
            else:
                print("\nSaving predictions to CouchDB as .ann attachments...")
                classifier.save_annotated(predictions)

            print(f"\n{'='*70}")
            print("Prediction Complete!" + (" (DRY RUN)" if dry_run else ""))
            print(f"{'='*70}")
            if dry_run:
                print(f"[DRY RUN] Would save predictions to CouchDB")
            else:
                print(f"✓ Predictions saved to CouchDB")
            print(f"  Database: {config['ingest_db_name']}")
            if verbosity >= 2:
                print(f"  Documents processed: {doc_count}")

    finally:
        # Clean up Spark session
        spark.stop()
        print("\nSpark session stopped.")


# ============================================================================
# Main Program
# ============================================================================

def update_taxonomy_flags_only(
    config: Dict[str, Any],
    dry_run: bool = False,
    limit: Optional[int] = None,
    doc_ids: Optional[List[str]] = None,
) -> None:
    """
    Update taxonomy flags for all documents without running predictions.

    This scans all documents with article.txt attachments and sets the
    'taxonomy' field based on whether they contain taxonomy abbreviations.

    Args:
        config: Environment configuration
        dry_run: If True, preview without saving changes
        limit: If set, process at most this many documents
        doc_ids: If set, only process these specific document IDs
    """
    import couchdb
    import re

    verbosity = config.get('verbosity', 1)
    abbrevs = config['taxonomy_abbrevs']

    print(f"\n{'='*70}")
    print("Update Taxonomy Flags")
    print(f"{'='*70}")
    print(f"CouchDB: {config['couchdb_url']}")
    print(f"Database: {config['ingest_db_name']}")
    print(f"Abbreviations: {', '.join(abbrevs[:10])}{'...' if len(abbrevs) > 10 else ''}")
    if dry_run:
        print("Mode: DRY RUN (no changes will be saved)")
    if limit:
        print(f"Limit: {limit} documents")
    if doc_ids:
        print(f"Document IDs: {', '.join(doc_ids[:5])}{'...' if len(doc_ids) > 5 else ''}")
    print()

    # Compile regex pattern for efficient single-pass checking
    abbrev_parts = [re.escape(abbrev.rstrip('.')) for abbrev in abbrevs]
    abbrev_pattern = re.compile(r'\b(' + '|'.join(abbrev_parts) + r')\.')

    # Connect to CouchDB
    couch_server = couchdb.Server(config['couchdb_url'])
    if config['couchdb_username'] and config['couchdb_password']:
        couch_server.resource.credentials = (config['couchdb_username'], config['couchdb_password'])

    if config['ingest_db_name'] not in couch_server:
        print(f"Error: Database '{config['ingest_db_name']}' not found")
        sys.exit(1)

    db = couch_server[config['ingest_db_name']]

    # Get list of document IDs to process
    if doc_ids:
        all_doc_ids = doc_ids
    else:
        print("Discovering documents in database...")
        all_doc_ids = [doc_id for doc_id in db if not doc_id.startswith('_design/')]
        print(f"  Found {len(all_doc_ids)} documents")

    # Apply limit
    if limit and len(all_doc_ids) > limit:
        print(f"  Limiting to {limit} documents")
        all_doc_ids = all_doc_ids[:limit]

    # Process documents
    print(f"\nScanning {len(all_doc_ids)} documents for taxonomy abbreviations...")

    total_docs = 0
    docs_with_txt = 0
    already_correct = 0
    needs_true = 0
    needs_false = 0
    errors = 0

    docs_to_update = []

    for doc_id in all_doc_ids:
        total_docs += 1
        if verbosity >= 1 and total_docs % 1000 == 0:
            print(f"  Scanned {total_docs} documents...")

        try:
            doc = db[doc_id]
            current_value = doc.get('taxonomy')

            # Check if document has article.txt attachment
            attachments = doc.get('_attachments', {})
            if 'article.txt' not in attachments:
                # No article.txt - skip (we can't determine taxonomy without text)
                continue

            docs_with_txt += 1

            # Fetch article.txt content
            try:
                article_content = db.get_attachment(doc_id, 'article.txt')
                if article_content is None:
                    continue

                if hasattr(article_content, 'read'):
                    text = article_content.read()
                    if isinstance(text, bytes):
                        text = text.decode('utf-8', errors='ignore')
                else:
                    text = str(article_content)

            except Exception as e:
                if verbosity >= 2:
                    print(f"  Error reading article.txt for {doc_id}: {e}")
                errors += 1
                continue

            # Check for taxonomy abbreviations
            has_taxonomy = bool(abbrev_pattern.search(text))

            if has_taxonomy and current_value is not True:
                needs_true += 1
                doc['taxonomy'] = True
                docs_to_update.append(doc)
                if verbosity >= 2:
                    print(f"  {doc_id}: will set taxonomy=True")
            elif not has_taxonomy and current_value is True:
                needs_false += 1
                doc['taxonomy'] = False
                docs_to_update.append(doc)
                if verbosity >= 2:
                    print(f"  {doc_id}: will set taxonomy=False")
            else:
                already_correct += 1

        except Exception as e:
            if verbosity >= 2:
                print(f"  Error reading {doc_id}: {e}")
            errors += 1
            continue

    print()
    print(f"  Total documents scanned: {total_docs}")
    print(f"  Documents with article.txt: {docs_with_txt}")
    print(f"  Already correct: {already_correct}")
    print(f"  Need to set True: {needs_true}")
    print(f"  Need to set False: {needs_false}")
    print(f"  Errors: {errors}")
    print(f"  Total to update: {len(docs_to_update)}")
    print()

    if not docs_to_update:
        print("No updates needed. All documents have correct taxonomy values.")
        return

    # Apply updates
    if dry_run:
        print("DRY RUN - No changes will be saved")
        if verbosity >= 1:
            print("  Documents that would be updated:")
            for doc in docs_to_update[:20]:
                print(f"    {doc['_id']}: taxonomy -> {doc['taxonomy']}")
            if len(docs_to_update) > 20:
                print(f"    ... and {len(docs_to_update) - 20} more")
    else:
        batch_size = 500
        print(f"Updating {len(docs_to_update)} documents in batches of {batch_size}...")
        success_count = 0
        error_count = 0

        for i in range(0, len(docs_to_update), batch_size):
            batch = docs_to_update[i:i + batch_size]
            try:
                results = db.update(batch)
                batch_success = sum(1 for ok, _, _ in results if ok)
                batch_errors = len(batch) - batch_success
                success_count += batch_success
                error_count += batch_errors

                if verbosity >= 1:
                    print(f"  Batch {i // batch_size + 1}: {batch_success} updated, {batch_errors} errors")

            except Exception as e:
                error_count += len(batch)
                print(f"  Batch {i // batch_size + 1}: Error - {e}")

        print()
        print(f"Successfully updated: {success_count}")
        print(f"Errors: {error_count}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Documents set to taxonomy=True: {needs_true}")
    print(f"Documents set to taxonomy=False: {needs_false}")
    print(f"Documents updated: {len(docs_to_update) if not dry_run else 0}")
    if dry_run:
        print("\nThis was a dry run. No changes were saved.")
        print("Remove --dry-run to actually update the documents.")
    print()


def main():
    """Main entry point for the prediction program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Predict labels using SKOL classifier from Redis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  """ + '\n  '.join([f"{k}: {v.get('name', k)}" for k, v in MODEL_CONFIGS.items()]) + """

Work Control Options:
  --dry-run               Preview what would be done without saving changes
  --skip-existing         Skip documents that already have .ann attachments
  --force                 Process even if .ann exists (overrides --skip-existing)
  --incremental           Process in batches, saving after each (crash-resistant)
  --incremental-batch-size N
                          Documents per batch when --incremental is set (default: 50)
  --taxonomy-filter       Only process documents with taxonomy abbreviations
  --retry-failed-extraction
                          Re-download PDF and retry on extraction failure (once per doc)
  --limit N               Process at most N documents
  --doc-id ID[,ID,...]    Process only specific document ID(s), comma-separated

Taxonomy-Only Mode:
  --update-taxonomy-flag-only
                          Only update taxonomy flags without running predictions.
                          Scans article.txt attachments for taxonomy abbreviations
                          and sets the 'taxonomy' field accordingly.

Configuration (via environment variables or command-line arguments):
  --couchdb-host          CouchDB host (default: 127.0.0.1:5984)
  --couchdb-username      CouchDB username (default: admin)
  --couchdb-password      CouchDB password (default: SU2orange!)
  --ingest-db-name        Database to predict on (default: skol_dev)
  --redis-host            Redis host (default: localhost)
  --redis-port            Redis port (default: 6379)
  --model-version         Model version tag (default: v2.0)
  --couchdb-pattern       File pattern to match (default: *.txt)
  --prediction-batch-size Batch size for predictions (default: 24)
  --num-workers           Number of workers (default: 4)
  --cores                 Number of Spark cores (default: 4)
  --spark-driver-memory   Spark driver memory (default: 4g)
  --spark-executor-memory Spark executor memory (default: 4g)

Environment Variables for Work Control:
  DRY_RUN=1               Same as --dry-run
  SKIP_EXISTING=1         Same as --skip-existing
  FORCE=1                 Same as --force
  INCREMENTAL=1           Same as --incremental
  TAXONOMY_FILTER=1       Same as --taxonomy-filter
  LIMIT=N                 Same as --limit N
  DOC_IDS=id1,id2,...     Same as --doc-id

Note: Command-line arguments override environment variables.
      Use --couchdb-pattern instead of --pattern
      Use --prediction-batch-size instead of --batch-size
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
        '--list-models',
        action='store_true',
        help='List available model configurations and exit'
    )

    parser.add_argument(
        '--update-taxonomy-flag-only',
        action='store_true',
        help='Only update taxonomy flags without running predictions'
    )

    parser.add_argument(
        '--retry-failed-extraction',
        action='store_true',
        help='Re-download PDF from pdf_url and retry on extraction failure'
    )

    args, _ = parser.parse_known_args()

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

    # Handle --update-taxonomy-flag-only mode
    if args.update_taxonomy_flag_only:
        try:
            update_taxonomy_flags_only(
                config=config,
                dry_run=config.get('dry_run', False),
                limit=config.get('limit'),
                doc_ids=config.get('doc_ids'),
            )
        except KeyboardInterrupt:
            print("\n\n✗ Operation interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n✗ Operation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

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
            read_text_override=args.read_text,
            save_text_override=args.save_text,
            dry_run=config.get('dry_run', False),
            skip_existing=config.get('skip_existing', False),
            force=config.get('force', False),
            incremental=config.get('incremental', False),
            incremental_batch_size=config.get('incremental_batch_size', 50),
            taxonomy_filter=config.get('taxonomy_filter', False),
            limit=config.get('limit'),
            doc_ids=config.get('doc_ids'),
            retry_failed_extraction=args.retry_failed_extraction,
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
