#!/usr/bin/env python3
"""
Example: Translating Taxa Descriptions to JSON

This script demonstrates how to use TaxaJSONTranslator to enrich taxa
descriptions with structured JSON features using a fine-tuned Mistral model.

Workflow:
1. Load taxa from CouchDB using TaxonExtractor
2. Initialize TaxaJSONTranslator with checkpoint
3. Translate descriptions to JSON
4. Validate and save results
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import TaxonExtractor
from taxa_json_translator import TaxaJSONTranslator


def main():
    """Run taxa translation example."""

    print("=" * 70)
    print("Taxa Description to JSON Translation Example")
    print("=" * 70)

    # Get configuration from environment
    couchdb_url = os.getenv("COUCHDB_URL", "http://localhost:5984")
    username = os.getenv("COUCHDB_USER", "admin")
    password = os.getenv("COUCHDB_PASSWORD", "password")
    ingest_db = os.getenv("INGEST_DB", "mycobank_annotations")
    taxon_db = os.getenv("TAXON_DB", "mycobank_taxa")
    checkpoint_path = os.getenv("MISTRAL_CHECKPOINT", None)

    # Initialize Spark
    print("\nInitializing Spark session...")
    spark = SparkSession.builder \
        .appName("Taxa JSON Translation Example") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    try:
        # Step 1: Load taxa from CouchDB
        print("\n" + "=" * 70)
        print("STEP 1: Loading Taxa from CouchDB")
        print("=" * 70)

        extractor = TaxonExtractor(
            spark=spark,
            ingest_couchdb_url=couchdb_url,
            ingest_db_name=ingest_db,
            taxon_db_name=taxon_db,
            ingest_username=username,
            ingest_password=password
        )

        taxa_df = extractor.load_taxa()
        count = taxa_df.count()

        print(f"\n✓ Loaded {count} taxa")

        if count == 0:
            print("\n⚠ No taxa found in database")
            print("  Run the extraction pipeline first to populate the database")
            return

        # Show sample taxa
        print("\nSample taxa:")
        taxa_df.select("taxon", "description").show(3, truncate=50)

        # Step 2: Initialize translator
        print("\n" + "=" * 70)
        print("STEP 2: Initializing TaxaJSONTranslator")
        print("=" * 70)

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"\nUsing fine-tuned checkpoint: {checkpoint_path}")
        elif checkpoint_path:
            print(f"\n⚠ Checkpoint not found: {checkpoint_path}")
            print("  Using base model instead")
            checkpoint_path = None
        else:
            print("\nNo checkpoint specified, using base model")
            print("  Set MISTRAL_CHECKPOINT env var to use fine-tuned model")

        translator = TaxaJSONTranslator(
            spark=spark,
            checkpoint_path=checkpoint_path,
            max_new_tokens=1024
        )

        # Step 3: Translate descriptions
        print("\n" + "=" * 70)
        print("STEP 3: Translating Descriptions to JSON")
        print("=" * 70)

        # Choose processing method based on dataset size
        if count <= 100:
            print("\nUsing standard processing (dataset is small)...")
            enriched_df = translator.translate_descriptions(taxa_df)
        else:
            print("\nUsing batch processing (dataset is large)...")
            batch_size = 10
            enriched_df = translator.translate_descriptions_batch(
                taxa_df,
                batch_size=batch_size
            )

        # Show sample results
        print("\nSample translations:")
        enriched_df.select(
            "taxon",
            "features_json"
        ).show(3, truncate=100)

        # Step 4: Validate JSON
        print("\n" + "=" * 70)
        print("STEP 4: Validating JSON Output")
        print("=" * 70)

        validated_df = translator.validate_json(enriched_df)

        # Show invalid entries if any
        invalid_df = validated_df.filter("json_valid = false")
        invalid_count = invalid_df.count()

        if invalid_count > 0:
            print(f"\n⚠ Found {invalid_count} invalid JSON entries:")
            invalid_df.select("taxon", "features_json").show(5, truncate=100)
        else:
            print("\n✓ All JSON entries are valid")

        # Step 5: Save results
        print("\n" + "=" * 70)
        print("STEP 5: Saving Results")
        print("=" * 70)

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Save all results
        all_output = output_dir / "taxa_with_features_all.parquet"
        translator.save_translations(
            validated_df,
            str(all_output),
            format="parquet"
        )

        # Save only valid entries
        valid_df = validated_df.filter("json_valid = true")
        valid_count = valid_df.count()

        if valid_count > 0:
            valid_output = output_dir / "taxa_with_features_valid.parquet"
            translator.save_translations(
                valid_df,
                str(valid_output),
                format="parquet"
            )
            print(f"\n✓ Saved {valid_count} valid entries to {valid_output}")

        # Step 6: Analysis
        print("\n" + "=" * 70)
        print("STEP 6: Sample Analysis")
        print("=" * 70)

        if valid_count > 0:
            # Show a complete example
            print("\nComplete example (first valid entry):")
            sample = valid_df.first()

            print(f"\nTaxon: {sample['taxon']}")
            print(f"\nDescription:")
            print(f"  {sample['description'][:200]}...")
            print(f"\nExtracted Features (JSON):")
            print(f"  {sample['features_json'][:500]}...")

            # Parse and display JSON structure
            import json
            try:
                features = json.loads(sample['features_json'])
                print(f"\nFeature categories: {list(features.keys())}")
                print(f"Total feature categories: {len(features)}")
            except:
                pass

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total taxa processed: {count}")
        print(f"Valid JSON outputs: {valid_count} ({100*valid_count/count:.1f}%)")
        print(f"Invalid JSON outputs: {invalid_count} ({100*invalid_count/count:.1f}%)")
        print(f"\nOutput saved to:")
        print(f"  All: {all_output}")
        if valid_count > 0:
            print(f"  Valid only: {valid_output}")

        print("\n" + "=" * 70)
        print("✓ Translation complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("\nStopping Spark session...")
        spark.stop()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
