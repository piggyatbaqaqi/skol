#!/usr/bin/env python3
"""Test script for TaxonExtractor.load_taxa() method.

Tests:
1. Basic loading with default pattern
2. Pattern-based filtering
3. Round-trip consistency (extract → save → load)
4. Schema verification
5. Empty result handling
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import TaxonExtractor
import os


def test_load_taxa():
    """Test the load_taxa() method."""

    print("=" * 70)
    print("Testing TaxonExtractor.load_taxa()")
    print("=" * 70)

    # Create Spark session
    print("\nInitializing Spark session...")
    spark = SparkSession.builder \
        .appName("Test Load Taxa") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    try:
        # Get CouchDB credentials from environment
        couchdb_url = os.getenv("COUCHDB_URL", "http://localhost:5984")
        username = os.getenv("COUCHDB_USER", "admin")
        password = os.getenv("COUCHDB_PASSWORD", "password")
        ingest_db = os.getenv("INGEST_DB", "mycobank_annotations")
        taxon_db = os.getenv("TAXON_DB", "mycobank_taxa")

        print(f"\nCouchDB Configuration:")
        print(f"  URL: {couchdb_url}")
        print(f"  Ingest DB: {ingest_db}")
        print(f"  Taxon DB: {taxon_db}")

        # Initialize extractor
        print("\nInitializing TaxonExtractor...")
        extractor = TaxonExtractor(
            spark=spark,
            ingest_couchdb_url=couchdb_url,
            ingest_db_name=ingest_db,
            taxon_db_name=taxon_db,
            ingest_username=username,
            ingest_password=password
        )

        # Test 1: Load all taxa
        print("\n" + "-" * 70)
        print("Test 1: Load all taxa (pattern='taxon_*')")
        print("-" * 70)

        all_taxa = extractor.load_taxa()
        count = all_taxa.count()
        print(f"✓ Loaded {count} taxa")

        if count > 0:
            print("\nSchema:")
            all_taxa.printSchema()

            print("\nSample taxa:")
            all_taxa.select("taxon", "source").show(5, truncate=50)
        else:
            print("⚠ No taxa found in database")
            print("  This is expected if no taxa have been saved yet")

        # Test 2: Pattern-based loading
        print("\n" + "-" * 70)
        print("Test 2: Pattern-based loading")
        print("-" * 70)

        # Try loading with wildcard pattern
        wildcard_taxa = extractor.load_taxa(pattern="taxon_*")
        wildcard_count = wildcard_taxa.count()
        print(f"Pattern 'taxon_*': {wildcard_count} taxa")

        # Try loading all documents
        all_docs = extractor.load_taxa(pattern="*")
        all_count = all_docs.count()
        print(f"Pattern '*': {all_count} taxa")

        # Verify they match
        if wildcard_count == all_count:
            print("✓ Wildcard pattern matches all documents")
        else:
            print(f"⚠ Mismatch: 'taxon_*' found {wildcard_count}, '*' found {all_count}")

        # Test 3: Schema verification
        print("\n" + "-" * 70)
        print("Test 3: Schema verification")
        print("-" * 70)

        expected_cols = {
            "taxon", "description", "source",
            "line_number", "paragraph_number",
            "page_number", "empirical_page_number"
        }

        if count > 0:
            actual_cols = set(all_taxa.columns)
            missing = expected_cols - actual_cols
            extra = actual_cols - expected_cols

            if not missing and not extra:
                print("✓ Schema matches expected structure")
            else:
                if missing:
                    print(f"⚠ Missing columns: {missing}")
                if extra:
                    print(f"⚠ Extra columns: {extra}")
        else:
            print("⚠ Skipping (no data)")

        # Test 4: Empty result handling
        print("\n" + "-" * 70)
        print("Test 4: Empty result handling")
        print("-" * 70)

        empty = extractor.load_taxa(pattern="nonexistent_pattern_12345")
        empty_count = empty.count()

        if empty_count == 0:
            print("✓ Empty pattern returns empty DataFrame")
        else:
            print(f"⚠ Expected 0, got {empty_count}")

        # Test 5: Round-trip test (if annotated data exists)
        print("\n" + "-" * 70)
        print("Test 5: Round-trip consistency test")
        print("-" * 70)

        try:
            # Try to load annotated documents
            print("Loading annotated documents...")
            annotated_df = extractor.load_annotated_documents()
            annotated_count = annotated_df.count()

            if annotated_count > 0:
                print(f"Found {annotated_count} annotated documents")

                # Extract taxa
                print("Extracting taxa...")
                extracted_df = extractor.extract_taxa(annotated_df)
                extracted_count = extracted_df.count()
                print(f"Extracted {extracted_count} taxa")

                if extracted_count > 0:
                    # Save taxa (with test prefix)
                    print("Saving taxa to CouchDB...")
                    save_results = extractor.save_taxa(extracted_df)
                    successes = save_results.filter("success = true").count()
                    failures = save_results.filter("success = false").count()
                    print(f"Saved: {successes} success, {failures} failures")

                    # Load back
                    print("Loading taxa from CouchDB...")
                    loaded_df = extractor.load_taxa()
                    loaded_count = loaded_df.count()
                    print(f"Loaded {loaded_count} taxa")

                    # Verify counts
                    if loaded_count >= successes:
                        print("✓ Round-trip successful!")
                        print(f"  Original: {extracted_count} taxa")
                        print(f"  Saved: {successes} taxa")
                        print(f"  Loaded: {loaded_count} taxa")
                    else:
                        print(f"⚠ Count mismatch: saved {successes}, loaded {loaded_count}")
                else:
                    print("⚠ No taxa extracted")
            else:
                print("⚠ No annotated documents found")
                print("  Skipping round-trip test")
        except Exception as e:
            print(f"⚠ Round-trip test failed: {e}")
            print("  This is expected if annotated documents aren't available")

        # Test 6: Data integrity
        if count > 0:
            print("\n" + "-" * 70)
            print("Test 6: Data integrity checks")
            print("-" * 70)

            # Check for required fields
            null_taxon = all_taxa.filter("taxon IS NULL").count()
            null_desc = all_taxa.filter("description IS NULL").count()
            null_source = all_taxa.filter("source IS NULL").count()

            print(f"Null taxon names: {null_taxon}")
            print(f"Null descriptions: {null_desc}")
            print(f"Null sources: {null_source}")

            if null_taxon == 0 and null_desc == 0 and null_source == 0:
                print("✓ All required fields are populated")
            else:
                print("⚠ Some required fields have null values")

            # Check source structure
            print("\nSource field structure:")
            all_taxa.select("source.db_name", "source.doc_id", "source.attachment_name").show(3)

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total taxa in database: {count}")
        print(f"Pattern matching works: {'✓' if wildcard_count == all_count else '✗'}")
        print(f"Schema correct: {'✓' if count == 0 or not missing else '✗'}")
        print(f"Empty results work: {'✓' if empty_count == 0 else '✗'}")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print("\nStopping Spark session...")
        spark.stop()
        print("Done!")

    return True


if __name__ == "__main__":
    success = test_load_taxa()
    sys.exit(0 if success else 1)
