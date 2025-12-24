#!/usr/bin/env python3
"""
Test access to skol_training database with SkolClassifierV2.

This verifies that the couchdb_training_database parameter works correctly.
"""

import os
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2


def main():
    """Test training database access."""
    print("=" * 70)
    print("Testing CouchDB Training Database Access")
    print("=" * 70)

    # Load environment variables
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

    couchdb_url = os.getenv('COUCHDB_URL', 'http://localhost:5984')
    username = os.getenv('COUCHDB_USER', 'admin')
    password = os.getenv('COUCHDB_PASSWORD', '')

    print(f"\nCouchDB URL: {couchdb_url}")
    print(f"Username: {username}")
    print(f"Main database: skol_dev")
    print(f"Training database: skol_training")

    # Create Spark session
    spark = SparkSession.builder \
        .appName("TestTrainingDB") \
        .master("local[2]") \
        .getOrCreate()

    print("\n✓ Spark session created")

    try:
        # Create classifier with training database
        print("\nCreating SkolClassifierV2 with couchdb_training_database...")

        classifier = SkolClassifierV2(
            spark=spark,
            input_source='couchdb',
            couchdb_url=couchdb_url,
            couchdb_database='skol_dev',  # For predictions
            couchdb_training_database='skol_training',  # For training
            couchdb_username=username,
            couchdb_password=password,
            couchdb_pattern='*.txt.ann',  # Pattern for training files
            extraction_mode='section',
            use_suffixes=True,
            output_dest='none',  # No output needed for this test
            verbosity=2
        )

        print("✓ Classifier created successfully")

        # Try to load training data (annotated data)
        print("\nLoading annotated training data from skol_training database...")

        df = classifier._load_annotated_data()

        count = df.count()

        print(f"\n✓ Successfully loaded {count} annotated training samples from skol_training!")

        # Check schema
        print(f"\nDataFrame schema columns: {df.columns}")

        # Verify section_name column exists
        if 'section_name' in df.columns:
            print("✓ section_name column present")
        else:
            print("✗ section_name column missing!")

        # Show a few samples
        print("\nSample data:")
        df.select("value", "label", "section_name").show(5, truncate=50)

        # Check labels distribution
        print("\nLabel distribution:")
        df.groupBy("label").count().orderBy("count", ascending=False).show(10)

        # Check section names distribution
        print("\nSection name distribution:")
        df.groupBy("section_name").count().orderBy("count", ascending=False).show(10)

        print("\n" + "=" * 70)
        print("✅ Training database access test PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("❌ Training database access test FAILED!")
        print("=" * 70)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
