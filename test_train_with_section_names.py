#!/usr/bin/env python3
"""
Test that SkolClassifierV2 can train on skol_training data with section features.
"""

import os
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2


def main():
    """Test full training workflow with section names."""
    print("=" * 70)
    print("Testing Classifier Training with Section Features")
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
    print(f"Training database: skol_training")

    # Create Spark session
    spark = SparkSession.builder \
        .appName("TestTrainingSectionNames") \
        .master("local[4]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print("\n✓ Spark session created")

    try:
        # Create classifier with training database and section features
        print("\nCreating SkolClassifierV2...")

        classifier = SkolClassifierV2(
            spark=spark,
            input_source='couchdb',
            couchdb_url=couchdb_url,
            couchdb_database='skol_dev',
            couchdb_training_database='skol_training',
            couchdb_username=username,
            couchdb_password=password,
            couchdb_pattern='*.txt.ann',
            extraction_mode='section',  # Use section tokenizer
            use_suffixes=True,
            section_name_vocab_size=50,
            word_vocab_size=800,
            suffix_vocab_size=200,
            output_dest='none',
            verbosity=2
        )

        print("✓ Classifier created")

        # Train on a subset of data (to speed up test)
        print("\nLoading training data...")
        annotated_df = classifier._load_annotated_data()

        # Take a sample for faster testing
        sample_size = 5000
        print(f"Sampling {sample_size} records for quick test...")
        sample_df = annotated_df.limit(sample_size)

        print(f"✓ Loaded {sample_df.count()} training samples")

        # Verify schema
        print(f"\nColumns: {sample_df.columns}")
        assert 'section_name' in sample_df.columns, "section_name missing!"

        # Train the model
        print("\nTraining classifier with section features...")
        results = classifier.fit(sample_df)

        print("\n" + "=" * 70)
        print("Training Results:")
        print("=" * 70)
        print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"  Train samples: {results.get('train_count', 0)}")
        print(f"  Test samples: {results.get('test_count', 0)}")

        if 'f1_score' in results:
            print(f"  F1 Score: {results['f1_score']:.4f}")

        if 'class_metrics' in results:
            print("\nPer-class metrics:")
            for label, metrics in results['class_metrics'].items():
                print(f"  {label}:")
                print(f"    Precision: {metrics.get('precision', 0):.4f}")
                print(f"    Recall: {metrics.get('recall', 0):.4f}")
                print(f"    F1: {metrics.get('f1', 0):.4f}")

        print("\n" + "=" * 70)
        print("✅ Training with section features SUCCESSFUL!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("❌ Training test FAILED!")
        print("=" * 70)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
