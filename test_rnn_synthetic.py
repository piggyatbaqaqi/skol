#!/usr/bin/env python3
"""
Quick test for RNN model using synthetic data.

This script creates a small synthetic dataset and trains an RNN model on it,
allowing for rapid iteration when debugging rnn_model.py.

Usage:
    python test_rnn_synthetic.py [--verbosity LEVEL]

Options:
    --verbosity LEVEL    Set verbosity (0-3), default 2
"""

import sys
import os
import time
import argparse

# CRITICAL: Force CPU-only mode BEFORE any imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

# Now import everything
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import random

print("=" * 70)
print("RNN Model Synthetic Data Test")
print("=" * 70)
print()
print("Environment:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"  Python: {sys.version.split()[0]}")
print()

# Parse arguments
parser = argparse.ArgumentParser(description='Test RNN model with synthetic data')
parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2, 3, 4, 5],
                    help='Verbosity level (0-5)')
parser.add_argument('--num-docs', type=int, default=10,
                    help='Number of documents to generate')
parser.add_argument('--lines-per-doc', type=int, default=20,
                    help='Lines per document')
parser.add_argument('--epochs', type=int, default=2,
                    help='Training epochs')
parser.add_argument('--window-size', type=int, default=10,
                    help='RNN window size')
args = parser.parse_args()

print(f"Configuration:")
print(f"  Verbosity: {args.verbosity}")
print(f"  Documents: {args.num_docs}")
print(f"  Lines per doc: {args.lines_per_doc}")
print(f"  Epochs: {args.epochs}")
print(f"  Window size: {args.window_size}")
print()

# Create Spark session
print("-" * 70)
print("Initializing Spark...")
print("-" * 70)

spark = SparkSession.builder \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+PrintFlagsFinal -XX:+UseContainerSupport -Dio.netty.tryReflectionSetAccessible=true")\
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
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

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

print(f"✓ Spark version: {spark.version}")
print()

# Generate synthetic data
print("-" * 70)
print("Generating synthetic data...")
print("-" * 70)

# Define 3 simple patterns for classification
LABELS = ["Misc-exposition", "Description", "Nomenclature"]

# Word patterns for each label
PATTERNS = {
    "Misc-exposition": ["this", "is", "about", "generally", "overall", "introduction"],
    "Description": ["describes", "shows", "defines", "represents", "indicates"],
    "Nomenclature": ["name", "called", "term", "symbol", "notation", "denoted"]
}

def generate_line(label):
    """Generate a synthetic line with words from the label's pattern."""
    words = PATTERNS[label]
    # Mix label-specific words with some random words
    line_words = random.choices(words, k=random.randint(3, 6))
    random_words = ["the", "a", "of", "and", "to", "in"]
    line_words.extend(random.choices(random_words, k=random.randint(2, 4)))
    random.shuffle(line_words)
    return " ".join(line_words)

def generate_document(doc_id, num_lines):
    """Generate a synthetic document with mixed labels."""
    lines = []
    labels = []
    for i in range(num_lines):
        # Mix of labels with some bias
        if i < num_lines // 3:
            label = "Misc-exposition"
        elif i < 2 * num_lines // 3:
            label = "Description"
        else:
            label = "Nomenclature"

        # Add some randomness
        if random.random() < 0.2:
            label = random.choice(LABELS)

        lines.append(generate_line(label))
        labels.append(label)

    return doc_id, lines, labels

# Generate documents
print(f"Generating {args.num_docs} documents with {args.lines_per_doc} lines each...")
documents = []
for i in range(args.num_docs):
    doc_id = f"doc_{i:03d}.txt"
    filename, lines, labels = generate_document(doc_id, args.lines_per_doc)
    for line, label in zip(lines, labels):
        documents.append((filename, line, label))

print(f"✓ Generated {len(documents)} total lines")
print(f"  Label distribution:")
for label in LABELS:
    count = sum(1 for _, _, l in documents if l == label)
    pct = 100.0 * count / len(documents)
    print(f"    {label}: {count} ({pct:.1f}%)")
print()

# Create DataFrame
# Note: The FeatureExtractor expects a column named "value" by default
schema = StructType([
    StructField("filename", StringType(), False),
    StructField("value", StringType(), False),  # Changed from "line" to "value"
    StructField("label", StringType(), False)
])

df = spark.createDataFrame(documents, schema)
print(f"✓ Created DataFrame with {df.count()} rows")
print()

if args.verbosity >= 3:
    print("Sample data:")
    df.show(10, truncate=60)
    print()

# Initialize classifier
print("-" * 70)
print("Initializing RNN Classifier...")
print("-" * 70)

sys.path.insert(0, os.path.dirname(__file__))
from skol_classifier import SkolClassifierV2

# Model parameters optimized for speed
model_params = {
    'input_size': 100,      # Small vocabulary size
    'hidden_size': 32,      # Small hidden size
    'num_layers': 1,        # Single layer
    'num_classes': 3,       # 3 labels
    'dropout': 0.2,
    'window_size': args.window_size,
    'batch_size': 8,        # Small batches
    'epochs': args.epochs,
    'num_workers': 2,
    'verbosity': args.verbosity
}

print(f"Model parameters:")
for key, value in model_params.items():
    print(f"  {key}: {value}")
print()

classifier = SkolClassifierV2(
    spark=spark,
    input_source='dataframe',
    output_dest=None,  # No output needed for test
    model_storage=None,  # No model saving needed
    model_type='rnn',
    use_suffixes=True,
    min_doc_freq=1,  # Low threshold for small dataset
    **model_params  # Spread the dict instead of passing as model_params=
)

print("✓ Classifier initialized")
print()

# Train model
print("=" * 70)
print("TRAINING")
print("=" * 70)
print()

try:
    start_time = time.time()
    results = classifier.fit(annotated_data=df)
    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Training time: {elapsed:.2f} seconds")
    print()
    print("Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

    # Test model save and load with Redis
    print("=" * 70)
    print("TESTING MODEL SAVE/LOAD (Redis)")
    print("=" * 70)
    print()

    # Split data for testing
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Setup Redis client
    import redis
    redis_key = "test_rnn_model_synthetic"
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        redis_client.ping()
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"⚠ WARNING: Could not connect to Redis: {e}")
        print("Skipping save/load test")
        redis_client = None

    if redis_client:
        # Configure the classifier for Redis storage
        classifier.model_storage = 'redis'
        classifier.redis_client = redis_client
        classifier.redis_key = redis_key

        # Save model to Redis
        print(f"Saving model to Redis key: {redis_key}")
        classifier.save_model()
        print("✓ Model saved to Redis")
        print()

        # Add a dummy 'value' column to test_df for prediction compatibility
        # The RNN model doesn't use this column, but the output formatter expects it
        from pyspark.sql.functions import lit
        test_df_with_value = test_df.withColumn("value", lit(""))

        # Apply feature pipeline to test data
        test_df_featured = classifier._feature_pipeline.transform(test_df)

        # Make predictions with original model
        # Use internal _model.predict and then add decoded labels manually
        print("Making predictions with ORIGINAL model...")
        original_predictions_raw = classifier._model.predict(test_df_featured)
        original_predictions = classifier._decode_predictions(original_predictions_raw)
        original_predictions = original_predictions.select("doc_id", "line_num", "prediction", "predicted_label")
        original_count = original_predictions.count()
        print(f"✓ Original model predictions: {original_count} rows")

        # Collect a few predictions for comparison
        original_sample = original_predictions.limit(10).collect()
        if original_count == 0:
            print("⚠ WARNING: Original model produced 0 predictions!")
        print()

        # Create a new classifier instance and load the model from Redis
        print("Loading model into NEW classifier instance from Redis...")
        classifier_loaded = SkolClassifierV2(
            spark=spark,
            input_source='dataframe',
            output_dest=None,
            model_storage='redis',
            redis_client=redis_client,
            redis_key=redis_key,
            model_type='rnn',
            use_suffixes=True,
            min_doc_freq=1,
            **model_params
        )
        classifier_loaded.load_model()
        print("✓ Model loaded from Redis")
        print(f"  DEBUG: Loaded model input_size: {classifier_loaded._model.input_size}")
        print(f"  DEBUG: Loaded model window_size: {classifier_loaded._model.window_size}")
        print()

        # Apply the LOADED classifier's feature pipeline to test data
        # (cannot reuse test_df_featured from original classifier)
        test_df_featured_loaded = classifier_loaded._feature_pipeline.transform(test_df)
        print(f"  DEBUG: Featured loaded data count: {test_df_featured_loaded.count()}")
        print(f"  DEBUG: Featured loaded columns: {test_df_featured_loaded.columns}")

        # Make predictions with loaded model
        print("Making predictions with LOADED model...")
        loaded_predictions_raw = classifier_loaded._model.predict(test_df_featured_loaded)
        print(f"  DEBUG: Raw predictions count: {loaded_predictions_raw.count()}")
        print(f"  DEBUG: Raw predictions columns: {loaded_predictions_raw.columns}")

        # Show a sample of raw predictions
        if loaded_predictions_raw.count() > 0:
            print("  DEBUG: Sample raw predictions:")
            loaded_predictions_raw.show(5, truncate=False)

        loaded_predictions = classifier_loaded._decode_predictions(loaded_predictions_raw)
        print(f"  DEBUG: After decode count: {loaded_predictions.count()}")

        loaded_predictions = loaded_predictions.select("doc_id", "line_num", "prediction", "predicted_label")
        loaded_count = loaded_predictions.count()
        print(f"✓ Loaded model predictions: {loaded_count} rows")

        # Collect a few predictions for comparison
        loaded_sample = loaded_predictions.limit(10).collect()
        if loaded_count == 0:
            print("⚠ WARNING: Loaded model produced 0 predictions!")
        print()

        # Compare predictions
        print("Comparing predictions...")
        print("-" * 70)
        print(f"{'Doc ID':<15} {'Line#':<7} {'Original':<12} {'Loaded':<12} {'Match':<10}")
        print("-" * 70)

        mismatches = 0
        for i in range(min(len(original_sample), len(loaded_sample))):
            orig = original_sample[i]
            load = loaded_sample[i]
            match = "✓" if orig.prediction == load.prediction else "✗"
            if orig.prediction != load.prediction:
                mismatches += 1
            print(f"{orig.doc_id:<15} {orig.line_num:<7} {orig.prediction:<12.1f} {load.prediction:<12.1f} {match:<10}")

        print("-" * 70)
        print()

        if mismatches == 0:
            print("✓✓✓ ALL PREDICTIONS MATCH ✓✓✓")
            print("The saved and loaded model produces identical predictions!")
        else:
            print(f"⚠ WARNING: {mismatches} predictions differ between original and loaded model")
        print()

        # Clean up Redis key
        try:
            redis_client.delete(redis_key)
            print(f"✓ Cleaned up Redis key: {redis_key}")
        except Exception as e:
            print(f"⚠ Could not clean up Redis key: {e}")
        print()

    # Success indicators
    if results.get('accuracy', 0) > 0:
        print("✓✓✓ SUCCESS ✓✓✓")
        print()
        print("The RNN model trained and evaluated successfully!")
        print("You can now iterate on rnn_model.py with confidence.")
    else:
        print("⚠ WARNING: Accuracy is 0")
        print("The model trained but may not be learning correctly.")

    sys.exit(0)

except Exception as e:
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("TRAINING FAILED")
    print("=" * 70)
    print()
    print(f"Time before failure: {elapsed:.2f} seconds")
    print()
    print(f"Error: {type(e).__name__}")
    print(f"Message: {e}")
    print()

    import traceback
    print("Full traceback:")
    print("-" * 70)
    traceback.print_exc()
    print("-" * 70)
    print()

    sys.exit(1)

finally:
    # Cleanup
    print()
    print("Stopping Spark session...")
    spark.stop()
    print("✓ Done")
