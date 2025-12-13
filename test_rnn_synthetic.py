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
import random
from turtle import pos

# CRITICAL: Force CPU-only mode BEFORE any imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

# Now import everything
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
import redis

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
parser.add_argument('--suppress-rnn', action='store_true',
                    help='Suppress RNN training for quick test')
parser.add_argument('--suppress-logistic', action='store_true',
                    help='Suppress Logistic training for quick test')
args = parser.parse_args()

print(f"Configuration:")
print(f"  Verbosity: {args.verbosity}")
print(f"  Documents: {args.num_docs}")
print(f"  Lines per doc: {args.lines_per_doc}")
print(f"  Epochs: {args.epochs}")
print(f"  Window size: {args.window_size}")
if args.suppress_rnn:
    print("  RNN training suppressed for quick test")
if args.suppress_logistic:
    print("  Logistic training suppressed for quick test")
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
    StructField("doc_id", StringType(), False),
    StructField("value", StringType(), False),
    StructField("label", StringType(), False),
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
    'line_level': True,
    'verbosity': args.verbosity
}

print(f"Model parameters:")
for key, value in model_params.items():
    print(f"  {key}: {value}")
print()

rnn_classifier = SkolClassifierV2(
    spark=spark,
    input_source='dataframe',
    output_dest=None,  # No output needed for test
    model_storage=None,  # No model saving needed
    model_type='rnn',
    use_suffixes=True,
    min_doc_freq=1,  # Low threshold for small dataset
    **model_params  # Spread the dict instead of passing as model_params=
)

logistic_classifier = SkolClassifierV2(
    spark=spark,
    input_source='dataframe',
    output_dest=None,  # No output needed for test
    model_storage=None,  # No model saving needed
    model_type='logistic',
    use_suffixes=True,
    min_doc_freq=1,  # Low threshold for small dataset
    **model_params  # Spread the dict instead of passing as model_params=
)

# Both classifier loaders produce the same result.
df = rnn_classifier.load_raw_from_df(df)
if args.verbosity >= 3:
    print("Sample annotated data:")
    df.show(10, truncate=60)
    print()

print("✓ Classifiers initialized")
print()

# Train models
print("=" * 70)
print("TRAINING")
print("=" * 70)
print()

# Split data for testing
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Setup Redis client
logistic_redis_key = "test_logistic_model_synthetic"
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    redis_client.ping()
    print("✓ Connected to Redis")
except Exception as e:
    print(f"⚠ WARNING: Could not connect to Redis: {e}")
    print("Skipping save/load test")
    redis_client = None

if not args.suppress_logistic:
    try:
        start_time = time.time()
        logistic_results = logistic_classifier.fit(annotated_data=df)
        elapsed = time.time() - start_time

        print()
        print("=" * 70)
        print("LOGISTIC TRAINING COMPLETE")
        print("=" * 70)
        print()
        print(f"Training time: {elapsed:.2f} seconds")
        print()

        print("LOGISTIC Results:")
        for key, value in logistic_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()

        # Test model save and load with Redis
        print("=" * 70)
        print("TESTING LOGISTIC MODEL SAVE/LOAD (Redis)")
        print("=" * 70)
        print()

        if redis_client:
            # Configure the classifier for Redis storage
            logistic_classifier.model_storage = 'redis'
            logistic_classifier.redis_client = redis_client
            logistic_classifier.redis_key = logistic_redis_key
            logistic_classifier.redis_expire = 3600  # 1 hour

            # Save models to Redis
            print(f"Saving logistic model to Redis key: {logistic_redis_key}")
            logistic_classifier.save_model()
            print("✓ LOGISTIC model saved to Redis")
            print()

            # Make predictions with original model
            logistic_original_predictions = logistic_classifier.predict(test_df)
            logistic_original_predictions = logistic_original_predictions.select("doc_id", "line_number", "prediction", "predicted_label", "value")
            logistic_original_count = logistic_original_predictions.count()
            print(f"✓ Original LOGISTIC model predictions: {logistic_original_count} rows")


            # Collect a few predictions for comparison
            logistic_original_sample = logistic_original_predictions.limit(10).collect()
            if logistic_original_count == 0:
                print("⚠ WARNING: Original LOGISTIC model produced 0 predictions!")
            print()


            # Create a new classifier instance and load the model from Redis
            print("Loading models into NEW classifier instances from Redis...")
            logistic_classifier_loaded = SkolClassifierV2(
                spark=spark,
                input_source='dataframe',
                output_dest=None,
                model_storage='redis',
                redis_client=redis_client,
                redis_key=logistic_redis_key,
                model_type='logistic',
                use_suffixes=True,
                min_doc_freq=1,
                **model_params
            )
            logistic_classifier_loaded.load_model()

            print("✓ Logistic model loaded from Redis")
            print()

            # Make predictions with loaded model
            print("Making predictions with LOADED model...")
            logistic_loaded_predictions = logistic_classifier_loaded.predict(test_df)
            print(f"  DEBUG: Raw LOGISTIC predictions count: {logistic_loaded_predictions.count()}")
            print(f"  DEBUG: Raw LOGISTIC predictions columns: {logistic_loaded_predictions.columns}")

            # Show a sample of raw predictions
            if logistic_loaded_predictions.count() > 0:
                print("  Sample LOGISTIC raw predictions:")
                logistic_loaded_predictions.show(5, truncate=False)

            # logistic_loaded_predictions = logistic_loaded_predictions.select("doc_id", "line_number", "prediction", "predicted_label", "value")
            logistic_loaded_count = logistic_loaded_predictions.count()
            print(f"✓ Loaded LOGISTIC model predictions: {logistic_loaded_count} rows")

            # Collect a few predictions for comparison
            logistic_loaded_sample = logistic_loaded_predictions.orderBy("doc_id", "line_number").limit(10).collect()
            if logistic_loaded_count == 0:
                print("⚠ WARNING: Loaded  LOGISTIC model produced 0 predictions!")
            print()

            # Compare predictions
            print("Comparing LOGISTIC predictions...")
            print("-" * 70)
            print(f"{'Doc ID':<15} {'Line#':<7} {'Original':<12} {'Loaded':<12} {'Match':<10}")
            print("-" * 70)

            mismatches = 0
            for i in range(min(len(logistic_original_sample), len(logistic_loaded_sample))):
                orig = logistic_original_sample[i]
                load = logistic_loaded_sample[i]
                match = "✓" if orig.prediction == load.prediction else "✗"
                if orig.prediction != load.prediction:
                    mismatches += 1
                print(f"{orig.doc_id:<15} {orig.line_number:<7} {orig.prediction:<12.1f} {load.prediction:<12.1f} {match:<10}")

            print("-" * 70)
            print()

            if mismatches == 0:
                print("✓✓✓ ALL LOGISTIC PREDICTIONS MATCH ✓✓✓")
                print("The saved and loaded model produces identical predictions!")
            else:
                print(f"⚠ WARNING: {mismatches} LOGISTIC predictions differ between original and loaded model")
            print()


            # Clean up Redis key
            try:
                redis_client.delete(logistic_redis_key)
                print(f"✓ Cleaned up Redis key: {logistic_redis_key}")
            except Exception as e:
                print(f"⚠ Could not clean up Redis keys: {e}")
            print()

        # Success indicators
        if logistic_results.get('accuracy', 0) > 0:
            print("✓✓✓ SUCCESS ✓✓✓")
            print()
            print("The LOGISTIC model trained and evaluated successfully!")
            print("You can now iterate on logistic_model.py with confidence.")
        else:
            print("⚠ WARNING: Accuracy is 0")
            print("The model trained but may not be learning correctly.")

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print("=" * 70)
        print("LOGISTIC TRAINING FAILED")
        print("=" * 70)
        print()
        print(f"Time before failure: {elapsed:.2f} seconds")
        print()
        print(f"Error: {type(e).__name__}")
        print(f"Message: {e}")
        print()
        # Cleanup
        print()
        print("Stopping Spark session...")
        spark.stop()
        print("✓ Done")
        sys.exit(1)

if not args.suppress_rnn:
    try:
        start_time = time.time()
        rnn_results = rnn_classifier.fit(annotated_data=df)
        elapsed = time.time() - start_time

        print()
        print("=" * 70)
        print("RNN TRAINING COMPLETE")
        print("=" * 70)
        print()
        print(f"Training time: {elapsed:.2f} seconds")
        print()

        print("RNN Results:")
        for key, value in rnn_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()

        # Test model save and load with Redis
        print("=" * 70)
        print("TESTING RNN MODEL SAVE/LOAD (Redis)")
        print("=" * 70)
        print()

        # Setup Redis client
        rnn_redis_key = "test_rnn_model_synthetic"
        if redis_client:
            # Configure the classifier for Redis storage
            rnn_classifier.model_storage = 'redis'
            rnn_classifier.redis_client = redis_client
            rnn_classifier.redis_key = rnn_redis_key
            rnn_classifier.redis_expire = 3600  # 1 hour

            # Save models to Redis
            print(f"Saving rnn model to Redis key: {rnn_redis_key}")
            rnn_classifier.save_model()
            print("✓ RNN model saved to Redis")
            print()

            # Make predictions with original model
            print("Making predictions with ORIGINAL model...")
            rnn_original_predictions = rnn_classifier.predict(test_df)
            # rnn_original_predictions = rnn_original_predictions.select("doc_id", "line_number", "prediction", "predicted_label", "value")
            rnn_original_count = rnn_original_predictions.count()
            print(f"✓ Original RNN model predictions: {rnn_original_count} rows")
            f_zeros = rnn_original_predictions.filter(F.col('prediction') == 0.0)
            if f_zeros.count() == rnn_original_count:
                raise ValueError("All predictions are zero, model may not be learning correctly.")

            # Collect a few predictions for comparison
            rnn_original_sample = rnn_original_predictions.limit(10).collect()
            if rnn_original_count == 0:
                print("⚠ WARNING: Original RNN model produced 0 predictions!")
            print()

            # Create a new classifier instance and load the model from Redis
            print("Loading models into NEW classifier instances from Redis...")
            rnn_classifier_loaded = SkolClassifierV2(
                spark=spark,
                input_source='dataframe',
                output_dest=None,
                model_storage='redis',
                redis_client=redis_client,
                redis_key=rnn_redis_key,
                model_type='rnn',
                use_suffixes=True,
                min_doc_freq=1,
                **model_params
            )
            rnn_classifier_loaded.load_model()

            print("✓ Models loaded from Redis")
            print(f"  DEBUG: Loaded RNN model input_size: {rnn_classifier_loaded._model.input_size}")
            print(f"  DEBUG: Loaded RNN model window_size: {rnn_classifier_loaded._model.window_size}")
            print()

            # Make predictions with loaded model
            print("Making predictions with LOADED model...")
            rnn_loaded_predictions = rnn_classifier_loaded.predict(test_df)
            print(f"  Raw RNN predictions count: {rnn_loaded_predictions.count()}")
            print(f"  Raw RNN predictions columns: {rnn_loaded_predictions.columns}")
            # Show a sample of raw predictions
            if rnn_loaded_predictions.count() > 0:
                print("  Sample RNN raw predictions:")
                rnn_loaded_predictions.show(5, truncate=False)


            # rnn_loaded_predictions = rnn_loaded_predictions.select("doc_id", "line_number", "prediction", "predicted_label", "value")
            rnn_loaded_count = rnn_loaded_predictions.count()
            print(f"✓ Loaded RNN model predictions: {rnn_loaded_count} rows")

            # Collect a few predictions for comparison
            rnn_loaded_sample = rnn_loaded_predictions.orderBy("doc_id", "line_number").limit(10).collect()
            if rnn_loaded_count == 0:
                print("⚠ WARNING: Loaded  RNN model produced 0 predictions!")
            print()

            # Compare predictions
            print("Comparing RNN predictions...")
            print("-" * 70)
            print(f"{'Doc ID':<15} {'Line#':<7} {'Original':<12} {'Loaded':<12} {'Match':<10}")
            print("-" * 70)

            mismatches = 0
            for i in range(min(len(rnn_original_sample), len(rnn_loaded_sample))):
                orig = rnn_original_sample[i]
                load = rnn_loaded_sample[i]
                match = "✓" if orig.prediction == load.prediction else "✗"
                if orig.prediction != load.prediction:
                    mismatches += 1
                print(f"{orig.doc_id:<15} {orig.line_number:<7} {orig.prediction:<12.1f} {load.prediction:<12.1f} {match:<10}")

            print("-" * 70)
            print()

            if mismatches == 0:
                print("✓✓✓ ALL RNN PREDICTIONS MATCH ✓✓✓")
                print("The saved and loaded model produces identical predictions!")
            else:
                print(f"⚠ WARNING: {mismatches} RNN predictions differ between original and loaded model")
            print()

            # Clean up Redis key
            try:
                redis_client.delete(rnn_redis_key)
                print(f"✓ Cleaned up Redis key: {rnn_redis_key}")
            except Exception as e:
                print(f"⚠ Could not clean up Redis keys: {e}")
            print()

        # Success indicators
        if rnn_results.get('accuracy', 0) > 0:
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

sys.exit(0)