"""
Example: Using GPU acceleration in RNN UDF predictions.

This example demonstrates how to enable and use GPU acceleration
for RNN model predictions in Spark executor UDFs.

IMPORTANT: This feature requires:
1. Worker nodes with NVIDIA GPUs
2. CUDA toolkit installed on workers
3. TensorFlow GPU version
4. Proper Spark GPU configuration

Run this script only if you have verified GPU availability on your cluster.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.linalg import Vectors
from skol_classifier.model import create_model
import numpy as np


def check_gpu_availability():
    """Check if GPU is available for TensorFlow."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow detected {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            return True
        else:
            print("✗ No GPUs detected by TensorFlow")
            return False
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def example_cpu_mode():
    """Example: Default CPU-only mode (safe, compatible)."""
    print("\n" + "="*70)
    print("EXAMPLE 1: CPU-Only Mode (Default)")
    print("="*70)

    spark = SparkSession.builder.appName("RNN CPU Mode").getOrCreate()

    # Create sample training data
    train_data = [
        (Vectors.dense([1.0] * 100), 0),
        (Vectors.dense([2.0] * 100), 1),
        (Vectors.dense([3.0] * 100), 2),
    ] * 10

    from pyspark.ml.linalg import VectorUDT
    schema = StructType([
        StructField("combined_idf", VectorUDT(), False),
        StructField("label_indexed", IntegerType(), False),
    ])

    train_df = spark.createDataFrame(train_data, schema=schema)
    train_df = train_df.withColumn("filename", F.lit("doc1.txt"))
    train_df = train_df.withColumn("line_number", F.monotonically_increasing_id())
    train_df = train_df.withColumn("value", F.lit("sample text"))

    # Create model with CPU-only mode (default)
    print("\nCreating RNN model with CPU-only mode...")
    model = create_model(
        model_type='rnn',
        input_size=100,
        hidden_size=64,
        num_layers=2,
        window_size=10,
        epochs=1,
        batch_size=4,
        use_gpu_in_udf=False,  # Explicitly disable GPU (this is the default)
        verbosity=2
    )

    # Train
    print("\nTraining model...")
    labels = ["Class0", "Class1", "Class2"]
    model.fit(train_df, labels=labels)

    # Predict (runs on CPU in UDFs)
    print("\nMaking predictions (CPU mode)...")
    predictions = model.predict(train_df)

    print(f"\n✓ Predictions completed on CPU")
    print(f"  Total predictions: {predictions.count()}")

    spark.stop()


def example_gpu_mode():
    """Example: GPU-enabled mode (requires GPU setup)."""
    print("\n" + "="*70)
    print("EXAMPLE 2: GPU-Enabled Mode")
    print("="*70)

    # Check GPU availability first
    if not check_gpu_availability():
        print("\n⚠ Skipping GPU mode example - no GPU detected")
        print("This example requires:")
        print("  - NVIDIA GPU with CUDA support")
        print("  - TensorFlow GPU version installed")
        print("  - Proper CUDA/cuDNN setup")
        return

    print("\n✓ GPU detected! Proceeding with GPU-enabled mode...")

    spark = SparkSession.builder.appName("RNN GPU Mode").getOrCreate()

    # Create sample training data
    from pyspark.sql import functions as F

    train_data = [
        (Vectors.dense(np.random.randn(100).tolist()), 0),
        (Vectors.dense(np.random.randn(100).tolist()), 1),
        (Vectors.dense(np.random.randn(100).tolist()), 2),
    ] * 20

    from pyspark.ml.linalg import VectorUDT
    schema = StructType([
        StructField("combined_idf", VectorUDT(), False),
        StructField("label_indexed", IntegerType(), False),
    ])

    train_df = spark.createDataFrame(train_data, schema=schema)
    train_df = train_df.withColumn("filename", F.lit("doc1.txt"))
    train_df = train_df.withColumn("line_number", F.monotonically_increasing_id())
    train_df = train_df.withColumn("value", F.lit("sample text"))

    # Create model with GPU enabled in UDFs
    print("\nCreating RNN model with GPU-enabled mode...")
    model = create_model(
        model_type='rnn',
        input_size=100,
        hidden_size=128,
        num_layers=2,
        window_size=10,
        epochs=2,
        batch_size=8,
        prediction_batch_size=32,  # Batch size for GPU inference
        use_gpu_in_udf=True,  # ⚡ Enable GPU in UDFs
        verbosity=2
    )

    # Train
    print("\nTraining model...")
    labels = ["Class0", "Class1", "Class2"]
    model.fit(train_df, labels=labels)

    # Predict (will attempt to use GPU in UDFs)
    print("\nMaking predictions (GPU mode)...")
    print("Check executor logs for GPU detection messages:")
    print("  [UDF PROBA] TensorFlow imported, N GPU(s) available")

    predictions = model.predict(train_df)

    print(f"\n✓ Predictions completed with GPU mode enabled")
    print(f"  Total predictions: {predictions.count()}")
    print("\nNote: Check Spark executor logs to verify GPU usage")
    print("  If no GPU was available, TensorFlow fell back to CPU")

    spark.stop()


def example_performance_comparison():
    """Example: Compare CPU vs GPU performance (if GPU available)."""
    print("\n" + "="*70)
    print("EXAMPLE 3: CPU vs GPU Performance Comparison")
    print("="*70)

    has_gpu = check_gpu_availability()

    if not has_gpu:
        print("\n⚠ GPU not available - skipping performance comparison")
        return

    import time
    from pyspark.sql import functions as F

    spark = SparkSession.builder.appName("CPU vs GPU Comparison").getOrCreate()

    # Create larger dataset for meaningful comparison
    print("\nCreating test dataset (200 documents)...")
    train_data = []
    for _ in range(200):
        train_data.append((Vectors.dense(np.random.randn(200).tolist()), np.random.randint(0, 3)))

    from pyspark.ml.linalg import VectorUDT
    schema = StructType([
        StructField("combined_idf", VectorUDT(), False),
        StructField("label_indexed", IntegerType(), False),
    ])

    train_df = spark.createDataFrame(train_data, schema=schema)
    train_df = train_df.withColumn("filename", F.lit("doc1.txt"))
    train_df = train_df.withColumn("line_number", F.monotonically_increasing_id())
    train_df = train_df.withColumn("value", F.lit("sample text"))
    train_df.cache()

    labels = ["Class0", "Class1", "Class2"]

    # Train once (we'll reuse model weights)
    print("\nTraining model...")
    base_model = create_model(
        model_type='rnn',
        input_size=200,
        hidden_size=128,
        num_layers=2,
        window_size=20,
        epochs=2,
        batch_size=16,
        verbosity=1
    )
    base_model.fit(train_df, labels=labels)

    # Test CPU mode
    print("\n--- Testing CPU Mode ---")
    cpu_model = create_model(
        model_type='rnn',
        input_size=200,
        hidden_size=128,
        num_layers=2,
        window_size=20,
        prediction_batch_size=32,
        use_gpu_in_udf=False,
        verbosity=1
    )
    cpu_model.set_model(base_model.keras_model)
    cpu_model.labels = labels

    start_time = time.time()
    cpu_predictions = cpu_model.predict(train_df)
    cpu_count = cpu_predictions.count()
    cpu_time = time.time() - start_time

    print(f"CPU Mode: {cpu_count} predictions in {cpu_time:.2f}s ({cpu_count/cpu_time:.1f} pred/sec)")

    # Test GPU mode
    print("\n--- Testing GPU Mode ---")
    gpu_model = create_model(
        model_type='rnn',
        input_size=200,
        hidden_size=128,
        num_layers=2,
        window_size=20,
        prediction_batch_size=64,  # Can use larger batch with GPU
        use_gpu_in_udf=True,
        verbosity=1
    )
    gpu_model.set_model(base_model.keras_model)
    gpu_model.labels = labels

    start_time = time.time()
    gpu_predictions = gpu_model.predict(train_df)
    gpu_count = gpu_predictions.count()
    gpu_time = time.time() - start_time

    print(f"GPU Mode: {gpu_count} predictions in {gpu_time:.2f}s ({gpu_count/gpu_time:.1f} pred/sec)")

    # Compare
    print("\n--- Performance Summary ---")
    print(f"CPU Time: {cpu_time:.2f}s")
    print(f"GPU Time: {gpu_time:.2f}s")
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"Speedup:  {speedup:.2f}x faster with GPU ⚡")
    else:
        print(f"Note: GPU was slower - may indicate GPU overhead for small dataset")

    train_df.unpersist()
    spark.stop()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GPU IN UDF EXAMPLES")
    print("="*70)

    # Example 1: CPU mode (always works)
    example_cpu_mode()

    # Example 2: GPU mode (requires GPU)
    example_gpu_mode()

    # Example 3: Performance comparison (if GPU available)
    example_performance_comparison()

    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. CPU mode (default) works everywhere but is slower")
    print("  2. GPU mode requires proper setup but is much faster")
    print("  3. Always test GPU mode before production use")
    print("  4. Monitor GPU usage with nvidia-smi on workers")
    print("\nFor more details, see: docs/GPU_IN_UDF.md")


if __name__ == '__main__':
    main()
