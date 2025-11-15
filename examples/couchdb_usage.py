"""
Example of using SKOL classifier with CouchDB for input/output

DISTRIBUTED ARCHITECTURE (foreachPartition approach):
=====================================================

This example demonstrates the distributed partition-level processing approach:

1. READING from CouchDB:
   - Driver fetches document IDs (lightweight metadata only)
   - Creates DataFrame with (doc_id, attachment_name) rows
   - DataFrame is distributed across N partitions
   - Each partition connects to CouchDB ONCE
   - That connection fetches ALL documents in that partition
   - Processing happens in parallel across all workers

2. WRITING to CouchDB:
   - Predictions DataFrame is distributed across partitions
   - Each partition connects to CouchDB ONCE
   - That connection saves ALL documents in that partition
   - All writes happen in parallel

3. EFFICIENCY GAINS:
   - Traditional: N connections (one per document)
   - foreachPartition: P connections (one per partition, where P << N)
   - Example: 100,000 documents with 100 partitions = 100 connections vs 100,000

4. SCALABILITY:
   - More worker nodes = faster processing
   - More partitions = more parallelism (up to number of cores)
   - No driver bottleneck - driver only handles metadata
   - Workers handle all actual I/O

See DISTRIBUTED_COUCHDB.md for detailed architecture documentation.
"""

import redis
from skol_classifier import SkolClassifier, CouchDBReader, CouchDBWriter


def example_read_from_couchdb():
    """
    Read text attachments from CouchDB using distributed processing.

    This uses the load_from_couchdb_distributed() function which:
    1. Fetches document IDs on the driver (lightweight)
    2. Distributes IDs across partitions
    3. Each partition connects to CouchDB once
    4. Each partition fetches all its assigned documents
    """

    # CouchDB connection details
    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    username = "admin"
    password = "password"

    # Create classifier to get Spark session
    classifier = SkolClassifier()

    # Load documents using distributed approach
    df = classifier.load_from_couchdb(
        couchdb_url=couchdb_url,
        database=database,
        username=username,
        password=password,
        pattern="*.txt"
    )

    print(f"Found {df.count()} text attachments")
    print("\nSample documents:")
    df.select("doc_id", "attachment_name").show(5, truncate=False)

    # Show partition distribution
    print(f"\nData distributed across {df.rdd.getNumPartitions()} partitions")
    print("Each partition will process its documents using a single CouchDB connection")


def example_classify_couchdb_data():
    """
    Load model from Redis, classify CouchDB data, save back to CouchDB.

    This demonstrates the complete distributed pipeline:
    1. Read documents from CouchDB (distributed across partitions)
    2. Classify using PySpark pipeline (distributed)
    3. Save results back to CouchDB (distributed writes)

    Each partition maintains a single CouchDB connection throughout.
    """

    # CouchDB settings
    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    username = "admin"
    password = "password"

    # Redis settings
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Initialize classifier - auto-loads model from Redis
    print("Loading classifier from Redis...")
    classifier = SkolClassifier(
        redis_client=redis_client,
        redis_key="production_model"
    )

    if classifier.labels is None:
        print("No model found in Redis. Please train a model first.")
        return

    print(f"Model loaded with labels: {classifier.labels}")

    # Process CouchDB data using distributed approach
    print("\nLoading and classifying documents from CouchDB...")
    print("(Documents fetched in parallel using foreachPartition)")
    predictions = classifier.predict_from_couchdb(
        couchdb_url=couchdb_url,
        database=database,
        username=username,
        password=password,
        pattern="*.txt"
    )

    # Show sample predictions
    print("\nSample predictions:")
    predictions.select(
        "doc_id", "attachment_name", "predicted_label"
    ).show(5, truncate=50)

    # Save results back to CouchDB using distributed writes
    print("\nSaving predictions back to CouchDB...")
    print("(Each partition writes its documents using a single connection)")
    results = classifier.save_to_couchdb(
        predictions=predictions,
        couchdb_url=couchdb_url,
        database=database,
        username=username,
        password=password,
        suffix=".ann"
    )

    # Report results
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\nSaved {successful} annotated files to CouchDB")
    if failed > 0:
        print(f"Failed to save {failed} files")
        for r in results:
            if not r['success']:
                print(f"  {r['doc_id']}/{r['attachment_name']}")


def example_complete_pipeline():
    """Complete pipeline: Train, save to Redis, classify CouchDB data."""

    from skol_classifier import get_file_list

    # Settings
    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    db_username = "admin"
    db_password = "password"

    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Step 1: Train model (or load from Redis)
    classifier = SkolClassifier(
        redis_client=redis_client,
        redis_key="skol_production_model"
    )

    if classifier.labels is None:
        print("Training new model...")
        annotated_files = get_file_list("/data/annotated", pattern="**/*.ann")
        results = classifier.fit(annotated_files)
        print(f"Training complete! F1: {results['f1_score']:.4f}")

        # Save to Redis
        classifier.save_to_redis()
        print("Model saved to Redis")
    else:
        print(f"Loaded model from Redis: {classifier.labels}")

    # Step 2: Process CouchDB documents
    print("\nProcessing CouchDB documents...")
    predictions = classifier.predict_from_couchdb(
        couchdb_url=couchdb_url,
        database=database,
        username=db_username,
        password=db_password
    )

    # Step 3: Save back to CouchDB
    print("Saving annotated results...")
    results = classifier.save_to_couchdb(
        predictions=predictions,
        couchdb_url=couchdb_url,
        database=database,
        username=db_username,
        password=db_password
    )

    print(f"Complete! Processed {len(results)} documents")


def example_manual_couchdb_workflow():
    """
    Manual workflow using CouchDB Reader/Writer directly.

    The CouchDBReader and CouchDBWriter convenience classes use the
    same distributed foreachPartition approach internally.
    """

    # CouchDB settings
    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    username = "admin"
    password = "password"

    # Initialize classifier
    classifier = SkolClassifier()
    # ... assume model is trained or loaded ...

    # Step 1: Read from CouchDB using distributed reader
    print("Reading documents from CouchDB...")
    print("(Using foreachPartition for distributed reads)")
    df = classifier.load_from_couchdb(
        couchdb_url=couchdb_url,
        database=database,
        username=username,
        password=password
    )

    print(f"Loaded {df.count()} documents")
    df.show(5, truncate=50)

    # Step 2: Process with classifier
    # (Use standard prediction methods on the distributed DataFrame)

    # Step 3: Write back using writer
    # The writer also uses foreachPartition for distributed writes
    writer = CouchDBWriter(couchdb_url, database, username, password)

    # Create a sample predictions DataFrame
    sample_predictions = classifier.spark.createDataFrame([
        ("doc123", "article.txt", "[@ Sample paragraph #Description]")
    ], ["doc_id", "attachment_name", "final_aggregated_pg"])

    # Save using distributed approach
    results = writer.save_from_dataframe(sample_predictions, suffix=".ann")

    print(f"Saved {len(results)} documents using distributed writes")


def example_batch_processing():
    """
    Process documents in batches from CouchDB.

    Note: With the foreachPartition approach, you typically don't need manual
    batching. Spark automatically distributes work across partitions.
    However, if you want to control memory usage or checkpoint progress,
    you can process in batches.
    """

    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Load classifier
    classifier = SkolClassifier(
        redis_client=redis_client,
        redis_key="production_model"
    )

    if classifier.labels is None:
        print("No model available")
        return

    # Load all documents using distributed approach
    df = classifier.load_from_couchdb(
        couchdb_url=couchdb_url,
        database=database,
        username="admin",
        password="password"
    )

    total_docs = df.count()
    print(f"Found {total_docs} documents to process")

    # Process in batches (optional - mainly for checkpointing)
    batch_size = 1000
    num_batches = (total_docs + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        offset = batch_num * batch_size
        print(f"\nProcessing batch {batch_num + 1}/{num_batches}...")

        # Get batch using limit/offset
        # Note: Each batch still uses foreachPartition internally
        batch_df = df.limit(batch_size).offset(offset)

        # Predict on batch (distributed)
        predictions = classifier.predict_raw_text(batch_df)

        # Save batch (distributed writes)
        results = classifier.save_to_couchdb(
            predictions=predictions,
            couchdb_url=couchdb_url,
            database=database,
            username="admin",
            password="password"
        )

        print(f"Batch {batch_num + 1} complete: {len(results)} documents saved")


def example_partitioning_and_parallelism():
    """
    Demonstrate how to control partitioning for optimal performance.

    Key concepts:
    - More partitions = more parallelism
    - Each partition creates ONE CouchDB connection
    - Rule of thumb: 2-4x the number of cores
    """

    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    classifier = SkolClassifier(
        redis_client=redis_client,
        redis_key="production_model"
    )

    # Load documents
    df = classifier.load_from_couchdb(
        couchdb_url=couchdb_url,
        database=database,
        username="admin",
        password="password"
    )

    print(f"Loaded {df.count()} documents")
    print(f"Default partitions: {df.rdd.getNumPartitions()}")

    # Repartition for better parallelism
    # If you have 100 CPU cores, use 200-400 partitions
    num_partitions = 100
    df_repartitioned = df.repartition(num_partitions)

    print(f"After repartitioning: {df_repartitioned.rdd.getNumPartitions()} partitions")
    print(f"This means {num_partitions} CouchDB connections will be used in parallel")
    print(f"(vs {df_repartitioned.count()} connections if we connected per-row)")

    # Process with optimized partitioning
    predictions = classifier.predict_raw_text(df_repartitioned)

    # Save with same partitioning
    # Each partition will reuse its connection for all saves
    results = classifier.save_to_couchdb(
        predictions=predictions,
        couchdb_url=couchdb_url,
        database=database,
        username="admin",
        password="password"
    )

    print(f"\nProcessed {len(results)} documents using {num_partitions} parallel connections")


def example_monitoring_progress():
    """
    Example of monitoring progress during distributed processing.

    Note: With foreachPartition, progress tracking is implicit through
    Spark's task tracking. Use Spark UI (http://localhost:4040) to monitor:
    - Number of active tasks (= number of partitions being processed)
    - Task duration
    - Failed tasks
    """

    couchdb_url = "http://localhost:5984"
    database = "skol_documents"

    classifier = SkolClassifier()

    print("Starting distributed CouchDB processing...")
    print("Monitor progress at: http://localhost:4040 (Spark UI)")

    # Load documents - watch Spark UI for task progress
    df = classifier.load_from_couchdb(
        couchdb_url=couchdb_url,
        database=database,
        username="admin",
        password="password"
    )

    # Cache for multiple operations
    df.cache()

    print(f"Loaded and cached {df.count()} documents")
    print(f"Processing across {df.rdd.getNumPartitions()} partitions")

    # View sample to trigger computation
    print("\nSample documents:")
    df.show(5)

    print("\nDistributed processing complete!")
    print("Check Spark UI for detailed task statistics")


if __name__ == "__main__":
    print("=" * 80)
    print("SKOL Classifier - CouchDB Integration Examples")
    print("Using distributed foreachPartition approach for scalability")
    print("=" * 80)

    print("\n" + "=" * 60)
    print("Example 1: Read from CouchDB (distributed)")
    print("=" * 60)
    example_read_from_couchdb()

    print("\n" + "=" * 60)
    print("Example 2: Classify CouchDB data (distributed pipeline)")
    print("=" * 60)
    example_classify_couchdb_data()

    print("\n" + "=" * 60)
    print("Example 3: Complete pipeline (train + process)")
    print("=" * 60)
    example_complete_pipeline()

    print("\n" + "=" * 60)
    print("Example 4: Partitioning and parallelism")
    print("=" * 60)
    example_partitioning_and_parallelism()

    print("\n" + "=" * 60)
    print("Example 5: Monitoring progress")
    print("=" * 60)
    example_monitoring_progress()

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("\nKey takeaways:")
    print("- Each partition creates ONE CouchDB connection")
    print("- That connection is reused for ALL documents in the partition")
    print("- More partitions = more parallelism (but also more connections)")
    print("- Monitor via Spark UI at http://localhost:4040")
    print("=" * 80)
