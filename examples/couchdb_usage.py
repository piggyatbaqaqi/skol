"""
Example of using SKOL classifier with CouchDB for input/output
"""

import redis
from skol_classifier import SkolClassifier, CouchDBReader, CouchDBWriter


def example_read_from_couchdb():
    """Read text attachments from CouchDB."""

    # CouchDB connection details
    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    username = "admin"
    password = "password"

    # Create reader
    reader = CouchDBReader(couchdb_url, database, username, password)

    # Get all .txt attachments
    attachments = reader.get_text_attachments(pattern="*.txt")

    print(f"Found {len(attachments)} text attachments")
    for att in attachments[:5]:  # Show first 5
        print(f"  Doc: {att['doc_id']}, Attachment: {att['attachment_name']}")
        print(f"    Content preview: {att['content'][:100]}...")


def example_classify_couchdb_data():
    """Load model from Redis, classify CouchDB data, save back to CouchDB."""

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

    # Process CouchDB data
    print("Loading and classifying documents from CouchDB...")
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
        "doc_id", "attachment_name", "predicted_label", "value"
    ).show(5, truncate=50)

    # Save results back to CouchDB
    print("\nSaving predictions back to CouchDB...")
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

    print(f"Saved {successful} annotated files to CouchDB")
    if failed > 0:
        print(f"Failed to save {failed} files")
        for r in results:
            if not r['success']:
                print(f"  Error for {r['doc_id']}/{r['attachment_name']}: {r['error']}")


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
    """Manual workflow using CouchDB Reader/Writer directly."""

    # CouchDB settings
    couchdb_url = "http://localhost:5984"
    database = "skol_documents"
    username = "admin"
    password = "password"

    # Initialize classifier
    classifier = SkolClassifier()
    # ... assume model is trained or loaded ...

    # Step 1: Read from CouchDB using reader
    print("Reading documents from CouchDB...")
    df = classifier.load_from_couchdb(
        couchdb_url=couchdb_url,
        database=database,
        username=username,
        password=password
    )

    print(f"Loaded {df.count()} documents")
    df.show(5, truncate=50)

    # Step 2: Process with classifier
    # (Use standard prediction methods)

    # Step 3: Write back using writer
    writer = CouchDBWriter(couchdb_url, database, username, password)

    # Example: Save a single annotated file
    results = writer.save_annotated_predictions([
        ("doc123", "article.txt", "[@ Sample paragraph #Description]")
    ])

    print(f"Saved: {results}")


def example_batch_processing():
    """Process documents in batches from CouchDB."""

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

    # Get CouchDB reader
    reader = CouchDBReader(couchdb_url, database, "admin", "password")

    # Get all documents
    attachments = reader.get_text_attachments()
    print(f"Found {len(attachments)} documents to process")

    # Process in batches
    batch_size = 100
    for i in range(0, len(attachments), batch_size):
        batch = attachments[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} documents)...")

        # Create DataFrame for batch
        batch_df = classifier.spark.createDataFrame(
            [(att['doc_id'], att['attachment_name'], att['content']) for att in batch],
            ['doc_id', 'attachment_name', 'value']
        )

        # Process batch
        # (Use standard processing pipeline)

        print(f"Batch {i//batch_size + 1} complete")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Read from CouchDB")
    print("=" * 60)
    example_read_from_couchdb()

    print("\n" + "=" * 60)
    print("Example 2: Classify CouchDB data")
    print("=" * 60)
    example_classify_couchdb_data()

    print("\n" + "=" * 60)
    print("Example 3: Complete pipeline")
    print("=" * 60)
    example_complete_pipeline()
