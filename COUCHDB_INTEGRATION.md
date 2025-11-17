## CouchDB Integration for SKOL Classifier

## Overview

The SKOL Classifier now supports reading raw text from CouchDB attachments and writing annotated results back to CouchDB. This is perfect for managing large document collections where:

- Raw documents are stored as `.txt` attachments in CouchDB
- Annotated results should be saved as `.txt.ann` attachments in the same documents
- Document IDs and attachment names need to be preserved

## Installation

CouchDB support requires the `CouchDB` package (version 1.2 or later):

```bash
pip install CouchDB>=1.2
```

Or install with updated requirements:

```bash
pip install -e .
```

## Features

1. **Distributed Processing**: Uses PySpark UDFs to process documents in parallel across cluster
2. **Read from CouchDB**: Load `.txt` attachments using distributed workers
3. **Automatic Processing**: Maintains document ID and attachment name through the pipeline
4. **Write to CouchDB**: Save annotated results in parallel using distributed UDFs
5. **Integration with Redis**: Combine CouchDB I/O with Redis model storage
6. **Scalability**: Handles large datasets by distributing I/O across Spark workers

> **Note**: This implementation uses **distributed UDFs** where each Spark worker connects to CouchDB independently. This avoids loading all data on the driver and enables horizontal scaling. See [DISTRIBUTED_COUCHDB.md](DISTRIBUTED_COUCHDB.md) for details.

## Quick Start

### Basic Workflow

```python
import redis
from skol_classifier import SkolClassifier

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Initialize classifier with Redis and CouchDB configuration
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="production_model",
    couchdb_url="http://localhost:5984",
    database="my_documents",
    username="admin",
    password="password"
)

# Process CouchDB documents
predictions = classifier.predict_from_couchdb()

# Save back to CouchDB as .ann attachments
results = classifier.save_to_couchdb(predictions=predictions, suffix=".ann")

print(f"Processed {len(results)} documents")
```

## API Reference

### SkolClassifier Methods

#### `load_from_couchdb(pattern="*.txt")`

Load raw text from CouchDB attachments.

Uses the CouchDB configuration set in the constructor.

**Args:**
- `pattern`: Pattern for attachment names (default: "*.txt")

**Returns:** DataFrame with columns: `doc_id`, `attachment_name`, `value`

**Example:**
```python
# Initialize with CouchDB configuration
classifier = SkolClassifier(
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)

df = classifier.load_from_couchdb(pattern="*.txt")
print(f"Loaded {df.count()} documents")
```

#### `predict_from_couchdb(pattern="*.txt", output_format="annotated")`

Load text from CouchDB, predict labels, and return predictions.

Uses the CouchDB configuration set in the constructor.

**Args:**
- `pattern`: Pattern for attachment names
- `output_format`: Output format ('annotated' or 'simple')

**Returns:** DataFrame with predictions, including `doc_id` and `attachment_name`

**Example:**
```python
# Initialize with CouchDB configuration
classifier = SkolClassifier(
    redis_client=redis_client,
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)

predictions = classifier.predict_from_couchdb()

# View predictions
predictions.select("doc_id", "attachment_name", "predicted_label").show()
```

#### `save_to_couchdb(predictions, suffix=".ann")`

Save annotated predictions back to CouchDB as attachments.

Uses the CouchDB configuration set in the constructor.

**Args:**
- `predictions`: DataFrame with predictions (must include `annotated_pg` column)
- `suffix`: Suffix to append to attachment names (default: ".ann")

**Returns:** List of results from CouchDB operations

**Example:**
```python
# Initialize with CouchDB configuration
classifier = SkolClassifier(
    redis_client=redis_client,
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)

results = classifier.save_to_couchdb(predictions=predictions, suffix=".ann")

# Check results
for r in results:
    if r['success']:
        print(f"Saved: {r['doc_id']}/{r['attachment_name']}")
    else:
        print(f"Failed: {r['doc_id']}")
```

### CouchDBReader Class

For more control over reading from CouchDB:

```python
from skol_classifier import CouchDBReader

reader = CouchDBReader(
    url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)

# Get all .txt attachments
attachments = reader.get_text_attachments(pattern="*.txt")

for att in attachments:
    print(f"Doc: {att['doc_id']}, File: {att['attachment_name']}")
    print(f"Content: {att['content'][:100]}...")

# Convert to Spark DataFrame
df = reader.to_spark_dataframe(spark_session, pattern="*.txt")
```

### CouchDBWriter Class

For more control over writing to CouchDB:

```python
from skol_classifier import CouchDBWriter

writer = CouchDBWriter(
    url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)

# Save single annotated file
results = writer.save_annotated_predictions([
    ("doc_id_123", "article.txt", "[@ Annotated content #Label]")
])

# Save from DataFrame
results = writer.save_from_dataframe(predictions_df, suffix=".ann")
```

## Data Format

### Input: CouchDB Documents

Documents in CouchDB should have text attachments:

```json
{
  "_id": "article_001",
  "_rev": "1-abc123",
  "title": "My Article",
  "_attachments": {
    "article.txt": {
      "content_type": "text/plain",
      "data": "...base64 encoded content..."
    }
  }
}
```

### Output: Annotated Attachments

The classifier adds new attachments with `.ann` suffix:

```json
{
  "_id": "article_001",
  "_rev": "2-def456",
  "title": "My Article",
  "_attachments": {
    "article.txt": {
      "content_type": "text/plain",
      "data": "...original content..."
    },
    "article.txt.ann": {
      "content_type": "text/plain",
      "data": "[@ Paragraph 1 #Nomenclature]\n[@ Paragraph 2 #Description]..."
    }
  }
}
```

### Multiple Attachments Per Document

**Important**: Documents can have multiple `.txt` attachments, and each will be processed independently:

```json
{
  "_id": "article_001",
  "_attachments": {
    "abstract.txt": {...},
    "methods.txt": {...},
    "results.txt": {...}
  }
}
```

After processing, each `.txt` file gets a corresponding `.ann` file:

```json
{
  "_id": "article_001",
  "_attachments": {
    "abstract.txt": {...},
    "abstract.txt.ann": {...},
    "methods.txt": {...},
    "methods.txt.ann": {...},
    "results.txt": {...},
    "results.txt.ann": {...}
  }
}
```

The system automatically:
- Finds ALL `.txt` attachments in each document
- Processes each one independently (in parallel)
- Saves each result as a separate `.ann` attachment

## Usage Patterns

### Pattern 1: Complete Pipeline

Train model, save to Redis, process CouchDB documents:

```python
import redis
from skol_classifier import SkolClassifier, get_file_list

# Settings
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
couchdb_url = "http://localhost:5984"
database = "documents"

# Initialize classifier with both Redis and CouchDB configuration
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="production_model",
    couchdb_url=couchdb_url,
    database=database,
    username="admin",
    password="password"
)

# Train if needed
if classifier.labels is None:
    print("Training model...")
    files = get_file_list("/data/annotated")
    classifier.fit(files)
    classifier.save_to_redis()
else:
    print(f"Model loaded: {classifier.labels}")

# Process CouchDB
predictions = classifier.predict_from_couchdb()

# Save results
results = classifier.save_to_couchdb(predictions=predictions)

print(f"Processed {len(results)} documents")
```

### Pattern 2: Batch Processing

Process documents in batches:

```python
from skol_classifier import CouchDBReader

reader = CouchDBReader(couchdb_url, database, username, password)
attachments = reader.get_text_attachments()

batch_size = 100
for i in range(0, len(attachments), batch_size):
    batch = attachments[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}...")

    # Create DataFrame for batch
    batch_df = classifier.spark.createDataFrame(
        [(att['doc_id'], att['attachment_name'], att['content']) for att in batch],
        ['doc_id', 'attachment_name', 'value']
    )

    # Process and save batch
    # ... processing code ...
```

### Pattern 3: Error Handling

Handle failures gracefully:

```python
predictions = classifier.predict_from_couchdb(
    couchdb_url=couchdb_url,
    database=database,
    username="admin",
    password="password"
)

results = classifier.save_to_couchdb(
    predictions=predictions,
    couchdb_url=couchdb_url,
    database=database,
    username="admin",
    password="password"
)

# Check for failures
failed = [r for r in results if not r['success']]
if failed:
    print(f"Failed to save {len(failed)} documents:")
    for r in failed:
        print(f"  {r['doc_id']}/{r['attachment_name']}: {r['error']}")

    # Retry failed documents
    # ... retry logic ...
```

### Pattern 4: Custom Attachment Suffix

Use different suffixes for different model versions:

```python
# Save with versioned suffix
results = classifier.save_to_couchdb(
    predictions=predictions,
    couchdb_url=couchdb_url,
    database=database,
    username="admin",
    password="password",
    suffix=".v2.ann"  # Custom suffix
)

# Document will have: article.txt.v2.ann
```

## CouchDB Setup

### Create Database

```bash
curl -X PUT http://admin:password@localhost:5984/documents
```

### Upload Document with Attachment

```bash
# Create document
curl -X PUT http://admin:password@localhost:5984/documents/doc001 \
  -H "Content-Type: application/json" \
  -d '{"title": "Sample Document"}'

# Add attachment
curl -X PUT http://admin:password@localhost:5984/documents/doc001/article.txt?rev=1-xxx \
  -H "Content-Type: text/plain" \
  --data-binary @article.txt
```

### View Attachments

```bash
# List documents
curl http://admin:password@localhost:5984/documents/_all_docs

# Get document with attachments
curl http://admin:password@localhost:5984/documents/doc001

# Download attachment
curl http://admin:password@localhost:5984/documents/doc001/article.txt
```

## Docker CouchDB

Run CouchDB in Docker for development:

```bash
docker run -d \
  --name couchdb \
  -p 5984:5984 \
  -e COUCHDB_USER=admin \
  -e COUCHDB_PASSWORD=password \
  couchdb:latest
```

Connect from Python:

```python
classifier = SkolClassifier()

predictions = classifier.predict_from_couchdb(
    couchdb_url="http://localhost:5984",  # Docker host
    database="documents",
    username="admin",
    password="password"
)
```

## Authentication

### Basic Authentication

```python
predictions = classifier.predict_from_couchdb(
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)
```

### No Authentication

```python
predictions = classifier.predict_from_couchdb(
    couchdb_url="http://localhost:5984",
    database="documents"
    # No username/password
)
```

### Environment Variables

```python
import os

predictions = classifier.predict_from_couchdb(
    couchdb_url=os.getenv("COUCHDB_URL"),
    database=os.getenv("COUCHDB_DATABASE"),
    username=os.getenv("COUCHDB_USER"),
    password=os.getenv("COUCHDB_PASSWORD")
)
```

## Performance Considerations

### Large Documents

- **Memory**: Large attachments are loaded into memory
- **Batching**: Process documents in batches for better memory management
- **Spark**: Leverage Spark's distributed processing for large datasets

### Network Latency

- **Local CouchDB**: Best performance with local CouchDB instance
- **Caching**: Consider caching models in Redis for faster access
- **Connection Pooling**: Reuse connections when processing multiple documents

### CouchDB Indexing

Create views for faster document retrieval:

```javascript
// Design document for finding text attachments
{
  "_id": "_design/attachments",
  "views": {
    "text_files": {
      "map": "function(doc) { if (doc._attachments) { for (var att in doc._attachments) { if (att.endsWith('.txt')) { emit(doc._id, att); } } } }"
    }
  }
}
```

## Troubleshooting

### Connection Errors

**Issue**: Cannot connect to CouchDB
```
Error: Connection refused
```

**Solution**: Check CouchDB is running and URL is correct
```bash
curl http://localhost:5984/
```

### Authentication Errors

**Issue**: Unauthorized
```
Error: 401 Unauthorized
```

**Solution**: Verify credentials
```python
# Test connection
import requests
r = requests.get(
    "http://localhost:5984/_all_dbs",
    auth=("admin", "password")
)
print(r.json())
```

### Document Update Conflicts

**Issue**: Document update conflict
```
Error: 409 Conflict
```

**Solution**: The document was modified between read and write. The code automatically handles this by fetching the latest `_rev` before updating.

### Attachment Too Large

**Issue**: Attachment exceeds size limit
```
Error: 413 Payload Too Large
```

**Solution**: Increase CouchDB's max attachment size in configuration

## Best Practices

1. **Use Redis for Models**: Store trained models in Redis for fast loading
2. **Batch Processing**: Process documents in batches to manage memory
3. **Error Handling**: Always check results for failed uploads
4. **Version Suffixes**: Use versioned suffixes (`.v1.ann`, `.v2.ann`) for different model versions
5. **Monitoring**: Monitor CouchDB disk usage as annotated files accumulate
6. **Backup**: Regularly backup CouchDB database
7. **Indexing**: Create CouchDB views for faster document queries

## Complete Example

See [examples/couchdb_usage.py](examples/couchdb_usage.py) for complete working examples.

## See Also

- [REDIS_INTEGRATION.md](REDIS_INTEGRATION.md) - Redis model storage
- [AUTO_LOAD_FEATURE.md](AUTO_LOAD_FEATURE.md) - Auto-loading models
- [skol_classifier/README.md](skol_classifier/README.md) - Full API documentation
- [CouchDB Documentation](https://docs.couchdb.org/)
