# Taxon Extraction Pipeline - CouchDB to CouchDB

## Overview

This module provides a complete UDF-based PySpark pipeline for extracting `Taxon` objects from annotated files stored in CouchDB and saving them back to CouchDB as structured JSON documents.

**Key Features:**
- **Idempotent operations**: Uses composite keys `(doc_id, url, line_number)` to prevent duplicates
- **Distributed processing**: Leverages PySpark's `mapPartitions` for scalability
- **Metadata preservation**: Tracks complete lineage from source to extracted taxa
- **Error handling**: Graceful failure recovery with detailed error reporting

## Architecture

### Data Flow

```
Ingest CouchDB Database (annotated files)
  ↓
CouchDBConnection.load_distributed() → DataFrame[doc_id, attachment_name, value, url]
  ↓
mapPartitions(extract_taxa_from_partition) → Parse & Extract Taxa → RDD[Taxon]
  ↓
mapPartitions(convert_taxa_to_rows) → Convert Taxon objects to Rows
  ↓
DataFrame[taxon, description, source, line_number, ...]
  ↓
mapPartitions(save_taxa_to_couchdb_partition) → Save with idempotent keys
  ↓
Taxon CouchDB Database (structured JSON documents)
  ↓
Results DataFrame[doc_id, success, error_message]
```

### Idempotency

Each taxon document is assigned a deterministic ID based on:
```python
doc_id = SHA256(source_doc_id + ":" + source_url + ":" + first_line_number)
```

This ensures:
- Re-running the pipeline won't create duplicates
- Updates to existing taxa are handled gracefully
- Document IDs are stable across runs

### Document Structure

Each taxon document in the taxon database contains:

```json
{
  "_id": "taxon_<sha256_hash>",
  "type": "taxon",
  "serial_number": "123",
  "source": {
    "doc_id": "article_2023_001",
    "url": "http://example.com/article",
    "db_name": "mycobank_annotations",
    "line_number": 42
  },
  "paragraphs": [
    {
      "serial_number": "123",
      "filename": "mycobank/article_2023_001/fulltext.txt.ann",
      "url": "http://example.com/article",
      "label": "Nomenclature",
      "paragraph_number": "5",
      "page_number": "10",
      "empirical_page_number": "8",
      "body": "Agaricus novus Author 2023..."
    },
    {
      "serial_number": "123",
      "filename": "mycobank/article_2023_001/fulltext.txt.ann",
      "url": "http://example.com/article",
      "label": "Description",
      "paragraph_number": "6",
      "page_number": "10",
      "empirical_page_number": "8",
      "body": "Pileus 3-8 cm diam., convex..."
    }
  ],
  "nomenclature_count": 1,
  "description_count": 1
}
```

## Usage

### Basic Usage

```python
from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import extract_and_save_taxa_pipeline

# Create Spark session
spark = SparkSession.builder \
    .appName("SKOL Taxon Extractor") \
    .getOrCreate()

# Run pipeline
results = extract_and_save_taxa_pipeline(
    spark=spark,
    ingest_couchdb_url="http://localhost:5984",
    ingest_db_name="mycobank_annotations",
    taxon_couchdb_url="http://localhost:5984",
    taxon_db_name="mycobank_taxa",
    ingest_username="admin",
    ingest_password="secret",
    taxon_username="admin",
    taxon_password="secret",
    pattern="*.txt.ann"
)

# Check results
total = results.count()
successes = results.filter("success = true").count()
failures = results.filter("success = false").count()

print(f"Total: {total}, Success: {successes}, Failed: {failures}")

# View failures
results.filter("success = false").show(truncate=False)
```

### Command-Line Usage

```bash
python extract_taxa_to_couchdb.py \
    --ingest-database mycobank_annotations \
    --taxon-database mycobank_taxa \
    --ingest-username admin \
    --ingest-password secret \
    --pattern "*.txt.ann"
```

**Arguments:**
- `--ingest-url`: CouchDB URL for ingest database (default: http://localhost:5984)
- `--ingest-database`: Name of ingest database (required)
- `--ingest-username`: Username for ingest database
- `--ingest-password`: Password for ingest database
- `--taxon-url`: CouchDB URL for taxon database (defaults to ingest-url)
- `--taxon-database`: Name of taxon database (required)
- `--taxon-username`: Username for taxon database (defaults to ingest-username)
- `--taxon-password`: Password for taxon database (defaults to ingest-password)
- `--pattern`: Attachment pattern (default: *.txt.ann)

### Advanced: Using Individual Functions

#### Extract Taxa from Partition

```python
from extract_taxa_to_couchdb import extract_taxa_from_partition, convert_taxa_to_rows

def process_partition(partition):
    """Custom partition processing - returns Taxon objects."""
    taxa = extract_taxa_from_partition(partition, "mycobank")
    # Convert to Rows for DataFrame
    return convert_taxa_to_rows(taxa)

# Use with mapPartitions
taxa_rdd = df.rdd.mapPartitions(process_partition)
# Or to work directly with Taxon objects:
def process_taxa(partition):
    return extract_taxa_from_partition(partition, "mycobank")

taxa_objects_rdd = df.rdd.mapPartitions(process_taxa)
# Do something with Taxon objects...
```

#### Save Taxa to CouchDB

```python
from extract_taxa_to_couchdb import save_taxa_to_couchdb_partition

def save_partition(partition):
    """Save taxa with custom settings."""
    return save_taxa_to_couchdb_partition(
        partition,
        "http://localhost:5984",
        "mycobank_taxa",
        username="admin",
        password="secret"
    )

results_rdd = taxa_df.rdd.mapPartitions(save_partition)
```

#### Custom Document ID Generation

```python
from extract_taxa_to_couchdb import generate_taxon_doc_id

# Generate deterministic ID
doc_id = generate_taxon_doc_id(
    doc_id="article_2023_001",
    url="http://example.com/article",
    line_number=42
)
# Returns: "taxon_<sha256_hash>"
```

## Functions Reference

### `extract_and_save_taxa_pipeline()`

Main pipeline function that orchestrates the entire extraction and saving process.

**Parameters:**
- `spark` (SparkSession): Active Spark session
- `ingest_couchdb_url` (str): URL of ingest CouchDB server
- `ingest_db_name` (str): Name of ingest database
- `taxon_couchdb_url` (str): URL of taxon CouchDB server
- `taxon_db_name` (str): Name of taxon database
- `ingest_username` (Optional[str]): Username for ingest database
- `ingest_password` (Optional[str]): Password for ingest database
- `taxon_username` (Optional[str]): Username for taxon database
- `taxon_password` (Optional[str]): Password for taxon database
- `pattern` (str): Pattern for attachment names (default: "*.txt.ann")

**Returns:**
- DataFrame with columns: `doc_id`, `success`, `error_message`

### `extract_taxa_from_partition()`

UDF for extracting taxa from a partition of CouchDB rows.

**Parameters:**
- `partition` (Iterator[Row]): Iterator of Rows with doc_id, attachment_name, value, url
- `ingest_db_name` (str): Database name for metadata tracking

**Yields:**
- Taxon objects with nomenclature and description paragraphs

### `convert_taxa_to_rows()`

Convert Taxon objects to PySpark Rows for DataFrame creation.

**Parameters:**
- `partition` (Iterator[Taxon]): Iterator of Taxon objects

**Yields:**
- PySpark Row objects with fields: taxon, description, source, line_number, paragraph_number, page_number, empirical_page_number

### `save_taxa_to_couchdb_partition()`

UDF for saving taxa to CouchDB (idempotent).

**Parameters:**
- `partition` (Iterator[Row]): Iterator of Rows with taxon data
- `couchdb_url` (str): CouchDB server URL
- `taxon_db_name` (str): Target database name
- `username` (Optional[str]): Optional username
- `password` (Optional[str]): Optional password

**Yields:**
- Rows with: `doc_id`, `success`, `error_message`

### `generate_taxon_doc_id()`

Generate deterministic document ID for idempotent writes.

**Parameters:**
- `doc_id` (str): Source document ID
- `url` (Optional[str]): URL from source line
- `line_number` (int): Line number from source

**Returns:**
- str: Document ID as "taxon_<sha256_hash>"

### `taxon_to_json_doc()`

Convert Taxon object to JSON document.

**Parameters:**
- `taxon` (Taxon): Taxon object to convert
- `first_nomenclature_para` (Paragraph): First nomenclature paragraph for metadata

**Returns:**
- Optional[Dict[str, Any]]: JSON-ready dictionary or None

## Schema Definitions

### Extract Schema
```python
StructType([
    StructField("source_doc_id", StringType(), False),
    StructField("source_url", StringType(), False),
    StructField("source_db_name", StringType(), False),
    StructField("taxon_json", StringType(), False),
    StructField("first_line_number", StringType(), False),
])
```

### Save Results Schema
```python
StructType([
    StructField("doc_id", StringType(), False),
    StructField("success", BooleanType(), False),
    StructField("error_message", StringType(), False),
])
```

## Error Handling

### Connection Errors
If CouchDB connection fails, all rows in the partition are marked as failed:
```python
Row(doc_id=<id>, success=False, error_message="Connection error")
```

### Individual Document Errors
If a specific document fails to save, only that document is marked as failed:
```python
Row(doc_id=<id>, success=False, error_message="Save error: <details>")
```

### Viewing Failures
```python
failures = results.filter("success = false")
failures.select("doc_id", "error_message").show(truncate=False)
```

## Performance Considerations

### Partition Size
- Each partition creates one CouchDB connection
- Optimal partition size: 100-1000 documents
- Adjust with `df.repartition(n)`

### Memory Usage
- Taxa are processed in streaming fashion within partitions
- Paragraphs are materialized temporarily for metadata extraction
- Monitor executor memory for large documents

### Network Efficiency
- One connection per partition minimizes overhead
- Bulk operations where possible
- Consider network latency between Spark and CouchDB

## Best Practices

### 1. Test with Small Dataset First
```python
# Limit to first 10 documents
test_df = df.limit(10)
results = extract_and_save_taxa_pipeline(...)
```

### 2. Monitor Progress
```python
# Cache results for multiple queries
results.cache()

# Check progress
results.groupBy("success").count().show()
```

### 3. Handle Failures
```python
# Re-process failures
failures = results.filter("success = false")
failure_ids = [row.doc_id for row in failures.collect()]

# Create filtered DataFrame and retry
retry_df = df.filter(df.doc_id.isin(failure_ids))
```

### 4. Database Setup
```bash
# Create databases ahead of time
curl -X PUT http://admin:secret@localhost:5984/mycobank_taxa
```

### 5. Idempotency Verification
```python
# Run pipeline twice - should get same results
results1 = extract_and_save_taxa_pipeline(...)
results2 = extract_and_save_taxa_pipeline(...)

# Both should succeed (updates, not duplicates)
assert results1.filter("success = true").count() == results2.filter("success = true").count()
```

## Troubleshooting

### Issue: "Database does not exist"
**Solution:** Create the taxon database before running:
```python
import couchdb
server = couchdb.Server("http://localhost:5984")
server.resource.credentials = ("admin", "secret")
server.create("mycobank_taxa")
```

### Issue: High memory usage
**Solution:** Reduce partition size:
```python
df = df.repartition(200)  # Increase number of partitions
```

### Issue: Slow processing
**Solution:** Check CouchDB network latency and increase parallelism:
```python
spark.conf.set("spark.default.parallelism", "100")
```

### Issue: Authentication failures
**Solution:** Verify credentials and permissions:
```bash
curl -u admin:secret http://localhost:5984/_session
```

## Integration with Existing Workflows

### Use with CouchDBConnection
```python
from skol_classifier.couchdb_io import CouchDBConnection

# Load annotated files
conn = CouchDBConnection(
    "http://localhost:5984",
    "mycobank_annotations",
    "admin",
    "secret"
)

df = conn.load_distributed(spark, "*.txt.ann")

# Extract and save taxa (using already-loaded df)
# ... (customize extract_taxa_from_partition to use pre-loaded df)
```

### Query Saved Taxa
```python
import couchdb

server = couchdb.Server("http://localhost:5984")
server.resource.credentials = ("admin", "secret")
db = server["mycobank_taxa"]

# Find all taxa with descriptions
for doc_id in db:
    doc = db[doc_id]
    if doc.get("type") == "taxon" and doc.get("description_count", 0) > 0:
        print(f"Taxon {doc['serial_number']}: {doc['description_count']} descriptions")
```

## Example: Complete Workflow

```python
from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import extract_and_save_taxa_pipeline

# 1. Setup
spark = SparkSession.builder \
    .appName("SKOL Taxon Extractor") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# 2. Run pipeline
print("Starting taxon extraction pipeline...")
results = extract_and_save_taxa_pipeline(
    spark=spark,
    ingest_couchdb_url="http://localhost:5984",
    ingest_db_name="mycobank_annotations",
    taxon_couchdb_url="http://localhost:5984",
    taxon_db_name="mycobank_taxa",
    ingest_username="admin",
    ingest_password="secret",
    taxon_username="admin",
    taxon_password="secret",
    pattern="*.txt.ann"
)

# 3. Analyze results
results.cache()

total = results.count()
successes = results.filter("success = true").count()
failures = results.filter("success = false").count()

print(f"\n=== Results ===")
print(f"Total documents processed: {total}")
print(f"Successful saves: {successes}")
print(f"Failed saves: {failures}")
print(f"Success rate: {100 * successes / total:.1f}%")

# 4. Handle failures
if failures > 0:
    print("\n=== Failed Documents ===")
    results.filter("success = false") \
        .select("doc_id", "error_message") \
        .show(20, truncate=False)

# 5. Verify in CouchDB
import couchdb
server = couchdb.Server("http://localhost:5984")
server.resource.credentials = ("admin", "secret")
db = server["mycobank_taxa"]

print(f"\n=== Database Statistics ===")
print(f"Total documents in taxon database: {len(db)}")

# 6. Cleanup
spark.stop()
```

## See Also

- [EXTRACTING_TAXON_OBJECTS.md](EXTRACTING_TAXON_OBJECTS.md) - Main extraction guide
- [COUCHDB_INTEGRATION_SUMMARY.md](COUCHDB_INTEGRATION_SUMMARY.md) - CouchDB integration overview
- [couchdb_file_README.md](couchdb_file_README.md) - CouchDB file reading module
- [skol_classifier/couchdb_io.py](skol_classifier/couchdb_io.py) - CouchDBConnection class
