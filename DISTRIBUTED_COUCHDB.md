# Distributed CouchDB Processing with PySpark

## Overview

The SKOL Classifier uses **distributed UDFs** (User-Defined Functions) to process CouchDB documents in parallel across a Spark cluster. This approach:

1. **Avoids loading all data on the driver** - Only document metadata is collected centrally
2. **Distributes I/O operations** - Each Spark worker connects to CouchDB independently
3. **Scales horizontally** - Processing speed increases with cluster size
4. **Handles large datasets** - Can process millions of documents efficiently

## Architecture

### Traditional Approach (Non-Distributed)
```
Driver Node:
  1. Connect to CouchDB
  2. Download ALL attachments to driver
  3. Create DataFrame from collected data
  4. Distribute data to workers

Problem: Driver becomes bottleneck and may run out of memory
```

### Distributed Approach (Current Implementation)
```
Driver Node:
  1. Connect to CouchDB
  2. Get list of document IDs only (lightweight)
  3. Create DataFrame with IDs
  4. Distribute IDs to workers

Worker Nodes (parallel):
  1. Each worker connects to CouchDB
  2. Each fetches only its assigned documents
  3. Processing happens in parallel
  4. Results written back in parallel

Benefit: Work distributed across cluster, no bottleneck
```

## How It Works

### Reading from CouchDB

```python
from skol_classifier import SkolClassifier

classifier = SkolClassifier()

# This uses distributed UDFs under the hood
df = classifier.load_from_couchdb(
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)

# df is a Spark DataFrame where:
# - Document IDs were collected on driver
# - Content is fetched lazily by workers when needed
# - Each worker connects to CouchDB independently
```

**What happens:**
1. Driver gets list of document IDs from CouchDB
2. Driver creates DataFrame with (doc_id, attachment_name) columns
3. When the DataFrame is processed:
   - Spark distributes rows to workers
   - Each worker runs a UDF that connects to CouchDB
   - Worker fetches its assigned documents
   - Processing happens in parallel

### Writing to CouchDB

```python
# predictions is a Spark DataFrame with results
results = classifier.save_to_couchdb(
    predictions,
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)
```

**What happens:**
1. DataFrame is distributed across workers
2. Each worker runs a UDF that:
   - Connects to CouchDB
   - Saves its assigned documents
   - Returns success/failure status
3. Results are collected on driver

## UDF Functions

### Fetch Attachment UDF

```python
from skol_classifier.couchdb_io import create_fetch_attachment_udf

# Create a UDF that runs on workers
fetch_udf = create_fetch_attachment_udf(
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)

# Apply to DataFrame
df_with_content = doc_ids_df.withColumn(
    "value",
    fetch_udf(col("doc_id"), col("attachment_name"))
)

# When this DataFrame is evaluated:
# - Each worker connects to CouchDB
# - Fetches content for its assigned rows
# - All in parallel
```

### Save Attachment UDF

```python
from skol_classifier.couchdb_io import create_save_attachment_udf

# Create a UDF that saves on workers
save_udf = create_save_attachment_udf(
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password",
    suffix=".ann"
)

# Apply to DataFrame
result_df = content_df.withColumn(
    "success",
    save_udf(col("doc_id"), col("attachment_name"), col("content"))
)

# When evaluated:
# - Each worker saves its assigned documents
# - All writes happen in parallel
```

## Performance Considerations

### Parallelism

The number of parallel operations depends on:
- **Number of Spark partitions**: More partitions = more parallelism
- **Number of worker cores**: Each core can run one task
- **DataFrame size**: Need enough rows to distribute

```python
# Check partitions
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Repartition for better parallelism
df = df.repartition(100)  # 100 parallel tasks
```

### CouchDB Connection Pooling

Each worker creates its own CouchDB connection. For better performance:

1. **Use connection pooling** in CouchDB
2. **Increase CouchDB max connections**
3. **Use CouchDB cluster** for distributed database

```ini
# CouchDB configuration
[httpd]
max_connections = 1000

[cluster]
n = 3  # 3-node cluster
```

### Memory Usage

With distributed UDFs:
- **Driver memory**: Only stores document IDs (small)
- **Worker memory**: Each stores only its partition's data
- **Total capacity**: Sum of all worker memory

### Network Considerations

- **Locality**: Place Spark workers close to CouchDB servers
- **Bandwidth**: Ensure sufficient network bandwidth
- **Latency**: Low latency between workers and CouchDB is crucial

## Scaling Examples

### Small Dataset (< 1000 documents)

```python
# Simple approach works fine
classifier = SkolClassifier()
predictions = classifier.predict_from_couchdb(
    couchdb_url="http://localhost:5984",
    database="small_db"
)
```

### Medium Dataset (1,000 - 100,000 documents)

```python
# Increase parallelism
classifier = SkolClassifier()

# Load data
df = classifier.load_from_couchdb(
    couchdb_url="http://localhost:5984",
    database="medium_db"
)

# Repartition for better parallelism
df = df.repartition(50)

# Process
# ... (processing steps)
```

### Large Dataset (> 100,000 documents)

```python
# Use Spark cluster with multiple workers
classifier = SkolClassifier()

# Load data
df = classifier.load_from_couchdb(
    couchdb_url="http://couchdb-cluster:5984",
    database="large_db"
)

# Optimize partitioning
# Rule of thumb: 2-4x number of cores
num_cores = 100  # total cores in cluster
df = df.repartition(num_cores * 3)

# Enable caching if processing multiple times
df.cache()

# Process in batches if needed
batch_size = 10000
total_docs = df.count()

for offset in range(0, total_docs, batch_size):
    batch_df = df.limit(batch_size).offset(offset)
    # Process batch...
```

## Monitoring

### Track Progress

```python
from pyspark import TaskContext

def fetch_with_progress(doc_id, attachment_name):
    context = TaskContext.get()
    if context:
        partition_id = context.partitionId()
        print(f"Partition {partition_id}: Processing {doc_id}")

    # ... fetch logic ...
```

### Monitor Spark UI

The Spark UI (http://localhost:4040) shows:
- Number of active tasks
- Task duration
- Data shuffle
- Failed tasks

### CouchDB Metrics

Monitor CouchDB:
```bash
# Check active connections
curl http://admin:password@localhost:5984/_active_tasks

# Monitor database stats
curl http://admin:password@localhost:5984/database/_info
```

## Error Handling

### Handling Failed Fetches

```python
# The UDF filters out errors automatically
df = classifier.load_from_couchdb(...)

# Check for errors in logs
# Failed fetches are logged but don't stop processing
```

### Retry Failed Saves

```python
results = classifier.save_to_couchdb(predictions, ...)

# Find failed saves
failed = [r for r in results if not r['success']]

if failed:
    print(f"Retrying {len(failed)} failed saves...")
    # Retry logic...
```

### Handling CouchDB Downtime

```python
def fetch_with_retry(doc_id, attachment_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Connect and fetch...
            return content
        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Advanced Patterns

### Process and Save in One UDF

Avoid passing large data through Spark by combining operations:

```python
from skol_classifier.couchdb_io import create_process_and_save_udf

# This UDF reads, processes, and saves all on the worker
process_save_udf = create_process_and_save_udf(
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)

# Apply to document IDs only
doc_ids_df.withColumn(
    "success",
    process_save_udf(col("doc_id"), col("attachment_name"), col("processed"))
)
```

### Batch Processing

```python
# Process in smaller batches to control memory
batch_size = 1000

# Get total count
total = df.count()

for i in range(0, total, batch_size):
    batch = df.offset(i).limit(batch_size)

    # Process batch
    predictions = classifier.predict_from_couchdb_df(batch)

    # Save batch
    classifier.save_to_couchdb(predictions, ...)
```

### Distributed Cache

Share expensive objects across UDF calls:

```python
from pyspark import SparkContext

def get_couchdb_connection(url, username, password):
    # Use broadcast variable to share connection config
    if not hasattr(get_couchdb_connection, 'server'):
        get_couchdb_connection.server = couchdb.Server(url)
        if username:
            get_couchdb_connection.server.resource.credentials = (username, password)

    return get_couchdb_connection.server
```

## Best Practices

1. **Partition Wisely**: Use 2-4x the number of cores
2. **Monitor Resource Usage**: Watch memory and network
3. **Handle Errors Gracefully**: Don't fail entire job on one document
4. **Use CouchDB Views**: Create views for faster document queries
5. **Enable Spark Caching**: Cache DataFrames used multiple times
6. **Test Scaling**: Start small and scale up gradually
7. **Log Appropriately**: Log errors but avoid excessive logging

## Troubleshooting

### Issue: Slow Processing

**Symptoms**: Long execution time, low CPU usage

**Solutions**:
- Increase number of partitions: `df.repartition(N)`
- Check Spark UI for task distribution
- Verify CouchDB isn't bottlenecked

### Issue: Out of Memory

**Symptoms**: Workers crash, OOM errors

**Solutions**:
- Reduce partition size: `df.repartition(more_partitions)`
- Process in batches
- Increase worker memory: `--executor-memory 8g`

### Issue: Connection Timeouts

**Symptoms**: Many failed fetches, timeout errors

**Solutions**:
- Increase CouchDB timeout settings
- Add retry logic to UDFs
- Check network connectivity
- Use CouchDB connection pooling

### Issue: Uneven Processing

**Symptoms**: Some workers idle while others work

**Solutions**:
- Repartition DataFrame: `df.repartition("doc_id")`
- Check data skew (some docs much larger)
- Use `coalesce()` to reduce partitions if needed

## See Also

- [COUCHDB_INTEGRATION.md](COUCHDB_INTEGRATION.md) - Basic CouchDB usage
- [examples/couchdb_usage.py](examples/couchdb_usage.py) - Code examples
- [Apache Spark UDF Documentation](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html)
- [CouchDB Performance Guide](https://docs.couchdb.org/en/stable/best-practices/index.html)
