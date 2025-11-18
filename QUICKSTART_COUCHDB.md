# Quick Start: Extracting Taxa from CouchDB

This guide gets you started with extracting `Taxon` objects from CouchDB in under 5 minutes.

## Prerequisites

```bash
pip install pyspark couchdb
```

## 1. Import Required Modules

```python
from pyspark.sql import SparkSession
from skol_classifier.couchdb_io import CouchDBConnection
from couchdb_file import read_couchdb_partition
from finder import parse_annotated, remove_interstitials
from taxon import group_paragraphs
```

## 2. Define Your Processing Function

```python
def process_partition_to_taxa(partition, db_name):
    """Convert CouchDB partition to taxa."""
    # Read lines with metadata
    lines = read_couchdb_partition(partition, db_name)

    # Parse paragraphs
    paragraphs = parse_annotated(lines)

    # Remove non-content paragraphs
    filtered = remove_interstitials(paragraphs)

    # Group into taxa
    taxa = group_paragraphs(filtered)

    # Convert to dictionaries with metadata
    for taxon in taxa:
        for para_dict in taxon.dictionaries():
            # Extract CouchDB metadata from filename
            parts = para_dict['filename'].split('/', 2)
            if len(parts) == 3:
                para_dict['db_name'] = parts[0]
                para_dict['doc_id'] = parts[1]
                para_dict['attachment_name'] = parts[2]
            yield para_dict
```

## 3. Set Up Spark and CouchDB

```python
# Create Spark session
spark = SparkSession.builder \
    .appName("TaxonExtractor") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Connect to CouchDB
conn = CouchDBConnection(
    couchdb_url="http://localhost:5984",
    database="mycobank_annotations",
    username="admin",
    password="your_password"
)
```

## 4. Load and Process Data

```python
# Load attachments from CouchDB (distributed)
df = conn.load_distributed(spark, "*.txt.ann")

# Process in parallel
taxa_rdd = df.rdd.mapPartitions(
    lambda part: process_partition_to_taxa(part, "mycobank")
)
```

## 5. Save Results

```python
from pyspark.sql.types import StructType, StructField, StringType

# Define schema
schema = StructType([
    StructField("serial_number", StringType(), False),
    StructField("filename", StringType(), False),
    StructField("db_name", StringType(), True),
    StructField("doc_id", StringType(), True),
    StructField("attachment_name", StringType(), True),
    StructField("label", StringType(), False),
    StructField("paragraph_number", StringType(), False),
    StructField("page_number", StringType(), False),
    StructField("empirical_page_number", StringType(), True),
    StructField("body", StringType(), False)
])

# Convert to DataFrame
taxa_df = taxa_rdd.toDF(schema)

# Save to Parquet
taxa_df.write.mode("overwrite").parquet("output/taxa.parquet")

# Or save to CSV
taxa_df.write.mode("overwrite").csv("output/taxa.csv", header=True)

# Show sample results
print(f"\nTotal paragraphs extracted: {taxa_df.count()}")
taxa_df.select("serial_number", "doc_id", "label").show(10)
```

## Complete Example Script

```python
#!/usr/bin/env python
"""Extract taxa from CouchDB - Complete Example"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from skol_classifier.couchdb_io import CouchDBConnection
from couchdb_file import read_couchdb_partition
from finder import parse_annotated, remove_interstitials
from taxon import group_paragraphs


def process_partition_to_taxa(partition, db_name):
    """Convert CouchDB partition to taxa."""
    lines = read_couchdb_partition(partition, db_name)
    paragraphs = parse_annotated(lines)
    filtered = remove_interstitials(paragraphs)
    taxa = group_paragraphs(filtered)

    for taxon in taxa:
        for para_dict in taxon.dictionaries():
            parts = para_dict['filename'].split('/', 2)
            if len(parts) == 3:
                para_dict['db_name'] = parts[0]
                para_dict['doc_id'] = parts[1]
                para_dict['attachment_name'] = parts[2]
            yield para_dict


def main():
    # Setup
    spark = SparkSession.builder \
        .appName("TaxonExtractor") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    conn = CouchDBConnection(
        couchdb_url="http://localhost:5984",
        database="mycobank_annotations",
        username="admin",
        password="password"
    )

    # Load from CouchDB
    print("Loading attachments from CouchDB...")
    df = conn.load_distributed(spark, "*.txt.ann")
    print(f"Found {df.count()} attachments")

    # Process in parallel
    print("Processing partitions to extract taxa...")
    taxa_rdd = df.rdd.mapPartitions(
        lambda part: process_partition_to_taxa(part, "mycobank")
    )

    # Define schema
    schema = StructType([
        StructField("serial_number", StringType(), False),
        StructField("filename", StringType(), False),
        StructField("db_name", StringType(), True),
        StructField("doc_id", StringType(), True),
        StructField("attachment_name", StringType(), True),
        StructField("label", StringType(), False),
        StructField("paragraph_number", StringType(), False),
        StructField("page_number", StringType(), False),
        StructField("empirical_page_number", StringType(), True),
        StructField("body", StringType(), False)
    ])

    # Convert to DataFrame
    taxa_df = taxa_rdd.toDF(schema)

    # Analyze
    print(f"\nTotal paragraphs: {taxa_df.count()}")
    print("\nLabel distribution:")
    taxa_df.groupBy("label").count().show()

    print("\nTaxa per document:")
    taxa_df.groupBy("doc_id", "serial_number").count() \
        .groupBy("doc_id").count() \
        .withColumnRenamed("count", "taxa_count") \
        .show()

    # Save
    print("\nSaving to parquet...")
    taxa_df.write.mode("overwrite").parquet("output/taxa.parquet")

    print("\nSaving to CSV...")
    taxa_df.write.mode("overwrite").csv("output/taxa.csv", header=True)

    print("\nDone!")
    spark.stop()


if __name__ == "__main__":
    main()
```

## Run It

```bash
python quickstart_example.py
```

## Expected Output

```
Loading attachments from CouchDB...
Found 150 attachments

Processing partitions to extract taxa...

Total paragraphs: 842

Label distribution:
+---------------+-----+
|          label|count|
+---------------+-----+
|  Nomenclature|  421|
|   Description|  421|
+---------------+-----+

Taxa per document:
+--------------------+----------+
|              doc_id|taxa_count|
+--------------------+----------+
|  article_2023_001  |        12|
|  article_2023_002  |         8|
|  article_2023_003  |        15|
+--------------------+----------+

Saving to parquet...
Saving to CSV...

Done!
```

## What You Get

The output contains:

- **serial_number**: Unique taxon ID
- **db_name**: Source database ("mycobank")
- **doc_id**: CouchDB document ID
- **attachment_name**: Filename (e.g., "article.txt.ann")
- **label**: "Nomenclature" or "Description"
- **paragraph_number**: Sequential paragraph number
- **page_number**: Page in document
- **body**: Full paragraph text

## Next Steps

1. **Analyze results**: Use Spark SQL to query the taxa DataFrame
2. **Filter data**: Select specific documents or labels
3. **Export different formats**: JSON, Parquet, CSV, etc.
4. **Scale up**: Increase Spark cluster size for larger datasets

## Troubleshooting

### CouchDB Connection Error
```python
# Check connection
conn = CouchDBConnection("http://localhost:5984", "mycobank")
print(conn.db.info())  # Should print database info
```

### No Attachments Found
```python
# Verify pattern matching
df = conn.load_distributed(spark, "*.txt.ann")
df.select("doc_id", "attachment_name").show()
```

### Memory Issues
```python
# Increase executor memory
spark = SparkSession.builder \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

## More Information

- **Full Documentation**: [EXTRACTING_TAXON_OBJECTS.md](EXTRACTING_TAXON_OBJECTS.md)
- **Module Reference**: [couchdb_file_README.md](couchdb_file_README.md)
- **Complete Examples**: [examples/extract_taxa_from_couchdb.py](examples/extract_taxa_from_couchdb.py)
- **Integration Summary**: [COUCHDB_INTEGRATION_SUMMARY.md](COUCHDB_INTEGRATION_SUMMARY.md)

## Command-Line Alternative

Use the provided example script:

```bash
python examples/extract_taxa_from_couchdb.py \
    --mode distributed \
    --database mycobank_annotations \
    --db-name mycobank \
    --username admin \
    --password secret \
    --pattern "*.txt.ann"
```

That's it! You're now extracting taxa from CouchDB with full metadata tracking and distributed processing.
