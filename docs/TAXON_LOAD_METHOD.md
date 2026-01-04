# TaxonExtractor.load_taxa() Method

## Overview

Added `load_taxa()` method to `TaxonExtractor` class that performs the inverse operation of `save_taxa()`, loading taxa from CouchDB and converting them back to a PySpark DataFrame.

## Implementation

### Method Signature

```python
def load_taxa(self, pattern: str = "taxon_*") -> DataFrame:
    """
    Load taxa from CouchDB taxon database.

    Args:
        pattern: Pattern for document IDs to load (default: "taxon_*")
                Use "*" to load all documents
                Use "taxon_abc*" to load specific subset

    Returns:
        DataFrame with taxa information matching the extract_taxa() schema
    """
```

### Architecture

The method uses a distributed UDF-based approach via `mapPartitions`:

1. **Get document IDs**: Query CouchDB to get list of matching document IDs
2. **Parallelize IDs**: Create RDD from document ID list
3. **Load per partition**: Each partition connects to CouchDB once and loads its documents
4. **Return DataFrame**: Convert loaded documents to DataFrame with consistent schema

### Key Design Decisions

#### 1. Pattern-Based Loading

Supports flexible pattern matching for document IDs:

```python
# Load all taxa
taxa_df = extractor.load_taxa()

# Load all taxa (explicit)
taxa_df = extractor.load_taxa(pattern="*")

# Load specific prefix
taxa_df = extractor.load_taxa(pattern="taxon_abc")

# Load with wildcard
taxa_df = extractor.load_taxa(pattern="taxon_abc*")

# Load exact document
taxa_df = extractor.load_taxa(pattern="taxon_123abc")
```

#### 2. Distributed Processing

Uses `mapPartitions` for efficient distributed loading:

```python
def load_partition(partition: Iterator[Row]) -> Iterator[Row]:
    """Load taxa from CouchDB for an entire partition."""
    # Connect to CouchDB once per partition (not per document)
    server = couchdb.Server(couchdb_url)
    db = server[db_name]

    # Process each document ID in the partition
    for row in partition:
        doc_id = row.doc_id
        if doc_id in db:
            doc = db[doc_id]
            # Convert to Row and yield
            yield Row(**taxon_data)
```

**Benefits**:
- One CouchDB connection per partition (not per document)
- Parallel processing across Spark workers
- Fault tolerant (each partition is independent)

#### 3. Schema Consistency

Returns DataFrame with same schema as `extract_taxa()`:

```python
StructType([
    StructField("taxon", StringType(), False),
    StructField("description", StringType(), False),
    StructField("source", MapType(StringType(), StringType()), False),
    StructField("line_number", IntegerType(), True),
    StructField("paragraph_number", IntegerType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("empirical_page_number", StringType(), True),
])
```

This ensures loaded taxa can be:
- Processed with same code as extracted taxa
- Re-saved without schema changes
- Used in downstream analytics

## Usage Examples

### Example 1: Load All Taxa

```python
from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import TaxonExtractor

spark = SparkSession.builder.appName("LoadTaxa").getOrCreate()

extractor = TaxonExtractor(
    spark=spark,
    ingest_couchdb_url="http://localhost:5984",
    ingest_db_name="mycobank_annotations",
    taxon_db_name="mycobank_taxa",
    ingest_username="admin",
    ingest_password="secret"
)

# Load all taxa
taxa_df = extractor.load_taxa()

print(f"Loaded {taxa_df.count()} taxa")
taxa_df.show(10)

# Analyze taxa
taxa_df.groupBy("source.db_name").count().show()
```

### Example 2: Load Specific Subset

```python
# Load taxa with specific prefix
subset_df = extractor.load_taxa(pattern="taxon_abc*")

print(f"Loaded {subset_df.count()} taxa with prefix 'taxon_abc'")

# Filter and process
nomenclature_taxa = subset_df.filter(
    col("taxon").contains("sp. nov.")
)

print(f"Found {nomenclature_taxa.count()} new species")
```

### Example 3: Round-Trip Test

```python
# Extract taxa from annotated documents
annotated_df = extractor.load_annotated_documents()
extracted_df = extractor.extract_taxa(annotated_df)

print(f"Extracted {extracted_df.count()} taxa")

# Save to CouchDB
save_results = extractor.save_taxa(extracted_df)
successes = save_results.filter("success = true").count()

print(f"Saved {successes} taxa to CouchDB")

# Load back from CouchDB
loaded_df = extractor.load_taxa()

print(f"Loaded {loaded_df.count()} taxa from CouchDB")

# Verify round-trip consistency
assert extracted_df.count() == loaded_df.count()
print("✓ Round-trip successful!")
```

### Example 4: Update Taxa

```python
# Load existing taxa
taxa_df = extractor.load_taxa()

# Add computed field
from pyspark.sql.functions import length

taxa_with_length = taxa_df.withColumn(
    "description_length",
    length(col("description"))
)

# Convert back to save format and update
# (Note: This would require additional processing to maintain CouchDB _rev)
```

## Helper Method: _get_matching_doc_ids

Private method to query CouchDB for matching document IDs:

```python
def _get_matching_doc_ids(self, pattern: str) -> list:
    """
    Get list of document IDs matching the pattern from CouchDB.

    Args:
        pattern: Pattern for document IDs

    Returns:
        List of matching document IDs
    """
```

**Pattern Matching Logic**:
- `"*"` → All documents (excluding `_design/` documents)
- `"prefix*"` → All documents starting with "prefix"
- `"exact"` → Single document with exact ID

**Implementation**:
```python
# Get all document IDs from database
all_doc_ids = [doc_id for doc_id in db if not doc_id.startswith('_design/')]

# Apply pattern filter
if pattern == "*":
    return all_doc_ids
elif pattern.endswith('*'):
    prefix = pattern[:-1]
    return [doc_id for doc_id in all_doc_ids if doc_id.startswith(prefix)]
else:
    return [doc_id for doc_id in all_doc_ids if doc_id == pattern]
```

## Performance Considerations

### Efficiency

1. **Batch Loading**: Connects to CouchDB once per partition, not per document
2. **Parallel Processing**: Distributes load across Spark workers
3. **Lazy Evaluation**: DataFrame operations are lazy until action is called

### Scalability

For large taxon databases:

```python
# Load in batches by pattern
patterns = ["taxon_a*", "taxon_b*", "taxon_c*"]

all_taxa = []
for pattern in patterns:
    batch_df = extractor.load_taxa(pattern=pattern)
    all_taxa.append(batch_df)

# Union all batches
from functools import reduce
complete_df = reduce(DataFrame.union, all_taxa)
```

### Optimization Tips

1. **Use specific patterns** when possible to reduce documents to load
2. **Repartition** if needed for better parallelism:
   ```python
   taxa_df = extractor.load_taxa()
   taxa_df = taxa_df.repartition(200)  # Increase partitions
   ```
3. **Cache** if reusing:
   ```python
   taxa_df = extractor.load_taxa()
   taxa_df.cache()
   # Multiple operations on taxa_df
   ```

## Comparison: save_taxa() vs load_taxa()

| Aspect | save_taxa() | load_taxa() |
|--------|-------------|-------------|
| **Direction** | DataFrame → CouchDB | CouchDB → DataFrame |
| **Input** | DataFrame with taxa | Pattern string |
| **Output** | Results DataFrame (success/failure) | Taxa DataFrame |
| **Schema** | Extract schema → CouchDB docs | CouchDB docs → Extract schema |
| **Idempotency** | Yes (upserts based on doc_id) | Yes (reads are naturally idempotent) |
| **Parallelization** | mapPartitions on taxa rows | mapPartitions on doc_id rows |
| **Error Handling** | Returns success/error per row | Skips failed docs, prints errors |

## Error Handling

The method handles errors gracefully:

```python
def load_partition(partition: Iterator[Row]) -> Iterator[Row]:
    try:
        # Connect and load
        server = couchdb.Server(couchdb_url)
        db = server[db_name]

        for row in partition:
            try:
                # Load document
                if doc_id in db:
                    yield Row(**taxon_data)
                else:
                    print(f"Document {doc_id} not found")
            except Exception as e:
                print(f"Error loading taxon {doc_id}: {e}")

    except Exception as e:
        print(f"Error connecting to CouchDB: {e}")
```

**Error Strategy**:
- Connection errors: Print error, skip partition
- Document errors: Print error, skip document
- Missing documents: Print warning, continue
- Invalid data: Print error, skip document

## Testing

```python
def test_load_taxa():
    """Test load_taxa() functionality."""
    spark = SparkSession.builder.getOrCreate()

    extractor = TaxonExtractor(
        spark=spark,
        ingest_couchdb_url="http://localhost:5984",
        ingest_db_name="test_ingest",
        taxon_db_name="test_taxa",
        ingest_username="admin",
        ingest_password="password"
    )

    # Test 1: Load all taxa
    all_taxa = extractor.load_taxa()
    assert all_taxa.count() > 0
    print("✓ Load all taxa works")

    # Test 2: Load with pattern
    subset = extractor.load_taxa(pattern="taxon_test*")
    assert subset.count() <= all_taxa.count()
    print("✓ Pattern matching works")

    # Test 3: Schema verification
    expected_cols = {"taxon", "description", "source", "line_number"}
    actual_cols = set(all_taxa.columns)
    assert expected_cols.issubset(actual_cols)
    print("✓ Schema is correct")

    # Test 4: Empty pattern
    empty = extractor.load_taxa(pattern="nonexistent*")
    assert empty.count() == 0
    print("✓ Empty result works")

    print("\n✓ All tests passed!")
```

## Future Enhancements

Potential improvements:

1. **Advanced Pattern Matching**: Support regex patterns
2. **Incremental Loading**: Load only documents modified since timestamp
3. **Selective Fields**: Load only specific fields to reduce memory
4. **CouchDB Views**: Use views for more efficient filtering
5. **Update Support**: Method to update existing taxa in CouchDB

## Related Methods

- `load_annotated_documents()`: Loads annotated documents from ingest DB
- `extract_taxa()`: Extracts taxa from annotated documents
- `save_taxa()`: Saves taxa DataFrame to CouchDB
- `run_pipeline()`: Complete extract and save pipeline

## Summary

The `load_taxa()` method provides:
- ✅ Inverse operation of `save_taxa()`
- ✅ Distributed loading via mapPartitions
- ✅ Pattern-based filtering
- ✅ Schema consistency
- ✅ Error handling
- ✅ Efficient batch loading
- ✅ Integration with existing TaxonExtractor workflow

This enables full round-trip capability: annotated documents → taxa extraction → CouchDB storage → DataFrame loading → further processing.
