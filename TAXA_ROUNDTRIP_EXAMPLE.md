# Taxa Extraction Round-Trip Example

## Overview

Complete example showing the full taxa extraction workflow:
1. Load annotated documents from CouchDB
2. Extract taxa using pattern matching
3. Save taxa to taxon database
4. Load taxa back from database
5. Verify data integrity

## Complete Workflow

### Step 1: Initialize TaxonExtractor

```python
from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import TaxonExtractor

# Create Spark session
spark = SparkSession.builder \
    .appName("Taxa Extraction Workflow") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Initialize extractor
extractor = TaxonExtractor(
    spark=spark,
    ingest_couchdb_url="http://localhost:5984",
    ingest_db_name="mycobank_annotations",
    taxon_db_name="mycobank_taxa",
    ingest_username="admin",
    ingest_password="password"
)
```

### Step 2: Load Annotated Documents

```python
# Load documents that have been annotated with YEDA format
annotated_df = extractor.load_annotated_documents()

print(f"Loaded {annotated_df.count()} annotated documents")

# Show structure
annotated_df.printSchema()
# root
#  |-- doc_id: string
#  |-- attachment_name: string
#  |-- value: string
#  |-- label: string
#  |-- paragraph_number: integer
#  |-- line_number: integer
#  |-- page_number: integer
#  |-- empirical_page_number: string

# Sample data
annotated_df.select("doc_id", "attachment_name", "label", "value").show(5, truncate=50)
```

### Step 3: Extract Taxa

```python
# Extract taxa from annotated paragraphs
extracted_df = extractor.extract_taxa(annotated_df)

print(f"Extracted {extracted_df.count()} taxa")

# Show schema
extracted_df.printSchema()
# root
#  |-- taxon: string (nullable = false)
#  |-- description: string (nullable = false)
#  |-- source: map (nullable = false)
#  |    |-- key: string
#  |    |-- value: string (valueContainsNull = true)
#  |-- line_number: integer (nullable = true)
#  |-- paragraph_number: integer (nullable = true)
#  |-- page_number: integer (nullable = true)
#  |-- empirical_page_number: string (nullable = true)

# Sample taxa
extracted_df.select("taxon", "description").show(5, truncate=50)
```

**Example Output**:
```
+--------------------------------------------------+--------------------------------------------------+
|taxon                                             |description                                       |
+--------------------------------------------------+--------------------------------------------------+
|Agaricus bisporus (J.E.Lange) Imbach sp. nov.   |Pileus 5–10 cm broad, convex, white to brown ... |
|Boletus edulis Bull. comb. nov.                  |Basidiomata fleshy, pores white becoming yell... |
+--------------------------------------------------+--------------------------------------------------+
```

### Step 4: Save Taxa to CouchDB

```python
# Save taxa to taxon database
save_results = extractor.save_taxa(extracted_df)

# Check results
total = save_results.count()
successes = save_results.filter("success = true").count()
failures = save_results.filter("success = false").count()

print(f"Save results:")
print(f"  Total: {total}")
print(f"  Success: {successes}")
print(f"  Failures: {failures}")

# Show failures (if any)
if failures > 0:
    print("\nFailed saves:")
    save_results.filter("success = false").select("doc_id", "error").show(truncate=50)
```

**Example Output**:
```
Save results:
  Total: 1234
  Success: 1234
  Failures: 0
```

### Step 5: Load Taxa from CouchDB

```python
# Load all taxa back from database
loaded_df = extractor.load_taxa()

print(f"Loaded {loaded_df.count()} taxa from CouchDB")

# Verify schema matches
loaded_df.printSchema()

# Sample loaded data
loaded_df.select("taxon", "source.doc_id", "source.attachment_name").show(5, truncate=50)
```

### Step 6: Verify Round-Trip Integrity

```python
from pyspark.sql.functions import col

# Compare counts
original_count = extracted_df.count()
loaded_count = loaded_df.count()

print(f"\nRound-trip verification:")
print(f"  Original taxa: {original_count}")
print(f"  Loaded taxa: {loaded_count}")
print(f"  Match: {'✓' if original_count == loaded_count else '✗'}")

# Verify specific taxon
sample_taxon = extracted_df.select("taxon", "description").first()
print(f"\nSample original taxon: {sample_taxon['taxon'][:50]}...")

# Find it in loaded data
loaded_sample = loaded_df.filter(
    col("taxon") == sample_taxon['taxon']
).select("taxon", "description").first()

if loaded_sample:
    print(f"Found in loaded data: {loaded_sample['taxon'][:50]}...")
    if loaded_sample['description'] == sample_taxon['description']:
        print("✓ Description matches")
    else:
        print("✗ Description differs")
else:
    print("✗ Taxon not found in loaded data")
```

## Pattern-Based Loading

### Load All Taxa

```python
# Load all taxa (default)
all_taxa = extractor.load_taxa()
print(f"All taxa: {all_taxa.count()}")
```

### Load Specific Subset

```python
# Load taxa with specific prefix
subset = extractor.load_taxa(pattern="taxon_abc*")
print(f"Taxa with prefix 'taxon_abc': {subset.count()}")
```

### Load by Exact ID

```python
# Load single taxon by exact ID
import hashlib

# Example: compute ID for a specific taxon
taxon_name = "Agaricus bisporus"
taxon_id = "taxon_" + hashlib.md5(taxon_name.encode()).hexdigest()

single_taxon = extractor.load_taxa(pattern=taxon_id)
if single_taxon.count() > 0:
    print(f"Found taxon: {single_taxon.first()['taxon']}")
```

## Analysis Examples

### Example 1: Taxa by Source Document

```python
from pyspark.sql.functions import col

# Load all taxa
taxa_df = extractor.load_taxa()

# Group by source document
by_source = taxa_df.groupBy("source.doc_id", "source.attachment_name").count()
by_source.orderBy(col("count").desc()).show()
```

**Output**:
```
+--------------------+--------------------+-----+
|              doc_id|     attachment_name|count|
+--------------------+--------------------+-----+
|doc_123             |its769.txt          |   45|
|doc_124             |its770.txt          |   38|
|doc_125             |its771.txt          |   32|
+--------------------+--------------------+-----+
```

### Example 2: New Species

```python
from pyspark.sql.functions import col

# Find new species (contain "sp. nov.")
new_species = taxa_df.filter(col("taxon").contains("sp. nov."))
print(f"Found {new_species.count()} new species")

new_species.select("taxon", "description").show(10, truncate=50)
```

### Example 3: Taxa by Page

```python
# Group by page number
by_page = taxa_df.groupBy("page_number").count()
by_page.orderBy("page_number").show()
```

### Example 4: Extract Genus Names

```python
from pyspark.sql.functions import split, col

# Extract first word (genus) from taxon name
genera = taxa_df.withColumn(
    "genus",
    split(col("taxon"), " ").getItem(0)
)

# Count unique genera
genus_counts = genera.groupBy("genus").count()
genus_counts.orderBy(col("count").desc()).show(20)
```

## Incremental Updates

### Scenario: Add New Annotated Documents

```python
# Initial extraction and save
annotated_df = extractor.load_annotated_documents()
extracted_df = extractor.extract_taxa(annotated_df)
save_results = extractor.save_taxa(extracted_df)
print(f"Saved {save_results.filter('success = true').count()} taxa")

# Later: Load and analyze
taxa_df = extractor.load_taxa()

# Add new annotated documents to CouchDB (external process)
# ...

# Re-run extraction
new_annotated_df = extractor.load_annotated_documents()
new_extracted_df = extractor.extract_taxa(new_annotated_df)

# Save new taxa (existing ones will be updated due to idempotent doc_id)
new_save_results = extractor.save_taxa(new_extracted_df)

# Reload to see updated data
updated_taxa_df = extractor.load_taxa()
print(f"Total taxa now: {updated_taxa_df.count()}")
```

## Error Handling

### Handle Missing Database

```python
try:
    taxa_df = extractor.load_taxa()
    print(f"Loaded {taxa_df.count()} taxa")
except Exception as e:
    print(f"Error loading taxa: {e}")
    # Database might not exist yet
    # Create it by running save_taxa() first
```

### Verify Save Success

```python
# Save taxa and check for failures
save_results = extractor.save_taxa(extracted_df)

failures = save_results.filter("success = false")
failure_count = failures.count()

if failure_count > 0:
    print(f"⚠ {failure_count} taxa failed to save")
    print("\nFailure details:")
    failures.select("doc_id", "error").show(truncate=False)

    # Retry failed ones
    failed_taxa = extracted_df.join(
        failures.select("doc_id"),
        extracted_df["source.doc_id"] == failures["doc_id"]
    )

    retry_results = extractor.save_taxa(failed_taxa)
    print(f"Retry: {retry_results.filter('success = true').count()} succeeded")
```

## Performance Optimization

### Batch Processing

```python
# For very large datasets, process in batches by pattern
patterns = ["taxon_a*", "taxon_b*", "taxon_c*", "taxon_d*"]

for pattern in patterns:
    print(f"\nProcessing {pattern}...")
    taxa_batch = extractor.load_taxa(pattern=pattern)
    count = taxa_batch.count()
    print(f"  Loaded {count} taxa")

    # Analyze this batch
    new_species = taxa_batch.filter(col("taxon").contains("sp. nov.")).count()
    print(f"  New species: {new_species}")
```

### Caching for Multiple Operations

```python
# Load once and cache
taxa_df = extractor.load_taxa()
taxa_df.cache()

# Multiple operations on cached data
print(f"Total taxa: {taxa_df.count()}")
print(f"New species: {taxa_df.filter(col('taxon').contains('sp. nov.')).count()}")
print(f"Unique genera: {taxa_df.select('taxon').distinct().count()}")

# Unpersist when done
taxa_df.unpersist()
```

### Repartition for Better Parallelism

```python
# Load and repartition
taxa_df = extractor.load_taxa()
taxa_df = taxa_df.repartition(200)  # Adjust based on cluster size

# Now operations will be more parallel
taxa_df.write.parquet("taxa_export.parquet")
```

## Integration with SKOL Classifier

### Classify and Extract in Pipeline

```python
from skol_classifier.classifier_v2 import SkolClassifierV2

# 1. Classify raw documents
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    couchdb_url="http://localhost:5984",
    db_name="mycobank_raw",
    line_level=False,
    model_type='logistic',
    output_format='annotated',
    coalesce_labels=True
)

# Train and predict
results = classifier.fit()
predictions = classifier.predict()

# 2. Save annotated results
classifier.save_annotated(predictions)

# 3. Extract taxa from annotated documents
extractor = TaxonExtractor(
    spark=spark,
    ingest_couchdb_url="http://localhost:5984",
    ingest_db_name="mycobank_raw",  # Same as classifier output
    taxon_db_name="mycobank_taxa",
    ingest_username="admin",
    ingest_password="password"
)

annotated_df = extractor.load_annotated_documents()
extracted_df = extractor.extract_taxa(annotated_df)
save_results = extractor.save_taxa(extracted_df)

# 4. Load and analyze
taxa_df = extractor.load_taxa()
print(f"Extracted {taxa_df.count()} taxa from classified documents")
```

## Complete Example Script

```python
#!/usr/bin/env python3
"""Complete taxa extraction round-trip example."""

from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import TaxonExtractor

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Taxa Extraction Round-Trip") \
        .master("local[*]") \
        .getOrCreate()

    try:
        # Initialize extractor
        extractor = TaxonExtractor(
            spark=spark,
            ingest_couchdb_url="http://localhost:5984",
            ingest_db_name="mycobank_annotations",
            taxon_db_name="mycobank_taxa",
            ingest_username="admin",
            ingest_password="password"
        )

        print("=== TAXA EXTRACTION ROUND-TRIP ===\n")

        # Step 1: Load annotated documents
        print("1. Loading annotated documents...")
        annotated_df = extractor.load_annotated_documents()
        print(f"   Loaded {annotated_df.count()} documents\n")

        # Step 2: Extract taxa
        print("2. Extracting taxa...")
        extracted_df = extractor.extract_taxa(annotated_df)
        original_count = extracted_df.count()
        print(f"   Extracted {original_count} taxa\n")

        # Step 3: Save to CouchDB
        print("3. Saving taxa to CouchDB...")
        save_results = extractor.save_taxa(extracted_df)
        successes = save_results.filter("success = true").count()
        failures = save_results.filter("success = false").count()
        print(f"   Success: {successes}, Failures: {failures}\n")

        # Step 4: Load from CouchDB
        print("4. Loading taxa from CouchDB...")
        loaded_df = extractor.load_taxa()
        loaded_count = loaded_df.count()
        print(f"   Loaded {loaded_count} taxa\n")

        # Step 5: Verify
        print("5. Verification:")
        print(f"   Original: {original_count}")
        print(f"   Saved: {successes}")
        print(f"   Loaded: {loaded_count}")

        if original_count == loaded_count == successes:
            print("   ✓ Round-trip successful!\n")
        else:
            print("   ✗ Counts don't match\n")

        # Step 6: Analyze
        print("6. Sample analysis:")
        loaded_df.groupBy("source.db_name").count().show()

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
```

## Related Documentation

- [TAXON_LOAD_METHOD.md](TAXON_LOAD_METHOD.md) - Implementation details
- [TEST_LOAD_TAXA.md](TEST_LOAD_TAXA.md) - Test suite
- [COUCHDB_INTEGRATION.md](COUCHDB_INTEGRATION.md) - CouchDB setup
- [CLASSIFIER_V2_API.md](CLASSIFIER_V2_API.md) - Classifier integration

## Summary

The complete taxa extraction workflow enables:
- ✅ Loading annotated documents from CouchDB
- ✅ Extracting taxa using pattern matching
- ✅ Saving taxa with idempotent document IDs
- ✅ Loading taxa with pattern-based filtering
- ✅ Round-trip data integrity
- ✅ Integration with SKOL classifier
- ✅ Distributed processing for scalability
