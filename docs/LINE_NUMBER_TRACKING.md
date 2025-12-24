# Line Number Tracking in Section Extraction Mode

## Overview

Line numbers are now fully tracked and preserved through the entire prediction pipeline when using `extraction_mode='section'`. This enables proper sorting of extracted sections within each document.

## Implementation Details

### 1. PDFSectionExtractor Output Schema

The [PDFSectionExtractor](../pdf_section_extractor.py) has always included `line_number` in its output schema:

```python
schema = StructType([
    StructField("value", StringType(), False),
    StructField("doc_id", StringType(), False),
    StructField("attachment_name", StringType(), False),
    StructField("paragraph_number", IntegerType(), False),
    StructField("line_number", IntegerType(), False),  # ← Line number field
    StructField("page_number", IntegerType(), False),
    StructField("empirical_page_number", IntegerType(), True),
    StructField("section_name", StringType(), True)
])
```

The `line_number` field tracks the line number of the first line of each section/paragraph in the original document.

### 2. Classifier Integration (Fixed)

**Problem**: The classifier's `_load_annotated_from_couchdb()` method was using traditional text loading for all extraction modes, which didn't preserve `line_number` for section mode.

**Solution**: Updated [classifier_v2.py:888-927](../skol_classifier/classifier_v2.py#L888-L927) to use section extraction when in section mode:

```python
def _load_annotated_from_couchdb(self) -> DataFrame:
    """Load annotated data from CouchDB."""
    database = self.couchdb_training_database or self.couchdb_database

    # For 'section' extraction mode, use PDFSectionExtractor
    # which preserves line_number and other metadata
    if self.extraction_mode == 'section':
        return self._load_sections_from_couchdb(database=database)

    # For 'line' and 'paragraph' modes, use traditional text loading
    # ... (rest of method)
```

### 3. Feature Pipeline Preservation

PySpark ML pipelines automatically preserve all input columns during transformation. The `line_number` column flows through:

1. Tokenization
2. TF-IDF vectorization (words, suffixes, section names)
3. Feature assembly
4. Label indexing

All intermediate transformations preserve the `line_number` column.

### 4. Output Sorting

The [CouchDBOutputWriter](../skol_classifier/output_formatters.py#L328-L351) already has logic to sort predictions by `line_number` when the column is present:

```python
if "line_number" in predictions.columns:
    predictions = (
        predictions.groupBy(groupby_col, attachment_col)
        .agg(
            expr("sort_array(collect_list(struct(line_number, annotated_value)))").alias("sorted_list")
        )
        .withColumn("annotated_value_ordered", expr("transform(sorted_list, x -> x.annotated_value)"))
        .withColumn("final_aggregated_pg", expr("array_join(annotated_value_ordered, '\n')"))
        .select(groupby_col, attachment_col, "final_aggregated_pg")
    )
```

This ensures that when predictions are aggregated and saved back to CouchDB, they maintain the original document order.

## Usage Example

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.appName("Example").getOrCreate()

# Train classifier with section extraction mode
clf = SkolClassifierV2(
    spark=spark,
    extraction_mode='section',  # Enable section extraction
    input_source='couchdb',
    couchdb_database='skol_training',
    couchdb_doc_ids=['doc1', 'doc2', 'doc3'],
    output_dest='couchdb',
    verbosity=1
)

# Train model
stats = clf.fit()

# Make predictions (line_number preserved throughout)
predictions = clf.predict()

# Save to CouchDB (sorted by line_number automatically)
clf.save_annotated(predictions)
```

## Verification

The complete pipeline has been verified to preserve `line_number`:

1. ✅ PDFSectionExtractor outputs `line_number`
2. ✅ Classifier loads training data with `line_number` (after fix)
3. ✅ Feature pipeline preserves `line_number`
4. ✅ Model predictions include `line_number`
5. ✅ Output formatter sorts by `line_number`

## Files Modified

- [skol_classifier/classifier_v2.py](../skol_classifier/classifier_v2.py)
  - Line 888-927: Updated `_load_annotated_from_couchdb()` to use section extraction for section mode
  - Line 990-1075: Updated `_load_sections_from_couchdb()` to accept optional database parameter

## Related Documentation

- [PDF Section Extractor](./PDF_DATAFRAME_MIGRATION_SUMMARY.md)
- [Text Attachment Implementation](./TXT_ATTACHMENT_IMPLEMENTATION.md)
- [Extraction Mode Migration](./EXTRACTION_MODE_MIGRATION.md)
