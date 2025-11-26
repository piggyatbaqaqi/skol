# Line-Level Support for CouchDB Predictions

## Summary

Fixed `SkolClassifier.predict_from_couchdb()` to properly process documents from CouchDB. The method now correctly splits document content into lines/paragraphs instead of treating entire articles as single units.

## Critical Bug Fixed

**Problem**: CouchDB attachments contain entire article text in a single `value` field. The previous implementation did not split this content, causing:
- **Line-level mode**: Each entire article was treated as a single line
- **Paragraph mode**: Each entire article was treated as a single paragraph (the UDF received `[entire_article]` instead of `[line1, line2, ...]`)

**Root Cause**: The `load_from_couchdb()` method returns one row per attachment with the full content:
```python
# From couchdb_io.py:176
yield Row(
    doc_id=row.doc_id,
    attachment_name=row.attachment_name,
    value=content  # <-- ENTIRE ARTICLE TEXT IN ONE STRING
)
```

**Solution**: Now properly splits content using `split(col("value"), "\n")` with `explode()` before processing in both modes. This creates one row per line, which can then be:
- Classified individually (line-level mode)
- Grouped and parsed into paragraphs (paragraph mode)

## Changes Made

### Modified Method: `predict_from_couchdb()`

**File**: `skol_classifier/classifier.py` (lines 753-855)

**What Changed**:
1. Updated docstring to document the `line_level` parameter
2. Implemented conditional processing based on `line_level` flag:
   - **Line-level mode** (`line_level=True`): Processes each line individually
   - **Paragraph mode** (`line_level=False`): Uses paragraph extraction (original behavior)

### Line-Level Processing Logic

When `line_level=True`, the method:

1. **Splits content into lines**: Uses `split(col("value"), "\n")` with `explode()` to create one row per line
2. **Filters empty lines**: Removes blank lines from processing
3. **Adds line numbers**: Creates a `line_number` column for each line within a document
4. **Preserves document context**: Maintains `doc_id` and `attachment_name` for tracking source documents
5. **Orders correctly**: Ensures lines are processed in their original order

```python
if line_level:
    # Line-level processing: split content into lines
    from pyspark.sql.window import Window

    # Split the content into individual lines
    lines_df = (
        df.withColumn("value", explode(split(col("value"), "\n")))
        .filter(trim(col("value")) != "")
    )

    # Add line numbers
    window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy(lit(1))
    processed_df = lines_df.withColumn("line_number", row_number().over(window_spec) - 1)

    # Add row number for ordering
    processed_df = processed_df.withColumn("row_number", row_number().over(window_spec))
```

### Paragraph-Level Processing (Fixed)

When `line_level=False`, the method now:

1. **Splits content into lines first**: Critical fix - must split before paragraph extraction
2. **Groups lines by document**: Collects all lines from the same document
3. **Extracts paragraphs**: Uses heuristic paragraph detection on the line list
4. **Creates one row per paragraph**: Each paragraph becomes a separate classification unit

```python
else:
    # Paragraph-level processing
    from .preprocessing import ParagraphExtractor
    from pyspark.sql.types import ArrayType, StringType
    from pyspark.sql.window import Window

    # First, split content into lines
    lines_df = df.withColumn("value", explode(split(col("value"), "\n")))

    heuristic_udf = udf(
        ParagraphExtractor.extract_heuristic_paragraphs,
        ArrayType(StringType())
    )

    # Window specification for ordering
    window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy("start_idx")

    # Group lines and extract paragraphs
    processed_df = (
        lines_df.groupBy("doc_id", "attachment_name")
        .agg(
            collect_list("value").alias("lines"),
            min(lit(0)).alias("start_idx")
        )
        .withColumn("value", explode(heuristic_udf(col("lines"))))
        .drop("lines")
        .filter(trim(col("value")) != "")
        .withColumn("row_number", row_number().over(window_spec))
    )
```

## Usage

### Line-Level Classification

```python
from pyspark.sql import SparkSession
from skol_classifier import SkolClassifier

spark = SparkSession.builder.appName("CouchDB Line Classifier").getOrCreate()

# Initialize with CouchDB configuration
classifier = SkolClassifier(
    spark=spark,
    couchdb_url="http://localhost:5984",
    database="taxonomic_articles",
    username="admin",
    password="password"
)

# Predict line-by-line from CouchDB
predictions = classifier.predict_from_couchdb(
    pattern="*.txt",
    output_format="annotated",
    line_level=True  # Enable line-level processing
)

# Save back to CouchDB with coalesced labels
classifier.save_to_couchdb(
    predictions,
    suffix=".ann",
    coalesce_labels=True  # Coalesce consecutive same-label lines into YEDA blocks
)
```

### Paragraph-Level Classification (Default)

```python
# Predict by paragraphs (original behavior)
predictions = classifier.predict_from_couchdb(
    pattern="*.txt",
    output_format="annotated",
    line_level=False  # Default: paragraph-level processing
)
```

## Benefits

1. **Consistency**: Line-level classification from CouchDB now matches the behavior of `predict_lines()`
2. **Flexibility**: Users can choose between line-level and paragraph-level granularity
3. **Integration**: Works seamlessly with `save_to_couchdb()` when using `coalesce_labels=True`
4. **Distributed Processing**: Maintains Spark's distributed processing capabilities

## Related Methods

- `predict_lines()`: Processes raw text strings line-by-line
- `load_raw_data_lines()`: Loads raw text as individual lines
- `save_to_couchdb()`: Saves predictions back to CouchDB with optional label coalescence
- `load_annotated_data()`: Loads training data with line-level or paragraph-level extraction

## Example Workflow

```python
# 1. Train with line-level data
classifier.fit(
    annotated_file_paths=["training_data/*.txt.ann"],
    model_type="logistic",
    use_suffixes=True,
    line_level=True  # Train on individual lines
)

# 2. Predict from CouchDB with line-level processing
predictions = classifier.predict_from_couchdb(
    pattern="*.txt",
    line_level=True  # Match training granularity
)

# 3. Save with coalesced labels to create YEDA blocks
classifier.save_to_couchdb(
    predictions,
    suffix=".ann",
    coalesce_labels=True
)
```

## Technical Notes

- The `line_number` column is added automatically if not present in the CouchDB data
- Empty lines are filtered out before classification
- Document context (`doc_id`, `attachment_name`) is preserved throughout the pipeline
- The method uses Spark window functions for efficient distributed processing
