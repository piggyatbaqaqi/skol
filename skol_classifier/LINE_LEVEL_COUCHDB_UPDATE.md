# Line-Level Support for CouchDB Predictions

## Summary

Added support for the `line_level` argument to `SkolClassifier.predict_from_couchdb()` method, enabling line-by-line classification of documents stored in CouchDB.

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

1. **Adds line numbers**: Creates a `line_number` column if not already present in the CouchDB data
2. **Filters empty lines**: Removes blank lines from processing
3. **Preserves document context**: Maintains `doc_id` and `attachment_name` for tracking source documents
4. **Orders by line number**: Ensures lines are processed in their original order

```python
if line_level:
    # Line-level processing
    from pyspark.sql.window import Window

    # Window specification for adding line numbers (needs a basic ordering first)
    window_spec_init = Window.partitionBy("doc_id", "attachment_name").orderBy(lit(1))

    # Add line numbers if not present
    if "line_number" not in df.columns:
        df = df.withColumn("line_number", row_number().over(window_spec_init) - 1)

    # Window specification for final ordering by line number
    window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy("line_number")

    # Filter empty lines and add row numbers
    processed_df = (
        df.filter(trim(col("value")) != "")
        .withColumn("row_number", row_number().over(window_spec))
    )
```

### Paragraph-Level Processing (Unchanged)

When `line_level=False`, the method continues to use the original paragraph extraction logic:

```python
else:
    # Paragraph-level processing
    from .preprocessing import ParagraphExtractor
    from pyspark.sql.types import ArrayType, StringType
    from pyspark.sql.window import Window

    heuristic_udf = udf(
        ParagraphExtractor.extract_heuristic_paragraphs,
        ArrayType(StringType())
    )

    # Window specification for ordering
    window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy("start_idx")

    # Group and extract paragraphs
    processed_df = (
        df.groupBy("doc_id", "attachment_name")
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
