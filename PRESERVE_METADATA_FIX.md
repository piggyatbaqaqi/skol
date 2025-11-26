# Preserve Metadata Columns Fix

## Issue

**Problem**: The `attachment_name` metadata column was being dropped during the prediction pipeline, preventing proper aggregation and saving of predictions to CouchDB.

**User Report**: "The attachment_name metadata has been removed by the time we're looking at the output of classifier.predict(). This should be preserved from data ingestion."

## Root Causes

### 1. Column Selection in _format_predictions

The `_format_predictions` method was using `.select()` with hardcoded column lists that didn't include `attachment_name`:

```python
# Before - drops attachment_name
elif self.output_format == 'labels':
    return predictions_df.select("doc_id", "predicted_label")
elif self.output_format == 'probs':
    return predictions_df.select("doc_id", "predicted_label", "probability")
```

### 2. Inconsistent Column Naming in Data Loaders

The `RawTextLoader.load_from_couchdb` method was creating `row_number` instead of `line_number`, causing inconsistency with other parts of the pipeline.

## Solutions

### 1. Preserve attachment_name in _format_predictions

Modified the method to conditionally include `attachment_name` if present:

```python
# After - preserves attachment_name
elif self.output_format == 'labels':
    cols = ["doc_id"]
    if "attachment_name" in predictions_df.columns:
        cols.append("attachment_name")
    cols.append("predicted_label")
    return predictions_df.select(*cols)
elif self.output_format == 'probs':
    cols = ["doc_id"]
    if "attachment_name" in predictions_df.columns:
        cols.append("attachment_name")
    cols.extend(["predicted_label", "probability"])
    return predictions_df.select(*cols)
```

### 2. Standardize Column Naming

Changed `row_number` to `line_number` in `RawTextLoader.load_from_couchdb`:

```python
# Before
.withColumn("row_number", row_number().over(window_spec))

# After
.withColumn("line_number", row_number().over(window_spec))
```

## Files Modified

### 1. [skol_classifier/classifier_v2.py](skol_classifier/classifier_v2.py:535-555)

**Method**: `_format_predictions`

**Changes**:
- Added conditional logic to preserve `attachment_name` when selecting columns
- Applied to both 'labels' and 'probs' output formats
- Maintains backward compatibility with file-based inputs (which don't have attachment_name)

**Before**:
```python
def _format_predictions(self, predictions_df: DataFrame) -> DataFrame:
    """Format predictions according to output_format setting."""
    if self.output_format == 'annotated':
        return self._format_as_annotated(predictions_df)
    elif self.output_format == 'labels':
        return predictions_df.select("doc_id", "predicted_label")
    elif self.output_format == 'probs':
        return predictions_df.select("doc_id", "predicted_label", "probability")
    else:
        return predictions_df
```

**After**:
```python
def _format_predictions(self, predictions_df: DataFrame) -> DataFrame:
    """Format predictions according to output_format setting."""
    if self.output_format == 'annotated':
        return self._format_as_annotated(predictions_df)
    elif self.output_format == 'labels':
        # Return just labels (preserve attachment_name if present)
        cols = ["doc_id"]
        if "attachment_name" in predictions_df.columns:
            cols.append("attachment_name")
        cols.append("predicted_label")
        return predictions_df.select(*cols)
    elif self.output_format == 'probs':
        # Return probabilities (preserve attachment_name if present)
        cols = ["doc_id"]
        if "attachment_name" in predictions_df.columns:
            cols.append("attachment_name")
        cols.extend(["predicted_label", "probability"])
        return predictions_df.select(*cols)
    else:
        return predictions_df
```

### 2. [skol_classifier/data_loaders.py](skol_classifier/data_loaders.py:355-377)

**Method**: `RawTextLoader.load_from_couchdb`

**Changes**:
- Changed `row_number` to `line_number` (lines 361 and 376)
- Ensures consistency with other data loading methods

**Before**:
```python
if line_level:
    # Line-level: explode lines and add row numbers
    window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy("doc_id")
    return (
        df.withColumn("value", explode(col("lines")))
        .drop("lines")
        .withColumn("row_number", row_number().over(window_spec))
    )
else:
    # Paragraph-level: use heuristic extraction
    ...
    return (
        df.withColumn("value", explode(heuristic_udf(col("lines"))))
        .drop("lines")
        .filter(trim(col("value")) != "")
        .withColumn("row_number", row_number().over(window_spec))
    )
```

**After**:
```python
if line_level:
    # Line-level: explode lines and add line numbers
    window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy("doc_id")
    return (
        df.withColumn("value", explode(col("lines")))
        .drop("lines")
        .withColumn("line_number", row_number().over(window_spec))
    )
else:
    # Paragraph-level: use heuristic extraction
    ...
    return (
        df.withColumn("value", explode(heuristic_udf(col("lines"))))
        .drop("lines")
        .filter(trim(col("value")) != "")
        .withColumn("line_number", row_number().over(window_spec))
    )
```

## Impact

### Benefits

1. **CouchDB Output Works**: Predictions can now be properly saved back to CouchDB because `attachment_name` is preserved
2. **Proper Aggregation**: Output formatters can group predictions by both `doc_id` and `attachment_name`
3. **Backward Compatible**: File-based inputs (which don't have `attachment_name`) continue to work
4. **Consistent Naming**: All data loaders now use `line_number` consistently

### Example Flow

**Before (Broken)**:
```
CouchDB Load → doc_id, attachment_name, value, line_number
    ↓
Feature Extraction → doc_id, attachment_name, value, line_number, features
    ↓
Model Prediction → doc_id, attachment_name, value, line_number, prediction
    ↓
Format Predictions → doc_id, predicted_label  ❌ attachment_name lost!
    ↓
Save to CouchDB → ❌ Can't aggregate by attachment_name
```

**After (Fixed)**:
```
CouchDB Load → doc_id, attachment_name, value, line_number
    ↓
Feature Extraction → doc_id, attachment_name, value, line_number, features
    ↓
Model Prediction → doc_id, attachment_name, value, line_number, prediction
    ↓
Format Predictions → doc_id, attachment_name, predicted_label  ✅ Preserved!
    ↓
Save to CouchDB → ✅ Can aggregate by (doc_id, attachment_name)
```

## Testing

To verify the fix:

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.getOrCreate()

# Test with CouchDB input/output
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    couchdb_url='http://localhost:5984',
    couchdb_database='mydb',
    couchdb_username='admin',
    couchdb_password='password',
    couchdb_pattern='*.txt',
    line_level=True,
    output_format='annotated',
    output_dest='couchdb',
    model_type='logistic'
)

# Train
results = classifier.fit()

# Predict - attachment_name should be preserved
predictions = classifier.predict()

# Verify columns
print(predictions.columns)  # Should include 'attachment_name'

# Save to CouchDB - should work now
classifier.save_annotated(predictions)
```

## Related Issues Fixed

This fix resolves:
- ✅ `attachment_name` preserved through prediction pipeline
- ✅ CouchDB output aggregation works correctly
- ✅ Consistent column naming (`line_number` everywhere)
- ✅ Backward compatibility with file-based inputs

## Notes

- The `attachment_name` column is only present when loading from CouchDB
- File-based inputs use `filename` instead and work as expected
- Output formatters check for both column names and handle appropriately
- The conditional logic ensures the code works with both CouchDB and file inputs
