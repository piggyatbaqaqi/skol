# Preserve attachment_name in Coalesced Output

## Issue

**Problem**: When `coalesce_labels=True`, the `YedaFormatter.coalesce_consecutive_labels()` method was grouping only by a single column (`filename` or `doc_id`), which lost the `attachment_name` column needed for CouchDB operations.

**User Report**: "In jupyter/its769_skol.ipynb the clause after 'Extract the taxa names and descriptions' still gives an error about a missing attachment_name field in 'Show sample predictions'"

## Root Cause

The `coalesce_consecutive_labels` method was grouping by only one column:

```python
# Before - loses attachment_name
groupby_col = "filename" if "filename" in predictions.columns else "doc_id"

return (
    predictions
    .groupBy(groupby_col)  # Only groups by one column!
    .agg(...)
    .select(groupby_col, "coalesced_annotations")  # Only selects one column!
)
```

For CouchDB data, documents can have multiple attachments, so we need to group by BOTH `doc_id` AND `attachment_name` to keep them separate.

## Solution

Modified the grouping logic to preserve both `doc_id` and `attachment_name` when both are present:

```python
# After - preserves both columns
if "attachment_name" in predictions.columns:
    groupby_cols = ["doc_id", "attachment_name"]
elif "filename" in predictions.columns:
    groupby_cols = ["filename"]
else:
    groupby_cols = ["doc_id"]

return (
    predictions
    .groupBy(*groupby_cols)  # Groups by all necessary columns
    .agg(...)
    .select(*groupby_cols, "coalesced_annotations")  # Preserves all grouping columns
)
```

## Files Modified

### [skol_classifier/output_formatters.py](skol_classifier/output_formatters.py:144-165)

**Method**: `YedaFormatter.coalesce_consecutive_labels`

**Changes**:
- Lines 144-152: Added logic to determine appropriate grouping columns
- Line 157: Changed to `groupBy(*groupby_cols)` to support multiple columns
- Line 164: Changed to `select(*groupby_cols, "coalesced_annotations")` to preserve all columns

**Before**:
```python
# Check if DataFrame has filename or doc_id
groupby_col = "filename" if "filename" in predictions.columns else "doc_id"

# Group by document and coalesce
return (
    predictions
    .groupBy(groupby_col)
    .agg(
        collect_list(
            expr("struct(line_number, value, predicted_label)")
        ).alias("rows")
    )
    .withColumn("coalesced_annotations", coalesce_udf(col("rows")))
    .select(groupby_col, "coalesced_annotations")
)
```

**After**:
```python
# Determine grouping columns
# For CouchDB data, group by both doc_id and attachment_name
# For file data, group by filename only
if "attachment_name" in predictions.columns:
    groupby_cols = ["doc_id", "attachment_name"]
elif "filename" in predictions.columns:
    groupby_cols = ["filename"]
else:
    groupby_cols = ["doc_id"]

# Group by document and coalesce
return (
    predictions
    .groupBy(*groupby_cols)
    .agg(
        collect_list(
            expr("struct(line_number, value, predicted_label)")
        ).alias("rows")
    )
    .withColumn("coalesced_annotations", coalesce_udf(col("rows")))
    .select(*groupby_cols, "coalesced_annotations")
)
```

## Impact

### Before (Broken)

**CouchDB data structure**:
```
doc_id="123", attachment_name="article.txt", line_number=1, value="line1", predicted_label="Nomenclature"
doc_id="123", attachment_name="article.txt", line_number=2, value="line2", predicted_label="Nomenclature"
doc_id="123", attachment_name="supplement.txt", line_number=1, value="line3", predicted_label="Description"
```

**After coalescing** (❌ Lost attachment_name):
```
doc_id="123", coalesced_annotations=["[@ line1\nline2\nline3 #...]"]
```
All attachments merged together! Can't distinguish between article.txt and supplement.txt.

### After (Fixed)

**After coalescing** (✅ Preserved attachment_name):
```
doc_id="123", attachment_name="article.txt", coalesced_annotations=["[@ line1\nline2 #Nomenclature*]"]
doc_id="123", attachment_name="supplement.txt", coalesced_annotations=["[@ line3 #Description*]"]
```
Each attachment is kept separate!

## Usage Impact

### For File-based Inputs

No change in behavior - still groups by `filename`:
```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/*.ann'],
    line_level=True,
    coalesce_labels=True
)
predictions = classifier.predict()
# Result: filename, coalesced_annotations
```

### For CouchDB Inputs

Now preserves both `doc_id` and `attachment_name`:
```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    couchdb_url='http://localhost:5984',
    couchdb_database='mydb',
    couchdb_pattern='*.txt',
    line_level=True,
    coalesce_labels=True
)
predictions = classifier.predict()
# Result: doc_id, attachment_name, coalesced_annotations
```

## Note on Coalesced Output Structure

When `coalesce_labels=True`, the output DataFrame structure changes:

**Without coalescing** (`coalesce_labels=False`):
- One row per line/paragraph
- Columns: `doc_id`, `attachment_name`, `value`, `line_number`, `predicted_label`, `annotated_value`
- Can call `.show()` to see individual predictions

**With coalescing** (`coalesce_labels=True`):
- One row per document (or per attachment for CouchDB)
- Columns: `doc_id`, `attachment_name`, `coalesced_annotations` (array of strings)
- `coalesced_annotations` contains merged annotation blocks
- Individual `predicted_label` and `value` columns are not present

### Notebook Compatibility

For notebooks that want to inspect predictions, you should check if coalescing is enabled:

```python
# Get predictions
predictions = classifier.predict()

# Show sample - handle both cases
if "coalesced_annotations" in predictions.columns:
    # Coalesced format
    print("Showing coalesced predictions:")
    predictions.select("doc_id", "attachment_name").show(5)
else:
    # Regular format
    print("Showing individual predictions:")
    predictions.select("doc_id", "attachment_name", "predicted_label").show(5)
```

## Related Issues Fixed

This fix resolves:
- ✅ `attachment_name` preserved when coalescing CouchDB predictions
- ✅ Each attachment processed separately (not merged with other attachments from same document)
- ✅ Proper grouping for both file-based and CouchDB inputs
- ✅ Correct aggregation for saving back to CouchDB

## Testing

To verify the fix:

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.getOrCreate()

# Test with CouchDB and coalescing
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    couchdb_url='http://localhost:5984',
    couchdb_database='mydb',
    couchdb_pattern='*.txt',
    line_level=True,
    coalesce_labels=True,
    output_format='annotated',
    model_type='logistic'
)

predictions = classifier.predict()

# Verify columns
print("Columns:", predictions.columns)
# Should include: ['doc_id', 'attachment_name', 'coalesced_annotations']

# Verify attachment_name is preserved
predictions.select("doc_id", "attachment_name").distinct().show()

# Save should work now
classifier.save_annotated(predictions)
```
