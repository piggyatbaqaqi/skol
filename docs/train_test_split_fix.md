# Train/Test Split Fix - Document-Level Splitting

**Date**: 2025-12-15
**File**: `skol_classifier/classifier_v2.py`
**Lines**: 365-406

## Problem Identified

The original `SkolClassifierV2.fit()` method was using `randomSplit()` to split training data into train and test sets:

```python
train_data, test_data = featured_df.randomSplit([0.8, 0.2], seed=42)
```

### Issues with This Approach

1. **Document Fragmentation**: Lines from the same document could be split between train and test sets
2. **Data Leakage**: Model could see parts of the same document in both training and evaluation
3. **Loss of Ordering**: No guarantee that lines within each dataset maintain proper ordering by line_number
4. **Unrealistic Evaluation**: In production, the model processes complete documents, not random lines

### Example of the Problem

```
Document A (10 lines):
  Lines 1-8 → Train set
  Lines 9-10 → Test set  ← WRONG! Same document split

Document B (5 lines):
  Lines 1-4 → Train set
  Line 5 → Test set      ← WRONG! Same document split
```

This violates the constraint: **"A single document (filename) should not be split between train_data and test_data."**

## Solution Implemented

### New Approach: Document-Level Splitting

```python
# Determine document column (filename or doc_id)
doc_col = "filename" if "filename" in featured_df.columns else "doc_id"

# Get unique documents and split them randomly
unique_docs = featured_df.select(doc_col).distinct()
unique_docs_with_rand = unique_docs.withColumn("rand", rand(seed=42))

# Split documents into train and test (80/20)
train_docs = unique_docs_with_rand.filter("rand < 0.8").select(doc_col)
test_docs = unique_docs_with_rand.filter("rand >= 0.8").select(doc_col)

# Filter featured_df based on document assignments
train_data = featured_df.join(train_docs, on=doc_col, how="inner")
test_data = featured_df.join(test_docs, on=doc_col, how="inner")

# Sort by doc_col and line_number to maintain ordering
if "line_number" in featured_df.columns:
    train_data = train_data.orderBy(doc_col, "line_number")
    test_data = test_data.orderBy(doc_col, "line_number")
```

### How It Works

1. **Extract unique documents**: Get list of all unique doc_ids/filenames
2. **Random assignment**: Assign each document randomly to train (80%) or test (20%)
3. **Filter by document**: Join back to original data to get all lines for assigned documents
4. **Sort within datasets**: Order by (doc_id, line_number) to maintain proper line ordering

### Example with Fix

```
Document A (10 lines):
  Lines 1-10 → Train set  ✓ All lines together

Document B (5 lines):
  Lines 1-5 → Test set    ✓ All lines together

Document C (8 lines):
  Lines 1-8 → Train set   ✓ All lines together
```

## Benefits

### 1. Document Integrity ✅
- All lines from a document stay together
- No partial documents in either set

### 2. Prevent Data Leakage ✅
- Model never sees parts of same document in both train and test
- More realistic evaluation of generalization

### 3. Maintain Ordering ✅
- Lines within each dataset are properly ordered by (doc_id, line_number)
- RNN can process sequences correctly

### 4. Reproducibility ✅
- Still uses seed=42 for reproducible splits
- Same documents assigned to train/test across runs

### 5. Proper Evaluation ✅
- Test set represents how model performs on completely new documents
- Evaluation metrics reflect true generalization capability

## Impact on Model Training

### Before Fix (Potential Issues)
- Test accuracy might be **artificially inflated** due to data leakage
- Model might "memorize" patterns from partial documents
- Evaluation doesn't reflect real-world performance

### After Fix (Correct Behavior)
- Test accuracy reflects **true generalization** to unseen documents
- Model cannot cheat by seeing parts of test documents during training
- Evaluation matches production scenario

## Performance Considerations

The new approach adds minimal overhead:
- One additional `distinct()` operation on document IDs
- Two `join()` operations (efficient with small unique doc list)
- One `orderBy()` operation (important for maintaining order)

The benefits far outweigh the small computational cost.

## Verification

The fix ensures:
1. ✅ No document is split between train and test
2. ✅ Lines maintain ordering by (doc_id, line_number)
3. ✅ Split is reproducible with seed
4. ✅ Approximately 80/20 split ratio maintained

## Related Documentation

- Full ordering analysis: [docs/rnn_ordering_analysis.md](rnn_ordering_analysis.md)
- Train test split code: [skol_classifier/classifier_v2.py:365-406](../skol_classifier/classifier_v2.py#L365-L406)

## Testing Recommendations

To verify the fix works correctly:

```python
# After training, check that no document appears in both sets
train_docs = train_data.select("doc_id").distinct().collect()
test_docs = test_data.select("doc_id").distinct().collect()

train_doc_set = set(row.doc_id for row in train_docs)
test_doc_set = set(row.doc_id for row in test_docs)

# Should be empty
overlap = train_doc_set & test_doc_set
assert len(overlap) == 0, f"Documents in both sets: {overlap}"

# Verify ordering within each dataset
for dataset in [train_data, test_data]:
    window = Window.partitionBy("doc_id").orderBy("line_number")
    ordered_check = dataset.withColumn(
        "expected_row_num",
        row_number().over(window)
    )
    # All rows should have sequential line numbers within doc_id
```

## Conclusion

This fix addresses a critical bug in the train/test splitting logic that could have led to:
- Overestimated model performance
- Data leakage between train and test sets
- Incorrect line ordering

The new document-level splitting approach ensures:
- Proper data separation
- Maintained ordering constraints
- Realistic model evaluation

**Status**: ✅ **FIXED** and ready for testing
