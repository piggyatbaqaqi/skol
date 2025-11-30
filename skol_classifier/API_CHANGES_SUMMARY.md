# API Changes Summary - Raw Text Input

## Overview

Updated the line-by-line classification API to accept raw text strings instead of file paths. This provides more flexibility for processing text from various sources (databases, APIs, CouchDB, etc.).

## Changed Methods

### 1. `load_raw_data_lines(text_contents: List[str])`

**Before:**
```python
def load_raw_data_lines(self, file_paths: List[str]) -> DataFrame:
    # Read files from disk
    df = self.spark.read.text(file_paths).withColumn("filename", input_file_name())
    ...
```

**After:**
```python
def load_raw_data_lines(self, text_contents: List[str]) -> DataFrame:
    # Process raw text strings directly
    data = []
    for doc_id, text in enumerate(text_contents):
        lines = text.split('\n')
        for line_num, line in enumerate(lines):
            data.append((f"doc_{doc_id}", line, line_num))

    df = self.spark.createDataFrame(data, ["filename", "value", "line_number"])
    ...
```

**Impact:**
- Now accepts raw text strings instead of file paths
- Each string in the list is treated as a separate document
- Documents are labeled as `doc_0`, `doc_1`, etc.

### 2. `predict_lines(text_contents: List[str], output_format: str = "yedda")`

**Before:**
```python
def predict_lines(self, file_paths: List[str], output_format: str = "yedda") -> DataFrame:
    raw_df = self.load_raw_data_lines(file_paths)
    ...
```

**After:**
```python
def predict_lines(self, text_contents: List[str], output_format: str = "yedda") -> DataFrame:
    raw_df = self.load_raw_data_lines(text_contents)
    ...
```

**Impact:**
- Now accepts raw text strings instead of file paths
- Allows classification of text from any source

### 3. `save_to_couchdb(predictions: DataFrame, suffix: str = ".ann", coalesce_labels: bool = False)`

**Before:**
```python
def save_to_couchdb(self, predictions: DataFrame, suffix: str = ".ann") -> List[Dict[str, Any]]:
    # Only supported paragraph-based aggregation
    ...
```

**After:**
```python
def save_to_couchdb(self, predictions: DataFrame, suffix: str = ".ann", coalesce_labels: bool = False) -> List[Dict[str, Any]]:
    if coalesce_labels:
        # For line-level predictions with YEDDA coalescence
        ...
    else:
        # Original paragraph-based aggregation
        ...
```

**Impact:**
- Added `coalesce_labels` parameter for line-level predictions
- When True, consecutive lines with same label are coalesced into YEDDA blocks
- Backward compatible - default behavior unchanged

## Migration Guide

### Old API Usage

```python
# Read files from disk
classifier = SkolClassifier(spark=spark)
predictions = classifier.predict_lines(['path/to/file1.txt', 'path/to/file2.txt'])
classifier.save_yedda_output(predictions, 'output_dir')
```

### New API Usage

```python
# Read text content first
classifier = SkolClassifier(spark=spark)

# Option 1: From files
with open('path/to/file1.txt', 'r') as f:
    text1 = f.read()
with open('path/to/file2.txt', 'r') as f:
    text2 = f.read()

predictions = classifier.predict_lines([text1, text2])
classifier.save_yedda_output(predictions, 'output_dir')

# Option 2: From CouchDB
from couchdb_file import CouchDBFile

couchdb = CouchDBFile('http://localhost:5984', 'database')
text1 = couchdb.get_attachment('doc1', 'article.txt')
text2 = couchdb.get_attachment('doc2', 'article.txt')

predictions = classifier.predict_lines([text1, text2])

# Save back to CouchDB with coalescence
results = classifier.save_to_couchdb(
    predictions,
    suffix='.ann',
    coalesce_labels=True  # NEW: Creates YEDDA blocks
)

# Option 3: From any source (database, API, etc.)
texts = fetch_texts_from_database()  # Returns List[str]
predictions = classifier.predict_lines(texts)
```

## Benefits

1. **Flexibility**: Process text from any source, not just files
2. **CouchDB Integration**: Seamlessly work with CouchDB attachments
3. **Memory Efficient**: No intermediate files needed
4. **Database Support**: Easy integration with SQL, NoSQL databases
5. **API Integration**: Process text from web APIs directly
6. **Testing**: Easier to write unit tests with raw strings

## Backward Compatibility

The changes are **not backward compatible** for code that:
- Calls `load_raw_data_lines()` with file paths
- Calls `predict_lines()` with file paths

**Migration is required** - read files first, then pass content.

The `save_to_couchdb()` change **is backward compatible**:
- Default `coalesce_labels=False` maintains original behavior
- Only affects users who want YEDDA coalescence

## Updated Examples

All examples have been updated:
- [test_line_classifier.py](test_line_classifier.py) - Now tests with raw text
- [example_line_classification.py](example_line_classification.py) - Shows raw text usage
- [YEDDA_INTEGRATION_GUIDE.md](../YEDDA_INTEGRATION_GUIDE.md) - Updated all patterns

## Testing

Run updated tests:
```bash
python skol_classifier/test_line_classifier.py
```

All tests pass with new API ✓

## See Also

- [README.md](README.md) - Updated API documentation
- [LINE_CLASSIFICATION_SUMMARY.md](LINE_CLASSIFICATION_SUMMARY.md) - Line classification overview
- [../YEDDA_INTEGRATION_GUIDE.md](../YEDDA_INTEGRATION_GUIDE.md) - Complete integration guide
