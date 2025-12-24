# Unified Extraction Architecture

## Overview

All three extraction modes (`'line'`, `'paragraph'`, `'section'`) now use a unified architecture with consistent data flow through `AnnotatedTextParser`. This eliminates code duplication and ensures consistent YEDDA label extraction across all modes.

## Architecture

### Critical Distinction: Training vs. Prediction

**Training Data** (from `couchdb_training_database`):
- ALWAYS reads from `.txt.ann` files (YEDDA annotated text)
- For ALL three extraction modes (line, paragraph, section)
- PDFs are NEVER used for training

**Prediction Data** (from `couchdb_database`):
- Reads from PDF files (unannotated documents)
- Only for section mode (line/paragraph modes use text files)
- Uses PDFSectionExtractor to extract structure

### Before Refactoring

**Line/Paragraph modes:**
```
CouchDBConnection → AnnotatedTextParser → DataFrame(value, label, section_name, ...)
```

**Section mode:**
```
PDFSectionExtractor (includes YEDDA parsing) → DataFrame(value, label, section_name, ...)
```

**Problem**: Duplicated YEDDA parsing logic in two places.

### After Refactoring

**Training (ALL modes):**
```
CouchDBConnection.load_distributed("*.txt.ann")
  ↓ (raw YEDDA annotated text from training database)
AnnotatedTextParser(extraction_mode='line'|'paragraph'|'section')
  ↓ (extracts YEDDA labels + splits text + detects sections)
DataFrame(doc_id, value, label, section_name)
```

**Prediction:**

**Line/Paragraph modes:**
```
CouchDBConnection.load_distributed("*.txt")
  ↓ (raw unannotated text)
DataFrame(doc_id, value)
```

**Section mode:**
```
PDFSectionExtractor.extract_from_multiple_documents()
  ↓ (sections with structure from PDFs)
DataFrame(doc_id, value, section_name, line_number, page_number, ...)
```

## Key Components

### 1. PDFSectionExtractor

**Responsibility**: Structural extraction only
- Extract sections/paragraphs from PDFs
- Detect section headers
- Track line numbers, page numbers
- Preserve YEDDA annotations IN THE TEXT (doesn't extract them)

**Output Schema:**
```python
StructType([
    StructField("value", StringType()),              # Text with YEDDA preserved
    StructField("doc_id", StringType()),
    StructField("attachment_name", StringType()),
    StructField("paragraph_number", IntegerType()),
    StructField("line_number", IntegerType()),       # First line of section
    StructField("page_number", IntegerType()),
    StructField("empirical_page_number", IntegerType(), True),
    StructField("section_name", StringType(), True)
])
```

### 2. AnnotatedTextParser

**Responsibility**: YEDDA label extraction for ALL modes
- Parses `[@content#Label*]` annotations
- Supports three extraction modes:
  - `'line'`: Splits into individual lines
  - `'paragraph'`: Treats each annotation block as a paragraph
  - `'section'`: Adaptive behavior (see below)

**Section Mode - Adaptive Behavior:**

Section mode handles both training and prediction data:

1. **Training** (plain `.txt.ann` files):
   - Input: `DataFrame(doc_id, value=annotated_text)`
   - Behavior: Parse like paragraphs + detect section names
   - Output: `DataFrame(doc_id, value, label, section_name)`

2. **Prediction** (pre-segmented from PDFSectionExtractor):
   - Input: `DataFrame(doc_id, value, line_number, page_number, section_name, ...)`
   - Behavior: Extract labels only, preserve ALL columns
   - Output: Same columns + `label`

**Detection Logic:**
```python
def parse(self, df: DataFrame) -> DataFrame:
    if self.extraction_mode == 'section':
        if 'line_number' in df.columns:
            # Pre-segmented (prediction): just extract labels
            return df.withColumn("label", extract_label_udf(col("value")))
        else:
            # Plain text (training): parse paragraphs + detect sections
            # Behavior similar to paragraph mode
            ...
```

### 3. Classifier Integration

**Training Data Loading** ([classifier_v2.py:888-928](../skol_classifier/classifier_v2.py#L888-L928)):

```python
def _load_annotated_from_couchdb(self) -> DataFrame:
    """Load training data from .txt.ann files (ALL modes)."""
    database = self.couchdb_training_database or self.couchdb_database

    # For ALL extraction modes, load from .txt.ann files
    # PDFs are only used for prediction, not training
    conn = CouchDBConnection(self.couchdb_url, database, ...)
    pattern = "*.txt.ann"  # YEDDA annotated text
    df = conn.load_distributed(self.spark, pattern)

    # Parse annotations using unified AnnotatedTextParser
    parser = AnnotatedTextParser(
        extraction_mode=self.extraction_mode,
        collapse_labels=self.collapse_labels
    )
    return parser.parse(df)
```

**Prediction Data Loading** ([classifier_v2.py:788-820](../skol_classifier/classifier_v2.py#L788-L820)):

```python
def _load_raw_from_couchdb(self) -> DataFrame:
    """Load prediction data from CouchDB."""
    if self.extraction_mode == 'section':
        # Load from PDFs using PDFSectionExtractor
        return self._load_sections_from_couchdb()
    else:
        # Load from text files for line/paragraph modes
        conn = CouchDBConnection(...)
        return conn.load_distributed(self.spark, pattern="*.txt")
```

## Benefits

### 1. Single Source of Truth
- YEDDA parsing logic exists in ONE place (`AnnotatedTextParser`)
- Consistent label extraction across all modes
- Easier to maintain and debug

### 2. Separation of Concerns
- **PDFSectionExtractor**: Structural extraction (PDF-specific)
- **AnnotatedTextParser**: Label extraction (format-specific)
- **Classifier**: Orchestration and model training

### 3. Metadata Preservation
- Section mode preserves ALL metadata (line_number, page_number, section_name, etc.)
- AnnotatedTextParser in section mode adds labels WITHOUT dropping columns
- Consistent schema across pipeline

### 4. Backwards Compatibility
- `AnnotatedTextParser` still accepts `line_level` parameter (deprecated)
- Automatic conversion: `line_level=True` → `extraction_mode='line'`
- Existing code continues to work

## Data Flow Examples

### Training: Section Mode from .txt.ann Files

1. **Load Annotated Text**:
```python
# CouchDBConnection loads .txt.ann files
conn = CouchDBConnection(...)
df = conn.load_distributed(spark, "*.txt.ann")
# Columns: doc_id, attachment_name, value
# value = "[@Introduction\nThis paper describes...#Misc-exposition*]\n[@Methods\n...#Methods*]"
```

2. **Parse Annotations**:
```python
# AnnotatedTextParser extracts labels and detects sections
parser = AnnotatedTextParser(extraction_mode='section')
annotated_df = parser.parse(df)
# Columns: doc_id, attachment_name, value, label, section_name
# value = "Introduction\nThis paper describes..."
# label = "Misc-exposition"
# section_name = "Introduction"
```

3. **Train Model**:
```python
# Model learns from (value, label, section_name)
clf.fit()
```

### Prediction: Section Mode from PDF Files

1. **Extract Structure**:
```python
# PDFSectionExtractor extracts sections from PDFs
extractor = PDFSectionExtractor(...)
sections_df = extractor.extract_from_multiple_documents(...)
# Columns: value, doc_id, line_number, page_number, section_name, ...
# value = "Introduction\nThis paper describes..." (no YEDDA annotations)
```

2. **Predict Labels**:
```python
# Model predicts labels for each section
predictions_df = clf.predict()
# Columns: value, doc_id, line_number, page_number, section_name, predicted_label
```

3. **Output Sorting**:
```python
# CouchDBOutputWriter sorts by line_number when present
predictions.groupBy(...).agg(
    sort_array(collect_list(struct(line_number, annotated_value)))
)
# Results saved back to CouchDB in original document order
```

## Migration Guide

### For Users

No changes required! The refactoring is transparent to users.

Existing code like this continues to work:
```python
clf = SkolClassifierV2(
    spark=spark,
    extraction_mode='section',  # or 'line' or 'paragraph'
    input_source='couchdb',
    ...
)
clf.fit()
```

### For Developers

If you're modifying extraction logic:

**DO**: Modify `AnnotatedTextParser` for YEDDA parsing changes
**DON'T**: Add YEDDA parsing to `PDFSectionExtractor`

**DO**: Modify `PDFSectionExtractor` for structural changes (sections, pages)
**DON'T**: Add text splitting logic to `AnnotatedTextParser.parse()` for section mode

## Testing

Test coverage ensures:
- ✅ Line numbers preserved through entire pipeline
- ✅ Labels correctly extracted for all modes
- ✅ Section names maintained
- ✅ All metadata columns preserved
- ✅ Line numbers correctly ordered

Run tests:
```bash
python3 test_unified_extraction.py
```

## Related Documentation

- [Line Number Tracking](./LINE_NUMBER_TRACKING.md) - Line number preservation details
- [Extraction Mode Migration](./EXTRACTION_MODE_MIGRATION.md) - Parameter rename guide
- [PDF Section Extractor](./PDF_DATAFRAME_MIGRATION_SUMMARY.md) - PDF processing details
