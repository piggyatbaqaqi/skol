# Unified Extraction Architecture

## Overview

All three extraction modes (`'line'`, `'paragraph'`, `'section'`) now use a unified architecture with consistent data flow through `AnnotatedTextParser`. This eliminates code duplication and ensures consistent YEDDA label extraction across all modes.

## Architecture

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

**All modes now use:**
```
Raw Data Source → Structural Extractor → AnnotatedTextParser → DataFrame(value, label, ...)
```

**Specific flows:**

**Line/Paragraph modes:**
```
CouchDBConnection.load_distributed()
  ↓ (raw annotated text)
AnnotatedTextParser(extraction_mode='line'|'paragraph')
  ↓ (extracts YEDDA labels + splits text)
DataFrame(doc_id, value, label, section_name, [line_number])
```

**Section mode:**
```
PDFSectionExtractor.extract_from_multiple_documents()
  ↓ (sections with structure, YEDDA preserved in text)
AnnotatedTextParser(extraction_mode='section')
  ↓ (extracts YEDDA labels, preserves all metadata)
DataFrame(doc_id, value, label, section_name, line_number, page_number, ...)
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
  - `'section'`: Treats input as pre-segmented (preserves structure)

**Key Method:**
```python
def parse(self, df: DataFrame) -> DataFrame:
    if self.extraction_mode == 'section':
        # Input already segmented, just extract labels
        # Preserves ALL existing columns
        return df.withColumn("label", extract_label_udf(col("value")))

    elif self.extraction_mode == 'line':
        # Extract labels AND split into lines
        ...

    elif self.extraction_mode == 'paragraph':
        # Extract labels AND split into paragraphs
        ...
```

### 3. Classifier Integration

**Unified Loading Path** ([classifier_v2.py:888-939](../skol_classifier/classifier_v2.py#L888-L939)):

```python
def _load_annotated_from_couchdb(self) -> DataFrame:
    database = self.couchdb_training_database or self.couchdb_database

    if self.extraction_mode == 'section':
        # Get structure from PDFSectionExtractor
        sections_df = self._load_sections_from_couchdb(database=database)

        # Extract labels through AnnotatedTextParser
        parser = AnnotatedTextParser(
            extraction_mode='section',
            collapse_labels=self.collapse_labels
        )
        return parser.parse(sections_df)

    else:  # 'line' or 'paragraph'
        # Load raw text
        conn = CouchDBConnection(...)
        df = conn.load_distributed(self.spark, pattern)

        # Extract labels and split text through AnnotatedTextParser
        parser = AnnotatedTextParser(
            extraction_mode=self.extraction_mode,
            collapse_labels=self.collapse_labels
        )
        return parser.parse(df)
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

## Data Flow Example

### Section Mode Training

1. **Load Structure**:
```python
# PDFSectionExtractor extracts sections
sections_df = extract_from_multiple_documents(...)
# Columns: value, doc_id, attachment_name, line_number, page_number, section_name
# value = "[@Introduction\nThis paper describes...#Misc-exposition*]"
```

2. **Extract Labels**:
```python
# AnnotatedTextParser extracts labels
parser = AnnotatedTextParser(extraction_mode='section')
annotated_df = parser.parse(sections_df)
# Columns: value, doc_id, attachment_name, line_number, page_number, section_name, label
# label = "Misc-exposition"
```

3. **Feature Extraction**:
```python
# Features are extracted from text
featured_df = feature_extractor.fit_transform(annotated_df)
# line_number preserved through transformations
```

4. **Output Sorting**:
```python
# CouchDBOutputWriter sorts by line_number when present
predictions.groupBy(...).agg(
    sort_array(collect_list(struct(line_number, annotated_value)))
)
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
