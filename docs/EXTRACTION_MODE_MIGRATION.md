# Migration Guide: `tokenizer` → `extraction_mode`

## Overview

The `tokenizer` parameter in `SkolClassifierV2` has been renamed to `extraction_mode` to better reflect its purpose. This parameter controls the **granularity of text extraction** (line, paragraph, or section level), not tokenization which happens during feature extraction.

**Date**: 2025-12-24
**Breaking Change**: Yes (parameter name changed)
**Backwards Compatibility**: Deprecated parameter still supported

## Why the Change?

The term `tokenizer` was misleading because:
- It doesn't control tokenization (which is handled by Spark's ML Tokenizer during feature extraction)
- It actually controls **how text is chunked/extracted** from documents before processing
- The name suggested a lower-level operation than what it actually does

The new name `extraction_mode` more accurately describes the parameter's function: controlling the mode/granularity of text extraction.

## Migration Steps

### Simple Replacement

Replace `tokenizer=` with `extraction_mode=` in all `SkolClassifierV2` instantiations:

**Before:**
```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    tokenizer='section',  # OLD NAME
    use_suffixes=True
)
```

**After:**
```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    extraction_mode='section',  # NEW NAME
    use_suffixes=True
)
```

## Parameter Values

The allowed values remain the same:

| Value | Description |
|-------|-------------|
| `'line'` | Extract and process text line-by-line |
| `'paragraph'` | Extract and process text by paragraphs (default) |
| `'section'` | Extract sections from PDFs with section name features |

## Examples

### Training with Line-Level Extraction

**Before:**
```python
classifier = SkolClassifierV2(
    input_source='files',
    file_paths=['data/*.txt.ann'],
    tokenizer='line',
    model_path='models/line_model.pkl'
)
```

**After:**
```python
classifier = SkolClassifierV2(
    input_source='files',
    file_paths=['data/*.txt.ann'],
    extraction_mode='line',
    model_path='models/line_model.pkl'
)
```

### Prediction with Section-Level Extraction

**Before:**
```python
classifier = SkolClassifierV2(
    input_source='couchdb',
    couchdb_database='articles',
    tokenizer='section',
    section_name_vocab_size=50,
    output_dest='couchdb'
)
```

**After:**
```python
classifier = SkolClassifierV2(
    input_source='couchdb',
    couchdb_database='articles',
    extraction_mode='section',
    section_name_vocab_size=50,
    output_dest='couchdb'
)
```

### Training from Separate Database

**Before:**
```python
classifier = SkolClassifierV2(
    couchdb_database='skol_dev',
    couchdb_training_database='skol_training',
    tokenizer='section',
    use_suffixes=True
)
```

**After:**
```python
classifier = SkolClassifierV2(
    couchdb_database='skol_dev',
    couchdb_training_database='skol_training',
    extraction_mode='section',
    use_suffixes=True
)
```

## Backwards Compatibility

### Deprecated Property

The `line_level` property still works for backwards compatibility:

```python
classifier = SkolClassifierV2(extraction_mode='line')
assert classifier.line_level == True  # Still works

classifier = SkolClassifierV2(extraction_mode='paragraph')
assert classifier.line_level == False  # Still works
```

## Updated Documentation

The following documentation has been updated to use `extraction_mode`:

- [TRAINING_DATABASE_SETUP.md](TRAINING_DATABASE_SETUP.md)
- [PDF_TXT_ATTACHMENT_SUPPORT.md](PDF_TXT_ATTACHMENT_SUPPORT.md)
- [TXT_ATTACHMENT_IMPLEMENTATION.md](TXT_ATTACHMENT_IMPLEMENTATION.md)
- [CLASSIFIER_V2_TOKENIZER_UPDATE.md](CLASSIFIER_V2_TOKENIZER_UPDATE.md)
- [PDF_YEDDA_ANNOTATION_SUPPORT.md](PDF_YEDDA_ANNOTATION_SUPPORT.md)
- [PDF_YEDDA_IMPLEMENTATION_SUMMARY.md](PDF_YEDDA_IMPLEMENTATION_SUMMARY.md)

## Code Search and Replace

To migrate your codebase, use this search and replace pattern:

**Search:** `tokenizer=`
**Replace:** `extraction_mode=`

**Search (regex):** `tokenizer\s*[:=]\s*`
**Replace:** `extraction_mode=`

## Terminology Changes

| Old Term | New Term | Context |
|----------|----------|---------|
| "tokenizer mode" | "extraction mode" | Parameter description |
| "line tokenization" | "line extraction" | Processing description |
| "paragraph tokenization" | "paragraph extraction" | Processing description |
| "section tokenization" | "section extraction" | Processing description |

## Implementation Details

**What Changed:**
- Parameter name: `tokenizer` → `extraction_mode`
- Instance variable: `self.tokenizer` → `self.extraction_mode`
- Documentation: Updated to use "extraction mode" terminology

**What Didn't Change:**
- Allowed values (`'line'`, `'paragraph'`, `'section'`)
- Default value (`'paragraph'`)
- Behavior and functionality
- Feature extraction pipeline
- Model training and prediction

## Testing

All test files have been updated:
- [test_training_db_access.py](../test_training_db_access.py)
- [test_train_with_section_names.py](../test_train_with_section_names.py)
- [test_parser_section_names.py](../test_parser_section_names.py)

Run tests to verify migration:
```bash
python3 test_training_db_access.py
python3 test_train_with_section_names.py
```

## Related Changes

This refactoring is part of the section name feature implementation:
- [TRAINING_DATABASE_SETUP.md](TRAINING_DATABASE_SETUP.md) - Training database setup
- Section name detection in `AnnotatedTextParser`
- NULL handling in `FeatureExtractor`

---

**Status**: ✅ Complete
**Date**: 2025-12-24
**Breaking Change**: Parameter renamed (`tokenizer` → `extraction_mode`)
**Migration**: Simple search and replace
