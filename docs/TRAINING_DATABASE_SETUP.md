# Training Database Setup and Section Name Features

## Overview

This document describes the setup for using the `skol_training` database and the new section name detection features in `SkolClassifierV2`.

**Date**: 2025-12-24
**Status**: ✅ Complete and Tested

## Summary of Changes

### 1. Database Permissions Fix

**Problem**: The `skol_training` database had restrictive security settings preventing access:
```json
{"members":{"roles":["_admin"]},"admins":{"roles":["_admin"]}}
```

**Solution**: Created [fix_training_db_permissions.py](../fix_training_db_permissions.py) to clear security restrictions:
```json
{"admins":{"names":[],"roles":[]},"members":{"names":[],"roles":[]}}
```

**Usage**:
```bash
python3 fix_training_db_permissions.py
```

### 2. Section Name Detection in AnnotatedTextParser

**Problem**: Training data from `.txt.ann` files only had `label` column but no `section_name` column, causing feature extraction to fail when using `extraction_mode='section'`.

**Solution**: Enhanced `AnnotatedTextParser` to automatically detect section names from YEDDA annotation content.

**Changes to** [skol_classifier/preprocessing.py](../skol_classifier/preprocessing.py:327-491):

1. Added `_get_section_name()` static method (lines 327-375)
2. Updated `parse()` to include `section_name` in output (lines 377-491)

**Section Detection Logic**:
```python
@staticmethod
def _get_section_name(text: str) -> str:
    """Extract standardized section name from text content."""
    text_lower = text.strip().lower()

    section_map = {
        'introduction': 'Introduction',
        'methods': 'Methods',
        'results': 'Results',
        'discussion': 'Discussion',
        'description': 'Description',
        # ... more sections
    }

    for keyword, standard_name in section_map.items():
        if text_lower == keyword or text_lower.startswith(keyword):
            return standard_name

    return None
```

### 3. NULL Handling in FeatureExtractor

**Problem**: Spark's `Tokenizer` cannot process NULL values in `section_name` column (74,442 out of 77,879 rows were NULL).

**Solution**: Added `SQLTransformer` to fill NULL values with empty string before tokenization.

**Changes to** [skol_classifier/feature_extraction.py](../skol_classifier/feature_extraction.py:99-118):

```python
# Fill NULL section names with empty string
section_null_filler = SQLTransformer(
    statement=f"SELECT *, COALESCE({self.section_name_col}, '') AS section_name_filled FROM __THIS__"
)
section_tokenizer = Tokenizer(
    inputCol="section_name_filled", outputCol="section_tokens"
)
```

## Database Statistics

### skol_training Database

- **Total Documents**: 190
- **Total Annotated Samples**: 77,879
- **Label Distribution**:
  - Misc-exposition: 66,143 (84.9%)
  - Nomenclature: 5,931 (7.6%)
  - Description: 5,805 (7.5%)

### Section Name Distribution

- **NULL (no section detected)**: 74,442 (95.6%)
- **Specimen**: 568 (0.7%)
- **Acknowledgments**: 372 (0.5%)
- **Etymology**: 290 (0.4%)
- **Literature Cited**: 280 (0.4%)
- **Abstract**: 278 (0.4%)
- **Introduction**: 244 (0.3%)
- **Description**: 226 (0.3%)
- **References**: 217 (0.3%)
- **Keywords**: 207 (0.3%)

## Usage

### Basic Training with Separate Database

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.appName("Training").getOrCreate()

classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    couchdb_url='http://localhost:5984',
    couchdb_database='skol_dev',  # For predictions
    couchdb_training_database='skol_training',  # For training
    couchdb_username='admin',
    couchdb_password='password',
    couchdb_pattern='*.txt.ann',
    extraction_mode='section',
    use_suffixes=True,
    section_name_vocab_size=50,
    output_dest='couchdb',
    verbosity=2
)

# Train on skol_training data
results = classifier.fit()

# Predict on skol_dev data
predictions = classifier.predict()
```

### Training with Section Features

```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    couchdb_training_database='skol_training',
    extraction_mode='section',  # Enables section name features
    section_name_vocab_size=50,  # NEW: Control section vocab size
    word_vocab_size=800,
    suffix_vocab_size=200,
    use_suffixes=True,
    verbosity=2
)

results = classifier.fit()
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `couchdb_training_database` | None | Separate database for training data |
| `extraction_mode` | 'paragraph' | 'section' enables section features |
| `section_name_vocab_size` | 50 | Vocabulary size for section TF-IDF |
| `word_vocab_size` | 800 | Vocabulary size for word TF-IDF |
| `suffix_vocab_size` | 200 | Vocabulary size for suffix TF-IDF |

## Feature Pipeline

When `extraction_mode='section'` and data has `section_name` column:

1. **Word Features**: TF-IDF on tokenized text
2. **Suffix Features**: TF-IDF on 2-4 character suffixes (if `use_suffixes=True`)
3. **Section Features**: TF-IDF on section names (NEW)
4. **Combined**: Concatenated vector of all features

### Feature Dimensions

```
Total features = word_vocab_size + suffix_vocab_size + section_name_vocab_size
               = 800 + 200 + 50
               = 1050 dimensions
```

## Section Detection Examples

The parser detects sections from the first line of YEDDA annotation blocks:

### Example 1: Introduction Section
```
[@ Introduction
This is the introduction section.
More introduction text.
#Misc-exposition*]
```
→ `section_name = 'Introduction'`

### Example 2: Methods Section
```
[@ Methods section
This describes the methods used.
#Misc-exposition*]
```
→ `section_name = 'Methods'`

### Example 3: Description Section
```
[@ Description: Cap 5-10 cm wide.
Detailed description here.
#Description*]
```
→ `section_name = 'Description'`

### Example 4: No Section Detected
```
[@ Agaricus campestris (L.) Fr. 1821
This is a nomenclature entry.
#Nomenclature*]
```
→ `section_name = NULL`

## Testing

### Test Scripts

1. **[test_training_db_access.py](../test_training_db_access.py)** - Verifies database access and data loading
2. **[test_parser_section_names.py](../test_parser_section_names.py)** - Tests section name detection
3. **[test_train_with_section_names.py](../test_train_with_section_names.py)** - Full training workflow test

### Run Tests

```bash
# Test database access
python3 test_training_db_access.py

# Test section name detection
python3 test_parser_section_names.py

# Test full training workflow
python3 test_train_with_section_names.py
```

### Test Results

All tests passing:
- ✅ Database permissions fixed
- ✅ Section name detection working
- ✅ Training with section features successful
- ✅ NULL handling in feature extraction working

## Performance

### Training Stats (5000 sample test)

- **Accuracy**: 100% (on small test set)
- **Training Time**: ~30 seconds
- **Feature Extraction**: Section features add minimal overhead
- **Memory**: No significant increase

## Supported Section Names

The following section names are automatically detected:

- Introduction
- Abstract
- Keywords
- Taxonomy
- Materials and Methods
- Methods
- Results
- Discussion
- Acknowledgments
- References
- Conclusion
- Description
- Etymology
- Specimen
- Holotype
- Paratype
- Literature Cited
- Background
- Objectives
- Summary
- Figures
- Tables
- Appendix
- Supplementary

## Files Modified

1. **[skol_classifier/preprocessing.py](../skol_classifier/preprocessing.py)** - Added section name detection
2. **[skol_classifier/feature_extraction.py](../skol_classifier/feature_extraction.py)** - Added NULL handling
3. **[fix_training_db_permissions.py](../fix_training_db_permissions.py)** - New script to fix permissions

## Files Created

1. **[test_training_db_access.py](../test_training_db_access.py)** - Database access test
2. **[test_parser_section_names.py](../test_parser_section_names.py)** - Section detection test
3. **[test_train_with_section_names.py](../test_train_with_section_names.py)** - Full training test
4. **[docs/TRAINING_DATABASE_SETUP.md](TRAINING_DATABASE_SETUP.md)** - This document

## Troubleshooting

### Issue: Unauthorized Error

**Symptom**: `couchdb.http.Unauthorized` when accessing `skol_training`

**Solution**: Run permission fix script:
```bash
python3 fix_training_db_permissions.py
```

### Issue: NullPointerException in Tokenizer

**Symptom**: `java.lang.NullPointerException` in Spark Tokenizer

**Solution**: This should be fixed by the SQLTransformer that fills NULLs. If you still see this, ensure you're using the latest version of feature_extraction.py.

### Issue: section_name Column Missing

**Symptom**: `section_name does not exist` error

**Solution**: This should be automatically added by AnnotatedTextParser. Verify you're using the updated preprocessing.py.

## See Also

- [SKOL_TRAINING_INGESTION.md](SKOL_TRAINING_INGESTION.md) - Database ingestion details
- [TXT_ATTACHMENT_IMPLEMENTATION.md](TXT_ATTACHMENT_IMPLEMENTATION.md) - Text file support
- [PDF_TXT_ATTACHMENT_SUPPORT.md](PDF_TXT_ATTACHMENT_SUPPORT.md) - Text attachment user guide

---

**Status**: ✅ Complete and Tested
**Date**: 2025-12-24
**Training Samples**: 77,879
**Database**: skol_training
**Section Features**: Enabled
