# Test Files Updated to V2 API

## Overview

Updated all test and example files to use `SkolClassifierV2` instead of the deprecated `SkolClassifier`.

## Files Modified

### 1. [skol_classifier/example_line_classification.py](skol_classifier/example_line_classification.py)

**Purpose**: Demonstrates line-by-line classification with YEDDA output

**Changes**:
- Changed import from `SkolClassifier` to `SkolClassifierV2`
- Completely rewrote example to use V2 unified API
- Removed references to deprecated methods
- Added demonstration of V2 configuration options

**Before**:
```python
from skol_classifier.classifier import SkolClassifier

classifier = SkolClassifier(spark=spark)
# Uses deprecated methods: load_annotated_data(), fit_features(), train_classifier()
```

**After**:
```python
from skol_classifier.classifier_v2 import SkolClassifierV2

classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=[str(f) for f in annotated_files],
    line_level=True,
    use_suffixes=False,
    model_type='logistic',
    output_format='annotated',
    coalesce_labels=True
)

results = classifier.fit()  # Single method call
predictions = classifier.predict()
```

**Key Features Demonstrated**:
- Unified API with single `fit()` call
- Line-level processing
- Label coalescing
- YEDDA format output
- Sample predictions and label distribution

### 2. [skol_classifier/test_line_classifier.py](skol_classifier/test_line_classifier.py)

**Purpose**: Tests line-by-line classification functionality

**Changes**:
- Changed import to use `SkolClassifierV2`
- Rewrote tests to use V2 API
- Added comprehensive test coverage for V2 features
- Tests both with and without training data

**Test Cases**:
1. **Initialization test**: Verify V2 API configuration
2. **Training test**: Train with line-level processing
3. **Prediction test**: Make and verify predictions
4. **YEDDA format test**: Verify output format
5. **Label distribution test**: Check predicted labels
6. **No-coalesce test**: Verify coalescing can be disabled

**Before**:
```python
from skol_classifier.classifier import SkolClassifier

classifier = SkolClassifier(spark=spark, auto_load=False)
result = classifier.coalesce_consecutive_labels(test_data)
df = classifier.load_raw_data_lines(raw_texts)
```

**After**:
```python
from skol_classifier.classifier_v2 import SkolClassifierV2

classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=[str(f) for f in annotated_files[:2]],
    line_level=True,
    use_suffixes=False,
    model_type='logistic',
    output_format='annotated'
)

results = classifier.fit()
predictions = classifier.predict()

# Verify structure
assert 'value' in predictions.columns
assert 'predicted_label' in predictions.columns
assert 'annotated_value' in predictions.columns
assert 'line_number' in predictions.columns
```

### 3. [skol_classifier/test_line_level_loading.py](skol_classifier/test_line_level_loading.py)

**Purpose**: Compares line-level vs paragraph-level data loading

**Changes**:
- Updated to use `SkolClassifierV2`
- Tests both processing modes with V2 API
- Verifies proper column presence and data structure

**Test Scenarios**:
1. **Paragraph-level loading**: `line_level=False`
2. **Line-level loading**: `line_level=True`
3. **Verification**: Checks expected counts and columns
4. **V2 features**: Verifies configuration options

**Before**:
```python
from skol_classifier.classifier import SkolClassifier

classifier = SkolClassifier(spark=spark, auto_load=False)
para_df = classifier.load_annotated_data([temp_file], line_level=False)
line_df = classifier.load_annotated_data([temp_file], line_level=True)
```

**After**:
```python
from skol_classifier.classifier_v2 import SkolClassifierV2

classifier_para = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=[temp_file],
    line_level=False,
    model_type='logistic',
    auto_load_model=False
)
para_df = classifier_para._load_annotated_data()

classifier_line = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=[temp_file],
    line_level=True,
    model_type='logistic',
    auto_load_model=False
)
line_df = classifier_line._load_annotated_data()
```

## Key Differences: V1 vs V2

### V1 API (Deprecated)
```python
# Multi-step process
classifier = SkolClassifier(spark=spark)
classifier.load_annotated_data(file_paths)
classifier.fit_features()
classifier.train_classifier()
predictions = classifier.predict_lines(raw_data)
classifier.save_yeda_output(predictions, output_dir)
```

### V2 API (Current)
```python
# Single-step configuration
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/*.ann'],
    line_level=True,
    coalesce_labels=True,
    output_format='annotated',
    model_type='logistic'
)

# Simple workflow
results = classifier.fit()
predictions = classifier.predict()
classifier.save_annotated(predictions)
```

## Benefits of V2 API

1. **Unified Configuration**: All settings in constructor
2. **Simpler Workflow**: Single `fit()` call instead of multiple steps
3. **Consistent Interface**: Same API for files and CouchDB
4. **Better Separation**: Prediction vs formatting are separate
5. **Flexible Output**: Multiple format options
6. **Proper Coalescing**: Applied at save time, not prediction time

## Running the Tests

### Example Script
```bash
python skol_classifier/example_line_classification.py
```

Expected output:
- Demonstrates V2 API usage
- Shows training and prediction
- Displays sample results and statistics

### Line Classifier Test
```bash
python skol_classifier/test_line_classifier.py
```

Expected output:
- Runs 5 test cases
- Verifies V2 functionality
- Shows "All tests passed!" if successful

### Line Level Loading Test
```bash
python skol_classifier/test_line_level_loading.py
```

Expected output:
- Compares paragraph vs line loading
- Verifies expected counts (3 paragraphs, 8 lines)
- Returns exit code 0 on success

## Migration Guide for Other Code

If you have code using `SkolClassifier`, migrate to `SkolClassifierV2`:

**Old Code**:
```python
from skol_classifier.classifier import SkolClassifier

classifier = SkolClassifier(spark=spark)
classifier.load_annotated_data(file_paths)
classifier.fit_features()
classifier.train_classifier()
predictions = classifier.predict_paragraph(data)
```

**New Code**:
```python
from skol_classifier.classifier_v2 import SkolClassifierV2

classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=file_paths,
    line_level=False,  # or True for line-level
    model_type='logistic'
)

results = classifier.fit()
predictions = classifier.predict()
```

## Related Documentation

- [CLASSIFIER_V2_API.md](CLASSIFIER_V2_API.md) - Complete V2 API documentation
- [COALESCE_DURING_SAVE_FIX.md](COALESCE_DURING_SAVE_FIX.md) - Coalescing behavior
- [PRESERVE_METADATA_FIX.md](PRESERVE_METADATA_FIX.md) - Metadata preservation
- [ANNOTATED_VALUE_RENAME.md](ANNOTATED_VALUE_RENAME.md) - Column naming

## Notes

- All test files now use only V2 API
- Tests are compatible with both training data present and absent
- Tests verify core V2 functionality
- Original `SkolClassifier` is still available but deprecated
