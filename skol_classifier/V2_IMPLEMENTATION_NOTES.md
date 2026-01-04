# SkolClassifierV2 Implementation Notes

## Overview

This document describes the implementation of the helper classes required for SkolClassifierV2 to function properly.

## Problem

The SkolClassifierV2 class was designed with references to helper classes that didn't exist as separate, modular components. The original SkolClassifier had all functionality embedded in a single large class (1128 lines).

## Solution

Created modular helper classes by extracting and refactoring functionality from the original classifier:

### 1. Feature Extraction (`feature_extraction.py`)

**Class: `FeatureExtractor`**
- Builds PySpark ML pipelines for text feature extraction
- Supports word TF-IDF and optional suffix TF-IDF
- Handles label indexing
- Methods:
  - `build_pipeline()` - Creates the feature extraction pipeline
  - `fit_transform(data)` - Fits and transforms training data
  - `transform(data)` - Transforms new data with fitted pipeline
  - `get_pipeline()` - Returns the fitted pipeline model
  - `get_labels()` - Returns label list
  - `get_features_col()` - Returns name of features column

### 2. Model Training (`model.py`)

**Class: `SkolModel`**
- Trains classification models (Logistic Regression, Random Forest, GBT)
- Handles prediction with label conversion
- Methods:
  - `build_classifier()` - Creates classifier based on model_type
  - `fit(train_data, labels)` - Trains the model
  - `predict(data)` - Makes predictions
  - `predict_with_labels(data)` - Predictions with label strings
  - `get_model()` / `set_model()` - Model accessor/mutator
  - `set_labels(labels)` - Sets label list

### 3. Data Loading (`data_loaders.py`)

**Class: `AnnotatedTextLoader`**
- Loads annotated text from files or CouchDB
- Supports line-level and paragraph-level extraction
- Parses YEDDA annotation format: `[@ text #Label*]`
- Methods:
  - `load_from_files(file_paths, collapse_labels, line_level)`
  - `load_from_couchdb(couchdb_url, database, username, password, pattern, collapse_labels, line_level)`

**Class: `RawTextLoader`**
- Loads raw (unannotated) text from files or CouchDB
- Uses heuristic paragraph extraction for paragraph-level
- Methods:
  - `load_from_files(file_paths, line_level)`
  - `load_from_couchdb(couchdb_url, database, username, password, pattern, line_level)`

### 4. Output Formatting (`output_formatters.py`)

**Class: `YeddaFormatter`**
- Formats predictions in YEDDA annotation format
- Coalesces consecutive same-label predictions (for line-level)
- Methods:
  - `format_predictions(predictions)` - Adds YEDDA formatting
  - `coalesce_consecutive_labels(predictions, line_level)` - Merges consecutive labels

**Class: `FileOutputWriter`**
- Writes predictions to local filesystem
- Supports annotated, labels-only, and probabilities formats
- Methods:
  - `save_annotated(predictions, output_path, coalesce_labels, line_level)`
  - `save_labels(predictions, output_path)`
  - `save_probabilities(predictions, output_path)`

**Class: `CouchDBOutputWriter`**
- Writes predictions back to CouchDB as attachments
- Uses distributed `save_distributed()` for efficiency
- Methods:
  - `save_annotated(predictions, suffix, coalesce_labels, line_level)`

## Integration with CouchDB

All CouchDB operations use the existing `CouchDBConnection` class:
- `load_distributed(spark, pattern)` - Loads attachments using mapPartitions
- `save_distributed(df, suffix)` - Saves attachments using mapPartitions

Key fixes made:
- Changed constructor parameter from `url` to `couchdb_url` for consistency
- Split text content into lines before processing in loaders
- Proper aggregation of predictions before saving to CouchDB

## Key Design Decisions

1. **Separation of Concerns**: Each class has a single, well-defined responsibility
2. **Reusability**: Classes can be used independently or composed
3. **Spark Integration**: All classes work with PySpark DataFrames
4. **Backward Compatibility**: Original SkolClassifier remains unchanged

## Testing

Basic import and instantiation tests pass successfully:
- All modules compile without syntax errors
- All classes can be imported
- SkolClassifierV2 can be instantiated with proper configuration
- Configuration validation works correctly

## Files Created

1. `/data/piggy/src/github.com/piggyatbaqaqi/skol/skol_classifier/feature_extraction.py`
2. `/data/piggy/src/github.com/piggyatbaqaqi/skol/skol_classifier/model.py`
3. `/data/piggy/src/github.com/piggyatbaqaqi/skol/skol_classifier/data_loaders.py`
4. `/data/piggy/src/github.com/piggyatbaqaqi/skol/skol_classifier/output_formatters.py`

## Files Modified

1. `/data/piggy/src/github.com/piggyatbaqaqi/skol/skol_classifier/classifier_v2.py`
   - Updated imports to use new helper classes

## Dependencies

The helper classes depend on:
- PySpark (sql, ml, ml.feature, ml.classification)
- skol_classifier.preprocessing (SuffixTransformer, ParagraphExtractor)
- skol_classifier.couchdb_io (CouchDBConnection)

## Next Steps

1. **Integration Testing**: Test complete workflows (train → predict → save)
2. **Redis Model Persistence**: Implement Redis save/load in classifier_v2.py
3. **Disk Model Persistence**: Implement disk save/load in classifier_v2.py
4. **Error Handling**: Add comprehensive error handling and logging
5. **Documentation**: Add docstring examples for common use cases
6. **Performance Testing**: Benchmark against original SkolClassifier

## Known Limitations

1. String-based input/output not yet implemented (planned for future)
2. Model persistence (Redis/disk) references in classifier_v2.py need implementation
3. No automated tests yet - only manual import/instantiation tests performed
4. Coalesced label output for CouchDB needs more testing

## Benefits of V2 Architecture

1. **Cleaner API**: All configuration in constructor, unified methods
2. **More Testable**: Modular components can be tested independently
3. **More Maintainable**: Smaller, focused classes easier to understand
4. **More Extensible**: Easy to add new input sources, output formats, or models
5. **Better Documentation**: Each class has clear purpose and API

## Migration from V1 to V2

The V1 API (SkolClassifier) remains available for backward compatibility. Projects can gradually migrate:

**V1 Example:**
```python
classifier = SkolClassifier(spark=spark)
annotated_df = classifier.load_annotated_data(['data/*.ann'], line_level=True)
classifier.fit_features(annotated_df, use_suffixes=True)
classifier.train_classifier(annotated_df, model_type='logistic')
classifier.save_to_disk('models/model.pkl')
```

**V2 Example:**
```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/*.ann'],
    model_storage='disk',
    model_path='models/model.pkl',
    extraction_mode='line',
    use_suffixes=True,
    model_type='logistic'
)
classifier.fit()
```

The V2 API is significantly more concise and configuration-driven.
