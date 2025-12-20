# predict_proba() Implementation Summary

## What Was Implemented

Added `predict_proba()` method to `TraditionalMLSkolModel` to enable probability distribution access for all traditional Spark ML models (Logistic Regression, Random Forest, Gradient Boosted Trees). This allows the unified `calculate_stats()` method in the base class to compute loss metrics for all model types.

## Changes Made

### 1. **[skol_classifier/model.py](../skol_classifier/model.py:38)** - Added Result Caching

**Added instance variable**:
```python
self._last_predictions: Optional[DataFrame] = None  # Cache for stats calculation
```

This cache stores the most recent predictions with probabilities, allowing `calculate_stats()` to access probability distributions even when called separately.

### 2. **[skol_classifier/model.py](../skol_classifier/model.py:114-156)** - Implemented `predict_proba()`

**New method**:
```python
def predict_proba(self, data: DataFrame) -> DataFrame:
    """
    Make predictions with probabilities using Spark ML pipeline.

    The results are cached in self._last_predictions for use by calculate_stats().

    Args:
        data: DataFrame with features

    Returns:
        DataFrame with predictions and probabilities column

    Raises:
        ValueError: If model hasn't been trained yet
    """
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import ArrayType, DoubleType

    if self.classifier_model is None:
        raise ValueError("No classifier model found. Train a model first.")

    # Transform returns prediction and probability columns
    predictions = self.classifier_model.transform(data)

    # Convert Spark ML's 'probability' Vector column to 'probabilities' array
    # for compatibility with calculate_stats()
    if "probability" in predictions.columns:
        @udf(returnType=ArrayType(DoubleType()))
        def vector_to_array(v):
            """Convert Spark ML Vector to list of floats."""
            if v is None:
                return []
            return [float(x) for x in v.toArray()]

        predictions = predictions.withColumn(
            "probabilities",
            vector_to_array(col("probability"))
        )

    # Cache for stats calculation
    self._last_predictions = predictions

    return predictions
```

**Key features**:
- Converts Spark ML's Vector format to array for compatibility
- Caches results for stats calculation
- Works with all Spark ML classifiers (LR, RF, GBT)

### 3. **[skol_classifier/model.py](../skol_classifier/model.py:158-174)** - Updated `predict()`

**Modified to use predict_proba**:
```python
def predict(self, data: DataFrame) -> DataFrame:
    """
    Make predictions using Spark ML pipeline.

    Uses predict_proba() internally and caches results for calculate_stats().

    Args:
        data: DataFrame with features

    Returns:
        DataFrame with predictions (includes probabilities column)

    Raises:
        ValueError: If model hasn't been trained yet
    """
    # Use predict_proba which caches the full results
    return self.predict_proba(data)
```

Now `predict()` delegates to `predict_proba()`, ensuring probabilities are always available.

## Benefits

### 1. **Unified Loss Metrics**

All model types now support loss calculation:
- **Before**: Only RNN models had loss metrics
- **After**: Logistic Regression, Random Forest, and GBT models also compute loss

### 2. **Consistent Interface**

All models now have the same prediction interface:
```python
# All models support this
predictions = model.predict_proba(data)
stats = model.calculate_stats(predictions)
# stats['loss'] is available for all models
```

### 3. **Better Model Comparison**

You can now compare models using loss metrics:
```python
# Train different models
lr_model = LogisticRegressionSkolModel(...)
rf_model = RandomForestSkolModel(...)

# Compare using same metrics
lr_stats = lr_model.calculate_stats(lr_predictions)
rf_stats = rf_model.calculate_stats(rf_predictions)

print(f"LR Loss:  {lr_stats['loss']:.4f}")
print(f"RF Loss:  {rf_stats['loss']:.4f}")
```

### 4. **Automatic Caching**

Results are automatically cached, so you don't need to keep the DataFrame around:
```python
# Predict once
model.predict(test_data)

# Stats can be calculated later using cached results
stats = model.calculate_stats(model._last_predictions)
```

## Testing

Test suite (`examples/test_predict_proba.py`) verifies:

1. ✓ Logistic Regression predict_proba() works correctly
2. ✓ Random Forest predict_proba() works correctly
3. ✓ Probabilities are in array format (not Spark ML Vector)
4. ✓ Probabilities sum to 1.0
5. ✓ calculate_stats() can compute loss metrics
6. ✓ Confusion matrix displayed at verbosity >= 2
7. ✓ predict() uses predict_proba() internally
8. ✓ Results cached in _last_predictions

All tests pass:
```bash
python examples/test_predict_proba.py
# ALL TESTS PASSED ✓
```

## Example: Comparing Models with Loss Metrics

```python
from pyspark.sql import SparkSession
from skol_classifier.model import (
    LogisticRegressionSkolModel,
    RandomForestSkolModel,
    GradientBoostedSkolModel
)

spark = SparkSession.builder.appName("Model Comparison").getOrCreate()

# Load your training and test data
train_df = ...  # Your training DataFrame
test_df = ...   # Your test DataFrame
labels = ["Class0", "Class1", "Class2"]

# Train multiple models
lr_model = LogisticRegressionSkolModel(verbosity=2)
lr_model.fit(train_df, labels=labels)

rf_model = RandomForestSkolModel(n_estimators=100, verbosity=2)
rf_model.fit(train_df, labels=labels)

gbt_model = GradientBoostedSkolModel(max_iter=50, verbosity=2)
gbt_model.fit(train_df, labels=labels)

# Make predictions with probabilities
lr_predictions = lr_model.predict_proba(test_df)
rf_predictions = rf_model.predict_proba(test_df)
gbt_predictions = gbt_model.predict_proba(test_df)

# Calculate stats (includes loss metrics and confusion matrix)
print("\n=== LOGISTIC REGRESSION ===")
lr_stats = lr_model.calculate_stats(lr_predictions)

print("\n=== RANDOM FOREST ===")
rf_stats = rf_model.calculate_stats(rf_predictions)

print("\n=== GRADIENT BOOSTED TREES ===")
gbt_stats = gbt_model.calculate_stats(gbt_predictions)

# Compare models
print("\n=== MODEL COMPARISON ===")
print(f"{'Model':<25} {'Accuracy':<10} {'F1':<10} {'Loss':<10}")
print("-" * 55)
print(f"{'Logistic Regression':<25} {lr_stats['accuracy']:<10.4f} {lr_stats['f1_score']:<10.4f} {lr_stats['loss']:<10.4f}")
print(f"{'Random Forest':<25} {rf_stats['accuracy']:<10.4f} {rf_stats['f1_score']:<10.4f} {rf_stats['loss']:<10.4f}")
print(f"{'Gradient Boosted Trees':<25} {gbt_stats['accuracy']:<10.4f} {gbt_stats['f1_score']:<10.4f} {gbt_stats['loss']:<10.4f}")
```

**Sample output**:
```
=== MODEL COMPARISON ===
Model                     Accuracy   F1         Loss
-------------------------------------------------------
Logistic Regression       0.8500     0.8450     0.4230
Random Forest             0.8800     0.8750     0.3210
Gradient Boosted Trees    0.9100     0.9050     0.2850
```

## Implementation Details

### Vector-to-Array Conversion

Spark ML classifiers return probabilities as a `Vector` object (DenseVector or SparseVector). The base class `calculate_stats()` expects an array for cross-entropy loss calculation. The conversion UDF handles this:

```python
@udf(returnType=ArrayType(DoubleType()))
def vector_to_array(v):
    """Convert Spark ML Vector to list of floats."""
    if v is None:
        return []
    return [float(x) for x in v.toArray()]
```

### Cross-Entropy Loss Calculation

The base class `calculate_stats()` method (moved from RNNSkolModel) computes cross-entropy loss:

```python
# Cross-entropy: -log(p(true_class))
prob_true_class = max(probabilities[int(true_label)], 1e-10)  # Avoid log(0)
loss = -np.log(prob_true_class)
```

This loss metric is now available for all model types.

## Backward Compatibility

✅ **Fully backward compatible**
- Existing code continues to work without changes
- `predict()` still returns predictions, just with added probabilities column
- Models that don't need probabilities can ignore the additional column
- No breaking changes to existing APIs

## Performance Considerations

- **Minimal overhead**: Vector-to-array conversion is efficient
- **Caching**: Results cached to avoid recomputation
- **Lazy evaluation**: Spark's lazy evaluation means conversion only happens when results are collected

## Related Changes

This implementation builds on the recent refactoring where `calculate_stats()` was moved from `RNNSkolModel` to `SkolModel` base class. Key related changes:

1. **[base_model.py](../skol_classifier/base_model.py:142-399)**: Comprehensive `calculate_stats()` implementation
2. **[rnn_model.py](../skol_classifier/rnn_model.py)**: Removed duplicate `calculate_stats()`, now uses base class
3. **All models now support**: Confusion matrix at verbosity >= 2, per-class metrics, loss calculation

## See Also

- **[Calculate Stats Refactoring](calculate_stats_refactoring.md)** - Details on unified stats implementation
- **[Base Model API](../skol_classifier/base_model.py)** - SkolModel interface
- **[Model Implementations](../skol_classifier/model.py)** - TraditionalMLSkolModel subclasses
- **[Test Suite](../examples/test_predict_proba.py)** - Comprehensive tests

---

**Implementation Date**: 2025-12-20
**Files Modified**: skol_classifier/model.py
**Tests Created**: examples/test_predict_proba.py
**Status**: ✅ Complete and tested
