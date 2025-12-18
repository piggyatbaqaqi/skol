# Custom Loss Function Serialization Fix

## Problem

When using class weights with RNN models, a custom `weighted_categorical_crossentropy` loss function is created. This caused serialization errors when:

1. **Distributed Prediction**: Model is serialized to JSON for Spark UDFs
2. **Model Loading**: Model is loaded from disk (.h5 files) or Redis

### Error Message

```
TypeError: Could not locate function 'weighted_categorical_crossentropy'.
Make sure custom classes and functions are decorated with `@keras.saving.register_keras_serializable()`.
```

### Root Cause

The custom loss function is created as a closure inside `build_bilstm_model()`:

```python
def build_bilstm_model(..., class_weights=None, labels=None):
    if class_weights is not None:
        # Create weight tensor
        weight_tensor = tf.constant(weight_list, dtype=tf.float32)

        # Create custom loss function (CLOSURE - captures weight_tensor)
        def weighted_categorical_crossentropy(y_true, y_pred):
            loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
            class_indices = tf.argmax(y_true, axis=-1)
            weights = tf.gather(weight_tensor, class_indices)
            weighted_loss = loss * weights
            return tf.reduce_mean(weighted_loss)

        loss_fn = weighted_categorical_crossentropy
```

When Keras serializes the model (via `to_json()` or `save()`), it saves:
- The model architecture
- The **name** of the loss function: `"weighted_categorical_crossentropy"`
- The compiled configuration

When deserializing, Keras tries to find this function but can't because:
1. It's not a registered Keras function
2. It's a dynamically created closure
3. Each model has a different closure (different `weight_tensor`)

## Solution

For **prediction only**, we don't need the loss function. The solution is to skip compilation when loading models:

### 1. Distributed Prediction (Spark UDF)

**File**: `skol_classifier/rnn_model.py`
**Line**: 971

```python
# Before
model = keras.models.model_from_json(model_config)

# After
model = keras.models.model_from_json(model_config, compile=False)
```

### 2. Loading from Disk

**File**: `skol_classifier/rnn_model.py`
**Line**: 1599

```python
# Before
self.keras_model = keras.models.load_model(path)

# After
self.keras_model = keras.models.load_model(path, compile=False)
```

### 3. Loading from Disk/Redis (via SkolClassifierV2)

**File**: `skol_classifier/classifier_v2.py`
**Line**: 1149

```python
# Before
keras_model = keras.models.load_model(str(classifier_path))

# After
keras_model = keras.models.load_model(str(classifier_path), compile=False)
```

## Why This Works

### `compile=False` Explained

When loading with `compile=False`:
- Keras loads **only the architecture** and **weights**
- Skips loading the optimizer, loss function, and metrics
- Model can still be used for prediction via `model.predict()`
- Much faster loading since no compilation step

### For Prediction
```python
# This works fine without compilation
predictions = model.predict(X)
probabilities = model.predict(X)  # softmax outputs
```

### For Training
If you need to continue training, call `fit()` which rebuilds the model:

```python
# Load model (not compiled)
classifier.load_model()

# Call fit() - this rebuilds and recompiles with current class_weights
classifier.fit()  # Works! Rebuilds model with loss function
```

## Impact

### ✅ What Still Works

1. **Prediction**: All prediction methods work normally
   - `classifier.predict()`
   - `classifier.predict_proba()`
   - Distributed prediction via Spark

2. **Training**: Can still train/retrain models
   - `classifier.fit()` rebuilds the model with proper loss
   - Class weights are reapplied

3. **Model Saving/Loading**: No changes to workflow
   - `classifier.save_model()`
   - `classifier.load_model()`

### ⚠️ Limitations

1. **Cannot inspect loss function**: After loading, `model.loss` is not available
   - This is fine - we don't use it for prediction
   - Training rebuilds it anyway

2. **Cannot continue training without fit()**: Can't call `model.fit()` directly on loaded Keras model
   - Solution: Use `classifier.fit()` which rebuilds the model

3. **Metrics not available**: Loaded model has no compiled metrics
   - Solution: Use `classifier.model.calculate_stats()` for evaluation

## Alternative Solutions Considered

### ❌ Registering Custom Loss Function

```python
@keras.saving.register_keras_serializable()
def weighted_categorical_crossentropy(y_true, y_pred):
    # Problem: Can't capture weight_tensor in decorated function
    # Each model has different weights
    pass
```

**Why not**: The decorator requires a static function, but our loss is a closure with model-specific weights.

### ❌ Custom Loss Class

```python
@keras.saving.register_keras_serializable()
class WeightedCategoricalCrossentropy(keras.losses.Loss):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def call(self, y_true, y_pred):
        # Implementation
        pass
```

**Why not**: More complex, requires changes to `build_bilstm_model()`, and still needs special handling for serialization of the weights array.

### ✅ `compile=False` (Chosen Solution)

**Why yes**:
- Simple one-line change
- No architectural changes needed
- Works for all use cases
- Standard Keras pattern for prediction-only loading
- Faster loading

## Testing

### Test That It Works

```python
from skol_classifier.classifier_v2 import SkolClassifierV2
import redis

# 1. Train with class weights
classifier = SkolClassifierV2(
    spark=spark,
    model_type='rnn',
    weight_strategy='inverse',  # Uses custom loss
    model_storage='redis',
    redis_client=redis.Redis(),
    redis_key='test_model'
)
classifier.fit()
classifier.save_model()

# 2. Load and predict (should work now!)
new_classifier = SkolClassifierV2(
    spark=spark,
    model_type='rnn',
    model_storage='redis',
    redis_client=redis.Redis(),
    redis_key='test_model',
    auto_load_model=True  # Loads with compile=False
)

# Prediction works
predictions = new_classifier.predict(test_data)  # ✅ Works!

# Can also retrain
new_classifier.fit()  # ✅ Works! Rebuilds with loss
```

### Verify Distributed Prediction

```python
# Large dataset - uses Spark UDFs
large_data = spark.read.parquet("large_dataset.parquet")

# This serializes model to JSON for UDFs
predictions = classifier.predict(large_data)  # ✅ Works!
```

## Files Modified

1. **`skol_classifier/rnn_model.py`**:
   - Line 971: `model_from_json(..., compile=False)`
   - Line 1599: `load_model(..., compile=False)`
   - Updated docstring for `load()` method

2. **`skol_classifier/classifier_v2.py`**:
   - Line 1149: `load_model(..., compile=False)`

## References

- Keras documentation: [Saving & Loading Models](https://keras.io/guides/serialization_and_saving/)
- TensorFlow guide: [Custom Losses](https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_losses)
- Related: `docs/class_weights_implementation_summary.md`
