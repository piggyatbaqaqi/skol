# Weight Strategy: Automatic Class Weights

## Overview

The `weight_strategy` parameter provides a simple way to automatically compute and apply class weights to handle class imbalance, without manually calculating or specifying weights.

**Supported for all model types:**
- Logistic Regression
- Random Forest
- Gradient Boosted Trees
- RNN/BiLSTM

Each model type implements weights differently internally, but the API is unified and automatic.

## Quick Start

### Before: Manual Class Weights

```python
# Old way: manually specify weights
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=annotated_files,
    model_type='rnn',
    class_weights={
        "Nomenclature": 100.0,
        "Description": 10.0,
        "Misc": 0.1
    },
    **other_config
)
```

### After: Automatic Weight Strategy

```python
# New way: automatic weight calculation
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=annotated_files,
    model_type='rnn',
    weight_strategy='inverse',  # Automatically computes and applies weights
    **other_config
)
```

## Available Strategies

### 1. `'inverse'` - Inverse Frequency (Recommended)

Weights are inversely proportional to class frequency, normalized to a reasonable range.

**Best for**: Balanced improvement across all classes

**Example output**:
```
[Classifier] Label Frequencies:
  Misc                     9523 ( 87.2%)
  Description              1234 ( 11.3%)
  Nomenclature              165 (  1.5%)
  Total                   10922 (100.0%)

[Classifier] Applied 'inverse' weight strategy:
  Nomenclature         100.00
  Description           14.19
  Misc                   0.10
```

### 2. `'balanced'` - Sklearn-style Balanced

Uses the sklearn formula: `n_samples / (n_classes * n_samples_per_class)`.

**Best for**: Standard balanced learning

**Example output**:
```
[Classifier] Applied 'balanced' weight strategy:
  Nomenclature          22.07
  Description            2.95
  Misc                   0.38
```

### 3. `'aggressive'` - Aggressive Upweighting

Squares the inverse frequency weights for stronger focus on minority classes.

**Best for**: Extreme class imbalance or when minority class recall is critical

**Example output**:
```
[Classifier] Applied 'aggressive' weight strategy:
  Nomenclature         100.00
  Description           20.14
  Misc                   0.01
```

## Complete Example

```python
import redis
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

# Initialize
spark = SparkSession.builder.appName("Auto Weights").getOrCreate()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Configure model with automatic weight strategy
model_config = {
    'model_type': 'rnn',
    'window_size': 15,
    'prediction_stride': 5,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'epochs': 6,
    'batch_size': 32,
    'verbosity': 1,
}

# Create classifier with weight strategy
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/annotated/*.ann'],
    auto_load_model=False,
    model_storage='redis',
    redis_client=redis_client,
    redis_key='rnn_auto_weights',
    weight_strategy='inverse',  # Automatic weight calculation
    **model_config
)

# Train - weights are automatically computed and applied
results = classifier.fit()

# Output shows automatic process:
# [Classifier] Label Frequencies:
#   Misc                     9523 ( 87.2%)
#   Description              1234 ( 11.3%)
#   Nomenclature              165 (  1.5%)
#   Total                   10922 (100.0%)
#
# [Classifier] Applied 'inverse' weight strategy:
#   Nomenclature         100.00
#   Description           14.19
#   Misc                   0.10
#
# [BiLSTM] Using weighted loss with class weights:
#   Description: 14.19
#   Misc: 0.10
#   Nomenclature: 100.0

# Evaluate
stats = results['test_stats']
print(f"Nomenclature F1: {stats['Nomenclature_f1']:.4f}")
print(f"Description F1: {stats['Description_f1']:.4f}")
```

## How It Works

When you specify `weight_strategy`, the classifier automatically:

1. **Enables frequency computation**: Sets `compute_label_frequencies=True` internally
2. **Loads annotated data**: Computes label frequencies from training data
3. **Calculates recommended weights**: Calls `get_recommended_class_weights(strategy=...)`
4. **Applies weights to model**: Stores weights in `model_params['class_weights']`
5. **Passes weights to RNN**: The weights flow through to the model's loss function

The process is fully automatic - you just specify the strategy and everything else happens during `fit()`.

## Choosing a Strategy

### Start with `'inverse'`

For most use cases, start with the inverse frequency strategy:

```python
classifier = SkolClassifierV2(
    ...,
    weight_strategy='inverse'
)
```

This provides balanced improvement across classes with reasonable weights.

### Use `'balanced'` for standard weighting

If you're familiar with sklearn's class_weight='balanced':

```python
classifier = SkolClassifierV2(
    ...,
    weight_strategy='balanced'
)
```

This uses the same formula as sklearn's balanced weights.

### Use `'aggressive'` for extreme imbalance

When minority classes are extremely rare and critical:

```python
classifier = SkolClassifierV2(
    ...,
    weight_strategy='aggressive'
)
```

This heavily penalizes errors on minority classes.

## Combining with Manual Weights

You can still manually specify weights if needed:

```python
# Manual weights override weight_strategy
classifier = SkolClassifierV2(
    ...,
    weight_strategy='inverse',  # This is ignored
    class_weights={             # These are used instead
        "Nomenclature": 150.0,
        "Description": 20.0,
        "Misc": 0.05
    }
)
```

If both `weight_strategy` and `class_weights` are provided, `class_weights` takes precedence.

## Advanced: Custom Weight Adjustment

If you want to see the computed weights before applying them:

```python
# Option 1: Compute manually then specify
classifier = SkolClassifierV2(
    ...,
    compute_label_frequencies=True,  # Enable frequency computation
    verbosity=1
)

# Fit feature extractor (without training model yet)
annotated_data = classifier._load_annotated_data()

# Get recommended weights
weights = classifier.get_recommended_class_weights(strategy='inverse')
print(f"Recommended weights: {weights}")

# Adjust if needed
weights['Nomenclature'] *= 1.5  # Boost Nomenclature even more
weights['Misc'] *= 0.5          # Reduce Misc even further

# Apply adjusted weights
classifier.model_params['class_weights'] = weights

# Now train with adjusted weights
results = classifier.fit(annotated_data=annotated_data)
```

## Monitoring

With `verbosity >= 1`, you'll see:

1. **Label frequencies** when data is loaded:
   ```
   [Classifier] Label Frequencies:
     Misc                     9523 ( 87.2%)
     Description              1234 ( 11.3%)
     Nomenclature              165 (  1.5%)
   ```

2. **Applied weights** when strategy is used:
   ```
   [Classifier] Applied 'inverse' weight strategy:
     Nomenclature         100.00
     Description           14.19
     Misc                   0.10
   ```

3. **Model confirmation** when training starts:
   ```
   [BiLSTM] Using weighted loss with class weights:
     Description: 14.19
     Misc: 0.10
     Nomenclature: 100.0
   ```

## Troubleshooting

### Weight strategy has no effect

**Symptom**: No message about weights being applied during training

**Cause**: Labels not available or weight_strategy not set

**Solution**:
1. Ensure `weight_strategy` is set to 'inverse', 'balanced', or 'aggressive'
2. Set `verbosity=1` to see weight application messages
3. Check that training data has labeled examples

### Weights seem wrong

**Symptom**: Computed weights don't match expectations

**Solution**: Check label frequencies and try different strategy:
```python
# See what frequencies were computed
classifier = SkolClassifierV2(..., weight_strategy='inverse', verbosity=1)
results = classifier.fit()  # Will print frequencies

# Try different strategy
classifier = SkolClassifierV2(..., weight_strategy='aggressive', verbosity=1)
```

### Want to disable weights

**Symptom**: Want to turn off automatic weights

**Solution**: Set `weight_strategy=None` (default):
```python
classifier = SkolClassifierV2(
    ...,
    weight_strategy=None  # No automatic weights
)
```

## Comparison with Manual Weights

| Feature | Manual `class_weights` | Automatic `weight_strategy` |
|---------|----------------------|---------------------------|
| Ease of use | Need to calculate weights | Single parameter |
| Flexibility | Full control over values | Three preset strategies |
| Reproducibility | Same weights every time | Adapts to data distribution |
| Code clarity | Explicit weights visible | Strategy name is self-documenting |
| Best for | Fine-tuned experiments | Quick iteration, standard use |

## References

- Implementation details: `docs/class_weights_implementation_summary.md`
- Manual class weights: `docs/class_weights_usage.md`
- All strategies: `docs/class_imbalance_strategies.md`
