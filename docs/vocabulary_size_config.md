# Configuring Vocabulary Sizes

## Overview

You can now control the vocabulary sizes for TF-IDF features through `SkolClassifierV2` configuration:

- **`word_vocab_size`**: Maximum number of unique words in the vocabulary (default: 800)
- **`suffix_vocab_size`**: Maximum number of unique suffixes in the vocabulary (default: 200)

These parameters control how many features are extracted from the text, which directly affects:
1. Model input size (for RNN models)
2. Memory usage during training
3. Model capacity to learn patterns

## Basic Usage

### Default Configuration (1000 total features)

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.appName("Default Vocab").getOrCreate()

classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/annotated/*.ann'],
    model_type='rnn',

    # Default vocabulary sizes (can omit these)
    word_vocab_size=800,     # 800 word features
    suffix_vocab_size=200,   # 200 suffix features
    # Total input_size = 1000

    verbosity=1
)
```

### Larger Vocabulary (2000 total features)

```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/annotated/*.ann'],
    model_type='rnn',

    # Larger vocabulary for better coverage
    word_vocab_size=1800,    # 1800 word features
    suffix_vocab_size=200,   # 200 suffix features
    # Total input_size = 2000 (auto-calculated)

    # RNN configuration
    hidden_size=256,
    num_layers=4,
    window_size=35,
    batch_size=3276,

    verbosity=1
)
```

### Without Suffixes (1800 features)

```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/annotated/*.ann'],
    model_type='rnn',

    # Only word features, no suffixes
    word_vocab_size=1800,
    use_suffixes=False,      # Disable suffix features
    # Total input_size = 1800 (no suffix features)

    verbosity=1
)
```

## Automatic input_size Calculation

For RNN models, `input_size` is **automatically calculated** based on vocabulary sizes:

```
input_size = word_vocab_size + suffix_vocab_size (if use_suffixes=True)
input_size = word_vocab_size (if use_suffixes=False)
```

You **don't need to specify `input_size`** manually - it will be set automatically.

### Manual Override (Not Recommended)

If you explicitly provide `input_size`, it will override the automatic calculation with a warning:

```python
classifier = SkolClassifierV2(
    spark=spark,
    model_type='rnn',

    word_vocab_size=1800,
    suffix_vocab_size=200,   # Auto would calculate input_size=2000
    input_size=2500,         # Manual override (will show warning)

    verbosity=1
)

# Output:
# [Classifier] WARNING: input_size (2500) doesn't match calculated size (2000)
# from word_vocab_size (1800) + suffix_vocab_size (200). Using user-provided value.
```

## Memory Impact

Increasing vocabulary size has **minimal GPU memory impact** (~2-3% per 1000 features):

| Configuration | Total Features | GPU Memory (Est.) | Notes |
|---------------|---------------|-------------------|-------|
| Default | 1000 (800+200) | Baseline | Standard configuration |
| Medium | 1500 (1300+200) | +250 MB | ~1.5x features |
| Large | 2000 (1800+200) | +500 MB | ~2x features |
| Very Large | 3000 (2800+200) | +1 GB | ~3x features |

**Why so small?** Most GPU memory is used for LSTM activations (batch_size × window_size × hidden_size × num_layers), which don't scale with input_size. Only the first layer weights and input batch scale with vocabulary size.

## Recommended Configurations

### For Small Datasets (<100 documents)

```python
word_vocab_size=500,      # Avoid overfitting with smaller vocab
suffix_vocab_size=100,
# Total: 600 features
```

### For Medium Datasets (100-500 documents)

```python
word_vocab_size=800,      # Default
suffix_vocab_size=200,
# Total: 1000 features
```

### For Large Datasets (>500 documents)

```python
word_vocab_size=1800,     # Larger vocab for better coverage
suffix_vocab_size=200,
# Total: 2000 features
```

### For Specialized/Technical Domains

```python
word_vocab_size=2500,     # Many technical terms
suffix_vocab_size=300,    # Morphological variations
# Total: 2800 features
```

## Complete Example: Doubling Vocabulary

```python
import redis
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.appName("Large Vocab").getOrCreate()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Train with doubled vocabulary
classifier = SkolClassifierV2(
    # Data source
    spark=spark,
    input_source='files',
    file_paths=['data/annotated/*.ann'],

    # Model type
    model_type='rnn',  # or 'hybrid'

    # Feature configuration - DOUBLED VOCABULARY
    word_vocab_size=1800,      # 800 → 1800 (2.25x more words)
    suffix_vocab_size=200,     # Keep same
    use_suffixes=True,
    # Auto input_size = 2000

    # RNN configuration
    hidden_size=256,
    num_layers=4,
    window_size=35,
    batch_size=3276,           # Should still fit in 24GB GPU
    epochs=18,

    # Class weights
    class_weights={
        'Nomenclature': 80.0,
        'Description': 8.0,
        'Misc-exposition': 0.2
    },

    # Storage
    model_storage='redis',
    redis_client=redis_client,
    redis_key='rnn_large_vocab',

    verbosity=1
)

# Train
print("Training with doubled vocabulary (2000 features)...")
results = classifier.fit()

print(f"\nResults:")
print(f"  Nomenclature F1: {results['test_stats']['Nomenclature_f1']:.4f}")
print(f"  Description F1:  {results['test_stats']['Description_f1']:.4f}")
print(f"  Misc F1:         {results['test_stats']['Misc-exposition_f1']:.4f}")
print(f"  Overall F1:      {results['test_stats']['f1_score']:.4f}")

# Save
classifier.save_model()
print(f"✓ Model saved to Redis: rnn_large_vocab")
```

## Hybrid Model with Large Vocabulary

```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/annotated/*.ann'],
    model_type='hybrid',

    # Feature configuration - shared by both models
    word_vocab_size=1800,
    suffix_vocab_size=200,
    # Both logistic and RNN will use 2000-dimensional features

    # Hybrid configuration
    nomenclature_threshold=0.6,

    # RNN parameters (for Description/Misc)
    rnn_params={
        'window_size': 35,
        'hidden_size': 256,
        'num_layers': 3,
        'epochs': 15,
        'batch_size': 4000,
        'class_weights': {
            'Nomenclature': 80.0,
            'Description': 8.0,
            'Misc-exposition': 0.2
        }
    },

    verbosity=1
)

results = classifier.fit()
```

## Validation

The system will automatically validate that vocabulary sizes match the feature extraction:

```python
# After training, check actual vocabulary sizes
print(f"Word vocab size: {classifier._feature_extractor.word_vocab_size}")
print(f"Suffix vocab size: {classifier._feature_extractor.suffix_vocab_size}")
print(f"Model input size: {classifier._model.input_size}")  # For RNN models
```

## Troubleshooting

### Problem: OOM (Out of Memory) Error

**Symptom**: GPU runs out of memory during training

**Cause**: Combined effect of large vocabulary + large batch_size

**Solution**: Reduce batch_size proportionally
```python
# If vocab increased from 1000 → 2000, reduce batch_size slightly
batch_size=3000,  # Down from 3276
```

### Problem: Model Not Using Full Vocabulary

**Symptom**: Feature counts show fewer than expected features

**Cause**: Not enough unique words/suffixes in training data

**Solution**: Check your data
```python
# Count unique words in your data
from pyspark.sql.functions import explode, split

words_df = annotated_data.select(explode(split("value", " ")).alias("word"))
unique_words = words_df.distinct().count()
print(f"Unique words in data: {unique_words}")

# Adjust vocab_size to match
word_vocab_size=min(unique_words, 1800)
```

### Problem: No Improvement with Larger Vocabulary

**Symptom**: F1 scores don't improve when increasing vocab size

**Cause**: Either (1) default vocab already captured most signal, or (2) need more model capacity

**Solution**: Also increase hidden_size
```python
# Increase both vocabulary AND model capacity
word_vocab_size=1800,
suffix_vocab_size=200,
hidden_size=384,     # Also increase from 256
num_layers=4
```

## Best Practices

1. **Start with defaults** (800+200=1000) and measure performance
2. **Increase vocabulary** if you have >500 documents with technical terms
3. **Check GPU memory** after increasing vocabulary
4. **Don't exceed your data's unique words** - no benefit
5. **Balance with model capacity** - larger vocab may need larger hidden_size
6. **Let input_size auto-calculate** - don't override manually

## See Also

- [Hybrid Model Usage](hybrid_model_usage.md) - Combining logistic and RNN
- [Class Weights Usage](class_weights_usage.md) - Handling class imbalance
- [RNN Model Parameters](../skol_classifier/rnn_model.py) - Full RNN configuration
