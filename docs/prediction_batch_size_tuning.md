# Prediction Batch Size Tuning Guide

## Overview

The `prediction_batch_size` parameter controls how many RNN windows are predicted in a single batch during inference. This parameter significantly affects both **throughput** (speed) and **memory usage**.

## What is Prediction Batch Size?

During prediction, the RNN model processes sequences using a sliding window approach. Each sequence may generate multiple windows that need predictions. The `prediction_batch_size` parameter controls the maximum number of windows to predict in a single `model.predict()` call.

### Example

If you have:
- 100 sequences in a Spark partition
- Each sequence generates 5 windows on average
- Total windows = 500

With different batch sizes:
- **`prediction_batch_size=64`**: 500 windows ÷ 64 = 8 prediction calls
- **`prediction_batch_size=128`**: 500 windows ÷ 128 = 4 prediction calls
- **`prediction_batch_size=32`**: 500 windows ÷ 32 = 16 prediction calls

**Fewer prediction calls = faster throughput** (due to GPU/CPU parallelization and reduced overhead)

## How to Set It

### When Creating a New Model

```python
from skol_classifier import RNNClassifier

rnn_classifier = RNNClassifier(
    model_type='rnn',
    model_params={
        'input_size': 100,
        'window_size': 50,
        'prediction_stride': 25,
        'prediction_batch_size': 64,  # Set here
        ...
    }
)
```

### When Loading an Existing Model

You can change `prediction_batch_size` **without retraining** because it only affects inference, not the model weights.

**Option 1: Override during construction with `auto_load_model`** (Recommended):

```python
from skol_classifier import SkolClassifierV2

# Load model and override runtime parameters in one step
rnn_classifier = SkolClassifierV2(
    model_storage='redis',
    redis_client=redis_client,
    redis_key='my_model',
    auto_load_model=True,
    model_params={
        'prediction_batch_size': 128,  # Override saved value
        'num_workers': 8,              # Also works for workers
        'verbosity': 2                 # And verbosity
    }
)

# Model is loaded with new parameters already applied
predictions = rnn_classifier.predict(test_data)
```

**Option 2: Manual load and adjust**:

```python
# Load existing model manually
rnn_classifier = SkolClassifierV2(
    model_storage='redis',
    redis_client=redis_client,
    redis_key='my_model'
)
rnn_classifier.load_model()

# Then adjust batch size
rnn_classifier._model.prediction_batch_size = 128

# Predict with new batch size
predictions = rnn_classifier.predict(test_data)
```

## Performance vs Memory Trade-off

### Larger Batch Size (e.g., 128, 256)
**Pros:**
- ✅ **Faster predictions** - Better GPU/CPU utilization
- ✅ **Fewer TensorFlow retracing warnings** - More consistent batch shapes
- ✅ **Lower overhead** - Amortized across more windows

**Cons:**
- ❌ **Higher memory usage** - More data loaded simultaneously
- ❌ **Risk of OOM errors** - If batch doesn't fit in GPU/RAM

### Smaller Batch Size (e.g., 16, 32)
**Pros:**
- ✅ **Lower memory usage** - Safer for limited hardware
- ✅ **More stable** - Less likely to crash

**Cons:**
- ❌ **Slower predictions** - Underutilizes hardware
- ❌ **More overhead** - More prediction calls needed

## Tuning Recommendations

### Default Starting Point

```python
prediction_batch_size=64  # Good balance for most systems
```

### Based on Hardware

| Hardware Configuration | Recommended Batch Size | Notes |
|------------------------|------------------------|-------|
| **Limited GPU** (4-6GB VRAM) | 32-48 | Conservative, prevents OOM |
| **Standard GPU** (8-12GB VRAM) | 64-96 | Default recommendation |
| **High-end GPU** (16GB+ VRAM) | 128-256 | Maximum throughput |
| **CPU-only** | 32-64 | CPU memory is usually abundant |

### Tuning Process

1. **Start with default (64)**
2. **Monitor resource usage** during first prediction
3. **Adjust based on observations:**
   - If GPU/RAM usage is low (< 70%): **increase** batch size
   - If hitting OOM errors: **decrease** batch size
   - If performance is acceptable: **keep current setting**

## Example Analysis: 4-Worker Setup

### Current Hardware Profile

**Observed unbatched performance (batch_size=1):**
- 4 Spark executors (workers)
- **Host RAM per executor:** ~2GB resident
- **GPU RAM per executor:** 412MB
- **Available resources:**
  - Host RAM: ~25GB free
  - GPU RAM: ~5GB free (6.6GB total)

### Memory Scaling Estimates

GPU memory usage scales as:
```
GPU_memory ≈ base_overhead + (batch_size × window_memory)
```

Where:
- **base_overhead**: Model weights + fixed buffers (200-300MB)
- **window_memory**: Input/activation/output tensors per window

**Projected GPU usage for different batch sizes:**

| Batch Size | GPU per Executor | Total GPU (4 executors) | Safety Margin | Recommendation |
|------------|------------------|-------------------------|---------------|----------------|
| 32 | ~1.0 GB | 4.0 GB | ✅ Very safe (2.6GB free) | Conservative |
| 48 | ~1.2 GB | 4.8 GB | ✅ Safe (1.8GB free) | Safe choice |
| **64** | ~1.5 GB | 6.0 GB | ✅ Good (0.6GB free) | **Recommended** |
| 96 | ~1.8 GB | 7.2 GB | ⚠️ Tight (may OOM) | Aggressive |
| 128 | ~2.2 GB | 8.8 GB | ❌ Will OOM | Too large |

### Recommendation for This Setup

**Start with `prediction_batch_size=64`**

- Projected GPU usage: ~6GB total (90% utilization)
- Good balance of speed and safety
- If stable, can try 96 for ~20% speedup
- If OOM occurs, drop to 48

## Monitoring and Troubleshooting

### Monitor GPU Usage

```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Look for:
# - GPU memory usage per process
# - Any OOM errors
```

### Monitor Host RAM

```bash
# Check memory usage
htop

# Or
top
```

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Symptoms:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
```python
# Reduce batch size
rnn_classifier._model.prediction_batch_size = 32
```

#### 2. Slow Predictions

**Symptoms:**
- Predictions taking much longer than expected
- Low GPU utilization (< 50%)

**Solution:**
```python
# Increase batch size to better utilize GPU
rnn_classifier._model.prediction_batch_size = 128

# Also consider increasing Spark partitions
sequenced_data = preprocessor.transform(data).repartition(100)
```

#### 3. TensorFlow Retracing Warnings

**Symptoms:**
```
WARNING:tensorflow: N out of M calls triggered tf.function retracing
```

**Context:**
- Small amounts of retracing are normal
- Excessive retracing (>20%) indicates varying input shapes
- Larger batch sizes reduce retracing frequency

**Not usually a problem**, but if concerned:
```python
# Increase batch size for more consistent batches
rnn_classifier._model.prediction_batch_size = 96
```

## Interaction with Other Parameters

### `prediction_batch_size` vs `window_size`

- **`window_size`**: Fixed by trained model, controls RNN input sequence length
- **`prediction_batch_size`**: Can be changed anytime, controls how many windows to process together

**Independent parameters** - changing `prediction_batch_size` doesn't affect model accuracy.

### `prediction_batch_size` vs Spark Partitions

Both affect how work is distributed:

- **Spark partitions**: Distribute sequences across executors
- **`prediction_batch_size`**: Batch windows within each executor

**Example:**
```python
# 1000 sequences, 5 windows each = 5000 total windows
# 10 partitions = 500 windows per partition
# prediction_batch_size=64 = ~8 batches per partition
data.repartition(10)
```

**Guidelines:**
- More partitions = better parallelization but more overhead
- Aim for 2-4x your executor count
- Typical: 20-50 partitions for 4-8 executors

### `prediction_batch_size` vs `batch_size`

**Different parameters for different purposes:**

- **`batch_size`**: Used during **training** to control gradient updates
- **`prediction_batch_size`**: Used during **inference** to control memory/speed

Can be set independently:
```python
RNNSkolModel(
    batch_size=32,              # Training batch size
    prediction_batch_size=96,   # Prediction batch size (can be larger)
    ...
)
```

## Best Practices

### 1. Start Conservative, Then Optimize

```python
# First run: use safe default
prediction_batch_size=64

# Monitor resources
# If GPU < 70%: increase to 96 or 128
# If OOM: decrease to 32 or 48
```

### 2. Profile Your Workload

Different workloads have different characteristics:

```python
# Short sequences (< window_size): fewer windows, can use larger batches
# Long sequences (>> window_size): many windows, may need smaller batches

# Check average sequence length
avg_length = df.groupBy('doc_id').count().agg({'count': 'avg'}).collect()[0][0]
windows_per_seq = avg_length / window_size

# If windows_per_seq > 10: consider smaller prediction_batch_size
```

### 3. Test Before Production

```python
# Test with small dataset first
test_predictions = rnn_classifier.predict(small_test_data)

# Monitor memory during test run
# Adjust if needed
# Then run on full dataset
```

### 4. Document Your Settings

```python
# Save your tuned configuration
config = {
    'model_name': 'rnn_classifier_v1',
    'window_size': 50,
    'prediction_stride': 25,
    'prediction_batch_size': 64,  # Tuned for 4 workers, 6.6GB GPU
    'hardware_notes': '4x executors, NVIDIA RTX with 6.6GB VRAM'
}
```

## Summary

- **Default: 64** - Good starting point for most systems
- **GPU memory is usually the bottleneck** - Watch nvidia-smi
- **Can change without retraining** - Experiment freely
- **Larger = faster but more memory** - Find your hardware's sweet spot
- **Monitor first, then optimize** - Don't guess, measure

## Quick Reference

### Recommended: Load with auto_load_model

```python
from skol_classifier import SkolClassifierV2

# Load model with runtime parameter overrides
classifier = SkolClassifierV2(
    model_storage='redis',
    redis_client=redis_client,
    redis_key='my_model',
    auto_load_model=True,
    model_params={
        'prediction_batch_size': 64,  # Your tuned value
        'num_workers': 4,
        'verbosity': 2
    }
)

# Monitor during prediction
# watch -n 1 nvidia-smi  (in another terminal)

# Predict
predictions = classifier.predict(data)
```

### Alternative: Manual adjustment

```python
# Load model
model = RNNClassifier.load_from_redis(redis_client, "model_name")

# Adjust batch size manually
model._model.prediction_batch_size = 64  # Your tuned value

# Predict
predictions = model.predict(data)
```
