# Sliding Window Prediction for RNN Models

## Overview

The RNN classifier in SKOL uses a sliding window approach to classify arbitrarily long documents, overcoming the fixed window size limitation inherent in recurrent neural networks.

## The Problem

### Fixed Window Size Limitation

RNN models have a fixed `window_size` parameter (e.g., 20 timesteps) that determines:
- How many lines the model can process at once during training
- The sequence length the model expects during prediction

**Without sliding windows:**
- A document with 1,196 lines would only get predictions for the **first 20 lines**
- Lines 21-1,196 would receive `null` predictions
- Most of the document would be unclassified

### Why This Happens

When using `arrays_zip` to combine predictions with features:
```python
arrays_zip("predictions", "sequence_features", "sequence_labels", "sequence_line_numbers")
```

If arrays have different lengths:
- `predictions`: [p1, p2, ..., p20] (20 items)
- `sequence_features`: [f1, f2, ..., f1196] (1,196 items)

Spark pads the shorter array with `null` values, causing predictions 21-1,196 to be null.

## The Solution: Sliding Windows

### Concept

Instead of making a single prediction on the first `window_size` lines, we:

1. **Create overlapping windows** that slide across the entire document
2. **Predict each window independently**
3. **Aggregate overlapping predictions** by averaging probability distributions
4. **Return predictions for all lines** in the document

### Visual Example

For a document with 50 lines, `window_size=20`, `stride=10`:

```
Document: [L1, L2, L3, ..., L48, L49, L50]

Window 1: [L1  - L20]           → Predictions for L1-L20
Window 2:      [L11 - L30]      → Predictions for L11-L30
Window 3:            [L21 - L40] → Predictions for L21-L40
Window 4:                  [L31 - L50] → Predictions for L31-L50

Lines with multiple predictions (averaged):
- L11-L20: Covered by Window 1 and Window 2
- L21-L30: Covered by Window 2 and Window 3
- L31-L40: Covered by Window 3 and Window 4
```

## Implementation

### Configuration Parameters

#### `window_size` (Training & Prediction)
- **Purpose**: Maximum sequence length the RNN can process
- **Training**: Sequences are padded/truncated to this length
- **Prediction**: Size of each sliding window
- **Typical values**: 20-100
- **Memory impact**: Higher values require more GPU/CPU memory

#### `prediction_stride` (Prediction Only)
- **Purpose**: How many lines to advance between windows
- **Default**: `None` (becomes `window_size` - non-overlapping)
- **Recommended**: `window_size // 2` (50% overlap)
- **Range**: 1 to `window_size`
- **Trade-off**: Smaller stride = more overlap = better quality but slower

### Algorithm

```python
# For each document/sequence
if sequence_length <= window_size:
    # Short sequence: single prediction
    predictions = predict_once(sequence)
else:
    # Long sequence: sliding windows
    prediction_counts = zeros(sequence_length)
    prediction_sums = zeros((sequence_length, num_classes))

    # Slide window across sequence
    for window_start in range(0, sequence_length, prediction_stride):
        window_end = min(window_start + window_size, sequence_length)
        window = sequence[window_start:window_end]

        # Predict this window
        window_predictions = model.predict(window)

        # Accumulate predictions
        for i, pred in enumerate(window_predictions):
            global_position = window_start + i
            prediction_sums[global_position] += pred  # prob distribution
            prediction_counts[global_position] += 1

    # Average overlapping predictions
    for i in range(sequence_length):
        avg_probs = prediction_sums[i] / prediction_counts[i]
        predictions[i] = argmax(avg_probs)
```

### Key Implementation Details

**Probability Averaging** ([rnn_model.py:870-913](../skol_classifier/rnn_model.py#L870-L913)):
- Each window produces a probability distribution over classes (e.g., [0.1, 0.7, 0.2])
- For overlapping positions, we sum all probability distributions
- Final prediction uses argmax of the averaged probabilities
- This is more robust than majority voting of class labels

**Padding Handling** ([rnn_model.py:881-884](../skol_classifier/rnn_model.py#L881-L884)):
- Last window may be shorter than `window_size`
- We pad it to `window_size` for the model
- But only use predictions for actual (non-padded) positions

**Memory Efficiency**:
- Predictions are made one window at a time
- Only accumulator arrays (`prediction_sums`, `prediction_counts`) span full sequence
- Memory usage: O(sequence_length × num_classes)

## Configuration Examples

### Example 1: Maximum Speed (No Overlap)

```python
model_config = {
    "window_size": 20,
    "prediction_stride": 20,  # or None (default)
}
```

**Behavior:**
- Document with 100 lines → 5 non-overlapping windows
- Each line predicted once
- Fastest option
- Potential discontinuities at window boundaries

**Use when:**
- Speed is critical
- Documents are well-structured with clear boundaries
- Quick prototyping

### Example 2: Balanced (50% Overlap)

```python
model_config = {
    "window_size": 20,
    "prediction_stride": 10,  # window_size // 2
}
```

**Behavior:**
- Document with 100 lines → 9 windows
- Most lines predicted twice (averaged)
- Good quality/speed balance
- **Recommended default**

**Use when:**
- Production deployment
- Good balance of accuracy and performance
- Most common use case

### Example 3: High Quality (75% Overlap)

```python
model_config = {
    "window_size": 20,
    "prediction_stride": 5,  # window_size // 4
}
```

**Behavior:**
- Document with 100 lines → 17 windows
- Most lines predicted 4 times (averaged)
- Highest quality predictions
- 4x slower than no overlap

**Use when:**
- Quality is paramount
- Processing time is not constrained
- Critical classification tasks
- Research/evaluation

### Example 4: Maximum Overlap (Line-by-Line)

```python
model_config = {
    "window_size": 20,
    "prediction_stride": 1,
}
```

**Behavior:**
- Document with 100 lines → 81 windows
- Each line predicted up to 20 times
- Extremely smooth transitions
- Very slow (20x slower than no overlap)

**Use when:**
- Absolute maximum quality needed
- Very small test sets
- Benchmarking
- Research experiments

## Performance Considerations

### Prediction Time

For a document with `N` lines:

| Stride | Windows | Time Multiplier | Overlap |
|--------|---------|-----------------|---------|
| `window_size` (20) | N/20 ≈ 5 | 1x (baseline) | 0% |
| `window_size // 2` (10) | N/10 ≈ 10 | ~2x | 50% |
| `window_size // 4` (5) | N/5 ≈ 20 | ~4x | 75% |
| 1 | N ≈ 100 | ~20x | 95% |

**Example**: Document with 1,000 lines, window_size=20
- Stride 20: 50 windows
- Stride 10: 99 windows (2x slower)
- Stride 5: 197 windows (4x slower)
- Stride 1: 981 windows (20x slower)

### Memory Usage

**Per sequence in UDF:**
```
prediction_counts: N × 4 bytes (int32)
prediction_sums:   N × num_classes × 4 bytes (float32)
Total: N × (4 + num_classes × 4) bytes
```

**Example**: 1,000 lines, 3 classes
- prediction_counts: 4 KB
- prediction_sums: 12 KB
- Total: ~16 KB per sequence

This is negligible compared to Spark's overhead.

### Distributed Processing

Sliding windows operate **within each sequence** in the Pandas UDF:
- Different documents are still processed in parallel across Spark partitions
- Sliding window overhead is per-document, not per-dataset
- Large datasets with many small documents scale well
- Large datasets with few huge documents may have load imbalance

## Usage in SKOL Classifier

### Training Configuration

```python
from skol_classifier.classifier_v2 import SkolClassifierV2

classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=annotated_files,
    model_type='rnn',
    model_storage='redis',

    # RNN-specific parameters
    window_size=20,           # Training window size
    prediction_stride=10,     # Prediction stride (50% overlap)
    hidden_size=128,
    num_layers=2,
    batch_size=16384,
    epochs=4,

    # Other parameters...
    line_level=True,
    use_suffixes=True,
    verbosity=2
)

# Train
results = classifier.fit()

# Save with stride configuration
classifier.save_model()
```

### Prediction

```python
# Load model (includes stride configuration)
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    model_storage='redis',
    auto_load_model=True,
    # prediction_stride is loaded from saved model
)

# Predict on long documents
raw_data = classifier.load_raw()
predictions = classifier.predict(raw_data)

# All lines classified (no nulls!)
predictions.filter("prediction IS NULL").count()  # Returns 0
```

### Monitoring and Debugging

The UDF logs to `/tmp/rnn_udf_debug_*.log`:

```bash
# View logs
ls -lth /tmp/rnn_udf_debug_* | head -5
cat $(ls -t /tmp/rnn_udf_debug_* | head -1)
```

**Log output example:**
```
[UDF START] Processing 102 sequences
[UDF START] Using sliding window: window_size=20, stride=10
[UDF SEQ 0] Long sequence: 1196 features, using sliding windows
[UDF SEQ 0] Used 119 windows, generated 1196 predictions
[UDF COMPLETE] Processed 102 sequences, 102 results
[UDF COMPLETE] Non-empty results: 102
[UDF COMPLETE] Empty results: 0
```

## Benefits

### ✅ Complete Coverage
- All lines in all documents receive predictions
- No null predictions regardless of document length

### ✅ Smooth Transitions
- Overlapping windows provide context from both sides
- Reduces boundary effects
- More consistent predictions

### ✅ Configurable Trade-offs
- Adjust stride to balance quality vs. speed
- Same model works for different use cases
- Easy to experiment

### ✅ Backward Compatible
- `prediction_stride=None` behaves like old code (first window only)
- Existing models work without changes
- Gradual migration path

## Limitations

### Training Still Uses Fixed Windows
- Training data is still windowed at `window_size`
- Model never sees sequences longer than `window_size` during training
- Sliding windows help prediction, not training

**Implication**: Model learns patterns within `window_size` context
- Longer-range dependencies (>20 lines apart) not learned
- Increasing `window_size` improves this but requires more memory

### Computational Cost
- Overlapping windows increase prediction time
- Trade-off between quality and speed
- May need larger Spark clusters for large datasets

### No Ground Truth for Overlaps
- When windows disagree, averaging assumes equal confidence
- Some windows may have better context than others
- Could weight by position within window (future improvement)

## Future Enhancements

### Adaptive Stride
```python
# Smaller stride near paragraph boundaries
# Larger stride within homogeneous regions
adaptive_stride = calculate_stride(context)
```

### Confidence Weighting
```python
# Weight predictions by distance from window center
# Central predictions may be more reliable
weights = gaussian_weights(window_size)
weighted_avg = sum(pred * weight) / sum(weights)
```

### Hierarchical Windowing
```python
# First pass: coarse windows (stride = window_size)
# Second pass: fine windows (stride = window_size // 4) on uncertain regions
if confidence < threshold:
    refine_with_smaller_stride()
```

## References

- **Implementation**: [skol_classifier/rnn_model.py:842-913](../skol_classifier/rnn_model.py#L842-L913)
- **Configuration**: [skol_classifier/rnn_model.py:224-264](../skol_classifier/rnn_model.py#L224-L264)
- **Save/Load**: [skol_classifier/classifier_v2.py:789-807](../skol_classifier/classifier_v2.py#L789-L807)

## Summary

Sliding window prediction enables RNN models to classify documents of any length by:
1. Breaking long sequences into overlapping windows
2. Predicting each window independently
3. Averaging predictions for overlapping positions
4. Returning complete predictions (no nulls)

**Recommended configuration**: `prediction_stride = window_size // 2` for a good balance of quality and speed.
