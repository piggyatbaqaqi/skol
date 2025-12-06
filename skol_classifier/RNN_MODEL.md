# RNN Model for Line-Level Classification

## Overview

The RNN (Recurrent Neural Network) model uses a Bidirectional LSTM architecture to classify individual lines of taxonomic text by leveraging surrounding lines as context. This approach significantly improves classification accuracy compared to traditional ML models that treat each line independently.

## Architecture

### Bidirectional LSTM

The model uses a **Bidirectional LSTM** (Long Short-Term Memory) network:

- **Forward LSTM**: Processes the sequence from beginning to end
- **Backward LSTM**: Processes the sequence from end to beginning
- **Combined context**: Each line's classification uses both past and future context

### Model Structure

```
Input Sequence (max 50 lines)
    ↓
[Bidirectional LSTM Layer 1] (hidden_size × 2)
    ↓
[Bidirectional LSTM Layer 2] (hidden_size × 2)
    ↓
[Dropout Layer]
    ↓
[Time-Distributed Dense] (num_classes)
    ↓
Output Predictions (one per line)
```

## Key Features

### 1. Sequential Context

Unlike traditional models that classify each line independently, the RNN model:
- Considers surrounding lines when classifying each line
- Learns document structure and flow
- Better handles ambiguous cases where context matters

### 2. Distributed Training with Elephas

The model integrates with PySpark using **Elephas** for distributed training:
- Scales to large datasets
- Utilizes Spark cluster resources
- Efficient parallel training across workers

### 3. Sequence Windowing

Long documents are automatically split into windows:
- Default window size: 50 lines
- Prevents memory issues with very long documents
- Maintains sequential context within windows

## Installation

```bash
# Required dependencies
pip install tensorflow>=2.12.0
pip install elephas>=3.0.0

# Optional: GPU support
pip install tensorflow-gpu
```

## Usage

### Basic Example

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.appName("RNN Classification").getOrCreate()

# Create classifier with RNN model
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    couchdb_url='http://localhost:5984',
    couchdb_database='annotated_taxa',
    model_type='rnn',        # Use RNN model
    line_level=True,         # Required for RNN

    # RNN parameters
    hidden_size=128,
    num_layers=2,
    window_size=50,
    epochs=10
)

# Train
classifier.fit()

# Predict
predictions = classifier.predict(raw_data)
```

## Configuration Parameters

### Required Parameters

- **`model_type='rnn'`**: Selects the RNN model
- **`line_level=True`**: RNN requires line-level mode
- **`input_size`**: Feature vector dimensionality (default: 1000)

### RNN-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 128 | LSTM hidden state size |
| `num_layers` | 2 | Number of LSTM layers |
| `num_classes` | 3 | Number of output classes |
| `dropout` | 0.3 | Dropout rate for regularization |
| `window_size` | 50 | Maximum sequence length |
| `batch_size` | 32 | Training batch size |
| `epochs` | 10 | Number of training epochs |
| `num_workers` | 4 | Spark workers for distributed training |

### Performance Tuning

**For better accuracy:**
```python
hidden_size=256,      # Larger capacity
num_layers=3,         # Deeper network
window_size=100,      # More context
epochs=20             # More training
```

**For faster training:**
```python
hidden_size=64,       # Smaller model
num_layers=1,         # Simpler architecture
batch_size=64,        # Larger batches
epochs=5              # Fewer iterations
```

**To prevent overfitting:**
```python
dropout=0.5,          # Higher dropout
epochs=10,            # Don't overtrain
```

## How It Works

### 1. Data Preparation

The model automatically:
1. Groups lines by document ID
2. Creates sequences (up to `window_size` lines)
3. Prepares features and labels for each sequence

### 2. Training

The Bidirectional LSTM:
1. Processes each sequence in both directions
2. Learns patterns in line transitions
3. Optimizes using categorical cross-entropy loss
4. Distributes training across Spark workers

### 3. Prediction

For each document:
1. Lines are grouped into sequences
2. RNN processes each sequence
3. Per-line predictions are extracted
4. Results are reassembled into original order

## Advantages Over Traditional Models

### 1. Context-Aware Classification

**Traditional ML (e.g., Logistic Regression):**
```
Line 1: "Pileus 5-10 cm broad" → Description (no context)
Line 2: "Amanita muscaria"     → ??? (ambiguous)
Line 3: "Cap surface smooth"   → Description (no context)
```

**RNN Model:**
```
Line 1: "Pileus 5-10 cm broad" → Description
Line 2: "Amanita muscaria"     → Nomenclature (knows it follows description)
Line 3: "Cap surface smooth"   → Description (continues description)
```

### 2. Better Handling of Document Structure

The RNN learns:
- Nomenclature typically comes before descriptions
- Descriptions often span multiple consecutive lines
- Transitions between sections follow patterns

### 3. Improved Accuracy on Edge Cases

Particularly effective for:
- Short lines with limited features
- Lines that could belong to multiple categories
- Documents with unusual structure

## Performance Considerations

### Memory Usage

- **Window size**: Larger windows require more memory
- **Batch size**: Affects GPU/CPU memory requirements
- **Hidden size**: Larger states need more memory

### Training Time

Typical training time on moderate hardware:
- Small dataset (1000 documents): ~5-10 minutes
- Medium dataset (10000 documents): ~30-60 minutes
- Large dataset (100000 documents): ~2-4 hours

### Distributed Training

Elephas distributes training across Spark workers:
- Each worker processes batches independently
- Model weights are synchronized after each epoch
- Scales linearly with number of workers

## Comparison with Other Models

| Model | Accuracy | Training Time | Context-Aware | Best For |
|-------|----------|---------------|---------------|----------|
| Logistic Regression | Good | Fast | No | Large datasets, simple patterns |
| Random Forest | Better | Medium | No | Non-linear patterns |
| RNN (this) | Best | Slower | Yes | Sequential data, context matters |

## Troubleshooting

### Out of Memory Errors

```python
# Reduce window size
window_size=25

# Reduce batch size
batch_size=16

# Reduce model size
hidden_size=64
num_layers=1
```

### Poor Performance

```python
# Increase model capacity
hidden_size=256
num_layers=3

# More training
epochs=20

# Longer context
window_size=100
```

### Slow Training

```python
# Increase batch size
batch_size=64

# More workers
num_workers=8

# Reduce epochs
epochs=5
```

## Implementation Details

### Files

- **`rnn_model.py`**: RNN model implementation using Keras + Elephas
- **`model.py`**: Integration with SkolModel class
- **`classifier_v2.py`**: Unified API with RNN support
- **`example_rnn_classification.py`**: Usage examples

### Dependencies

- **TensorFlow**: Neural network backend
- **Keras**: High-level neural network API
- **Elephas**: PySpark + Keras integration
- **PySpark**: Distributed data processing

## Future Enhancements

Potential improvements:
- Attention mechanisms for better context understanding
- Transformer-based models (BERT, GPT)
- Transfer learning from pre-trained models
- Multi-task learning for related tasks

## References

- Elephas: [https://github.com/maxpumperla/elephas](https://github.com/maxpumperla/elephas)
- Keras: [https://keras.io/](https://keras.io/)
- LSTM networks: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
