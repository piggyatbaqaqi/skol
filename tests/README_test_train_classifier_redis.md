# Test Program: test_train_classifier_redis.py

## Quick Start

```bash
# Train RNN model (default)
python tests/test_train_classifier_redis.py

# Train Logistic Regression model
python tests/test_train_classifier_redis.py --model-type logistic

# Force retraining
python tests/test_train_classifier_redis.py --force

# See all options
python tests/test_train_classifier_redis.py --help
```

## What This Program Does

This standalone test program extracts and automates the classifier training workflow from the Jupyter notebook (`jupyter/ist769_skol.ipynb`). It:

1. **Loads annotated training data** from `data/annotated/`
2. **Configures a Spark session** with CouchDB integration
3. **Trains a text classifier** using SkolClassifierV2
4. **Saves the model to Redis** with automatic expiration
5. **Reports performance metrics** (accuracy, precision, recall, F1)

## Why Use This Test Program?

### Advantages over Notebook
- ✅ **Automated**: No manual cell execution
- ✅ **Reproducible**: Same configuration every time
- ✅ **Scriptable**: Can be used in automation/CI/CD
- ✅ **Command-line**: Easy to run from terminal
- ✅ **Testable**: Proper error handling and exit codes

### Use Cases
- Quick model retraining
- Testing configuration changes
- Automated model updates
- Batch training jobs
- CI/CD integration

## Prerequisites

### Required Services
- ✓ Redis server (localhost:6379)
- ✓ CouchDB server (localhost:5984)
- ✓ Apache Spark

### Required Data
- ✓ Annotated training files in `data/annotated/*.ann`

### Python Packages
```bash
pip install pyspark redis tensorflow keras scikit-learn
```

## Model Types

### RNN (BiLSTM) - Default
- **Training time**: ~15-30 minutes (depends on data size and GPU)
- **Memory**: ~17GB GPU memory
- **Accuracy**: ~0.80
- **Best for**: High accuracy, large datasets

### Logistic Regression
- **Training time**: ~2-5 minutes
- **Memory**: ~4GB RAM
- **Accuracy**: ~0.75
- **Best for**: Quick baselines, testing

## Output

The program saves:
1. **Redis Key**: `skol:classifier:model:{model_type}-v1.0`
2. **Expiration**: 2 days (configurable)
3. **Contents**: Complete trained model with metadata

## Verify Model in Redis

```bash
# Check if model exists
redis-cli EXISTS skol:classifier:model:rnn-v1.0

# View expiration time
redis-cli TTL skol:classifier:model:rnn-v1.0

# Get model size
redis-cli MEMORY USAGE skol:classifier:model:rnn-v1.0
```

## Configuration

Edit the script to change:
- Redis connection (host, port, db)
- CouchDB connection (host, credentials)
- Model parameters (hidden_size, epochs, etc.)
- Spark configuration (memory, cores)

## Documentation

Full documentation available in:
- [docs/test_train_classifier_redis.md](../docs/test_train_classifier_redis.md)

## Source

Extracted from notebook block:
```python
# Train classifier on annotated data and save to Redis using SkolClassifierV2
```

Located at: `jupyter/ist769_skol.ipynb`

## Example Output

```
Connecting to Redis at 127.0.0.1:6379...

Configuring RNN model...
Model configuration:
  model_type: rnn
  hidden_size: 128
  num_layers: 2
  epochs: 4
  ...

Loading annotated files from: .../data/annotated
Found 42 annotated files

Creating Spark session...

======================================================================
Training classifier with SkolClassifierV2...
======================================================================

Starting training...
Epoch 1/4
████████████████████████████████████ 100/100 [00:38s]
...
Epoch 4/4
████████████████████████████████████ 100/100 [00:40s]

======================================================================
Training complete!
======================================================================
  Accuracy:  0.7990
  Precision: 0.7990
  Recall:    1.0000
  F1 Score:  0.7098

Saving model to Redis...
✓ Model saved to Redis with key: skol:classifier:model:rnn-v1.0
  Expiration: 172800 seconds (2.0 days)

✓ Training complete!
```

## Exit Codes

- `0`: Success
- `1`: Error (missing data, training failure, etc.)

## Troubleshooting

### Common Issues

**"Directory does not exist"**
→ Create `data/annotated/` and add `.ann` files

**"Redis connection refused"**
→ Start Redis: `redis-server`

**"Out of memory"**
→ Reduce `batch_size` in script or use `--model-type logistic`

**"No annotated files found"**
→ Add YEDDA-annotated `.ann` files to `data/annotated/`

See full troubleshooting guide in [docs/test_train_classifier_redis.md](../docs/test_train_classifier_redis.md)

## Related Files

- **Test script**: `tests/test_train_classifier_redis.py` (this file)
- **Documentation**: `docs/test_train_classifier_redis.md`
- **Source notebook**: `jupyter/ist769_skol.ipynb`
- **Classifier**: `skol_classifier/classifier_v2.py`
- **RNN model**: `skol_classifier/rnn_model.py`
