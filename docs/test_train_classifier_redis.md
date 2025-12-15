# Test Program: Train Classifier with Redis Storage

## Overview

`tests/test_train_classifier_redis.py` is a standalone test program extracted from the Jupyter notebook (`jupyter/ist769_skol.ipynb`) that trains a SKOL text classifier and saves it to Redis.

This program demonstrates:
- Training a text classifier on annotated taxonomic data
- Saving the trained model to Redis for fast retrieval
- Using SkolClassifierV2 with the unified API
- Supporting both RNN (BiLSTM) and Logistic Regression models

## Source

This test program is extracted from the notebook block:
```
# Train classifier on annotated data and save to Redis using SkolClassifierV2
```

## Prerequisites

### System Requirements
- Python 3.8+
- Redis server running on localhost:6379
- CouchDB server running on localhost:5984
- Apache Spark
- Sufficient memory (16GB+ recommended for RNN model)
- CUDA-capable GPU (optional, for faster RNN training)

### Python Dependencies
```bash
pip install pyspark redis tensorflow keras scikit-learn
```

### Data Requirements
Annotated training files must be available in:
```
/data/piggy/src/github.com/piggyatbaqaqi/skol/data/annotated/
```

Files should have `.ann` extension (YEDDA annotation format).

## Usage

### Basic Usage

Train RNN model (default):
```bash
cd /data/piggy/src/github.com/piggyatbaqaqi/skol
python tests/test_train_classifier_redis.py
```

### Train Logistic Regression Model
```bash
python tests/test_train_classifier_redis.py --model-type logistic
```

### Force Retraining
If a model already exists in Redis, use `--force` to retrain:
```bash
python tests/test_train_classifier_redis.py --force
```

### Command-Line Arguments

- `--model-type {rnn,logistic}`: Model type to train (default: rnn)
- `--force`: Force retraining even if model already exists in Redis

### Examples

Train RNN model with forced retraining:
```bash
python tests/test_train_classifier_redis.py --model-type rnn --force
```

Train Logistic Regression:
```bash
python tests/test_train_classifier_redis.py --model-type logistic
```

## Configuration

The script uses these default configurations (can be modified in the script):

### Redis Configuration
```python
redis_host = "127.0.0.1"
redis_port = 6379
redis_db = 0
classifier_model_name = "skol:classifier:model:{model_type}-v1.0"
classifier_model_expire = 172800  # 2 days in seconds
```

### CouchDB Configuration
```python
couchdb_host = "127.0.0.1:5984"
couchdb_username = "admin"
couchdb_password = "SU2orange!"
ingest_db_name = "skol_dev"
```

### Spark Configuration
```python
cores = 2
driver_memory = "16g"
executor_memory = "20g"
```

## Model Configurations

### RNN (BiLSTM) Model

```python
{
    "name": "RNN BiLSTM (line-level, advanced config)",
    "model_type": "rnn",
    "use_suffixes": True,
    "line_level": True,
    "input_size": 1000,
    "hidden_size": 128,
    "num_layers": 2,
    "num_classes": 3,
    "dropout": 0.3,
    "window_size": 20,
    "prediction_stride": 20,  # No overlap
    "prediction_batch_size": 32,
    "batch_size": 16384,  # ~16.7GB GPU memory
    "epochs": 4,
    "num_workers": 2,
    "verbosity": 2,
}
```

**Expected Performance (4 epochs):**
- Test Accuracy:  0.7990
- Test Precision: 0.7990
- Test Recall:    1.0000
- Test F1 Score:  0.7098

**Memory Requirements:**
- GPU Memory: ~16.7GB (with batch_size=16384)
- Training time: ~38-40 seconds per step

### Logistic Regression Model

```python
{
    "name": "Logistic Regression (line-level, words + suffixes)",
    "model_type": "logistic",
    "use_suffixes": True,
    "maxIter": 10,
    "regParam": 0.01,
    "line_level": True
}
```

**Characteristics:**
- Much faster training than RNN
- Lower memory requirements
- Good baseline performance
- No GPU required

## Output

The program outputs:
1. Configuration details
2. Training progress
3. Final model performance metrics
4. Redis storage confirmation

Example output:
```
Connecting to Redis at 127.0.0.1:6379...

Configuring RNN model...
Model configuration:
  model_type: rnn
  use_suffixes: True
  line_level: True
  input_size: 1000
  hidden_size: 128
  ...

Loading annotated files from: /data/piggy/src/github.com/piggyatbaqaqi/skol/data/annotated
Found 42 annotated files

Creating Spark session...

======================================================================
Training classifier with SkolClassifierV2...
======================================================================

Starting training...
Epoch 1/4
...
Epoch 4/4

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

## Troubleshooting

### Error: Directory does not exist
**Problem:** Annotated data directory not found

**Solution:** Ensure annotated training data is available at:
```
/data/piggy/src/github.com/piggyatbaqaqi/skol/data/annotated/
```

### Error: No annotated files found
**Problem:** No `.ann` files in the annotated directory

**Solution:** Add YEDDA-annotated `.ann` files to the data directory

### Error: Redis connection refused
**Problem:** Redis server not running

**Solution:** Start Redis server:
```bash
redis-server
```

### Error: Out of memory (GPU)
**Problem:** Insufficient GPU memory for RNN model

**Solution:** Reduce `batch_size` in the configuration:
```python
"batch_size": 8192,  # Use smaller batch size
```

### Error: CouchDB connection failed
**Problem:** CouchDB not running or incorrect credentials

**Solution:**
1. Start CouchDB server
2. Verify credentials in the script
3. Ensure database `skol_dev` exists

## Redis Model Keys

Models are stored in Redis with these keys:

- RNN model: `skol:classifier:model:rnn-v1.0`
- Logistic model: `skol:classifier:model:logistic-v1.0`

### Check if Model Exists

```bash
redis-cli EXISTS skol:classifier:model:rnn-v1.0
```

### View Model Expiration

```bash
redis-cli TTL skol:classifier:model:rnn-v1.0
```

### Delete Model

```bash
redis-cli DEL skol:classifier:model:rnn-v1.0
```

## Integration with Notebook

This test program can be used to:
1. Train models outside the notebook environment
2. Test model training in CI/CD pipelines
3. Automate model retraining on schedule
4. Verify configuration changes before running in notebook

## Related Files

- Source notebook: `jupyter/ist769_skol.ipynb`
- Classifier implementation: `skol_classifier/classifier_v2.py`
- Model implementations: `skol_classifier/rnn_model.py`, `skol_classifier/model.py`
- Utilities: `skol_classifier/utils.py`

## Notes

- The script uses the same configuration as the notebook for consistency
- Models are automatically saved to Redis with expiration
- Use `--force` to retrain existing models
- GPU is recommended but not required for RNN training
- The script will exit with error code 1 on failure

## Future Enhancements

Potential improvements:
- Add command-line arguments for Redis/CouchDB configuration
- Support for custom data paths
- Model evaluation on test set
- Export model metrics to file
- Integration with MLflow for experiment tracking
