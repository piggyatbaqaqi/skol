# Redis Integration for SKOL Classifier

## Overview

The SKOL Classifier now supports model persistence using Redis, allowing you to save and load trained models efficiently. This is particularly useful for:

- **Model versioning**: Store multiple model versions with different Redis keys
- **Distributed systems**: Share models across different machines/processes
- **Fast loading**: Quick model retrieval from Redis cache
- **Production deployments**: Centralized model storage for microservices

## Installation

Add Redis to your dependencies:

```bash
pip install redis>=4.0.0
```

Or install with the updated requirements:

```bash
pip install -e .
```

## New Features

### 1. Constructor Changes

The `SkolClassifier` constructor now accepts Redis configuration:

```python
SkolClassifier(
    spark=None,
    redis_client=None,           # NEW: Redis client connection
    redis_key="skol_classifier_model",  # NEW: Key name for Redis storage
    auto_load=True               # NEW: Auto-load model if key exists
)
```

### 2. Auto-Load Feature

By default, if you provide a `redis_client` and the `redis_key` exists in Redis, the model will be **automatically loaded** during initialization. This means you don't need to call `load_from_redis()` explicitly!

```python
# Model automatically loads if "my_model" exists in Redis
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="my_model"
)

# Check if model was loaded
if classifier.labels is not None:
    print("Model loaded successfully!")
else:
    print("No model found, need to train")
```

To disable auto-loading:
```python
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="my_model",
    auto_load=False  # Don't auto-load
)
```

### 3. New Methods

#### `save_to_redis(redis_client=None, redis_key=None)`

Save trained models to Redis as a compressed tar.gz archive containing:
- Pipeline model (feature extraction)
- Classifier model (trained classifier)
- Metadata (labels, version info)

**Returns**: `True` if successful, `False` otherwise

**Example**:
```python
success = classifier.save_to_redis()
if success:
    print("Model saved!")
```

#### `load_from_redis(redis_client=None, redis_key=None)`

Load previously saved models from Redis.

**Returns**: `True` if successful, `False` otherwise

**Example**:
```python
classifier = SkolClassifier(redis_client=redis_client)
success = classifier.load_from_redis(redis_key="my_model")
if success:
    print("Model loaded!")
```

#### `save_to_disk(path)`

Save trained models to a local directory (alternative to Redis).

**Example**:
```python
classifier.save_to_disk("/models/skol_v1")
```

#### `load_from_disk(path)`

Load models from a local directory.

**Example**:
```python
classifier.load_from_disk("/models/skol_v1")
```

## Usage Patterns

### Pattern 1: Training and Saving

```python
import redis
from skol_classifier import SkolClassifier, get_file_list

# Connect to Redis
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False  # IMPORTANT: Must be False for binary data
)

# Initialize with Redis
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="production_model_v1"
)

# Train
files = get_file_list("/data/annotated")
results = classifier.fit(files)

# Save to Redis
if classifier.save_to_redis():
    print("Model saved successfully!")
```

### Pattern 2: Loading and Predicting (Auto-Load)

```python
import redis
from skol_classifier import SkolClassifier, get_file_list

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Initialize - model auto-loads if key exists!
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="production_model_v1"
)

# Check if model was loaded
if classifier.labels is not None:
    print(f"Model loaded with labels: {classifier.labels}")

    # Use the loaded model
    raw_files = get_file_list("/data/raw")
    predictions = classifier.predict_raw_text(raw_files)
else:
    print("No model found in Redis, need to train first")
```

### Pattern 3: Train-or-Load Pattern

The auto-load feature makes it easy to implement a "train if needed, otherwise load" pattern:

```python
import redis
from skol_classifier import SkolClassifier, get_file_list

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Initialize - auto-loads if model exists
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="my_production_model"
)

# Check if we need to train
if classifier.labels is None:
    print("Training new model...")
    annotated_files = get_file_list("/data/annotated")
    classifier.fit(annotated_files)
    classifier.save_to_redis()
else:
    print(f"Using cached model: {classifier.labels}")

# Use the model (trained or loaded)
raw_files = get_file_list("/data/raw")
predictions = classifier.predict_raw_text(raw_files)
```

### Pattern 4: Multiple Model Versions

```python
import redis
from skol_classifier import SkolClassifier, get_file_list

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
files = get_file_list("/data/annotated")

# Train different models (disable auto_load to avoid loading existing models)
lr_model = SkolClassifier(redis_client=redis_client, auto_load=False)
lr_model.fit(files, model_type="logistic")
lr_model.save_to_redis(redis_key="model_logistic_v1")

rf_model = SkolClassifier(redis_client=redis_client, auto_load=False)
rf_model.fit(files, model_type="random_forest")
rf_model.save_to_redis(redis_key="model_rf_v1")

# Later, load specific version (auto-loads by default)
lr_classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="model_logistic_v1"
)
print(f"LR model loaded: {lr_classifier.labels is not None}")

rf_classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="model_rf_v1"
)
print(f"RF model loaded: {rf_classifier.labels is not None}")
```

### Pattern 5: Production Deployment with Custom Redis Config

```python
import redis
from skol_classifier import SkolClassifier

# Production Redis with authentication and SSL
redis_client = redis.Redis(
    host='redis.production.com',
    port=6380,
    db=0,
    password='secure_password',
    ssl=True,
    ssl_cert_reqs='required',
    decode_responses=False
)

# Load production model
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="production_model"
)

classifier.load_from_redis()
```

## Implementation Details

### How Models are Stored

1. **Temporary Directory**: Models are first saved to a temp directory
2. **Archiving**: Directory is compressed into a tar.gz archive in memory
3. **Redis Storage**: Archive is stored as a binary blob in Redis
4. **Cleanup**: Temporary directory is automatically cleaned up

### Data Structure in Redis

```
redis_key → tar.gz archive containing:
    ├── pipeline_model/       (PySpark PipelineModel)
    │   └── [model files]
    ├── classifier_model/     (PySpark PipelineModel)
    │   └── [model files]
    └── metadata.json         (labels, version info)
```

### Error Handling

All save/load methods include error handling:
- Return `False` on failure
- Print error messages to stdout
- Clean up temporary resources
- Raise `ValueError` for configuration issues

## Redis Configuration

### Local Development

```python
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False
)
```

### Docker Redis

```python
redis_client = redis.Redis(
    host='redis-container',  # Docker service name
    port=6379,
    db=0,
    decode_responses=False
)
```

### Redis Cluster

```python
from redis.cluster import RedisCluster

redis_client = RedisCluster(
    host='redis-cluster.example.com',
    port=6379,
    decode_responses=False
)
```

### Redis Sentinel

```python
from redis.sentinel import Sentinel

sentinel = Sentinel([('sentinel1', 26379), ('sentinel2', 26379)])
redis_client = sentinel.master_for('mymaster', decode_responses=False)
```

## Best Practices

1. **Key Naming**: Use descriptive keys with versioning
   ```python
   redis_key = f"skol_model_{model_type}_{date}_{version}"
   ```

2. **Model Versioning**: Store multiple versions for rollback
   ```python
   classifier.save_to_redis(redis_key="model_v1_backup")
   classifier.save_to_redis(redis_key="model_v2_current")
   ```

3. **Error Handling**: Always check return values
   ```python
   if not classifier.load_from_redis():
       # Fallback to disk or retrain
       classifier.load_from_disk("/backup/model")
   ```

4. **Binary Mode**: Always use `decode_responses=False` for Redis client
   ```python
   # CORRECT
   redis.Redis(decode_responses=False)

   # WRONG - will cause errors
   redis.Redis(decode_responses=True)
   ```

5. **Memory Management**: Large models may consume significant Redis memory
   - Monitor Redis memory usage
   - Use appropriate Redis eviction policies
   - Consider compression trade-offs

## Troubleshooting

### Issue: "No Redis client available"
**Solution**: Pass `redis_client` to constructor or method call

### Issue: "No model found in Redis with key: X"
**Solution**: Verify key exists using `redis-cli KEYS '*'`

### Issue: Model fails to deserialize
**Solution**: Ensure `decode_responses=False` in Redis client

### Issue: Out of memory errors
**Solution**: Check model size and Redis maxmemory configuration

## Performance Considerations

- **Model Size**: Typical models are 10-100 MB compressed
- **Save Time**: ~5-30 seconds depending on model size
- **Load Time**: ~3-10 seconds from Redis
- **Network**: Loading from remote Redis adds latency

## Migration from Disk-Only

If you have existing disk-based models:

```python
# Load from disk
classifier = SkolClassifier()
classifier.load_from_disk("/old/model/path")

# Save to Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
classifier.save_to_redis(redis_client=redis_client, redis_key="migrated_model_v1")
```

## See Also

- [examples/redis_usage.py](examples/redis_usage.py) - Complete Redis examples
- [skol_classifier/README.md](skol_classifier/README.md) - Full API documentation
- [Redis Python Documentation](https://redis-py.readthedocs.io/)
