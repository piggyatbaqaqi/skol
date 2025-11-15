# Changelog

## Version 0.1.0 (Current)

### Features

#### Redis Integration
- Added Redis support for model persistence
- Models stored as compressed tar.gz archives in Redis
- Auto-load feature: automatically loads models from Redis if key exists
- Methods: `save_to_redis()`, `load_from_redis()`
- Disk persistence also available: `save_to_disk()`, `load_from_disk()`

#### CouchDB Integration
- Read raw text from CouchDB document attachments
- Write annotated results back to CouchDB as new attachments
- Preserves document IDs and attachment names through pipeline
- Methods: `load_from_couchdb()`, `predict_from_couchdb()`, `save_to_couchdb()`
- `CouchDBReader` and `CouchDBWriter` classes for low-level access

#### Model Auto-Loading
- `auto_load=True` parameter in constructor (default)
- Automatically checks Redis for existing models on initialization
- Enables "train-or-load" pattern with minimal code

### API

#### New Constructor Parameters
- `redis_client`: Optional Redis client connection
- `redis_key`: Key name for Redis storage (default: "skol_classifier_model")
- `auto_load`: Auto-load from Redis if key exists (default: True)

#### New Methods

**Redis:**
- `save_to_redis(redis_client=None, redis_key=None)` - Save models to Redis
- `load_from_redis(redis_client=None, redis_key=None)` - Load models from Redis
- `save_to_disk(path)` - Save models to disk
- `load_from_disk(path)` - Load models from disk

**CouchDB:**
- `load_from_couchdb(couchdb_url, database, username=None, password=None, pattern="*.txt")` - Load text from CouchDB
- `predict_from_couchdb(couchdb_url, database, username=None, password=None, pattern="*.txt", output_format="annotated")` - Process CouchDB documents
- `save_to_couchdb(predictions, couchdb_url, database, username=None, password=None, suffix=".ann")` - Save to CouchDB

### New Modules

- `skol_classifier/couchdb_io.py` - CouchDB I/O utilities
  - `CouchDBReader` class
  - `CouchDBWriter` class
  - Factory functions: `create_couchdb_reader()`, `create_couchdb_writer()`

### Dependencies

- Added `redis>=4.0.0`
- Added `requests>=2.25.0`
- Fixed `spark-nlp` package name (was `sparknlp>=3.0.0`)

### Documentation

- [REDIS_INTEGRATION.md](REDIS_INTEGRATION.md) - Complete Redis integration guide
- [AUTO_LOAD_FEATURE.md](AUTO_LOAD_FEATURE.md) - Auto-load feature documentation
- [COUCHDB_INTEGRATION.md](COUCHDB_INTEGRATION.md) - CouchDB integration guide
- Updated [skol_classifier/README.md](skol_classifier/README.md) with new features

### Examples

- [examples/redis_usage.py](examples/redis_usage.py) - Redis integration examples
- [examples/couchdb_usage.py](examples/couchdb_usage.py) - CouchDB integration examples
- Updated [examples/basic_usage.py](examples/basic_usage.py) and [examples/model_comparison.py](examples/model_comparison.py)

### Bug Fixes

- Fixed package name in requirements: `spark-nlp` (not `sparknlp>=3.0.0`)

## Migration Guide

### From Local Files to CouchDB

**Before:**
```python
from skol_classifier import SkolClassifier, get_file_list

classifier = SkolClassifier()
files = get_file_list("/data/raw", pattern="**/*.txt")
predictions = classifier.predict_raw_text(files)
classifier.save_annotated_output(predictions, "/output/annotated")
```

**After:**
```python
from skol_classifier import SkolClassifier

classifier = SkolClassifier()
predictions = classifier.predict_from_couchdb(
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)
classifier.save_to_couchdb(
    predictions,
    couchdb_url="http://localhost:5984",
    database="documents",
    username="admin",
    password="password"
)
```

### Adding Redis Caching

**Before:**
```python
classifier = SkolClassifier()
# Train every time
files = get_file_list("/data/annotated")
classifier.fit(files)
```

**After:**
```python
import redis

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)
classifier = SkolClassifier(redis_client=redis_client, redis_key="my_model")

# Auto-loads if exists, otherwise train and save
if classifier.labels is None:
    files = get_file_list("/data/annotated")
    classifier.fit(files)
    classifier.save_to_redis()
```

## Breaking Changes

None. All changes are backward compatible.

## Contributors

- Christopher Murphy
- La Monte Henry Piggy Yarroll
- David Caspers
