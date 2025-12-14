# Redis and CouchDB Methods Report

This report documents all methods that use Redis or CouchDB across classes imported in `jupyter/ist769_skol.ipynb`.

## Classes with CouchDB Methods

| Module | Class Name | Alias | CouchDB Methods |
|--------|------------|-------|-----------------|
| `skol_classifier.couchdb_io` | `CouchDBConnection` | `CDBC` | `_connect()`, `db` (property), `get_all_doc_ids()`, `get_document_list()`, `fetch_partition()`, `save_partition()`, `load_distributed()`, `save_distributed()`, `process_partition_with_func()` |
| `skol_classifier.output_formatters` | `CouchDBOutputWriter` | `CDBOW` | `__init__()`, `save_annotated()` |
| `skol_classifier.classifier_v2` | `SkolClassifierV2` | `SC` | `load_raw()`, `_load_raw_from_couchdb()`, `_load_annotated_from_couchdb()`, `_save_to_couchdb()` |
| `couchdb_file` | `CouchDBFile` | `CDBF` | `__init__()`, `filename` (property), `doc_id` (property), `attachment_name` (property), `db_name` (property), `human_url` (property), `read_couchdb_partition()` (function), `read_couchdb_rows()` (function), `read_couchdb_files_from_connection()` (function) |
| `taxa_json_translator` | `TaxaJSONTranslator` | `TJT` | `__init__()`, `load_taxa()`, `save_taxa()` |
| `dr_drafts_mycosearch.data` | `SKOL_TAXA` | `STX` | `__init__()`, `load_data()` |

## Classes with Redis Methods

| Module | Class Name | Alias | Redis Methods |
|--------|------------|-------|---------------|
| `skol_classifier.classifier_v2` | `SkolClassifierV2` | `SC` | `_save_model_to_redis()`, `_load_model_from_redis()` |
| `taxon_clusterer` | `TaxonClusterer` | `TC` | `__init__()`, `load_embeddings()` |
| `dr_drafts_mycosearch.compute_embeddings` | `EmbeddingsComputer` | `EC` | `__init__()`, `write_embeddings_to_redis()` |

## Summary Statistics

- **Total Classes Analyzed**: 8
- **Classes with CouchDB methods**: 6
- **Classes with Redis methods**: 3
- **Total CouchDB methods**: 28
- **Total Redis methods**: 5

## Method Names for Copying into Notebook

### CouchDBConnection (CDBC)
```python
_connect, db, get_all_doc_ids, get_document_list, fetch_partition, save_partition, load_distributed, save_distributed, process_partition_with_func
```

### CouchDBOutputWriter (CDBOW)
```python
__init__, save_annotated
```

### SkolClassifierV2 (SC) - CouchDB
```python
load_raw, _load_raw_from_couchdb, _load_annotated_from_couchdb, _save_to_couchdb
```

### SkolClassifierV2 (SC) - Redis
```python
_save_model_to_redis, _load_model_from_redis
```

### CouchDBFile (CDBF)
```python
__init__, filename, doc_id, attachment_name, db_name, human_url, read_couchdb_partition, read_couchdb_rows, read_couchdb_files_from_connection
```

### TaxaJSONTranslator (TJT)
```python
__init__, load_taxa, save_taxa
```

### TaxonClusterer (TC) - Redis
```python
__init__, load_embeddings
```

### SKOL_TAXA (STX) - CouchDB
```python
__init__, load_data
```

### EmbeddingsComputer (EC) - Redis
```python
__init__, write_embeddings_to_redis
```

## Detailed Method Descriptions

### CouchDBConnection (CDBC)

All methods in this class interact with CouchDB:

- `_connect()` - Initializes CouchDB server connection using `couchdb.Server()`
- `db` (property) - Returns database object, connects if necessary
- `get_all_doc_ids()` - Queries CouchDB to retrieve document IDs from database
- `get_document_list()` - Retrieves list of documents with attachments from CouchDB
- `fetch_partition()` - Fetches CouchDB attachment content for a partition using `db.get_attachment()`
- `save_partition()` - Saves annotated content to CouchDB using `db.put_attachment()` and `db.save()`
- `load_distributed()` - Distributed load from CouchDB using mapPartitions
- `save_distributed()` - Distributed save to CouchDB using mapPartitions
- `process_partition_with_func()` - Generic read-process-save operation with CouchDB

### CouchDBOutputWriter (CDBOW)

- `__init__()` - Creates CouchDBConnection instance
- `save_annotated()` - Saves predictions to CouchDB by calling `self.conn.save_distributed()`

### SkolClassifierV2 (SC)

**CouchDB Methods:**
- `load_raw()` - Loads raw text from CouchDB (routes to `_load_raw_from_couchdb()`)
- `_load_raw_from_couchdb()` - Creates CouchDBConnection and calls `conn.load_distributed()`
- `_load_annotated_from_couchdb()` - Loads annotated data from CouchDB using CouchDBConnection
- `_save_to_couchdb()` - Saves predictions to CouchDB using CouchDBOutputWriter

**Redis Methods:**
- `_save_model_to_redis()` - Saves trained model to Redis using `self.redis_client.set()` and `self.redis_client.expire()`
- `_load_model_from_redis()` - Loads model from Redis using `self.redis_client.get()`

### CouchDBFile (CDBF)

All methods and properties in this class relate to CouchDB:

- `__init__()` - Initializes a CouchDB file-like object from attachment content
- `filename` (property) - Returns composite identifier for CouchDB documents
- `doc_id` (property) - Returns CouchDB document ID
- `attachment_name` (property) - Returns attachment filename
- `db_name` (property) - Returns CouchDB database name
- `human_url` (property) - Returns URL from CouchDB row
- `read_couchdb_partition()` - Reads annotated files from CouchDB rows in a PySpark partition
- `read_couchdb_rows()` - Reads annotated files from a list of CouchDB rows
- `read_couchdb_files_from_connection()` - Loads and reads annotated files from CouchDB using CouchDBConnection

### TaxaJSONTranslator (TJT)

All methods interact with CouchDB:

- `__init__()` - Stores CouchDB URL and credentials
- `load_taxa()` - Loads taxa from CouchDB using CouchDBConnection, queries documents with `conn.get_all_doc_ids()`, and uses `conn.db` to retrieve documents
- `save_taxa()` - Saves taxa DataFrame to CouchDB using `couchdb.Server()` and `db.save()` within a mapPartitions function

### TaxonClusterer (TC)

All methods interact with Redis:

- `__init__()` - Creates Redis client connection using `redis.Redis()`
- `load_embeddings()` - Loads embeddings from Redis using `self.redis_client.exists()` and `self.redis_client.get()`, with pickle deserialization

### SKOL_TAXA (STX)

All methods interact with CouchDB:

- `__init__()` - Stores CouchDB connection parameters and calls `load_data()`
- `load_data()` - Connects to CouchDB using `couchdb.Server()`, accesses database, fetches all taxon documents, and loads them into a pandas DataFrame

### EmbeddingsComputer (EC)

Methods that interact with Redis:

- `__init__()` - Stores Redis connection parameters (url, username, password, db, expire time)
- `write_embeddings_to_redis()` - Writes computed embeddings to Redis using `redis.from_url()`, `r.set()`, and `r.expire()`

---

**Generated**: 2025-12-14
**Purpose**: Documentation for IST769 SKOL project notebook integration
