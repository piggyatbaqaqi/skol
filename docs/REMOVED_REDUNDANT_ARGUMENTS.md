# Removed Redundant Command-Line Arguments

## Overview

Following the implementation of centralized command-line argument parsing in `bin/env_config.py`, redundant argument definitions have been removed from individual scripts. Configuration can now be set via command-line arguments to `env_config` rather than duplicating argument parsers in each script.

## Summary of Changes

### 1. bin/predict_classifier.py

**Removed Arguments:**
- `--pattern` (use `--couchdb-pattern` instead)
- `--batch-size` (use `--prediction-batch-size` instead)

**Migration:**

Before:
```bash
./bin/with_skol bin/predict_classifier.py \
    --model logistic_sections \
    --pattern "*.txt" \
    --batch-size 96
```

After:
```bash
./bin/with_skol bin/predict_classifier.py \
    --model logistic_sections \
    --couchdb-pattern "*.txt" \
    --prediction-batch-size 96
```

### 2. bin/extract_taxa_to_couchdb.py

**Removed Arguments:**
- `--ingest-url` (use `--ingest-url` via env_config)
- `--ingest-database` (use `--ingest-database` via env_config)
- `--ingest-username` (use `--ingest-username` via env_config)
- `--ingest-password` (use `--ingest-password` via env_config)
- `--taxon-url` (use `--taxon-url` via env_config)
- `--taxon-database` (use `--taxon-database` via env_config)
- `--taxon-username` (use `--taxon-username` via env_config)
- `--taxon-password` (use `--taxon-password` via env_config)
- `--pattern` (use `--pattern` via env_config)

**Kept Script-Specific Arguments:**
- `--debug-trace`
- `--debug-doc-id`

**Migration:**

Before:
```bash
python3 bin/extract_taxa_to_couchdb.py \
    --ingest-database mycobank_annotations \
    --taxon-database mycobank_taxa \
    --ingest-url http://localhost:5984 \
    --pattern "*.txt.ann"
```

After:
```bash
python3 bin/extract_taxa_to_couchdb.py \
    --ingest-database mycobank_annotations \
    --taxon-database mycobank_taxa \
    --ingest-url http://localhost:5984 \
    --pattern "*.txt.ann"
```

Note: The command-line syntax is unchanged because `env_config.py` now handles these arguments!

### 3. fixes/regenerate_txt_with_pages.py

**Removed Arguments:**
- `--couchdb-url` (use `--couchdb-url` via env_config)
- `--couchdb-username` (use `--couchdb-username` via env_config)
- `--couchdb-password` (use `--couchdb-password` via env_config)

**Modified Arguments:**
- `--database`: Now optional, falls back to `--couchdb-database` or `$COUCHDB_DATABASE`

**Migration:**

Before:
```bash
python3 fixes/regenerate_txt_with_pages.py \
    --database skol_dev \
    --couchdb-url http://localhost:5984 \
    --couchdb-username admin \
    --couchdb-password secret
```

After:
```bash
# Using env_config arguments
python3 fixes/regenerate_txt_with_pages.py \
    --couchdb-database skol_dev \
    --couchdb-url http://localhost:5984 \
    --couchdb-username admin \
    --couchdb-password secret

# Or keep using --database (for backward compatibility)
python3 fixes/regenerate_txt_with_pages.py \
    --database skol_dev \
    --couchdb-url http://localhost:5984
```

### 4. tests/test_page_marker_preservation.py

**Modified Arguments:**
- `--database`: Now optional, falls back to `--couchdb-database`, `$COUCHDB_DATABASE`, or `skol_dev`

**Migration:**

Before:
```bash
python3 tests/test_page_marker_preservation.py \
    --doc-id 0e4ec0213f3e540c9503efce61e58fe9 \
    --database skol_dev
```

After (unchanged, but can also use env_config):
```bash
# Using script-specific argument
python3 tests/test_page_marker_preservation.py \
    --doc-id 0e4ec0213f3e540c9503efce61e58fe9 \
    --database skol_dev

# Or using env_config argument
python3 tests/test_page_marker_preservation.py \
    --doc-id 0e4ec0213f3e540c9503efce61e58fe9 \
    --couchdb-database skol_dev
```

## Benefits

1. **Consistency**: All scripts use the same argument names for configuration
2. **Less duplication**: Argument parsing logic is centralized in `env_config.py`
3. **Easier maintenance**: Changes to configuration options only need to be made in one place
4. **Better documentation**: All configuration options are documented in one location

## Argument Name Mapping

| Configuration Key | Command-Line Argument | Description |
|-------------------|----------------------|-------------|
| `couchdb_database` | `--couchdb-database` | CouchDB database name |
| `couchdb_pattern` | `--couchdb-pattern` | File pattern to match (e.g., *.txt) |
| `prediction_batch_size` | `--prediction-batch-size` | Batch size for predictions |
| `ingest_database` | `--ingest-database` | Ingest database name |
| `taxon_database` | `--taxon-database` | Taxon database name |
| `pattern` | `--pattern` | Pattern for annotated files |

## Backward Compatibility

Most scripts maintain backward compatibility:

- **predict_classifier.py**: Use `--couchdb-pattern` instead of `--pattern`, `--prediction-batch-size` instead of `--batch-size`
- **extract_taxa_to_couchdb.py**: Fully compatible (arguments handled by env_config)
- **regenerate_txt_with_pages.py**: Mostly compatible (`--database` still works)
- **test_page_marker_preservation.py**: Fully compatible

## See Also

- [ENV_CONFIG_ARGUMENTS.md](ENV_CONFIG_ARGUMENTS.md) - Complete documentation of env_config command-line arguments
- [bin/env_config.py](../bin/env_config.py) - Centralized configuration module
