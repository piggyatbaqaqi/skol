# Centralized Verbosity Configuration

## Overview

Verbosity level is now centralized in `bin/env_config.py` and can be configured via command-line arguments or environment variables across all SKOL scripts. This eliminates redundant `--verbosity` arguments in individual scripts.

## Configuration

### Command-Line Argument

```bash
./bin/with_skol bin/predict_classifier.py --verbosity 2
```

### Environment Variable

```bash
export VERBOSITY=2
./bin/with_skol bin/predict_classifier.py
```

### Default Value

If not specified, verbosity defaults to `1` (info level).

## Verbosity Levels

- **0 (silent)**: No output except errors
- **1 (info)**: Standard informational messages (default)
- **2 (debug)**: Detailed debug information

## Scripts Updated

### 1. bin/train_classifier.py

**Before:**
```bash
python3 bin/train_classifier.py --model logistic_sections --verbosity 2
```

**After:**
```bash
# Use env_config verbosity
python3 bin/train_classifier.py --model logistic_sections --verbosity 2
```

Note: The command-line syntax is the same, but `--verbosity` is now handled by `env_config` rather than the script itself.

**Changes:**
- Removed `--verbosity` argument from script-specific parser
- Removed `verbosity_override` parameter from `train_classifier()` function
- Now uses `config['verbosity']` directly

### 2. bin/predict_classifier.py

**Before:**
```bash
python3 bin/predict_classifier.py --model logistic_sections --verbosity 1
```

**After:**
```bash
# Use env_config verbosity
python3 bin/predict_classifier.py --model logistic_sections --verbosity 1
```

**Changes:**
- Removed `--verbosity` argument from script-specific parser
- Removed `verbosity_override` parameter from `predict_and_save()` function
- Now uses `config['verbosity']` directly

### 3. bin/embed_taxa.py

**Before:**
```bash
python3 bin/embed_taxa.py --force --verbosity 2
```

**After:**
```bash
# Use env_config verbosity
python3 bin/embed_taxa.py --force --verbosity 2
```

**Changes:**
- Removed `--verbosity` argument from script-specific parser
- Now uses `config['verbosity']` directly in function call

### 4. fixes/regenerate_txt_with_pages.py

**Before:**
```bash
python3 fixes/regenerate_txt_with_pages.py --database skol_dev --verbosity 2
```

**After:**
```bash
# Use env_config verbosity
python3 fixes/regenerate_txt_with_pages.py --database skol_dev --verbosity 2
```

**Changes:**
- Removed `--verbosity` argument from script-specific parser
- Now uses `config['verbosity']` directly

### 5. bin/extract_taxa_to_couchdb.py

**Before:**
```bash
python3 bin/extract_taxa_to_couchdb.py --ingest-database mycobank_annotations --taxon-database mycobank_taxa
```

**After:**
```bash
# Use env_config verbosity
python3 bin/extract_taxa_to_couchdb.py --ingest-database mycobank_annotations --taxon-database mycobank_taxa --verbosity 2
```

**Changes:**
- Added `verbosity` parameter to `TaxonExtractor.__init__()`
- Updated error messages and print statements to respect verbosity levels:
  - `verbosity >= 1`: Standard error messages
  - `verbosity >= 2`: Debug schema information and failed document details
- Now uses `config['verbosity']` throughout the extraction pipeline

## Benefits

1. **Consistency**: All scripts use the same verbosity configuration
2. **No duplication**: Verbosity parsing logic exists only in `env_config.py`
3. **Global control**: Set verbosity once via environment variable for all scripts
4. **Easier maintenance**: Changes to verbosity handling happen in one place

## Usage Examples

### Set Verbosity for All Scripts

```bash
# Set via environment variable
export VERBOSITY=2

# All scripts will use debug verbosity
./bin/with_skol bin/train_classifier.py --model logistic_sections
./bin/with_skol bin/predict_classifier.py --model logistic_sections
python3 bin/embed_taxa.py --force
```

### Override Per Script

```bash
# Default verbosity is 1
export VERBOSITY=1

# Override for specific script
./bin/with_skol bin/predict_classifier.py --verbosity 2 --model logistic_sections
```

### Silent Mode

```bash
# Suppress output
./bin/with_skol bin/train_classifier.py --verbosity 0 --model logistic_sections
```

## Implementation Details

### env_config.py

Added verbosity to the configuration dictionary:

```python
base_config = {
    # ... other config ...

    # General settings
    'verbosity': int(os.environ.get('VERBOSITY', '1')),
}

# Integer arguments
for key in ['redis_port', 'embedding_expire', 'prediction_batch_size', 'num_workers', 'cores', 'verbosity']:
    arg_name = '--' + key.replace('_', '-')
    parser.add_argument(arg_name, type=int, default=None, dest=key)
```

### Script Updates

Scripts now use `config['verbosity']` instead of `args.verbosity`:

```python
# Old
parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=None)
model_config['verbosity'] = args.verbosity

# New
config = get_env_config()
model_config['verbosity'] = config['verbosity']
```

## Backward Compatibility

The command-line interface remains the same - scripts still accept `--verbosity`, but it's now processed by `env_config.py` rather than each individual script.

## See Also

- [ENV_CONFIG_ARGUMENTS.md](ENV_CONFIG_ARGUMENTS.md) - Complete documentation of env_config arguments
- [REMOVED_REDUNDANT_ARGUMENTS.md](REMOVED_REDUNDANT_ARGUMENTS.md) - Documentation of other removed redundant arguments
