# Taxa JSON Translation - Quick Start Guide

## Overview

Translate taxa descriptions to structured JSON features using a fine-tuned Mistral model.

## Installation

```bash
pip install transformers peft accelerate bitsandbytes torch
```

## Basic Usage

### 1. Simple Translation

```python
from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import TaxonExtractor
from taxa_json_translator import TaxaJSONTranslator

# Initialize
spark = SparkSession.builder.master("local[*]").getOrCreate()

extractor = TaxonExtractor(
    spark=spark,
    ingest_couchdb_url="http://localhost:5984",
    ingest_db_name="mycobank_annotations",
    taxon_db_name="mycobank_taxa",
    ingest_username="admin",
    ingest_password="password"
)

translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./mistral_checkpoints/checkpoint-100"
)

# Load, translate, save
taxa_df = extractor.load_taxa()
enriched_df = translator.translate_descriptions(taxa_df)
translator.save_translations(enriched_df, "output/taxa_features.parquet")

spark.stop()
```

### 2. With Validation

```python
# Translate
enriched_df = translator.translate_descriptions(taxa_df)

# Validate
validated_df = translator.validate_json(enriched_df)

# Save only valid
valid_df = validated_df.filter("json_valid = true")
translator.save_translations(valid_df, "output/valid_taxa.parquet")
```

### 3. Batch Processing

```python
# For large datasets
enriched_df = translator.translate_descriptions_batch(
    taxa_df,
    batch_size=20  # Adjust based on GPU memory
)
```

### 4. Single Description

```python
description = "Pileus 5-10 cm, white. Stipe 8-12 cm."
features = translator.translate_single(description)

print(features)
# {'pileus': {'diameter': ['5-10 cm'], 'color': ['white']}, ...}
```

## Running the Example

```bash
# Set environment variables
export COUCHDB_URL="http://localhost:5984"
export COUCHDB_USER="admin"
export COUCHDB_PASSWORD="password"
export MISTRAL_CHECKPOINT="./mistral_checkpoints/checkpoint-100"

# Run example
python example_taxa_translation.py
```

## Common Patterns

### Pattern 1: Filter by Source

```python
# Load specific taxa
taxa_df = extractor.load_taxa(pattern="taxon_abc*")

# Translate
enriched_df = translator.translate_descriptions(taxa_df)
```

### Pattern 2: Save Multiple Formats

```python
# Save as Parquet
translator.save_translations(enriched_df, "output/taxa.parquet", format="parquet")

# Save as JSON
translator.save_translations(enriched_df, "output/taxa.json", format="json")

# Save as CSV
translator.save_translations(enriched_df, "output/taxa.csv", format="csv")
```

### Pattern 3: Custom Prompt

```python
custom_prompt = "Extract only morphological features as JSON."

translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./checkpoints/checkpoint-100",
    prompt=custom_prompt
)
```

### Pattern 4: CPU-Only

```python
# For machines without GPU
translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./checkpoints/checkpoint-100",
    device="cpu",
    load_in_4bit=False
)
```

## Configuration Options

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spark` | (required) | SparkSession instance |
| `checkpoint_path` | `None` | Path to fine-tuned checkpoint |
| `base_model_id` | `"mistralai/Mistral-7B-Instruct-v0.3"` | Base model |
| `max_length` | `2048` | Max input tokens |
| `max_new_tokens` | `1024` | Max output tokens |
| `device` | `"cuda"` | Device ("cuda" or "cpu") |
| `load_in_4bit` | `True` | Enable 4-bit quantization |

### Method Parameters

**translate_descriptions()**:
- `taxa_df`: Input DataFrame
- `description_col`: Description column name (default: "description")
- `output_col`: Output column name (default: "features_json")

**translate_descriptions_batch()**:
- Same as above, plus:
- `batch_size`: Batch size (default: 10)

## Output Format

### DataFrame Schema

**Input** (from `load_taxa()`):
```
root
 |-- taxon: string
 |-- description: string
 |-- source: map
 |-- line_number: integer
 |-- paragraph_number: integer
 |-- page_number: integer
 |-- empirical_page_number: string
```

**Output** (after `translate_descriptions()`):
```
root
 |-- taxon: string
 |-- description: string
 |-- source: map
 |-- line_number: integer
 |-- paragraph_number: integer
 |-- page_number: integer
 |-- empirical_page_number: string
 |-- features_json: string  ← NEW
```

**Output** (after `validate_json()`):
```
... (all above columns)
 |-- json_valid: boolean  ← NEW
```

### JSON Structure

```json
{
  "pileus": {
    "diameter": ["5-10 cm"],
    "shape": ["convex", "plane"],
    "color": ["white", "brown"]
  },
  "stipe": {
    "length": ["8-12 cm"],
    "thickness": ["1-2 cm"]
  },
  "spores": {
    "dimensions": ["7-9 x 5-6 μm"],
    "shape": ["ellipsoid"]
  }
}
```

## Troubleshooting

### Error: CUDA out of memory

**Solution 1**: Enable 4-bit quantization (default)
```python
translator = TaxaJSONTranslator(spark=spark, load_in_4bit=True)
```

**Solution 2**: Use smaller batches
```python
enriched_df = translator.translate_descriptions_batch(taxa_df, batch_size=5)
```

**Solution 3**: Use CPU
```python
translator = TaxaJSONTranslator(spark=spark, device="cpu", load_in_4bit=False)
```

### Error: Checkpoint not found

```python
from pathlib import Path

checkpoint = "./checkpoints/checkpoint-100"
if not Path(checkpoint).exists():
    checkpoint = None  # Use base model

translator = TaxaJSONTranslator(spark=spark, checkpoint_path=checkpoint)
```

### Issue: Invalid JSON outputs

```python
# Validate and filter
validated_df = translator.validate_json(enriched_df)
valid_df = validated_df.filter("json_valid = true")

# Show invalid entries
invalid_df = validated_df.filter("json_valid = false")
invalid_df.select("taxon", "features_json").show()
```

## Performance Tips

### 1. Cache DataFrame

```python
taxa_df.cache()
enriched_df = translator.translate_descriptions(taxa_df)
taxa_df.unpersist()
```

### 2. Repartition

```python
taxa_df = taxa_df.repartition(10)
enriched_df = translator.translate_descriptions(taxa_df)
```

### 3. Process in Chunks

```python
patterns = ["taxon_a*", "taxon_b*", "taxon_c*"]
for pattern in patterns:
    chunk = extractor.load_taxa(pattern=pattern)
    enriched = translator.translate_descriptions(chunk)
    translator.save_translations(enriched, f"output/{pattern}.parquet")
```

### 4. Use Batch Mode

```python
# Faster for datasets > 100 rows
enriched_df = translator.translate_descriptions_batch(
    taxa_df,
    batch_size=20
)
```

## Complete Example

```python
#!/usr/bin/env python3
"""Complete taxa translation example."""

from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import TaxonExtractor
from taxa_json_translator import TaxaJSONTranslator

# Initialize
spark = SparkSession.builder \
    .appName("Taxa Translation") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Load taxa
extractor = TaxonExtractor(
    spark=spark,
    ingest_couchdb_url="http://localhost:5984",
    ingest_db_name="mycobank_annotations",
    taxon_db_name="mycobank_taxa",
    ingest_username="admin",
    ingest_password="password"
)

taxa_df = extractor.load_taxa()
print(f"Loaded {taxa_df.count()} taxa")

# Translate
translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./mistral_checkpoints/checkpoint-100"
)

enriched_df = translator.translate_descriptions_batch(
    taxa_df,
    batch_size=20
)

# Validate
validated_df = translator.validate_json(enriched_df)
valid_df = validated_df.filter("json_valid = true")

# Save
translator.save_translations(
    valid_df,
    "output/taxa_with_features.parquet"
)

print("✓ Complete!")
spark.stop()
```

## Next Steps

- Read [TAXA_JSON_TRANSLATOR.md](TAXA_JSON_TRANSLATOR.md) for detailed documentation
- See [example_taxa_translation.py](example_taxa_translation.py) for full example
- Check [TAXA_ROUNDTRIP_EXAMPLE.md](TAXA_ROUNDTRIP_EXAMPLE.md) for pipeline integration

## Environment Variables

```bash
# CouchDB configuration
export COUCHDB_URL="http://localhost:5984"
export COUCHDB_USER="admin"
export COUCHDB_PASSWORD="password"
export INGEST_DB="mycobank_annotations"
export TAXON_DB="mycobank_taxa"

# Mistral configuration
export MISTRAL_CHECKPOINT="./mistral_checkpoints/checkpoint-100"
export HUGGING_FACE_TOKEN="your_token_here"
```

## API Reference

### TaxaJSONTranslator

**Methods**:
- `translate_descriptions(taxa_df)` → DataFrame with JSON column
- `translate_descriptions_batch(taxa_df, batch_size)` → DataFrame with JSON column
- `translate_single(description)` → Dict[str, Any]
- `validate_json(translated_df)` → DataFrame with validation column
- `save_translations(df, path, format)` → None

**Properties**:
- `model` - Lazy-loaded Mistral model
- `tokenizer` - Lazy-loaded tokenizer

## FAQ

**Q: Do I need a GPU?**
A: No, but it's much faster. Use `device="cpu"` for CPU-only.

**Q: How much memory do I need?**
A: With 4-bit quantization: ~8GB GPU memory. Without: ~28GB.

**Q: Can I use the base model without fine-tuning?**
A: Yes, set `checkpoint_path=None`. Results will be less structured.

**Q: How do I fine-tune the model?**
A: See `mistral_transfer_learning.py` for training utilities.

**Q: What if JSON parsing fails?**
A: Returns empty dict `{}`. Use `validate_json()` to identify failures.

**Q: Can I customize the prompt?**
A: Yes, pass custom `prompt` parameter to constructor.
