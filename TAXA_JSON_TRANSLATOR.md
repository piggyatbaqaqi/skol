# TaxaJSONTranslator Class

## Overview

The `TaxaJSONTranslator` class provides an optimized interface for translating taxa descriptions into structured JSON using a fine-tuned Mistral model. It's designed specifically to work with PySpark DataFrames created by `TaxonExtractor.load_taxa()`.

## Key Features

- ✅ **DataFrame-optimized**: Works seamlessly with PySpark DataFrames
- ✅ **Lazy loading**: Model and tokenizer loaded only when needed
- ✅ **Batch processing**: Efficient processing of multiple descriptions
- ✅ **4-bit quantization**: Memory-efficient model loading
- ✅ **JSON validation**: Built-in validation of generated JSON
- ✅ **Flexible output**: Save to Parquet, JSON, or CSV
- ✅ **Multi-GPU support**: Automatic parallelization across GPUs

## Architecture

### Design Principles

1. **Encapsulation**: All Mistral functionality in single class
2. **Lazy Loading**: Models loaded only when inference is performed
3. **PySpark Integration**: Native DataFrame operations
4. **Error Handling**: Graceful degradation on JSON parsing errors

### Workflow

```
Taxa DataFrame → TaxaJSONTranslator → Enriched DataFrame
                      ↓
                  Mistral Model
                      ↓
                 JSON Features
```

## Class Interface

### Constructor

```python
translator = TaxaJSONTranslator(
    spark: SparkSession,
    checkpoint_path: Optional[str] = None,
    base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_length: int = 2048,
    max_new_tokens: int = 1024,
    prompt: str = DEFAULT_PROMPT,
    device: str = "cuda",
    load_in_4bit: bool = True,
    use_auth_token: bool = True
)
```

**Parameters**:
- `spark`: SparkSession instance (required)
- `checkpoint_path`: Path to fine-tuned checkpoint (optional)
- `base_model_id`: Hugging Face model ID
- `max_length`: Maximum input sequence length
- `max_new_tokens`: Maximum tokens to generate
- `prompt`: Instruction prompt for model
- `device`: "cuda" or "cpu"
- `load_in_4bit`: Enable 4-bit quantization
- `use_auth_token`: Use Hugging Face authentication

### Main Methods

#### 1. translate_descriptions()

```python
enriched_df = translator.translate_descriptions(
    taxa_df: DataFrame,
    description_col: str = "description",
    output_col: str = "features_json"
) -> DataFrame
```

**Purpose**: Add JSON column to DataFrame using UDF

**Use when**: Small to medium datasets (< 10,000 rows)

**Example**:
```python
taxa_df = extractor.load_taxa()
enriched_df = translator.translate_descriptions(taxa_df)
enriched_df.select("taxon", "features_json").show()
```

#### 2. translate_descriptions_batch()

```python
enriched_df = translator.translate_descriptions_batch(
    taxa_df: DataFrame,
    description_col: str = "description",
    output_col: str = "features_json",
    batch_size: int = 10
) -> DataFrame
```

**Purpose**: Batch processing for efficiency

**Use when**: Medium datasets or when you want progress tracking

**Example**:
```python
enriched_df = translator.translate_descriptions_batch(
    taxa_df,
    batch_size=20
)
```

#### 3. translate_single()

```python
features = translator.translate_single(
    description: str
) -> Dict[str, Any]
```

**Purpose**: Translate single description (returns dict)

**Example**:
```python
description = "Pileus 5-10 cm broad, convex, white to brown..."
features = translator.translate_single(description)
print(features.keys())
```

#### 4. validate_json()

```python
validated_df = translator.validate_json(
    translated_df: DataFrame,
    json_col: str = "features_json"
) -> DataFrame
```

**Purpose**: Add validation column and print statistics

**Example**:
```python
validated_df = translator.validate_json(enriched_df)
validated_df.filter("json_valid = false").show()
```

#### 5. save_translations()

```python
translator.save_translations(
    translated_df: DataFrame,
    output_path: str,
    format: str = "parquet"
)
```

**Purpose**: Save results to disk

**Formats**: "parquet", "json", "csv"

**Example**:
```python
translator.save_translations(
    enriched_df,
    "output/taxa_features.parquet"
)
```

## Usage Examples

### Example 1: Basic Usage

```python
from pyspark.sql import SparkSession
from extract_taxa_to_couchdb import TaxonExtractor
from taxa_json_translator import TaxaJSONTranslator

# Initialize Spark
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

# Initialize translator
translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./mistral_checkpoints/checkpoint-100"
)

# Translate descriptions
enriched_df = translator.translate_descriptions(taxa_df)

# Show results
enriched_df.select("taxon", "description", "features_json").show(5, truncate=50)

# Save
translator.save_translations(enriched_df, "output/taxa_features.parquet")

spark.stop()
```

### Example 2: With Validation

```python
# Translate
enriched_df = translator.translate_descriptions(taxa_df)

# Validate
validated_df = translator.validate_json(enriched_df)

# Show invalid entries
print("\nInvalid JSON entries:")
validated_df.filter("json_valid = false").select(
    "taxon", "description", "features_json"
).show(truncate=50)

# Save only valid ones
valid_df = validated_df.filter("json_valid = true")
translator.save_translations(valid_df, "output/valid_taxa_features.parquet")
```

### Example 3: Batch Processing with Progress

```python
# Process in batches of 20
enriched_df = translator.translate_descriptions_batch(
    taxa_df,
    batch_size=20
)

# Each batch prints progress:
# Batch 1/50
# Batch 2/50
# ...
```

### Example 4: Custom Prompt

```python
custom_prompt = '''Extract morphological features from this fungal description.
Output JSON with these top-level keys: pileus, stipe, lamellae, spores.
Each key should contain a dictionary of attributes and their values.
'''

translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./checkpoints/fungi-specific-100",
    prompt=custom_prompt
)

enriched_df = translator.translate_descriptions(taxa_df)
```

### Example 5: Single Description Translation

```python
description = """
Pileus 5-10 cm broad, convex to plane, viscid when moist,
white to cream colored. Lamellae free, crowded, white.
Stipe 6-12 cm long, 1-2 cm thick, white, hollow.
Spores 7-9 x 5-6 μm, ellipsoid, smooth.
"""

features = translator.translate_single(description)

print(json.dumps(features, indent=2))
```

**Output**:
```json
{
  "pileus": {
    "diameter": ["5-10 cm"],
    "shape": ["convex", "plane"],
    "surface": ["viscid when moist"],
    "color": ["white", "cream"]
  },
  "lamellae": {
    "attachment": ["free"],
    "spacing": ["crowded"],
    "color": ["white"]
  },
  "stipe": {
    "length": ["6-12 cm"],
    "thickness": ["1-2 cm"],
    "color": ["white"],
    "structure": ["hollow"]
  },
  "spores": {
    "dimensions": ["7-9 x 5-6 μm"],
    "shape": ["ellipsoid"],
    "surface": ["smooth"]
  }
}
```

### Example 6: Using Base Model (No Checkpoint)

```python
# Use base model without fine-tuning
translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path=None  # No checkpoint = base model
)

enriched_df = translator.translate_descriptions(taxa_df)
```

### Example 7: CPU-Only Inference

```python
# For machines without GPU
translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./checkpoints/checkpoint-100",
    device="cpu",
    load_in_4bit=False  # 4-bit only works on GPU
)

enriched_df = translator.translate_descriptions(taxa_df)
```

## Integration with Full Pipeline

### Complete Workflow: Annotation → Extraction → Translation

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2
from extract_taxa_to_couchdb import TaxonExtractor
from taxa_json_translator import TaxaJSONTranslator

spark = SparkSession.builder \
    .appName("Complete Pipeline") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Step 1: Classify raw documents
print("Step 1: Classifying documents...")
classifier = SkolClassifierV2(
    spark=spark,
    input_source='couchdb',
    couchdb_url="http://localhost:5984",
    db_name="mycobank_raw",
    line_level=False,
    model_type='logistic',
    output_format='annotated',
    coalesce_labels=True
)

results = classifier.fit()
predictions = classifier.predict()
classifier.save_annotated(predictions)

# Step 2: Extract taxa
print("\nStep 2: Extracting taxa...")
extractor = TaxonExtractor(
    spark=spark,
    ingest_couchdb_url="http://localhost:5984",
    ingest_db_name="mycobank_raw",
    taxon_db_name="mycobank_taxa",
    ingest_username="admin",
    ingest_password="password"
)

annotated_df = extractor.load_annotated_documents()
extracted_df = extractor.extract_taxa(annotated_df)
save_results = extractor.save_taxa(extracted_df)

print(f"Saved {save_results.filter('success = true').count()} taxa")

# Step 3: Load and translate
print("\nStep 3: Translating descriptions to JSON...")
taxa_df = extractor.load_taxa()

translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./mistral_checkpoints/checkpoint-100"
)

enriched_df = translator.translate_descriptions_batch(
    taxa_df,
    batch_size=20
)

# Step 4: Validate and save
print("\nStep 4: Validating and saving...")
validated_df = translator.validate_json(enriched_df)
valid_df = validated_df.filter("json_valid = true")

translator.save_translations(
    valid_df,
    "output/complete_taxa_features.parquet"
)

print("\n✓ Pipeline complete!")
spark.stop()
```

## Performance Considerations

### Memory Management

**4-bit Quantization**:
- Reduces model memory from ~28GB to ~7GB
- Minimal accuracy loss
- Enabled by default

**Batch Size**:
- Larger batches = more memory usage
- Smaller batches = more overhead
- Recommended: 10-20 for most cases

### Processing Speed

**GPU vs CPU**:
- GPU: ~1-2 seconds per description
- CPU: ~10-30 seconds per description

**Batch Mode**:
- Slightly faster than UDF mode
- Better progress tracking
- Recommended for datasets > 100 rows

### Optimization Tips

```python
# 1. Cache taxa DataFrame if reusing
taxa_df.cache()

# 2. Repartition for better parallelism
taxa_df = taxa_df.repartition(10)

# 3. Use batch mode for large datasets
enriched_df = translator.translate_descriptions_batch(
    taxa_df,
    batch_size=50  # Adjust based on GPU memory
)

# 4. Process in chunks for very large datasets
pattern_prefixes = ["taxon_a*", "taxon_b*", "taxon_c*"]
for pattern in pattern_prefixes:
    chunk_df = extractor.load_taxa(pattern=pattern)
    enriched_chunk = translator.translate_descriptions(chunk_df)
    translator.save_translations(
        enriched_chunk,
        f"output/taxa_{pattern}.parquet"
    )
```

## Error Handling

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Option 1: Enable 4-bit quantization
translator = TaxaJSONTranslator(
    spark=spark,
    load_in_4bit=True
)

# Option 2: Reduce batch size
enriched_df = translator.translate_descriptions_batch(
    taxa_df,
    batch_size=5  # Smaller batches
)

# Option 3: Use CPU
translator = TaxaJSONTranslator(
    spark=spark,
    device="cpu",
    load_in_4bit=False
)
```

#### 2. Invalid JSON Output

**Issue**: Model generates malformed JSON

**Detection**:
```python
validated_df = translator.validate_json(enriched_df)
invalid_df = validated_df.filter("json_valid = false")

print(f"Invalid JSON count: {invalid_df.count()}")
invalid_df.select("taxon", "description", "features_json").show(truncate=50)
```

**Solutions**:
- Use fine-tuned checkpoint (better JSON formatting)
- Adjust prompt to emphasize JSON structure
- Post-process with custom validation

#### 3. Checkpoint Not Found

**Error**: `OSError: ./checkpoints/checkpoint-100 does not exist`

**Solution**:
```python
# Check if checkpoint exists
checkpoint_path = "./checkpoints/checkpoint-100"
if not Path(checkpoint_path).exists():
    print(f"Checkpoint not found, using base model")
    checkpoint_path = None

translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path=checkpoint_path
)
```

## Comparison: TaxaJSONTranslator vs Direct Usage

### Direct Usage (from mistral_transfer_learning.py)

```python
# Load model
base_model = load_base_model(BASE_MODEL_ID)
tokenizer = create_tokenizer(BASE_MODEL_ID)
model = load_finetuned_model(base_model, checkpoint_path, tokenizer)

# Process each description
for row in descriptions:
    prompt = make_prompt(DEFAULT_PROMPT, row['description'])
    output = generate_response(model, tokenizer, prompt)
    json_obj = extract_json(output)
    # ... manual DataFrame construction
```

### Using TaxaJSONTranslator

```python
# Initialize once
translator = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path=checkpoint_path
)

# Process entire DataFrame
enriched_df = translator.translate_descriptions(taxa_df)
```

**Benefits**:
- ✅ **Simpler**: Single line instead of loop
- ✅ **Integrated**: Native PySpark DataFrame operations
- ✅ **Lazy**: Model loaded only when needed
- ✅ **Validated**: Built-in JSON validation
- ✅ **Robust**: Automatic error handling

## Advanced Usage

### Custom JSON Extraction

```python
class CustomTaxaTranslator(TaxaJSONTranslator):
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Custom JSON extraction logic."""
        # Your custom parsing logic
        # ...
        return parsed_json

translator = CustomTaxaTranslator(
    spark=spark,
    checkpoint_path="./checkpoints/checkpoint-100"
)
```

### Progress Tracking

```python
from pyspark.sql.functions import monotonically_increasing_id

# Add ID column for tracking
taxa_df = taxa_df.withColumn("id", monotonically_increasing_id())
total = taxa_df.count()

# Process with progress
enriched_df = translator.translate_descriptions(taxa_df)

# Show progress during save
enriched_df.write \
    .mode("overwrite") \
    .format("parquet") \
    .option("compression", "snappy") \
    .save("output/taxa_features.parquet")
```

### Multi-Model Ensemble

```python
# Use multiple checkpoints
translator1 = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./checkpoints/checkpoint-50"
)

translator2 = TaxaJSONTranslator(
    spark=spark,
    checkpoint_path="./checkpoints/checkpoint-100"
)

# Translate with both
enriched1 = translator1.translate_descriptions(taxa_df)
enriched2 = translator2.translate_descriptions(taxa_df)

# Compare results
# ...
```

## Testing

### Unit Test Example

```python
def test_taxa_translator():
    """Test TaxaJSONTranslator functionality."""
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    try:
        # Create test data
        test_data = [
            {
                "taxon": "Test species",
                "description": "Pileus 5 cm, white. Stipe 10 cm.",
                "source": {"db_name": "test", "doc_id": "test1"}
            }
        ]
        taxa_df = spark.createDataFrame(test_data)

        # Initialize translator
        translator = TaxaJSONTranslator(
            spark=spark,
            checkpoint_path=None  # Use base model for testing
        )

        # Translate
        enriched_df = translator.translate_descriptions(taxa_df)

        # Verify
        assert "features_json" in enriched_df.columns
        result = enriched_df.first()
        assert result['features_json'] is not None

        # Validate
        validated_df = translator.validate_json(enriched_df)
        assert "json_valid" in validated_df.columns

        print("✓ Test passed")

    finally:
        spark.stop()
```

## Related Documentation

- [TAXON_LOAD_METHOD.md](TAXON_LOAD_METHOD.md) - Loading taxa from CouchDB
- [TAXA_ROUNDTRIP_EXAMPLE.md](TAXA_ROUNDTRIP_EXAMPLE.md) - Complete extraction workflow
- [mistral_transfer_learning.py](mistral_transfer_learning.py) - Underlying Mistral utilities

## Summary

The `TaxaJSONTranslator` class provides:

- ✅ **Encapsulated Interface**: All Mistral functionality in one class
- ✅ **PySpark Integration**: Native DataFrame operations
- ✅ **Lazy Loading**: Efficient resource management
- ✅ **Batch Processing**: Scalable to large datasets
- ✅ **Validation**: Built-in JSON validation
- ✅ **Flexibility**: Multiple output formats and processing modes
- ✅ **Error Handling**: Graceful degradation
- ✅ **Documentation**: Comprehensive examples and guides

This class is optimized for translating taxa descriptions to structured JSON in a production PySpark environment.
