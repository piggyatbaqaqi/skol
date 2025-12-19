# Strings Output: Getting Annotated Text as List

## Overview

When using `SkolClassifierV2`, you can now get predictions as a **list of annotated strings** by setting `output_dest='strings'`. This is useful when you want to:

- Process predictions programmatically without writing to disk
- Display predictions in a UI or web service
- Pass predictions to another pipeline component
- Quickly inspect results in a notebook

## Basic Usage

```python
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.appName("Strings Output").getOrCreate()

# Create classifier with strings output
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/raw/*.txt'],
    output_dest='strings',  # Return as list of strings
    model_type='rnn',
    auto_load_model=True,
    model_storage='redis',
    redis_client=redis_client,
    redis_key='my_model',
    verbosity=1
)

# Make predictions
predictions_df = classifier.predict()

# Get annotated strings (returns List[str])
annotated_strings = classifier.save_annotated(predictions_df)

# Use the strings
for i, doc in enumerate(annotated_strings):
    print(f"\n=== Document {i+1} ===")
    print(doc)
```

## Output Format

Each string in the returned list represents one complete document with all annotations in **YEDDA format**:

```
[@ First line of text #Description*]
[@ Second line of text #Description*]
[@ Species name here #Nomenclature*]
[@ More description text #Description*]
```

### Format Details

- **One string per document**: If you have 10 input documents, you'll get a list of 10 strings
- **Line-separated annotations**: Within each document string, annotations are separated by newlines
- **YEDDA format**: `[@ text #Label*]` - standard annotation format
- **Ordered by line number**: If predictions have `line_number`, they're sorted correctly

## Return Value

```python
annotated_strings = classifier.save_annotated(predictions_df)
```

**Return type**: `Optional[List[str]]`
- Returns `List[str]` when `output_dest='strings'`
- Returns `None` when `output_dest='files'` or `output_dest='couchdb'`

## With Label Coalescing

You can enable label coalescing to merge consecutive lines with the same label:

```python
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/raw/*.txt'],
    output_dest='strings',
    model_type='rnn',
    line_level=True,
    coalesce_labels=True,  # Merge consecutive same-label lines
    verbosity=1
)

predictions_df = classifier.predict()
annotated_strings = classifier.save_annotated(predictions_df)

# Output will have merged blocks
# Instead of:
#   [@ Line 1 #Description*]
#   [@ Line 2 #Description*]
# You'll get:
#   [@ Line 1
#   Line 2 #Description*]
```

## Complete Example: Web Service

```python
import redis
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

app = Flask(__name__)

# Initialize once
spark = SparkSession.builder.appName("API Server").getOrCreate()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict labels for text documents."""
    # Get text from request
    texts = request.json.get('texts', [])

    # Create DataFrame from strings
    from pyspark.sql.types import StructType, StructField, StringType
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("value", StringType(), False),
    ])
    data = [(f"doc_{i}", text) for i, text in enumerate(texts)]
    raw_df = spark.createDataFrame(data, schema)

    # Create classifier with strings output
    classifier = SkolClassifierV2(
        spark=spark,
        input_source='files',  # Source doesn't matter, we pass df to predict()
        file_paths=['dummy'],   # Required but not used
        output_dest='strings',  # Return as strings
        model_type='rnn',
        auto_load_model=True,
        model_storage='redis',
        redis_client=redis_client,
        redis_key='taxonomy_model',
        line_level=True,
        verbosity=0
    )

    # Make predictions
    predictions_df = classifier.predict(raw_df)

    # Get annotated strings
    annotated_texts = classifier.save_annotated(predictions_df)

    # Return as JSON
    return jsonify({
        'predictions': annotated_texts,
        'count': len(annotated_texts)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Request example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This is a taxonomic description of a new species.",
      "Genus species Author, 2024"
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    "[@ This is a taxonomic description of a new species. #Description*]",
    "[@ Genus species Author, 2024 #Nomenclature*]"
  ],
  "count": 2
}
```

## Complete Example: Batch Processing

```python
import redis
from pyspark.sql import SparkSession
from skol_classifier.classifier_v2 import SkolClassifierV2

spark = SparkSession.builder.appName("Batch Processing").getOrCreate()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

# Load model once
classifier = SkolClassifierV2(
    spark=spark,
    input_source='files',
    file_paths=['data/input/*.txt'],
    output_dest='strings',
    model_type='rnn',
    auto_load_model=True,
    model_storage='redis',
    redis_client=redis_client,
    redis_key='taxonomy_model',
    line_level=True,
    verbosity=1
)

# Process multiple batches
input_batches = [
    'data/batch1/*.txt',
    'data/batch2/*.txt',
    'data/batch3/*.txt'
]

all_results = []

for batch in input_batches:
    print(f"\nProcessing {batch}...")

    # Update file paths
    classifier.file_paths = [batch]

    # Predict
    predictions_df = classifier.predict()

    # Get annotated strings
    annotated = classifier.save_annotated(predictions_df)

    all_results.extend(annotated)
    print(f"  Processed {len(annotated)} documents")

# Save all results
with open('all_predictions.txt', 'w') as f:
    for i, result in enumerate(all_results):
        f.write(f"\n{'='*70}\n")
        f.write(f"Document {i+1}\n")
        f.write(f"{'='*70}\n")
        f.write(result)
        f.write("\n")

print(f"\n✓ Total processed: {len(all_results)} documents")
print(f"✓ Saved to all_predictions.txt")
```

## Comparison: output_dest Options

| output_dest | Returns | Use Case |
|-------------|---------|----------|
| `'strings'` | `List[str]` | Programmatic access, APIs, notebooks |
| `'files'` | `None` | Save to disk for archival |
| `'couchdb'` | `None` | Save to database for storage |

## Common Use Cases

### 1. Interactive Notebook Analysis

```python
# Predict and immediately inspect
predictions = classifier.predict()
results = classifier.save_annotated(predictions)

# Quick inspection
for doc in results[:3]:  # First 3 documents
    print(doc)
    print("-" * 50)
```

### 2. Extract Specific Labels

```python
# Get predictions as strings
predictions_df = classifier.predict()
annotated = classifier.save_annotated(predictions_df)

# Extract only Nomenclature annotations
nomenclature_lines = []
for doc in annotated:
    for line in doc.split('\n'):
        if '#Nomenclature*]' in line:
            # Extract text between [@ and #
            text = line.split('[@ ')[1].split(' #')[0]
            nomenclature_lines.append(text)

print(f"Found {len(nomenclature_lines)} Nomenclature annotations:")
for name in nomenclature_lines[:10]:
    print(f"  - {name}")
```

### 3. Convert to Different Format

```python
import json

# Get predictions
predictions_df = classifier.predict()
annotated = classifier.save_annotated(predictions_df)

# Convert to JSON
output = []
for i, doc in enumerate(annotated):
    annotations = []
    for line in doc.split('\n'):
        if line.startswith('[@ '):
            # Parse YEDDA format
            text = line.split('[@ ')[1].split(' #')[0]
            label = line.split('#')[1].split('*]')[0]
            annotations.append({'text': text, 'label': label})

    output.append({
        'document_id': i,
        'annotations': annotations
    })

# Save as JSON
with open('predictions.json', 'w') as f:
    json.dump(output, f, indent=2)
```

## Performance Notes

- **Memory**: All results are collected to the driver, so be mindful with very large datasets
- **For large datasets**: Consider using `output_dest='files'` and processing in batches
- **Recommended limit**: ~10,000 documents per call to `save_annotated()`

## Troubleshooting

### Problem: Empty list returned

**Cause**: No predictions in DataFrame

**Solution**: Check that `predict()` returned results
```python
predictions_df = classifier.predict()
print(f"Predictions count: {predictions_df.count()}")
results = classifier.save_annotated(predictions_df)
```

### Problem: Memory error with large datasets

**Cause**: Collecting too many results to driver

**Solution**: Process in smaller batches or use `output_dest='files'`
```python
# Instead of processing all at once, batch it
for batch_path in batch_paths:
    classifier.file_paths = [batch_path]
    predictions = classifier.predict()
    results = classifier.save_annotated(predictions)
    # Process results immediately
    process_results(results)
```

### Problem: Annotations out of order

**Cause**: Missing `line_number` in predictions

**Solution**: Ensure data has `line_number` column
```python
# Check if line_number exists
print(predictions_df.columns)
# Should include 'line_number'
```

## See Also

- [SkolClassifierV2 Documentation](../README.md) - Full API reference
- [Output Formatters](../skol_classifier/output_formatters.py) - YEDDA formatting details
- [Complete Examples](../examples/) - More usage examples
