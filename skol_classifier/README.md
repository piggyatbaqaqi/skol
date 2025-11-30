# SKOL Text Classifier

A PySpark-based text classification pipeline for taxonomic literature, designed to automatically classify paragraphs into categories: Nomenclature, Description, and Miscellaneous exposition.

Created by: Christopher Murphy, La Monte Yarroll, David Caspers

## Features

- **Multiple Classification Models**: Logistic Regression and Random Forest
- **Advanced Feature Engineering**: TF-IDF with optional word suffix features (2-4 characters)
- **Automated Paragraph Detection**: Heuristic-based paragraph extraction from raw text
- **Line-by-Line Classification**: Optional line-level classification with YEDDA format output
- **Scalable Processing**: Built on Apache Spark for handling large document collections
- **Model Persistence**: Save and load models to/from Redis or disk
- **CouchDB Integration**: Read from and write to CouchDB attachments
- **Easy-to-use API**: Simple interface for training and prediction

## Installation

### Prerequisites

- Python 3.7+
- Java 8 or 11 (required for Spark)

### Install Dependencies

```bash
pip install pyspark sparknlp regex
```

### Install the Module

From the repository root:

```bash
cd skol
pip install -e .
```

Or add the `skol_classifier` directory to your Python path.

## Quick Start

### Basic Usage

```python
from skol_classifier import SkolClassifier, get_file_list

# Initialize classifier
classifier = SkolClassifier()

# Get annotated training files
annotated_files = get_file_list(
    "/path/to/annotated/data",
    pattern="**/*.txt.ann"
)

# Train the model
results = classifier.fit(
    annotated_file_paths=annotated_files,
    model_type="logistic",
    use_suffixes=True,
    test_size=0.2
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score: {results['f1_score']:.4f}")

# Predict on raw text
raw_files = get_file_list("/path/to/raw/data", pattern="**/*.txt")
predictions = classifier.predict_raw_text(raw_files)

# Save annotated output
classifier.save_annotated_output(predictions, "/path/to/output")
```

## API Reference

### SkolClassifier

Main classifier class for training and prediction.

#### Methods

**`__init__(spark=None, redis_client=None, redis_key='skol_classifier_model', auto_load=True)`**
- Initialize the classifier
- `spark`: Optional SparkSession (creates one if not provided)
- `redis_client`: Optional Redis client connection for model persistence
- `redis_key`: Key name to use in Redis for storing the model
- `auto_load`: If True, automatically loads model from Redis if key exists (default: True)

**`fit(annotated_file_paths, model_type='logistic', use_suffixes=True, test_size=0.2, **model_params)`**
- Complete training pipeline
- Returns: Dictionary with evaluation metrics

**`load_annotated_data(file_paths, collapse_labels=True)`**
- Load and preprocess annotated training data
- Returns: DataFrame with paragraphs and labels

**`load_raw_data(file_paths)`**
- Load and preprocess raw text files
- Returns: DataFrame with extracted paragraphs

**`fit_features(data, use_suffixes=True, min_doc_freq=10)`**
- Fit feature extraction pipeline
- Returns: Transformed DataFrame with features

**`train_classifier(train_data, model_type='logistic', features_col='combined_idf', **model_params)`**
- Train classification model
- `model_type`: 'logistic' or 'random_forest'
- Returns: Fitted model

**`predict(data)`**
- Make predictions on feature DataFrame
- Returns: DataFrame with predictions

**`predict_raw_text(file_paths, output_format='annotated')`**
- Process and predict labels for raw text
- Returns: DataFrame with predictions

**`save_annotated_output(predictions, output_path)`**
- Save annotated predictions to disk

**`evaluate(predictions, verbose=True)`**
- Evaluate model performance
- Returns: Dictionary with metrics (accuracy, precision, recall, F1)

**`save_to_redis(redis_client=None, redis_key=None)`**
- Save trained models to Redis
- Returns: True if successful, False otherwise

**`load_from_redis(redis_client=None, redis_key=None)`**
- Load trained models from Redis
- Returns: True if successful, False otherwise

**`save_to_disk(path)`**
- Save trained models to disk
- `path`: Directory path to save the models

**`load_from_disk(path)`**
- Load trained models from disk
- `path`: Directory path containing the saved models

### Utility Functions

**`get_file_list(folder, pattern='**/*.txt*', exclude_pattern='Sydowia')`**
- List files matching pattern in a folder
- Returns: List of file paths

**`create_evaluators()`**
- Create evaluation metrics for classification
- Returns: Dictionary of evaluators

**`calculate_stats(predictions, evaluators=None, verbose=True)`**
- Calculate evaluation statistics
- Returns: Dictionary with metrics

### Preprocessing

**`SuffixTransformer(inputCol='words', outputCol='suffixes')`**
- Custom transformer for extracting word suffixes

**`ParagraphExtractor`**
- Static methods for paragraph extraction
  - `extract_annotated_paragraphs(lines)`: Extract annotated paragraphs
  - `extract_heuristic_paragraphs(lines)`: Heuristic paragraph detection
  - `collapse_labels(label)`: Collapse labels to 3 categories

## Examples

### Example 1: Train with Different Models

```python
from skol_classifier import SkolClassifier, get_file_list

classifier = SkolClassifier()
files = get_file_list("/data/annotated", pattern="**/*.ann")

# Logistic Regression
results_lr = classifier.fit(
    files, model_type="logistic", use_suffixes=True
)

# Random Forest
classifier_rf = SkolClassifier()
results_rf = classifier_rf.fit(
    files, model_type="random_forest", numTrees=100
)
```

### Example 2: Manual Pipeline Control

```python
from skol_classifier import SkolClassifier

classifier = SkolClassifier()

# Load data
annotated_df = classifier.load_annotated_data(file_paths)

# Extract features
features = classifier.fit_features(annotated_df, use_suffixes=True)

# Split data
train, test = features.randomSplit([0.8, 0.2], seed=42)

# Train
classifier.train_classifier(
    train, model_type="logistic", features_col="combined_idf"
)

# Predict
predictions = classifier.predict(test)

# Evaluate
stats = classifier.evaluate(predictions)
```

### Example 3: Save and Load Models with Redis

```python
import redis
from skol_classifier import SkolClassifier, get_file_list

# Connect to Redis
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False  # Important for binary data
)

# Train and save to Redis
classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="skol_model_v1"
)

annotated_files = get_file_list("/data/annotated")
classifier.fit(annotated_files)

# Save to Redis
classifier.save_to_redis()
print("Model saved to Redis!")

# Later, load from Redis (auto-loads if key exists)
new_classifier = SkolClassifier(
    redis_client=redis_client,
    redis_key="skol_model_v1"
)
# Model is automatically loaded! Check if loaded:
if new_classifier.labels is not None:
    print(f"Model loaded with labels: {new_classifier.labels}")

# Use loaded model
raw_files = get_file_list("/data/raw")
predictions = new_classifier.predict_raw_text(raw_files)
```

### Example 4: Save and Load Models from Disk

```python
from skol_classifier import SkolClassifier, get_file_list

# Train model
classifier = SkolClassifier()
annotated_files = get_file_list("/data/annotated")
classifier.fit(annotated_files)

# Save to disk
classifier.save_to_disk("/models/skol_classifier")

# Later, load from disk
new_classifier = SkolClassifier()
new_classifier.load_from_disk("/models/skol_classifier")

# Use loaded model
raw_files = get_file_list("/data/raw")
predictions = new_classifier.predict_raw_text(raw_files)
```

### Example 5: Process Raw Documents

```python
from skol_classifier import SkolClassifier, get_file_list

# Assume model is already trained
classifier = SkolClassifier()
# ... train classifier ...

# Process new documents
raw_files = get_file_list("/data/raw", pattern="**/*.txt")
predictions = classifier.predict_raw_text(raw_files)

# View predictions
predictions.select("filename", "predicted_label", "value").show()

# Save annotated versions
classifier.save_annotated_output(predictions, "/output/annotated")
```

## Data Format

### Annotated Data Format

Annotated files should use the format:
```
[@ This is a nomenclature paragraph. #Nomenclature]
[@ This is a description paragraph. #Description]
[@ This is miscellaneous text. #Misc-exposition]
```

### Output Format

Predictions are saved in the same format:
```
[@ Predicted paragraph text. #predicted_label]
```

## Model Performance

Based on the original notebook results:

| Model | Features | Accuracy | Precision | Recall | F1 Score |
|-------|----------|----------|-----------|--------|----------|
| Logistic Regression | Words only | 0.9418 | 0.9603 | 0.9666 | 0.9414 |
| Logistic Regression | Suffixes only | 0.9396 | 0.9593 | 0.9649 | 0.9392 |
| Logistic Regression | Combined | 0.9421 | 0.9640 | 0.9630 | 0.9419 |
| Random Forest | Words only | 0.7878 | 0.7878 | 1.0000 | 0.6943 |

## Label Categories

The classifier recognizes three main categories:

1. **Nomenclature**: Taxonomic nomenclature and naming
2. **Description**: Morphological descriptions
3. **Misc-exposition**: All other content (discussions, materials/methods, etc.)

## Advanced Configuration

### Model Parameters

**Logistic Regression:**
```python
classifier.fit(
    files,
    model_type="logistic",
    maxIter=10,        # Maximum iterations
    regParam=0.01      # Regularization parameter
)
```

**Random Forest:**
```python
classifier.fit(
    files,
    model_type="random_forest",
    numTrees=100,      # Number of trees
    seed=42            # Random seed
)
```

### Feature Engineering

```python
# Words only
classifier.fit_features(data, use_suffixes=False)

# Words + suffixes (2-4 character endings)
classifier.fit_features(data, use_suffixes=True)

# Adjust minimum document frequency
classifier.fit_features(data, min_doc_freq=5)
```

## Troubleshooting

### Spark Memory Issues

If processing large datasets, increase Spark memory:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SKOL") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

classifier = SkolClassifier(spark=spark)
```

### Java Version Issues

Ensure Java 8 or 11 is installed and `JAVA_HOME` is set correctly.

## Line-by-Line Classification with YEDDA Output

In addition to paragraph-based classification, the classifier now supports line-by-line classification with YEDDA (Yet Another Entity Detection and Annotation) format output.

### Why Line-by-Line Classification?

- More granular control over text segmentation
- Better for documents where paragraph detection is unreliable
- Produces YEDDA-formatted output for use with the yedda_parser module
- Consecutive lines with the same label are automatically coalesced into blocks

### Usage

```python
from skol_classifier import SkolClassifier

# Initialize and load model
classifier = SkolClassifier()

# Read raw text content
with open('article.txt', 'r') as f:
    text_content = f.read()

# Classify lines (not paragraphs) - pass raw text strings
predictions = classifier.predict_lines([text_content])

# Save as YEDDA format (coalesces consecutive same-label lines)
classifier.save_yedda_output(predictions, 'output_dir')

# Or save to CouchDB with coalescence
results = classifier.save_to_couchdb(
    predictions,
    suffix='.ann',
    coalesce_labels=True  # Enable YEDDA block coalescence
)
```

### YEDDA Format Output

The output coalesces consecutive lines with the same label into YEDDA blocks:

```
[@ Glomus mosseae Nicolson & Gerdemann, 1963.
≡ Glomus mosseae (Nicolson & Gerdemann) C. Walker & A. Schüssler
#Nomenclature*]
[@ Key characters: Spores formed singly or in loose clusters.
Spore wall structure: mono- to multiple-layered.
#Description*]
[@ This species is commonly found in temperate regions.
It forms arbuscular mycorrhizal associations.
#Misc-exposition*]
```

### API Methods

#### `load_raw_data_lines(text_contents: List[str]) -> DataFrame`

Load raw text strings as individual lines (not paragraphs).

**Parameters:**
- `text_contents`: List of raw text strings

#### `predict_lines(text_contents: List[str], output_format: str = "yedda") -> DataFrame`

Predict labels for individual lines. Returns DataFrame with line-level predictions.

**Parameters:**
- `text_contents`: List of raw text strings
- `output_format`: 'yedda', 'annotated', or 'simple'

#### `save_yedda_output(predictions: DataFrame, output_path: str) -> None`

Save predictions in YEDDA format with automatic label coalescence.

**Parameters:**
- `predictions`: DataFrame from `predict_lines()`
- `output_path`: Directory to save output files

#### `save_to_couchdb(predictions: DataFrame, suffix: str = ".ann", coalesce_labels: bool = False) -> List[Dict[str, Any]]`

Save predictions to CouchDB with optional label coalescence.

**Parameters:**
- `predictions`: DataFrame with predictions
- `suffix`: Suffix to append to attachment names (default: ".ann")
- `coalesce_labels`: If True, coalesce consecutive lines with same label into YEDDA blocks

#### `coalesce_consecutive_labels(lines_data: List[Dict[str, Any]]) -> str`

Static method to coalesce consecutive lines with the same label into YEDDA blocks.

### Example

See `example_line_classification.py` for a complete example.

```bash
python skol_classifier/example_line_classification.py
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- New features include tests
- Documentation is updated

## License

[Add appropriate license information]

## Citation

If you use this classifier in your research, please cite:

```
Murphy, C., Yarroll, L.M., & Caspers, D. (2025).
SKOL II: Text Classification Pipeline for Taxonomic Literature.
```

## Contact

For questions or issues, please contact the authors or open an issue on GitHub.
