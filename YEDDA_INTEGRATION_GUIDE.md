# YEDDA Integration Guide

This guide shows how to use the YEDDA parser module with the SKOL classifier for end-to-end text processing.

## Overview

The YEDDA (Yet Another Entity Detection and Annotation) format provides a standard way to annotate text with labels. The integration between `yeda_parser` and `skol_classifier` enables:

1. **Classification**: Use `skol_classifier` to classify text line-by-line
2. **Annotation**: Output is automatically formatted as YEDDA blocks
3. **Parsing**: Use `yeda_parser` to load YEDDA into DataFrames
4. **Analysis**: Perform queries, statistics, and transformations

## Quick Start

### 1. Classify Text and Generate YEDDA

```python
from pyspark.sql import SparkSession
from skol_classifier import SkolClassifier

spark = SparkSession.builder.appName("Classify").getOrCreate()

# Initialize classifier with trained model
classifier = SkolClassifier(spark=spark)

# Read raw text content
with open('data/article1.txt', 'r') as f:
    text1 = f.read()
with open('data/article2.txt', 'r') as f:
    text2 = f.read()

# Classify raw text strings line-by-line
predictions = classifier.predict_lines([text1, text2])

# Save as YEDDA format (auto-coalesces consecutive same-label lines)
classifier.save_yeda_output(predictions, 'output/yeda_annotations')
```

### 2. Parse YEDDA into DataFrame

```python
from yeda_parser import yeda_file_to_spark_df

# Parse YEDDA file with metadata
df = yeda_file_to_spark_df(
    'output/yeda_annotations/article1.txt',
    spark,
    metadata={'classifier_version': '1.0', 'date': '2025-01-01'}
)

# View results
df.show()
# +-------------+-----------+-----------+--------------------+
# |        label|       line|line_number|            metadata|
# +-------------+-----------+-----------+--------------------+
# |Nomenclature|Line 1     |          0|{"classifier_ver...|
# |Nomenclature|Line 2     |          1|{"classifier_ver...|
# +-------------+-----------+-----------+--------------------+
```

### 3. Analyze Results

```python
from yeda_parser import get_label_statistics

# Get label distribution
stats = get_label_statistics(df)
stats.show()
# +---------------+-----+----------+
# |          label|count|percentage|
# +---------------+-----+----------+
# |Nomenclature   |  150|     45.00|
# |Description    |   80|     24.00|
# |Misc-exposition|  103|     31.00|
# +---------------+-----+----------+

# Filter by label
nomenclature = df.filter(df.label == "Nomenclature")

# Search within labels
species = nomenclature.filter(nomenclature.line.contains("mosseae"))
```

## Complete Pipeline Example

```python
from pyspark.sql import SparkSession
from skol_classifier import SkolClassifier
from yeda_parser import (
    yeda_file_to_spark_df,
    get_label_statistics,
    extract_metadata_field
)

# Initialize Spark
spark = SparkSession.builder \
    .appName("YEDDA Pipeline") \
    .master("local[*]") \
    .getOrCreate()

# Step 1: Classify
print("Step 1: Classifying text...")
classifier = SkolClassifier(spark=spark)

# Read text content
with open('input/article.txt', 'r') as f:
    text_content = f.read()

predictions = classifier.predict_lines([text_content])
classifier.save_yeda_output(predictions, 'output/yeda')

# Step 2: Parse
print("Step 2: Parsing YEDDA...")
df = yeda_file_to_spark_df(
    'output/yeda/article.txt',
    spark,
    metadata={
        'source': 'taxonomic_journal',
        'year': 2025,
        'classifier': 'skol_v1'
    }
)

# Step 3: Analyze
print("Step 3: Analyzing results...")
stats = get_label_statistics(df)
stats.show()

# Extract metadata
df_with_source = extract_metadata_field(df, 'source')
df_with_source.select('line', 'label', 'source').show(10)

# Save processed results
df.write.parquet('output/processed.parquet')

spark.stop()
```

## Workflow Patterns

### Pattern 1: Batch Processing

Process multiple files and consolidate results:

```python
# Read multiple files
text_contents = []
for filepath in ['corpus/file1.txt', 'corpus/file2.txt', 'corpus/file3.txt']:
    with open(filepath, 'r') as f:
        text_contents.append(f.read())

# Classify raw text strings
predictions = classifier.predict_lines(text_contents)

# Save all as YEDDA
classifier.save_yeda_output(predictions, 'output/yeda_batch')

# Load all YEDDA files
from glob import glob
yeda_files = glob('output/yeda_batch/**/*.txt', recursive=True)

# Parse and combine
dfs = [yeda_file_to_spark_df(f, spark) for f in yeda_files]
combined_df = dfs[0]
for df in dfs[1:]:
    combined_df = combined_df.union(df)

# Analyze combined results
get_label_statistics(combined_df).show()
```

### Pattern 2: Quality Checking

Review classification results:

```python
# Read and classify
with open('article.txt', 'r') as f:
    text = f.read()

predictions = classifier.predict_lines([text])
classifier.save_yeda_output(predictions, 'output/yeda')

# Parse
df = yeda_file_to_spark_df('output/yeda/article.txt', spark)

# Find potential issues
short_nomenclature = df.filter(
    (df.label == "Nomenclature") &
    (length(df.line) < 20)
)
short_nomenclature.show()

# Check label transitions (via line_number gaps)
from pyspark.sql.functions import lag, col
from pyspark.sql.window import Window

window = Window.orderBy("line_number")
df_with_prev = df.withColumn("prev_label", lag("label").over(window))

transitions = df_with_prev.filter(col("label") != col("prev_label"))
transitions.select("line_number", "prev_label", "label", "line").show()
```

### Pattern 3: Re-annotation

Update annotations based on rules:

```python
# Parse existing YEDDA
df = yeda_file_to_spark_df('output/yeda/article.txt', spark)

# Apply correction rules
from pyspark.sql.functions import when

df_corrected = df.withColumn(
    "corrected_label",
    when(
        (df.label == "Misc-exposition") & df.line.contains("gen. nov."),
        "Nomenclature"
    ).otherwise(df.label)
)

# Save corrected version
# (You'd need to write back to YEDDA format - see yeda_parser docs)
```

## Data Flow Diagram

```
Raw Text Files
      ↓
[skol_classifier.predict_lines()]
      ↓
Line-level Predictions (DataFrame)
      ↓
[skol_classifier.save_yeda_output()]
      ↓
YEDDA Format Files (consecutive labels coalesced)
      ↓
[yeda_parser.yeda_file_to_spark_df()]
      ↓
Structured DataFrame with Metadata
      ↓
Analysis / Queries / Export
```

## File Formats

### Input (Raw Text)
```
Glomus mosseae Nicolson & Gerdemann, 1963.
≡ Glomus mosseae (Nicolson & Gerdemann) C. Walker

Key characters: Spores formed singly.
Spore wall: mono- to multiple-layered.

This species is common.
```

### Intermediate (YEDDA)
```
[@ Glomus mosseae Nicolson & Gerdemann, 1963.
≡ Glomus mosseae (Nicolson & Gerdemann) C. Walker
#Nomenclature*]
[@
#Misc-exposition*]
[@ Key characters: Spores formed singly.
Spore wall: mono- to multiple-layered.
#Description*]
[@
This species is common.
#Misc-exposition*]
```

### Output (DataFrame Schema)
```
root
 |-- label: string (nullable = false)
 |-- line: string (nullable = false)
 |-- line_number: integer (nullable = false)
 |-- metadata: string (nullable = false)
```

## Best Practices

1. **Add Metadata**: Always include provenance information (source, date, version)
2. **Validate Results**: Use `get_label_statistics()` to check distribution
3. **Save Checkpoints**: Save intermediate YEDDA files for reproducibility
4. **Use Parquet**: For large datasets, save final DataFrames as Parquet
5. **Document Labels**: Keep track of your label taxonomy

## See Also

- `yeda_parser/README.md`: Complete YEDDA parser documentation
- `skol_classifier/README.md`: Classifier documentation
- `skol_classifier/example_line_classification.py`: Line classification example
- `yeda_parser/example_yeda_to_spark.py`: YEDDA parsing example
