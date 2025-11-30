# YEDDA Parser Module

A Python module for parsing YEDDA (Yet Another Entity Detection and Annotation) format text and converting it to PySpark DataFrames.

## YEDDA Format

YEDDA is an annotation format used for labeling text data. Each annotation block has the structure:

```
[@ <text content>
#<label>*]
```

Example:
```
[@ Glomus mosseae Nicolson & Gerdemann, 1963.
#Nomenclature*]
```

## Features

- Parse YEDDA-annotated strings or files
- Convert to PySpark DataFrames with line-level granularity
- Compute label statistics
- Preserve line numbers within annotation blocks

## Installation

Requires:
- Python 3.7+
- PySpark 3.0+

```bash
pip install pyspark
```

## Usage

### Basic Parsing

```python
from yedda_parser import parse_yedda_string

text = """[@ Line 1
Line 2
#Label1*]
[@ Line 3
#Label2*]"""

result = parse_yedda_string(text)
# Returns: [('Label1', 'Line 1', 0), ('Label1', 'Line 2', 1), ('Label2', 'Line 3', 0)]
```

### Convert to PySpark DataFrame

```python
from pyspark.sql import SparkSession
from yedda_parser import yedda_to_spark_df

spark = SparkSession.builder.appName("YEDDA Parser").getOrCreate()

text = "[@ First line\nSecond line\n#Nomenclature*]"
df = yedda_to_spark_df(text, spark)

df.show()
# +-------------+-----------+-----------+
# |        label|       line|line_number|
# +-------------+-----------+-----------+
# |Nomenclature|First line |          0|
# |Nomenclature|Second line|          1|
# +-------------+-----------+-----------+
```

### Parse Files

```python
from yedda_parser import yedda_file_to_spark_df

df = yedda_file_to_spark_df('article_reference.txt', spark)
print(f"Parsed {df.count()} lines")
```

### Get Label Statistics

```python
from yedda_parser import get_label_statistics

stats = get_label_statistics(df)
stats.show()
# +---------------+-----+----------+
# |          label|count|percentage|
# +---------------+-----+----------+
# |Misc-exposition|  708|     45.15|
# |   Nomenclature|  650|     41.45|
# |    Description|  129|      8.23|
# +---------------+-----+----------+
```

### Example Queries

```python
# Filter by label
nomenclature_df = df.filter(df.label == "Nomenclature")

# Search for specific content
glomus_df = df.filter(df.line.contains("Glomus"))

# Extract years from nomenclature
from pyspark.sql.functions import regexp_extract

year_df = df.withColumn("year", regexp_extract(df.line, r'\b(19|20)\d{2}\b', 0))
```

### Save Results

```python
# Save as Parquet
df.write.parquet("output.parquet")

# Save as CSV
df.write.csv("output.csv", header=True)

# Save as JSON
df.write.json("output.json")
```

## Module Functions

### `parse_yedda_string(yedda_text: str) -> List[Tuple[str, str, int]]`

Parse YEDDA-annotated string into list of (label, line, line_number) tuples.

### `parse_yedda_file(filepath: str) -> List[Tuple[str, str, int]]`

Parse YEDDA-annotated file into list of tuples.

### `yedda_to_spark_df(yedda_text: str, spark: SparkSession) -> DataFrame`

Convert YEDDA-annotated text to PySpark DataFrame.

Returns DataFrame with columns:
- `label` (string): Annotation label
- `line` (string): Text content
- `line_number` (int): 0-indexed line number within annotation block

### `yedda_file_to_spark_df(filepath: str, spark: SparkSession) -> DataFrame`

Convert YEDDA-annotated file to PySpark DataFrame.

### `get_label_statistics(df: DataFrame) -> DataFrame`

Compute statistics about labels in the DataFrame.

Returns DataFrame with columns:
- `label`: The annotation label
- `count`: Number of lines with this label
- `percentage`: Percentage of total lines

## Example Script

See `example_yedda_to_spark.py` for a complete example that:
- Loads a YEDDA file
- Creates a Spark DataFrame
- Computes statistics
- Runs example queries
- Saves results

Run it with (from the skol root directory):
```bash
python yedda_parser/example_yedda_to_spark.py
```

## Testing

Run the test suite (from the skol root directory):
```bash
python -m pytest yedda_parser/test_yedda_parser.py -v
```

## Labels in article_reference.txt

The example file uses three labels:

- **Nomenclature**: Formal taxonomic names and citations
- **Description**: Morphological descriptions, diagnoses, key characters
- **Misc-exposition**: General text (introduction, methods, discussion)

## Notes

- The regex parser strips leading/trailing whitespace around annotation blocks
- Empty lines within blocks are preserved
- Line numbers are 0-indexed within each annotation block
- Multi-line content is automatically split into individual lines

## License

Part of the skol taxonomic text processing toolkit.
