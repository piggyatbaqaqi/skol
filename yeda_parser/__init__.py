"""Parse YEDDA (Yet Another Entity Detection and Annotation) format.

YEDDA format consists of annotated text blocks:
    [@ <text content>
    #<label>*]

This module provides functions to parse YEDDA-annotated strings and convert
them to structured formats like PySpark DataFrames.
"""

import re
import json
from typing import List, Tuple, Optional, Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def parse_yeda_string(yeda_text: str) -> List[Tuple[str, str, int]]:
    """Parse a YEDDA-annotated string into (label, line, line_number) tuples.

    Args:
        yeda_text: String containing YEDDA annotations in format:
                   [@ <text>
                   #<label>*]

    Returns:
        List of tuples (label, line_text, line_number) for each line in the text.
        line_number is 0-indexed within each annotation block.

    Example:
        >>> text = "[@ Line 1\\nLine 2\\n#Label*]"
        >>> parse_yeda_string(text)
        [('Label', 'Line 1', 0), ('Label', 'Line 2', 1)]
    """
    results = []

    # Pattern to match YEDDA blocks: [@ ... #label*]
    # Using DOTALL to match across newlines
    pattern = r'\[@\s*(.*?)\s*#([^*]+)\*\]'

    for match in re.finditer(pattern, yeda_text, re.DOTALL):
        content = match.group(1)
        label = match.group(2).strip()

        # Split content into lines
        lines = content.split('\n')

        # Add each line with its label
        for line_num, line in enumerate(lines):
            # Skip empty lines at the end (trailing newlines)
            if line or line_num < len(lines) - 1:
                results.append((label, line, line_num))

    return results


def parse_yeda_file(filepath: str) -> List[Tuple[str, str, int]]:
    """Parse a YEDDA-annotated file into (label, line, line_number) tuples.

    Args:
        filepath: Path to file containing YEDDA annotations.

    Returns:
        List of tuples (label, line_text, line_number) for each line in the file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        yeda_text = f.read()

    return parse_yeda_string(yeda_text)


def yeda_to_spark_df(
    yeda_text: str,
    spark: Optional[SparkSession] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """Convert YEDDA-annotated text to a PySpark DataFrame.

    Args:
        yeda_text: String containing YEDDA annotations.
        spark: SparkSession instance. If None, creates a new session.
        metadata: Optional dictionary of metadata to attach to every row.
                 Useful for provenance tracking (e.g., source file, timestamp,
                 processing version, etc.). Will be serialized to JSON string.

    Returns:
        PySpark DataFrame with columns:
            - label (string): The annotation label
            - line (string): The text content of the line
            - line_number (int): 0-indexed line number within the annotation block
            - metadata (string): JSON string of metadata (if provided)

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("test").getOrCreate()
        >>> text = "[@ First line\\nSecond line\\n#Nomenclature*]"
        >>> metadata = {"source": "article.txt", "version": "1.0"}
        >>> df = yeda_to_spark_df(text, spark, metadata)
        >>> df.show()
        +-------------+-----------+-----------+--------------------+
        |        label|       line|line_number|            metadata|
        +-------------+-----------+-----------+--------------------+
        |Nomenclature|First line |          0|{"source": "artic...|
        |Nomenclature|Second line|          1|{"source": "artic...|
        +-------------+-----------+-----------+--------------------+
    """
    # Create SparkSession if not provided
    if spark is None:
        spark = SparkSession.builder \
            .appName("YEDDA Parser") \
            .getOrCreate()

    # Parse YEDDA text
    parsed_data = parse_yeda_string(yeda_text)

    # Add metadata to each row if provided
    if metadata is not None:
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        parsed_data = [(label, line, line_num, metadata_json)
                      for label, line, line_num in parsed_data]

        # Define schema with metadata
        schema = StructType([
            StructField("label", StringType(), False),
            StructField("line", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("metadata", StringType(), False)
        ])
    else:
        # Define schema without metadata
        schema = StructType([
            StructField("label", StringType(), False),
            StructField("line", StringType(), False),
            StructField("line_number", IntegerType(), False)
        ])

    # Create DataFrame
    df = spark.createDataFrame(parsed_data, schema)

    return df


def yeda_file_to_spark_df(
    filepath: str,
    spark: Optional[SparkSession] = None,
    metadata: Optional[Dict[str, Any]] = None,
    auto_add_filepath: bool = True
) -> DataFrame:
    """Convert YEDDA-annotated file to a PySpark DataFrame.

    Args:
        filepath: Path to file containing YEDDA annotations.
        spark: SparkSession instance. If None, creates a new session.
        metadata: Optional dictionary of metadata to attach to every row.
                 Useful for provenance tracking.
        auto_add_filepath: If True, automatically adds 'source_file' to metadata.
                          Default True.

    Returns:
        PySpark DataFrame with columns:
            - label (string): The annotation label
            - line (string): The text content of the line
            - line_number (int): 0-indexed line number within the annotation block
            - metadata (string): JSON string of metadata (if metadata provided or
                                auto_add_filepath is True)

    Example:
        >>> df = yeda_file_to_spark_df('article.txt', spark,
        ...                            metadata={'version': '1.0'})
        >>> # metadata will include both 'source_file' and 'version'
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        yeda_text = f.read()

    # Combine auto-added filepath with user metadata
    if auto_add_filepath:
        combined_metadata = {'source_file': filepath}
        if metadata:
            combined_metadata.update(metadata)
        return yeda_to_spark_df(yeda_text, spark, combined_metadata)
    else:
        return yeda_to_spark_df(yeda_text, spark, metadata)


def get_label_statistics(df: DataFrame) -> DataFrame:
    """Get statistics about labels in the DataFrame.

    Args:
        df: DataFrame from yeda_to_spark_df()

    Returns:
        DataFrame with columns:
            - label: The annotation label
            - count: Number of lines with this label
            - percentage: Percentage of total lines
    """
    from pyspark.sql.functions import count, col, round as spark_round

    total_count = df.count()

    stats = df.groupBy("label") \
        .agg(count("*").alias("count")) \
        .withColumn("percentage",
                   spark_round((col("count") / total_count) * 100, 2)) \
        .orderBy(col("count").desc())

    return stats


def extract_metadata_field(df: DataFrame, field_name: str) -> DataFrame:
    """Extract a specific field from the JSON metadata column.

    Args:
        df: DataFrame with metadata column
        field_name: Name of the field to extract from JSON

    Returns:
        DataFrame with additional column named after the field

    Example:
        >>> df_with_source = extract_metadata_field(df, 'source_file')
        >>> df_with_source.select('line', 'source_file').show()
    """
    from pyspark.sql.functions import get_json_object

    return df.withColumn(
        field_name,
        get_json_object(df.metadata, f'$.{field_name}')
    )


def add_metadata_to_existing_df(
    df: DataFrame,
    metadata: Dict[str, Any]
) -> DataFrame:
    """Add metadata column to an existing DataFrame without metadata.

    Args:
        df: DataFrame without metadata column
        metadata: Dictionary of metadata to add

    Returns:
        DataFrame with metadata column added

    Example:
        >>> df_no_meta = yeda_to_spark_df(text, spark)
        >>> meta = {'source': 'article.txt', 'date': '2025-01-01'}
        >>> df_with_meta = add_metadata_to_existing_df(df_no_meta, meta)
    """
    from pyspark.sql.functions import lit

    metadata_json = json.dumps(metadata, ensure_ascii=False)
    return df.withColumn('metadata', lit(metadata_json))


# Example usage and testing
if __name__ == "__main__":
    # Example YEDDA text
    example_text = """[@ ISSN (print) 0093-4666
© 2011. Mycotaxon, Ltd.
ISSN (online) 2154-8889
#Misc-exposition*]
[@ Glomus hoi S.M. Berch & Trappe, Mycologia 77: 654. 1985.
#Nomenclature*]
[@ Key characters: Spores formed singly or in very loose, small clusters.
Spores with a mono-to-multiple layered spore wall.
#Description*]"""

    print("Parsing YEDDA text...")
    parsed = parse_yeda_string(example_text)

    print(f"\nParsed {len(parsed)} lines:")
    for label, line, line_num in parsed[:5]:
        print(f"  [{label}] Line {line_num}: {line[:60]}...")

    print("\nCreating PySpark DataFrame...")
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder \
            .appName("YEDDA Parser Test") \
            .master("local[*]") \
            .getOrCreate()

        # Example 1: DataFrame without metadata
        df = yeda_to_spark_df(example_text, spark)

        print("\nDataFrame Schema (without metadata):")
        df.printSchema()

        print("\nFirst 5 rows:")
        df.show(5, truncate=50)

        # Example 2: DataFrame with metadata
        import datetime
        metadata = {
            'source': 'example_text',
            'parser_version': '1.0',
            'timestamp': datetime.datetime.now().isoformat(),
            'annotator': 'automated'
        }

        df_with_meta = yeda_to_spark_df(example_text, spark, metadata)

        print("\nDataFrame Schema (with metadata):")
        df_with_meta.printSchema()

        print("\nFirst 5 rows with metadata:")
        df_with_meta.show(5, truncate=40)

        print("\nExtracted source field:")
        df_extracted = extract_metadata_field(df_with_meta, 'source')
        df_extracted.select('label', 'source').show(5)

        print("\nLabel Statistics:")
        stats = get_label_statistics(df)
        stats.show()

        spark.stop()

    except ImportError:
        print("PySpark not installed. Install with: pip install pyspark")
    except Exception as e:
        print(f"Error: {e}")
