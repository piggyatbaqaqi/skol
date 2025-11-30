"""
Data loading module for SKOL classifier.

This module provides classes for loading annotated and raw text data
from various sources (files, CouchDB, strings).
"""

from typing import List, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    input_file_name, collect_list, regexp_extract, col, udf,
    explode, trim, row_number, min as sql_min, monotonically_increasing_id
)
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
from pyspark.sql.window import Window

from .preprocessing import ParagraphExtractor
from .couchdb_io import CouchDBConnection


class AnnotatedTextLoader:
    """
    Loads annotated text data from various sources.

    Supports file-based and CouchDB-based loading with paragraph-level
    or line-level extraction.
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the loader.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark

    def load_from_files(
        self,
        file_paths: List[str],
        collapse_labels: bool = True,
        line_level: bool = False
    ) -> DataFrame:
        """
        Load and preprocess annotated data from files.

        Args:
            file_paths: List of paths to annotated files
            collapse_labels: Whether to collapse labels to 3 main categories
            line_level: If True, extract individual lines instead of paragraphs

        Returns:
            Preprocessed DataFrame with paragraphs/lines and labels
        """
        # Read annotated files and add line ID to preserve order
        ann_df = (
            self.spark.read.text(file_paths)
            .withColumn("filename", input_file_name())
            .withColumn("_line_id", monotonically_increasing_id())
        )

        if line_level:
            # Line-level extraction: parse each line from YEDDA blocks
            def extract_yedda_lines(lines: List[str]) -> List[Tuple[str, str, int]]:
                """Extract individual lines from YEDDA annotation blocks."""
                import re
                results = []
                pattern = r'\[@\s*(.*?)\s*#([^\*]+)\*\]'

                for match in re.finditer(pattern, '\n'.join(lines), re.DOTALL):
                    content = match.group(1)
                    label = match.group(2).strip()

                    # Split content into lines
                    content_lines = content.split('\n')
                    for line_num, line in enumerate(content_lines):
                        if line or line_num < len(content_lines) - 1:
                            results.append((label, line, line_num))

                return results

            # UDF to extract lines
            extract_udf = udf(
                extract_yedda_lines,
                ArrayType(StructType([
                    StructField("label", StringType(), False),
                    StructField("value", StringType(), False),
                    StructField("line_number", IntegerType(), False)
                ]))
            )

            # Extract lines with ordering preserved
            grouped_df = (
                ann_df.orderBy("_line_id")
                .groupBy("filename")
                .agg(collect_list("value").alias("lines"))
                .withColumn("line_data", explode(extract_udf(col("lines"))))
                .select(
                    "filename",
                    col("line_data.label").alias("label"),
                    col("line_data.value").alias("value"),
                    col("line_data.line_number")
                )
            )
        else:
            # Paragraph-level extraction (original behavior)
            extract_udf = udf(
                ParagraphExtractor.extract_annotated_paragraphs,
                ArrayType(StringType())
            )

            # Group and extract paragraphs with ordering preserved
            grouped_df = (
                ann_df.orderBy("_line_id")
                .groupBy("filename")
                .agg(collect_list("value").alias("lines"))
                .withColumn("value", explode(extract_udf(col("lines"))))
                .drop("lines")
            )

            # Extract labels
            label_pattern = r"#(\S+?)(?:\*)?]"
            lead_pattern = r"^\[@"
            trail_pattern = label_pattern + r"$"
            clean_pattern = lead_pattern + r"(.*)" + trail_pattern

            grouped_df = grouped_df.withColumn(
                "label", regexp_extract(col("value"), label_pattern, 1)
            ).withColumn(
                "value", regexp_extract(col("value"), clean_pattern, 1)
            )

        # Optionally collapse labels
        if collapse_labels:
            collapse_udf = udf(
                ParagraphExtractor.collapse_labels,
                StringType()
            )
            grouped_df = grouped_df.withColumn(
                "label", collapse_udf(col("label"))
            )

        return grouped_df

    def load_from_couchdb(
        self,
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt.ann",
        collapse_labels: bool = True,
        line_level: bool = False
    ) -> DataFrame:
        """
        Load annotated data from CouchDB.

        Args:
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
            pattern: Pattern to match attachment names
            collapse_labels: Whether to collapse labels
            line_level: If True, extract lines instead of paragraphs

        Returns:
            DataFrame with annotated data
        """
        conn = CouchDBConnection(
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

        # Load annotated attachments from CouchDB
        df = conn.load_distributed(
            spark=self.spark,
            pattern=pattern
        )

        # Split content into lines for processing
        from pyspark.sql.functions import split as sql_split
        df = df.withColumn("lines", sql_split(col("value"), "\n"))

        # Process similar to file loading
        if line_level:
            def extract_yedda_lines(lines: List[str]) -> List[Tuple[str, str, int]]:
                import re
                results = []
                pattern_re = r'\[@\s*(.*?)\s*#([^\*]+)\*\]'

                for match in re.finditer(pattern_re, '\n'.join(lines), re.DOTALL):
                    content = match.group(1)
                    label = match.group(2).strip()

                    content_lines = content.split('\n')
                    for line_num, line in enumerate(content_lines):
                        if line or line_num < len(content_lines) - 1:
                            results.append((label, line, line_num))

                return results

            extract_udf = udf(
                extract_yedda_lines,
                ArrayType(StructType([
                    StructField("label", StringType(), False),
                    StructField("value", StringType(), False),
                    StructField("line_number", IntegerType(), False)
                ]))
            )

            df = (
                df.withColumn("line_data", explode(extract_udf(col("lines"))))
                .select(
                    "doc_id",
                    "attachment_name",
                    col("line_data.label").alias("label"),
                    col("line_data.value").alias("value"),
                    col("line_data.line_number")
                )
            )
        else:
            extract_udf = udf(
                ParagraphExtractor.extract_annotated_paragraphs,
                ArrayType(StringType())
            )

            df = (
                df.withColumn("value", explode(extract_udf(col("lines"))))
                .drop("lines")
            )

            label_pattern = r"#(\S+?)(?:\*)?]"
            lead_pattern = r"^\[@"
            trail_pattern = label_pattern + r"$"
            clean_pattern = lead_pattern + r"(.*)" + trail_pattern

            df = df.withColumn(
                "label", regexp_extract(col("value"), label_pattern, 1)
            ).withColumn(
                "value", regexp_extract(col("value"), clean_pattern, 1)
            )

        if collapse_labels:
            collapse_udf = udf(
                ParagraphExtractor.collapse_labels,
                StringType()
            )
            df = df.withColumn("label", collapse_udf(col("label")))

        return df


class RawTextLoader:
    """
    Loads raw (unannotated) text data from various sources.

    Supports file-based and CouchDB-based loading with heuristic
    paragraph extraction or line-level processing.
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the loader.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark

    def load_from_files(
        self,
        file_paths: List[str],
        line_level: bool = False
    ) -> DataFrame:
        """
        Load and preprocess raw text data from files.

        Args:
            file_paths: List of paths to raw text files
            line_level: If True, process individual lines instead of paragraphs

        Returns:
            Preprocessed DataFrame with paragraphs or lines
        """
        # Read raw files and add line ID to preserve order
        df = (
            self.spark.read.text(file_paths)
            .withColumn("filename", input_file_name())
            .withColumn("_line_id", monotonically_increasing_id())
        )

        if line_level:
            # Line-level: add row numbers preserving order
            window_spec = Window.partitionBy("filename").orderBy("_line_id")
            return df.withColumn("row_number", row_number().over(window_spec))
        else:
            # Paragraph-level: use heuristic extraction
            heuristic_udf = udf(
                ParagraphExtractor.extract_heuristic_paragraphs,
                ArrayType(StringType())
            )

            # Window specification for ordering
            window_spec = Window.partitionBy("filename").orderBy("_line_id")

            return (
                df.orderBy("_line_id")
                .groupBy("filename")
                .agg(collect_list("value").alias("lines"))
                .withColumn("value", explode(heuristic_udf(col("lines"))))
                .drop("lines")
                .filter(trim(col("value")) != "")
                .withColumn("row_number", row_number().over(window_spec))
            )

    def load_from_couchdb(
        self,
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt",
        line_level: bool = False
    ) -> DataFrame:
        """
        Load raw text data from CouchDB.

        Args:
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
            pattern: Pattern to match attachment names
            line_level: If True, process lines instead of paragraphs

        Returns:
            DataFrame with raw text data
        """
        conn = CouchDBConnection(
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

        # Load raw attachments from CouchDB
        df = conn.load_distributed(
            spark=self.spark,
            pattern=pattern
        )

        # Split content into lines for processing
        from pyspark.sql.functions import split as sql_split
        df = df.withColumn("lines", sql_split(col("value"), "\n"))

        if line_level:
            # Line-level: explode lines and add line numbers
            window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy("doc_id")
            return (
                df.withColumn("value", explode(col("lines")))
                .drop("lines")
                .withColumn("line_number", row_number().over(window_spec))
            )
        else:
            # Paragraph-level: use heuristic extraction
            heuristic_udf = udf(
                ParagraphExtractor.extract_heuristic_paragraphs,
                ArrayType(StringType())
            )

            window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy("doc_id")

            return (
                df.withColumn("value", explode(heuristic_udf(col("lines"))))
                .drop("lines")
                .filter(trim(col("value")) != "")
                .withColumn("line_number", row_number().over(window_spec))
            )
