"""
Line-level extraction mode implementation.
"""

import re
from typing import List, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    input_file_name, collect_list, regexp_extract, col, udf,
    explode, monotonically_increasing_id, row_number, split as sql_split
)
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, IntegerType
from pyspark.sql.window import Window

from skol import constants

from .mode import ExtractionMode
from ..preprocessing import ParagraphExtractor
from ..couchdb_io import CouchDBConnection


class LineExtractionMode(ExtractionMode):
    """Line-level extraction mode."""

    @property
    def name(self) -> str:
        return 'line'

    @property
    def is_line_mode(self) -> bool:
        return True

    def load_raw_from_files(
        self,
        spark: SparkSession,
        file_paths: List[str]
    ) -> DataFrame:
        """Load raw text from files at line level."""
        df = (
            spark.read.text(file_paths)
            .withColumn("filename", input_file_name())
            .withColumn("_line_id", monotonically_increasing_id())
        )

        # Line-level: add row numbers preserving order
        window_spec = Window.partitionBy("filename").orderBy("_line_id")
        return df.withColumn("row_number", row_number().over(window_spec))

    def load_raw_from_couchdb(
        self,
        spark: SparkSession,
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt"
    ) -> DataFrame:
        """Load raw text from CouchDB at line level."""
        from pyspark.sql.functions import when, regexp_extract

        conn = CouchDBConnection(
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

        df = conn.load_distributed(spark=spark, pattern=pattern)
        df = df.withColumn("lines", sql_split(col("value"), "\n"))

        # Line-level: explode lines and add line numbers
        window_spec = Window.partitionBy("doc_id", "attachment_name") \
            .orderBy("doc_id")

        exploded_df = (
            df.withColumn("value", explode(col("lines")))
            .drop("lines")
            .withColumn("line_number", row_number().over(window_spec))
        )

        marker_pattern = re.compile(constants.pdf_page_pattern)

        # Mark PDF page markers (format: "--- PDF Page N Label L ---")
        # These will be preserved but not classified
        return exploded_df.withColumn(
            "is_page_marker",
            col("value").rlike(marker_pattern.pattern)
        )

    def load_annotated_from_files(
        self,
        spark: SparkSession,
        file_paths: List[str],
        collapse_labels: bool = True
    ) -> DataFrame:
        """Load annotated text from files at line level."""
        # Read annotated files
        ann_df = (
            spark.read.text(file_paths)
            .withColumn("filename", input_file_name())
            .withColumn("_line_id", monotonically_increasing_id())
        )

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

    def load_annotated_from_couchdb(
        self,
        spark: SparkSession,
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt.ann",
        collapse_labels: bool = True
    ) -> DataFrame:
        """Load annotated text from CouchDB at line level."""
        conn = CouchDBConnection(
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

        df = conn.load_distributed(spark=spark, pattern=pattern)
        df = df.withColumn("lines", sql_split(col("value"), "\n"))

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

        if collapse_labels:
            collapse_udf = udf(
                ParagraphExtractor.collapse_labels,
                StringType()
            )
            df = df.withColumn("label", collapse_udf(col("label")))

        return df
