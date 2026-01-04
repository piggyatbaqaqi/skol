"""
Paragraph-level extraction mode implementation.
"""

from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    input_file_name, collect_list, regexp_extract, col, udf,
    explode, trim, row_number, monotonically_increasing_id,
    split as sql_split
)
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window

from .mode import ExtractionMode
from ..preprocessing import ParagraphExtractor
from ..couchdb_io import CouchDBConnection


class ParagraphExtractionMode(ExtractionMode):
    """Paragraph-level extraction mode."""

    @property
    def name(self) -> str:
        return 'paragraph'

    def load_raw_from_files(
        self,
        spark: SparkSession,
        file_paths: List[str]
    ) -> DataFrame:
        """Load raw text from files at paragraph level."""
        df = (
            spark.read.text(file_paths)
            .withColumn("filename", input_file_name())
            .withColumn("_line_id", monotonically_increasing_id())
        )

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

    def load_raw_from_couchdb(
        self,
        spark: SparkSession,
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt"
    ) -> DataFrame:
        """Load raw text from CouchDB at paragraph level."""
        conn = CouchDBConnection(
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

        df = conn.load_distributed(spark=spark, pattern=pattern)
        df = df.withColumn("lines", sql_split(col("value"), "\n"))

        # Paragraph-level: use heuristic extraction
        heuristic_udf = udf(
            ParagraphExtractor.extract_heuristic_paragraphs,
            ArrayType(StringType())
        )

        window_spec = Window.partitionBy("doc_id", "attachment_name") \
            .orderBy("doc_id")

        return (
            df.withColumn("value", explode(heuristic_udf(col("lines"))))
            .drop("lines")
            .filter(trim(col("value")) != "")
            .withColumn("line_number", row_number().over(window_spec))
        )

    def load_annotated_from_files(
        self,
        spark: SparkSession,
        file_paths: List[str],
        collapse_labels: bool = True
    ) -> DataFrame:
        """Load annotated text from files at paragraph level."""
        ann_df = (
            spark.read.text(file_paths)
            .withColumn("filename", input_file_name())
            .withColumn("_line_id", monotonically_increasing_id())
        )

        # Paragraph-level extraction
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
        """Load annotated text from CouchDB at paragraph level."""
        conn = CouchDBConnection(
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

        df = conn.load_distributed(spark=spark, pattern=pattern)
        df = df.withColumn("lines", sql_split(col("value"), "\n"))

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
