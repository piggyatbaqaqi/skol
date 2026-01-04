"""
Line-level extraction mode.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

from .base import AnnotatedTextParser
from ..preprocessing import ParagraphExtractor


class LineAnnotatedTextParser(AnnotatedTextParser):
    """Parser for line-level extraction mode."""

    @property
    def extraction_mode(self) -> str:
        return 'line'

    def parse(self, df: DataFrame) -> DataFrame:
        """
        Parse YEDDA-annotated text into individual lines.

        Args:
            df: DataFrame with columns (doc_id, human_url, attachment_name, value)

        Returns:
            DataFrame with columns (doc_id, human_url, attachment_name, label, value, line_number, section_name)
        """
        def extract_yedda_lines(text: str):
            """Extract individual lines from YEDDA annotation blocks with section detection."""
            import re
            results = []
            pattern = r'\[@\s*(.*?)\s*#([^\*]+)\*\]'

            for match in re.finditer(pattern, text, re.DOTALL):
                content = match.group(1)
                label = match.group(2).strip()

                # Collapse labels if requested
                if self.collapse_labels:
                    label = ParagraphExtractor.collapse_labels(label)

                # Detect section name from first line of content
                first_line = content.split('\n')[0] if content else ""
                section_name = AnnotatedTextParser._get_section_name(first_line)

                # Split content into lines
                content_lines = content.split('\n')
                for line_num, line in enumerate(content_lines):
                    if line or line_num < len(content_lines) - 1:
                        results.append((label, line, line_num, section_name))

            return results

        # UDF to extract lines
        extract_udf = udf(
            extract_yedda_lines,
            ArrayType(StructType([
                StructField("label", StringType(), False),
                StructField("value", StringType(), False),
                StructField("line_number", IntegerType(), False),
                StructField("section_name", StringType(), True)
            ]))
        )

        # Extract lines and explode
        result_df = (
            df.withColumn("line_data", explode(extract_udf(col("value"))))
            .select(
                "doc_id",
                "human_url",
                "attachment_name",
                col("line_data.label").alias("label"),
                col("line_data.value").alias("value"),
                col("line_data.line_number"),
                col("line_data.section_name")
            )
        )

        return result_df
