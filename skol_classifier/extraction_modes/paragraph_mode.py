"""
Paragraph-level extraction mode.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

from .base import AnnotatedTextParser
from ..preprocessing import ParagraphExtractor


class ParagraphAnnotatedTextParser(AnnotatedTextParser):
    """Parser for paragraph-level extraction mode."""

    @property
    def extraction_mode(self) -> str:
        return 'paragraph'

    def parse(self, df: DataFrame) -> DataFrame:
        """
        Parse YEDDA-annotated text into paragraphs.

        Args:
            df: DataFrame with columns (doc_id, human_url, attachment_name, value)

        Returns:
            DataFrame with columns (doc_id, human_url, attachment_name, label, value, section_name)
        """
        def extract_yedda_paragraphs(text: str):
            """Extract paragraphs from YEDDA annotation blocks with section detection."""
            import re
            results = []
            pattern = r'\[@\s*(.*?)\s*#([^\*]+)\*\]'

            for match in re.finditer(pattern, text, re.DOTALL):
                content = match.group(1).strip()
                label = match.group(2).strip()

                # Collapse labels if requested
                if self.collapse_labels:
                    label = ParagraphExtractor.collapse_labels(label)

                # Detect section name from first line of content
                first_line = content.split('\n')[0] if content else ""
                section_name = AnnotatedTextParser._get_section_name(first_line)

                if content:
                    results.append((label, content, section_name))

            return results

        # UDF to extract paragraphs
        extract_udf = udf(
            extract_yedda_paragraphs,
            ArrayType(StructType([
                StructField("label", StringType(), False),
                StructField("value", StringType(), False),
                StructField("section_name", StringType(), True)
            ]))
        )

        # Extract paragraphs and explode
        result_df = (
            df.withColumn("paragraph_data", explode(extract_udf(col("value"))))
            .select(
                "doc_id",
                "human_url",
                "attachment_name",
                col("paragraph_data.label").alias("label"),
                col("paragraph_data.value").alias("value"),
                col("paragraph_data.section_name")
            )
        )

        return result_df
