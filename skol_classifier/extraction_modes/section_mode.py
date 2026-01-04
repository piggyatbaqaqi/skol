"""
Section-level extraction mode.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

from .base import AnnotatedTextParser
from ..preprocessing import ParagraphExtractor


class SectionAnnotatedTextParser(AnnotatedTextParser):
    """Parser for section-level extraction mode."""

    @property
    def extraction_mode(self) -> str:
        return 'section'

    def parse(self, df: DataFrame) -> DataFrame:
        """
        Parse YEDDA-annotated text at section level.

        Section-level extraction has two modes:
        1. Pre-segmented input (from PDFSectionExtractor): preserve structure, extract labels
        2. Plain text input (from .txt.ann files): parse like paragraphs with section detection

        Args:
            df: DataFrame with columns (doc_id, human_url, attachment_name, value)
                For pre-segmented input, may also include (line_number, page_number, section_name, etc.)

        Returns:
            DataFrame with columns (doc_id, human_url, attachment_name, label, value, section_name)
            For pre-segmented input, preserves all input metadata columns
        """
        # Check if input has pre-existing structure (line_number column indicates this)
        if 'line_number' in df.columns:
            # Pre-segmented: just extract YEDDA labels, preserve all metadata
            result_df = self._parse_presegmented(df)
        else:
            # Plain text: parse like paragraphs with section detection
            # This handles training data from .txt.ann files
            result_df = self._parse_plain_text(df)

        return result_df

    def _parse_presegmented(self, df: DataFrame) -> DataFrame:
        """
        Parse pre-segmented section data (from PDFSectionExtractor).

        Args:
            df: DataFrame with pre-existing structure columns

        Returns:
            DataFrame with extracted labels, preserving all metadata
        """
        def extract_yedda_label(text: str):
            """Extract label from YEDDA annotation block."""
            import re
            # Match YEDDA format: [@ content #Label*]
            pattern = r'\[@\s*.*?\s*#([^\*]+)\*\]'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                label = match.group(1).strip()
                # Collapse labels if requested
                if self.collapse_labels:
                    label = ParagraphExtractor.collapse_labels(label)
                return label
            return None

        # UDF to extract label
        extract_label_udf = udf(extract_yedda_label, StringType())

        # Apply UDF to extract label, preserve all existing columns
        result_df = df.withColumn("label", extract_label_udf(col("value")))

        # Filter out rows without labels (if any)
        result_df = result_df.filter(col("label").isNotNull())

        return result_df

    def _parse_plain_text(self, df: DataFrame) -> DataFrame:
        """
        Parse plain text with section detection (from .txt.ann files).

        Args:
            df: DataFrame with plain text columns

        Returns:
            DataFrame with extracted paragraphs and section names
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
