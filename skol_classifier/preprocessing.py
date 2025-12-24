"""
Text preprocessing utilities for SKOL classifier
"""

import regex as re
from typing import List
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, explode, collect_list, regexp_extract
from pyspark.sql.types import ArrayType, StringType


SUFFIXES=r'(ae}al|am|an|ar|ba|be|bi|ca|ch|ci|ck|da|di|ea|ed|ei|en|er|es|ev|gi|ha|he|ia|ic|id|ii|is|ix|íz|la|le|li|ll|ma|me|na|nd|ni|ns|o|oa|oé|of|oi|on|or|os|ox|pa|ph|ps|ra|re|ri|rt|sa|se|si|ta|te|ti|ts|ty|ua|ud|um|up|us|va|vá|xa|ya|yi|ys|za|zi)'


# Regex patterns for nomenclature detection
NOMENCLATURE_RE = re.compile(
    r'^([-\w≡=.*|:]*\s+)?'
    r'('
        r'([A-Z]\w*' + SUFFIXES + r'?)'
        r'|(\w+\b)'
    r')'
    r'\s'
    r'('
        r'(\w+' + SUFFIXES + r'?)'
        r'|(\w+\b)'
    r')'
    r'.*'
    r'('
        r'nov\.|'
        r'nov\.\s?(comb\.|sp\.)|'
        r'\(?in\.?\s?ed\.?\)?|'
        r'\(?nom\.\s?(prov\.|sanct\.)\)?|'
        r'emend\..*|'
        r'\(?\b[12]\d{3}\b(?:\s+.[12]\d{3}\b.)?\)?[^\n]{0,9}'
    r')'
    r'[-\s—]*'
    r'(\(?Fig|Plate\)?)?$'
)

TABLE_KEYWORDS = ['table', 'tab.', 'tab', 'tbl.', 'tbl']
FIGURE_KEYWORDS = ['fig', 'fig.', 'figg.', 'figs', 'figs.', 'figure', 'photo', 'plate', 'plates']
TAXON_PATTERN = (
    r'(nov\.|nov\.\s?(comb\.|sp\.)|\(?in\.?\s?ed\.?\)?|\(?nom\.\s?sanct\.?\)?|emend\..*)'
    r'|\b[12]\d{3}\b.{0,3}'
    r'\s*(\(?Fig[^\)]*\)?)?$'
)


class SuffixTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """
    Custom PySpark Transformer that extracts word suffixes (2-4 characters).

    This transformer is MLWritable and MLReadable, so it can be saved and loaded
    as part of PySpark ML pipelines.
    """

    def __init__(self, inputCol: str = "words", outputCol: str = "suffixes"):
        """
        Initialize the SuffixTransformer.

        Args:
            inputCol: Column name containing tokenized words
            outputCol: Column name for output suffixes
        """
        super(SuffixTransformer, self).__init__()
        self._setDefault(inputCol="words", outputCol="suffixes")
        self.setParams(inputCol=inputCol, outputCol=outputCol)

    def setParams(self, inputCol: str = "words", outputCol: str = "suffixes"):
        """
        Set parameters for the transformer.

        Args:
            inputCol: Column name containing tokenized words
            outputCol: Column name for output suffixes

        Returns:
            self
        """
        return self._set(inputCol=inputCol, outputCol=outputCol)

    @staticmethod
    def extract_suffixes(words: List[str]) -> List[str]:
        """
        Extract 2, 3, and 4 character suffixes from words.

        Args:
            words: List of words

        Returns:
            List of suffixes
        """
        retval = []
        for word in words:
            retval.extend([word[-2:], word[-3:], word[-4:]])
        return retval

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Transform the DataFrame by adding suffix column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with suffix column added
        """
        suffixes_udf = udf(self.extract_suffixes, ArrayType(StringType()))

        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        if isinstance(df.schema[input_col].dataType, ArrayType):
            return df.withColumn(output_col, suffixes_udf(df[input_col]))
        else:
            raise ValueError(
                f"Column '{input_col}' is not an array type but a "
                f"{type(df.schema[input_col].dataType)}."
            )


class ParagraphExtractor:
    """
    Extracts paragraphs from text files using heuristic rules.
    """

    @staticmethod
    def extract_annotated_paragraphs(lines: List[str]) -> List[str]:
        """
        Extract and aggregate text between '[' and ']' markers for annotated data.

        Args:
            lines: List of text lines

        Returns:
            List of aggregated paragraphs
        """
        aggregated = []
        temp = []
        inside_brackets = False

        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("["):
                inside_brackets = True
                temp = [stripped_line]
            elif stripped_line.endswith("]") and inside_brackets:
                temp.append(stripped_line)
                aggregated.append(" ".join(temp))
                temp = []
                inside_brackets = False
            elif inside_brackets:
                temp.append(stripped_line)

        return aggregated

    @staticmethod
    def extract_heuristic_paragraphs(lines: List[str]) -> List[str]:
        """
        Heuristically identify paragraphs from raw text.

        Uses multiple heuristics including:
        - Nomenclature detection
        - Table/figure detection
        - Blank lines
        - Line length
        - Special patterns (synonyms, taxons)

        Args:
            lines: List of text lines

        Returns:
            List of identified paragraphs
        """
        aggregated = []
        temp = []
        inside_table = False
        inside_figure = False

        for line in lines:
            # Handle table content
            if inside_table:
                if len(line.strip()) < 45:
                    temp.append(line)
                    continue
                elif all(len(sentence.strip()) > 45 for sentence in temp):
                    temp.append(line)
                    continue
                inside_table = False
                temp = [line]
                continue

            # Handle figure content
            if inside_figure:
                if not (line.strip().endswith(".") or line.strip().endswith(":") or line.strip() == ""):
                    temp.append(line)
                else:
                    inside_figure = False
                    temp = [line]
                continue

            # Detect nomenclature
            if NOMENCLATURE_RE.match(line):
                if temp:
                    aggregated.append(" ".join(temp))
                temp = [line]
                continue

            # Page break
            if line.startswith('\f'):
                if temp:
                    aggregated.append(" ".join(temp))
                temp = [line]
                continue

            # Leading tab
            if line.startswith('\t'):
                if temp:
                    aggregated.append(" ".join(temp))
                temp = [line]
                continue

            # Detect table
            if any(line.lower().startswith(keyword) for keyword in TABLE_KEYWORDS):
                inside_table = True
                if temp:
                    aggregated.append(" ".join(temp))
                temp = [line]
                continue

            # Blank line
            if line.strip() == "":
                if temp and temp[-1] == "":
                    continue
                else:
                    if temp:
                        aggregated.append(" ".join(temp))
                    temp = [line]
                continue

            # Detect figure
            if any(line.lower().startswith(keyword) for keyword in FIGURE_KEYWORDS):
                if temp:
                    aggregated.append(" ".join(temp))
                temp = [line]
                inside_figure = True
                continue

            # Detect hyphen
            if line.startswith('-'):
                if temp:
                    aggregated.append(" ".join(temp))
                temp = [line]
                continue

            # Detect synonym
            if re.search(r'\([Ss]yn.*\)$', line):
                temp.append(line)
                if temp:
                    aggregated.append(" ".join(temp))
                temp = []
                continue

            # Detect taxon
            if re.search(TAXON_PATTERN, line):
                temp.append(line)
                if temp:
                    aggregated.append(" ".join(temp))
                temp = []
                continue

            # Short line ends paragraph
            if len(line.strip()) < 45:
                temp.append(line)
                if temp:
                    aggregated.append(" ".join(temp))
                temp = []
                continue

            # Otherwise, append to current paragraph
            temp.append(line)

        return aggregated

    @staticmethod
    def collapse_labels(label: str) -> str:
        """
        Collapse labels to three main categories.

        Args:
            label: Original label

        Returns:
            Collapsed label (Nomenclature, Description, or Misc-exposition)
        """
        if label not in ['Nomenclature', 'Description']:
            return 'Misc-exposition'
        return label



class AnnotatedTextParser:
    """
    Parser for YEDDA-annotated text stored in CouchDB.

    Extracts labeled text from YEDDA annotation format:
    [@content#Label*]

    Supports line-level, paragraph-level, and section-level extraction.
    Detects section names from content for compatibility with section-based features.
    """

    def __init__(self, extraction_mode: str = 'paragraph', collapse_labels: bool = True, line_level: bool = None):
        """
        Initialize the AnnotatedTextParser.

        Args:
            extraction_mode: Extraction granularity - 'line', 'paragraph', or 'section'
            collapse_labels: If True, collapse labels to 3 main categories
            line_level: DEPRECATED - use extraction_mode instead. For backwards compatibility only.
        """
        # Handle backwards compatibility
        if line_level is not None:
            import warnings
            warnings.warn(
                "line_level parameter is deprecated, use extraction_mode='line' instead",
                DeprecationWarning,
                stacklevel=2
            )
            self.extraction_mode = 'line' if line_level else 'paragraph'
        else:
            self.extraction_mode = extraction_mode

        self.collapse_labels = collapse_labels

    @property
    def line_level(self) -> bool:
        """Backwards compatibility property."""
        return self.extraction_mode == 'line'

    @staticmethod
    def _get_section_name(text: str) -> str:
        """
        Extract standardized section name from text content.

        Args:
            text: Text to analyze (typically first line or header)

        Returns:
            Standardized section name, or None if not a known section
        """
        text_lower = text.strip().lower()

        # Map of section keywords to standardized names
        section_map = {
            'introduction': 'Introduction',
            'abstract': 'Abstract',
            'key words': 'Keywords',
            'keywords': 'Keywords',
            'taxonomy': 'Taxonomy',
            'materials and methods': 'Materials and Methods',
            'methods': 'Methods',
            'results': 'Results',
            'discussion': 'Discussion',
            'acknowledgments': 'Acknowledgments',
            'acknowledgements': 'Acknowledgments',
            'references': 'References',
            'conclusion': 'Conclusion',
            'description': 'Description',
            'etymology': 'Etymology',
            'specimen': 'Specimen',
            'holotype': 'Holotype',
            'paratype': 'Paratype',
            'literature cited': 'Literature Cited',
            'background': 'Background',
            'objectives': 'Objectives',
            'summary': 'Summary',
            'figures': 'Figures',
            'tables': 'Tables',
            'appendix': 'Appendix',
            'supplementary': 'Supplementary'
        }

        # Check for exact matches or starts with
        for keyword, standard_name in section_map.items():
            if text_lower == keyword or text_lower.startswith(keyword):
                return standard_name

        return None

    def parse(self, df: DataFrame) -> DataFrame:
        """
        Parse YEDDA-annotated text from a DataFrame.

        Args:
            df: DataFrame with columns (doc_id, human_url, attachment_name, value)
                where value contains YEDDA-annotated text
                For section mode, may also include (line_number, page_number, section_name, etc.)

        Returns:
            DataFrame with columns (doc_id, human_url, attachment_name, label, value, section_name)
            For extraction_mode='line', also includes line_number column
            For extraction_mode='section', preserves all input metadata columns
        """
        from pyspark.sql.types import StructType, StructField, IntegerType

        if self.extraction_mode == 'section':
            # Section-level extraction has two modes:
            # 1. Pre-segmented input (from PDFSectionExtractor): preserve structure, extract labels
            # 2. Plain text input (from .txt.ann files): parse like paragraphs with section detection

            # Check if input has pre-existing structure (line_number column indicates this)
            if 'line_number' in df.columns:
                # Pre-segmented: just extract YEDDA labels, preserve all metadata
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
            else:
                # Plain text: parse like paragraphs with section detection
                # This handles training data from .txt.ann files
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

        elif self.extraction_mode == 'line':
            # Line-level extraction: parse each line from YEDDA blocks
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
        elif self.extraction_mode == 'paragraph':
            # Paragraph-level extraction
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
        else:
            raise ValueError(
                f"Unknown extraction_mode: {self.extraction_mode}. "
                f"Must be 'line', 'paragraph', or 'section'"
            )

        return result_df


__all__ = [
    'SuffixTransformer',
    'ParagraphExtractor',
    'AnnotatedTextParser',
    'SUFFIXES',
    'NOMENCLATURE_RE',
    'TABLE_KEYWORDS',
    'FIGURE_KEYWORDS',
    'TAXON_PATTERN',
]
