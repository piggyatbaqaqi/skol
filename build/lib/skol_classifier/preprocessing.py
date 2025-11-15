"""
Text preprocessing utilities for SKOL classifier
"""

import regex as re
from typing import List
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, explode, collect_list, regexp_extract
from pyspark.sql.types import ArrayType, StringType


# Regex patterns for nomenclature detection
NOMENCLATURE_RE = re.compile(
    r'^([-\w≡=.*|:]*\s+)?'
    r'('
        r'([A-Z]\w*(aceae|idae|iformes|ales|ineae|inae)?)'
        r'|(\w+\b)'
    r')'
    r'\s'
    r'('
        r'(\w+(aceae|idae|iformes|ales|ineae|inae)?)'
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


class SuffixTransformer(Transformer):
    """
    Custom PySpark Transformer that extracts word suffixes (2-4 characters).
    """

    def __init__(self, inputCol: str = "words", outputCol: str = "suffixes"):
        """
        Initialize the SuffixTransformer.

        Args:
            inputCol: Column name containing tokenized words
            outputCol: Column name for output suffixes
        """
        super().__init__()
        self.input_col = inputCol
        self.output_col = outputCol

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

        if isinstance(df.schema[self.input_col].dataType, ArrayType):
            return df.withColumn(self.output_col, suffixes_udf(df[self.input_col]))
        else:
            raise ValueError(
                f"Column '{self.input_col}' is not an array type but a "
                f"{type(self.input_col)}."
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