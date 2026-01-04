"""
Base class for extraction modes and annotated text parsers.
"""

from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, col


class AnnotatedTextParser(ABC):
    """
    Abstract base class for parsing YEDDA-annotated text.

    Different extraction modes (line, paragraph, section) have different ways of
    parsing YEDDA annotation format: [@content#Label*]
    """

    def __init__(self, collapse_labels: bool = True):
        """
        Initialize the parser.

        Args:
            collapse_labels: If True, collapse labels to 3 main categories
        """
        self.collapse_labels = collapse_labels

    @property
    @abstractmethod
    def extraction_mode(self) -> str:
        """Return the string name of this extraction mode ('line', 'paragraph', or 'section')."""
        pass

    @property
    def line_level(self) -> bool:
        """Backwards compatibility property: returns True if extraction_mode is 'line'."""
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

    @abstractmethod
    def parse(self, df: DataFrame) -> DataFrame:
        """
        Parse YEDDA-annotated text from a DataFrame.

        Args:
            df: DataFrame with columns (doc_id, human_url, attachment_name, value)
                where value contains YEDDA-annotated text

        Returns:
            DataFrame with columns specific to the extraction mode
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(extraction_mode='{self.extraction_mode}')"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(collapse_labels={self.collapse_labels})"
