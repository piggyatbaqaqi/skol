"""
ExtractionMode classes for handling mode-specific behavior.

These classes encapsulate mode-specific logic that was previously
handled with if statements checking extraction_mode strings.
"""

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from ..classifier_v2 import TaxaClassifier


class ExtractionMode(ABC):
    """
    Base class for extraction modes.

    Encapsulates mode-specific behavior for different text extraction
    granularities (line, paragraph, section).
    """

    @property
    def name(self) -> str:
        """Return the string name of this extraction mode."""
        raise NotImplementedError

    @property
    def is_line_mode(self) -> bool:
        """Return True if this is line-level extraction mode."""
        return False

    def load_from_files(self, classifier: 'TaxaClassifier') -> 'DataFrame':
        """
        Load raw text from local files.

        Args:
            classifier: The TaxaClassifier instance

        Returns:
            DataFrame with raw text data
        """
        # Default implementation for line and paragraph modes
        from ..preprocessing import RawTextLoader

        loader = RawTextLoader(classifier.spark)
        df = loader.load_files(
            classifier.file_paths,
            line_level=self.is_line_mode
        )

        return classifier.load_raw_from_df(df)

    def load_from_couchdb(self, classifier: 'TaxaClassifier') -> 'DataFrame':
        """
        Load raw text from CouchDB.

        Args:
            classifier: The TaxaClassifier instance

        Returns:
            DataFrame with raw text data
        """
        # Default implementation for line and paragraph modes
        from ..classifier_v2 import CouchDBConnection

        conn = CouchDBConnection(
            classifier.couchdb_url,
            classifier.couchdb_database,
            classifier.couchdb_username,
            classifier.couchdb_password
        )

        # Load using attachment pattern
        if not classifier.couchdb_attachment_pattern:
            raise ValueError(
                "couchdb_attachment_pattern must be specified for "
                "CouchDB-based loading"
            )

        if classifier.verbosity >= 1:
            print(
                f"[Classifier] Loading training data from CouchDB: "
                f"{classifier.couchdb_database}"
            )
            print(f"[Classifier] Attachment pattern: "
                  f"{classifier.couchdb_attachment_pattern}")

        df = conn.load_distributed(
            classifier.spark,
            classifier.couchdb_attachment_pattern
        )

        return classifier.load_raw_from_df(df)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other) -> bool:
        """Allow comparison with string for backwards compatibility."""
        if isinstance(other, str):
            return self.name == other
        return isinstance(other, ExtractionMode) and self.name == other.name


class LineExtractionMode(ExtractionMode):
    """Line-level extraction mode."""

    @property
    def name(self) -> str:
        return 'line'

    @property
    def is_line_mode(self) -> bool:
        return True


class ParagraphExtractionMode(ExtractionMode):
    """Paragraph-level extraction mode."""

    @property
    def name(self) -> str:
        return 'paragraph'


class SectionExtractionMode(ExtractionMode):
    """Section-level extraction mode."""

    @property
    def name(self) -> str:
        return 'section'

    def load_from_files(self, classifier: 'TaxaClassifier') -> 'DataFrame':
        """
        Load sections from PDF files.

        Note: For 'section' extraction mode, PDFs must be stored in CouchDB.
        The file_paths parameter is not supported for section mode.
        Use input_source='couchdb' instead.
        """
        raise NotImplementedError(
            "Section-based tokenization requires PDFs to be stored in "
            "CouchDB. "
            "Please use input_source='couchdb' with tokenizer='section'. "
            "The PDFSectionExtractor only supports loading from CouchDB "
            "attachments."
        )

    def load_from_couchdb(self, classifier: 'TaxaClassifier') -> 'DataFrame':
        """
        Load sections from PDF documents in CouchDB using
        PDFSectionExtractor.

        Args:
            classifier: The TaxaClassifier instance

        Returns:
            DataFrame with section data
        """
        # Delegate to classifier's _load_sections_from_couchdb method
        return classifier._load_sections_from_couchdb()
