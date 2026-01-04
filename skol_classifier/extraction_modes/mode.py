"""
ExtractionMode classes for handling mode-specific behavior.

These classes encapsulate mode-specific logic for different text extraction
granularities (line, paragraph, section), including data loading.
"""

from abc import ABC
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame


class ExtractionMode(ABC):
    """
    Base class for extraction modes.

    Encapsulates mode-specific behavior for different text extraction
    granularities (line, paragraph, section), including loading data
    from files and CouchDB.
    """

    @property
    def name(self) -> str:
        """Return the string name of this extraction mode."""
        raise NotImplementedError

    @property
    def is_line_mode(self) -> bool:
        """Return True if this is line-level extraction mode."""
        return False

    def load_raw_from_files(
        self,
        spark: 'SparkSession',
        file_paths: List[str]
    ) -> 'DataFrame':
        """
        Load raw (unannotated) text from local files.

        Args:
            spark: SparkSession instance
            file_paths: List of file paths to load

        Returns:
            DataFrame with raw text data
        """
        raise NotImplementedError

    def load_raw_from_couchdb(
        self,
        spark: 'SparkSession',
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt"
    ) -> 'DataFrame':
        """
        Load raw text from CouchDB.

        Args:
            spark: SparkSession instance
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
            pattern: Pattern to match attachment names

        Returns:
            DataFrame with raw text data
        """
        raise NotImplementedError

    def load_annotated_from_files(
        self,
        spark: 'SparkSession',
        file_paths: List[str],
        collapse_labels: bool = True
    ) -> 'DataFrame':
        """
        Load annotated text from local files.

        Args:
            spark: SparkSession instance
            file_paths: List of file paths to load
            collapse_labels: Whether to collapse labels to 3 main categories

        Returns:
            DataFrame with annotated data (includes label column)
        """
        raise NotImplementedError

    def load_annotated_from_couchdb(
        self,
        spark: 'SparkSession',
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt.ann",
        collapse_labels: bool = True
    ) -> 'DataFrame':
        """
        Load annotated text from CouchDB.

        Args:
            spark: SparkSession instance
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
            pattern: Pattern to match attachment names
            collapse_labels: Whether to collapse labels

        Returns:
            DataFrame with annotated data
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other) -> bool:
        """Allow comparison with string for backwards compatibility."""
        if isinstance(other, str):
            return self.name == other
        return isinstance(other, ExtractionMode) and self.name == other.name
