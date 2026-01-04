"""
Data loading module for SKOL classifier.

This module provides backward-compatible wrapper classes for loading
annotated and raw text data. The actual implementations have been moved
to extraction_modes/ to properly separate concerns by extraction mode.

These classes are maintained for backwards compatibility and delegate
to the appropriate extraction mode implementations.
"""

from typing import List
from pyspark.sql import SparkSession, DataFrame

from .extraction_modes import get_mode


class AnnotatedTextLoader:
    """
    Loads annotated text data from various sources.

    This is a backward-compatibility wrapper that delegates to the
    appropriate extraction mode implementation.

    Supports file-based and CouchDB-based loading with paragraph-level
    or line-level extraction.
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the loader.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark

    def load_from_files(
        self,
        file_paths: List[str],
        collapse_labels: bool = True,
        line_level: bool = False
    ) -> DataFrame:
        """
        Load and preprocess annotated data from files.

        Args:
            file_paths: List of paths to annotated files
            collapse_labels: Whether to collapse labels to 3 main categories
            line_level: If True, extract individual lines instead of
                paragraphs

        Returns:
            Preprocessed DataFrame with paragraphs/lines and labels
        """
        mode_name = 'line' if line_level else 'paragraph'
        mode = get_mode(mode_name)
        return mode.load_annotated_from_files(
            spark=self.spark,
            file_paths=file_paths,
            collapse_labels=collapse_labels
        )

    def load_from_couchdb(
        self,
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt.ann",
        collapse_labels: bool = True,
        line_level: bool = False
    ) -> DataFrame:
        """
        Load annotated data from CouchDB.

        Args:
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
            pattern: Pattern to match attachment names
            collapse_labels: Whether to collapse labels
            line_level: If True, extract lines instead of paragraphs

        Returns:
            DataFrame with annotated data
        """
        mode_name = 'line' if line_level else 'paragraph'
        mode = get_mode(mode_name)
        return mode.load_annotated_from_couchdb(
            spark=self.spark,
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password,
            pattern=pattern,
            collapse_labels=collapse_labels
        )


class RawTextLoader:
    """
    Loads raw (unannotated) text data from various sources.

    This is a backward-compatibility wrapper that delegates to the
    appropriate extraction mode implementation.

    Supports file-based and CouchDB-based loading with heuristic
    paragraph extraction or line-level processing.
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize the loader.

        Args:
            spark: SparkSession instance
        """
        self.spark = spark

    def load_from_files(
        self,
        file_paths: List[str],
        line_level: bool = False
    ) -> DataFrame:
        """
        Load and preprocess raw text data from files.

        Args:
            file_paths: List of paths to raw text files
            line_level: If True, process individual lines instead of
                paragraphs

        Returns:
            Preprocessed DataFrame with paragraphs or lines
        """
        mode_name = 'line' if line_level else 'paragraph'
        mode = get_mode(mode_name)
        return mode.load_raw_from_files(
            spark=self.spark,
            file_paths=file_paths
        )

    def load_from_couchdb(
        self,
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.txt",
        line_level: bool = False
    ) -> DataFrame:
        """
        Load raw text data from CouchDB.

        Args:
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
            pattern: Pattern to match attachment names
            line_level: If True, process lines instead of paragraphs

        Returns:
            DataFrame with raw text data
        """
        mode_name = 'line' if line_level else 'paragraph'
        mode = get_mode(mode_name)
        return mode.load_raw_from_couchdb(
            spark=self.spark,
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password,
            pattern=pattern
        )
