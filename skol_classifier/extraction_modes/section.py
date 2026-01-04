"""
Section-level extraction mode implementation.
"""

from typing import List
from pyspark.sql import SparkSession, DataFrame

from .mode import ExtractionMode


class SectionExtractionMode(ExtractionMode):
    """Section-level extraction mode."""

    @property
    def name(self) -> str:
        return 'section'

    def load_raw_from_files(
        self,
        spark: SparkSession,
        file_paths: List[str]
    ) -> DataFrame:
        """
        Section mode does not support loading from files.

        Section extraction requires PDF processing with PDFSectionExtractor,
        which only works with CouchDB attachments.
        """
        raise NotImplementedError(
            "Section-based extraction requires PDFs to be stored in CouchDB. "
            "Please use load_raw_from_couchdb() instead. "
            "The PDFSectionExtractor only supports loading from CouchDB "
            "attachments."
        )

    def load_raw_from_couchdb(
        self,
        spark: SparkSession,
        couchdb_url: str,
        database: str,
        username: str,
        password: str,
        pattern: str = "*.pdf"
    ) -> DataFrame:
        """
        Load sections from PDF documents in CouchDB.

        This uses PDFSectionExtractor to extract sections from PDFs.

        Args:
            spark: SparkSession instance
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
            pattern: Pattern to match PDF attachments (default: *.pdf)

        Returns:
            DataFrame with section data including columns:
            (doc_id, attachment_name, page_number, line_number,
             section_name, value)
        """
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        # Add parent directory to path to import pdf_section_extractor
        parent_dir = Path(__file__).parent.parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from pdf_section_extractor import PDFSectionExtractor

        # Create extractor
        extractor = PDFSectionExtractor(
            spark=spark,
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

        # Load sections from PDFs
        return extractor.extract_sections()

    def load_annotated_from_files(
        self,
        spark: SparkSession,
        file_paths: List[str],
        collapse_labels: bool = True
    ) -> DataFrame:
        """
        Section mode does not support loading annotated data from files.

        Use CouchDB-based loading instead.
        """
        raise NotImplementedError(
            "Section-based extraction requires PDFs to be stored in CouchDB. "
            "Please use load_annotated_from_couchdb() instead."
        )

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
        """
        Load annotated sections from CouchDB.

        For section mode, annotated data should already be in section format
        from PDFSectionExtractor. This method loads pre-segmented annotated
        sections.

        Args:
            spark: SparkSession instance
            couchdb_url: CouchDB server URL
            database: Database name
            username: CouchDB username
            password: CouchDB password
            pattern: Pattern to match attachment names
            collapse_labels: Whether to collapse labels

        Returns:
            DataFrame with annotated section data
        """
        from ..couchdb_io import CouchDBConnection

        conn = CouchDBConnection(
            couchdb_url=couchdb_url,
            database=database,
            username=username,
            password=password
        )

        # Load annotated sections
        df = conn.load_distributed(spark=spark, pattern=pattern)

        # For section mode with pre-segmented data, we expect the data
        # to already have section structure (line_number, section_name, etc.)
        # The AnnotatedTextParser will handle the actual parsing

        return df
