"""CouchDB-aware file reading for annotated documents in PySpark."""

from typing import Any, Dict, Iterator, List, Optional
from pyspark.sql import Row

from line import Line
from fileobj import FileObject


class CouchDBFile(FileObject):
    """
    File-like object that reads from CouchDB attachment content.

    This class extends FileObject to support reading text from CouchDB
    attachments while preserving database metadata (doc_id, attachment_name,
    and database name).
    """

    _doc_id: str
    _attachment_name: str
    _db_name: str
    _human_url: Optional[str]
    _pdf_url: Optional[str]
    _ingest: Optional[Dict[str, Any]]
    _content_lines: List[str]

    def __init__(
        self,
        content: str,
        doc_id: str,
        attachment_name: str,
        db_name: str,
        human_url: Optional[str] = None,
        pdf_url: Optional[str] = None,
        ingest: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize CouchDBFile from attachment content.

        Args:
            content: Text content from CouchDB attachment
            doc_id: CouchDB document ID
            attachment_name: Name of the attachment (e.g., "article.txt.ann")
            db_name: Database name where document is stored (ingest_db_name)
            human_url: Optional human-readable URL from the CouchDB document
            pdf_url: Optional PDF URL from the CouchDB document
            ingest: Optional full ingest document (without _attachments/_rev)
        """
        self._doc_id = doc_id
        self._attachment_name = attachment_name
        self._db_name = db_name
        self._human_url = human_url
        self._pdf_url = pdf_url
        self._ingest = ingest
        self._line_number = 0
        self._page_number = 1
        self._pdf_page = 0  # Will be updated when PDF page markers are encountered
        self._pdf_label = None  # Will be updated when PDF page markers are encountered
        self._empirical_page_number = None
        self._char_offset = 0  # Cumulative character position for span tracking

        # Split content into lines
        self._content_lines = content.split('\n')

    def _get_content_iterator(self) -> Iterator[str]:
        """Get iterator over content lines."""
        return iter(self._content_lines)

    @property
    def filename(self) -> str:
        """
        Return a composite identifier for CouchDB documents.

        Format: db_name/doc_id/attachment_name
        This allows tracking the source of each line.
        """
        return f"{self._db_name}/{self._doc_id}/{self._attachment_name}"

    @property
    def doc_id(self) -> str:
        """CouchDB document ID."""
        return self._doc_id

    @property
    def attachment_name(self) -> str:
        """Attachment filename."""
        return self._attachment_name

    @property
    def db_name(self) -> str:
        """Database name (ingest_db_name)."""
        return self._db_name

    @property
    def human_url(self) -> Optional[str]:
        """Human-readable URL from the CouchDB document."""
        return self._human_url

    @property
    def pdf_url(self) -> Optional[str]:
        """PDF URL from the CouchDB document."""
        return self._pdf_url

    @property
    def ingest(self) -> Optional[Dict[str, Any]]:
        """Full ingest document (without _attachments/_rev)."""
        return self._ingest


def read_couchdb_partition(
    partition: Iterator[Row],
    db_name: str
) -> Iterator[Line]:
    """
    Read annotated files from CouchDB rows in a PySpark partition.

    This is the UDF alternative to read_files() for CouchDB-backed data.
    It processes rows containing CouchDB attachment content and yields
    Line objects that preserve database metadata.

    Args:
        partition: Iterator of PySpark Rows with columns:
            - attachment_name: Attachment filename
            - value: Text content from attachment
            - ingest: Full ingest document containing all metadata:
                - _id: CouchDB document ID
                - url: Human-readable URL
                - pdf_url: PDF URL
                - (other ingest fields)
        db_name: Database name to store in metadata (ingest_db_name)

    Yields:
        Line objects with content and CouchDB metadata (doc_id, human_url,
        pdf_url, attachment_name, db_name, ingest)

    Example:
        >>> # In a PySpark context
        >>> from pyspark.sql.functions import col
        >>> from couchdb_file import read_couchdb_partition
        >>>
        >>> # Assume df has columns: attachment_name, value, ingest
        >>> def process_partition(partition):
        ...     lines = read_couchdb_partition(partition, "mycobank")
        ...     # Process lines with finder.parse_annotated()
        ...     return lines
        >>>
        >>> result = df.rdd.mapPartitions(process_partition)
    """
    for row in partition:
        # Extract ingest dict - all metadata comes from here
        ingest = getattr(row, 'ingest', None) or {}

        # Extract metadata from ingest using canonical field names
        doc_id = ingest.get('_id', 'unknown')
        human_url = ingest.get('url')
        pdf_url = ingest.get('pdf_url')

        # Create CouchDBFile object from row data
        file_obj = CouchDBFile(
            content=row.value,
            doc_id=doc_id,
            attachment_name=row.attachment_name,
            db_name=db_name,
            human_url=human_url,
            pdf_url=pdf_url,
            ingest=ingest
        )

        # Yield all lines from this file
        yield from file_obj.read_line()


def read_couchdb_rows(
    rows: List[Row],
    db_name: str
) -> Iterator[Line]:
    """
    Read annotated files from a list of CouchDB rows.

    This is a convenience function for non-distributed processing or testing.
    For production use with PySpark, use read_couchdb_partition().

    Args:
        rows: List of Rows with columns:
            - doc_id: CouchDB document ID
            - attachment_name: Attachment filename
            - value: Text content from attachment
        db_name: Database name to store in metadata

    Yields:
        Line objects with content and CouchDB metadata

    Example:
        >>> from couchdb_file import read_couchdb_rows
        >>>
        >>> # Collect rows from DataFrame
        >>> rows = df.collect()
        >>>
        >>> # Process all lines
        >>> lines = read_couchdb_rows(rows, "mycobank")
        >>> paragraphs = parse_annotated(lines)
        >>> taxa = group_paragraphs(paragraphs)
    """
    return read_couchdb_partition(iter(rows), db_name)


# Convenience function for integration with CouchDBConnection
def read_couchdb_files_from_connection(
    conn,  # CouchDBConnection
    spark,  # SparkSession
    db_name: str,
    pattern: str = "*.txt.ann"
) -> Iterator[Line]:
    """
    Load and read annotated files from CouchDB using CouchDBConnection.

    This function integrates CouchDBConnection.load_distributed() with
    read_couchdb_rows() to provide a complete pipeline from database to lines.

    Args:
        conn: CouchDBConnection instance
        spark: SparkSession
        db_name: Database name for metadata (ingest_db_name)
        pattern: Pattern for attachment names (default: "*.txt.ann")

    Returns:
        Iterator of Line objects with CouchDB metadata

    Example:
        >>> from skol_classifier.couchdb_io import CouchDBConnection
        >>> from couchdb_file import read_couchdb_files_from_connection
        >>> from finder import parse_annotated
        >>> from taxon import group_paragraphs
        >>>
        >>> # Connect to CouchDB
        >>> conn = CouchDBConnection(
        ...     "http://localhost:5984",
        ...     "mycobank",
        ...     "user",
        ...     "pass"
        ... )
        >>>
        >>> # Load files
        >>> lines = read_couchdb_files_from_connection(
        ...     conn, spark, "mycobank", "*.txt.ann"
        ... )
        >>>
        >>> # Parse and extract taxa
        >>> paragraphs = parse_annotated(lines)
        >>> taxa = group_paragraphs(paragraphs)
    """
    # Load data from CouchDB
    df = conn.load_distributed(spark, pattern)

    # Collect rows (for small datasets) or use in distributed context
    rows = df.collect()

    # Read lines with metadata
    return read_couchdb_rows(rows, db_name)
