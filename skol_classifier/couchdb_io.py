"""
CouchDB I/O utilities for SKOL classifier using distributed foreachPartition
"""

from typing import List, Optional, Iterator
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    StringType, StructType, StructField, BooleanType
)
import couchdb


class CouchDBConnection:
    """
    Manages CouchDB connection and provides I/O operations.

    This class encapsulates connection parameters and provides an idempotent
    connection method that can be safely called multiple times.
    """

    # Shared schema definitions (DRY principle)
    LOAD_SCHEMA = StructType([
        StructField("doc_id", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("value", StringType(), False),
    ])

    SAVE_SCHEMA = StructType([
        StructField("doc_id", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("success", BooleanType(), False),
    ])

    def __init__(
        self,
        couchdb_url: str,
        database: str,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize CouchDB connection parameters.

        Args:
            couchdb_url: CouchDB server URL (e.g., "http://localhost:5984")
            database: Database name
            username: Optional username for authentication
            password: Optional password for authentication
        """
        self.couchdb_url = couchdb_url
        self.database = database
        self.username = username
        self.password = password
        self._server = None
        self._db = None

    def _connect(self):
        """
        Idempotent connection method that returns a CouchDB server object.

        This method can be called multiple times safely - it will only create
        a connection if one doesn't already exist.

        Returns:
            couchdb.Server: Connected CouchDB server object
        """
        if self._server is None:
            self._server = couchdb.Server(self.couchdb_url)
            if self.username and self.password:
                self._server.resource.credentials = (self.username, self.password)

        if self._db is None:
            self._db = self._server[self.database]

        return self._server

    @property
    def db(self):
        """Get the database object, connecting if necessary."""
        if self._db is None:
            self._connect()
        return self._db

    def get_all_doc_ids(self, pattern: str = "*") -> List[str]:
        """
        Get list of document IDs matching the pattern from CouchDB.

        Args:
            pattern: Pattern for document IDs (e.g., "taxon_*", "*")
                    - "*" matches all non-design documents
                    - "prefix*" matches documents starting with prefix
                    - "exact" matches exactly

        Returns:
            List of matching document IDs
        """
        print("DEBUG: Before the try.")
        # try:
        db = self.db
        print("DEBUG: Connected to CouchDB database:", self.database)
        print("DEBUG: Total documents in DB:", len(db))
        print("DEBUG: Fetching document IDs with pattern:", pattern)
        print("DEBUG: Sample document IDs:", list(db)[:10])

        # Get all document IDs (excluding design documents)
        all_doc_ids = [doc_id for doc_id in list(db) if not doc_id.startswith('_design/')]

        # Filter by pattern
        if pattern == "*":
            # Return all non-design documents
            return all_doc_ids
        elif pattern.endswith('*'):
            # Prefix matching
            prefix = pattern[:-1]
            return [doc_id for doc_id in all_doc_ids if doc_id.startswith(prefix)]
        else:
            # Exact match
            return [doc_id for doc_id in all_doc_ids if doc_id == pattern]

        # except Exception as e:
        #     print(f"Error getting document IDs from CouchDB: {e}")
        #     return []

    def get_document_list(
        self,
        spark: SparkSession,
        pattern: str = "*.txt"
    ) -> DataFrame:
        """
        Get a list of documents with text attachments from CouchDB.

        This only fetches document metadata (not content) to create a DataFrame
        that can be processed in parallel. Creates ONE ROW per attachment, so if
        a document has multiple attachments matching the pattern, it will have
        multiple rows in the resulting DataFrame.

        Args:
            spark: SparkSession
            pattern: Pattern for attachment names (e.g., "*.txt")

        Returns:
            DataFrame with columns: doc_id, attachment_name
            One row per (doc_id, attachment_name) pair
        """
        # Connect to CouchDB (driver only)
        db = self.db

        # Get all documents with attachments matching pattern
        doc_list = []
        for doc_id in db:
            try:
                doc = db[doc_id]
                attachments = doc.get('_attachments', {})

                # Loop through ALL attachments in the document
                for att_name in attachments.keys():
                    # Check if attachment matches pattern
                    # Pattern matching: "*.txt" matches files ending with .txt
                    if pattern == "*.txt" and att_name.endswith('.txt'):
                        doc_list.append((doc_id, att_name))
                    elif pattern == "*.*" or pattern == "*":
                        # Match all attachments
                        doc_list.append((doc_id, att_name))
                    elif pattern.startswith("*.") and att_name.endswith(pattern[1:]):
                        # Generic pattern matching for *.ext
                        doc_list.append((doc_id, att_name))
            except Exception:
                # Skip documents we can't read
                continue

        # Create DataFrame with document IDs and attachment names
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False)
        ])

        return spark.createDataFrame(doc_list, schema)

    def fetch_partition(
        self,
        partition: Iterator[Row]
    ) -> Iterator[Row]:
        """
        Fetch CouchDB attachments for an entire partition.

        This function is designed to be used with foreachPartition or mapPartitions.
        It creates a single CouchDB connection per partition and reuses it for all rows.

        Args:
            partition: Iterator of Rows with doc_id and attachment_name

        Yields:
            Rows with doc_id, attachment_name, and value (content).
        """
        # Connect to CouchDB once per partition
        try:
            db = self.db

            # Process all rows in partition with same connection
            # Note: Each row represents one (doc_id, attachment_name) pair
            # If a document has multiple .txt attachments, there will be multiple rows
            for row in partition:
                try:
                    doc = db[row.doc_id]

                    # Get the specific attachment for this row
                    if row.attachment_name in doc.get('_attachments', {}):
                        attachment = db.get_attachment(doc, row.attachment_name)
                        if attachment:
                            content = attachment.read().decode('utf-8', errors='ignore')

                            yield Row(
                                doc_id=row.doc_id,
                                attachment_name=row.attachment_name,
                                value=content
                            )
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error fetching {row.doc_id}/{row.attachment_name}: {e}")
                    continue

        except Exception as e:
            print(f"Error connecting to CouchDB: {e}")
            return

    def save_partition(
        self,
        partition: Iterator[Row],
        suffix: str = ".ann"
    ) -> Iterator[Row]:
        """
        Save annotated content to CouchDB for an entire partition.

        This function is designed to be used with foreachPartition or mapPartitions.
        It creates a single CouchDB connection per partition and reuses it for all rows.

        Args:
            partition: Iterator of Rows with doc_id, attachment_name, final_aggregated_pg
            suffix: Suffix to append to attachment names

        Yields:
            Rows with doc_id, attachment_name, and success status.
        """
        # Connect to CouchDB once per partition
        try:
            db = self.db

            # Process all rows in partition with same connection
            # Note: Each row represents one (doc_id, attachment_name) pair
            # If a document had multiple .txt files, we save multiple .ann files
            for row in partition:
                success = False
                try:
                    doc = db[row.doc_id]

                    # Create new attachment name by appending suffix
                    # e.g., "article.txt" becomes "article.txt.ann"
                    new_attachment_name = f"{row.attachment_name}{suffix}"

                    # Save the annotated content as a new attachment
                    db.put_attachment(
                        doc,
                        row.final_aggregated_pg.encode('utf-8'),
                        filename=new_attachment_name,
                        content_type='text/plain'
                    )

                    success = True

                except Exception as e:
                    print(f"Error saving {row.doc_id}/{row.attachment_name}: {e}")

                yield Row(
                    doc_id=row.doc_id,
                    attachment_name=row.attachment_name,
                    success=success
                )

        except Exception as e:
            print(f"Error connecting to CouchDB: {e}")
            # Yield failures for all rows
            for row in partition:
                yield Row(
                    doc_id=row.doc_id,
                    attachment_name=row.attachment_name,
                    success=False
                )

    def load_distributed(
        self,
        spark: SparkSession,
        pattern: str = "*.txt"
    ) -> DataFrame:
        """
        Load text attachments from CouchDB using foreachPartition.

        This function:
        1. Gets list of documents (on driver)
        2. Creates a DataFrame with doc IDs
        3. Uses mapPartitions to fetch content efficiently (one connection per partition)

        Args:
            spark: SparkSession
            pattern: Pattern for attachment names

        Returns:
            DataFrame with columns: doc_id, attachment_name, and value.
        """
        # Get document list
        doc_df = self.get_document_list(spark, pattern)

        # Use mapPartitions for efficient batch fetching
        # Create new connection instance with same params for workers
        conn_params = (self.couchdb_url, self.database, self.username, self.password)

        def fetch_partition(partition):
            # Each worker creates its own connection
            conn = CouchDBConnection(*conn_params)
            return conn.fetch_partition(partition)

        # Apply mapPartitions using shared schema
        result_df = doc_df.rdd.mapPartitions(fetch_partition).toDF(self.LOAD_SCHEMA)

        return result_df

    def save_distributed(
        self,
        df: DataFrame,
        suffix: str = ".ann"
    ) -> DataFrame:
        """
        Save annotated predictions to CouchDB using foreachPartition.

        This function uses mapPartitions where each partition creates a single
        CouchDB connection and reuses it for all rows.

        Args:
            df: DataFrame with columns: doc_id, attachment_name, final_aggregated_pg
            suffix: Suffix to append to attachment names

        Returns:
            DataFrame with doc_id, attachment_name, and success columns
        """
        # Use mapPartitions for efficient batch saving
        # Create new connection instance with same params for workers
        conn_params = (self.couchdb_url, self.database, self.username, self.password)

        def save_partition(partition):
            # Each worker creates its own connection
            conn = CouchDBConnection(*conn_params)
            return conn.save_partition(partition, suffix)

        # Apply mapPartitions using shared schema
        result_df = df.rdd.mapPartitions(save_partition).toDF(self.SAVE_SCHEMA)

        return result_df

    def process_partition_with_func(
        self,
        partition: Iterator[Row],
        processor_func,
        suffix: str = ".ann"
    ) -> Iterator[Row]:
        """
        Generic function to read, process, and save in one partition operation.

        This allows custom processing logic while maintaining single connection per partition.

        Args:
            partition: Iterator of Rows
            processor_func: Function to process content (takes content string, returns processed string)
            suffix: Suffix for output attachment

        Yields:
            Rows with processing results, including success status for logging.
        """
        try:
            db = self.db

            for row in partition:
                try:
                    doc = db[row.doc_id]

                    # Fetch
                    if row.attachment_name in doc.get('_attachments', {}):
                        attachment = db.get_attachment(doc, row.attachment_name)
                        if attachment:
                            content = attachment.read().decode('utf-8', errors='ignore')

                            # Process
                            processed = processor_func(content)

                            # Save
                            new_attachment_name = f"{row.attachment_name}{suffix}"
                            db.put_attachment(
                                doc,
                                processed.encode('utf-8'),
                                filename=new_attachment_name,
                                content_type='text/plain'
                            )

                            yield Row(
                                doc_id=row.doc_id,
                                attachment_name=row.attachment_name,
                                success=True
                            )
                            continue

                except Exception as e:
                    print(f"Error processing {row.doc_id}/{row.attachment_name}: {e}")

                yield Row(
                    doc_id=row.doc_id,
                    attachment_name=row.attachment_name,
                    success=False
                )

        except Exception as e:
            print(f"Error connecting to CouchDB: {e}")
            for row in partition:
                yield Row(
                    doc_id=row.doc_id,
                    attachment_name=row.attachment_name,
                    success=False
                )
