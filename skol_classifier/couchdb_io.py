"""
CouchDB I/O utilities for SKOL classifier using distributed foreachPartition
"""

from typing import Any, Dict, List, Optional, Iterator
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    StringType, StructType, StructField, BooleanType, MapType
)
import couchdb


class CouchDBConnection:
    """
    Manages CouchDB connection and provides I/O operations.

    This class encapsulates connection parameters and provides an idempotent
    connection method that can be safely called multiple times.
    """

    # Shared schema definitions (DRY principle)
    # All metadata (doc_id, url, pdf_url) is inside 'ingest' - no redundant columns
    LOAD_SCHEMA = StructType([
        StructField("attachment_name", StringType(), False),
        StructField("value", StringType(), False),
        StructField("ingest", MapType(StringType(), StringType(), valueContainsNull=True), False),
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
        db = self.db

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

    def get_document_list(
        self,
        spark: SparkSession,
        pattern: str = "*.txt"
    ) -> DataFrame:
        """
        Get a list of documents with text attachments from CouchDB.

        This only fetches document metadata (not content) to create a DataFrame
        that can be processed in parallel. For annotation files (*.ann pattern),
        only ONE attachment per document is returned, preferring .pdf.ann over
        .txt.ann when both exist.

        Args:
            spark: SparkSession
            pattern: Pattern for attachment names (e.g., "*.txt", "*.ann")

        Returns:
            DataFrame with columns: doc_id, attachment_name
            One row per (doc_id, attachment_name) pair
        """
        # Connect to CouchDB (driver only)
        db = self.db

        # For *.ann pattern, we need to deduplicate per document
        # Prefer .pdf.ann over .txt.ann (PDF is the canonical source)
        is_ann_pattern = pattern == "*.ann"

        # Get all documents with attachments matching pattern
        # Use dict to deduplicate by doc_id for annotation files
        doc_attachments: dict = {}  # doc_id -> list of matching attachments

        for doc_id in db:
            try:
                doc = db[doc_id]
                attachments = doc.get('_attachments', {})

                # Loop through ALL attachments in the document
                for att_name in attachments.keys():
                    # Check if attachment matches pattern
                    # Pattern matching: "*.txt" matches files ending with .txt
                    matched = False
                    if pattern == "*.txt" and att_name.endswith('.txt'):
                        matched = True
                    elif pattern == "*.*" or pattern == "*":
                        # Match all attachments
                        matched = True
                    elif pattern.startswith("*.") and att_name.endswith(pattern[1:]):
                        # Generic pattern matching for *.ext
                        matched = True

                    if matched:
                        if doc_id not in doc_attachments:
                            doc_attachments[doc_id] = []
                        doc_attachments[doc_id].append(att_name)
            except Exception:
                # Skip documents we can't read
                continue

        # Build final list, deduplicating for *.ann pattern
        doc_list = []
        for doc_id, att_names in doc_attachments.items():
            if is_ann_pattern and len(att_names) > 1:
                # Prefer .pdf.ann over .txt.ann
                pdf_ann = [a for a in att_names if a.endswith('.pdf.ann')]
                txt_ann = [a for a in att_names if a.endswith('.txt.ann')]
                if pdf_ann:
                    doc_list.append((doc_id, pdf_ann[0]))
                elif txt_ann:
                    doc_list.append((doc_id, txt_ann[0]))
                else:
                    # Neither .pdf.ann nor .txt.ann, just pick first
                    doc_list.append((doc_id, att_names[0]))
            else:
                # Single attachment or not *.ann pattern - include all
                for att_name in att_names:
                    doc_list.append((doc_id, att_name))

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
            Rows with attachment_name, value, and ingest.
            The ingest field contains the full ingest document (without _attachments/_rev).
            All metadata (doc_id as _id, url, pdf_url) is accessed from ingest.
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

                            # Create ingest record without attachments and _rev
                            # Convert all values to strings for MapType compatibility
                            ingest_record: Dict[str, Optional[str]] = {}
                            for k, v in doc.items():
                                if k not in ('_attachments', '_rev'):
                                    ingest_record[k] = str(v) if v is not None else None

                            yield Row(
                                attachment_name=row.attachment_name,
                                value=content,
                                ingest=ingest_record
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
                       and optionally human_url
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

                    # Update human_url field if provided
                    if hasattr(row, 'human_url') and row.human_url:
                        doc['url'] = row.human_url
                        db.save(doc)
                        # Reload doc to get updated _rev
                        doc = db[row.doc_id]

                    # Create new attachment name by appending suffix
                    # e.g., "article.txt" becomes "article.txt.ann"
                    new_attachment_name = f"{row.attachment_name}{suffix}"

                    # Save the annotated content as a new attachment
                    db.put_attachment(
                        doc,
                        row.final_aggregated_pg.encode('utf-8'),
                        filename=new_attachment_name,
                        content_type='text/plain; charset=utf-8'
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

        # Extract doc_id and human_url from ingest map for backwards compatibility
        # Many consumers expect these as top-level columns
        from pyspark.sql.functions import col
        result_df = result_df.withColumn("doc_id", col("ingest._id")) \
                             .withColumn("human_url", col("ingest.url"))

        return result_df

    def save_distributed(
        self,
        df: DataFrame,
        suffix: str = ".ann",
        verbosity: int = 1,
        incremental: bool = False
    ) -> DataFrame:
        """
        Save annotated predictions to CouchDB using foreachPartition.

        This function uses mapPartitions where each partition creates a single
        CouchDB connection and reuses it for all rows.

        Args:
            df: DataFrame with columns: doc_id, attachment_name, final_aggregated_pg
            suffix: Suffix to append to attachment names
            verbosity: Logging level (0=none, 1=warnings, 2=info, 3=debug)
            incremental: If True, use immediate writes (crash-resistant but slower)

        Returns:
            DataFrame with doc_id, attachment_name, and success columns
        """
        from .instrumentation import SparkInstrumentation

        # Initialize instrumentation
        instr = SparkInstrumentation(verbosity=verbosity)

        instr.log(2, "\n" + "="*70)
        instr.log(2, "save_distributed: Starting CouchDB save operation")
        if incremental:
            instr.log(2, "  Mode: INCREMENTAL (immediate writes)")
        instr.log(2, "="*70)

        # Analyze input DataFrame
        instr.analyze_dataframe(df, "save_input", count=False)

        # Use mapPartitions for efficient batch saving
        # Create new connection instance with same params for workers
        # IMPORTANT: Keep conn_params minimal to reduce closure size
        conn_params = (self.couchdb_url, self.database, self.username, self.password)

        if incremental:
            # Incremental mode: use foreachPartition for immediate writes
            # This writes each document as soon as it's processed, making the
            # operation crash-resistant and providing real-time progress visibility
            return self._save_incremental(df, suffix, verbosity, conn_params)

        def save_partition(partition):
            # Each worker creates its own connection
            conn = CouchDBConnection(*conn_params)
            return conn.save_partition(partition, suffix)

        # Measure closure size before applying
        instr.measure_closure_size(save_partition, "save_partition")

        instr.log(2, "  Applying mapPartitions...")

        # Apply mapPartitions using shared schema
        result_df = df.rdd.mapPartitions(save_partition).toDF(self.SAVE_SCHEMA)

        instr.log(2, "✓ mapPartitions complete")
        instr.log(2, "="*70 + "\n")

        # Print metrics summary if requested
        if verbosity >= 2:
            print(instr.get_metrics_summary())

        return result_df

    def _save_incremental(
        self,
        df: DataFrame,
        suffix: str,
        verbosity: int,
        conn_params: tuple
    ) -> DataFrame:
        """
        Save predictions incrementally using foreachPartition with immediate writes.

        Each document's .ann attachment is saved immediately after processing,
        making the operation crash-resistant. Progress is visible in real-time.

        Args:
            df: DataFrame with doc_id, attachment_name, final_aggregated_pg columns
            suffix: Suffix to append to attachment names
            verbosity: Logging verbosity
            conn_params: Tuple of (couchdb_url, database, username, password)

        Returns:
            DataFrame with save results (success/failure per document)
        """
        from pyspark.sql import Row
        from pyspark.sql.types import StructType, StructField, StringType, BooleanType
        import sys

        # Accumulator-like tracking via broadcast or print statements
        # Since we can't easily return data from foreachPartition,
        # we'll collect the data first and process iteratively

        # Collect all rows to driver (for incremental processing)
        # This is acceptable because we want immediate, sequential writes
        rows = df.collect()
        total = len(rows)

        if verbosity >= 1:
            print(f"  Processing {total} documents incrementally...")

        # Connect to CouchDB on the driver
        db = self.db
        results = []
        success_count = 0
        failure_count = 0

        for i, row in enumerate(rows):
            success = False
            try:
                doc = db[row.doc_id]

                # Update human_url field if provided
                if hasattr(row, 'human_url') and row.human_url:
                    doc['url'] = row.human_url
                    db.save(doc)
                    doc = db[row.doc_id]

                # Create new attachment name
                new_attachment_name = f"{row.attachment_name}{suffix}"

                # Save immediately
                db.put_attachment(
                    doc,
                    row.final_aggregated_pg.encode('utf-8'),
                    filename=new_attachment_name,
                    content_type='text/plain; charset=utf-8'
                )
                success = True
                success_count += 1

                if verbosity >= 2:
                    print(f"    [{i+1}/{total}] Saved {row.doc_id}/{new_attachment_name}")

            except Exception as e:
                failure_count += 1
                if verbosity >= 1:
                    print(f"    [{i+1}/{total}] ERROR saving {row.doc_id}: {e}")

            results.append(Row(
                doc_id=row.doc_id,
                attachment_name=row.attachment_name,
                success=success
            ))

            # Progress indicator every 10 documents
            if verbosity >= 1 and (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{total} ({success_count} saved, {failure_count} failed)")
                sys.stdout.flush()

        if verbosity >= 1:
            print(f"  ✓ Incremental save complete: {success_count} saved, {failure_count} failed")

        # Convert results back to DataFrame
        spark = df.sparkSession
        return spark.createDataFrame(results, self.SAVE_SCHEMA)

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
                                content_type='text/plain; charset=utf-8'
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
