"""
CouchDB I/O utilities for SKOL classifier using distributed foreachPartition
"""

from typing import Optional, List, Dict, Any, Iterator
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    StringType, StructType, StructField, BooleanType
)
import couchdb


def get_document_list(
    spark: SparkSession,
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    pattern: str = "*.txt"
) -> DataFrame:
    """
    Get a list of documents with text attachments from CouchDB.

    This only fetches document metadata (not content) to create a DataFrame
    that can be processed in parallel.

    Args:
        spark: SparkSession
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password
        pattern: Pattern for attachment names

    Returns:
        DataFrame with columns: doc_id, attachment_name
    """
    # Connect to CouchDB (driver only)
    if username and password:
        server = couchdb.Server(couchdb_url)
        server.resource.credentials = (username, password)
    else:
        server = couchdb.Server(couchdb_url)

    db = server[database]

    # Get all documents with attachments matching pattern
    doc_list = []
    for doc_id in db:
        try:
            doc = db[doc_id]
            attachments = doc.get('_attachments', {})

            for att_name in attachments.keys():
                # Check if attachment matches pattern
                if pattern == "*.txt" and att_name.endswith('.txt'):
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


def fetch_partition_from_couchdb(
    partition: Iterator[Row],
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> Iterator[Row]:
    """
    Fetch CouchDB attachments for an entire partition.

    This function is designed to be used with foreachPartition or mapPartitions.
    It creates a single CouchDB connection per partition and reuses it for all rows.

    Args:
        partition: Iterator of Rows with doc_id and attachment_name
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password

    Yields:
        Rows with doc_id, attachment_name, and value (content)
    """
    # Connect to CouchDB once per partition
    try:
        if username and password:
            server = couchdb.Server(couchdb_url)
            server.resource.credentials = (username, password)
        else:
            server = couchdb.Server(couchdb_url)

        db = server[database]

        # Process all rows in partition with same connection
        for row in partition:
            try:
                doc = db[row.doc_id]

                # Get attachment
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


def save_partition_to_couchdb(
    partition: Iterator[Row],
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    suffix: str = ".ann"
) -> Iterator[Row]:
    """
    Save annotated content to CouchDB for an entire partition.

    This function is designed to be used with foreachPartition or mapPartitions.
    It creates a single CouchDB connection per partition and reuses it for all rows.

    Args:
        partition: Iterator of Rows with doc_id, attachment_name, final_aggregated_pg
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password
        suffix: Suffix to append to attachment names

    Yields:
        Rows with doc_id, attachment_name, and success status
    """
    # Connect to CouchDB once per partition
    try:
        if username and password:
            server = couchdb.Server(couchdb_url)
            server.resource.credentials = (username, password)
        else:
            server = couchdb.Server(couchdb_url)

        db = server[database]

        # Process all rows in partition with same connection
        for row in partition:
            success = False
            try:
                doc = db[row.doc_id]

                # Create new attachment name
                new_attachment_name = f"{row.attachment_name}{suffix}"

                # Save attachment
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


def load_from_couchdb_distributed(
    spark: SparkSession,
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
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
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password
        pattern: Pattern for attachment names

    Returns:
        DataFrame with columns: doc_id, attachment_name, value
    """
    # Get document list
    doc_df = get_document_list(
        spark, couchdb_url, database, username, password, pattern
    )

    # Use mapPartitions for efficient batch fetching
    def fetch_partition(partition):
        return fetch_partition_from_couchdb(
            partition, couchdb_url, database, username, password
        )

    # Define output schema
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("value", StringType(), False)
    ])

    # Apply mapPartitions
    result_df = doc_df.rdd.mapPartitions(fetch_partition).toDF(schema)

    return result_df


def save_to_couchdb_distributed(
    df: DataFrame,
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    suffix: str = ".ann"
) -> DataFrame:
    """
    Save annotated predictions to CouchDB using foreachPartition.

    This function uses mapPartitions where each partition creates a single
    CouchDB connection and reuses it for all rows.

    Args:
        df: DataFrame with columns: doc_id, attachment_name, final_aggregated_pg
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password
        suffix: Suffix to append to attachment names

    Returns:
        DataFrame with doc_id, attachment_name, and success columns
    """
    # Use mapPartitions for efficient batch saving
    def save_partition(partition):
        return save_partition_to_couchdb(
            partition, couchdb_url, database, username, password, suffix
        )

    # Define output schema
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("success", BooleanType(), False)
    ])

    # Apply mapPartitions
    result_df = df.rdd.mapPartitions(save_partition).toDF(schema)

    return result_df


def process_partition_with_couchdb(
    partition: Iterator[Row],
    couchdb_url: str,
    database: str,
    processor_func,
    username: Optional[str] = None,
    password: Optional[str] = None,
    suffix: str = ".ann"
) -> Iterator[Row]:
    """
    Generic function to read, process, and save in one partition operation.

    This allows custom processing logic while maintaining single connection per partition.

    Args:
        partition: Iterator of Rows
        couchdb_url: CouchDB server URL
        database: Database name
        processor_func: Function to process content (takes content string, returns processed string)
        username: Optional username
        password: Optional password
        suffix: Suffix for output attachment

    Yields:
        Rows with processing results
    """
    try:
        if username and password:
            server = couchdb.Server(couchdb_url)
            server.resource.credentials = (username, password)
        else:
            server = couchdb.Server(couchdb_url)

        db = server[database]

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


# Convenience classes for backward compatibility
class CouchDBReader:
    """
    Reader for CouchDB using distributed foreachPartition.
    """

    def __init__(
        self,
        url: str,
        database: str,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """Initialize CouchDB reader."""
        self.url = url
        self.database = database
        self.username = username
        self.password = password

    def to_spark_dataframe(
        self,
        spark: SparkSession,
        pattern: str = "*.txt"
    ) -> DataFrame:
        """Load attachments as Spark DataFrame using foreachPartition."""
        return load_from_couchdb_distributed(
            spark,
            self.url,
            self.database,
            self.username,
            self.password,
            pattern
        )


class CouchDBWriter:
    """
    Writer for CouchDB using distributed foreachPartition.
    """

    def __init__(
        self,
        url: str,
        database: str,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """Initialize CouchDB writer."""
        self.url = url
        self.database = database
        self.username = username
        self.password = password

    def save_from_dataframe(
        self,
        df: DataFrame,
        suffix: str = ".ann"
    ) -> List[Dict[str, Any]]:
        """
        Save from DataFrame using foreachPartition.

        Returns list of results (for compatibility).
        """
        result_df = save_to_couchdb_distributed(
            df,
            self.url,
            self.database,
            self.username,
            self.password,
            suffix
        )

        # Collect results
        results = []
        for row in result_df.collect():
            results.append({
                'doc_id': row.doc_id,
                'attachment_name': f"{row.attachment_name}{suffix}",
                'success': row.success
            })

        return results


def create_couchdb_reader(
    url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> CouchDBReader:
    """Factory function to create a CouchDB reader."""
    return CouchDBReader(url, database, username, password)


def create_couchdb_writer(
    url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> CouchDBWriter:
    """Factory function to create a CouchDB writer."""
    return CouchDBWriter(url, database, username, password)
