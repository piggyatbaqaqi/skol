"""
CouchDB I/O utilities for SKOL classifier using distributed UDFs
"""

from typing import Optional, List, Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, lit, struct
from pyspark.sql.types import (
    StringType, StructType, StructField, ArrayType, BooleanType
)
import couchdb


def create_fetch_attachment_udf(
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    pattern: str = "*.txt"
):
    """
    Create a UDF that fetches attachment content from CouchDB.

    This UDF runs distributed across Spark workers, each connecting to CouchDB
    independently to fetch their assigned documents.

    Args:
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password
        pattern: Pattern for attachment names

    Returns:
        UDF function that takes (doc_id, attachment_name) and returns content
    """
    def fetch_attachment(doc_id: str, attachment_name: str) -> Optional[str]:
        """Fetch a single attachment from CouchDB."""
        try:
            # Connect to CouchDB (each worker creates its own connection)
            if username and password:
                server = couchdb.Server(f"{couchdb_url}")
                server.resource.credentials = (username, password)
            else:
                server = couchdb.Server(couchdb_url)

            db = server[database]
            doc = db[doc_id]

            # Get attachment
            if attachment_name in doc.get('_attachments', {}):
                attachment = db.get_attachment(doc, attachment_name)
                if attachment:
                    return attachment.read().decode('utf-8', errors='ignore')

            return None
        except Exception as e:
            # Return error message instead of None for debugging
            return f"ERROR: {str(e)}"

    return udf(fetch_attachment, StringType())


def create_save_attachment_udf(
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    suffix: str = ".ann"
):
    """
    Create a UDF that saves annotated content back to CouchDB.

    This UDF runs distributed across Spark workers, each saving their
    assigned documents independently.

    Args:
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password
        suffix: Suffix to append to attachment names

    Returns:
        UDF function that takes (doc_id, attachment_name, content) and returns success status
    """
    def save_attachment(doc_id: str, attachment_name: str, content: str) -> bool:
        """Save annotated content as a new attachment."""
        try:
            # Connect to CouchDB
            if username and password:
                server = couchdb.Server(couchdb_url)
                server.resource.credentials = (username, password)
            else:
                server = couchdb.Server(couchdb_url)

            db = server[database]
            doc = db[doc_id]

            # Create new attachment name
            new_attachment_name = f"{attachment_name}{suffix}"

            # Save attachment
            db.put_attachment(
                doc,
                content.encode('utf-8'),
                filename=new_attachment_name,
                content_type='text/plain'
            )

            return True
        except Exception as e:
            print(f"Error saving {doc_id}/{attachment_name}: {e}")
            return False

    return udf(save_attachment, BooleanType())


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


def load_from_couchdb_distributed(
    spark: SparkSession,
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    pattern: str = "*.txt"
) -> DataFrame:
    """
    Load text attachments from CouchDB using distributed UDFs.

    This function:
    1. Gets list of documents (on driver)
    2. Creates a DataFrame with doc IDs
    3. Uses UDF to fetch content in parallel across workers

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

    # Create UDF to fetch content
    fetch_udf = create_fetch_attachment_udf(
        couchdb_url, database, username, password, pattern
    )

    # Apply UDF to fetch content in parallel
    result_df = doc_df.withColumn(
        "value",
        fetch_udf(col("doc_id"), col("attachment_name"))
    )

    # Filter out failed fetches
    result_df = result_df.filter(
        (col("value").isNotNull()) &
        (~col("value").startswith("ERROR:"))
    )

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
    Save annotated predictions to CouchDB using distributed UDFs.

    This function uses a UDF that runs on each worker to save attachments
    in parallel.

    Args:
        df: DataFrame with columns: doc_id, attachment_name, final_aggregated_pg
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password
        suffix: Suffix to append to attachment names

    Returns:
        DataFrame with additional 'success' column indicating save status
    """
    # Create UDF to save content
    save_udf = create_save_attachment_udf(
        couchdb_url, database, username, password, suffix
    )

    # Apply UDF to save content in parallel
    result_df = df.withColumn(
        "success",
        save_udf(
            col("doc_id"),
            col("attachment_name"),
            col("final_aggregated_pg")
        )
    )

    return result_df


def create_process_and_save_udf(
    couchdb_url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    suffix: str = ".ann"
):
    """
    Create a UDF that both reads and writes in a single operation.

    This is useful when you want to process and save in one step,
    avoiding the need to pass large content through Spark.

    Args:
        couchdb_url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password
        suffix: Suffix for new attachment

    Returns:
        UDF that takes (doc_id, attachment_name, processed_content) and returns success
    """
    def process_and_save(doc_id: str, attachment_name: str, processed_content: str) -> bool:
        """Read original, combine with processed, and save."""
        try:
            # Connect to CouchDB
            if username and password:
                server = couchdb.Server(couchdb_url)
                server.resource.credentials = (username, password)
            else:
                server = couchdb.Server(couchdb_url)

            db = server[database]
            doc = db[doc_id]

            # Create new attachment name
            new_attachment_name = f"{attachment_name}{suffix}"

            # Save processed content as new attachment
            db.put_attachment(
                doc,
                processed_content.encode('utf-8'),
                filename=new_attachment_name,
                content_type='text/plain'
            )

            return True
        except Exception as e:
            print(f"Error processing {doc_id}/{attachment_name}: {e}")
            return False

    return udf(process_and_save, BooleanType())


# Convenience class for backward compatibility
class CouchDBReader:
    """
    Reader for CouchDB using distributed UDFs.
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
        """Load attachments as Spark DataFrame using distributed UDFs."""
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
    Writer for CouchDB using distributed UDFs.
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
        Save from DataFrame using distributed UDFs.

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
