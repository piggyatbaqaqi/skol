"""
CouchDB I/O utilities for SKOL classifier
"""

from typing import Optional, List, Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StringType, StructType, StructField, ArrayType
import requests
from requests.auth import HTTPBasicAuth


class CouchDBReader:
    """
    Reader for extracting text attachments from CouchDB documents.
    """

    def __init__(
        self,
        url: str,
        database: str,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize CouchDB reader.

        Args:
            url: CouchDB server URL (e.g., "http://localhost:5984")
            database: Database name
            username: Optional username for authentication
            password: Optional password for authentication
        """
        self.url = url.rstrip('/')
        self.database = database
        self.auth = HTTPBasicAuth(username, password) if username else None
        self.db_url = f"{self.url}/{self.database}"

    def get_all_docs(self, include_attachments: bool = True) -> List[Dict[str, Any]]:
        """
        Get all documents from the database.

        Args:
            include_attachments: Include attachment metadata

        Returns:
            List of document dictionaries
        """
        params = {
            'include_docs': 'true',
            'attachments': 'true' if include_attachments else 'false'
        }

        response = requests.get(
            f"{self.db_url}/_all_docs",
            params=params,
            auth=self.auth
        )
        response.raise_for_status()

        data = response.json()
        return [row['doc'] for row in data.get('rows', [])]

    def get_attachment(
        self,
        doc_id: str,
        attachment_name: str
    ) -> bytes:
        """
        Get attachment content from a document.

        Args:
            doc_id: Document ID
            attachment_name: Name of the attachment

        Returns:
            Attachment content as bytes
        """
        url = f"{self.db_url}/{doc_id}/{attachment_name}"
        response = requests.get(url, auth=self.auth)
        response.raise_for_status()
        return response.content

    def get_text_attachments(
        self,
        pattern: str = "*.txt"
    ) -> List[Dict[str, Any]]:
        """
        Get all .txt attachments from all documents.

        Args:
            pattern: Pattern for attachment names (default: "*.txt")

        Returns:
            List of dictionaries with doc_id, attachment_name, and content
        """
        docs = self.get_all_docs()
        attachments = []

        for doc in docs:
            doc_id = doc.get('_id', '')
            doc_attachments = doc.get('_attachments', {})

            for att_name, att_meta in doc_attachments.items():
                # Check if attachment matches pattern
                if pattern == "*.txt" and att_name.endswith('.txt'):
                    # Get attachment content
                    content = self.get_attachment(doc_id, att_name)

                    attachments.append({
                        'doc_id': doc_id,
                        'attachment_name': att_name,
                        'content': content.decode('utf-8', errors='ignore')
                    })

        return attachments

    def to_spark_dataframe(
        self,
        spark: SparkSession,
        pattern: str = "*.txt"
    ) -> DataFrame:
        """
        Load text attachments as a Spark DataFrame.

        Args:
            spark: SparkSession
            pattern: Pattern for attachment names

        Returns:
            DataFrame with columns: doc_id, attachment_name, value
        """
        attachments = self.get_text_attachments(pattern)

        # Convert to DataFrame
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("value", StringType(), False)
        ])

        # Create list of rows
        rows = [(att['doc_id'], att['attachment_name'], att['content'])
                for att in attachments]

        return spark.createDataFrame(rows, schema)


class CouchDBWriter:
    """
    Writer for saving annotated text back to CouchDB as attachments.
    """

    def __init__(
        self,
        url: str,
        database: str,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize CouchDB writer.

        Args:
            url: CouchDB server URL (e.g., "http://localhost:5984")
            database: Database name
            username: Optional username for authentication
            password: Optional password for authentication
        """
        self.url = url.rstrip('/')
        self.database = database
        self.auth = HTTPBasicAuth(username, password) if username else None
        self.db_url = f"{self.url}/{self.database}"

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dictionary
        """
        response = requests.get(f"{self.db_url}/{doc_id}", auth=self.auth)
        response.raise_for_status()
        return response.json()

    def put_attachment(
        self,
        doc_id: str,
        attachment_name: str,
        content: str,
        content_type: str = "text/plain"
    ) -> Dict[str, Any]:
        """
        Add or update an attachment to a document.

        Args:
            doc_id: Document ID
            attachment_name: Name for the attachment
            content: Attachment content (string)
            content_type: MIME type (default: "text/plain")

        Returns:
            Response dictionary from CouchDB
        """
        # Get current document to get _rev
        doc = self.get_document(doc_id)
        rev = doc['_rev']

        # Upload attachment
        url = f"{self.db_url}/{doc_id}/{attachment_name}"
        headers = {'Content-Type': content_type}

        response = requests.put(
            url,
            data=content.encode('utf-8'),
            headers=headers,
            params={'rev': rev},
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()

    def save_annotated_predictions(
        self,
        predictions: List[Tuple[str, str, str]],
        suffix: str = ".ann"
    ) -> List[Dict[str, Any]]:
        """
        Save annotated predictions back to CouchDB.

        Args:
            predictions: List of (doc_id, attachment_name, annotated_content) tuples
            suffix: Suffix to append to attachment names (default: ".ann")

        Returns:
            List of response dictionaries from CouchDB
        """
        results = []

        for doc_id, attachment_name, content in predictions:
            # Create new attachment name
            new_attachment_name = f"{attachment_name}{suffix}"

            try:
                result = self.put_attachment(
                    doc_id,
                    new_attachment_name,
                    content
                )
                results.append({
                    'doc_id': doc_id,
                    'attachment_name': new_attachment_name,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'doc_id': doc_id,
                    'attachment_name': new_attachment_name,
                    'success': False,
                    'error': str(e)
                })

        return results

    def save_from_dataframe(
        self,
        df: DataFrame,
        suffix: str = ".ann"
    ) -> List[Dict[str, Any]]:
        """
        Save annotated predictions from a Spark DataFrame to CouchDB.

        Args:
            df: DataFrame with columns: doc_id, attachment_name, final_aggregated_pg
            suffix: Suffix to append to attachment names

        Returns:
            List of results
        """
        # Collect DataFrame to driver
        rows = df.select("doc_id", "attachment_name", "final_aggregated_pg").collect()

        # Convert to list of tuples
        predictions = [
            (row.doc_id, row.attachment_name, row.final_aggregated_pg)
            for row in rows
        ]

        return self.save_annotated_predictions(predictions, suffix)


def create_couchdb_reader(
    url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> CouchDBReader:
    """
    Factory function to create a CouchDB reader.

    Args:
        url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password

    Returns:
        CouchDBReader instance
    """
    return CouchDBReader(url, database, username, password)


def create_couchdb_writer(
    url: str,
    database: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> CouchDBWriter:
    """
    Factory function to create a CouchDB writer.

    Args:
        url: CouchDB server URL
        database: Database name
        username: Optional username
        password: Optional password

    Returns:
        CouchDBWriter instance
    """
    return CouchDBWriter(url, database, username, password)
