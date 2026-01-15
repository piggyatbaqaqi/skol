"""
Tests for couchdb_io.py module.

Run with: pytest skol_classifier/couchdb_io_test.py -v
"""

from unittest.mock import MagicMock, patch
import pytest
from pyspark.sql import SparkSession, Row

from .couchdb_io import CouchDBConnection

# Module path for patching - couchdb is imported directly in couchdb_io
COUCHDB_MODULE_PATH = 'skol_classifier.couchdb_io.couchdb'


# Fixtures
@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    session = SparkSession.builder \
        .appName("CouchDBIOTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def mock_server():
    """Create a mock CouchDB server."""
    server = MagicMock()
    server.resource.credentials = None
    return server


@pytest.fixture
def mock_db():
    """Create a mock CouchDB database."""
    db = MagicMock()
    return db


class TestCouchDBConnectionInit:
    """Tests for CouchDBConnection initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        assert conn.couchdb_url == "http://localhost:5984"
        assert conn.database == "test_db"
        assert conn.username is None
        assert conn.password is None
        assert conn._server is None
        assert conn._db is None

    def test_init_with_credentials(self):
        """Test initialization with credentials."""
        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db",
            username="admin",
            password="secret"
        )

        assert conn.username == "admin"
        assert conn.password == "secret"


class TestCouchDBConnectionConnect:
    """Tests for CouchDBConnection._connect method."""

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_connect_without_credentials(self, mock_server_class):
        """Test connection without credentials."""
        mock_server = MagicMock()
        mock_db = MagicMock()
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        conn._connect()

        mock_server_class.assert_called_once_with("http://localhost:5984")
        assert conn._server is mock_server

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_connect_with_credentials(self, mock_server_class):
        """Test connection with credentials."""
        mock_server = MagicMock()
        mock_server.resource = MagicMock()
        mock_db = MagicMock()
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db",
            username="admin",
            password="secret"
        )

        conn._connect()

        assert mock_server.resource.credentials == ("admin", "secret")

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_connect_idempotent(self, mock_server_class):
        """Test that multiple connects don't create multiple servers."""
        mock_server = MagicMock()
        mock_db = MagicMock()
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        # Call connect multiple times
        conn._connect()
        conn._connect()
        conn._connect()

        # Server should only be created once
        mock_server_class.assert_called_once()


class TestCouchDBConnectionDbProperty:
    """Tests for CouchDBConnection.db property."""

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_db_property_connects_if_needed(self, mock_server_class):
        """Test that db property triggers connection if not connected."""
        mock_server = MagicMock()
        mock_db = MagicMock()
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        # Access db property
        db = conn.db

        assert db is mock_db
        mock_server_class.assert_called_once()


class TestCouchDBConnectionGetAllDocIds:
    """Tests for CouchDBConnection.get_all_doc_ids method."""

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_get_all_doc_ids_all(self, mock_server_class):
        """Test getting all document IDs."""
        mock_server = MagicMock()
        mock_db = MagicMock()
        mock_db.__iter__ = MagicMock(return_value=iter([
            "doc1", "doc2", "_design/test", "doc3"
        ]))
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        result = conn.get_all_doc_ids("*")

        # Should exclude design documents
        assert "doc1" in result
        assert "doc2" in result
        assert "doc3" in result
        assert "_design/test" not in result

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_get_all_doc_ids_prefix(self, mock_server_class):
        """Test getting document IDs with prefix pattern."""
        mock_server = MagicMock()
        mock_db = MagicMock()
        mock_db.__iter__ = MagicMock(return_value=iter([
            "taxon_1", "taxon_2", "other_doc"
        ]))
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        result = conn.get_all_doc_ids("taxon_*")

        assert "taxon_1" in result
        assert "taxon_2" in result
        assert "other_doc" not in result

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_get_all_doc_ids_exact(self, mock_server_class):
        """Test getting document ID with exact match."""
        mock_server = MagicMock()
        mock_db = MagicMock()
        mock_db.__iter__ = MagicMock(return_value=iter([
            "doc1", "doc2", "exact_doc"
        ]))
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        result = conn.get_all_doc_ids("exact_doc")

        assert result == ["exact_doc"]


class TestCouchDBConnectionGetDocumentList:
    """Tests for CouchDBConnection.get_document_list method."""

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_get_document_list_txt_pattern(self, mock_server_class, spark):
        """Test getting document list with .txt pattern."""
        mock_server = MagicMock()
        mock_db = MagicMock()

        # Create mock documents with attachments
        mock_doc1 = {
            "_attachments": {
                "article.txt": {},
                "image.png": {}
            }
        }
        mock_doc2 = {
            "_attachments": {
                "paper.txt": {}
            }
        }
        mock_doc3 = {}  # No attachments

        def mock_getitem(doc_id):
            if doc_id == "doc1":
                return mock_doc1
            elif doc_id == "doc2":
                return mock_doc2
            else:
                return mock_doc3

        mock_db.__iter__ = MagicMock(return_value=iter(["doc1", "doc2", "doc3"]))
        mock_db.__getitem__ = mock_getitem
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        result = conn.get_document_list(spark, "*.txt")
        rows = result.collect()

        # Should find article.txt and paper.txt
        assert len(rows) == 2
        doc_attachments = [(r.doc_id, r.attachment_name) for r in rows]
        assert ("doc1", "article.txt") in doc_attachments
        assert ("doc2", "paper.txt") in doc_attachments


class TestCouchDBConnectionFetchPartition:
    """Tests for CouchDBConnection.fetch_partition method."""

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_fetch_partition_basic(self, mock_server_class):
        """Test basic partition fetching."""
        mock_server = MagicMock()
        mock_db = MagicMock()

        # Create mock document
        mock_doc = {
            "url": "http://example.com",
            "pdf_url": "http://example.com/doc.pdf",
            "_attachments": {
                "article.txt": {}
            }
        }

        # Create mock attachment
        mock_attachment = MagicMock()
        mock_attachment.read.return_value = b"This is the content."

        mock_db.__getitem__ = MagicMock(return_value=mock_doc)
        mock_db.get_attachment = MagicMock(return_value=mock_attachment)
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        # Create partition iterator
        partition = iter([
            Row(doc_id="doc1", attachment_name="article.txt")
        ])

        results = list(conn.fetch_partition(partition))

        assert len(results) == 1
        assert results[0].doc_id == "doc1"
        assert results[0].value == "This is the content."
        assert results[0].human_url == "http://example.com"


class TestCouchDBConnectionSavePartition:
    """Tests for CouchDBConnection.save_partition method."""

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_save_partition_basic(self, mock_server_class):
        """Test basic partition saving."""
        mock_server = MagicMock()
        mock_db = MagicMock()

        mock_doc = {"_id": "doc1", "_rev": "1-abc"}
        mock_db.__getitem__ = MagicMock(return_value=mock_doc)
        mock_db.save = MagicMock()
        mock_db.put_attachment = MagicMock()
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        # Create partition iterator with annotated content
        partition = iter([
            Row(
                doc_id="doc1",
                attachment_name="article.txt",
                final_aggregated_pg="Annotated content here"
            )
        ])

        results = list(conn.save_partition(partition, suffix=".ann"))

        assert len(results) == 1
        assert results[0].doc_id == "doc1"
        assert results[0].success is True

        # Verify put_attachment was called
        mock_db.put_attachment.assert_called_once()
        call_args = mock_db.put_attachment.call_args
        assert call_args[1]["filename"] == "article.txt.ann"


class TestCouchDBConnectionSchemas:
    """Tests for CouchDBConnection schema definitions."""

    def test_load_schema(self):
        """Test LOAD_SCHEMA has expected fields."""
        schema = CouchDBConnection.LOAD_SCHEMA

        field_names = [f.name for f in schema.fields]
        assert "doc_id" in field_names
        assert "human_url" in field_names
        assert "pdf_url" in field_names
        assert "attachment_name" in field_names
        assert "value" in field_names

    def test_save_schema(self):
        """Test SAVE_SCHEMA has expected fields."""
        schema = CouchDBConnection.SAVE_SCHEMA

        field_names = [f.name for f in schema.fields]
        assert "doc_id" in field_names
        assert "attachment_name" in field_names
        assert "success" in field_names


class TestCouchDBConnectionProcessPartition:
    """Tests for CouchDBConnection.process_partition_with_func method."""

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_process_partition_with_func(self, mock_server_class):
        """Test processing partition with custom function."""
        mock_server = MagicMock()
        mock_db = MagicMock()

        mock_doc = {
            "_attachments": {"article.txt": {}}
        }
        mock_attachment = MagicMock()
        mock_attachment.read.return_value = b"Original content"

        mock_db.__getitem__ = MagicMock(return_value=mock_doc)
        mock_db.get_attachment = MagicMock(return_value=mock_attachment)
        mock_db.put_attachment = MagicMock()
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        # Define processor function
        def processor(content):
            return content.upper()

        # Create partition
        partition = iter([
            Row(doc_id="doc1", attachment_name="article.txt")
        ])

        results = list(conn.process_partition_with_func(
            partition, processor, suffix=".processed"
        ))

        assert len(results) == 1
        assert results[0].success is True

        # Verify put_attachment was called with processed content
        call_args = mock_db.put_attachment.call_args
        assert call_args[0][1] == b"ORIGINAL CONTENT"
        assert call_args[1]["filename"] == "article.txt.processed"


class TestCouchDBConnectionDistributedMethods:
    """Tests for load_distributed and save_distributed methods."""

    @patch(COUCHDB_MODULE_PATH + '.Server')
    def test_load_distributed(self, mock_server_class, spark):
        """Test load_distributed method structure."""
        mock_server = MagicMock()
        mock_db = MagicMock()
        mock_db.__iter__ = MagicMock(return_value=iter([]))
        mock_server.__getitem__ = MagicMock(return_value=mock_db)
        mock_server_class.return_value = mock_server

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )

        # With empty database, should return empty DataFrame
        result = conn.load_distributed(spark, "*.txt")

        assert result.count() == 0
        # Should have expected columns
        assert set(result.columns) == {
            "doc_id", "human_url", "pdf_url", "attachment_name", "value"
        }

    @patch('skol_classifier.couchdb_io.SparkInstrumentation')
    @patch.object(CouchDBConnection, 'save_partition')
    def test_save_distributed_creates_result_df(
        self, mock_save_partition, mock_instr_class, spark
    ):
        """Test that save_distributed creates proper result DataFrame."""
        # Mock instrumentation to avoid toDebugString issues
        mock_instr = MagicMock()
        mock_instr_class.return_value = mock_instr

        # Create mock save_partition that yields success results
        def mock_save(partition, suffix):
            for row in partition:
                yield Row(
                    doc_id=row.doc_id,
                    attachment_name=row.attachment_name,
                    success=True
                )

        mock_save_partition.side_effect = mock_save

        conn = CouchDBConnection(
            couchdb_url="http://localhost:5984",
            database="test_db"
        )
        # Pre-connect to avoid actual connection
        conn._server = MagicMock()
        conn._db = MagicMock()

        # Create test DataFrame
        data = [
            Row(
                doc_id="doc1",
                attachment_name="article.txt",
                final_aggregated_pg="Annotated content"
            )
        ]
        df = spark.createDataFrame(data)

        result = conn.save_distributed(df, suffix=".ann", verbosity=0)

        # Should have expected columns
        assert set(result.columns) == {"doc_id", "attachment_name", "success"}
