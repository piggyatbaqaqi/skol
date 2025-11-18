"""
Unit tests for CouchDB file reading functionality.
"""

import unittest
from pyspark.sql import Row

# Add parent directory to path for imports
import sys
sys.path.insert(0, '..')

from line import Line
from couchdb_file import CouchDBFile, read_couchdb_partition


class TestCouchDBFile(unittest.TestCase):
    """Test CouchDBFile class."""

    def test_couchdb_file_basic(self):
        """Test basic CouchDBFile creation and reading."""
        content = "Line 1\nLine 2\nLine 3"
        doc_id = "doc123"
        attachment_name = "test.txt.ann"
        db_name = "test_db"

        file_obj = CouchDBFile(content, doc_id, attachment_name, db_name)

        # Check properties
        self.assertEqual(file_obj.doc_id, doc_id)
        self.assertEqual(file_obj.attachment_name, attachment_name)
        self.assertEqual(file_obj.db_name, db_name)
        self.assertEqual(file_obj.filename, f"{db_name}/{doc_id}/{attachment_name}")

        # Read lines
        lines = list(file_obj.read_line())
        self.assertEqual(len(lines), 3)

        # Check first line
        self.assertIsInstance(lines[0], Line)
        self.assertEqual(lines[0].line, "Line 1")
        self.assertEqual(lines[0].doc_id, doc_id)
        self.assertEqual(lines[0].attachment_name, attachment_name)
        self.assertEqual(lines[0].db_name, db_name)
        self.assertEqual(lines[0].line_number, 1)

    def test_couchdb_file_annotated_content(self):
        """Test CouchDBFile with annotated content."""
        content = "[@Species nova#Nomenclature*]\n[@Description text#Description*]"
        file_obj = CouchDBFile(content, "doc1", "test.txt.ann", "mycobank")

        lines = list(file_obj.read_line())
        self.assertEqual(len(lines), 2)

        # First line should have label start marker
        self.assertTrue(lines[0].contains_start())
        self.assertEqual(lines[0].end_label(), "Nomenclature")

        # Second line should have label end marker
        self.assertTrue(lines[1].contains_start())
        self.assertEqual(lines[1].end_label(), "Description")

    def test_couchdb_file_page_numbers(self):
        """Test page number tracking."""
        content = "Page 1\n\fPage 2\n\fPage 3"
        file_obj = CouchDBFile(content, "doc1", "test.txt", "db1")

        lines = list(file_obj.read_line())

        # Page numbers should increment at form feed
        self.assertEqual(lines[0].page_number, 1)
        self.assertEqual(lines[1].page_number, 2)
        self.assertEqual(lines[2].page_number, 3)

    def test_couchdb_file_metadata_preserved(self):
        """Test that metadata is preserved through line objects."""
        content = "Test line"
        doc_id = "important_doc"
        attachment = "data.txt.ann"
        db = "production_db"

        file_obj = CouchDBFile(content, doc_id, attachment, db)
        lines = list(file_obj.read_line())

        line = lines[0]
        self.assertEqual(line.doc_id, doc_id)
        self.assertEqual(line.attachment_name, attachment)
        self.assertEqual(line.db_name, db)
        self.assertEqual(line.filename, f"{db}/{doc_id}/{attachment}")


class TestReadCouchDBPartition(unittest.TestCase):
    """Test read_couchdb_partition function."""

    def test_read_partition_single_row(self):
        """Test reading a single row."""
        rows = [
            Row(
                doc_id="doc1",
                attachment_name="file1.txt.ann",
                value="Line 1\nLine 2"
            )
        ]

        lines = list(read_couchdb_partition(iter(rows), "test_db"))

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].line, "Line 1")
        self.assertEqual(lines[0].doc_id, "doc1")
        self.assertEqual(lines[0].db_name, "test_db")

    def test_read_partition_multiple_rows(self):
        """Test reading multiple rows (multiple files)."""
        rows = [
            Row(doc_id="doc1", attachment_name="file1.txt", value="File 1 Line 1"),
            Row(doc_id="doc2", attachment_name="file2.txt", value="File 2 Line 1\nFile 2 Line 2")
        ]

        lines = list(read_couchdb_partition(iter(rows), "db1"))

        # Should have 3 lines total (1 from first file, 2 from second)
        self.assertEqual(len(lines), 3)

        # Check first file
        self.assertEqual(lines[0].doc_id, "doc1")
        self.assertEqual(lines[0].attachment_name, "file1.txt")

        # Check second file
        self.assertEqual(lines[1].doc_id, "doc2")
        self.assertEqual(lines[2].doc_id, "doc2")

    def test_read_partition_with_annotations(self):
        """Test reading annotated content from partition."""
        rows = [
            Row(
                doc_id="annotated_doc",
                attachment_name="taxa.txt.ann",
                value="[@Nomenclature paragraph#Nomenclature*]\n[@Description paragraph#Description*]"
            )
        ]

        lines = list(read_couchdb_partition(iter(rows), "mycobank"))

        self.assertEqual(len(lines), 2)

        # Check annotation markers
        self.assertTrue(lines[0].contains_start())
        self.assertEqual(lines[0].end_label(), "Nomenclature")
        self.assertTrue(lines[1].contains_start())
        self.assertEqual(lines[1].end_label(), "Description")

        # Check metadata preserved
        self.assertEqual(lines[0].doc_id, "annotated_doc")
        self.assertEqual(lines[0].db_name, "mycobank")


class TestLineFilenameWithCouchDB(unittest.TestCase):
    """Test composite filename generation for CouchDB Lines."""

    def test_filename_format(self):
        """Test that filename follows db_name/doc_id/attachment_name format."""
        content = "Test"
        file_obj = CouchDBFile(content, "my_doc", "data.txt", "production")

        lines = list(file_obj.read_line())
        filename = lines[0].filename

        self.assertEqual(filename, "production/my_doc/data.txt")

        # Should be parseable
        parts = filename.split('/', 2)
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], "production")
        self.assertEqual(parts[1], "my_doc")
        self.assertEqual(parts[2], "data.txt")

    def test_filename_with_slashes_in_attachment(self):
        """Test filename when attachment name contains path separators."""
        # CouchDB attachment names can contain slashes
        content = "Test"
        file_obj = CouchDBFile(content, "doc1", "folder/subfolder/file.txt", "db1")

        lines = list(file_obj.read_line())
        filename = lines[0].filename

        # Should preserve the attachment path
        self.assertEqual(filename, "db1/doc1/folder/subfolder/file.txt")


class TestIntegrationWithParsers(unittest.TestCase):
    """Test integration with existing parser functions."""

    def test_with_parse_annotated(self):
        """Test that Line with CouchDB metadata works with parse_annotated()."""
        from finder import parse_annotated

        rows = [
            Row(
                doc_id="test_doc",
                attachment_name="test.txt.ann",
                value="[@First paragraph#Nomenclature*]\n[@Second paragraph#Description*]"
            )
        ]

        lines = read_couchdb_partition(iter(rows), "test_db")
        paragraphs = list(parse_annotated(lines))

        self.assertEqual(len(paragraphs), 2)

        # Check that metadata is preserved in paragraphs
        self.assertEqual(str(paragraphs[0].top_label()), "Nomenclature")
        self.assertEqual(paragraphs[0].filename, "test_db/test_doc/test.txt.ann")

        self.assertEqual(str(paragraphs[1].top_label()), "Description")
        self.assertEqual(paragraphs[1].filename, "test_db/test_doc/test.txt.ann")

    def test_with_group_paragraphs(self):
        """Test full pipeline: CouchDB → Lines → Paragraphs → Taxa."""
        from finder import parse_annotated
        from taxon import group_paragraphs

        # Simulate a complete taxon with nomenclature and description
        content = (
            "[@Fungus novus Author 1999#Nomenclature*]\n"
            "[@This is the description of the species.#Description*]"
        )

        rows = [
            Row(doc_id="paper_2023", attachment_name="article.txt.ann", value=content)
        ]

        # Full pipeline
        lines = read_couchdb_partition(iter(rows), "mycobank")
        paragraphs = parse_annotated(lines)
        taxa = list(group_paragraphs(paragraphs))

        # Should have one taxon
        self.assertEqual(len(taxa), 1)

        taxon = taxa[0]
        self.assertTrue(taxon.has_nomenclature())
        self.assertTrue(taxon.has_description())

        # Check metadata in output
        dicts = list(taxon.dictionaries())
        self.assertEqual(len(dicts), 2)  # One for nomenclature, one for description

        # Both should have the same filename from CouchDB
        for d in dicts:
            self.assertEqual(d['filename'], "mycobank/paper_2023/article.txt.ann")


if __name__ == '__main__':
    unittest.main()
