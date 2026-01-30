"""
Unit tests for ingest field propagation and dual-format reading.

Tests the get_ingest_field() helper function and ingest propagation
through the data pipeline (CouchDBFile -> Line -> Taxon).
"""

import unittest
from pyspark.sql import Row

# Add parent directory to path for imports
import sys
sys.path.insert(0, '..')

from taxon import get_ingest_field
from couchdb_file import CouchDBFile, read_couchdb_partition
from line import Line


class TestGetIngestField(unittest.TestCase):
    """Test get_ingest_field() helper function."""

    def test_ingest_only_id(self):
        """Test record with only ingest field - get _id."""
        record = {
            'ingest': {
                '_id': 'doc123',
                'url': 'https://example.com/article',
                'pdf_url': 'https://example.com/article.pdf',
            }
        }
        self.assertEqual(get_ingest_field(record, '_id'), 'doc123')

    def test_ingest_only_url(self):
        """Test record with only ingest field - get url."""
        record = {
            'ingest': {
                '_id': 'doc123',
                'url': 'https://example.com/article',
                'pdf_url': 'https://example.com/article.pdf',
            }
        }
        self.assertEqual(
            get_ingest_field(record, 'url'),
            'https://example.com/article'
        )

    def test_ingest_only_pdf_url(self):
        """Test record with only ingest field - get pdf_url."""
        record = {
            'ingest': {
                '_id': 'doc123',
                'url': 'https://example.com/article',
                'pdf_url': 'https://example.com/article.pdf',
            }
        }
        self.assertEqual(
            get_ingest_field(record, 'pdf_url'),
            'https://example.com/article.pdf'
        )

    def test_no_ingest_returns_default(self):
        """Test record without ingest field returns default."""
        record = {}
        self.assertIsNone(get_ingest_field(record, '_id'))
        self.assertEqual(
            get_ingest_field(record, '_id', default='unknown'),
            'unknown'
        )

    def test_ingest_with_all_fields(self):
        """Test record with all expected ingest fields."""
        record = {
            'ingest': {
                '_id': 'doc123',
                'url': 'https://example.com/article',
                'pdf_url': 'https://example.com/article.pdf',
            }
        }
        self.assertEqual(get_ingest_field(record, '_id'), 'doc123')
        self.assertEqual(get_ingest_field(record, 'url'), 'https://example.com/article')
        self.assertEqual(
            get_ingest_field(record, 'pdf_url'),
            'https://example.com/article.pdf'
        )

    def test_missing_field_returns_default(self):
        """Test that missing fields return the default value."""
        record = {
            'ingest': {
                '_id': 'doc123',
            }
        }
        self.assertIsNone(get_ingest_field(record, 'url'))
        self.assertIsNone(get_ingest_field(record, 'pdf_url'))
        self.assertEqual(
            get_ingest_field(record, 'url', default='no_url'),
            'no_url'
        )

    def test_empty_record_returns_default(self):
        """Test that empty record returns default."""
        record = {}
        self.assertIsNone(get_ingest_field(record, '_id'))
        self.assertEqual(
            get_ingest_field(record, '_id', default='unknown'),
            'unknown'
        )

    def test_nested_field_access(self):
        """Test accessing nested fields in ingest."""
        record = {
            'ingest': {
                '_id': 'doc123',
                'metadata': {
                    'author': 'Smith',
                    'year': 2023,
                }
            }
        }
        self.assertEqual(
            get_ingest_field(record, 'metadata', 'author'),
            'Smith'
        )
        self.assertEqual(
            get_ingest_field(record, 'metadata', 'year'),
            2023
        )

    def test_none_ingest_returns_default(self):
        """Test that None ingest value returns default."""
        record = {
            'ingest': None,
        }
        self.assertIsNone(get_ingest_field(record, '_id'))
        self.assertEqual(
            get_ingest_field(record, '_id', default='unknown'),
            'unknown'
        )


class TestCouchDBFileIngestPropagation(unittest.TestCase):
    """Test ingest field propagation through CouchDBFile."""

    def test_ingest_passed_to_line(self):
        """Test that ingest is propagated to Line objects."""
        content = "Test line"
        ingest = {
            '_id': 'doc123',
            'url': 'https://example.com/article',
            'pdf_url': 'https://example.com/article.pdf',
            'title': 'Test Article',
        }

        file_obj = CouchDBFile(
            content=content,
            doc_id="doc123",
            attachment_name="test.txt.ann",
            db_name="test_db",
            ingest=ingest
        )

        lines = list(file_obj.read_line())
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].ingest, ingest)

    def test_ingest_none_by_default(self):
        """Test that ingest is None when not provided."""
        content = "Test line"
        file_obj = CouchDBFile(
            content=content,
            doc_id="doc123",
            attachment_name="test.txt.ann",
            db_name="test_db"
        )

        lines = list(file_obj.read_line())
        self.assertEqual(len(lines), 1)
        self.assertIsNone(lines[0].ingest)


class TestReadCouchDBPartitionIngest(unittest.TestCase):
    """Test ingest propagation through read_couchdb_partition."""

    def test_ingest_from_row(self):
        """Test that ingest is propagated from Row to Line."""
        ingest = {
            '_id': 'doc123',
            'url': 'https://example.com/article',
            'pdf_url': 'https://example.com/article.pdf',
        }
        rows = [
            Row(
                doc_id="doc123",
                attachment_name="file.txt.ann",
                value="Line 1\nLine 2",
                ingest=ingest
            )
        ]

        lines = list(read_couchdb_partition(iter(rows), "test_db"))

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].ingest, ingest)
        self.assertEqual(lines[1].ingest, ingest)

    def test_ingest_missing_from_row(self):
        """Test handling rows without ingest field."""
        rows = [
            Row(
                doc_id="doc123",
                attachment_name="file.txt.ann",
                value="Line 1"
            )
        ]

        lines = list(read_couchdb_partition(iter(rows), "test_db"))

        self.assertEqual(len(lines), 1)
        self.assertIsNone(lines[0].ingest)


class TestTaxonIngestOutput(unittest.TestCase):
    """Test ingest field in Taxon.as_row() output."""

    def test_as_row_includes_ingest(self):
        """Test that Taxon.as_row() includes ingest field."""
        from finder import parse_annotated
        from taxon import group_paragraphs

        ingest = {
            '_id': 'doc123',
            'url': 'https://example.com/article',
            'pdf_url': 'https://example.com/article.pdf',
            'title': 'A new species',
        }

        content = (
            "[@Fungus novus Author 1999#Nomenclature*]\n"
            "[@This is the description.#Description*]"
        )

        rows = [
            Row(
                doc_id="doc123",
                attachment_name="article.txt.ann",
                value=content,
                ingest=ingest
            )
        ]

        lines = read_couchdb_partition(iter(rows), "test_db")
        paragraphs = parse_annotated(lines)
        taxa = list(group_paragraphs(paragraphs))

        self.assertEqual(len(taxa), 1)

        taxon_row = taxa[0].as_row()

        # Check that ingest is in output
        self.assertIn('ingest', taxon_row)
        self.assertEqual(taxon_row['ingest'], ingest)

        # Source field has been removed (Phase 4 complete)
        self.assertNotIn('source', taxon_row)

    def test_as_row_ingest_none_when_not_provided(self):
        """Test that as_row() has None ingest when not provided."""
        from finder import parse_annotated
        from taxon import group_paragraphs

        content = (
            "[@Fungus novus Author 1999#Nomenclature*]\n"
            "[@This is the description.#Description*]"
        )

        rows = [
            Row(
                doc_id="doc123",
                attachment_name="article.txt.ann",
                value=content
            )
        ]

        lines = read_couchdb_partition(iter(rows), "test_db")
        paragraphs = parse_annotated(lines)
        taxa = list(group_paragraphs(paragraphs))

        self.assertEqual(len(taxa), 1)

        taxon_row = taxa[0].as_row()

        # ingest should be empty dict when not provided
        self.assertIn('ingest', taxon_row)

        # Source field has been removed (Phase 4 complete)
        self.assertNotIn('source', taxon_row)


if __name__ == '__main__':
    unittest.main()
