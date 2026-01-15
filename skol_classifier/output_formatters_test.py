"""
Tests for output_formatters.py module.

Run with: pytest skol_classifier/output_formatters_test.py -v
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, BooleanType
)

from .output_formatters import YeddaFormatter, FileOutputWriter


# Fixtures
@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    session = SparkSession.builder \
        .appName("OutputFormatterTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def sample_predictions(spark):
    """Create sample predictions for testing."""
    data = [
        ("doc1", "article.txt", 1, "Russula cyanoxantha species.", "Nomenclature", False),
        ("doc1", "article.txt", 2, "The cap is convex.", "Description", False),
        ("doc1", "article.txt", 3, "Color is purple.", "Description", False),
        ("doc1", "article.txt", 4, "Found in forests.", "Misc-exposition", False),
    ]
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("line_number", IntegerType(), False),
        StructField("value", StringType(), False),
        StructField("predicted_label", StringType(), False),
        StructField("is_page_marker", BooleanType(), False),
    ])
    return spark.createDataFrame(data, schema)


@pytest.fixture
def predictions_with_page_markers(spark):
    """Create predictions with page markers for testing."""
    data = [
        ("doc1", "article.txt", 1, "Introduction text.", "Description", False),
        ("doc1", "article.txt", 2, "More description.", "Description", False),
        ("doc1", "article.txt", 3, "--- PDF Page 1 ---", None, True),
        ("doc1", "article.txt", 4, "Nomenclature text.", "Nomenclature", False),
        ("doc1", "article.txt", 5, "Species name.", "Nomenclature", False),
        ("doc1", "article.txt", 6, "--- PDF Page 2 ---", None, True),
        ("doc1", "article.txt", 7, "Final description.", "Description", False),
    ]
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("line_number", IntegerType(), False),
        StructField("value", StringType(), True),  # Nullable for page markers
        StructField("predicted_label", StringType(), True),  # Nullable for page markers
        StructField("is_page_marker", BooleanType(), False),
    ])
    return spark.createDataFrame(data, schema)


class TestYeddaFormatter:
    """Tests for YeddaFormatter class."""

    def test_init_defaults(self):
        """Test default initialization."""
        formatter = YeddaFormatter()

        assert formatter.coalesce_labels is False
        assert formatter.line_level is False

    def test_init_with_params(self):
        """Test initialization with parameters."""
        formatter = YeddaFormatter(coalesce_labels=True, line_level=True)

        assert formatter.coalesce_labels is True
        assert formatter.line_level is True

    def test_format_predictions_basic(self, sample_predictions):
        """Test basic prediction formatting."""
        result = YeddaFormatter.format_predictions(sample_predictions)

        assert "annotated_value" in result.columns

        # Check format of first row
        rows = result.collect()
        first_row = rows[0]
        assert first_row.annotated_value.startswith("[@ ")
        assert first_row.annotated_value.endswith("*]")
        assert "#Nomenclature" in first_row.annotated_value

    def test_format_predictions_yedda_format(self, sample_predictions):
        """Test that YEDDA format is correct."""
        result = YeddaFormatter.format_predictions(sample_predictions)
        rows = result.orderBy("line_number").collect()

        # First row should be: [@ Russula cyanoxantha species. #Nomenclature*]
        assert rows[0].annotated_value == "[@ Russula cyanoxantha species. #Nomenclature*]"

        # Second row should be: [@ The cap is convex. #Description*]
        assert rows[1].annotated_value == "[@ The cap is convex. #Description*]"

    def test_format_method_without_coalesce(self, sample_predictions):
        """Test format method without coalescing."""
        formatter = YeddaFormatter(coalesce_labels=False)
        result = formatter.format(sample_predictions)

        assert "annotated_value" in result.columns
        assert result.count() == sample_predictions.count()

    def test_format_method_with_coalesce(self, sample_predictions):
        """Test format method with coalescing."""
        formatter = YeddaFormatter(coalesce_labels=True, line_level=True)
        result = formatter.format(sample_predictions)

        assert "coalesced_annotations" in result.columns

        # Should have one row per document
        assert result.count() == 1


class TestYeddaFormatterCoalesce:
    """Tests for coalesce_consecutive_labels method."""

    def test_coalesce_consecutive_labels(self, sample_predictions):
        """Test coalescing consecutive labels."""
        result = YeddaFormatter.coalesce_consecutive_labels(
            sample_predictions, line_level=True
        )

        assert "coalesced_annotations" in result.columns

        # Verify result
        rows = result.collect()
        assert len(rows) == 1  # One document

        annotations = rows[0].coalesced_annotations
        # Should have 3 blocks:
        # 1. Nomenclature (line 1)
        # 2. Description (lines 2-3 coalesced)
        # 3. Misc-exposition (line 4)
        assert len(annotations) == 3

    def test_coalesce_groups_same_labels(self, spark):
        """Test that consecutive same labels are grouped."""
        data = [
            ("doc1", "article.txt", 1, "Line 1", "Description", False),
            ("doc1", "article.txt", 2, "Line 2", "Description", False),
            ("doc1", "article.txt", 3, "Line 3", "Description", False),
        ]
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("value", StringType(), False),
            StructField("predicted_label", StringType(), False),
            StructField("is_page_marker", BooleanType(), False),
        ])
        df = spark.createDataFrame(data, schema)

        result = YeddaFormatter.coalesce_consecutive_labels(df, line_level=True)
        rows = result.collect()
        annotations = rows[0].coalesced_annotations

        # Should have 1 block (all same label)
        assert len(annotations) == 1
        assert "Line 1" in annotations[0]
        assert "Line 2" in annotations[0]
        assert "Line 3" in annotations[0]
        assert "#Description*]" in annotations[0]

    def test_coalesce_alternating_labels(self, spark):
        """Test coalescing with alternating labels."""
        data = [
            ("doc1", "article.txt", 1, "Line 1", "A", False),
            ("doc1", "article.txt", 2, "Line 2", "B", False),
            ("doc1", "article.txt", 3, "Line 3", "A", False),
            ("doc1", "article.txt", 4, "Line 4", "B", False),
        ]
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("value", StringType(), False),
            StructField("predicted_label", StringType(), False),
            StructField("is_page_marker", BooleanType(), False),
        ])
        df = spark.createDataFrame(data, schema)

        result = YeddaFormatter.coalesce_consecutive_labels(df, line_level=True)
        rows = result.collect()
        annotations = rows[0].coalesced_annotations

        # Should have 4 blocks (alternating labels)
        assert len(annotations) == 4

    def test_coalesce_page_markers_preserved(self, predictions_with_page_markers):
        """Test that page markers are preserved and break coalescing."""
        result = YeddaFormatter.coalesce_consecutive_labels(
            predictions_with_page_markers, line_level=True
        )
        rows = result.collect()
        annotations = rows[0].coalesced_annotations

        # Page markers should be preserved as-is
        page_markers = [a for a in annotations if "PDF Page" in a]
        assert len(page_markers) == 2
        assert "--- PDF Page 1 ---" in annotations
        assert "--- PDF Page 2 ---" in annotations

        # Page markers should NOT be wrapped in YEDDA format
        for marker in page_markers:
            assert not marker.startswith("[@ ")

    def test_coalesce_page_markers_break_blocks(self, predictions_with_page_markers):
        """Test that page markers break coalescing blocks."""
        result = YeddaFormatter.coalesce_consecutive_labels(
            predictions_with_page_markers, line_level=True
        )
        rows = result.collect()
        annotations = rows[0].coalesced_annotations

        # Description blocks should be split by page marker
        # Block 1: Description (lines 1-2)
        # Page marker
        # Block 2: Nomenclature (lines 4-5)
        # Page marker
        # Block 3: Description (line 7)

        # Count non-page-marker blocks
        yedda_blocks = [a for a in annotations if a.startswith("[@ ")]

        # Lines 1-2 coalesced, lines 4-5 coalesced, line 7 separate
        assert len(yedda_blocks) == 3

    def test_coalesce_no_coalesce_paragraph_level(self, sample_predictions):
        """Test that paragraph-level data is not coalesced."""
        result = YeddaFormatter.coalesce_consecutive_labels(
            sample_predictions, line_level=False
        )

        # Should return unchanged (no coalescing)
        assert result.count() == sample_predictions.count()

    def test_coalesce_multiple_documents(self, spark):
        """Test coalescing with multiple documents."""
        data = [
            ("doc1", "article.txt", 1, "Doc1 Line 1", "A", False),
            ("doc1", "article.txt", 2, "Doc1 Line 2", "A", False),
            ("doc2", "article.txt", 1, "Doc2 Line 1", "B", False),
            ("doc2", "article.txt", 2, "Doc2 Line 2", "B", False),
        ]
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("value", StringType(), False),
            StructField("predicted_label", StringType(), False),
            StructField("is_page_marker", BooleanType(), False),
        ])
        df = spark.createDataFrame(data, schema)

        result = YeddaFormatter.coalesce_consecutive_labels(df, line_level=True)

        # Should have 2 rows (one per document)
        assert result.count() == 2

        # Each document should have 1 coalesced block
        rows = result.collect()
        for row in rows:
            assert len(row.coalesced_annotations) == 1


class TestFileOutputWriter:
    """Tests for FileOutputWriter class."""

    def test_save_annotated_basic(self, sample_predictions, tmp_path):
        """Test basic file output."""
        from pyspark.sql.functions import lit

        # Add filename column for file-based output
        predictions_with_filename = sample_predictions.withColumn(
            "filename", lit("test_file.txt")
        )

        # Note: FileOutputWriter.save_annotated is a static method
        # We'll test the output in a simple way
        output_path = str(tmp_path / "output")

        # The method expects specific columns, add annotated_value
        formatted = YeddaFormatter.format_predictions(predictions_with_filename)

        # For now, just verify the DataFrame was formatted correctly
        assert "annotated_value" in formatted.columns


class TestYeddaFormatterEdgeCases:
    """Edge case tests for YeddaFormatter."""

    def test_empty_dataframe(self, spark):
        """Test with empty DataFrame."""
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("value", StringType(), False),
            StructField("predicted_label", StringType(), False),
            StructField("is_page_marker", BooleanType(), False),
        ])
        empty_df = spark.createDataFrame([], schema)

        result = YeddaFormatter.format_predictions(empty_df)
        assert result.count() == 0

    def test_single_line(self, spark):
        """Test with single line document."""
        data = [("doc1", "article.txt", 1, "Single line", "Label", False)]
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("value", StringType(), False),
            StructField("predicted_label", StringType(), False),
            StructField("is_page_marker", BooleanType(), False),
        ])
        df = spark.createDataFrame(data, schema)

        result = YeddaFormatter.coalesce_consecutive_labels(df, line_level=True)
        rows = result.collect()

        assert len(rows) == 1
        assert len(rows[0].coalesced_annotations) == 1

    def test_special_characters_in_text(self, spark):
        """Test with special characters in text."""
        data = [
            ("doc1", "article.txt", 1, "Text with [brackets] and #hash", "Label", False),
            ("doc1", "article.txt", 2, "Text with *asterisks* and @at", "Label", False),
        ]
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("value", StringType(), False),
            StructField("predicted_label", StringType(), False),
            StructField("is_page_marker", BooleanType(), False),
        ])
        df = spark.createDataFrame(data, schema)

        result = YeddaFormatter.format_predictions(df)
        rows = result.orderBy("line_number").collect()

        # Special characters should be preserved
        assert "[brackets]" in rows[0].annotated_value
        assert "#hash" in rows[0].annotated_value
        assert "*asterisks*" in rows[1].annotated_value

    def test_unicode_text(self, spark):
        """Test with Unicode characters."""
        data = [
            ("doc1", "article.txt", 1, "Émile français 中文 日本語", "Label", False),
        ]
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("attachment_name", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("value", StringType(), False),
            StructField("predicted_label", StringType(), False),
            StructField("is_page_marker", BooleanType(), False),
        ])
        df = spark.createDataFrame(data, schema)

        result = YeddaFormatter.format_predictions(df)
        row = result.collect()[0]

        # Unicode should be preserved
        assert "Émile" in row.annotated_value
        assert "中文" in row.annotated_value
        assert "日本語" in row.annotated_value
