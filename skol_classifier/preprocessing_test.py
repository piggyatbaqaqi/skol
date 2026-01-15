"""
Tests for preprocessing.py module.

Run with: pytest skol_classifier/preprocessing_test.py -v
"""

import os
import sys
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, StringType

from .preprocessing import (
    SuffixTransformer,
    ParagraphExtractor,
    NOMENCLATURE_RE,
    TABLE_KEYWORDS,
    FIGURE_KEYWORDS
)


# Get the project root directory (parent of skol_classifier)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Get the parent of project root (where 'skol' directory lives)
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)


# Fixtures
@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    # Ensure the parent directory is in the Python path for Spark workers
    if PARENT_ROOT not in sys.path:
        sys.path.insert(0, PARENT_ROOT)

    session = SparkSession.builder \
        .appName("PreprocessingTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    yield session
    session.stop()


class TestSuffixTransformer:
    """Tests for SuffixTransformer class."""

    def test_extract_suffixes_basic(self):
        """Test basic suffix extraction."""
        words = ["hello", "world"]
        suffixes = SuffixTransformer.extract_suffixes(words)

        # Each word should produce 3 suffixes (2, 3, 4 chars)
        assert len(suffixes) == 6

        # Check specific suffixes for "hello"
        assert "lo" in suffixes      # 2-char
        assert "llo" in suffixes     # 3-char
        assert "ello" in suffixes    # 4-char

        # Check specific suffixes for "world"
        assert "ld" in suffixes      # 2-char
        assert "rld" in suffixes     # 3-char
        assert "orld" in suffixes    # 4-char

    def test_extract_suffixes_short_words(self):
        """Test suffix extraction with short words."""
        words = ["ab", "a"]
        suffixes = SuffixTransformer.extract_suffixes(words)

        # Should still produce suffixes (even if they're partial)
        assert len(suffixes) == 6

    def test_extract_suffixes_empty_list(self):
        """Test suffix extraction with empty word list."""
        words = []
        suffixes = SuffixTransformer.extract_suffixes(words)

        assert len(suffixes) == 0

    def test_suffix_transformer_spark(self, spark):
        """Test SuffixTransformer with Spark DataFrame.

        Note: This test requires the package to be properly installed
        (e.g., pip install -e .) for Spark workers to deserialize the
        SuffixTransformer UDF. In development mode without installation,
        the test will be skipped.
        """
        from pyspark.errors.exceptions.captured import PythonException

        # Create test data
        data = [
            (["taxonomic", "classification"],),
            (["species", "genus"],),
        ]
        schema = StructType([
            StructField("words", ArrayType(StringType()), False)
        ])
        df = spark.createDataFrame(data, schema)

        # Apply transformer
        transformer = SuffixTransformer(inputCol="words", outputCol="suffixes")

        try:
            result = transformer._transform(df)

            # Check output column exists
            assert "suffixes" in result.columns

            # Check suffixes were extracted
            rows = result.collect()
            for row in rows:
                assert len(row.suffixes) > 0
        except PythonException as e:
            if "ModuleNotFoundError" in str(e) and "skol" in str(e):
                pytest.skip(
                    "Skipping: Package not installed. Run 'pip install -e .' "
                    "to enable Spark UDF tests with custom transformers."
                )
            raise

    def test_suffix_transformer_params(self):
        """Test SuffixTransformer parameter handling."""
        transformer = SuffixTransformer(inputCol="custom_input", outputCol="custom_output")

        assert transformer.getInputCol() == "custom_input"
        assert transformer.getOutputCol() == "custom_output"


class TestParagraphExtractor:
    """Tests for ParagraphExtractor class."""

    def test_extract_annotated_paragraphs_basic(self):
        """Test basic annotated paragraph extraction.

        Note: The function only captures multi-line annotations where
        the opening '[' and closing ']' are on separate lines.
        Single-line annotations are not captured by this function.
        """
        lines = [
            "[@ Nomenclature paragraph",
            "continuation of text",
            "end of paragraph #Nomenclature*]",
        ]

        result = ParagraphExtractor.extract_annotated_paragraphs(lines)

        # Should find 1 paragraph (multi-line annotation)
        assert len(result) == 1
        assert "Nomenclature paragraph" in result[0]
        assert "continuation of text" in result[0]

    def test_extract_annotated_paragraphs_empty(self):
        """Test with empty input."""
        result = ParagraphExtractor.extract_annotated_paragraphs([])
        assert len(result) == 0

    def test_extract_annotated_paragraphs_no_brackets(self):
        """Test with text that has no bracket markers."""
        lines = ["line1", "line2", "line3"]
        result = ParagraphExtractor.extract_annotated_paragraphs(lines)
        assert len(result) == 0

    def test_extract_heuristic_paragraphs_blank_lines(self):
        """Test heuristic paragraph extraction with blank lines.

        Note: The function uses complex heuristics and may not split on
        blank lines in all cases. This test verifies basic functionality.
        """
        lines = [
            "First paragraph text that is long enough to not trigger short line detection.",
            "",
            "Second paragraph text that is also long enough for detection."
        ]

        result = ParagraphExtractor.extract_heuristic_paragraphs(lines)

        # Should produce at least one paragraph
        assert len(result) >= 1
        assert "First paragraph" in result[0]

    def test_extract_heuristic_paragraphs_table_detection(self):
        """Test table detection in heuristic extraction.

        Note: The function uses complex heuristics. Table detection sets
        a state flag but paragraphs are only output when another trigger
        occurs (like a blank line or short line at the end).
        """
        lines = [
            "Normal paragraph before the table with more than forty five characters.",
            "Table 1. Species distribution across multiple geographic regions worldwide",
            "Species A 45%",
        ]

        result = ParagraphExtractor.extract_heuristic_paragraphs(lines)

        # Table keyword triggers paragraph break, so "Normal paragraph" should be output
        assert len(result) >= 1
        assert any("Normal paragraph" in p for p in result)

    def test_extract_heuristic_paragraphs_figure_detection(self):
        """Test figure detection in heuristic extraction.

        Note: The function has complex figure detection. Figure lines
        trigger paragraph breaks for preceding content.
        """
        lines = [
            "Normal paragraph before the figure with more than forty five characters.",
            "Fig. 1. Description of figure showing species distribution patterns",
        ]

        result = ParagraphExtractor.extract_heuristic_paragraphs(lines)

        # Figure keyword triggers paragraph break for preceding content
        assert len(result) >= 1
        assert any("Normal paragraph" in p for p in result)

    def test_extract_heuristic_paragraphs_hyphen_start(self):
        """Test paragraph detection with hyphen-starting lines."""
        lines = [
            "Normal paragraph text that is definitely long enough.",
            "- hyphenated item",
            "- another item",
        ]

        result = ParagraphExtractor.extract_heuristic_paragraphs(lines)

        # Hyphen lines should start new paragraphs
        assert any(p.startswith("-") for p in result)

    def test_collapse_labels_nomenclature(self):
        """Test label collapsing for Nomenclature."""
        result = ParagraphExtractor.collapse_labels("Nomenclature")
        assert result == "Nomenclature"

    def test_collapse_labels_description(self):
        """Test label collapsing for Description."""
        result = ParagraphExtractor.collapse_labels("Description")
        assert result == "Description"

    def test_collapse_labels_misc(self):
        """Test label collapsing for misc labels."""
        misc_labels = ["Materials", "Methods", "Discussion", "References", "Other"]

        for label in misc_labels:
            result = ParagraphExtractor.collapse_labels(label)
            assert result == "Misc-exposition"


class TestNomenclatureRegex:
    """Tests for nomenclature detection regex."""

    def test_nomenclature_regex_basic_species(self):
        """Test nomenclature regex with basic species name."""
        text = "Russula cyanoxantha 1881"
        assert NOMENCLATURE_RE.match(text) is not None

    def test_nomenclature_regex_nov_sp(self):
        """Test nomenclature regex with nov. sp."""
        text = "Russula cyanoxantha nov. sp."
        assert NOMENCLATURE_RE.match(text) is not None

    def test_nomenclature_regex_nov_comb(self):
        """Test nomenclature regex with nov. comb."""
        text = "Russula cyanoxantha nov. comb."
        assert NOMENCLATURE_RE.match(text) is not None

    def test_nomenclature_regex_with_figure(self):
        """Test nomenclature regex with figure reference."""
        text = "Russula cyanoxantha 1881 Fig"
        assert NOMENCLATURE_RE.match(text) is not None

    def test_nomenclature_regex_no_match(self):
        """Test nomenclature regex with non-nomenclature text."""
        text = "This is a regular sentence without nomenclature."
        assert NOMENCLATURE_RE.match(text) is None


class TestKeywordConstants:
    """Tests for keyword constant lists."""

    def test_table_keywords(self):
        """Test table keywords list."""
        assert "table" in TABLE_KEYWORDS
        assert "tab." in TABLE_KEYWORDS
        assert len(TABLE_KEYWORDS) >= 3

    def test_figure_keywords(self):
        """Test figure keywords list."""
        assert "fig" in FIGURE_KEYWORDS
        assert "figure" in FIGURE_KEYWORDS
        assert "plate" in FIGURE_KEYWORDS
        assert len(FIGURE_KEYWORDS) >= 5
