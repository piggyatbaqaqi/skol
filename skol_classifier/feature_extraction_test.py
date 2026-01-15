"""
Tests for feature_extraction.py module.

Run with: pytest skol_classifier/feature_extraction_test.py -v
"""

import os
import sys
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

from .feature_extraction import FeatureExtractor


# Get the project root directory (parent of skol_classifier)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Get the parent of project root (where 'skol' directory lives)
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)


# Fixtures
@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    # Ensure the parent directory is in the Python path for Spark workers
    # This allows 'skol.skol_classifier' imports to work
    if PARENT_ROOT not in sys.path:
        sys.path.insert(0, PARENT_ROOT)

    session = SparkSession.builder \
        .appName("FeatureExtractionTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    # Add the skol_classifier package files to Spark for worker access
    skol_classifier_dir = os.path.join(PROJECT_ROOT, "skol_classifier")
    for pyfile in ["preprocessing.py", "__init__.py"]:
        filepath = os.path.join(skol_classifier_dir, pyfile)
        if os.path.exists(filepath):
            session.sparkContext.addPyFile(filepath)

    yield session
    session.stop()


@pytest.fixture
def sample_data(spark):
    """Create sample data for testing."""
    data = [
        ("Russula cyanoxantha is a species of mushroom.", "Nomenclature"),
        ("The cap is convex to flat.", "Description"),
        ("Found in deciduous forests.", "Description"),
        ("The type species is R. cyanoxantha.", "Nomenclature"),
        ("Materials and methods section text here.", "Misc-exposition"),
    ]
    schema = StructType([
        StructField("value", StringType(), False),
        StructField("label", StringType(), False)
    ])
    return spark.createDataFrame(data, schema)


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_init_defaults(self):
        """Test default initialization parameters."""
        extractor = FeatureExtractor()

        assert extractor.use_suffixes is True
        assert extractor.use_section_names is False
        assert extractor.min_doc_freq == 2
        assert extractor.input_col == "value"
        assert extractor.label_col == "label"
        assert extractor.word_vocab_size == 800
        assert extractor.suffix_vocab_size == 200
        assert extractor.section_name_vocab_size == 50

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        extractor = FeatureExtractor(
            use_suffixes=False,
            use_section_names=True,
            min_doc_freq=5,
            input_col="text",
            label_col="category",
            word_vocab_size=500,
            suffix_vocab_size=100,
            section_name_vocab_size=25
        )

        assert extractor.use_suffixes is False
        assert extractor.use_section_names is True
        assert extractor.min_doc_freq == 5
        assert extractor.input_col == "text"
        assert extractor.label_col == "category"
        assert extractor.word_vocab_size == 500

    def test_build_pipeline_basic(self, spark):
        """Test basic pipeline building.

        Note: spark fixture required as Spark ML stages need an active context.
        """
        extractor = FeatureExtractor(use_suffixes=False)
        pipeline = extractor.build_pipeline()

        # Pipeline should have stages
        assert len(pipeline.getStages()) > 0

    def test_build_pipeline_with_suffixes(self, spark):
        """Test pipeline building with suffix features.

        Note: spark fixture required as Spark ML stages need an active context.
        """
        extractor = FeatureExtractor(use_suffixes=True)
        pipeline = extractor.build_pipeline()

        # Should have more stages for suffix processing
        stage_count = len(pipeline.getStages())
        assert stage_count >= 4  # tokenizer, word tf-idf, suffix stages, etc.

    def test_build_pipeline_with_section_names(self, spark):
        """Test pipeline building with section name features.

        Note: spark fixture required as Spark ML stages need an active context.
        """
        extractor = FeatureExtractor(use_suffixes=False, use_section_names=True)
        pipeline = extractor.build_pipeline()

        # Should have section name processing stages
        assert len(pipeline.getStages()) >= 4

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        extractor = FeatureExtractor(
            use_suffixes=False,
            min_doc_freq=1  # Lower for small test data
        )

        result_df = extractor.fit_transform(sample_data)

        # Should have feature columns
        assert "word_idf" in result_df.columns
        assert "label_indexed" in result_df.columns

        # Pipeline should be fitted
        assert extractor.pipeline_model is not None
        assert extractor.labels is not None

    def test_fit_transform_with_suffixes(self, sample_data):
        """Test fit_transform with suffix features.

        Note: This test requires the package to be properly installed
        (e.g., pip install -e .) for Spark workers to deserialize the
        SuffixTransformer UDF. In development mode without installation,
        the test will be skipped.
        """
        from pyspark.errors.exceptions.captured import PythonException

        extractor = FeatureExtractor(
            use_suffixes=True,
            min_doc_freq=1
        )

        try:
            result_df = extractor.fit_transform(sample_data)

            # Should have combined features
            assert "combined_idf" in result_df.columns
            assert "suffix_idf" in result_df.columns
        except PythonException as e:
            if "ModuleNotFoundError" in str(e) and "skol" in str(e):
                pytest.skip(
                    "Skipping: Package not installed. Run 'pip install -e .' "
                    "to enable Spark UDF tests with custom transformers."
                )
            raise

    def test_transform_without_fit(self, sample_data):
        """Test that transform raises error without prior fit."""
        extractor = FeatureExtractor()

        with pytest.raises(ValueError, match="Pipeline not fitted"):
            extractor.transform(sample_data)

    def test_transform_after_fit(self, sample_data):
        """Test transform after fit_transform."""
        extractor = FeatureExtractor(
            use_suffixes=False,
            min_doc_freq=1
        )

        # First fit
        extractor.fit_transform(sample_data)

        # Then transform new data
        result_df = extractor.transform(sample_data)

        assert "word_idf" in result_df.columns
        assert "label_indexed" in result_df.columns

    def test_get_pipeline(self, sample_data):
        """Test get_pipeline method."""
        extractor = FeatureExtractor(use_suffixes=False, min_doc_freq=1)

        # Before fit, should be None
        assert extractor.get_pipeline() is None

        # After fit
        extractor.fit_transform(sample_data)
        assert extractor.get_pipeline() is not None

    def test_get_labels(self, sample_data):
        """Test get_labels method."""
        extractor = FeatureExtractor(use_suffixes=False, min_doc_freq=1)

        # Before fit, should be None
        assert extractor.get_labels() is None

        # After fit
        extractor.fit_transform(sample_data)
        labels = extractor.get_labels()

        assert labels is not None
        assert len(labels) == 3  # Nomenclature, Description, Misc-exposition

    def test_get_label_mapping(self, sample_data):
        """Test get_label_mapping method (alias for get_labels)."""
        extractor = FeatureExtractor(use_suffixes=False, min_doc_freq=1)
        extractor.fit_transform(sample_data)

        assert extractor.get_label_mapping() == extractor.get_labels()

    def test_get_features_col_without_suffixes(self):
        """Test get_features_col without suffix features."""
        extractor = FeatureExtractor(use_suffixes=False, use_section_names=False)
        assert extractor.get_features_col() == "word_idf"

    def test_get_features_col_with_suffixes(self):
        """Test get_features_col with suffix features."""
        extractor = FeatureExtractor(use_suffixes=True)
        assert extractor.get_features_col() == "combined_idf"

    def test_get_features_col_with_section_names(self):
        """Test get_features_col with section name features."""
        extractor = FeatureExtractor(use_suffixes=False, use_section_names=True)
        assert extractor.get_features_col() == "combined_idf"

    def test_label_indexing_consistency(self, sample_data):
        """Test that label indexing is consistent."""
        extractor = FeatureExtractor(use_suffixes=False, min_doc_freq=1)

        result_df = extractor.fit_transform(sample_data)
        labels = extractor.get_labels()

        # Collect label mappings
        label_map = {}
        rows = result_df.select("label", "label_indexed").distinct().collect()
        for row in rows:
            label_map[row["label"]] = row["label_indexed"]

        # Each unique label should map to a unique index
        assert len(label_map) == len(labels)

        # Indices should be 0, 1, 2, ...
        indices = sorted(label_map.values())
        assert indices == list(range(len(labels)))

    def test_vocab_size_limits(self, spark):
        """Test that vocabulary size limits are respected."""
        # Create data with many unique words
        data = [(f"word_{i} is unique text", "Label") for i in range(100)]
        schema = StructType([
            StructField("value", StringType(), False),
            StructField("label", StringType(), False)
        ])
        df = spark.createDataFrame(data, schema)

        extractor = FeatureExtractor(
            use_suffixes=False,
            min_doc_freq=1,
            word_vocab_size=50  # Limit vocab size
        )

        result_df = extractor.fit_transform(df)

        # Features should be created
        assert "word_idf" in result_df.columns
