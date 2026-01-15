"""
Tests for classifier_v2.py module.

Run with: pytest skol_classifier/classifier_v2_test.py -v
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from .classifier_v2 import SkolClassifierV2


# Fixtures
@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    session = SparkSession.builder \
        .appName("ClassifierV2Tests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def sample_annotated_data(spark):
    """Create sample annotated data for testing."""
    data = [
        ("doc1", "line1.txt", 1, "Russula cyanoxantha species.", "Nomenclature"),
        ("doc1", "line1.txt", 2, "The cap is convex.", "Description"),
        ("doc1", "line1.txt", 3, "Found in forests.", "Misc-exposition"),
        ("doc2", "line2.txt", 1, "Amanita muscaria 1881", "Nomenclature"),
        ("doc2", "line2.txt", 2, "Red cap with white spots.", "Description"),
    ]
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("filename", StringType(), False),
        StructField("line_number", IntegerType(), False),
        StructField("value", StringType(), False),
        StructField("label", StringType(), False),
    ])
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_raw_data(spark):
    """Create sample raw data for testing."""
    data = [
        ("doc1", "article.txt", 1, "Some text here."),
        ("doc1", "article.txt", 2, "More text here."),
        ("doc2", "article.txt", 1, "Different document text."),
    ]
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("attachment_name", StringType(), False),
        StructField("line_number", IntegerType(), False),
        StructField("value", StringType(), False),
    ])
    return spark.createDataFrame(data, schema)


class TestSkolClassifierV2Init:
    """Tests for SkolClassifierV2 initialization."""

    def test_init_with_files_source(self, spark):
        """Test initialization with files input source."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output'
        )

        assert classifier.input_source == 'files'
        assert classifier.file_paths == ['test/*.txt']
        assert classifier.output_dest == 'files'
        assert classifier.output_path == '/tmp/output'

    def test_init_with_couchdb_source(self, spark):
        """Test initialization with CouchDB input source."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='couchdb',
            couchdb_url='http://localhost:5984',
            couchdb_database='test_db',
            couchdb_username='admin',
            couchdb_password='secret',
            output_dest='couchdb'
        )

        assert classifier.input_source == 'couchdb'
        assert classifier.couchdb_url == 'http://localhost:5984'
        assert classifier.couchdb_database == 'test_db'

    def test_init_default_values(self, spark):
        """Test default initialization values."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output'
        )

        assert classifier.extraction_mode.name in ['line', 'paragraph', 'section']
        assert classifier.use_suffixes is True
        assert classifier.min_doc_freq == 2
        assert classifier.word_vocab_size == 800
        assert classifier.model_type == 'logistic'

    def test_init_with_model_storage_disk(self, spark):
        """Test initialization with disk model storage."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            model_storage='disk',
            model_path='/tmp/model.pkl'
        )

        assert classifier.model_storage == 'disk'
        assert classifier.model_path == '/tmp/model.pkl'

    def test_init_with_model_storage_redis(self, spark):
        """Test initialization with Redis model storage."""
        mock_redis = MagicMock()

        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            model_storage='redis',
            redis_client=mock_redis,
            redis_key='test:model'
        )

        assert classifier.model_storage == 'redis'
        assert classifier.redis_client is mock_redis
        assert classifier.redis_key == 'test:model'


class TestSkolClassifierV2Validation:
    """Tests for configuration validation."""

    def test_validate_files_source_requires_paths(self, spark):
        """Test that files source requires file_paths."""
        with pytest.raises(ValueError, match="file_paths must be provided"):
            SkolClassifierV2(
                spark=spark,
                input_source='files',
                output_dest='files',
                output_path='/tmp/output'
            )

    def test_validate_couchdb_source_requires_url(self, spark):
        """Test that CouchDB source requires URL and database."""
        with pytest.raises(ValueError, match="couchdb_url and couchdb_database"):
            SkolClassifierV2(
                spark=spark,
                input_source='couchdb',
                output_dest='files',
                output_path='/tmp/output'
            )

    def test_validate_files_dest_requires_path(self, spark):
        """Test that files destination requires output_path."""
        with pytest.raises(ValueError, match="output_path must be provided"):
            SkolClassifierV2(
                spark=spark,
                input_source='files',
                file_paths=['test/*.txt'],
                output_dest='files'
            )

    def test_validate_disk_storage_requires_path(self, spark):
        """Test that disk storage requires model_path."""
        with pytest.raises(ValueError, match="model_path must be provided"):
            SkolClassifierV2(
                spark=spark,
                input_source='files',
                file_paths=['test/*.txt'],
                output_dest='files',
                output_path='/tmp/output',
                model_storage='disk'
            )

    def test_validate_redis_storage_requires_client(self, spark):
        """Test that Redis storage requires client and key."""
        with pytest.raises(ValueError, match="redis_client and redis_key"):
            SkolClassifierV2(
                spark=spark,
                input_source='files',
                file_paths=['test/*.txt'],
                output_dest='files',
                output_path='/tmp/output',
                model_storage='redis'
            )


class TestSkolClassifierV2LineLevelProperty:
    """Tests for the line_level property."""

    def test_line_level_true_for_line_mode(self, spark):
        """Test line_level is True for line extraction mode."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            extraction_mode='line'
        )

        assert classifier.line_level is True

    def test_line_level_false_for_paragraph_mode(self, spark):
        """Test line_level is False for paragraph extraction mode."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            extraction_mode='paragraph'
        )

        assert classifier.line_level is False


class TestSkolClassifierV2LabelFrequencies:
    """Tests for label frequency computation."""

    def test_get_label_frequencies_none_initially(self, spark):
        """Test that label frequencies are None before fit."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output'
        )

        assert classifier.get_label_frequencies() is None

    def test_compute_label_frequencies(self, spark, sample_annotated_data):
        """Test that label frequencies are computed when enabled."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            compute_label_frequencies=True,
            verbosity=0
        )

        # Manually set frequencies (simulating fit)
        classifier._label_frequencies = {
            "Nomenclature": 2,
            "Description": 2,
            "Misc-exposition": 1
        }

        frequencies = classifier.get_label_frequencies()
        assert frequencies is not None
        assert frequencies["Nomenclature"] == 2
        assert frequencies["Description"] == 2
        assert frequencies["Misc-exposition"] == 1


class TestSkolClassifierV2ClassWeights:
    """Tests for class weight computation."""

    def test_get_recommended_class_weights_without_frequencies(self, spark):
        """Test that class weights return None when frequencies not computed."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            verbosity=0
        )

        weights = classifier.get_recommended_class_weights()
        assert weights is None

    def test_get_recommended_class_weights_inverse(self, spark):
        """Test inverse strategy for class weights."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            verbosity=0
        )

        # Set frequencies
        classifier._label_frequencies = {
            "Nomenclature": 10,
            "Description": 100,
            "Misc": 1000
        }

        weights = classifier.get_recommended_class_weights(strategy='inverse')

        assert weights is not None
        # Rarest class should have highest weight
        assert weights["Nomenclature"] > weights["Description"]
        assert weights["Description"] > weights["Misc"]

    def test_get_recommended_class_weights_balanced(self, spark):
        """Test balanced strategy for class weights."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            verbosity=0
        )

        classifier._label_frequencies = {
            "A": 100,
            "B": 100,
            "C": 100
        }

        weights = classifier.get_recommended_class_weights(strategy='balanced')

        assert weights is not None
        # Equal frequencies should give equal weights
        assert weights["A"] == weights["B"] == weights["C"]


class TestSkolClassifierV2PredictErrors:
    """Tests for predict method error handling."""

    def test_predict_without_model_raises_error(self, spark, sample_raw_data):
        """Test that predict raises error when model not trained."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output'
        )

        with pytest.raises(ValueError, match="Model not trained"):
            classifier.predict(sample_raw_data)


class TestSkolClassifierV2SaveModelErrors:
    """Tests for save_model method error handling."""

    def test_save_model_without_training_raises_error(self, spark):
        """Test that save_model raises error when no model trained."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            model_storage='disk',
            model_path='/tmp/model.pkl'
        )

        with pytest.raises(ValueError, match="No model to save"):
            classifier.save_model()

    def test_save_model_without_storage_raises_error(self, spark):
        """Test that save_model raises error when no storage configured."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output'
        )

        # Mock that model exists
        classifier._model = MagicMock()
        classifier._feature_pipeline = MagicMock()

        with pytest.raises(ValueError, match="model_storage not configured"):
            classifier.save_model()


class TestSkolClassifierV2LoadModelErrors:
    """Tests for load_model method error handling."""

    def test_load_model_without_storage_raises_error(self, spark):
        """Test that load_model raises error when no storage configured."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output'
        )

        with pytest.raises(ValueError, match="model_storage not configured"):
            classifier.load_model()


class TestSkolClassifierV2SaveAnnotatedErrors:
    """Tests for save_annotated method error handling."""

    def test_save_annotated_invalid_dest(self, spark):
        """Test save_annotated with invalid destination."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output'
        )

        # Override output_dest to invalid value
        classifier.output_dest = 'invalid'

        mock_predictions = MagicMock()
        with pytest.raises(ValueError, match="save_annotated.*not supported"):
            classifier.save_annotated(mock_predictions)


class TestSkolClassifierV2FormatAsStrings:
    """Tests for _format_as_strings method."""

    def test_format_as_strings_basic(self, spark):
        """Test basic string formatting."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='strings',
            output_path='/tmp/output'
        )

        # Create test predictions DataFrame with annotated_value
        data = [
            ("doc1", 1, "[@ Line 1 #Label1*]"),
            ("doc1", 2, "[@ Line 2 #Label2*]"),
        ]
        schema = StructType([
            StructField("doc_id", StringType(), False),
            StructField("line_number", IntegerType(), False),
            StructField("annotated_value", StringType(), False),
        ])
        predictions = spark.createDataFrame(data, schema)

        result = classifier._format_as_strings(predictions)

        assert isinstance(result, list)
        assert len(result) == 1  # One document


class TestSkolClassifierV2ExtractionModes:
    """Tests for different extraction modes."""

    def test_extraction_mode_line(self, spark):
        """Test line extraction mode initialization."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            extraction_mode='line'
        )

        assert classifier.extraction_mode.name == 'line'
        assert classifier.line_level is True

    def test_extraction_mode_paragraph(self, spark):
        """Test paragraph extraction mode initialization."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            extraction_mode='paragraph'
        )

        assert classifier.extraction_mode.name == 'paragraph'
        assert classifier.line_level is False

    def test_extraction_mode_section(self, spark):
        """Test section extraction mode initialization."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='couchdb',
            couchdb_url='http://localhost:5984',
            couchdb_database='test_db',
            output_dest='couchdb',
            extraction_mode='section'
        )

        assert classifier.extraction_mode.name == 'section'


class TestSkolClassifierV2WeightStrategy:
    """Tests for weight strategy auto-enable."""

    def test_weight_strategy_enables_frequency_computation(self, spark):
        """Test that weight_strategy auto-enables frequency computation."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output',
            weight_strategy='inverse'
        )

        assert classifier.compute_label_frequencies is True

    def test_no_weight_strategy_no_auto_enable(self, spark):
        """Test that no weight_strategy doesn't auto-enable frequency computation."""
        classifier = SkolClassifierV2(
            spark=spark,
            input_source='files',
            file_paths=['test/*.txt'],
            output_dest='files',
            output_path='/tmp/output'
        )

        assert classifier.compute_label_frequencies is False
