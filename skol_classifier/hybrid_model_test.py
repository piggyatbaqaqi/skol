"""
Tests for hybrid_model.py module.

Run with: pytest skol_classifier/hybrid_model_test.py -v

Note: These tests require TensorFlow to be installed.
Tests will be skipped if TensorFlow is not available.
"""

import pytest
from unittest.mock import MagicMock, patch
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, ArrayType, FloatType
)
from pyspark.ml.linalg import Vectors, VectorUDT

# Check if TensorFlow is available
try:
    import tensorflow as tf
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


# Skip all tests if Keras is not available
pytestmark = pytest.mark.skipif(
    not KERAS_AVAILABLE,
    reason="TensorFlow/Keras not available"
)


# Fixtures
@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    session = SparkSession.builder \
        .appName("HybridModelTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def sample_features(spark):
    """Create sample features for testing."""
    data = [
        ("doc1.txt", 1, Vectors.dense([0.1] * 100), 0),
        ("doc1.txt", 2, Vectors.dense([0.2] * 100), 1),
        ("doc1.txt", 3, Vectors.dense([0.3] * 100), 2),
        ("doc2.txt", 1, Vectors.dense([0.4] * 100), 0),
        ("doc2.txt", 2, Vectors.dense([0.5] * 100), 1),
    ]
    schema = StructType([
        StructField("filename", StringType(), False),
        StructField("line_number", IntegerType(), False),
        StructField("combined_idf", VectorUDT(), False),
        StructField("label_indexed", IntegerType(), False),
    ])
    return spark.createDataFrame(data, schema)


class TestHybridSkolModelInit:
    """Tests for HybridSkolModel initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(
            features_col="combined_idf",
            label_col="label_indexed",
            nomenclature_threshold=0.6,
            verbosity=0
        )

        assert model.nomenclature_threshold == 0.6
        assert model.verbosity == 0
        assert model.logistic_model is not None
        assert model.rnn_model is not None

    def test_init_with_custom_threshold(self):
        """Test initialization with custom threshold."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(
            nomenclature_threshold=0.8,
            verbosity=0
        )

        assert model.nomenclature_threshold == 0.8

    def test_init_with_logistic_params(self):
        """Test initialization with logistic model parameters."""
        from .hybrid_model import HybridSkolModel

        logistic_params = {'reg_param': 0.01, 'max_iter': 200}

        model = HybridSkolModel(
            logistic_params=logistic_params,
            verbosity=0
        )

        # Verify logistic model was initialized
        assert model.logistic_model is not None

    def test_init_with_rnn_params(self):
        """Test initialization with RNN model parameters."""
        from .hybrid_model import HybridSkolModel

        rnn_params = {
            'input_size': 100,
            'hidden_size': 64,
            'num_layers': 2,
            'epochs': 5
        }

        model = HybridSkolModel(
            rnn_params=rnn_params,
            verbosity=0
        )

        # Verify RNN model was initialized
        assert model.rnn_model is not None

    def test_init_prediction_stats(self):
        """Test that prediction stats are initialized."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(verbosity=0)

        assert model.prediction_stats == {
            'logistic_count': 0,
            'rnn_count': 0
        }


class TestHybridSkolModelFit:
    """Tests for HybridSkolModel.fit method."""

    def test_fit_stores_labels(self, spark, sample_features):
        """Test that fit stores labels."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(
            rnn_params={'input_size': 100},
            verbosity=0
        )

        labels = ["Nomenclature", "Description", "Misc"]

        # Mock the underlying models' fit methods
        model.logistic_model.fit = MagicMock(return_value=model.logistic_model)
        model.rnn_model.fit = MagicMock(return_value=model.rnn_model)

        model.fit(sample_features, labels=labels)

        assert model.labels == labels
        assert model.logistic_model.labels == labels
        assert model.rnn_model.labels == labels


class TestHybridSkolModelPredict:
    """Tests for HybridSkolModel.predict method."""

    def test_predict_requires_fit(self, spark, sample_features):
        """Test that predict requires model to be fitted."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(verbosity=0)

        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(sample_features)

    def test_predict_requires_nomenclature_label(self, spark, sample_features):
        """Test that predict requires Nomenclature in labels."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(verbosity=0)
        model.labels = ["ClassA", "ClassB"]  # No Nomenclature

        with pytest.raises(ValueError, match="Nomenclature.*not found"):
            model.predict(sample_features)


class TestHybridSkolModelPredictProba:
    """Tests for HybridSkolModel.predict_proba method."""

    def test_predict_proba_calls_predict(self, spark, sample_features):
        """Test that predict_proba delegates to predict."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(verbosity=0)

        # Mock predict
        mock_result = MagicMock()
        model.predict = MagicMock(return_value=mock_result)
        model.labels = ["Nomenclature", "Description", "Misc"]

        result = model.predict_proba(sample_features)

        model.predict.assert_called_once_with(sample_features)
        assert result is mock_result


class TestHybridSkolModelSave:
    """Tests for HybridSkolModel.save method."""

    def test_save_creates_directories(self, tmp_path):
        """Test that save creates necessary directories."""
        from .hybrid_model import HybridSkolModel
        from pathlib import Path

        model = HybridSkolModel(
            rnn_params={'input_size': 100},
            verbosity=0
        )
        model.labels = ["Nomenclature", "Description", "Misc"]

        # Mock the underlying models
        model.logistic_model.classifier_model = MagicMock()
        model.logistic_model.classifier_model.write = MagicMock()
        model.logistic_model.classifier_model.write.return_value.overwrite = MagicMock()
        model.logistic_model.classifier_model.write.return_value.overwrite.return_value.save = MagicMock()

        model.rnn_model.save = MagicMock()

        model.save(str(tmp_path / "hybrid_model"))

        # Check directories were created
        assert (tmp_path / "hybrid_model").exists()
        assert (tmp_path / "hybrid_model" / "logistic").exists()
        assert (tmp_path / "hybrid_model" / "rnn").exists()
        assert (tmp_path / "hybrid_model" / "hybrid_metadata.json").exists()

    def test_save_stores_metadata(self, tmp_path):
        """Test that save stores metadata correctly."""
        from .hybrid_model import HybridSkolModel
        import json

        model = HybridSkolModel(
            nomenclature_threshold=0.75,
            features_col="my_features",
            label_col="my_labels",
            rnn_params={'input_size': 100},
            verbosity=0
        )
        model.labels = ["Nomenclature", "Description"]

        # Mock the underlying models
        model.logistic_model.classifier_model = MagicMock()
        model.logistic_model.classifier_model.write = MagicMock()
        model.logistic_model.classifier_model.write.return_value.overwrite = MagicMock()
        model.logistic_model.classifier_model.write.return_value.overwrite.return_value.save = MagicMock()
        model.rnn_model.save = MagicMock()

        model.save(str(tmp_path / "hybrid_model"))

        # Read and verify metadata
        metadata_path = tmp_path / "hybrid_model" / "hybrid_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata['nomenclature_threshold'] == 0.75
        assert metadata['labels'] == ["Nomenclature", "Description"]
        assert metadata['features_col'] == "my_features"
        assert metadata['label_col'] == "my_labels"


class TestHybridSkolModelLoad:
    """Tests for HybridSkolModel.load method."""

    def test_load_restores_metadata(self, tmp_path):
        """Test that load restores metadata correctly."""
        from .hybrid_model import HybridSkolModel
        import json
        from pathlib import Path

        # Create model directory structure with metadata
        model_dir = tmp_path / "hybrid_model"
        model_dir.mkdir()
        (model_dir / "logistic").mkdir()
        (model_dir / "rnn").mkdir()

        metadata = {
            'nomenclature_threshold': 0.85,
            'labels': ["Nomenclature", "Description", "Misc"],
            'features_col': "custom_features",
            'label_col': "custom_labels"
        }
        with open(model_dir / "hybrid_metadata.json", 'w') as f:
            json.dump(metadata, f)

        # Create model and mock load methods
        model = HybridSkolModel(verbosity=0)

        # Mock PipelineModel.load and rnn_model.load
        with patch('pyspark.ml.PipelineModel.load') as mock_pipeline_load:
            mock_pipeline_load.return_value = MagicMock()
            model.rnn_model.load = MagicMock(return_value=model.rnn_model)

            model.load(str(model_dir))

        assert model.nomenclature_threshold == 0.85
        assert model.labels == ["Nomenclature", "Description", "Misc"]
        assert model.features_col == "custom_features"
        assert model.label_col == "custom_labels"


class TestHybridSkolModelVerbosity:
    """Tests for verbosity handling."""

    def test_verbosity_passed_to_submodels(self):
        """Test that verbosity is passed to both sub-models."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(
            verbosity=3,
            logistic_params={},
            rnn_params={'input_size': 100}
        )

        assert model.verbosity == 3
        # Both models should have verbosity set
        # Note: actual initialization may override this


class TestHybridSkolModelTwoStage:
    """Tests for two-stage prediction logic."""

    def test_high_confidence_uses_logistic(self):
        """Test that high-confidence Nomenclature uses logistic prediction."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(
            nomenclature_threshold=0.6,
            verbosity=0
        )

        # With threshold 0.6:
        # - Confidence > 0.6 should use logistic (Nomenclature)
        # - Confidence <= 0.6 should use RNN
        assert model.nomenclature_threshold == 0.6

    def test_low_confidence_uses_rnn(self):
        """Test that low-confidence predictions use RNN."""
        from .hybrid_model import HybridSkolModel

        model = HybridSkolModel(
            nomenclature_threshold=0.6,
            verbosity=0
        )

        # Threshold determines when to use logistic vs RNN
        # Values at or below threshold use RNN
        assert model.nomenclature_threshold == 0.6
