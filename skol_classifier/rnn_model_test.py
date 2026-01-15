"""
Tests for rnn_model.py module.

Run with: pytest skol_classifier/rnn_model_test.py -v

Note: These tests require TensorFlow to be installed.
Tests will be skipped if TensorFlow is not available.
"""

import pytest
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
)
from pyspark.ml.linalg import Vectors, VectorUDT

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
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
        .appName("RNNModelTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def sample_features(spark):
    """Create sample features for testing."""
    # Create simple feature vectors
    data = [
        ("doc1", 1, Vectors.dense([0.1, 0.2, 0.3, 0.4, 0.5]), 0),
        ("doc1", 2, Vectors.dense([0.2, 0.3, 0.4, 0.5, 0.6]), 1),
        ("doc1", 3, Vectors.dense([0.3, 0.4, 0.5, 0.6, 0.7]), 1),
        ("doc1", 4, Vectors.dense([0.4, 0.5, 0.6, 0.7, 0.8]), 2),
        ("doc2", 1, Vectors.dense([0.5, 0.6, 0.7, 0.8, 0.9]), 0),
        ("doc2", 2, Vectors.dense([0.6, 0.7, 0.8, 0.9, 1.0]), 1),
    ]
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("line_number", IntegerType(), False),
        StructField("combined_idf", VectorUDT(), False),
        StructField("label_indexed", IntegerType(), False),
    ])
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_features_unlabeled(spark):
    """Create sample features without labels for prediction testing."""
    data = [
        ("doc1", 1, Vectors.dense([0.1, 0.2, 0.3, 0.4, 0.5])),
        ("doc1", 2, Vectors.dense([0.2, 0.3, 0.4, 0.5, 0.6])),
        ("doc1", 3, Vectors.dense([0.3, 0.4, 0.5, 0.6, 0.7])),
    ]
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("line_number", IntegerType(), False),
        StructField("combined_idf", VectorUDT(), False),
    ])
    return spark.createDataFrame(data, schema)


class TestBuildBilstmModel:
    """Tests for build_bilstm_model function."""

    def test_build_basic_model(self):
        """Test building a basic BiLSTM model."""
        from .rnn_model import build_bilstm_model

        model = build_bilstm_model(
            input_shape=(10, 5),
            num_classes=3,
            hidden_size=32,
            num_layers=1,
            dropout=0.1
        )

        assert model is not None
        assert isinstance(model, keras.Model)

    def test_build_model_with_class_weights(self):
        """Test building model with class weights."""
        from .rnn_model import build_bilstm_model

        class_weights = {"A": 10.0, "B": 1.0, "C": 0.1}
        labels = ["A", "B", "C"]

        model = build_bilstm_model(
            input_shape=(10, 5),
            num_classes=3,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            class_weights=class_weights,
            labels=labels
        )

        assert model is not None

    def test_build_model_with_focal_labels(self):
        """Test building model with focal labels."""
        from .rnn_model import build_bilstm_model

        labels = ["A", "B", "C"]
        focal_labels = ["A", "B"]

        model = build_bilstm_model(
            input_shape=(10, 5),
            num_classes=3,
            hidden_size=32,
            num_layers=1,
            dropout=0.1,
            labels=labels,
            focal_labels=focal_labels
        )

        assert model is not None

    def test_build_model_mutual_exclusion(self):
        """Test that class_weights and focal_labels are mutually exclusive."""
        from .rnn_model import build_bilstm_model

        class_weights = {"A": 1.0, "B": 1.0}
        labels = ["A", "B"]
        focal_labels = ["A"]

        with pytest.raises(ValueError, match="mutually exclusive"):
            build_bilstm_model(
                input_shape=(10, 5),
                num_classes=2,
                class_weights=class_weights,
                labels=labels,
                focal_labels=focal_labels
            )

    def test_build_model_requires_labels_with_weights(self):
        """Test that class_weights requires labels parameter."""
        from .rnn_model import build_bilstm_model

        class_weights = {"A": 1.0, "B": 1.0}

        with pytest.raises(ValueError, match="labels parameter is required"):
            build_bilstm_model(
                input_shape=(10, 5),
                num_classes=2,
                class_weights=class_weights
            )

    def test_build_model_multiple_layers(self):
        """Test building model with multiple LSTM layers."""
        from .rnn_model import build_bilstm_model

        model = build_bilstm_model(
            input_shape=(10, 5),
            num_classes=3,
            hidden_size=64,
            num_layers=3,
            dropout=0.2
        )

        assert model is not None
        # Count Bidirectional layers
        bidirectional_layers = [
            l for l in model.layers
            if 'bidirectional' in l.name.lower()
        ]
        assert len(bidirectional_layers) == 3


class TestSequencePreprocessor:
    """Tests for SequencePreprocessor class."""

    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        from .rnn_model import SequencePreprocessor

        preprocessor = SequencePreprocessor(
            inputCol="features",
            outputCol="seq_features",
            docIdCol="doc_id",
            lineNumberCol="line_number",
            labelCol="label",
            window_size=50
        )

        assert preprocessor.inputCol == "features"
        assert preprocessor.outputCol == "seq_features"
        assert preprocessor.window_size == 50

    def test_preprocessor_get_columns(self):
        """Test preprocessor getter methods."""
        from .rnn_model import SequencePreprocessor

        preprocessor = SequencePreprocessor(
            inputCol="my_features",
            outputCol="my_output",
            lineNumberCol="my_line_no"
        )

        assert preprocessor.getInputCol() == "my_features"
        assert preprocessor.getOutputCol() == "my_output"
        assert preprocessor.getLineNoCol() == "my_line_no"

    def test_preprocessor_transform(self, spark, sample_features):
        """Test preprocessor transformation."""
        from .rnn_model import SequencePreprocessor

        preprocessor = SequencePreprocessor(
            inputCol="combined_idf",
            outputCol="sequence_features",
            docIdCol="doc_id",
            lineNumberCol="line_number",
            labelCol="label_indexed",
            window_size=50
        )

        result = preprocessor.transform(sample_features)

        # Should have grouped by doc_id
        assert result.count() == 2  # Two documents

        # Should have sorted_data column
        assert "sorted_data" in result.columns


class TestRNNSkolModel:
    """Tests for RNNSkolModel class."""

    def test_rnn_model_init(self):
        """Test RNN model initialization."""
        from .rnn_model import RNNSkolModel

        model = RNNSkolModel(
            input_size=100,
            hidden_size=64,
            num_layers=2,
            num_classes=3,
            dropout=0.3,
            window_size=50,
            batch_size=32,
            epochs=5,
            verbosity=0
        )

        assert model.input_size == 100
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.num_classes == 3
        assert model.window_size == 50

    def test_rnn_model_init_with_class_weights(self):
        """Test RNN model initialization with class weights."""
        from .rnn_model import RNNSkolModel

        class_weights = {"Label1": 10.0, "Label2": 1.0}

        model = RNNSkolModel(
            input_size=100,
            num_classes=2,
            class_weights=class_weights,
            verbosity=0
        )

        assert model.class_weights == class_weights

    def test_rnn_model_init_mutual_exclusion(self):
        """Test that class_weights and focal_labels are mutually exclusive."""
        from .rnn_model import RNNSkolModel

        with pytest.raises(ValueError, match="mutually exclusive"):
            RNNSkolModel(
                input_size=100,
                num_classes=3,
                class_weights={"A": 1.0},
                focal_labels=["A"],
                verbosity=0
            )

    def test_rnn_model_default_prediction_stride(self):
        """Test that prediction_stride defaults to window_size."""
        from .rnn_model import RNNSkolModel

        model = RNNSkolModel(
            input_size=100,
            window_size=50,
            verbosity=0
        )

        assert model.prediction_stride == 50

    def test_rnn_model_custom_prediction_stride(self):
        """Test custom prediction_stride."""
        from .rnn_model import RNNSkolModel

        model = RNNSkolModel(
            input_size=100,
            window_size=50,
            prediction_stride=25,
            verbosity=0
        )

        assert model.prediction_stride == 25

    def test_rnn_model_predict_without_fit_raises(self, spark, sample_features_unlabeled):
        """Test that predict raises error without training."""
        from .rnn_model import RNNSkolModel

        model = RNNSkolModel(
            input_size=5,
            num_classes=3,
            verbosity=0
        )

        # Clear the classifier_model to simulate untrained state
        model.classifier_model = None

        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(sample_features_unlabeled)

    def test_rnn_model_process_row_to_windows(self, spark, sample_features):
        """Test _process_row_to_windows method."""
        from .rnn_model import RNNSkolModel, SequencePreprocessor

        model = RNNSkolModel(
            input_size=5,
            num_classes=3,
            window_size=3,
            verbosity=0
        )

        # Create preprocessor and transform data
        preprocessor = SequencePreprocessor(
            inputCol="combined_idf",
            docIdCol="doc_id",
            lineNumberCol="line_number",
            labelCol="label_indexed",
            window_size=3
        )
        sequenced = preprocessor.transform(sample_features)
        row = sequenced.first()

        windows = model._process_row_to_windows(row)

        # Should have created windows
        assert len(windows) > 0

        # Each window should have features and labels
        for features, labels in windows:
            assert len(features) == 3  # window_size
            assert len(labels) == 3


class TestRNNSkolModelEdgeCases:
    """Edge case tests for RNNSkolModel."""

    def test_rnn_model_with_gpu_warning(self, capsys):
        """Test GPU warning is shown when use_gpu_in_udf is True."""
        from .rnn_model import RNNSkolModel

        model = RNNSkolModel(
            input_size=100,
            use_gpu_in_udf=True,
            verbosity=1
        )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "GPU" in captured.out

    def test_rnn_model_keras_not_available(self):
        """Test error when Keras is not available."""
        # This test verifies the import check works
        # In practice, if we're here, Keras is available
        # But we can verify the KERAS_AVAILABLE flag is set correctly
        from .rnn_model import KERAS_AVAILABLE
        assert KERAS_AVAILABLE is True


class TestRNNSkolModelSetModel:
    """Tests for RNNSkolModel.set_model method."""

    def test_set_model_updates_state(self):
        """Test that set_model properly updates model state."""
        from .rnn_model import RNNSkolModel, build_bilstm_model

        model = RNNSkolModel(
            input_size=5,
            num_classes=3,
            window_size=10,
            verbosity=0
        )

        # Create a dummy Keras model
        keras_model = build_bilstm_model(
            input_shape=(10, 5),
            num_classes=3
        )

        model.set_model(keras_model)

        assert model.keras_model is keras_model
        assert model.classifier_model is keras_model
        assert model.model_weights is not None
        assert len(model.model_weights) > 0


class TestRNNSkolModelLoadSave:
    """Tests for RNNSkolModel save/load methods."""

    def test_save_without_model_raises(self, tmp_path):
        """Test that save raises error without trained model."""
        from .rnn_model import RNNSkolModel

        model = RNNSkolModel(
            input_size=100,
            num_classes=3,
            verbosity=0
        )
        model.classifier_model = None

        with pytest.raises(ValueError, match="No model to save"):
            model.save(str(tmp_path / "model.h5"))

    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading a model."""
        from .rnn_model import RNNSkolModel, build_bilstm_model

        # Create model with trained weights
        model = RNNSkolModel(
            input_size=5,
            num_classes=3,
            window_size=10,
            verbosity=0
        )

        keras_model = build_bilstm_model(
            input_shape=(10, 5),
            num_classes=3
        )
        model.classifier_model = keras_model

        # Save model
        model_path = str(tmp_path / "test_model")
        model.save(model_path)

        # Create new model and load
        model2 = RNNSkolModel(
            input_size=5,
            num_classes=3,
            window_size=10,
            verbosity=0
        )
        model2.load(model_path)

        assert model2.classifier_model is not None
        assert model2.model_weights is not None
