"""
Tests for model.py and base_model.py modules.

Run with: pytest skol_classifier/model_test.py -v
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from .base_model import SkolModel
from .model import (
    create_model,
    LogisticRegressionSkolModel,
    RandomForestSkolModel,
    GradientBoostedSkolModel,
    TraditionalMLSkolModel
)


# Fixtures
@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing."""
    session = SparkSession.builder \
        .appName("ModelTests") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    yield session
    session.stop()


@pytest.fixture
def sample_features(spark):
    """Create sample feature data for testing models."""
    data = [
        (0.0, Vectors.dense([1.0, 0.5, 0.2, 0.1, 0.3])),
        (1.0, Vectors.dense([0.2, 0.8, 0.9, 0.7, 0.6])),
        (0.0, Vectors.dense([0.9, 0.4, 0.3, 0.2, 0.1])),
        (1.0, Vectors.dense([0.1, 0.9, 0.8, 0.6, 0.5])),
        (2.0, Vectors.dense([0.3, 0.3, 0.3, 0.8, 0.9])),
        (2.0, Vectors.dense([0.4, 0.2, 0.2, 0.9, 0.8])),
        (0.0, Vectors.dense([0.8, 0.5, 0.4, 0.1, 0.2])),
        (1.0, Vectors.dense([0.2, 0.7, 0.8, 0.5, 0.4])),
    ]
    return spark.createDataFrame(data, ["label_indexed", "combined_idf"])


class TestCreateModel:
    """Tests for create_model factory function."""

    def test_create_logistic_model(self):
        """Test creating logistic regression model."""
        model = create_model("logistic")

        assert isinstance(model, LogisticRegressionSkolModel)
        assert model.features_col == "combined_idf"
        assert model.label_col == "label_indexed"

    def test_create_random_forest_model(self):
        """Test creating random forest model."""
        model = create_model("random_forest")

        assert isinstance(model, RandomForestSkolModel)

    def test_create_gradient_boosted_model(self):
        """Test creating gradient boosted model."""
        model = create_model("gradient_boosted")

        assert isinstance(model, GradientBoostedSkolModel)

    def test_create_model_with_custom_cols(self):
        """Test creating model with custom column names."""
        model = create_model(
            "logistic",
            features_col="my_features",
            label_col="my_labels"
        )

        assert model.features_col == "my_features"
        assert model.label_col == "my_labels"

    def test_create_model_with_labels(self):
        """Test creating model with label list."""
        labels = ["Nomenclature", "Description", "Misc"]
        model = create_model("logistic", labels=labels)

        assert model.labels == labels

    def test_create_model_unknown_type(self):
        """Test that unknown model type raises error."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model("unknown_model_type")

    def test_create_model_with_class_weights(self):
        """Test creating model with class weights."""
        weights = {"Nomenclature": 10.0, "Description": 1.0, "Misc": 1.0}
        model = create_model("logistic", class_weights=weights)

        assert model.class_weights == weights

    def test_create_model_with_params(self):
        """Test creating model with additional parameters."""
        model = create_model("logistic", maxIter=20, regParam=0.1)

        assert model.model_params.get("maxIter") == 20
        assert model.model_params.get("regParam") == 0.1


class TestLogisticRegressionSkolModel:
    """Tests for LogisticRegressionSkolModel class."""

    def test_build_classifier(self):
        """Test building logistic regression classifier."""
        model = LogisticRegressionSkolModel()
        classifier = model.build_classifier()

        assert classifier is not None
        assert classifier.getFeaturesCol() == "combined_idf"
        assert classifier.getLabelCol() == "label_indexed"

    def test_fit(self, sample_features):
        """Test fitting logistic regression model."""
        model = LogisticRegressionSkolModel(
            labels=["Class0", "Class1", "Class2"],
            verbosity=0
        )

        fitted = model.fit(sample_features, labels=["Class0", "Class1", "Class2"])

        assert fitted is not None
        assert model.classifier_model is not None

    def test_predict(self, sample_features):
        """Test predictions with logistic regression."""
        model = LogisticRegressionSkolModel(verbosity=0)
        model.fit(sample_features, labels=["Class0", "Class1", "Class2"])

        predictions = model.predict(sample_features)

        assert "prediction" in predictions.columns
        assert predictions.count() == sample_features.count()

    def test_predict_without_fit(self, sample_features):
        """Test that predict raises error without fit."""
        model = LogisticRegressionSkolModel()

        with pytest.raises(ValueError, match="No classifier model found"):
            model.predict(sample_features)


class TestRandomForestSkolModel:
    """Tests for RandomForestSkolModel class."""

    def test_build_classifier(self):
        """Test building random forest classifier."""
        model = RandomForestSkolModel()
        classifier = model.build_classifier()

        assert classifier is not None

    def test_build_classifier_with_params(self):
        """Test building random forest with custom params."""
        model = RandomForestSkolModel(n_estimators=50, max_depth=10)
        classifier = model.build_classifier()

        assert classifier.getNumTrees() == 50
        assert classifier.getMaxDepth() == 10

    def test_fit_and_predict(self, sample_features):
        """Test fit and predict with random forest."""
        model = RandomForestSkolModel(verbosity=0)
        model.fit(sample_features, labels=["Class0", "Class1", "Class2"])

        predictions = model.predict(sample_features)

        assert "prediction" in predictions.columns


class TestGradientBoostedSkolModel:
    """Tests for GradientBoostedSkolModel class."""

    def test_build_classifier(self):
        """Test building GBT classifier."""
        model = GradientBoostedSkolModel()
        classifier = model.build_classifier()

        assert classifier is not None

    def test_build_classifier_with_params(self):
        """Test building GBT with custom params."""
        model = GradientBoostedSkolModel(max_iter=30, max_depth=3)
        classifier = model.build_classifier()

        assert classifier.getMaxIter() == 30
        assert classifier.getMaxDepth() == 3


class TestSkolModelBase:
    """Tests for SkolModel base class methods."""

    def test_get_model_before_fit(self):
        """Test get_model returns None before fitting."""
        model = LogisticRegressionSkolModel()
        assert model.get_model() is None

    def test_get_model_after_fit(self, sample_features):
        """Test get_model returns model after fitting."""
        model = LogisticRegressionSkolModel(verbosity=0)
        model.fit(sample_features, labels=["Class0", "Class1", "Class2"])

        assert model.get_model() is not None

    def test_set_model(self):
        """Test set_model method."""
        model = LogisticRegressionSkolModel()
        model.set_model("dummy_model")

        assert model.get_model() == "dummy_model"

    def test_set_labels(self):
        """Test set_labels method."""
        model = LogisticRegressionSkolModel()
        labels = ["A", "B", "C"]
        model.set_labels(labels)

        assert model.labels == labels

    def test_predict_with_labels(self, sample_features):
        """Test predict_with_labels method."""
        labels = ["Class0", "Class1", "Class2"]
        model = LogisticRegressionSkolModel(verbosity=0)
        model.fit(sample_features, labels=labels)

        predictions = model.predict_with_labels(sample_features)

        assert "predicted_label" in predictions.columns

        # Check that predicted labels are from the label list
        predicted_labels = predictions.select("predicted_label").distinct().collect()
        for row in predicted_labels:
            assert row["predicted_label"] in labels

    def test_predict_with_labels_no_model(self, sample_features):
        """Test predict_with_labels raises error without model."""
        model = LogisticRegressionSkolModel()

        with pytest.raises(ValueError, match="No classifier model found"):
            model.predict_with_labels(sample_features)

    def test_predict_with_labels_no_labels(self, sample_features):
        """Test predict_with_labels raises error without labels."""
        model = LogisticRegressionSkolModel(verbosity=0)
        model.fit(sample_features)  # Fit without labels
        model.labels = None  # Clear labels

        with pytest.raises(ValueError, match="No labels found"):
            model.predict_with_labels(sample_features)


class TestClassWeights:
    """Tests for class weight functionality."""

    def test_add_instance_weights(self, sample_features):
        """Test adding instance weights."""
        model = LogisticRegressionSkolModel(
            class_weights={"Class0": 10.0, "Class1": 1.0, "Class2": 5.0},
            verbosity=0
        )
        model.labels = ["Class0", "Class1", "Class2"]

        weighted_df = model._add_instance_weights(sample_features)

        assert "instance_weight" in weighted_df.columns

        # Verify weights are correct
        rows = weighted_df.select("label_indexed", "instance_weight").distinct().collect()
        weight_map = {int(row["label_indexed"]): row["instance_weight"] for row in rows}

        assert weight_map[0] == 10.0  # Class0
        assert weight_map[1] == 1.0   # Class1
        assert weight_map[2] == 5.0   # Class2

    def test_add_instance_weights_no_weights(self, sample_features):
        """Test that no weights are added when class_weights is None."""
        model = LogisticRegressionSkolModel(verbosity=0)

        result_df = model._add_instance_weights(sample_features)

        assert "instance_weight" not in result_df.columns

    def test_fit_with_class_weights(self, sample_features):
        """Test fitting model with class weights."""
        model = LogisticRegressionSkolModel(
            class_weights={"Class0": 10.0, "Class1": 1.0, "Class2": 1.0},
            verbosity=0
        )

        # Should fit without errors
        fitted = model.fit(sample_features, labels=["Class0", "Class1", "Class2"])

        assert fitted is not None


class TestCalculateStats:
    """Tests for calculate_stats method."""

    def test_calculate_stats_basic(self, sample_features):
        """Test basic statistics calculation."""
        model = LogisticRegressionSkolModel(verbosity=0)
        model.fit(sample_features, labels=["Class0", "Class1", "Class2"])

        predictions = model.predict(sample_features)

        # Calculate stats - predictions already have probabilities from predict_proba
        stats = model.calculate_stats(predictions, verbose=False)

        assert "accuracy" in stats
        assert "precision" in stats
        assert "recall" in stats
        assert "f1_score" in stats
        assert "total_predictions" in stats

        # Metrics should be between 0 and 1
        assert 0 <= stats["accuracy"] <= 1
        assert 0 <= stats["precision"] <= 1
        assert 0 <= stats["recall"] <= 1
        assert 0 <= stats["f1_score"] <= 1

    def test_calculate_stats_per_class(self, sample_features):
        """Test per-class statistics."""
        model = LogisticRegressionSkolModel(verbosity=0)
        model.fit(sample_features, labels=["Class0", "Class1", "Class2"])

        predictions = model.predict(sample_features)
        stats = model.calculate_stats(predictions, verbose=False)

        # Should have per-class metrics
        for class_name in ["Class0", "Class1", "Class2"]:
            assert f"{class_name}_accuracy" in stats
            assert f"{class_name}_precision" in stats
            assert f"{class_name}_recall" in stats
            assert f"{class_name}_f1" in stats
            assert f"{class_name}_support" in stats
