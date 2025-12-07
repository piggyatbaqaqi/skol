"""
Model training module for SKOL classifier.

This module provides the SkolModel class for training and predicting
with various classification models.
"""

from typing import Optional, List
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.feature import IndexToString
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat, lit

try:
    from .rnn_model import RNNSkolModel
    RNN_AVAILABLE = True
except ImportError:
    RNN_AVAILABLE = False


class SkolModel:
    """
    Trains and applies classification models.

    Supports multiple model types: Logistic Regression, Random Forest,
    and Gradient Boosted Trees.
    """

    def __init__(
        self,
        model_type: str = "logistic",
        features_col: str = "combined_idf",
        label_col: str = "label_indexed",
        **model_params
    ):
        """
        Initialize the model.

        Args:
            model_type: Type of classifier ('logistic', 'random_forest',
                       'gradient_boosted', 'rnn')
            features_col: Name of features column
            label_col: Name of label column
            **model_params: Additional model parameters
        """
        self.model_type = model_type
        self.features_col = features_col
        self.label_col = label_col
        self.model_params = model_params
        self.classifier_model: Optional[PipelineModel] = None
        self.labels: Optional[List[str]] = model_params.get("labels", None)
        self.rnn_model: Optional['RNNSkolModel'] = None

    def build_classifier(self):
        """
        Build the classifier based on model_type.

        Returns:
            Classifier instance (or None for RNN which uses different API)

        Raises:
            ValueError: If model_type is not recognized
        """
        if self.model_type == "logistic":
            return LogisticRegression(
                family="multinomial",
                featuresCol=self.features_col,
                labelCol=self.label_col,
                maxIter=self.model_params.get("maxIter", 10),
                regParam=self.model_params.get("regParam", 0.01)
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                featuresCol=self.features_col,
                labelCol=self.label_col,
                numTrees=self.model_params.get("n_estimators", 100),
                seed=self.model_params.get("seed", 42)
            )
        elif self.model_type == "gradient_boosted":
            return GBTClassifier(
                featuresCol=self.features_col,
                labelCol=self.label_col,
                maxIter=self.model_params.get("max_iter", 50),
                maxDepth=self.model_params.get("max_depth", 5),
                seed=self.model_params.get("seed", 42)
            )
        elif self.model_type == "rnn":
            if not RNN_AVAILABLE:
                raise ValueError(
                    "RNN model requires TensorFlow and Elephas. "
                    "Install with: pip install tensorflow elephas"
                )
            # RNN model is built separately, return None
            return None
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                "Choose 'logistic', 'random_forest', 'gradient_boosted', "
                "or 'rnn'."
            )

    def fit(self, train_data: DataFrame, labels: Optional[List[str]] = None) -> Optional[PipelineModel]:
        """
        Train the classification model.

        Args:
            train_data: Training DataFrame with features
            labels: Optional list of label strings

        Returns:
            Fitted classifier pipeline model (or None for RNN)
        """
        if labels is not None:
            self.labels = labels

        if self.model_type == "rnn":
            # Build RNN model
            input_size = self.model_params.get("input_size", 1000)
            self.rnn_model = RNNSkolModel(
                input_size=input_size,
                hidden_size=self.model_params.get("hidden_size", 128),
                num_layers=self.model_params.get("num_layers", 2),
                num_classes=self.model_params.get("num_classes", 3),
                dropout=self.model_params.get("dropout", 0.3),
                window_size=self.model_params.get("window_size", 50),
                batch_size=self.model_params.get("batch_size", 32),
                epochs=self.model_params.get("epochs", 10),
                num_workers=self.model_params.get("num_workers", 4),
                features_col=self.features_col,
                label_col=self.label_col,
                verbosity=self.model_params.get("verbosity", 2)
            )
            # Fit RNN model
            self.classifier_model = self.rnn_model.fit(train_data, labels)
            return self.classifier_model
        else:
            # Traditional ML models
            classifier = self.build_classifier()
            pipeline = Pipeline(stages=[classifier])
            self.classifier_model = pipeline.fit(train_data)
            return self.classifier_model

    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions on data.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with predictions

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.classifier_model is None:
            raise ValueError("No classifier model found. Train a model first.")

        if self.model_type == "rnn" and self.rnn_model is not None:
            return self.rnn_model.predict(data)
        else:
            return self.classifier_model.transform(data)  # pyright: ignore[reportUnknownMemberType]

    def predict_with_labels(self, data: DataFrame) -> DataFrame:
        """
        Make predictions and convert indices to label strings.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with predictions including predicted_label column

        Raises:
            ValueError: If model hasn't been trained yet or labels not set
        """
        if self.classifier_model is None:
            raise ValueError("No classifier model found. Train a model first.")
        if self.labels is None:
            raise ValueError("No labels found. Train a model first.")

        if self.model_type == "rnn" and self.rnn_model is not None:
            return self.rnn_model.predict_with_labels(data)
        else:
            predictions = self.classifier_model.transform(data)

            # Convert label indices to strings
            converter = IndexToString(
                inputCol="prediction",
                outputCol="predicted_label",
                labels=self.labels
            )
            return converter.transform(predictions)

    def get_model(self) -> Optional[PipelineModel]:
        """Get the fitted model."""
        return self.classifier_model

    def set_model(self, model: PipelineModel) -> None:
        """Set the model (useful for loading)."""
        self.classifier_model = model

    def set_labels(self, labels: List[str]) -> None:
        """Set the labels (useful for loading)."""
        self.labels = labels
