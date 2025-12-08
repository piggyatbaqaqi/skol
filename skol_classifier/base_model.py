"""
Base model class for SKOL classifier.

This module provides the abstract base class that all model implementations
inherit from, avoiding circular import issues.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict
from pyspark.sql import DataFrame
from pyspark.ml.feature import IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


class SkolModel(ABC):
    """
    Abstract base class for SKOL classification models.

    Defines the interface that all model types must implement.
    """

    def __init__(
        self,
        features_col: str = "combined_idf",
        label_col: str = "label_indexed",
        **model_params
    ):
        """
        Initialize the model.

        Args:
            features_col: Name of features column
            label_col: Name of label column
            **model_params: Additional model-specific parameters
        """
        self.features_col = features_col
        self.label_col = label_col
        self.model_params = model_params
        self.classifier_model: Optional[Any] = None
        self.labels: Optional[List[str]] = model_params.get("labels", None)

    @abstractmethod
    def fit(self, train_data: DataFrame, labels: Optional[List[str]] = None) -> Any:
        """
        Train the classification model.

        Args:
            train_data: Training DataFrame with features
            labels: Optional list of label strings

        Returns:
            Fitted model (type varies by implementation)
        """
        pass

    @abstractmethod
    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions on data.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with predictions
        """
        pass

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

        predictions = self.predict(data)

        # Convert label indices to strings
        converter = IndexToString(
            inputCol="prediction",
            outputCol="predicted_label",
            labels=self.labels
        )
        return converter.transform(predictions)

    def get_model(self) -> Optional[Any]:
        """Get the fitted model."""
        return self.classifier_model

    def set_model(self, model: Any) -> None:
        """Set the model (useful for loading)."""
        self.classifier_model = model

    def set_labels(self, labels: List[str]) -> None:
        """Set the labels (useful for loading)."""
        self.labels = labels

    def _create_evaluators(self) -> Dict[str, MulticlassClassificationEvaluator]:
        """
        Create evaluation metrics for this model type.

        Returns:
            Dictionary containing evaluators for various metrics
        """
        # Default implementation for standard multiclass classification
        evaluators = {
            'accuracy': MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol="prediction",
                metricName="accuracy"
            ),
            'precision': MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol="prediction",
                metricName="precisionByLabel"
            ),
            'recall': MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol="prediction",
                metricName="recallByLabel"
            ),
            'f1': MulticlassClassificationEvaluator(
                labelCol=self.label_col,
                predictionCol="prediction",
                metricName="f1"
            )
        }
        return evaluators

    def calculate_stats(
        self,
        predictions: DataFrame,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Calculate evaluation statistics for predictions.

        Args:
            predictions: DataFrame with predictions and labels
            verbose: Whether to print statistics

        Returns:
            Dictionary containing accuracy, precision, recall, f1_score
        """
        evaluators = self._create_evaluators()

        stats = {
            'accuracy': evaluators['accuracy'].evaluate(predictions),
            'precision': evaluators['precision'].evaluate(predictions),
            'recall': evaluators['recall'].evaluate(predictions),
            'f1_score': evaluators['f1'].evaluate(predictions)
        }

        if verbose:
            print(f"Test Accuracy: {stats['accuracy']:.4f}")
            print(f"Test Precision: {stats['precision']:.4f}")
            print(f"Test Recall: {stats['recall']:.4f}")
            print(f"Test F1 Score: {stats['f1_score']:.4f}")

        return stats
