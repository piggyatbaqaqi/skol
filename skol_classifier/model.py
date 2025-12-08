"""
Model training module for SKOL classifier.

This module provides specialized subclasses for different model types
using proper inheritance.
"""

from abc import abstractmethod
from typing import Optional, List
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier
)
from pyspark.sql import DataFrame

from .base_model import SkolModel


class TraditionalMLSkolModel(SkolModel):
    """
    Base class for traditional Spark ML classification models.

    Handles the common pipeline pattern used by LogisticRegression,
    RandomForest, GradientBoostedTrees, etc.
    """

    @abstractmethod
    def build_classifier(self):
        """
        Build the Spark ML classifier.

        Returns:
            Configured classifier instance
        """
        pass

    def fit(self, train_data: DataFrame, labels: Optional[List[str]] = None) -> PipelineModel:
        """
        Train the classification model using Spark ML pipeline.

        Args:
            train_data: Training DataFrame with features
            labels: Optional list of label strings

        Returns:
            Fitted PipelineModel
        """
        if labels is not None:
            self.labels = labels

        classifier = self.build_classifier()
        pipeline = Pipeline(stages=[classifier])
        self.classifier_model = pipeline.fit(train_data)
        return self.classifier_model

    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions using Spark ML pipeline.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with predictions

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if self.classifier_model is None:
            raise ValueError("No classifier model found. Train a model first.")

        return self.classifier_model.transform(data)  # pyright: ignore[reportUnknownMemberType]


class LogisticRegressionSkolModel(TraditionalMLSkolModel):
    """Logistic Regression model implementation."""

    def build_classifier(self):
        """Build Logistic Regression classifier."""
        return LogisticRegression(
            family="multinomial",
            featuresCol=self.features_col,
            labelCol=self.label_col,
            maxIter=self.model_params.get("maxIter", 10),
            regParam=self.model_params.get("regParam", 0.01)
        )


class RandomForestSkolModel(TraditionalMLSkolModel):
    """Random Forest model implementation."""

    def build_classifier(self):
        """Build Random Forest classifier."""
        # Get max_depth, only pass to RandomForest if not None
        max_depth = self.model_params.get("max_depth", None)
        kwargs = {
            "featuresCol": self.features_col,
            "labelCol": self.label_col,
            "numTrees": self.model_params.get("n_estimators", 100),
            "seed": self.model_params.get("seed", 42)
        }
        if max_depth is not None:
            kwargs["maxDepth"] = max_depth

        return RandomForestClassifier(**kwargs)


class GradientBoostedSkolModel(TraditionalMLSkolModel):
    """Gradient Boosted Trees model implementation."""

    def build_classifier(self):
        """Build Gradient Boosted Trees classifier."""
        return GBTClassifier(
            featuresCol=self.features_col,
            labelCol=self.label_col,
            maxIter=self.model_params.get("max_iter", 50),
            maxDepth=self.model_params.get("max_depth", 5),
            seed=self.model_params.get("seed", 42)
        )


def create_model(
    model_type: str = "logistic",
    features_col: str = "combined_idf",
    label_col: str = "label_indexed",
    **model_params
) -> SkolModel:
    """
    Factory function to create the appropriate model based on type.

    Args:
        model_type: Type of classifier ('logistic', 'random_forest',
                   'gradient_boosted', 'rnn')
        features_col: Name of features column
        label_col: Name of label column
        **model_params: Additional model parameters

    Returns:
        Instance of appropriate SkolModel subclass

    Raises:
        ValueError: If model_type is not recognized or RNN dependencies missing
    """
    if model_type == "logistic":
        return LogisticRegressionSkolModel(
            features_col=features_col,
            label_col=label_col,
            **model_params
        )
    elif model_type == "random_forest":
        return RandomForestSkolModel(
            features_col=features_col,
            label_col=label_col,
            **model_params
        )
    elif model_type == "gradient_boosted":
        return GradientBoostedSkolModel(
            features_col=features_col,
            label_col=label_col,
            **model_params
        )
    elif model_type == "rnn":
        try:
            from .rnn_model import RNNSkolModel
            return RNNSkolModel(
                input_size=model_params.get("input_size", 1000),
                hidden_size=model_params.get("hidden_size", 128),
                num_layers=model_params.get("num_layers", 2),
                num_classes=model_params.get("num_classes", 3),
                dropout=model_params.get("dropout", 0.3),
                window_size=model_params.get("window_size", 50),
                batch_size=model_params.get("batch_size", 32),
                epochs=model_params.get("epochs", 10),
                num_workers=model_params.get("num_workers", 4),
                features_col=features_col,
                label_col=label_col,
                verbosity=model_params.get("verbosity", 2),
                **{k: v for k, v in model_params.items()
                   if k not in ['input_size', 'hidden_size', 'num_layers', 'num_classes',
                               'dropout', 'window_size', 'batch_size', 'epochs',
                               'num_workers', 'verbosity']}
            )
        except ImportError:
            raise ValueError(
                "RNN model requires TensorFlow. "
                "Install with: pip install tensorflow"
            )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Choose 'logistic', 'random_forest', 'gradient_boosted', or 'rnn'."
        )
