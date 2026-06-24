"""
Model training module for SKOL classifier.

This module provides specialized subclasses for different model types
using proper inheritance.
"""

from abc import abstractmethod
from typing import Optional, List, Dict
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier
)
from pyspark.sql import DataFrame
from pyspark.sql.functions import when, col

from .base_model import SkolModel


class TraditionalMLSkolModel(SkolModel):
    """
    Base class for traditional Spark ML classification models.

    Handles the common pipeline pattern used by LogisticRegression,
    RandomForest, GradientBoostedTrees, etc.
    """

    def __init__(
        self,
        features_col: str = "combined_idf",
        label_col: str = "label_indexed",
        **model_params
    ):
        """Initialize the model with optional class weights."""
        super().__init__(features_col=features_col, label_col=label_col, **model_params)
        self.class_weights: Optional[Dict[str, float]] = model_params.get("class_weights", None)
        self.weight_col = "instance_weight" if self.class_weights is not None else None
        self._last_predictions: Optional[DataFrame] = None  # Cache for stats calculation

    @abstractmethod
    def build_classifier(self):
        """
        Build the Spark ML classifier.

        Returns:
            Configured classifier instance
        """
        pass

    def _add_instance_weights(self, data: DataFrame) -> DataFrame:
        """
        Add instance weight column based on class weights.

        Args:
            data: DataFrame with label_col

        Returns:
            DataFrame with instance_weight column added
        """
        if self.class_weights is None or self.labels is None or self.weight_col is None:
            return data

        if len(self.labels) == 0:
            return data

        # Build a when-otherwise chain to map label indices to weights
        # Start with the first label
        weight_expr = None
        for idx, label in enumerate(self.labels):
            weight = self.class_weights.get(label, 1.0)
            if weight_expr is None:
                weight_expr = when(col(self.label_col) == idx, weight)
            else:
                weight_expr = weight_expr.when(col(self.label_col) == idx, weight)

        # Add default weight of 1.0 for any unmatched labels
        # weight_expr is guaranteed to be not None here due to len check above
        assert weight_expr is not None
        weight_expr = weight_expr.otherwise(1.0)

        return data.withColumn(self.weight_col, weight_expr)

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

        # Add instance weights if class weights are specified
        if self.class_weights is not None:
            if self.labels is None:
                raise ValueError("Labels must be provided to use class weights")
            train_data = self._add_instance_weights(train_data)
            if hasattr(self, 'verbosity') and self.model_params.get('verbosity', 0) >= 1:
                print(f"\n[{self.__class__.__name__}] Using class weights:")
                sorted_weights = sorted(self.class_weights.items(), key=lambda x: x[1], reverse=True)
                for label, weight in sorted_weights:
                    print(f"  {label:<20} {weight:>6.2f}")
                print()

        classifier = self.build_classifier()
        pipeline = Pipeline(stages=[classifier])
        self.classifier_model = pipeline.fit(train_data)
        return self.classifier_model

    def predict_proba(self, data: DataFrame) -> DataFrame:
        """
        Make predictions with probabilities using Spark ML pipeline.

        The results are cached in self._last_predictions for use by calculate_stats().

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with predictions and probabilities column

        Raises:
            ValueError: If model hasn't been trained yet
        """
        from pyspark.sql.functions import udf, col
        from pyspark.sql.types import ArrayType, DoubleType

        if self.classifier_model is None:
            raise ValueError("No classifier model found. Train a model first.")

        # Transform returns prediction and probability columns
        predictions = self.classifier_model.transform(data)  # pyright: ignore[reportUnknownMemberType]

        # Convert Spark ML's 'probability' Vector column to 'probabilities' array
        # for compatibility with calculate_stats()
        if "probability" in predictions.columns:
            @udf(returnType=ArrayType(DoubleType()))
            def vector_to_array(v):  # pyright: ignore[reportUnknownParameterType]
                """Convert Spark ML Vector to list of floats."""
                if v is None:
                    return []
                return [float(x) for x in v.toArray()]  # pyright: ignore[reportUnknownMemberType]

            predictions = predictions.withColumn(
                "probabilities",
                vector_to_array(col("probability"))
            )

        # Cache for stats calculation
        self._last_predictions = predictions

        return predictions

    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions using Spark ML pipeline.

        Uses predict_proba() internally and caches results for calculate_stats().

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with predictions (includes probabilities column)

        Raises:
            ValueError: If model hasn't been trained yet
        """
        # Use predict_proba which caches the full results
        return self.predict_proba(data)


class LogisticRegressionSkolModel(TraditionalMLSkolModel):
    """Logistic Regression model implementation."""

    def build_classifier(self):
        """Build Logistic Regression classifier."""
        kwargs = {
            "family": "multinomial",
            "featuresCol": self.features_col,
            "labelCol": self.label_col,
            "maxIter": self.model_params.get("maxIter", 10),
            "regParam": self.model_params.get("regParam", 0.01)
        }

        # Add weight column if class weights are specified
        if self.weight_col is not None:
            kwargs["weightCol"] = self.weight_col

        return LogisticRegression(**kwargs)


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

        # Add weight column if class weights are specified
        if self.weight_col is not None:
            kwargs["weightCol"] = self.weight_col

        return RandomForestClassifier(**kwargs)


class GradientBoostedSkolModel(TraditionalMLSkolModel):
    """Gradient Boosted Trees model implementation."""

    def build_classifier(self):
        """Build Gradient Boosted Trees classifier."""
        kwargs = {
            "featuresCol": self.features_col,
            "labelCol": self.label_col,
            "maxIter": self.model_params.get("max_iter", 50),
            "maxDepth": self.model_params.get("max_depth", 5),
            "seed": self.model_params.get("seed", 42)
        }

        # Add weight column if class weights are specified
        if self.weight_col is not None:
            kwargs["weightCol"] = self.weight_col

        return GBTClassifier(**kwargs)


def create_model(
    model_type: str = "logistic",
    features_col: str = "combined_idf",
    label_col: str = "label_indexed",
    labels: Optional[List[str]] = None,
    **model_params
) -> SkolModel:
    """
    Factory function to create the appropriate model based on type.

    Args:
        model_type: Type of classifier ('logistic', 'random_forest',
                   'gradient_boosted')
        features_col: Name of features column
        label_col: Name of label column
        labels: Optional list of label strings (e.g., ["Nomenclature", "Description", "Misc"])
                Required for any model using class weights
        **model_params: Additional model parameters
                       Can include:
                       - 'class_weights': dict mapping label strings to weights

    Returns:
        Instance of appropriate SkolModel subclass

    Raises:
        ValueError: If model_type is not recognized

    Notes:
        Loss functions for handling class imbalance:
        - Logistic/RandomForest/GBT: 'class_weights' converted to instance
          weights via weightCol
    """
    if model_type == "logistic":
        model = LogisticRegressionSkolModel(
            features_col=features_col,
            label_col=label_col,
            **model_params
        )
        if labels is not None:
            model.labels = labels
        return model
    elif model_type == "random_forest":
        model = RandomForestSkolModel(
            features_col=features_col,
            label_col=label_col,
            **model_params
        )
        if labels is not None:
            model.labels = labels
        return model
    elif model_type == "gradient_boosted":
        model = GradientBoostedSkolModel(
            features_col=features_col,
            label_col=label_col,
            **model_params
        )
        if labels is not None:
            model.labels = labels
        return model
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Choose 'logistic', 'random_forest', or 'gradient_boosted'."
        )
