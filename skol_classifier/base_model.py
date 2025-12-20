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
        self.verbosity: int = model_params.get("verbosity", 1)

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
        Calculate comprehensive evaluation statistics for predictions.

        Includes overall metrics, per-class metrics, and confusion matrix.
        Confusion matrix is printed at verbosity >= 2.

        Args:
            predictions: DataFrame with predictions and labels.
                        Expected to have columns: prediction, label_col
                        Optionally: probabilities (for loss calculation)
            verbose: Whether to print statistics (deprecated, use self.verbosity)

        Returns:
            Dictionary containing:
            - Overall: accuracy, precision, recall, f1_score, loss (if probabilities available)
            - Per-class: {class_name}_accuracy, {class_name}_precision,
                        {class_name}_recall, {class_name}_f1, {class_name}_loss,
                        {class_name}_support
            - total_predictions: total count
        """
        from pyspark.sql.functions import col, avg, udf
        from pyspark.sql.types import DoubleType
        import numpy as np

        if self.verbosity >= 3:
            print("[Stats] Calculating statistics for predictions")
            print(f"[Stats] Predictions schema: {predictions.schema}")

        # Verify required columns are present
        required_cols = {"prediction", self.label_col}
        actual_cols = set(predictions.columns)

        if not required_cols.issubset(actual_cols):
            missing = required_cols - actual_cols
            raise ValueError(
                f"Predictions DataFrame missing required columns: {missing}. "
                f"Available columns: {actual_cols}"
            )

        # Check if we have probabilities column for loss calculation
        has_probabilities = "probabilities" in actual_cols

        # If no probabilities, we can't calculate loss
        if not has_probabilities and self.verbosity >= 2:
            print("[Stats] INFO: 'probabilities' column not found. Loss metrics will not be calculated.")

        # Select columns we need for evaluation
        select_cols = ["prediction", self.label_col]
        if has_probabilities:
            select_cols.append("probabilities")

        eval_predictions = predictions.select(*select_cols)

        # Filter out any null predictions
        null_count = eval_predictions.filter(col("prediction").isNull()).count()
        if null_count > 0:
            if self.verbosity >= 1:
                print(f"[Stats] WARNING: Filtering out {null_count} null predictions")
            eval_predictions = eval_predictions.filter(col("prediction").isNotNull())

        if self.verbosity >= 3:
            print(f"[Stats] Evaluating {eval_predictions.count()} predictions")

        # Use parent class method to create evaluators and calculate overall stats
        evaluators = self._create_evaluators()

        stats = {
            'accuracy': evaluators['accuracy'].evaluate(eval_predictions),
            'precision': evaluators['precision'].evaluate(eval_predictions),
            'recall': evaluators['recall'].evaluate(eval_predictions),
            'f1_score': evaluators['f1'].evaluate(eval_predictions)
        }

        # Calculate overall loss if we have probabilities
        if has_probabilities:
            # Define UDF to calculate cross-entropy loss
            @udf(returnType=DoubleType())
            def cross_entropy_loss_udf(probabilities: Optional[List[float]], true_label: int) -> float:
                """Calculate cross-entropy loss for a single prediction."""
                if probabilities is None or len(probabilities) == 0:
                    return 0.0
                if true_label < 0 or true_label >= len(probabilities):
                    return 0.0
                # Cross-entropy: -log(p(true_class))
                prob_true_class = max(probabilities[int(true_label)], 1e-10)  # Avoid log(0)
                return float(-np.log(prob_true_class))

            # Add loss column to eval_predictions
            eval_predictions_with_loss = eval_predictions.withColumn(
                "loss",
                cross_entropy_loss_udf(col("probabilities"), col(self.label_col).cast("int"))
            )

            # Calculate average loss
            avg_loss = eval_predictions_with_loss.select(avg("loss")).first()[0]
            stats['loss'] = float(avg_loss) if avg_loss is not None else 0.0
        else:
            stats['loss'] = float('nan')
            eval_predictions_with_loss = eval_predictions

        # Determine number of classes
        # Try to get from labels first, otherwise infer from data
        if self.labels is not None:
            num_classes = len(self.labels)
        else:
            # Infer from max label index in data
            max_label = eval_predictions.agg({"prediction": "max", self.label_col: "max"}).first()
            num_classes = max(int(max_label[0] or 0), int(max_label[1] or 0)) + 1

        # Calculate per-class metrics
        per_class_stats: Dict[str, float] = {}
        class_names = self.labels if self.labels else None

        for class_idx in range(num_classes):
            # Get class name if available
            class_name = class_names[class_idx] if class_names else f"class_{class_idx}"

            # Filter predictions for this class (true positives + false negatives)
            class_actual = eval_predictions.filter(col(self.label_col) == class_idx)
            class_count = class_actual.count()

            if class_count > 0:
                # Calculate per-class accuracy (recall for this class)
                class_correct = class_actual.filter(col("prediction") == class_idx).count()
                class_accuracy = float(class_correct) / float(class_count)

                # Calculate precision: TP / (TP + FP)
                # How many predicted as this class were actually this class
                predicted_as_class = eval_predictions.filter(col("prediction") == class_idx)
                predicted_count = predicted_as_class.count()
                if predicted_count > 0:
                    true_positives = predicted_as_class.filter(col(self.label_col) == class_idx).count()
                    class_precision = float(true_positives) / float(predicted_count)
                else:
                    class_precision = 0.0

                # Calculate recall: TP / (TP + FN)
                # How many actual instances of this class were correctly predicted
                class_recall = float(class_correct) / float(class_count)

                # Calculate F1 score
                if class_precision + class_recall > 0:
                    class_f1 = 2.0 * (class_precision * class_recall) / (class_precision + class_recall)
                else:
                    class_f1 = 0.0

                # Calculate per-class loss if we have probabilities
                if has_probabilities:
                    class_actual_with_loss = eval_predictions_with_loss.filter(col(self.label_col) == class_idx)
                    class_avg_loss = class_actual_with_loss.select(avg("loss")).first()[0]
                    class_loss = float(class_avg_loss) if class_avg_loss is not None else 0.0
                else:
                    class_loss = float('nan')

                # Store per-class metrics
                per_class_stats[f"{class_name}_accuracy"] = class_accuracy
                per_class_stats[f"{class_name}_precision"] = class_precision
                per_class_stats[f"{class_name}_recall"] = class_recall
                per_class_stats[f"{class_name}_f1"] = class_f1
                per_class_stats[f"{class_name}_loss"] = class_loss
                per_class_stats[f"{class_name}_support"] = float(class_count)
            else:
                # No instances of this class in test set
                per_class_stats[f"{class_name}_accuracy"] = 0.0
                per_class_stats[f"{class_name}_precision"] = 0.0
                per_class_stats[f"{class_name}_recall"] = 0.0
                per_class_stats[f"{class_name}_f1"] = 0.0
                per_class_stats[f"{class_name}_loss"] = float('nan')
                per_class_stats[f"{class_name}_support"] = 0.0

        # Add per-class stats to overall stats
        stats.update(per_class_stats)

        # Calculate confusion matrix for additional insights
        total_count = eval_predictions.count()
        confusion_data: List[Dict[str, int]] = []
        for true_class in range(num_classes):
            for pred_class in range(num_classes):
                count = eval_predictions.filter(
                    (col(self.label_col) == true_class) & (col("prediction") == pred_class)
                ).count()
                confusion_data.append({
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'count': count
                })

        # Store confusion matrix info
        stats['total_predictions'] = float(total_count)

        # Print formatted statistics
        if self.verbosity >= 1:
            print("\n" + "=" * 70)
            print("Model Evaluation Statistics (Line-Level)")
            print("=" * 70)
            print(f"\nOverall Metrics:")
            print(f"  Accuracy:  {stats['accuracy']:.4f}")
            print(f"  Precision: {stats['precision']:.4f}")
            print(f"  Recall:    {stats['recall']:.4f}")
            print(f"  F1 Score:  {stats['f1_score']:.4f}")
            if has_probabilities:
                print(f"  Loss:      {stats['loss']:.4f}")
            print(f"  Total Predictions: {total_count}")

            # Print per-class metrics
            print(f"\nPer-Class Metrics:")
            if has_probabilities:
                print(f"{'Class':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Loss':<10} {'Support':<10}")
                print("-" * 80)
            else:
                print(f"{'Class':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
                print("-" * 70)

            for class_idx in range(num_classes):
                class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
                metric_name = class_names[class_idx] if class_names else f'class_{class_idx}'
                acc = per_class_stats.get(f"{metric_name}_accuracy", 0.0)
                prec = per_class_stats.get(f"{metric_name}_precision", 0.0)
                rec = per_class_stats.get(f"{metric_name}_recall", 0.0)
                f1 = per_class_stats.get(f"{metric_name}_f1", 0.0)
                loss = per_class_stats.get(f"{metric_name}_loss", float('nan'))
                support = per_class_stats.get(f"{metric_name}_support", 0)

                if has_probabilities:
                    print(f"{class_name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {loss:<10.4f} {support:<10.0f}")
                else:
                    print(f"{class_name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {support:<10.0f}")

            # Print confusion matrix at verbosity >= 2
            if self.verbosity >= 2:
                print(f"\nConfusion Matrix:")
                print(f"{'True \\ Pred':<15}", end="")
                for pred_class in range(num_classes):
                    pred_name = class_names[pred_class] if class_names else f"C{pred_class}"
                    print(f"{pred_name:<12}", end="")
                print()
                print("-" * (15 + 12 * num_classes))

                for true_class in range(num_classes):
                    true_name = class_names[true_class] if class_names else f"Class {true_class}"
                    print(f"{true_name:<15}", end="")
                    for pred_class in range(num_classes):
                        count = next(
                            (item['count'] for item in confusion_data
                             if item['true_class'] == true_class and item['predicted_class'] == pred_class),
                            0
                        )
                        print(f"{count:<12}", end="")
                    print()

            print("=" * 70 + "\n")

        return stats
