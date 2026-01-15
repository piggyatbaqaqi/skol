"""
Hybrid model combining logistic regression and RNN for two-stage prediction.

This module implements a two-stage pipeline where:
1. Logistic regression identifies Nomenclature with high confidence
2. RNN handles all other predictions (Description and Misc)

This leverages the strengths of both models:
- Logistic: Excellent at TF-IDF patterns (italicized names, author citations)
- RNN: Strong at sequential context for fragmented descriptions
"""

from typing import Optional, List, Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel

from .base_model import SkolModel
from .model import LogisticRegressionSkolModel
from .rnn_model import RNNSkolModel


class HybridSkolModel(SkolModel):
    """
    Hybrid model combining logistic regression and RNN predictions.

    Uses a two-stage approach:
    1. Logistic model predicts Nomenclature with confidence threshold
    2. RNN model handles all other cases

    This combines the best of both approaches:
    - Logistic excels at identifying Nomenclature via TF-IDF features
    - RNN handles sequential patterns for Description and Misc
    """

    def __init__(
        self,
        features_col: str = "combined_idf",
        label_col: str = "label_indexed",
        nomenclature_threshold: float = 0.6,
        input_size: int = 1000,
        logistic_params: Optional[Dict[str, Any]] = None,
        rnn_params: Optional[Dict[str, Any]] = None,
        **model_params
    ):
        """
        Initialize hybrid model with both logistic and RNN components.

        Args:
            features_col: Name of features column
            label_col: Name of label column
            nomenclature_threshold: Confidence threshold for logistic Nomenclature predictions
                                   (0.0-1.0, default 0.6)
            input_size: Size of input feature vectors for RNN (default 1000)
            logistic_params: Parameters for logistic regression model
            rnn_params: Parameters for RNN model (can override input_size)
            **model_params: Additional parameters (verbosity, etc.)
        """
        super().__init__(features_col=features_col, label_col=label_col, **model_params)

        self.nomenclature_threshold = nomenclature_threshold
        self.verbosity = model_params.get('verbosity', 2)

        # Initialize logistic model
        logistic_config = logistic_params or {}
        logistic_config.update({'verbosity': self.verbosity})
        self.logistic_model = LogisticRegressionSkolModel(
            features_col=features_col,
            label_col=label_col,
            **logistic_config
        )

        # Initialize RNN model - use input_size from rnn_params if provided, else use default
        rnn_config = rnn_params or {}
        if 'input_size' not in rnn_config:
            rnn_config['input_size'] = input_size
        rnn_config.update({'verbosity': self.verbosity})
        self.rnn_model = RNNSkolModel(
            features_col=features_col,
            label_col=label_col,
            **rnn_config
        )

        # Track which model made each prediction
        self.prediction_stats = {
            'logistic_count': 0,
            'rnn_count': 0
        }

    def fit(self, train_data: DataFrame, labels: Optional[List[str]] = None) -> 'HybridSkolModel':
        """
        Train both logistic and RNN models.

        Args:
            train_data: Training DataFrame with features
            labels: Optional list of label strings

        Returns:
            Self for method chaining
        """
        if labels is not None:
            self.labels = labels
            self.logistic_model.labels = labels
            self.rnn_model.labels = labels

        if self.verbosity >= 1:
            print("\n[Hybrid Model] Training two-stage pipeline")
            print(f"[Hybrid Model] Nomenclature threshold: {self.nomenclature_threshold}")
            print(f"[Hybrid Model] Stage 1: Training Logistic Regression for Nomenclature detection")

        # Train logistic model
        self.logistic_model.fit(train_data, labels=self.labels)

        if self.verbosity >= 1:
            print(f"[Hybrid Model] Stage 2: Training RNN for Description/Misc classification")

        # Train RNN model
        self.rnn_model.fit(train_data, labels=self.labels)

        if self.verbosity >= 1:
            print("[Hybrid Model] Two-stage pipeline training completed\n")

        return self

    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions using two-stage pipeline.

        Stage 1: Get logistic Nomenclature predictions with high confidence
        Stage 2: Use RNN for all other cases

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with 'prediction' column and metadata about which model was used
        """
        if self.labels is None:
            raise ValueError("Model must be fitted before making predictions")

        # Find Nomenclature index
        try:
            nomenclature_idx = self.labels.index('Nomenclature')
        except ValueError:
            raise ValueError("'Nomenclature' not found in labels. Hybrid model requires Nomenclature class.")

        if self.verbosity >= 2:
            print(f"[Hybrid Predict] Using two-stage prediction")
            print(f"[Hybrid Predict]   Logistic threshold for Nomenclature: {self.nomenclature_threshold}")
            print(f"[Hybrid Predict]   Nomenclature class index: {nomenclature_idx}")

        # Stage 1: Get logistic probabilities
        logistic_probs = self.logistic_model.predict_proba(data)

        # Extract Nomenclature confidence
        @udf(DoubleType())
        def get_nomenclature_confidence(probs):
            """Extract Nomenclature probability from probability vector."""
            if probs is None or len(probs) <= nomenclature_idx:
                return 0.0
            return float(probs[nomenclature_idx])

        logistic_with_conf = logistic_probs.withColumn(
            'nomenclature_conf',
            get_nomenclature_confidence(col('probabilities'))
        )

        # Identify high-confidence Nomenclature predictions
        high_conf_mask = col('nomenclature_conf') > self.nomenclature_threshold

        # Create prediction source indicator
        logistic_with_conf = logistic_with_conf.withColumn(
            'prediction_source',
            when(high_conf_mask, lit('logistic')).otherwise(lit('rnn'))
        )

        # Stage 2: Get RNN predictions for low-confidence cases
        rnn_preds = self.rnn_model.predict(data)

        # Join predictions
        # Keep line_number and filename for joining
        logistic_subset = logistic_with_conf.select(
            'filename', 'line_number',
            col('probabilities').alias('logistic_probabilities'),
            'nomenclature_conf',
            'prediction_source'
        )

        rnn_subset = rnn_preds.select(
            'filename', 'line_number',
            col('prediction').alias('rnn_prediction'),
            col('probabilities').alias('rnn_probabilities')
        )

        # Join on filename and line_number
        combined = data.join(
            logistic_subset,
            on=['filename', 'line_number'],
            how='left'
        ).join(
            rnn_subset,
            on=['filename', 'line_number'],
            how='left'
        )

        # Select final prediction based on confidence
        final_predictions = combined.withColumn(
            'prediction',
            when(
                col('nomenclature_conf') > self.nomenclature_threshold,
                lit(float(nomenclature_idx))  # Use logistic's Nomenclature prediction
            ).otherwise(
                col('rnn_prediction')  # Use RNN's prediction
            )
        ).withColumn(
            'probabilities',
            when(
                col('nomenclature_conf') > self.nomenclature_threshold,
                col('logistic_probabilities')
            ).otherwise(
                col('rnn_probabilities')
            )
        )

        # Track prediction statistics
        if self.verbosity >= 2:
            source_counts = final_predictions.groupBy('prediction_source').count().collect()
            for row in source_counts:
                count = row['count']
                source = row['prediction_source']
                if source == 'logistic':
                    self.prediction_stats['logistic_count'] = count
                else:
                    self.prediction_stats['rnn_count'] = count

            total = sum(self.prediction_stats.values())
            if total > 0:
                logistic_pct = 100.0 * self.prediction_stats['logistic_count'] / total
                rnn_pct = 100.0 * self.prediction_stats['rnn_count'] / total
                print(f"[Hybrid Predict] Prediction sources:")
                print(f"[Hybrid Predict]   Logistic (Nomenclature): {self.prediction_stats['logistic_count']} ({logistic_pct:.1f}%)")
                print(f"[Hybrid Predict]   RNN (Description/Misc): {self.prediction_stats['rnn_count']} ({rnn_pct:.1f}%)")

        return final_predictions

    def predict_proba(self, data: DataFrame) -> DataFrame:
        """
        Get probability predictions using two-stage pipeline.

        Returns probabilities from whichever model made the final prediction.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with 'probabilities' column
        """
        # Use predict() which already computes probabilities
        return self.predict(data)

    def save(self, path: str) -> None:
        """
        Save both models to disk.

        Args:
            path: Base path for saving models
                  Logistic saved to: {path}/logistic/
                  RNN saved to: {path}/rnn/
        """
        from pathlib import Path

        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Save logistic model
        logistic_path = base_path / "logistic"
        logistic_path.mkdir(exist_ok=True)

        if self.logistic_model.classifier_model is not None:
            self.logistic_model.classifier_model.write().overwrite().save(str(logistic_path))

        # Save RNN model
        rnn_path = base_path / "rnn"
        rnn_path.mkdir(exist_ok=True)
        self.rnn_model.save(str(rnn_path))

        # Save hybrid model metadata
        import json
        metadata = {
            'nomenclature_threshold': self.nomenclature_threshold,
            'labels': self.labels,
            'features_col': self.features_col,
            'label_col': self.label_col
        }
        with open(base_path / "hybrid_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.verbosity >= 1:
            print(f"[Hybrid Model] Saved to {path}")
            print(f"[Hybrid Model]   Logistic: {logistic_path}")
            print(f"[Hybrid Model]   RNN: {rnn_path}")

    def load(self, path: str) -> None:
        """
        Load both models from disk.

        Args:
            path: Base path where models were saved
        """
        from pathlib import Path
        from pyspark.ml import PipelineModel

        base_path = Path(path)

        # Load metadata
        import json
        with open(base_path / "hybrid_metadata.json", 'r') as f:
            metadata = json.load(f)

        self.nomenclature_threshold = metadata['nomenclature_threshold']
        self.labels = metadata['labels']
        self.features_col = metadata['features_col']
        self.label_col = metadata['label_col']

        # Load logistic model
        logistic_path = base_path / "logistic"
        self.logistic_model.classifier_model = PipelineModel.load(str(logistic_path))
        self.logistic_model.labels = self.labels

        # Load RNN model
        rnn_path = base_path / "rnn"
        self.rnn_model.load(str(rnn_path))
        self.rnn_model.labels = self.labels

        if self.verbosity >= 1:
            print(f"[Hybrid Model] Loaded from {path}")
            print(f"[Hybrid Model]   Nomenclature threshold: {self.nomenclature_threshold}")
