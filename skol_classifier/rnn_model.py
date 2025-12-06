"""
RNN-based model for sequential line classification with context using Keras + Elephas.

This module provides an LSTM/GRU-based model that uses surrounding lines
as context to improve classification accuracy for individual lines. It integrates
with PySpark using Elephas for distributed training.
"""

from typing import Optional, List, Dict, Tuple
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, collect_list, struct, array, size, explode, posexplode
from pyspark.ml import Pipeline, PipelineModel, Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.utils import to_categorical
    from elephas.ml_model import ElephasEstimator
    ELEPHAS_AVAILABLE = True
except ImportError:
    ELEPHAS_AVAILABLE = False
    print("Warning: TensorFlow or Elephas not available. RNN model will not work.")


def build_bilstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3
) -> keras.Model:
    """
    Build a Bidirectional LSTM model for sequence classification.

    Args:
        input_shape: Shape of input (sequence_length, feature_dim)
        num_classes: Number of output classes
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    model = models.Sequential()

    # First LSTM layer
    model.add(layers.Bidirectional(
        layers.LSTM(hidden_size, return_sequences=True, dropout=dropout),
        input_shape=input_shape
    ))

    # Additional LSTM layers
    for _ in range(num_layers - 1):
        model.add(layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True, dropout=dropout)
        ))

    # Time-distributed dense layer for per-timestep classification
    model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='softmax')))

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


class SequencePreprocessor(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """
    Transformer that converts flat features into sequences grouped by document.

    This preprocessor groups lines by document ID and creates sequences with
    a maximum window size, preparing data for RNN training.
    """

    window_size = Param(
        Params._dummy(),
        "window_size",
        "Maximum sequence length for windowing long documents"
    )

    def __init__(
        self,
        inputCol: str = "features",
        outputCol: str = "sequence_features",
        docIdCol: str = "doc_id",
        labelCol: str = "label_indexed",
        window_size: int = 50
    ):
        """
        Initialize the sequence preprocessor.

        Args:
            inputCol: Column containing feature vectors
            outputCol: Column for output sequences
            docIdCol: Column containing document IDs
            labelCol: Column containing labels
            window_size: Maximum sequence length
        """
        super(SequencePreprocessor, self).__init__()
        self._setDefault(
            inputCol=inputCol,
            outputCol=outputCol,
            window_size=window_size
        )
        self.setParams(
            inputCol=inputCol,
            outputCol=outputCol,
            docIdCol=docIdCol,
            labelCol=labelCol,
            window_size=window_size
        )
        self.docIdCol = docIdCol
        self.labelCol = labelCol

    def setParams(
        self,
        inputCol: str = "features",
        outputCol: str = "sequence_features",
        docIdCol: str = "doc_id",
        labelCol: str = "label_indexed",
        window_size: int = 50
    ):
        """Set parameters."""
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Transform flat features into sequences.

        Groups by document ID and creates sequences with windowing for long documents.

        Args:
            df: Input DataFrame with columns: doc_id, features, label_indexed

        Returns:
            DataFrame with sequence_features and sequence_labels columns
        """
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        window = self.getOrDefault(self.window_size)

        # Group by document and collect features and labels
        grouped = df.groupBy(self.docIdCol).agg(
            collect_list(input_col).alias("feature_list"),
            collect_list(self.labelCol).alias("label_list")
        )

        # Window long sequences
        from pyspark.sql.functions import udf, explode
        from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType

        def window_sequence(features, labels):
            """Split long sequences into windows."""
            results = []
            for i in range(0, len(features), window):
                end_idx = min(i + window, len(features))
                results.append((features[i:end_idx], labels[i:end_idx]))
            return results

        window_schema = ArrayType(
            StructType([
                StructField("features", ArrayType(df.schema[input_col].dataType)),
                StructField("labels", ArrayType(df.schema[self.labelCol].dataType))
            ])
        )

        window_udf = udf(window_sequence, window_schema)

        # Apply windowing and explode
        windowed = grouped.withColumn(
            "windows",
            window_udf(col("feature_list"), col("label_list"))
        )

        result = windowed.select(
            self.docIdCol,
            explode(col("windows")).alias("window")
        ).select(
            self.docIdCol,
            col("window.features").alias(output_col),
            col("window.labels").alias("sequence_labels")
        )

        return result


class RNNSkolModel:
    """
    RNN-based classifier for line-level classification with sequential context.

    This classifier uses a Bidirectional LSTM to process sequences of lines,
    allowing it to use surrounding lines as context when classifying each line.
    Integrates with PySpark using Elephas for distributed training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        window_size: int = 50,
        batch_size: int = 32,
        epochs: int = 10,
        num_workers: int = 4,
        features_col: str = "combined_idf",
        label_col: str = "label_indexed"
    ):
        """
        Initialize RNN classifier.

        Args:
            input_size: Size of input feature vectors
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            window_size: Maximum sequence length for windowing
            batch_size: Batch size for training
            epochs: Number of training epochs
            num_workers: Number of Spark workers for distributed training
            features_col: Name of features column
            label_col: Name of label column
        """
        if not ELEPHAS_AVAILABLE:
            raise ImportError(
                "TensorFlow and Elephas are required for RNN model. "
                "Install with: pip install tensorflow elephas"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.features_col = features_col
        self.label_col = label_col

        # Build Keras model
        self.keras_model = build_bilstm_model(
            input_shape=(window_size, input_size),
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # Elephas estimator (will be created during fit)
        self.elephas_estimator: Optional[ElephasEstimator] = None
        self.classifier_model: Optional[PipelineModel] = None
        self.labels: Optional[List[str]] = None

    def fit(self, train_data: DataFrame, labels: Optional[List[str]] = None) -> PipelineModel:
        """
        Train the RNN classification model using Elephas for distributed training.

        Args:
            train_data: Training DataFrame with features and labels
            labels: Optional list of label strings

        Returns:
            Fitted classifier pipeline model
        """
        if labels is not None:
            self.labels = labels

        # Create sequence preprocessor
        preprocessor = SequencePreprocessor(
            inputCol=self.features_col,
            outputCol="sequence_features",
            docIdCol="doc_id",
            labelCol=self.label_col,
            window_size=self.window_size
        )

        # Create Elephas estimator
        self.elephas_estimator = ElephasEstimator()
        self.elephas_estimator.set_keras_model_config(self.keras_model.to_json())
        self.elephas_estimator.set_optimizer_config(
            optimizers.Adam(learning_rate=0.001).get_config()
        )
        self.elephas_estimator.set_mode("synchronous")
        self.elephas_estimator.set_loss("categorical_crossentropy")
        self.elephas_estimator.set_metrics(['accuracy'])
        self.elephas_estimator.set_epochs(self.epochs)
        self.elephas_estimator.set_batch_size(self.batch_size)
        self.elephas_estimator.set_num_workers(self.num_workers)
        self.elephas_estimator.set_features_col("sequence_features")
        self.elephas_estimator.set_label_col("sequence_labels")

        # Build pipeline
        pipeline = Pipeline(stages=[preprocessor, self.elephas_estimator])

        # Fit pipeline
        print("Training RNN model with Elephas...")
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
        return self.classifier_model.transform(data)

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

        predictions = self.classifier_model.transform(data)

        # Convert label indices to strings
        from pyspark.ml.feature import IndexToString

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
