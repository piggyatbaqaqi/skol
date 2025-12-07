"""
RNN-based model for sequential line classification with context using Keras + Pandas UDFs.

This module provides an LSTM/GRU-based model that uses surrounding lines
as context to improve classification accuracy for individual lines. It uses
PySpark Pandas UDFs for distributed training instead of Elephas.
"""

from typing import Optional, List, Tuple
import numpy as np
import pickle
import tempfile
import os

# Configure TensorFlow GPU settings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, collect_list, pandas_udf, struct, array
from pyspark.sql.types import ArrayType, FloatType, BinaryType, StructType, StructField, StringType
from pyspark.ml import Transformer
import pandas as pd

try:
    import tensorflow as tf

    # Configure GPU to prevent CUDA_ERROR_INVALID_HANDLE
    # This must happen before any TensorFlow operations
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Try to enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
    except Exception as e:
        # If GPU configuration fails, try to force CPU
        print(f"GPU configuration failed: {e}. Attempting to use CPU...")
        try:
            tf.config.set_visible_devices([], 'GPU')
            print("Forced CPU-only mode")
        except Exception as e2:
            print(f"Could not force CPU mode: {e2}")

    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.utils import to_categorical

    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. RNN model will not work.")


def build_bilstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3
) -> 'keras.Model':
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


class SequencePreprocessor(Transformer):
    """
    Transformer that converts flat features into sequences grouped by document.

    This preprocessor groups lines by document ID and creates sequences with
    a maximum window size, preparing data for RNN training.
    """

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
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.docIdCol = docIdCol
        self.labelCol = labelCol
        self.window_size = window_size

    def getInputCol(self):
        """Get input column name."""
        return self.inputCol

    def getOutputCol(self):
        """Get output column name."""
        return self.outputCol

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
        window = self.window_size

        # Simply group by document and collect all features and labels
        # We'll do windowing in the fit() method after collecting
        grouped = df.groupBy(self.docIdCol).agg(
            collect_list(input_col).alias("sequence_features"),
            collect_list(self.labelCol).alias("sequence_labels")
        )

        return grouped


class RNNSkolModel:
    """
    RNN-based classifier using Pandas UDFs for distributed training.

    This implementation uses PySpark's Pandas UDFs to distribute training
    across partitions, avoiding the Elephas compatibility issues.
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
            num_workers: Number of Spark workers (unused in Pandas UDF approach)
            features_col: Name of features column
            label_col: Name of label column
        """
        if not KERAS_AVAILABLE:
            raise ImportError(
                "TensorFlow and Keras are required for RNN model. "
                "Install with: pip install tensorflow"
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

        self.classifier_model = None
        self.labels = None
        self.model_weights = None

    def fit(self, train_data: DataFrame, labels: Optional[List[str]] = None) -> 'RNNSkolModel':
        """
        Train the RNN classification model using standard Keras training.

        Since RNN training requires sequences and doesn't parallelize well across
        documents, we collect the data and train on the driver.

        Args:
            train_data: Training DataFrame with features and labels
            labels: Optional list of label strings

        Returns:
            Self (fitted model)
        """
        if labels is not None:
            self.labels = labels

        print("Preparing sequences for RNN training...")

        # Determine document ID column (CouchDB uses 'doc_id', files use 'filename')
        doc_id_col = "doc_id" if "doc_id" in train_data.columns else "filename"

        # Create sequence preprocessor
        preprocessor = SequencePreprocessor(
            inputCol=self.features_col,
            outputCol="sequence_features",
            docIdCol=doc_id_col,
            labelCol=self.label_col,
            window_size=self.window_size
        )

        # Transform data into sequences
        sequenced_data = preprocessor.transform(train_data)

        # Collect sequences to driver for training
        print("Collecting sequences for training...")
        collected = sequenced_data.collect()

        if len(collected) == 0:
            raise ValueError("No sequences generated from training data")

        # Prepare training data with windowing
        X_train = []
        y_train = []

        for row in collected:
            features = row.sequence_features
            labels_seq = row.sequence_labels

            # Convert sparse vectors to dense arrays
            dense_features = []
            for feat in features:
                if feat is None:
                    # Skip None values
                    continue
                elif hasattr(feat, 'toArray'):
                    # SparseVector from PySpark
                    dense_features.append(feat.toArray().tolist())
                elif isinstance(feat, (list, tuple)):
                    dense_features.append(list(feat))
                else:
                    # Try to convert to list
                    try:
                        dense_features.append(list(feat))
                    except TypeError:
                        # Skip if not iterable
                        continue

            # Convert labels to list
            dense_labels = [float(l) for l in labels_seq if l is not None]

            # Create windows from this document
            for i in range(0, len(dense_features), self.window_size):
                window_features = dense_features[i:i + self.window_size]
                window_labels = dense_labels[i:i + self.window_size]

                # Skip if window is too small
                if len(window_features) == 0:
                    continue

                # Pad or truncate to window_size
                if len(window_features) < self.window_size:
                    # Pad with zeros
                    padding = [[0.0] * self.input_size] * (self.window_size - len(window_features))
                    window_features = window_features + padding
                    window_labels = window_labels + [0.0] * (self.window_size - len(window_labels))
                else:
                    window_features = window_features[:self.window_size]
                    window_labels = window_labels[:self.window_size]

                X_train.append(window_features)
                y_train.append(window_labels)

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)

        # Convert labels to one-hot encoding
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)

        print(f"Training RNN model on {len(X_train)} sequences...")
        print(f"  Input shape: {X_train.shape}")
        print(f"  Output shape: {y_train_cat.shape}")

        # Train model
        self.keras_model.fit(
            X_train,
            y_train_cat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            verbose=1
        )

        # Store model weights for distribution
        self.model_weights = self.keras_model.get_weights()
        self.classifier_model = self.keras_model

        return self

    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions on data.

        Args:
            data: DataFrame to predict on

        Returns:
            DataFrame with predictions
        """
        if self.classifier_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Determine document ID column
        doc_id_col = "doc_id" if "doc_id" in data.columns else "filename"

        # Create sequence preprocessor
        preprocessor = SequencePreprocessor(
            inputCol=self.features_col,
            outputCol="sequence_features",
            docIdCol=doc_id_col,
            labelCol=self.label_col if self.label_col in data.columns else "dummy_label",
            window_size=self.window_size
        )

        # Add dummy labels if not present (for prediction)
        if self.label_col not in data.columns:
            from pyspark.sql.functions import lit
            data = data.withColumn(self.label_col, lit(0.0))

        # Transform into sequences
        sequenced_data = preprocessor.transform(data)

        # Broadcast model weights for UDF
        model_config = self.keras_model.to_json()
        model_weights = self.model_weights
        input_size = self.input_size
        window_size = self.window_size
        num_classes = self.num_classes

        # Define prediction UDF
        @pandas_udf(ArrayType(FloatType()))
        def predict_sequence(features_series: pd.Series) -> pd.Series:
            """Predict on sequences using the trained model."""
            # Rebuild model from config and weights
            model = keras.models.model_from_json(model_config)
            model.set_weights(model_weights)

            results = []
            for features in features_series:
                # Pad or truncate
                if len(features) < window_size:
                    padding = [[0.0] * input_size] * (window_size - len(features))
                    features = features + padding
                else:
                    features = features[:window_size]

                # Predict
                X = np.array([features], dtype=np.float32)
                preds = model.predict(X, verbose=0)[0]
                # Get argmax for each timestep
                pred_classes = [float(np.argmax(p)) for p in preds]
                results.append(pred_classes[:len(features_series)])

            return pd.Series(results)

        # Apply prediction
        predictions = sequenced_data.withColumn(
            "predictions",
            predict_sequence(col("sequence_features"))
        )

        return predictions

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.classifier_model is None:
            raise ValueError("No model to save")

        self.classifier_model.save(path)

    def load(self, path: str) -> 'RNNSkolModel':
        """Load model from disk."""
        self.keras_model = keras.models.load_model(path)
        self.classifier_model = self.keras_model
        self.model_weights = self.keras_model.get_weights()
        return self
