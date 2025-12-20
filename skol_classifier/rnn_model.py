"""
RNN-based model for sequential line classification with context using Keras + Pandas UDFs.

This module provides an LSTM/GRU-based model that uses surrounding lines
as context to improve classification accuracy for individual lines. It uses
PySpark Pandas UDFs for distributed training instead of Elephas.
"""

import gc
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import os

# Configure TensorFlow GPU settings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, collect_list, lit, pandas_udf
)
from pyspark.sql.types import ArrayType, FloatType

import pandas as pd

from .base_model import SkolModel

try:
    import tensorflow as tf

    # Configure GPU to prevent CUDA_ERROR_INVALID_HANDLE
    # This must happen before any TensorFlow operations
    USE_CPU_ONLY = False
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Check GPU details to see if compute capability is supported
            print(f"Detected GPUs: {[gpu.name for gpu in gpus]}")
            if not USE_CPU_ONLY:
                # Try to enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
            else:
                tf.config.set_visible_devices([], 'GPU')
                print("Forced CPU-only mode due to GPU compatibility issues")
        else:
            print("No GPUs detected, using CPU")
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
    dropout: float = 0.3,
    class_weights: Optional[Dict[str, float]] = None,
    labels: Optional[List[str]] = None,
    focal_labels: Optional[List[str]] = None
) -> 'keras.Model':
    """
    Build a Bidirectional LSTM model for sequence classification.

    Args:
        input_shape: Shape of input (sequence_length, feature_dim)
        num_classes: Number of output classes
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        class_weights: Optional dict mapping class label strings to weights.
                      Example: {"Nomenclature": 100.0, "Description": 10.0, "Misc": 0.1}
                      Higher weights penalize errors on that class more heavily.
                      Use for addressing class imbalance with weighted cross-entropy.
        labels: List of label strings in order (e.g., ["Nomenclature", "Description", "Misc"])
                Required if class_weights or focal_labels is provided.
        focal_labels: Optional list of label strings to focus on for F1-based loss.
                     Example: ["Nomenclature", "Description"]
                     If provided, uses mean F1 loss for these labels only.
                     Cannot be used with class_weights (mutually exclusive).

    Returns:
        Compiled Keras model

    Raises:
        RuntimeError: If GPU initialization fails with helpful error message
        ValueError: If class_weights/focal_labels provided but labels is None,
                   or if both class_weights and focal_labels are provided
    """
    try:
        model = models.Sequential()

        # Add Input layer as the first layer (recommended by Keras)
        model.add(layers.Input(shape=input_shape))

        # First LSTM layer
        model.add(layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=True, dropout=dropout)
        ))

        # Additional LSTM layers
        for _ in range(num_layers - 1):
            model.add(layers.Bidirectional(
                layers.LSTM(hidden_size, return_sequences=True, dropout=dropout)
            ))

        # Time-distributed dense layer for per-timestep classification
        model.add(layers.TimeDistributed(layers.Dense(num_classes, activation='softmax')))

        # Validate mutually exclusive loss options
        if class_weights is not None and focal_labels is not None:
            raise ValueError("class_weights and focal_labels are mutually exclusive. Use one or the other.")

        # Create loss function based on parameters
        if focal_labels is not None:
            # Use F1-based loss for specified labels
            if labels is None:
                raise ValueError("labels parameter is required when focal_labels is provided")

            import tensorflow as tf

            # Map focal label strings to indices
            focal_indices = []
            for focal_label in focal_labels:
                if focal_label not in labels:
                    raise ValueError(f"focal_label '{focal_label}' not found in labels: {labels}")
                focal_indices.append(labels.index(focal_label))

            focal_indices_tensor = tf.constant(focal_indices, dtype=tf.int32)

            def mean_f1_loss(y_true, y_pred):
                """
                Mean F1 loss for specified focal labels.

                Computes soft F1 score for each focal label and returns 1 - mean(F1)
                so that minimizing the loss maximizes F1.

                Args:
                    y_true: One-hot encoded true labels, shape (batch, timesteps, num_classes)
                    y_pred: Predicted probabilities, shape (batch, timesteps, num_classes)

                Returns:
                    Loss scalar (1 - mean F1)
                """
                epsilon = tf.keras.backend.epsilon()

                f1_scores = []
                for focal_idx in focal_indices:
                    # Extract true and predicted for this class
                    # Shape: (batch, timesteps)
                    y_true_class = y_true[:, :, focal_idx]
                    y_pred_class = y_pred[:, :, focal_idx]

                    # Compute soft true positives, false positives, false negatives
                    # TP: sum of (true * pred) - when both are high
                    # FP: sum of ((1-true) * pred) - when pred is high but true is low
                    # FN: sum of (true * (1-pred)) - when true is high but pred is low

                    tp = tf.reduce_sum(y_true_class * y_pred_class)
                    fp = tf.reduce_sum((1.0 - y_true_class) * y_pred_class)
                    fn = tf.reduce_sum(y_true_class * (1.0 - y_pred_class))

                    # Compute soft precision and recall
                    precision = tp / (tp + fp + epsilon)
                    recall = tp / (tp + fn + epsilon)

                    # Compute F1 score
                    f1 = 2 * precision * recall / (precision + recall + epsilon)

                    f1_scores.append(f1)

                # Mean F1 across focal labels
                mean_f1 = tf.reduce_mean(tf.stack(f1_scores))

                # Return 1 - F1 so minimizing loss maximizes F1
                return 1.0 - mean_f1

            loss_fn = mean_f1_loss

            # Print focal labels for visibility
            print(f"[BiLSTM] Using mean F1 loss for focal labels:")
            for focal_label in focal_labels:
                print(f"  {focal_label}")

        elif class_weights is not None:
            if labels is None:
                raise ValueError("labels parameter is required when class_weights is provided")

            # Create weighted categorical cross-entropy loss
            import tensorflow as tf

            # Map label strings to indices and build weight tensor
            weight_list = []
            for i, label in enumerate(labels):
                weight = class_weights.get(label, 1.0)
                weight_list.append(weight)

            weight_tensor = tf.constant(weight_list, dtype=tf.float32)

            def weighted_categorical_crossentropy(y_true, y_pred):
                """
                Weighted categorical cross-entropy loss for class imbalance.

                Args:
                    y_true: One-hot encoded true labels, shape (batch, timesteps, num_classes)
                    y_pred: Predicted probabilities, shape (batch, timesteps, num_classes)

                Returns:
                    Weighted loss scalar
                """
                # Clip predictions to avoid log(0)
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

                # Calculate per-sample cross-entropy loss
                # Shape: (batch, timesteps)
                loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

                # Apply class weights based on true class
                # Get class indices from one-hot encoded labels
                class_indices = tf.argmax(y_true, axis=-1)  # Shape: (batch, timesteps)

                # Gather weights for each sample based on its true class
                weights = tf.gather(weight_tensor, class_indices)  # Shape: (batch, timesteps)

                # Apply weights
                weighted_loss = loss * weights

                return tf.reduce_mean(weighted_loss)

            loss_fn = weighted_categorical_crossentropy

            # Print class weights for visibility
            print(f"[BiLSTM] Using weighted loss with class weights:")
            for label, weight in sorted(class_weights.items()):
                print(f"  {label}: {weight:.1f}")
        else:
            loss_fn = 'categorical_crossentropy'
            print("[BiLSTM] Using standard categorical cross-entropy loss (no class weights)")

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss_fn,
            metrics=['accuracy']
        )

        return model

    except Exception as e:
        error_msg = str(e)
        if 'CUDA_ERROR' in error_msg or 'INVALID_PTX' in error_msg or 'INVALID_HANDLE' in error_msg:
            raise RuntimeError(
                f"Failed to build model due to GPU error: {error_msg}\n\n"
                "Your GPU may not be fully supported by this TensorFlow version.\n"
                "To fix this, restart your Python session and set this environment variable "
                "BEFORE importing skol_classifier:\n\n"
                "    import os\n"
                "    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU-only mode\n"
                "    # Now import skol_classifier\n\n"
                "Or run your script with: CUDA_VISIBLE_DEVICES='' python your_script.py\n"
            ) from e
        else:
            # Re-raise other errors as-is
            raise


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
        lineNumberCol: str = "line_number",
        labelCol: str = "label_indexed",
        valueCol: str = "value",
        window_size: int = 50
    ):
        """
        Initialize the sequence preprocessor.

        Args:
            inputCol: Column containing feature vectors
            outputCol: Column for output sequences
            docIdCol: Column containing document IDs
            lineNoCol: Column containing line numbers (for sorting)
            labelCol: Column containing labels
            window_size: Maximum sequence length
        """
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.docIdCol = docIdCol
        self.lineNoCol = lineNumberCol
        self.labelCol = labelCol
        self.valueCol = valueCol
        self.window_size = window_size

    def getInputCol(self):
        """Get input column name."""
        return self.inputCol

    def getOutputCol(self):
        """Get output column name."""
        return self.outputCol

    def getLineNoCol(self):
        """Get line number column name."""
        return self.lineNoCol

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Transform flat features into sequences.

        Groups by document ID and creates sequences with windowing for long documents.

        Args:
            df: Input DataFrame with columns: doc_id, line_number, features, label_indexed

        Returns:
            DataFrame with sequence_features and optional sequence_labels columns
        """
        has_labels = self.labelCol in df.columns
        input_col = self.getInputCol()

        # Simply group by document and collect all features and labels
        # We'll do windowing in the fit() method after collecting
        # The groupBy should produce one row per document.
        # The collect_list elements are ordered by line number using struct-based sorting.

        cols = [self.lineNoCol, self.docIdCol]
        for col in df.columns:
            if col not in cols:
                cols.append(col)

        fcols = [F.col(c) for c in cols]
        struct_col = F.struct(*fcols).alias("zipped_arrays")

        # Collect structs containing line number etc, then sort and extract
        grouped = df.groupBy(self.docIdCol).agg(
            F.sort_array(
                collect_list(struct_col)
            ).alias("sorted_data"),
            F.first(self.valueCol).alias(self.valueCol)
        )

        return grouped


class RNNSkolModel(SkolModel):
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
        prediction_stride: Optional[int] = None,
        prediction_batch_size: int = 64,
        batch_size: int = 32,
        epochs: int = 10,
        num_workers: int = 4,
        features_col: str = "combined_idf",
        label_col: str = "label_indexed",
        line_no_col: str = "line_number",
        verbosity: int = 3,
        name: str = "RNN_BiLSTM",
        class_weights: Optional[Dict[str, float]] = None,
        focal_labels: Optional[List[str]] = None
    ):
        """
        Initialize RNN classifier.

        Args:
            input_size: Size of input feature vectors
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            window_size: Maximum sequence length for training and prediction windows
            prediction_stride: Stride for sliding window during prediction.
                             If None, defaults to window_size (non-overlapping windows).
                             Use smaller values (e.g., window_size // 2) for overlapping windows
                             to classify longer documents.
            prediction_batch_size: Maximum number of windows to predict in a single batch.
                                  Controls memory usage during prediction. Larger values are faster
                                  but use more memory. Default: 64
            batch_size: Batch size for training
            epochs: Number of training epochs
            num_workers: Number of Spark workers (unused in Pandas UDF approach)
            features_col: Name of features column
            label_col: Name of label column
            line_no_col: Name of line number column
            verbosity: Verbosity level for logging
            name: Name of the model
            class_weights: Optional dict mapping class label strings to weights.
                          Example: {"Nomenclature": 100.0, "Description": 10.0, "Misc": 0.1}
                          Higher weights penalize errors on that class more heavily.
                          Use for addressing class imbalance with weighted cross-entropy.
                          Cannot be used with focal_labels (mutually exclusive).
                          Note: Labels must be provided in fit() for this to work.
            focal_labels: Optional list of label strings to focus on for F1-based loss.
                         Example: ["Nomenclature", "Description"]
                         Uses mean F1 loss for these labels only, ignoring others.
                         Cannot be used with class_weights (mutually exclusive).
                         Note: Labels must be provided in fit() for this to work.
        """
        if not KERAS_AVAILABLE:
            raise ImportError(
                "TensorFlow and Keras are required for RNN model. "
                "Install with: pip install tensorflow"
            )

        # Initialize parent class
        super().__init__(
            features_col=features_col,
            label_col=label_col
        )

        # Store RNN-specific parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.window_size = window_size
        self.prediction_stride = prediction_stride if prediction_stride is not None else window_size
        self.prediction_batch_size = prediction_batch_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.verbosity = verbosity
        self.name = name
        self.class_weights = class_weights
        self.focal_labels = focal_labels

        # Validate mutually exclusive options
        if class_weights is not None and focal_labels is not None:
            raise ValueError("class_weights and focal_labels are mutually exclusive. Use one or the other.")

        # Build Keras model (without class weights/focal labels initially, will rebuild in fit())
        self.keras_model = build_bilstm_model(
            input_shape=(window_size, input_size),
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            class_weights=None,  # Will rebuild with weights/focal labels in fit() when labels are available
            labels=None,
            focal_labels=None
        )

        self.model_weights = None

    def _process_row_to_windows(self, row):
        """Process a single row into training windows."""
        # Access the sorted_data array of structs
        sorted_data = row.sorted_data

        # Extract features and labels from the struct array
        dense_features = []
        dense_labels = []

        for item in sorted_data:
            # Extract the feature from the struct
            feat = item[self.features_col]
            if feat is None:
                continue

            # Convert to dense array
            if hasattr(feat, 'toArray'):
                dense_features.append(feat.toArray().tolist())
            elif isinstance(feat, (list, tuple)):
                dense_features.append(list(feat))
            else:
                try:
                    dense_features.append(list(feat))
                except TypeError:
                    continue

            # Extract the label from the struct
            label = item[self.label_col]
            if label is not None:
                dense_labels.append(float(label))

        # Create windows from this document
        windows = []
        for i in range(0, len(dense_features), self.window_size):
            window_features = dense_features[i:i + self.window_size]
            window_labels = dense_labels[i:i + self.window_size]

            if len(window_features) == 0:
                continue

            # Pad or truncate to window_size
            if len(window_features) < self.window_size:
                padding = [[0.0] * self.input_size] * (self.window_size - len(window_features))
                window_features = window_features + padding
                window_labels = window_labels + [0.0] * (self.window_size - len(window_labels))
            else:
                window_features = window_features[:self.window_size]
                window_labels = window_labels[:self.window_size]

            windows.append((window_features, window_labels))

        return windows

    def _data_generator(self, sequenced_data, batch_size, chunk_size=50):
        """
        Generator that yields batches of training data without loading everything into memory.

        Processes documents in small chunks to minimize memory footprint.

        Args:
            sequenced_data: DataFrame with sequence_features and optionally sequence_labels
            batch_size: Number of sequences per batch
            chunk_size: Number of documents to process at once (default: 50)

        Yields:
            Tuple of (X_batch, y_batch) as numpy arrays
        """
        import random
        import gc

        # Get total count for planning
        total_docs = sequenced_data.count()
        if self.verbosity >= 5:
            print(f"[RNN Generator] Processing {total_docs} documents in chunks of {chunk_size}...")

        # Create list of document indices
        doc_indices = list(range(total_docs))

        # Batch accumulator
        X_batch = []
        y_batch = []

        epoch_num = 0
        batch_num = 0

        while True:  # Keras generators need to loop infinitely (for multiple epochs)
            epoch_num += 1
            if self.verbosity >= 5:
                print(f"[RNN Generator] Starting epoch {epoch_num}")

            # Shuffle document order for each epoch
            random.shuffle(doc_indices)

            # Process documents in chunks
            chunk_num = 0
            for chunk_start in range(0, total_docs, chunk_size):
                chunk_num += 1
                chunk_end = min(chunk_start + chunk_size, total_docs)

                if self.verbosity >= 3:
                    print(f"[RNN Generator] Epoch {epoch_num}, Chunk {chunk_num}/{(total_docs + chunk_size - 1) // chunk_size}: docs {chunk_start}-{chunk_end}")

                # Get this chunk of documents using skip/limit
                # Note: This is not perfectly efficient but avoids loading all data
                if self.verbosity >= 5:
                    print(f"[RNN Generator] Collecting chunk documents...")
                chunk_docs = (sequenced_data
                             .limit(chunk_end)
                             .collect()[chunk_start:chunk_end])
                if self.verbosity >= 5:
                    print(f"[RNN Generator] Got {len(chunk_docs)} documents in chunk")

                # Process each document in the chunk
                doc_num = 0
                for row in chunk_docs:
                    doc_num += 1
                    windows = self._process_row_to_windows(row)

                    if doc_num % 10 == 0 and self.verbosity >= 5:
                        print(f"[RNN Generator] Processing doc {doc_num}/{len(chunk_docs)} in chunk, {len(windows)} windows, batch accumulator size: {len(X_batch)}")

                    for window_features, window_labels in windows:
                        X_batch.append(window_features)
                        y_batch.append(window_labels)

                        # Yield when we have a full batch
                        if len(X_batch) >= batch_size:
                            batch_num += 1
                            if self.verbosity >= 5:
                                print(f"[RNN Generator] Creating batch {batch_num} (size {batch_size})")
                            X = np.array(X_batch[:batch_size], dtype=np.float32)
                            y = np.array(y_batch[:batch_size], dtype=np.int32)
                            if self.verbosity >= 5:
                                print(f"[RNN Generator] Converting to categorical...")
                            y_cat = to_categorical(y, num_classes=self.num_classes)

                            if self.verbosity >= 5:
                                print(f"[RNN Generator] Yielding batch {batch_num}")
                            yield X, y_cat

                            # Keep overflow for next batch
                            X_batch = X_batch[batch_size:]
                            y_batch = y_batch[batch_size:]

                            # Force garbage collection
                            del X, y, y_cat
                            gc.collect()

                # Clear chunk from memory
                if self.verbosity >= 5:
                    print(f"[RNN Generator] Clearing chunk {chunk_num} from memory")
                del chunk_docs
                gc.collect()

            # Yield remaining data at end of epoch if we have any
            if X_batch:
                batch_num += 1
                if self.verbosity >= 5:
                    print(f"[RNN Generator] End of epoch {epoch_num}, yielding final batch {batch_num} with {len(X_batch)} samples")
                X = np.array(X_batch, dtype=np.float32)
                y = np.array(y_batch, dtype=np.int32)
                y_cat = to_categorical(y, num_classes=self.num_classes)
                yield X, y_cat
                X_batch = []
                y_batch = []
                del X, y, y_cat
                gc.collect()

            if self.verbosity >= 5:
                print(f"[RNN Generator] Completed epoch {epoch_num}")

    def fit(self, train_data: DataFrame, labels: Optional[List[str]] = None) -> 'RNNSkolModel':
        """
        Train the RNN classification model using Keras generators.

        Uses a generator-based approach to avoid loading all data into memory,
        which prevents OOM errors with large datasets.

        Args:
            train_data: Training DataFrame with features and labels
            labels: Optional list of label strings

        Returns:
            Self (fitted model)
        """
        if self.verbosity >= 3:
            print("[RNN Fit] Starting RNN model training")

        if labels is not None:
            self.labels = labels
            if self.verbosity >= 3:
                print(f"[RNN Fit] Set labels: {labels}")

        if self.verbosity >= 3:
            print("[RNN Fit] Preparing sequences for RNN training...")

        # Determine document ID column (CouchDB uses 'doc_id', files use 'filename')
        doc_id_col = "doc_id" if "doc_id" in train_data.columns else "filename"
        line_no_col = "line_number" if "line_number" in train_data.columns else "dummy_line_number"
        if self.verbosity >= 3:
            print(f"[RNN Fit] Using document ID column: {doc_id_col}")

        # Create sequence preprocessor
        if self.verbosity >= 3:
            print("[RNN Fit] Creating sequence preprocessor...")
        preprocessor = SequencePreprocessor(
            inputCol=self.features_col,
            outputCol="sequence_features",
            docIdCol=doc_id_col,
            lineNumberCol=line_no_col,
            labelCol=self.label_col,
            window_size=self.window_size
        )

        # Transform data into sequences
        if self.verbosity >= 3:
            print("[RNN Fit] Transforming data into sequences...")
            print(f"[RNN Fit]   Input data columns: {train_data.columns}")
        sequenced_data = preprocessor.transform(train_data)

        # Cache to avoid recomputation
        if self.verbosity >= 3:
            print("[RNN Fit] Caching sequenced data...")
        sequenced_data = sequenced_data.cache()

        if self.verbosity >= 3:
            print(f"[RNN Fit] Sequenced data count: {sequenced_data.count()}")
            sequenced_data.show(5)  # DEBUG

        # Detect actual feature dimension from data
        if self.verbosity >= 3:
            print("[RNN Fit] Detecting actual feature dimension...")
        # Access the sorted_data array of structs
        sample_row = sequenced_data.select("sorted_data").first()
        if sample_row and sample_row.sorted_data:
            # Get the first struct from the array
            first_struct = sample_row.sorted_data[0]
            # Extract the feature from the struct
            first_feature = first_struct[self.features_col]
            if hasattr(first_feature, 'size'):
                actual_input_size = first_feature.size
            elif hasattr(first_feature, 'toArray'):
                actual_input_size = len(first_feature.toArray())
            elif hasattr(first_feature, '__len__'):
                actual_input_size = len(first_feature)
            else:
                actual_input_size = self.input_size

            if actual_input_size != self.input_size:
                if self.verbosity >= 1:
                    print(f"[RNN Fit] WARNING: Model input_size ({self.input_size}) != actual feature size ({actual_input_size})")
                    print(f"[RNN Fit] Rebuilding model with correct input size...")

                # Update input_size and rebuild model
                self.input_size = actual_input_size
                self.keras_model = build_bilstm_model(
                    input_shape=(self.window_size, self.input_size),
                    num_classes=self.num_classes,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    class_weights=self.class_weights,
                    labels=self.labels if hasattr(self, 'labels') else None,
                    focal_labels=self.focal_labels
                )
                if self.verbosity >= 2:
                    print(f"[RNN Fit] Model rebuilt with input_size={self.input_size}")
            else:
                if self.verbosity >= 3:
                    print(f"[RNN Fit] Feature dimension matches: {actual_input_size}")

                # Even if dimensions match, rebuild if we have class weights/focal labels and haven't applied them yet
                needs_rebuild = (
                    (self.class_weights is not None or self.focal_labels is not None) and
                    hasattr(self, 'labels') and self.labels is not None
                )
                if needs_rebuild:
                    if self.verbosity >= 2:
                        loss_type = "focal F1 loss" if self.focal_labels is not None else "class weights"
                        print(f"[RNN Fit] Rebuilding model to apply {loss_type}...")
                    self.keras_model = build_bilstm_model(
                        input_shape=(self.window_size, self.input_size),
                        num_classes=self.num_classes,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        class_weights=self.class_weights,
                        labels=self.labels,
                        focal_labels=self.focal_labels
                    )

        # Estimate number of sequences for steps_per_epoch
        # Sample a few documents to estimate average windows per document
        if self.verbosity >= 3:
            print("[RNN Fit] Sampling documents to estimate training size...")
        sample = sequenced_data.limit(3).collect()  # Reduced from 10 to 3
        if self.verbosity >= 3:
            print(f"[RNN Fit] Got {len(sample)} sample documents")

        if self.verbosity >= 5:
            print(f"[RNN Fit] Sample documents:\n{sample}")

        if len(sample) == 0:
            raise ValueError("No sequences generated from training data")

        if self.verbosity >= 3:
            print("[RNN Fit] Counting total documents...")
        total_docs = sequenced_data.count()
        if self.verbosity >= 3:
            print(f"[RNN Fit] Total documents: {total_docs}")

        if self.verbosity >= 3:
            print("[RNN Fit] Estimating windows per document (memory-efficient)...")
        # Don't process full windows, just count lines to estimate
        total_windows = 0
        for idx, row in enumerate(sample):
            # Just count the number of rows (lines) in sorted_data
            num_rows = len(row.sorted_data) if row.sorted_data else 0
            # Estimate windows as ceil(num_rows / window_size)
            doc_windows = max(1, (num_rows + self.window_size - 1) // self.window_size)
            total_windows += doc_windows
            if self.verbosity >= 4:
                print(f"[RNN Fit]   Sample {idx+1}: {num_rows} lines -> ~{doc_windows} windows")
            # Clear this sample from memory
            del row
            gc.collect()

        avg_windows = total_windows / len(sample)
        estimated_sequences = int(total_docs * avg_windows)
        steps_per_epoch = max(1, estimated_sequences // self.batch_size)

        # Clean up sample data
        del sample
        gc.collect()

        if self.verbosity >= 1:
            print(f"[RNN Fit] Training RNN model with generator-based approach...")
            print(f"[RNN Fit]   Estimated sequences: ~{estimated_sequences}")
            print(f"[RNN Fit]   Batch size: {self.batch_size}")
            print(f"[RNN Fit]   Steps per epoch: {steps_per_epoch}")
            print(f"[RNN Fit]   Epochs: {self.epochs}")
            print(f"[RNN Fit]   Window size: {self.window_size}")
            print(f"[RNN Fit]   Stride size: {self.prediction_stride}")
            print(f"[RNN Fit]   Input size: {self.input_size}")

        # Create generator
        if self.verbosity >= 1:
            print("[RNN Fit] Creating data generator...")
        train_generator = self._data_generator(sequenced_data, self.batch_size, chunk_size=25)

        # Train model with generator
        if self.verbosity >= 1:
            print("[RNN Fit] Starting Keras model.fit()...")
        try:
            self.keras_model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                verbose=1
            )
            if self.verbosity >= 1:
                print("[RNN Fit] Keras model.fit() completed successfully")
        except Exception as e:
            if self.verbosity >= 1:
                print(f"[RNN Fit] ERROR during model.fit(): {e}")
            raise

        # Store model weights for distribution
        if self.verbosity >= 3:
            print("[RNN Fit] Storing model weights...")
        self.model_weights = self.keras_model.get_weights()
        self.classifier_model = self.keras_model

        # Unpersist cached data
        if self.verbosity >= 3:
            print("[RNN Fit] Unpersisting cached data...")
        sequenced_data.unpersist()
        if self.verbosity >= 1:
            print("[RNN Fit] Training completed successfully")
        return self

    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions on data.

        Reimplemented to use predict_proba() and add argmax to get predictions.

        Args:
            data: DataFrame to predict on

        Returns:
            DataFrame with both 'prediction' and 'probabilities' columns
        """
        if self.verbosity >= 2:
            print("[RNN Predict] Starting prediction [CODE VERSION 2025-12-16]")
            print(f"[RNN Predict] Input data columns: {data.columns}")
            print(f"[RNN Predict] Input data count: {data.count()}")
            print(f"[RNN Predict] Verbosity level: {self.verbosity}")

        if self.classifier_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Use predict_proba to get probability distributions
        if self.verbosity >= 2:
            print("[RNN Predict] Calling predict_proba to get probabilities")

        proba_result = self.predict_proba(data)

        # Add prediction column by taking argmax of probabilities
        if self.verbosity >= 2:
            print("[RNN Predict] Adding prediction column via argmax")

        # Define UDF to compute argmax of probability array
        from pyspark.sql.functions import udf
        from pyspark.sql.types import IntegerType
        from typing import Optional

        @udf(returnType=IntegerType())
        def argmax_udf(proba_array: Optional[List[float]]) -> int:
            """Return the index of the maximum probability."""
            if proba_array is None or len(proba_array) == 0:
                return 0
            return int(max(range(len(proba_array)), key=lambda i: float(proba_array[i])))

        result = proba_result.withColumn("prediction", argmax_udf(col("probabilities")).cast("double"))

        if self.verbosity >= 2:
            print(f"[RNN Predict] Result columns: {result.columns}")
        if self.verbosity >= 3:
            print("[RNN Predict]: Result DataFrame schema:")
            result.printSchema()
            print("[RNN Predict]: Result DataFrame sample:")
            result.show(5, truncate=False)

        if self.verbosity >= 1:
            print("[RNN Predict] Prediction completed successfully")
        return result

    def predict_proba(self, data: DataFrame) -> DataFrame:
        """
        Make predictions on data, returning probability distributions.

        Args:
            data: DataFrame to predict on

        Returns:
            DataFrame with probability distributions for each class.
            The 'probabilities' column contains an array of probabilities,
            one per class (e.g., [0.8, 0.15, 0.05] for 3 classes).
        """
        if self.verbosity >= 2:
            print("[RNN Predict Proba] Starting probability prediction [CODE VERSION 2025-12-16]")
            print(f"[RNN Predict Proba] Input data columns: {data.columns}")
            print(f"[RNN Predict Proba] Input data count: {data.count()}")
            print(f"[RNN Predict Proba] Verbosity level: {self.verbosity}")

        if self.classifier_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Import tqdm for progress bar
        try:
            from tqdm.auto import tqdm
            tqdm_available = True
        except ImportError:
            tqdm_available = False
            if self.verbosity >= 2:
                print("[RNN Predict Proba] tqdm not available, progress bar disabled")

        # Determine document ID column
        doc_id_col = "doc_id" if "doc_id" in data.columns else "filename"
        line_no_col = "line_number" if "line_number" in data.columns else "dummy_line_number"
        if self.verbosity >= 3:
            print(f"[RNN Predict Proba] Using document ID column: {doc_id_col}")

        # Create sequence preprocessor
        if self.verbosity >= 3:
            print("[RNN Predict Proba] Creating sequence preprocessor")
        preprocessor = SequencePreprocessor(
            inputCol=self.features_col,
            outputCol="sequence_features",
            docIdCol=doc_id_col,
            lineNumberCol=line_no_col,
            labelCol=self.label_col if self.label_col in data.columns else "dummy_label",
            window_size=self.window_size
        )

        has_labels = self.label_col in data.columns
        if self.verbosity >= 3:
            print(f"[RNN Predict Proba] Has labels: {has_labels}")

        # Transform into sequences
        if self.verbosity >= 2:
            print("[RNN Predict Proba] Transforming data into sequences")
        sequenced_data = preprocessor.transform(data)
        if self.verbosity >= 3:
            print(f"[RNN Predict Proba] Sequenced data columns: {sequenced_data.columns}")

        # Get total sequence count for progress bar
        total_sequences = sequenced_data.count() if tqdm_available else 0
        if tqdm_available and self.verbosity >= 1:
            print(f"[RNN Predict Proba] Total sequences to process: {total_sequences}")

        # Create accumulator for progress tracking
        if tqdm_available:
            from pyspark import SparkContext
            sc = SparkContext.getOrCreate()
            sequences_processed_acc = sc.accumulator(0)
        else:
            sequences_processed_acc = None

        # Broadcast model weights for UDF
        if self.verbosity >= 2:
            print(f"[RNN Predict Proba] Preparing model config and weights for UDF")
            print(f"[RNN Predict Proba] model_weights is None: {self.model_weights is None}")
            if self.model_weights:
                print(f"[RNN Predict Proba] model_weights count: {len(self.model_weights)}")
            print(f"[RNN Predict Proba] Using sliding window: window_size={self.window_size}, stride={self.prediction_stride}")
        model_config = self.keras_model.to_json()
        model_weights = self.model_weights
        input_size = self.input_size
        window_size = self.window_size
        num_classes = self.num_classes
        prediction_stride = self.prediction_stride
        prediction_batch_size = self.prediction_batch_size
        features_col = self.features_col  # Capture for UDF closure
        verbosity = self.verbosity  # Capture for UDF closure
        show_progress = tqdm_available and self.verbosity >= 1

        # Define prediction UDF that returns probability distributions
        @pandas_udf(ArrayType(ArrayType(FloatType())))
        def predict_proba_sequence(sorted_data_series: pd.Series) -> pd.Series:
            """Predict probability distributions on sequences using the trained model.

            Args:
                sorted_data_series: Series of arrays of Row objects (structs),
                                   each containing features and other fields

            Returns:
                Series where each element is a list of probability arrays,
                one array per line in the sequence.
            """
            import os
            import numpy as np
            import tempfile
            import time

            # Create a debug log file in /tmp for this executor
            log_file = f"/tmp/rnn_proba_udf_debug_{os.getpid()}_{int(time.time())}.log"

            def log(msg):
                """Write to both stderr and file."""
                import sys
                try:
                    with open(log_file, 'a') as f:
                        f.write(f"{msg}\n")
                        f.flush()
                except Exception:
                    pass
                if verbosity >= 3:
                    print(msg, file=sys.stderr)

            log(f"[UDF PROBA START] Processing {len(sorted_data_series)} sequences")
            log(f"[UDF PROBA START] Config: input_size={input_size}, window_size={window_size}, num_classes={num_classes}")
            log(f"[UDF PROBA START] Log file: {log_file}")

            try:
                # Force CPU-only mode in executors to prevent CUDA errors
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                log("[UDF PROBA] Set CUDA_VISIBLE_DEVICES to empty string")

                # Import TensorFlow/Keras inside UDF after setting CPU mode
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                    # Double-check GPU is disabled
                    tf.config.set_visible_devices([], 'GPU')
                    log("[UDF PROBA] TensorFlow imported and GPU disabled")
                except Exception as e:
                    log(f"[UDF PROBA WARNING] TensorFlow config issue: {e}")
                    pass

                # Rebuild model from config and weights
                # Provide dummy loss functions for deserialization (won't be used for prediction)
                try:
                    # Define dummy loss functions for deserialization
                    def weighted_categorical_crossentropy(y_true, y_pred):
                        """Dummy loss function for model deserialization. Not used for prediction."""
                        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

                    def mean_f1_loss(y_true, y_pred):
                        """Dummy loss function for model deserialization. Not used for prediction."""
                        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

                    # Rebuild with custom objects to handle custom loss functions
                    custom_objects = {
                        'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
                        'mean_f1_loss': mean_f1_loss
                    }
                    model = keras.models.model_from_json(model_config, custom_objects=custom_objects)
                    model.set_weights(model_weights)
                    log(f"[UDF PROBA] Model rebuilt successfully")
                except Exception as e:
                    log(f"[UDF PROBA FATAL] Model rebuild failed: {e}")
                    raise RuntimeError(
                        f"Failed to rebuild model in executor: {e}\n"
                        "This may be due to GPU compatibility issues. "
                        "Ensure CUDA_VISIBLE_DEVICES='' is set before starting Spark."
                    )

                results = []
                log(f"[UDF PROBA] Initialization complete, starting prediction loop")
            except Exception as e:
                # If initialization fails, return empty lists for all sequences
                log(f"[UDF PROBA ERROR] Initialization failed: {e}")
                import sys
                import traceback
                traceback.print_exc(file=sys.stderr)
                log(f"[UDF PROBA ERROR] Returning empty arrays for all {len(sorted_data_series)} sequences")
                return pd.Series([[]] * len(sorted_data_series))

            # PHASE 1: Extract features and prepare all windows for batched prediction
            sequence_metadata = []
            all_windows = []

            log(f"[UDF PROBA PHASE 1] Extracting features from {len(sorted_data_series)} sequences")

            for seq_idx, sorted_data in enumerate(sorted_data_series):
                try:
                    # Extract features from the struct array
                    dense_features = []

                    for row in sorted_data:
                        # Extract the feature from the struct
                        feat = row[features_col]
                        if feat is None:
                            continue

                        # Convert to dense array
                        try:
                            if isinstance(feat, dict):
                                from pyspark.ml.linalg import Vectors
                                if feat.get('type') == 0:  # DenseVector
                                    vec = Vectors.dense(feat['values'])
                                elif feat.get('type') == 1:  # SparseVector
                                    vec = Vectors.sparse(feat['size'], feat['indices'], feat['values'])
                                else:
                                    raise ValueError(f"Unknown vector type: {feat.get('type')}")
                                dense_arr = np.array(vec.toArray(), dtype=np.float32)
                            elif hasattr(feat, 'toArray'):
                                dense_arr = np.array(feat.toArray(), dtype=np.float32)
                            elif isinstance(feat, np.ndarray):
                                dense_arr = np.asarray(feat, dtype=np.float32)
                            elif isinstance(feat, (list, tuple)):
                                dense_arr = np.array(feat, dtype=np.float32)
                            else:
                                dense_arr = np.array(list(feat), dtype=np.float32)
                        except Exception:
                            dense_arr = np.zeros(input_size, dtype=np.float32)

                        if len(dense_arr) == 0:
                            continue

                        # Ensure each feature vector is exactly input_size
                        if len(dense_arr) < input_size:
                            padding = np.zeros(input_size - len(dense_arr), dtype=np.float32)
                            dense_arr = np.concatenate([dense_arr, padding])
                        elif len(dense_arr) > input_size:
                            dense_arr = dense_arr[:input_size]

                        dense_features.append(dense_arr)

                    # Handle empty sequences
                    if len(dense_features) == 0:
                        dense_features = [np.zeros(input_size, dtype=np.float32)]

                    sequence_length = len(dense_features)

                    # Prepare windows for this sequence
                    if sequence_length <= window_size:
                        # Short sequence: pad to window_size and create one window
                        if sequence_length < window_size:
                            padding = [np.zeros(input_size, dtype=np.float32)] * (window_size - sequence_length)
                            padded_features = dense_features + padding
                        else:
                            padded_features = dense_features

                        window_array = np.array(padded_features, dtype=np.float32)
                        all_windows.append(window_array)

                        sequence_metadata.append({
                            'seq_idx': seq_idx,
                            'type': 'short',
                            'sequence_length': sequence_length,
                            'window_indices': [len(all_windows) - 1]
                        })
                    else:
                        # Long sequence: create sliding windows
                        window_indices = []
                        window_positions = []

                        for window_start in range(0, sequence_length, prediction_stride):
                            window_end = min(window_start + window_size, sequence_length)
                            window_features = dense_features[window_start:window_end]

                            if len(window_features) < window_size:
                                padding = [np.zeros(input_size, dtype=np.float32)] * (window_size - len(window_features))
                                window_features = window_features + padding

                            window_array = np.array(window_features, dtype=np.float32)
                            all_windows.append(window_array)
                            window_indices.append(len(all_windows) - 1)

                            actual_window_length = min(window_size, window_end - window_start)
                            window_positions.append((window_start, actual_window_length))

                        sequence_metadata.append({
                            'seq_idx': seq_idx,
                            'type': 'sliding',
                            'sequence_length': sequence_length,
                            'window_indices': window_indices,
                            'window_positions': window_positions
                        })

                except Exception as e:
                    if seq_idx < 10 or seq_idx % 1000 == 0:
                        log(f"[UDF PROBA ERROR] Sequence {seq_idx} failed during feature extraction: {e}")
                    sequence_metadata.append({
                        'seq_idx': seq_idx,
                        'type': 'failed',
                        'error': str(e)
                    })

            # PHASE 2: Batch predict all windows
            log(f"[UDF PROBA PHASE 2] Predicting on {len(all_windows)} windows from {len(sorted_data_series)} sequences")

            if len(all_windows) == 0:
                log(f"[UDF PROBA ERROR] No windows to predict - returning empty results")
                return pd.Series([[]] * len(sorted_data_series))

            # Predict in batches
            total_windows = len(all_windows)
            num_batches = (total_windows + prediction_batch_size - 1) // prediction_batch_size
            log(f"[UDF PROBA PHASE 2] Using {num_batches} batches of max size {prediction_batch_size}")

            all_predictions = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * prediction_batch_size
                end_idx = min(start_idx + prediction_batch_size, total_windows)

                batch_windows = all_windows[start_idx:end_idx]
                X_batch = np.array(batch_windows, dtype=np.float32)

                if batch_idx == 0 or batch_idx == num_batches - 1:
                    log(f"[UDF PROBA PHASE 2] Batch {batch_idx+1}/{num_batches}: shape {X_batch.shape}")

                # Predict this batch - returns probability distributions
                batch_preds = model.predict(X_batch, verbose=0)
                all_predictions.append(batch_preds)

            # Concatenate all batch predictions
            all_predictions = np.concatenate(all_predictions, axis=0)
            log(f"[UDF PROBA PHASE 2] Total predictions shape: {all_predictions.shape}")

            # PHASE 3: Reconstruct results for each sequence
            log(f"[UDF PROBA PHASE 3] Reconstructing probability results for {len(sequence_metadata)} sequences")

            results = [None] * len(sorted_data_series)

            for metadata in sequence_metadata:
                seq_idx = metadata['seq_idx']

                if metadata['type'] == 'failed':
                    results[seq_idx] = []
                    continue

                if metadata['type'] == 'short':
                    # Single window - extract probabilities for actual sequence length
                    window_idx = metadata['window_indices'][0]
                    sequence_length = metadata['sequence_length']
                    preds = all_predictions[window_idx][:sequence_length]  # Shape: (sequence_length, num_classes)
                    # Convert to list of lists (each element is a probability distribution)
                    proba_arrays = [p.tolist() for p in preds]
                    results[seq_idx] = proba_arrays

                elif metadata['type'] == 'sliding':
                    # Multiple sliding windows - average overlapping predictions
                    sequence_length = metadata['sequence_length']
                    window_indices = metadata['window_indices']
                    window_positions = metadata['window_positions']

                    # Initialize prediction accumulator
                    prediction_counts = np.zeros(sequence_length, dtype=np.int32)
                    prediction_sums = np.zeros((sequence_length, num_classes), dtype=np.float32)

                    # Accumulate predictions from all windows
                    for window_idx, (window_start, actual_window_length) in zip(window_indices, window_positions):
                        window_preds = all_predictions[window_idx]  # Shape: (window_size, num_classes)

                        for i in range(actual_window_length):
                            global_pos = window_start + i
                            if global_pos < sequence_length:
                                prediction_sums[global_pos] += window_preds[i]
                                prediction_counts[global_pos] += 1

                    # Average overlapping predictions and return probability distributions
                    proba_arrays = []
                    for i in range(sequence_length):
                        if prediction_counts[i] > 0:
                            avg_probs = prediction_sums[i] / prediction_counts[i]
                            proba_arrays.append(avg_probs.tolist())
                        else:
                            # Default to uniform distribution if no predictions
                            uniform_probs = [1.0 / num_classes] * num_classes
                            proba_arrays.append(uniform_probs)

                    results[seq_idx] = proba_arrays

            log(f"[UDF PROBA COMPLETE] Processed {len(sorted_data_series)} sequences, {len(all_windows)} total windows")
            log(f"[UDF PROBA COMPLETE] Non-empty results: {sum(1 for r in results if r and len(r) > 0)}")
            log(f"[UDF PROBA COMPLETE] Empty results: {sum(1 for r in results if not r or len(r) == 0)}")

            # Update progress accumulator
            if sequences_processed_acc is not None:
                sequences_processed_acc.add(len(sorted_data_series))

            return pd.Series(results)

        # Apply prediction
        if self.verbosity >= 2:
            print("[RNN Predict Proba] Applying probability prediction UDF to sequences")

        predictions = sequenced_data.withColumn(
            "probabilities",
            predict_proba_sequence(col("sorted_data"))
        )

        # Check for empty or null prediction arrays
        if show_progress:
            import threading
            import time

            pbar = tqdm(total=total_sequences, desc="Predicting probabilities", unit="seq")

            def update_progress():
                """Background thread to update progress bar."""
                last_value = 0
                while not stop_progress.is_set():
                    current = sequences_processed_acc.value
                    if current > last_value:
                        pbar.update(current - last_value)
                        last_value = current
                    time.sleep(0.5)

            stop_progress = threading.Event()
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()

            try:
                total_seqs = predictions.count()
                final_value = sequences_processed_acc.value
                pbar.update(final_value - pbar.n)
            finally:
                stop_progress.set()
                progress_thread.join(timeout=1.0)
                pbar.close()
        else:
            total_seqs = predictions.count()

        empty_preds = predictions.filter(F.size(col("probabilities")) == 0)
        empty_count = empty_preds.count()
        null_preds = predictions.filter(col("probabilities").isNull())
        null_count = null_preds.count()

        if empty_count > 0 or null_count > 0:
            print(f"\n{'='*70}")
            print(f"[RNN Predict Proba] PREDICTION FAILURE DETECTED!")
            print(f"  Total sequences: {total_seqs}")
            print(f"  Empty probability arrays: {empty_count}")
            print(f"  Null probability arrays: {null_count}")
            print(f"  Success rate: {((total_seqs - empty_count - null_count) / total_seqs * 100):.1f}%")
            print(f"{'='*70}")

        if self.verbosity >= 3:
            print("[RNN Predict Proba]: Predictions DataFrame schema:")
            predictions.printSchema()
            print("[RNN Predict Proba]: Predictions DataFrame sample:")
            predictions.show(5, truncate=False)

        # Explode probabilities back to line-level
        if self.verbosity >= 2:
            print("[RNN Predict Proba] Exploding probabilities for line-level results")

        # Combine probabilities array with sorted_data array using arrays_zip
        predictions_with_data = predictions.withColumn(
            "zipped_arrays",
            F.arrays_zip("sorted_data", "probabilities")
        )

        # Posexplode the zipped arrays
        predictions_exploded = predictions_with_data.select(
            predictions_with_data[doc_id_col],
            predictions_with_data["value"],
            F.posexplode(predictions_with_data["zipped_arrays"]).alias("pos", "col")
        )

        # Extract fields from the nested struct
        from pyspark.sql.types import StructType
        col_struct_type = predictions_exploded.schema["col"].dataType

        sorted_data_struct_type = None
        if isinstance(col_struct_type, StructType):
            for field in col_struct_type.fields:
                if field.name == "sorted_data":
                    sorted_data_struct_type = field.dataType
                    break

        if sorted_data_struct_type and isinstance(sorted_data_struct_type, StructType):
            skip_fields = {doc_id_col, "value"}

            result = predictions_exploded.select(
                predictions_exploded[doc_id_col],
                predictions_exploded["pos"],
                predictions_exploded["value"].alias("value"),
                predictions_exploded["col"]["probabilities"].alias("probabilities"),
                *[predictions_exploded["col"]["sorted_data"][field.name].alias(field.name)
                  for field in sorted_data_struct_type.fields
                  if field.name not in skip_fields]
            ).cache()
        else:
            result = predictions_exploded.select(
                predictions_exploded[doc_id_col],
                predictions_exploded["pos"],
                predictions_exploded["value"].alias("value"),
                predictions_exploded["col"]["probabilities"].alias("probabilities")
            ).cache()

        if self.verbosity >= 2:
            print(f"[RNN Predict Proba] Result columns: {result.columns}")
        if self.verbosity >= 3:
            print("[RNN Predict Proba]: Result DataFrame schema:")
            result.printSchema()
            print("[RNN Predict Proba]: Result DataFrame sample:")
            result.show(5, truncate=False)

        if self.verbosity >= 1:
            print("[RNN Predict Proba] Probability prediction completed successfully")
        return result

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.classifier_model is None:
            raise ValueError("No model to save")

        self.classifier_model.save(path)

    def set_model(self, model: Any) -> None:
        """Set the model (useful for loading from Redis)."""
        self.keras_model = model
        self.classifier_model = model
        self.model_weights = model.get_weights()
        if self.verbosity >= 2:
            print(f"[RNN set_model] Model set, weights count: {len(self.model_weights) if self.model_weights else 0}")

    def load(self, path: str) -> 'RNNSkolModel':
        """Load model from disk.

        Note: Model is loaded without compilation. If you need to continue training,
        call fit() which will rebuild and recompile the model.
        """
        # Define dummy loss functions for deserialization (not used for prediction)
        def weighted_categorical_crossentropy(y_true, y_pred):
            """Dummy loss function for model deserialization. Not used for prediction."""
            import tensorflow as tf
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        def mean_f1_loss(y_true, y_pred):
            """Dummy loss function for model deserialization. Not used for prediction."""
            import tensorflow as tf
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # Load without compiling to avoid issues with custom loss functions
        # For prediction, we don't need the loss function
        # For training, fit() will rebuild the model anyway
        custom_objects = {
            'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
            'mean_f1_loss': mean_f1_loss
        }
        self.keras_model = keras.models.load_model(path, custom_objects=custom_objects, compile=False)
        self.classifier_model = self.keras_model
        self.model_weights = self.keras_model.get_weights()
        return self
