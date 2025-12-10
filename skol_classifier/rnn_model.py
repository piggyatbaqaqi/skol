"""
RNN-based model for sequential line classification with context using Keras + Pandas UDFs.

This module provides an LSTM/GRU-based model that uses surrounding lines
as context to improve classification accuracy for individual lines. It uses
PySpark Pandas UDFs for distributed training instead of Elephas.
"""

import gc
from typing import Optional, List, Tuple, Dict
import numpy as np
import os

# Configure TensorFlow GPU settings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, collect_list, lit, pandas_udf, posexplode
)
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml import Transformer
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

    Raises:
        RuntimeError: If GPU initialization fails with helpful error message
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

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
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
        batch_size: int = 32,
        epochs: int = 10,
        num_workers: int = 4,
        features_col: str = "combined_idf",
        label_col: str = "label_indexed",
        verbosity: int = 3,
        name: str = "RNN_BiLSTM"
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
            verbosity: Verbosity level for logging
            name: Name of the model
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
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.verbosity = verbosity
        self.name = name

        # Build Keras model
        self.keras_model = build_bilstm_model(
            input_shape=(window_size, input_size),
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        self.model_weights = None

    def _process_row_to_windows(self, row):
        """Process a single row into training windows."""
        features = row.sequence_features
        labels_seq = row.sequence_labels

        # Convert sparse vectors to dense arrays
        dense_features = []
        for feat in features:
            if feat is None:
                continue
            elif hasattr(feat, 'toArray'):
                dense_features.append(feat.toArray().tolist())
            elif isinstance(feat, (list, tuple)):
                dense_features.append(list(feat))
            else:
                try:
                    dense_features.append(list(feat))
                except TypeError:
                    continue

        # Convert labels to list
        dense_labels = [float(l) for l in labels_seq if l is not None]

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
            sequenced_data: DataFrame with sequence_features and sequence_labels
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
        if self.verbosity >= 3:
            print(f"[RNN Fit] Using document ID column: {doc_id_col}")

        # Create sequence preprocessor
        if self.verbosity >= 3:
            print("[RNN Fit] Creating sequence preprocessor...")
        preprocessor = SequencePreprocessor(
            inputCol=self.features_col,
            outputCol="sequence_features",
            docIdCol=doc_id_col,
            labelCol=self.label_col,
            window_size=self.window_size
        )

        # Transform data into sequences
        if self.verbosity >= 3:
            print("[RNN Fit] Transforming data into sequences...")
        sequenced_data = preprocessor.transform(train_data)

        # Cache to avoid recomputation
        if self.verbosity >= 3:
            print("[RNN Fit] Caching sequenced data...")
        sequenced_data = sequenced_data.cache()

        # Detect actual feature dimension from data
        if self.verbosity >= 3:
            print("[RNN Fit] Detecting actual feature dimension...")
        sample_row = sequenced_data.select("sequence_features").first()
        if sample_row and sample_row.sequence_features:
            first_feature = sample_row.sequence_features[0]
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
                    dropout=self.dropout
                )
                if self.verbosity >= 2:
                    print(f"[RNN Fit] Model rebuilt with input_size={self.input_size}")
            else:
                if self.verbosity >= 3:
                    print(f"[RNN Fit] Feature dimension matches: {actual_input_size}")

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
            # Just count the number of features (lines) without converting to dense
            num_features = len(row.sequence_features) if row.sequence_features else 0
            # Estimate windows as ceil(num_features / window_size)
            doc_windows = max(1, (num_features + self.window_size - 1) // self.window_size)
            total_windows += doc_windows
            if self.verbosity >= 4:
                print(f"[RNN Fit]   Sample {idx+1}: {num_features} lines -> ~{doc_windows} windows")
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

        Args:
            data: DataFrame to predict on

        Returns:
            DataFrame with predictions
        """
        if self.verbosity >= 2:
            print("[RNN Predict] Starting prediction")
            print(f"[RNN Predict] Input data columns: {data.columns}")
            print(f"[RNN Predict] Input data count: {data.count()}")

        if self.classifier_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Determine document ID column
        doc_id_col = "doc_id" if "doc_id" in data.columns else "filename"
        if self.verbosity >= 3:
            print(f"[RNN Predict] Using document ID column: {doc_id_col}")

        # Create sequence preprocessor
        if self.verbosity >= 3:
            print("[RNN Predict] Creating sequence preprocessor")
        preprocessor = SequencePreprocessor(
            inputCol=self.features_col,
            outputCol="sequence_features",
            docIdCol=doc_id_col,
            labelCol=self.label_col if self.label_col in data.columns else "dummy_label",
            window_size=self.window_size
        )

        has_labels = self.label_col in data.columns
        if self.verbosity >= 3:
            print(f"[RNN Predict] Has labels: {has_labels}")

        # Add dummy labels if not present (for prediction)
        if not has_labels:
            data = data.withColumn(self.label_col, lit(0.0))
            if self.verbosity >= 3:
                print("[RNN Predict] Added dummy labels for prediction")

        # Transform into sequences
        if self.verbosity >= 2:
            print("[RNN Predict] Transforming data into sequences")
        sequenced_data = preprocessor.transform(data)
        if self.verbosity >= 3:
            print(f"[RNN Predict] Sequenced data columns: {sequenced_data.columns}")

        # Broadcast model weights for UDF
        if self.verbosity >= 3:
            print("[RNN Predict] Preparing model config and weights for UDF")
        model_config = self.keras_model.to_json()
        model_weights = self.model_weights
        input_size = self.input_size
        window_size = self.window_size

        # Define prediction UDF
        @pandas_udf(ArrayType(FloatType()))
        def predict_sequence(features_series: pd.Series) -> pd.Series:
            """Predict on sequences using the trained model."""
            import os
            import numpy as np

            # Force CPU-only mode in executors to prevent CUDA errors
            # This is critical for GPUs with unsupported compute capabilities
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

            # Import TensorFlow/Keras inside UDF after setting CPU mode
            try:
                import tensorflow as tf
                # Double-check GPU is disabled
                tf.config.set_visible_devices([], 'GPU')
                from tensorflow import keras
            except Exception as e:
                # If TensorFlow config fails, try to continue anyway
                # The CPU-only env var should be sufficient
                pass

            # Rebuild model from config and weights
            try:
                model = keras.models.model_from_json(model_config)
                model.set_weights(model_weights)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to rebuild model in executor: {e}\n"
                    "This may be due to GPU compatibility issues. "
                    "Ensure CUDA_VISIBLE_DEVICES='' is set before starting Spark."
                )

            results = []
            for features in features_series:
                # Track original length before padding/truncating
                original_length = len(features)

                # Pad or truncate
                if len(features) < window_size:
                    padding = [[0.0] * input_size] * (window_size - len(features))
                    features = features + padding
                else:
                    features = features[:window_size]

                # Predict
                X = np.array([features], dtype=np.float32)
                try:
                    preds = model.predict(X, verbose=0)[0]
                except Exception as e:
                    raise RuntimeError(
                        f"Prediction failed in executor: {e}\n"
                        "This may be a CUDA/GPU error. Ensure CPU-only mode is active."
                    )

                # Get argmax for each timestep
                pred_classes = [float(np.argmax(p)) for p in preds]
                # Slice to original length (before padding/truncating)
                results.append(pred_classes[:original_length])

            return pd.Series(results)

        # Apply prediction
        if self.verbosity >= 2:
            print("[RNN Predict] Applying prediction UDF to sequences")
        predictions = sequenced_data.withColumn(
            "predictions",
            predict_sequence(col("sequence_features"))
        )
        if self.verbosity >= 3:
            print(f"[RNN Predict] Predictions columns: {predictions.columns}")

        # For line-level predictions, we need to explode the sequences back to individual lines
        # Use posexplode to get both position and value
        if has_labels:
            if self.verbosity >= 2:
                print("[RNN Predict] Exploding predictions and labels for evaluation")

            # If we have labels (e.g., for evaluation), explode both predictions and labels
            if self.verbosity >= 3:
                print("[RNN Predict] Exploding predictions")
            predictions_exploded = predictions.select(
                col(doc_id_col).alias("filename"),
                posexplode(col("predictions")).alias("pos", "prediction")
            )

            # Explode labels separately
            if self.verbosity >= 3:
                print("[RNN Predict] Exploding labels")
            labels_exploded = predictions.select(
                col(doc_id_col).alias("filename"),
                posexplode(col("sequence_labels")).alias("pos", self.label_col)
            )

            # Join on filename and position to align predictions with labels
            # Cast prediction to DoubleType as required by Spark ML evaluators
            if self.verbosity >= 2:
                print("[RNN Predict] Joining predictions with labels")
            result = predictions_exploded.join(
                labels_exploded,
                on=["filename", "pos"],
                how="inner"
            ).select(
                "filename",
                col("prediction").cast("double"),
                self.label_col
            )
            if self.verbosity >= 2:
                print(f"[RNN Predict] Result columns: {result.columns}")
        else:
            if self.verbosity >= 2:
                print("[RNN Predict] Exploding predictions (no labels)")
            # No labels, just return predictions
            result = predictions.select(
                col(doc_id_col).alias("filename"),
                posexplode(col("predictions")).alias("pos", "prediction")
            ).select(
                "filename",
                col("prediction").cast("double"),
                self.label_col
            )

        if self.verbosity >= 1:
            print("[RNN Predict] Prediction completed successfully")
        return result

    def calculate_stats(
        self,
        predictions: DataFrame,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Calculate evaluation statistics for RNN predictions.

        Overrides the base method to handle RNN-specific prediction format,
        which includes a 'filename' column and line-level predictions that
        have been exploded from sequences.

        Args:
            predictions: DataFrame with predictions and labels.
                        Expected to have columns: filename, prediction, label_col

        Returns:
            Dictionary containing accuracy, precision, recall, f1_score
        """
        if self.verbosity >= 3:
            print("[RNN Stats] Calculating statistics for RNN predictions")
            print(f"[RNN Stats] Predictions schema: {predictions.schema}")

        # Verify required columns are present
        required_cols = {"prediction", self.label_col}
        actual_cols = set(predictions.columns)

        if not required_cols.issubset(actual_cols):
            missing = required_cols - actual_cols
            raise ValueError(
                f"Predictions DataFrame missing required columns: {missing}. "
                f"Available columns: {actual_cols}"
            )

        # RNN predictions may have extra columns like 'filename' which evaluators ignore
        # We can use the predictions as-is, but let's select only the columns needed
        # for evaluation to ensure compatibility
        eval_predictions = predictions.select("prediction", self.label_col)

        if self.verbosity >= 3:
            print(f"[RNN Stats] Evaluating {eval_predictions.count()} line-level predictions")

        # Use parent class method to create evaluators and calculate stats
        evaluators = self._create_evaluators()

        stats = {
            'accuracy': evaluators['accuracy'].evaluate(eval_predictions),
            'precision': evaluators['precision'].evaluate(eval_predictions),
            'recall': evaluators['recall'].evaluate(eval_predictions),
            'f1_score': evaluators['f1'].evaluate(eval_predictions)
        }

        if self.verbosity >= 1:
            print("=" * 50)
            print("RNN Model Evaluation Statistics (Line-Level)")
            print("=" * 50)
            print(f"Test Accuracy:  {stats['accuracy']:.4f}")
            print(f"Test Precision: {stats['precision']:.4f}")
            print(f"Test Recall:    {stats['recall']:.4f}")
            print(f"Test F1 Score:  {stats['f1_score']:.4f}")
            print("=" * 50)

        return stats

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
