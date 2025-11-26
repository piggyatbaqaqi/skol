"""
SkolClassifierV2: Unified API for taxonomic text classification.

This module provides a cleaner, more unified interface for training and predicting
taxonomic labels from text, with configuration-driven behavior instead of multiple
methods for different sources/destinations.

Key improvements over SkolClassifier:
- Single constructor controls all configuration
- Unified methods: load_raw(), fit(), predict(), save_annotated(), load_model(), save_model()
- Mutually exclusive optional parameters for different sources (CouchDB vs files vs strings)
- Cleaner API with fewer methods

Example usage:

    # Train from files, save model to disk
    classifier = SkolClassifierV2(
        input_source='files',
        file_paths=['data/train/*.txt.ann'],
        model_storage='disk',
        model_path='models/my_model.pkl',
        line_level=True,
        use_suffixes=True,
        model_type='logistic'
    )
    classifier.fit()

    # Predict from CouchDB, save to CouchDB
    classifier = SkolClassifierV2(
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='taxonomic_articles',
        couchdb_username='admin',
        couchdb_password='password',
        couchdb_pattern='*.txt',
        output_dest='couchdb',
        output_couchdb_suffix='.ann',
        model_storage='disk',
        model_path='models/my_model.pkl',
        line_level=True,
        coalesce_labels=True
    )
    raw_df = classifier.load_raw()
    predictions_df = classifier.predict(raw_df)
    classifier.save_annotated(predictions_df)
"""

from typing import Optional, List, Dict, Any, Literal
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel

# Import from existing classifier module
from .preprocessing import FeatureExtractor
from .model import SkolModel
from .couchdb_io import CouchDBConnection


class SkolClassifierV2:
    """
    Unified classifier for taxonomic text with configuration-driven behavior.

    This class provides a clean API with all configuration in the constructor
    and unified methods for loading, training, predicting, and saving.

    Constructor Parameters:
    ----------------------
    Core Configuration:
        spark: SparkSession for distributed processing

    Input Configuration (mutually exclusive groups):
        input_source: Where to load data from ('files', 'couchdb', or 'strings')

        For input_source='files':
            file_paths: List of file paths or glob patterns

        For input_source='couchdb':
            couchdb_url: CouchDB server URL
            couchdb_database: Database name
            couchdb_username: Optional username
            couchdb_password: Optional password
            couchdb_pattern: Attachment pattern (e.g., '*.txt')

    Output Configuration:
        output_dest: Where to save predictions ('files', 'couchdb', or 'strings')
        output_path: Directory for saving files (if output_dest='files')
        output_couchdb_suffix: Suffix for CouchDB attachments (if output_dest='couchdb')

    Model Storage Configuration:
        model_storage: Where to store trained model ('disk', 'redis', or None)
        model_path: Path for disk storage
        redis_client: Redis client instance for redis storage
        redis_key: Key for redis storage
        auto_load_model: Whether to load model from storage on initialization

    Processing Configuration:
        line_level: Whether to process at line level (True) or paragraph level (False)
        collapse_labels: Whether to collapse similar labels during training
        coalesce_labels: Whether to merge consecutive same-label predictions
        output_format: Format for predictions ('annotated', 'labels', 'probs')

    Feature Configuration:
        use_suffixes: Whether to use word suffix features
        min_doc_freq: Minimum document frequency for word features

    Model Configuration:
        model_type: Type of model ('logistic', 'random_forest', 'gradient_boosted')
        **model_params: Additional parameters passed to the model

    Methods:
    -------
    load_raw() -> DataFrame:
        Load raw data from configured input source

    fit(annotated_data: Optional[DataFrame] = None) -> Dict[str, Any]:
        Train model on annotated data (loads from input_source if not provided)

    predict(raw_data: Optional[DataFrame] = None) -> DataFrame:
        Make predictions on raw data (loads from input_source if not provided)

    save_annotated(predictions: DataFrame) -> None:
        Save predictions to configured output destination

    load_model() -> None:
        Load model from configured storage

    save_model() -> None:
        Save model to configured storage
    """

    def __init__(
        self,
        # Core
        spark: Optional[SparkSession] = None,

        # Input configuration
        input_source: Literal['files', 'couchdb', 'strings'] = 'files',
        file_paths: Optional[List[str]] = None,
        couchdb_url: Optional[str] = None,
        couchdb_database: Optional[str] = None,
        couchdb_username: Optional[str] = None,
        couchdb_password: Optional[str] = None,
        couchdb_pattern: Optional[str] = None,

        # Output configuration
        output_dest: Literal['files', 'couchdb', 'strings'] = 'files',
        output_path: Optional[str] = None,
        output_couchdb_suffix: Optional[str] = '.ann',

        # Model storage configuration
        model_storage: Optional[Literal['disk', 'redis']] = None,
        model_path: Optional[str] = None,
        redis_client: Optional[Any] = None,
        redis_key: Optional[str] = None,
        auto_load_model: bool = False,

        # Processing configuration
        line_level: bool = False,
        collapse_labels: bool = True,
        coalesce_labels: bool = False,
        output_format: Literal['annotated', 'labels', 'probs'] = 'annotated',

        # Feature configuration
        use_suffixes: bool = True,
        min_doc_freq: int = 2,

        # Model configuration
        model_type: str = 'logistic',
        **model_params
    ):
        # Core
        self.spark = spark or SparkSession.builder.getOrCreate()

        # Input configuration
        self.input_source = input_source
        self.file_paths = file_paths
        self.couchdb_url = couchdb_url
        self.couchdb_database = couchdb_database
        self.couchdb_username = couchdb_username
        self.couchdb_password = couchdb_password
        self.couchdb_pattern = couchdb_pattern or '*.txt'

        # Output configuration
        self.output_dest = output_dest
        self.output_path = output_path
        self.output_couchdb_suffix = output_couchdb_suffix

        # Model storage configuration
        self.model_storage = model_storage
        self.model_path = model_path
        self.redis_client = redis_client
        self.redis_key = redis_key

        # Processing configuration
        self.line_level = line_level
        self.collapse_labels = collapse_labels
        self.coalesce_labels = coalesce_labels
        self.output_format = output_format

        # Feature configuration
        self.use_suffixes = use_suffixes
        self.min_doc_freq = min_doc_freq

        # Model configuration
        self.model_type = model_type
        self.model_params = model_params

        # Internal state
        self._feature_extractor: Optional[FeatureExtractor] = None
        self._feature_pipeline: Optional[PipelineModel] = None
        self._model: Optional[SkolModel] = None
        self._label_mapping: Optional[Dict[str, int]] = None
        self._reverse_label_mapping: Optional[Dict[int, str]] = None

        # Validate configuration
        self._validate_config()

        # Auto-load model if requested
        if auto_load_model and model_storage:
            self.load_model()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate input source configuration
        if self.input_source == 'files':
            if not self.file_paths:
                raise ValueError("file_paths must be provided when input_source='files'")
        elif self.input_source == 'couchdb':
            if not self.couchdb_url or not self.couchdb_database:
                raise ValueError(
                    "couchdb_url and couchdb_database must be provided when input_source='couchdb'"
                )

        # Validate output destination configuration
        if self.output_dest == 'files':
            if not self.output_path:
                raise ValueError("output_path must be provided when output_dest='files'")
        elif self.output_dest == 'couchdb':
            if not self.couchdb_url or not self.couchdb_database:
                raise ValueError(
                    "couchdb_url and couchdb_database must be provided when output_dest='couchdb'"
                )

        # Validate model storage configuration
        if self.model_storage == 'disk':
            if not self.model_path:
                raise ValueError("model_path must be provided when model_storage='disk'")
        elif self.model_storage == 'redis':
            if not self.redis_client or not self.redis_key:
                raise ValueError(
                    "redis_client and redis_key must be provided when model_storage='redis'"
                )

    def load_raw(self) -> DataFrame:
        """
        Load raw (unannotated) data from configured input source.

        Returns:
            DataFrame with raw text data

        Raises:
            ValueError: If input_source is not properly configured
        """
        if self.input_source == 'files':
            return self._load_raw_from_files()
        elif self.input_source == 'couchdb':
            return self._load_raw_from_couchdb()
        else:
            raise ValueError(f"load_raw() not supported for input_source='{self.input_source}'")

    def fit(self, annotated_data: Optional[DataFrame] = None) -> Dict[str, Any]:
        """
        Train model on annotated data.

        Args:
            annotated_data: Optional DataFrame with annotations. If not provided,
                          loads from configured input source.

        Returns:
            Dictionary with training statistics

        Raises:
            ValueError: If no annotated data is provided or available
        """
        # Load annotated data if not provided
        if annotated_data is None:
            annotated_data = self._load_annotated_data()

        # Build feature pipeline
        self._feature_extractor = FeatureExtractor(
            use_suffixes=self.use_suffixes,
            min_doc_freq=self.min_doc_freq
        )

        # Fit features and transform data
        featured_df = self._feature_extractor.fit_transform(annotated_data)

        # Store feature pipeline
        self._feature_pipeline = self._feature_extractor.get_pipeline()

        # Train model
        self._model = SkolModel(model_type=self.model_type, **self.model_params)
        stats = self._model.fit(featured_df)

        # Store label mappings
        self._label_mapping = self._feature_extractor.get_label_mapping()
        self._reverse_label_mapping = {v: k for k, v in self._label_mapping.items()}

        # Auto-save model if configured
        if self.model_storage:
            self.save_model()

        return stats

    def predict(self, raw_data: Optional[DataFrame] = None) -> DataFrame:
        """
        Make predictions on raw data.

        Args:
            raw_data: Optional DataFrame with raw text. If not provided,
                     loads from configured input source.

        Returns:
            DataFrame with predictions

        Raises:
            ValueError: If model is not trained or loaded
            ValueError: If no raw data is provided or available
        """
        if self._model is None or self._feature_pipeline is None:
            raise ValueError("Model not trained or loaded. Call fit() or load_model() first.")

        # Load raw data if not provided
        if raw_data is None:
            raw_data = self.load_raw()

        # Apply feature pipeline
        featured_df = self._feature_pipeline.transform(raw_data)

        # Make predictions
        predictions_df = self._model.predict(featured_df)

        # Convert label indices back to strings
        predictions_df = self._decode_predictions(predictions_df)

        # Format output
        predictions_df = self._format_predictions(predictions_df)

        return predictions_df

    def save_annotated(self, predictions: DataFrame) -> None:
        """
        Save predictions to configured output destination.

        Args:
            predictions: DataFrame with predictions to save

        Raises:
            ValueError: If output destination is not properly configured
        """
        if self.output_dest == 'files':
            self._save_to_files(predictions)
        elif self.output_dest == 'couchdb':
            self._save_to_couchdb(predictions)
        else:
            raise ValueError(
                f"save_annotated() not supported for output_dest='{self.output_dest}'"
            )

    def load_model(self) -> None:
        """
        Load model from configured storage.

        Raises:
            ValueError: If model_storage is not configured
            FileNotFoundError: If model file doesn't exist (for disk storage)
        """
        if self.model_storage == 'disk':
            self._load_model_from_disk()
        elif self.model_storage == 'redis':
            self._load_model_from_redis()
        else:
            raise ValueError("model_storage not configured")

    def save_model(self) -> None:
        """
        Save model to configured storage.

        Raises:
            ValueError: If model_storage is not configured or model not trained
        """
        if self._model is None or self._feature_pipeline is None:
            raise ValueError("No model to save. Call fit() first.")

        if self.model_storage == 'disk':
            self._save_model_to_disk()
        elif self.model_storage == 'redis':
            self._save_model_to_redis()
        else:
            raise ValueError("model_storage not configured")

    # Private helper methods

    def _load_raw_from_files(self) -> DataFrame:
        """Load raw text from local files."""
        from .preprocessing import RawTextLoader

        loader = RawTextLoader(self.spark)
        return loader.load_files(
            self.file_paths,
            line_level=self.line_level
        )

    def _load_raw_from_couchdb(self) -> DataFrame:
        """Load raw text from CouchDB."""
        conn = CouchDBConnection(
            self.couchdb_url,
            self.couchdb_database,
            self.couchdb_username,
            self.couchdb_password
        )

        df = conn.load_distributed(self.spark, self.couchdb_pattern)

        # Split into lines if line_level mode
        if self.line_level:
            from pyspark.sql.functions import explode, split, col, trim, row_number, lit
            from pyspark.sql.window import Window

            df = df.withColumn("value", explode(split(col("value"), "\\n")))
            df = df.filter(trim(col("value")) != "")

            # Add line numbers
            window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy(lit(1))
            df = df.withColumn("line_number", row_number().over(window_spec) - 1)

        return df

    def _load_annotated_data(self) -> DataFrame:
        """Load annotated data for training."""
        if self.input_source == 'files':
            return self._load_annotated_from_files()
        elif self.input_source == 'couchdb':
            return self._load_annotated_from_couchdb()
        else:
            raise ValueError(
                f"Cannot load annotated data from input_source='{self.input_source}'"
            )

    def _load_annotated_from_files(self) -> DataFrame:
        """Load annotated data from files."""
        from .preprocessing import AnnotatedTextLoader

        loader = AnnotatedTextLoader(self.spark)
        return loader.load_files(
            self.file_paths,
            line_level=self.line_level,
            collapse_labels=self.collapse_labels
        )

    def _load_annotated_from_couchdb(self) -> DataFrame:
        """Load annotated data from CouchDB."""
        # Load raw annotations from CouchDB
        conn = CouchDBConnection(
            self.couchdb_url,
            self.couchdb_database,
            self.couchdb_username,
            self.couchdb_password
        )

        # Look for .ann files for training
        pattern = self.couchdb_pattern
        if not pattern.endswith('.ann'):
            pattern = pattern.replace('.txt', '.txt.ann')

        df = conn.load_distributed(self.spark, pattern)

        # Parse annotations
        from .preprocessing import AnnotatedTextParser

        parser = AnnotatedTextParser(line_level=self.line_level)
        return parser.parse(df)

    def _decode_predictions(self, predictions_df: DataFrame) -> DataFrame:
        """Convert label indices back to label strings."""
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        # Create UDF to map indices to labels
        label_map = self._reverse_label_mapping

        def decode_label(idx: int) -> str:
            return label_map.get(idx, f"UNKNOWN_{idx}")

        decode_udf = udf(decode_label, StringType())

        return predictions_df.withColumn(
            "predicted_label",
            decode_udf(predictions_df["prediction"])
        )

    def _format_predictions(self, predictions_df: DataFrame) -> DataFrame:
        """Format predictions according to output_format setting."""
        if self.output_format == 'annotated':
            # Return full annotated format
            return self._format_as_annotated(predictions_df)
        elif self.output_format == 'labels':
            # Return just labels
            return predictions_df.select("doc_id", "predicted_label")
        elif self.output_format == 'probs':
            # Return probabilities
            return predictions_df.select("doc_id", "predicted_label", "probability")
        else:
            return predictions_df

    def _format_as_annotated(self, predictions_df: DataFrame) -> DataFrame:
        """Format predictions as YEDA-style annotated blocks."""
        from .formatting import YedaFormatter

        formatter = YedaFormatter(coalesce_labels=self.coalesce_labels)
        return formatter.format(predictions_df)

    def _save_to_files(self, predictions: DataFrame) -> None:
        """Save predictions to local files."""
        from .output import FileOutputWriter

        writer = FileOutputWriter(self.output_path)
        writer.write(predictions)

    def _save_to_couchdb(self, predictions: DataFrame) -> None:
        """Save predictions to CouchDB."""
        conn = CouchDBConnection(
            self.couchdb_url,
            self.couchdb_database,
            self.couchdb_username,
            self.couchdb_password
        )

        conn.save_predictions(
            predictions,
            suffix=self.output_couchdb_suffix,
            coalesce_labels=self.coalesce_labels
        )

    def _save_model_to_disk(self) -> None:
        """Save model to disk."""
        import pickle

        model_path = Path(self.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self._model,
            'feature_pipeline': self._feature_pipeline,
            'label_mapping': self._label_mapping,
            'config': {
                'line_level': self.line_level,
                'use_suffixes': self.use_suffixes,
                'min_doc_freq': self.min_doc_freq,
                'model_type': self.model_type,
                'model_params': self.model_params
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

    def _load_model_from_disk(self) -> None:
        """Load model from disk."""
        import pickle

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self._model = model_data['model']
        self._feature_pipeline = model_data['feature_pipeline']
        self._label_mapping = model_data['label_mapping']
        self._reverse_label_mapping = {v: k for k, v in self._label_mapping.items()}

        # Optionally update config from saved model
        # (or warn if mismatch)

    def _save_model_to_redis(self) -> None:
        """Save model to Redis."""
        import pickle

        model_data = {
            'model': self._model,
            'feature_pipeline': self._feature_pipeline,
            'label_mapping': self._label_mapping,
            'config': {
                'line_level': self.line_level,
                'use_suffixes': self.use_suffixes,
                'min_doc_freq': self.min_doc_freq,
                'model_type': self.model_type,
                'model_params': self.model_params
            }
        }

        serialized = pickle.dumps(model_data)
        self.redis_client.set(self.redis_key, serialized)

    def _load_model_from_redis(self) -> None:
        """Load model from Redis."""
        import pickle

        serialized = self.redis_client.get(self.redis_key)
        if not serialized:
            raise ValueError(f"No model found in Redis with key: {self.redis_key}")

        model_data = pickle.loads(serialized)

        self._model = model_data['model']
        self._feature_pipeline = model_data['feature_pipeline']
        self._label_mapping = model_data['label_mapping']
        self._reverse_label_mapping = {v: k for k, v in self._label_mapping.items()}
