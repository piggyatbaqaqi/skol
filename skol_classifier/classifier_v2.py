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

    # Train from files, save model to disk (line-level extraction)
    classifier = SkolClassifierV2(
        input_source='files',
        file_paths=['data/train/*.txt.ann'],
        model_storage='disk',
        model_path='models/my_model.pkl',
        extraction_mode='line',
        use_suffixes=True,
        model_type='logistic'
    )
    classifier.fit()

    # Predict from CouchDB PDFs with section-level extraction
    classifier = SkolClassifierV2(
        input_source='couchdb',
        couchdb_url='http://localhost:5984',
        couchdb_database='taxonomic_articles',
        couchdb_username='admin',
        couchdb_password='password',
        # couchdb_doc_ids auto-discovers all PDFs if not specified
        output_dest='couchdb',
        output_couchdb_suffix='.ann',
        model_storage='disk',
        model_path='models/my_model.pkl',
        extraction_mode='section',
        section_filter=['Introduction', 'Methods', 'Results'],  # Optional filtering
        coalesce_labels=True
    )
    raw_df = classifier.load_raw()
    predictions_df = classifier.predict(raw_df)
    classifier.save_annotated(predictions_df)
"""

from typing import Optional, List, Dict, Any, Literal, cast
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import explode, split, col, trim, row_number, lit
from pyspark.sql.window import Window

# Import helper classes
from .feature_extraction import FeatureExtractor
from .base_model import SkolModel
from .model import create_model
from .output_formatters import YeddaFormatter, FileOutputWriter

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
            couchdb_training_database: Optional separate database for training data
                                      If specified, training data is loaded from this database
                                      while predictions use couchdb_database
            couchdb_username: Optional username
            couchdb_password: Optional password
            couchdb_pattern: Attachment pattern (e.g., '*.txt')
            couchdb_doc_ids: Optional list of document IDs (for section mode)
                           If not provided in section mode, auto-discovers all PDFs

    Output Configuration:
        output_dest: Where to save predictions ('files', 'couchdb', or 'strings')
        output_path: Directory for saving files (if output_dest='files')
        output_couchdb_suffix: Suffix for CouchDB attachments (if output_dest='couchdb')

    Model Storage Configuration:
        model_storage: Where to store trained model ('disk', 'redis', or None)
        model_path: Path for disk storage
        redis_client: Redis client instance for redis storage
        redis_expire: If not None, expiration time for redis storage
        redis_key: Key for redis storage
        auto_load_model: Whether to load model from storage on initialization

    Processing Configuration:
        extraction_mode: Text extraction granularity ('line', 'paragraph', or 'section')
                  'line': Extract and process text line-by-line (equivalent to old line_level=True)
                  'paragraph': Extract and process text by paragraphs (equivalent to old line_level=False)
                  'section': Extract sections from PDFs with section name features
        section_filter: Optional list of section names to include (for section mode)
                       Example: ['Introduction', 'Methods', 'Results']
        read_text: If True, read from existing .txt attachment instead of converting PDF
                   Only applies to section extraction mode
        save_text: If True, save extracted PDF text as .txt attachment
                   Only applies to section extraction mode
                   If both read_text and save_text are True: always convert PDF and replace .txt
                   If read_text=True and save_text=False: convert PDF but don't save
        collapse_labels: Whether to collapse similar labels during training
        coalesce_labels: Whether to merge consecutive same-label predictions
        output_format: Format for predictions ('annotated', 'labels', 'probs')

    Feature Configuration:
        use_suffixes: Whether to use word suffix features
        min_doc_freq: Minimum document frequency for word features
        word_vocab_size: Maximum vocabulary size for word features (default: 800)
        suffix_vocab_size: Maximum vocabulary size for suffix features (default: 200)
        section_name_vocab_size: Maximum vocabulary size for section name features (default: 50)
                                Used when extraction_mode='section' to create TF-IDF features from section names

    Model Configuration:
        model_type: Type of model ('logistic', 'random_forest', 'gradient_boosted', 'rnn')
        **model_params: Additional parameters passed to the model

        RNN Model Parameters (when model_type='rnn'):
            input_size: Size of input feature vectors (default: auto-calculated from vocab sizes)
                       Automatically set to word_vocab_size + suffix_vocab_size
            hidden_size: LSTM hidden state size (default: 128)
            num_layers: Number of LSTM layers (default: 2)
            num_classes: Number of output classes (default: 3)
            dropout: Dropout rate (default: 0.3)
            window_size: Maximum sequence length (default: 50)
            batch_size: Batch size for training (default: 32)
            epochs: Number of training epochs (default: 10)
            num_workers: Number of Spark workers (default: 4)

    Methods:
    -------
    load_raw() -> DataFrame:
        Load raw data from configured input source

    fit(annotated_data: Optional[DataFrame] = None) -> Dict[str, Any]:
        Train model on annotated data (loads from input_source if not provided)

    predict(raw_data: Optional[DataFrame] = None) -> DataFrame:
        Make predictions on raw data (loads from input_source if not provided)

    save_annotated(predictions: DataFrame) -> Optional[List[str]]:
        Save predictions to configured output destination
        Returns List[str] if output_dest='strings', None otherwise

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
        couchdb_training_database: Optional[str] = None,  # Separate database for training
        couchdb_username: Optional[str] = None,
        couchdb_password: Optional[str] = None,
        couchdb_pattern: Optional[str] = None,
        couchdb_doc_ids: Optional[List[str]] = None,  # For section mode

        # Output configuration
        output_dest: Literal['files', 'couchdb', 'strings'] = 'files',
        output_path: Optional[str] = None,
        output_couchdb_suffix: Optional[str] = '.ann',

        # Model storage configuration
        model_storage: Optional[Literal['disk', 'redis']] = None,
        model_path: Optional[str] = None,
        redis_client: Optional[Any] = None,
        redis_key: Optional[str] = None,
        redis_expire: Optional[int] = None,
        auto_load_model: bool = False,

        # Processing configuration
        extraction_mode: Literal['line', 'paragraph', 'section'] = 'paragraph',
        section_filter: Optional[List[str]] = None,  # Filter by section names (for section mode)
        read_text: bool = False,  # Read from .txt attachment instead of converting PDF
        save_text: Optional[Literal['eager', 'lazy']] = None,  # 'eager': always save, 'lazy': save if missing, None: don't save
        collapse_labels: bool = True,
        coalesce_labels: bool = False,
        output_format: Literal['annotated', 'labels', 'probs'] = 'annotated',
        compute_label_frequencies: bool = False,

        # Feature configuration
        use_suffixes: bool = True,
        min_doc_freq: int = 2,
        word_vocab_size: int = 800,
        suffix_vocab_size: int = 200,
        section_name_vocab_size: int = 50,

        # Model configuration
        model_type: str = 'logistic',
        weight_strategy: Optional[Literal['inverse', 'balanced', 'aggressive']] = None,
        min_weight: float = 0.1,
        max_weight: float = 100.0,
        verbosity: int = 1,
        **model_params: Any
    ):
        # Core
        self.spark = spark or SparkSession.builder.getOrCreate()

        # Input configuration
        self.input_source = input_source
        self.file_paths = file_paths
        self.couchdb_url = couchdb_url
        self.couchdb_database = couchdb_database
        self.couchdb_training_database = couchdb_training_database
        self.couchdb_username = couchdb_username
        self.couchdb_password = couchdb_password
        self.couchdb_pattern = couchdb_pattern or '*.txt'
        self.couchdb_doc_ids = couchdb_doc_ids

        # Output configuration
        self.output_dest = output_dest
        self.output_path = output_path
        self.output_couchdb_suffix = output_couchdb_suffix

        # Model storage configuration
        self.model_storage = model_storage
        self.model_path = model_path
        self.redis_client = redis_client
        self.redis_key = redis_key
        self.redis_expire = redis_expire

        # Processing configuration
        from .extraction_modes import get_mode
        self.extraction_mode = get_mode(extraction_mode)
        self.section_filter = section_filter
        self.read_text = read_text
        self.save_text = save_text
        self.collapse_labels = collapse_labels
        self.coalesce_labels = coalesce_labels
        self.output_format = output_format
        self.weight_strategy = weight_strategy
        self.min_weight = min_weight
        self.max_weight = max_weight
        # Auto-enable frequency computation if weight strategy is specified
        self.compute_label_frequencies = compute_label_frequencies or (weight_strategy is not None)

        # Feature configuration
        self.use_suffixes = use_suffixes
        self.min_doc_freq = min_doc_freq
        self.word_vocab_size = word_vocab_size
        self.suffix_vocab_size = suffix_vocab_size
        self.section_name_vocab_size = section_name_vocab_size

        # Model configuration
        self.model_type = model_type
        self.model_params = model_params
        self.model_params['verbosity'] = verbosity
        self.verbosity = verbosity

        # Internal state
        self._feature_extractor: Optional[FeatureExtractor] = None
        self._feature_pipeline: Optional[PipelineModel] = None
        self._model: Optional[SkolModel] = None
        self._label_mapping: Optional[Dict[str, int]] = None
        self._reverse_label_mapping: Optional[Dict[int, str]] = None
        self._label_frequencies: Optional[Dict[str, int]] = None

        # Validate configuration
        self._validate_config()

        # Auto-load model if requested
        if auto_load_model and model_storage:
            self.load_model()

    @property
    def line_level(self) -> bool:
        """Backwards compatibility property: returns True if extraction_mode is 'line'."""
        return self.extraction_mode == 'line'

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

        # Apply weight strategy if specified
        if self.weight_strategy is not None:
            # Type checker narrowing: at this point weight_strategy is one of the literals
            strategy = cast(Literal['inverse', 'balanced', 'aggressive'], self.weight_strategy)
            recommended_weights = self.get_recommended_class_weights(
                strategy=strategy,
                min_weight=self.min_weight,
                max_weight=self.max_weight
            )
            if recommended_weights is not None:
                self.model_params['class_weights'] = recommended_weights
                if self.verbosity >= 1:
                    print(f"\n[Classifier] Applied '{self.weight_strategy}' weight strategy:")
                    sorted_weights = sorted(recommended_weights.items(), key=lambda x: x[1], reverse=True)
                    for label, weight in sorted_weights:
                        print(f"  {label:<20} {weight:>6.2f}")
                    print()
            else:
                if self.verbosity >= 1:
                    print(f"[Classifier] WARNING: Could not compute weights for strategy '{self.weight_strategy}' - label frequencies not available")

        # Build feature pipeline
        # Enable section name features when using 'section' extraction mode
        use_section_names = (self.extraction_mode == 'section')

        self._feature_extractor = FeatureExtractor(
            use_suffixes=self.use_suffixes,
            use_section_names=use_section_names,
            min_doc_freq=self.min_doc_freq,
            word_vocab_size=self.word_vocab_size,
            suffix_vocab_size=self.suffix_vocab_size,
            section_name_vocab_size=self.section_name_vocab_size
        )

        # Fit features and transform data
        featured_df = self._feature_extractor.fit_transform(annotated_data)

        # Cache featured DataFrame to avoid recomputation and reduce memory pressure
        featured_df = featured_df.persist()

        # Store feature pipeline
        self._feature_pipeline = self._feature_extractor.get_pipeline()

        # Get the features column name based on configuration
        features_col = self._feature_extractor.get_features_col()

        # Get labels from feature extractor (available after fit_transform)
        labels = self._feature_extractor.get_labels()

        # Calculate input_size based on vocabulary sizes
        # This ensures consistency between feature extraction and model input
        calculated_input_size = self.word_vocab_size + (self.suffix_vocab_size if self.use_suffixes else 0)

        # Set input_size in model_params if not already specified
        if 'input_size' not in self.model_params:
            self.model_params['input_size'] = calculated_input_size
        elif self.model_params['input_size'] != calculated_input_size:
            # Warn if user-provided input_size doesn't match vocabulary configuration
            if self.verbosity >= 1:
                print(f"[Classifier] WARNING: input_size ({self.model_params['input_size']}) doesn't match "
                      f"calculated size ({calculated_input_size}) from word_vocab_size ({self.word_vocab_size}) "
                      f"+ suffix_vocab_size ({self.suffix_vocab_size}). Using user-provided value.")

        # Train model with correct features column using factory
        # The label column is always "label_indexed" from the feature extractor
        # Pass labels to create_model so RNN can use them for class weights
        self._model = create_model(
            model_type=self.model_type,
            features_col=features_col,
            label_col="label_indexed",
            labels=labels,  # Pass labels for class weight support
            **self.model_params
        )

        # Fit model and pass labels for later use
        self._model.fit(featured_df, labels=labels)

        # Check if model has verbosity for logging
        model_verbosity = getattr(self._model, 'verbosity', 0)
        if model_verbosity >= 1:
            print("[Classifier Fit] Model training completed, starting evaluation")

        # Store label mappings (labels is a list like ['Label1', 'Label2'])
        labels_list = self._feature_extractor.get_label_mapping()
        if labels_list is not None:
            # Create dict mapping from label to index
            self._label_mapping = {label: i for i, label in enumerate(labels_list)}
            # Create reverse mapping from index to label
            self._reverse_label_mapping = {i: label for i, label in enumerate(labels_list)}
            if model_verbosity >= 2:
                print(f"[Classifier Fit] Stored label mappings for {len(labels_list)} labels")

        # Split data for evaluation
        # IMPORTANT: Split by document (filename/doc_id), not by row
        # to ensure all lines from a document stay together
        if model_verbosity >= 1:
            print("[Classifier Fit] Splitting data for evaluation (80/20)")

        # Determine which column to use for document grouping
        doc_col = "filename" if "filename" in featured_df.columns else "doc_id"

        if model_verbosity >= 2:
            print(f"[Classifier Fit] Grouping by document column: {doc_col}")

        # Get unique documents and split them randomly
        from pyspark.sql.functions import rand
        unique_docs = featured_df.select(doc_col).distinct()

        if model_verbosity >= 2:
            doc_count = unique_docs.count()
            print(f"[Classifier Fit] Total unique documents: {doc_count}")

        # Add random column for splitting
        unique_docs_with_rand = unique_docs.withColumn("rand", rand(seed=42))

        # Split documents into train and test (80/20)
        train_docs = unique_docs_with_rand.filter("rand < 0.8").select(doc_col)
        test_docs = unique_docs_with_rand.filter("rand >= 0.8").select(doc_col)

        # Filter featured_df to get train and test data based on document assignments
        train_data = featured_df.join(train_docs, on=doc_col, how="inner")
        test_data = featured_df.join(test_docs, on=doc_col, how="inner")

        # Sort by doc_col and line_number to maintain ordering within documents
        if "line_number" in featured_df.columns:
            train_data = train_data.orderBy(doc_col, "line_number")
            test_data = test_data.orderBy(doc_col, "line_number")
            if model_verbosity >= 2:
                print("[Classifier Fit] Data sorted by document and line_number")
        elif "paragraph_number" in featured_df.columns:
            train_data = train_data.orderBy(doc_col, "paragraph_number")
            test_data = test_data.orderBy(doc_col, "paragraph_number")
            if model_verbosity >= 2:
                print("[Classifier Fit] Data sorted by document and paragraph_number")
        else:
            train_data = train_data.orderBy(doc_col)
            test_data = test_data.orderBy(doc_col)
            if model_verbosity >= 2:
                print("[Classifier Fit] Warning: No line_number column, sorted by document only")

        if model_verbosity >= 2:
            print("[Classifier Fit] Counting split data...")
            train_count = train_data.count()
            test_count = test_data.count()
            print(f"[Classifier Fit]   Train data count: {train_count}")
            print(f"[Classifier Fit]   Test data count: {test_count}")

        if model_verbosity >= 3:
            print("[Classifier Fit] Test data schema:")
            test_data.printSchema()
            print("[Classifier Fit] Test data sample:")
            test_data.show(5)

        # Make predictions on test set
        if model_verbosity >= 1:
            print("[Classifier Fit] Making predictions on test set")
        test_predictions = self._model.predict(test_data)

        if model_verbosity >= 1:
            print("[Classifier Fit] Predictions completed, validating output")

        # Debug: Check if predictions are empty
        if model_verbosity >= 1:  # Changed from >= 3 to ensure it runs
            pred_count = test_predictions.count()
            print(f"[Classifier Fit] Predictions count: {pred_count}")
            if pred_count == 0:
                print("[Classifier Fit] ERROR: Predictions DataFrame is EMPTY!")
                print("[Classifier Fit] This means the UDF returned no results.")
            else:
                print(f"[Classifier Fit] Predictions columns: {test_predictions.columns}")
                if model_verbosity >= 1:
                    print("[Classifier Fit] First prediction:")
                    first_pred = test_predictions.first()
                    print(f"  {first_pred}")

        # Calculate stats using model's method
        if model_verbosity >= 1:
            print("[Classifier Fit] Calculating statistics")
        stats = self._model.calculate_stats(test_predictions, verbose=False)

        if model_verbosity >= 1:
            print("[Classifier Fit] Statistics calculated, adding metadata")
        stats['train_size'] = train_data.count()
        stats['test_size'] = test_data.count()
        stats['class_frequencies'] = test_predictions.select("label_indexed").groupBy("label_indexed").count().collect()
        if model_verbosity >= 2:
            print(f"[Classifier Fit] Final stats: {stats}")

        # Unpersist featured DataFrame to free memory
        if model_verbosity >= 2:
            print("[Classifier Fit] Unpersisting featured DataFrame and result")
        featured_df.unpersist()
        test_predictions.unpersist()

        if model_verbosity >= 1:
            print("[Classifier Fit] Evaluation complete, returning stats")
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

        # Separate page markers from regular lines (line mode only)
        page_markers_df = None
        if "is_page_marker" in raw_data.columns:
            # Save page markers for later reinsertion
            page_markers_df = raw_data.filter(col("is_page_marker") == True)
            # Filter out page markers before classification
            raw_data = raw_data.filter(col("is_page_marker") == False)

        # Apply feature pipeline
        featured_df = self._feature_pipeline.transform(raw_data)

        # Make predictions
        predictions_df = self._model.predict(featured_df)

        # Convert label indices back to strings
        predictions_df = self._decode_predictions(predictions_df)

        # Format output
        predictions_df = self._format_predictions(predictions_df)

        # Reinsert page markers if they were separated
        if page_markers_df is not None:
            predictions_df = self._reinsert_page_markers(predictions_df, page_markers_df)

        return predictions_df

    def save_annotated(self, predictions: DataFrame) -> Optional[List[str]]:
        """
        Save predictions to configured output destination.

        Args:
            predictions: DataFrame with predictions to save

        Returns:
            List of annotated strings if output_dest='strings', None otherwise

        Raises:
            ValueError: If output destination is not properly configured
        """
        if self.output_dest == 'files':
            self._save_to_files(predictions)
            return None
        elif self.output_dest == 'couchdb':
            self._save_to_couchdb(predictions)
            return None
        elif self.output_dest == 'strings':
            return self._format_as_strings(predictions)
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

    def get_label_frequencies(self) -> Optional[Dict[str, int]]:
        """
        Get computed label frequencies.

        Returns:
            Dictionary mapping label strings to counts, or None if not computed.
            Example: {"Nomenclature": 100, "Description": 1000, "Misc": 10000}

        Note:
            Frequencies are only available if compute_label_frequencies=True was
            set in the constructor and fit() has been called.
        """
        return self._label_frequencies

    def get_recommended_class_weights(
        self,
        strategy: Literal['inverse', 'balanced', 'aggressive'] = 'inverse',
        min_weight: float = 0.1,
        max_weight: float = 100.0
    ) -> Optional[Dict[str, float]]:
        """
        Compute recommended class weights based on label frequencies.

        Args:
            strategy: Weighting strategy:
                     'inverse' - Inverse frequency weighting (recommended)
                     'balanced' - Sklearn-style balanced weighting
                     'aggressive' - More aggressive weights for rare classes
            min_weight: Minimum weight value (for common classes)
            max_weight: Maximum weight value (for rare classes)

        Returns:
            Dictionary mapping label strings to recommended weights,
            or None if frequencies not computed.
            Example: {"Nomenclature": 100.0, "Description": 10.0, "Misc": 1.0}

        Note:
            Requires compute_label_frequencies=True and fit() to have been called.

        Example:
            classifier = SkolClassifierV2(
                spark=spark,
                compute_label_frequencies=True,
                ...
            )
            classifier.fit()
            weights = classifier.get_recommended_class_weights(strategy='inverse')
            print(weights)
            # Output: {"Nomenclature": 100.0, "Description": 10.0, "Misc": 0.1}
        """
        if self._label_frequencies is None:
            if self.verbosity >= 1:
                print("[Classifier] WARNING: Label frequencies not computed. "
                      "Set compute_label_frequencies=True in constructor.")
            return None

        if len(self._label_frequencies) == 0:
            return None

        # Get total count and frequencies
        total = sum(self._label_frequencies.values())
        num_classes = len(self._label_frequencies)

        weights = {}

        if strategy == 'inverse':
            # Inverse frequency: weight = total / (num_classes * frequency)
            # Then normalize to range [min_weight, max_weight]
            raw_weights = {}
            for label, count in self._label_frequencies.items():
                raw_weights[label] = total / (num_classes * count)

            # Normalize to range
            min_raw = min(raw_weights.values())
            max_raw = max(raw_weights.values())
            range_raw = max_raw - min_raw

            if range_raw > 0:
                for label, raw_weight in raw_weights.items():
                    normalized = (raw_weight - min_raw) / range_raw
                    weights[label] = min_weight + normalized * (max_weight - min_weight)
            else:
                # All classes have same frequency
                for label in raw_weights:
                    weights[label] = (min_weight + max_weight) / 2

        elif strategy == 'balanced':
            # Sklearn-style balanced: weight = total / (num_classes * frequency)
            for label, count in self._label_frequencies.items():
                weights[label] = total / (num_classes * count)

        elif strategy == 'aggressive':
            # Aggressive: square the inverse frequency for more extreme weights
            raw_weights = {}
            for label, count in self._label_frequencies.items():
                raw_weights[label] = (total / (num_classes * count)) ** 2

            # Normalize to range
            min_raw = min(raw_weights.values())
            max_raw = max(raw_weights.values())
            range_raw = max_raw - min_raw

            if range_raw > 0:
                for label, raw_weight in raw_weights.items():
                    normalized = (raw_weight - min_raw) / range_raw
                    weights[label] = min_weight + normalized * (max_weight - min_weight)
            else:
                for label in raw_weights:
                    weights[label] = (min_weight + max_weight) / 2

        if self.verbosity >= 1:
            print(f"\n[Classifier] Recommended Class Weights (strategy='{strategy}'):")
            # Sort by weight descending
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for label, weight in sorted_weights:
                freq = self._label_frequencies[label]
                print(f"  {label:<20} {weight:>8.2f} (frequency: {freq})")
            print()

        return weights

    # Private helper methods

    def _load_raw_from_files(self) -> DataFrame:
        """Load raw text from local files."""
        # Delegate to extraction mode - section mode will raise NotImplementedError
        df = self.extraction_mode.load_raw_from_files(
            spark=self.spark,
            file_paths=self.file_paths
        )

        return self.load_raw_from_df(df)

    def _load_raw_from_couchdb(self) -> DataFrame:
        """Load raw text from CouchDB."""
        # For section mode, use the classifier's special section loading logic
        # which handles doc ID discovery and verbosity
        if self.extraction_mode.name == 'section':
            return self._load_sections_from_couchdb()

        # For line and paragraph modes, delegate to extraction mode
        df = self.extraction_mode.load_raw_from_couchdb(
            spark=self.spark,
            couchdb_url=self.couchdb_url,
            database=self.couchdb_database,
            username=self.couchdb_username,
            password=self.couchdb_password,
            pattern=self.couchdb_pattern
        )

        return self.load_raw_from_df(df)

    def load_raw_from_df(self, df: DataFrame) -> DataFrame:
        """Load raw text from provided DataFrame."""
        if "doc_id" not in df.columns:
            df = df.withColumn("doc_id", col("filename"))
        if "attachment_name" not in df.columns:
            df = df.withColumn("attachment_name", lit("main.txt"))

        # Split into lines if line_level mode
        if self.line_level:
            df = df.withColumn("value", explode(split(col("value"), "\\n")))
            df = df.filter(trim(col("value")) != "")

            # Add line numbers
            window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy(lit(1))
            df = df.withColumn("line_number", row_number().over(window_spec) - 1)
        return df

    def _load_annotated_data(self) -> DataFrame:
        """
        Load annotated data for training.

        If compute_label_frequencies is True, computes and stores label frequencies
        in self._label_frequencies as a dict mapping label strings to counts.

        Returns:
            DataFrame with annotated data
        """
        if self.input_source == 'files':
            df = self._load_annotated_from_files()
        elif self.input_source == 'couchdb':
            df = self._load_annotated_from_couchdb()
        else:
            raise ValueError(
                f"Cannot load annotated data from input_source='{self.input_source}'"
            )

        # Compute label frequencies if requested
        if self.compute_label_frequencies:
            self._compute_label_frequencies(df)

        return df

    def _compute_label_frequencies(self, df: DataFrame) -> None:
        """
        Compute and store label frequencies from annotated data.

        Args:
            df: DataFrame with 'label' column containing label strings
        """
        if 'label' not in df.columns:
            if self.verbosity >= 1:
                print("[Classifier] WARNING: Cannot compute label frequencies - 'label' column not found")
            return

        # Count labels
        label_counts = df.groupBy('label').count().collect()

        # Store as dictionary
        self._label_frequencies = {row['label']: row['count'] for row in label_counts}

        if self.verbosity >= 1:
            print(f"\n[Classifier] Label Frequencies:")
            # Sort by count descending
            sorted_labels = sorted(self._label_frequencies.items(), key=lambda x: x[1], reverse=True)
            total = sum(count for _, count in sorted_labels)
            for label, count in sorted_labels:
                percentage = (count / total) * 100
                print(f"  {label:<20} {count:>8} ({percentage:>5.1f}%)")
            print(f"  {'Total':<20} {total:>8} (100.0%)")
            print()

    def _load_annotated_from_files(self) -> DataFrame:
        """Load annotated data from files."""
        return self.extraction_mode.load_annotated_from_files(
            spark=self.spark,
            file_paths=self.file_paths,
            collapse_labels=self.collapse_labels
        )

    def _load_annotated_from_couchdb(self) -> DataFrame:
        """Load annotated data from CouchDB.

        Uses couchdb_training_database if specified, otherwise uses couchdb_database.

        IMPORTANT: Training data ALWAYS comes from .txt.ann files (YEDDA annotated text),
        regardless of extraction_mode. PDFs are only used for prediction (unannotated data).
        """
        # Use training database if specified, otherwise use main database
        database = self.couchdb_training_database or self.couchdb_database

        if self.verbosity >= 2:
            if self.couchdb_training_database:
                print(f"[Classifier] Loading training data from database: {database}")
            else:
                print(f"[Classifier] Loading training data from main database: {database}")

        # For ALL extraction modes, load from .txt.ann files (annotated text)
        # PDFs are only used for prediction, not training
        conn = CouchDBConnection(
            self.couchdb_url,
            database,
            self.couchdb_username,
            self.couchdb_password
        )

        # Look for .ann files for training
        pattern = self.couchdb_pattern
        if not pattern.endswith('.ann'):
            pattern = pattern.replace('.txt', '.txt.ann')

        df = conn.load_distributed(self.spark, pattern)

        # Parse annotations using unified AnnotatedTextParser
        from .preprocessing import AnnotatedTextParser

        parser = AnnotatedTextParser(
            extraction_mode=self.extraction_mode.name,
            collapse_labels=self.collapse_labels
        )
        return parser.parse(df)

    def _load_sections_from_files(self) -> DataFrame:
        """
        Load sections from PDF files.

        Note: For 'section' extraction mode, PDFs must be stored in CouchDB.
        The file_paths parameter is not supported for section mode.
        Use input_source='couchdb' instead.
        """
        raise NotImplementedError(
            "Section-based tokenization requires PDFs to be stored in CouchDB. "
            "Please use input_source='couchdb' with tokenizer='section'. "
            "The PDFSectionExtractor only supports loading from CouchDB attachments."
        )

    def _discover_pdf_documents(self, database: Optional[str] = None) -> List[str]:
        """
        Discover all documents with PDF attachments in CouchDB.

        Returns:
            List of document IDs that have PDF attachments
        """
        import couchdb

        if database is None:
            database = self.couchdb_database

        if self.verbosity >= 2:
            print(f"[Classifier] Discovering PDF documents in database: {database}")

        # Connect to CouchDB
        if self.couchdb_username and self.couchdb_password:
            server = couchdb.Server(self.couchdb_url)
            server.resource.credentials = (self.couchdb_username, self.couchdb_password)
        else:
            server = couchdb.Server(self.couchdb_url)

        db = server[database]

        # Query all documents
        doc_ids_with_pdfs = []

        if self.verbosity >= 2:
            print(f"[Classifier] Querying database for documents with PDF attachments...")

        # Iterate through all documents
        for doc_id in db:
            try:
                doc = db[doc_id]

                # Check if document has attachments
                if '_attachments' in doc:
                    # Check if any attachment is a PDF
                    for att_name, att_info in doc['_attachments'].items():
                        if att_name.lower().endswith('.pdf') or \
                           att_info.get('content_type', '').startswith('application/pdf'):
                            doc_ids_with_pdfs.append(doc_id)
                            if self.verbosity >= 3:
                                print(f"[Classifier]   Found PDF in document: {doc_id} ({att_name})")
                            break  # Found a PDF, no need to check other attachments

            except Exception as e:
                if self.verbosity >= 2:
                    print(f"[Classifier] Warning: Could not check document {doc_id}: {e}")
                continue

        return doc_ids_with_pdfs

    def _load_sections_from_couchdb(self, database: Optional[str] = None) -> DataFrame:
        """Load sections from PDF documents in CouchDB using PDFSectionExtractor.

        Args:
            database: Optional database name. If not provided, uses self.couchdb_database.
        """
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        # Add parent directory to path to import pdf_section_extractor
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from pdf_section_extractor import PDFSectionExtractor

        # Use provided database or fall back to instance variable
        db = database or self.couchdb_database
        if not db:
            raise ValueError("Database name must be provided either as parameter or via couchdb_database")

        # Get document IDs (either provided or auto-discover)
        if self.couchdb_doc_ids:
            doc_ids = self.couchdb_doc_ids
            if self.verbosity >= 1:
                print(f"[Classifier] Loading sections from {len(doc_ids)} specified PDF documents")
        else:
            # Auto-discover documents with PDF attachments
            if self.verbosity >= 1:
                print(f"[Classifier] Auto-discovering PDF documents in database: {db}")

            doc_ids = self._discover_pdf_documents(db)

            if not doc_ids:
                raise ValueError(
                    f"No PDF documents found in database '{db}'. "
                    "Please ensure documents have PDF attachments or provide couchdb_doc_ids explicitly."
                )

            if self.verbosity >= 1:
                print(f"[Classifier] Found {len(doc_ids)} documents with PDF attachments")

        if self.verbosity >= 1:
            print(f"[Classifier] Database: {db}")

        # Create extractor
        extractor = PDFSectionExtractor(
            couchdb_url=self.couchdb_url,
            username=self.couchdb_username,
            password=self.couchdb_password,
            spark=self.spark,
            verbosity=max(0, self.verbosity - 1),  # Reduce verbosity for extractor
            read_text=self.read_text,
            save_text=self.save_text
        )

        # Extract sections from multiple documents
        sections_df = extractor.extract_from_multiple_documents(
            database=db,
            doc_ids=doc_ids
        )

        # Apply section filter if specified
        if self.section_filter:
            if self.verbosity >= 1:
                print(f"[Classifier] Filtering sections: {self.section_filter}")

            original_count = sections_df.count()
            sections_df = sections_df.filter(
                sections_df.section_name.isin(self.section_filter)
            )
            filtered_count = sections_df.count()

            if self.verbosity >= 1:
                print(f"[Classifier] Kept {filtered_count}/{original_count} sections after filtering")

        if self.verbosity >= 1:
            total_sections = sections_df.count()
            print(f"[Classifier] Total sections: {total_sections}")

            # Show section name distribution
            if self.verbosity >= 2 and "section_name" in sections_df.columns:
                print(f"[Classifier] Section distribution:")
                section_counts = sections_df.filter(sections_df.section_name.isNotNull()) \
                    .groupBy("section_name").count() \
                    .orderBy("count", ascending=False)
                section_counts.show(10, truncate=False)

        return sections_df

    def _decode_predictions(self, predictions_df: DataFrame) -> DataFrame:
        """Convert label indices back to label strings."""
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        # Standardize column names for consistency
        # RNN models use 'filename' and 'pos', others use 'doc_id'
        if "filename" in predictions_df.columns and "doc_id" not in predictions_df.columns:
            predictions_df = predictions_df.withColumnRenamed("filename", "doc_id")
        if "pos" in predictions_df.columns and "line_number" not in predictions_df.columns:
            predictions_df = predictions_df.withColumnRenamed("pos", "line_number")

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
            # Return just labels (preserve attachment_name if present)
            cols = ["doc_id"]
            if "attachment_name" in predictions_df.columns:
                cols.append("attachment_name")
            cols.append("predicted_label")
            return predictions_df.select(*cols)
        elif self.output_format == 'probs':
            # Return probabilities (preserve attachment_name if present)
            cols = ["doc_id"]
            if "attachment_name" in predictions_df.columns:
                cols.append("attachment_name")
            cols.extend(["predicted_label", "probability"])
            return predictions_df.select(*cols)
        else:
            return predictions_df

    def _reinsert_page_markers(
        self,
        predictions_df: DataFrame,
        page_markers_df: DataFrame
    ) -> DataFrame:
        """
        Reinsert PDF page markers into predictions DataFrame.

        Page markers are preserved but not classified. They are added back
        with a special 'PAGE_MARKER' pseudo-label to maintain proper ordering.

        Args:
            predictions_df: DataFrame with predictions
            page_markers_df: DataFrame with page markers

        Returns:
            Combined DataFrame with page markers reinserted
        """
        from pyspark.sql.functions import lit

        # If output_format is 'annotated', page markers need annotated_value column
        if "annotated_value" in predictions_df.columns:
            # Page markers don't get classified - preserve their raw value
            page_markers_with_label = page_markers_df.withColumn(
                "annotated_value", col("value")  # Keep raw marker text
            )
        else:
            # For non-annotated format, just add the value
            page_markers_with_label = page_markers_df

        # Add columns that predictions_df has but page markers don't
        for column in predictions_df.columns:
            if column not in page_markers_with_label.columns:
                # Use None for most columns, False for boolean columns
                if column == "is_page_marker":
                    page_markers_with_label = page_markers_with_label.withColumn(
                        column, lit(True)
                    )
                else:
                    page_markers_with_label = page_markers_with_label.withColumn(
                        column, lit(None)
                    )

        # Union predictions with page markers
        combined_df = predictions_df.union(page_markers_with_label)

        # Sort by line_number to restore original order (if line_number exists)
        if "line_number" in combined_df.columns:
            combined_df = combined_df.orderBy("doc_id", "attachment_name", "line_number")

        return combined_df

    def _format_as_annotated(self, predictions_df: DataFrame) -> DataFrame:
        """
        Format predictions as YEDDA-style annotated blocks.

        Note: This does NOT apply coalescing. Coalescing is only applied
        during save_annotated() to preserve line-level data for inspection.
        """
        formatter = YeddaFormatter(
            coalesce_labels=False,  # Never coalesce in predict()
            line_level=self.line_level
        )
        return formatter.format(predictions_df)

    def _save_to_files(self, predictions: DataFrame) -> None:
        """Save predictions to local files."""
        writer = FileOutputWriter(self.output_path)
        writer.write(predictions)

    def _save_to_couchdb(self, predictions: DataFrame) -> None:
        """Save predictions to CouchDB."""
        from skol_classifier.output_formatters import CouchDBOutputWriter

        writer = CouchDBOutputWriter(
            couchdb_url=self.couchdb_url,
            database=self.couchdb_database,
            username=self.couchdb_username,
            password=self.couchdb_password
        )

        writer.save_annotated(
            predictions,
            suffix=self.output_couchdb_suffix,
            coalesce_labels=self.coalesce_labels,
            line_level=self.line_level
        )

    def _format_as_strings(self, predictions: DataFrame) -> List[str]:
        """
        Format predictions as a list of annotated strings.

        Each string in the list represents one document with all its
        annotations joined by newlines.

        Args:
            predictions: DataFrame with predictions

        Returns:
            List of annotated strings (one per document)
        """
        from pyspark.sql.functions import expr, collect_list

        # Format predictions if not already formatted
        if "annotated_value" not in predictions.columns:
            formatter = YeddaFormatter(
                coalesce_labels=self.coalesce_labels,
                line_level=self.line_level
            )
            predictions = formatter.format(predictions)

        # Apply coalescing if requested and not already done
        if self.coalesce_labels and self.line_level:
            if "coalesced_annotations" not in predictions.columns:
                predictions = YeddaFormatter.coalesce_consecutive_labels(
                    predictions, line_level=True
                )

        # Determine grouping column
        if "filename" in predictions.columns:
            groupby_col = "filename"
        elif "doc_id" in predictions.columns:
            groupby_col = "doc_id"
        else:
            # If no grouping column, treat as single document
            groupby_col = None

        # If coalesced, we have coalesced_annotations column
        if self.coalesce_labels and self.line_level:
            if groupby_col:
                # Collect coalesced annotations per document
                aggregated = predictions.groupBy(groupby_col).agg(
                    expr("array_join(coalesced_annotations, '\n')").alias(
                        "final_annotated"
                    )
                )
            else:
                # Single document
                aggregated = predictions.select(
                    expr("array_join(coalesced_annotations, '\n')").alias(
                        "final_annotated"
                    )
                )
        else:
            # Not coalesced - aggregate annotated_value
            if groupby_col:
                # Check if we have line_number for ordering
                if "line_number" in predictions.columns:
                    # Order by line_number within each document
                    aggregated = (
                        predictions.groupBy(groupby_col)
                        .agg(
                            expr("sort_array(collect_list(struct(line_number, "
                                 "annotated_value))) AS sorted_list")
                        )
                        .withColumn(
                            "annotated_value_ordered",
                            expr("transform(sorted_list, x -> x.annotated_value)")
                        )
                        .withColumn(
                            "final_annotated",
                            expr("array_join(annotated_value_ordered, '\n')")
                        )
                        .select("final_annotated")
                    )
                else:
                    # No line_number, just collect
                    aggregated = (
                        predictions.groupBy(groupby_col)
                        .agg(collect_list("annotated_value").alias("annotations"))
                        .withColumn(
                            "final_annotated",
                            expr("array_join(annotations, '\n')")
                        )
                        .select("final_annotated")
                    )
            else:
                # Single document without grouping
                if "line_number" in predictions.columns:
                    aggregated = predictions.orderBy("line_number").agg(
                        collect_list("annotated_value").alias("annotations")
                    ).withColumn(
                        "final_annotated",
                        expr("array_join(annotations, '\n')")
                    ).select("final_annotated")
                else:
                    aggregated = predictions.agg(
                        collect_list("annotated_value").alias("annotations")
                    ).withColumn(
                        "final_annotated",
                        expr("array_join(annotations, '\n')")
                    ).select("final_annotated")

        # Collect to list
        result = aggregated.select("final_annotated").collect()
        return [row["final_annotated"] for row in result]

    def _save_model_to_disk(self) -> None:
        """Save model to disk using PySpark's native save."""
        import json
        import shutil

        if self._model is None or self._feature_pipeline is None:
            raise ValueError("No model to save. Train a model first.")

        model_path = Path(self.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a directory for the model
        model_dir = model_path.parent / model_path.stem
        model_dir.mkdir(exist_ok=True)

        # Save feature pipeline using PySpark's save
        pipeline_path = model_dir / "feature_pipeline"
        if pipeline_path.exists():
            shutil.rmtree(pipeline_path)
        self._feature_pipeline.save(str(pipeline_path))

        # Save classifier model using PySpark's save
        classifier_model = self._model.get_model()
        if classifier_model is None:
            raise ValueError("Classifier model not trained")
        classifier_path = model_dir / "classifier_model.h5"
        if classifier_path.exists():
            shutil.rmtree(classifier_path)
        classifier_model.save(str(classifier_path))

        # Save metadata as JSON
        metadata = {
            'label_mapping': self._label_mapping,
            'config': {
                'tokenizer': self.extraction_mode.name,
                'use_suffixes': self.use_suffixes,
                'min_doc_freq': self.min_doc_freq,
                'model_type': self.model_type,
                'model_params': self.model_params
            },
            'version': '2.1'
        }
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_model_from_disk(self) -> None:
        """Load model from disk using PySpark's native load."""
        import json
        from pyspark.ml import PipelineModel

        model_path = Path(self.model_path)
        model_dir = model_path.parent / model_path.stem

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Load feature pipeline
        pipeline_path = model_dir / "feature_pipeline"
        self._feature_pipeline = PipelineModel.load(str(pipeline_path))

        # Load metadata first to know model type
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self._label_mapping = metadata['label_mapping']
        self._reverse_label_mapping = {v: k for k, v in self._label_mapping.items()}
        model_type = metadata['config']['model_type']

        # Load classifier model (different approach for RNN vs traditional ML)
        classifier_path = model_dir / "classifier_model.h5"
        if model_type == 'rnn':
            # For RNN models, load the Keras .h5 file directly
            from tensorflow import keras
            import tensorflow as tf

            # Define dummy loss functions for deserialization
            def weighted_categorical_crossentropy(y_true, y_pred):
                """Dummy loss function for model deserialization. Not used for prediction."""
                return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

            def mean_f1_loss(y_true, y_pred):
                """Dummy loss function for model deserialization. Not used for prediction."""
                return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

            custom_objects = {
                'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
                'mean_f1_loss': mean_f1_loss
            }
            classifier_model = keras.models.load_model(
                str(classifier_path),
                custom_objects=custom_objects,
                compile=False
            )
        else:
            # For traditional ML models, load as PipelineModel
            classifier_model = PipelineModel.load(str(classifier_path))

        # Recreate the SkolModel wrapper using factory
        features_col = self._feature_extractor.get_features_col() if self._feature_extractor else "combined_idf"

        # Merge saved model params with any new params provided in constructor
        # New params override saved params for runtime-tunable parameters
        saved_params = metadata['config'].get('model_params', {})
        merged_params = saved_params.copy()

        # Override runtime-tunable parameters if provided
        if self.model_params:
            # These parameters can be changed without retraining
            runtime_tunable = {
                'prediction_batch_size',
                'prediction_stride',
                'num_workers',
                'verbosity',
                'batch_size'  # Training batch size, can be changed for future fine-tuning
            }
            for param, value in self.model_params.items():
                if param in runtime_tunable:
                    merged_params[param] = value
                    if self.verbosity >= 2:
                        print(f"[Load Model] Overriding {param}: {saved_params.get(param)} -> {value}")

        # Get labels from label mapping
        labels_list = list(self._label_mapping.keys()) if self._label_mapping else None

        self._model = create_model(
            model_type=model_type,
            features_col=features_col,
            label_col="label_indexed",
            labels=labels_list,  # Pass labels for class weight support
            **merged_params
        )
        self._model.set_model(classifier_model)
        if labels_list:
            self._model.set_labels(labels_list)

    def _save_model_to_redis(self) -> None:
        """Save model to Redis using tar archive."""
        import json
        import tempfile
        import shutil
        import tarfile
        import io

        if self._model is None or self._feature_pipeline is None:
            raise ValueError("No model to save. Train a model first.")

        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="skol_model_v2_")
            temp_path = Path(temp_dir)

            # Save feature pipeline
            pipeline_path = temp_path / "feature_pipeline"
            self._feature_pipeline.save(str(pipeline_path))

            # Save classifier model
            classifier_model = self._model.get_model()
            if classifier_model is None:
                raise ValueError("Classifier model not trained")
            classifier_path = temp_path / "classifier_model.h5"
            classifier_model.save(str(classifier_path))

            # Save metadata
            # For RNN models, save the actual model parameters (not the original params)
            if self.model_type == 'rnn':
                actual_model_params = {
                    'input_size': self._model.input_size,
                    'hidden_size': self._model.hidden_size,
                    'num_layers': self._model.num_layers,
                    'num_classes': self._model.num_classes,
                    'dropout': self._model.dropout,
                    'window_size': self._model.window_size,
                    'batch_size': self._model.batch_size,
                    'epochs': self._model.epochs,
                    'num_workers': self._model.num_workers,
                    'verbosity': self._model.verbosity,
                }
                if hasattr(self._model, 'prediction_stride'):
                    actual_model_params['prediction_stride'] = self._model.prediction_stride
                if hasattr(self._model, 'prediction_batch_size'):
                    actual_model_params['prediction_batch_size'] = self._model.prediction_batch_size
                if hasattr(self._model, 'name'):
                    actual_model_params['name'] = self._model.name
            else:
                actual_model_params = self.model_params

            metadata = {
                'label_mapping': self._label_mapping,
                'config': {
                    'tokenizer': self.extraction_mode.name,
                    'use_suffixes': self.use_suffixes,
                    'min_doc_freq': self.min_doc_freq,
                    'model_type': self.model_type,
                    'model_params': actual_model_params
                },
                'version': '2.1'
            }
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create tar archive
            archive_buffer = io.BytesIO()
            with tarfile.open(fileobj=archive_buffer, mode='w:gz') as tar:
                tar.add(temp_path, arcname='.')

            # Save to Redis
            archive_data = archive_buffer.getvalue()
            self.redis_client.set(self.redis_key, archive_data)
            if self.redis_expire is not None:
                self.redis_client.expire(self.redis_key, self.redis_expire)

        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def _load_model_from_redis(self) -> None:
        """Load model from Redis tar archive."""
        import json
        import tempfile
        import shutil
        import tarfile
        import io
        from pyspark.ml import PipelineModel

        serialized = self.redis_client.get(self.redis_key)
        if not serialized:
            raise ValueError(f"No model found in Redis with key: {self.redis_key}")

        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="skol_model_load_v2_")
            temp_path = Path(temp_dir)

            # Extract tar archive
            archive_buffer = io.BytesIO(serialized)
            with tarfile.open(fileobj=archive_buffer, mode='r:gz') as tar:
                tar.extractall(temp_path)

            # Load metadata first to know model type
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self._label_mapping = metadata['label_mapping']
            self._reverse_label_mapping = {v: k for k, v in self._label_mapping.items()}
            model_type = metadata['config']['model_type']

            # Load feature pipeline
            pipeline_path = temp_path / "feature_pipeline"
            self._feature_pipeline = PipelineModel.load(str(pipeline_path))

            # Load classifier model (different approach for RNN vs traditional ML)
            classifier_path = temp_path / "classifier_model.h5"

            if model_type == 'rnn':
                # For RNN models, load the Keras .h5 file directly
                # Load without compiling to avoid issues with custom loss functions
                from tensorflow import keras
                import tensorflow as tf

                # Define dummy loss functions for deserialization
                def weighted_categorical_crossentropy(y_true, y_pred):
                    """Dummy loss function for model deserialization. Not used for prediction."""
                    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

                def mean_f1_loss(y_true, y_pred):
                    """Dummy loss function for model deserialization. Not used for prediction."""
                    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

                custom_objects = {
                    'weighted_categorical_crossentropy': weighted_categorical_crossentropy,
                    'mean_f1_loss': mean_f1_loss
                }
                keras_model = keras.models.load_model(
                    str(classifier_path),
                    custom_objects=custom_objects,
                    compile=False
                )
                classifier_model = keras_model  # This is the Keras model itself
            else:
                # For traditional ML models, load as PipelineModel
                classifier_model = PipelineModel.load(str(classifier_path))

            # Recreate the SkolModel wrapper using factory
            features_col = self._feature_extractor.get_features_col() if self._feature_extractor else "combined_idf"

            # Merge saved model params with any new params provided in constructor
            # New params override saved params for runtime-tunable parameters
            saved_params = metadata['config'].get('model_params', {})
            merged_params = saved_params.copy()

            # Override runtime-tunable parameters if provided
            if self.model_params:
                # These parameters can be changed without retraining
                runtime_tunable = {
                    'prediction_batch_size',
                    'prediction_stride',
                    'num_workers',
                    'verbosity',
                    'batch_size'  # Training batch size, can be changed for future fine-tuning
                }
                for param, value in self.model_params.items():
                    if param in runtime_tunable:
                        merged_params[param] = value
                        if self.verbosity >= 2:
                            print(f"[Load Model] Overriding {param}: {saved_params.get(param)} -> {value}")

            # Get labels from label mapping
            labels_list = list(self._label_mapping.keys()) if self._label_mapping else None

            self._model = create_model(
                model_type=model_type,
                features_col=features_col,
                label_col="label_indexed",
                labels=labels_list,  # Pass labels for class weight support
                **merged_params
            )
            self._model.set_model(classifier_model)
            if labels_list:
                self._model.set_labels(labels_list)

        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
