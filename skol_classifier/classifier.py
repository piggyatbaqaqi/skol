"""
Main classifier module for SKOL text classification
"""

import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    input_file_name, collect_list, regexp_extract, col, udf,
    explode, split, trim, row_number, min, expr, concat, lit
)
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Tokenizer, CountVectorizer, IDF, StringIndexer, VectorAssembler, IndexToString
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import IndexToString
from pyspark.sql.types import (
    StructType as LineStructType,
    StructField as LineStructField,
    IntegerType as LineIntegerType,
    ArrayType as LineArrayType,
    ArrayType, StringType, StructType, StructField, IntegerType
)

import sparknlp  # typing: ignore[import-untyped]

from .couchdb_io import CouchDBConnection
from .preprocessing import SuffixTransformer, ParagraphExtractor
from .utils import calculate_stats


class SkolClassifier:
    """
    Text classifier for taxonomic literature.

    Supports multiple classification models (Logistic Regression, Random Forest)
    and feature types (word TF-IDF, suffix TF-IDF, combined).
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        redis_client: Optional[Any] = None,
        redis_key: str = "skol_classifier_model",
        auto_load: bool = True,
        couchdb_url: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize the SKOL classifier.

        Args:
            spark: SparkSession (creates one if not provided)
            redis_client: Redis client connection (optional, for model persistence)
            redis_key: Key name to use in Redis for storing the model
            auto_load: If True and redis_client is provided, automatically load
                      model from Redis if the key exists (default: True)
            couchdb_url: CouchDB server URL (e.g., "http://localhost:5984")
            database: CouchDB database name
            username: CouchDB username (optional)
            password: CouchDB password (optional)
        """
        if spark is None:
            self.spark = sparknlp.start()
        else:
            self.spark = spark

        self.pipeline_model: Optional[PipelineModel] = None
        self.classifier_model: Optional[PipelineModel] = None
        self.labels: Optional[List[str]] = None

        # Redis configuration
        self.redis_client = redis_client
        self.redis_key = redis_key

        # CouchDB configuration
        self.couchdb_url = couchdb_url
        self.database = database
        self.username = username
        self.password = password

        # Automatically load from Redis if key exists
        if auto_load and redis_client is not None:
            try:
                # Check if key exists in Redis
                if redis_client.exists(redis_key):
                    self.load_from_redis()
            except Exception:
                # Silently fail if loading doesn't work
                # Model will remain uninitialized
                pass

    def load_annotated_data(
        self,
        file_paths: List[str],
        collapse_labels: bool = True,
        line_level: bool = False
    ) -> DataFrame:
        """
        Load and preprocess annotated data.

        Args:
            file_paths: List of paths to annotated files
            collapse_labels: Whether to collapse labels to 3 main categories
            line_level: If True, extract individual lines instead of paragraphs

        Returns:
            Preprocessed DataFrame with paragraphs/lines and labels
        """
        # Read annotated files
        ann_df = self.spark.read.text(file_paths).withColumn(
            "filename", input_file_name()
        )

        if line_level:
            # Line-level extraction: parse each line from YEDA blocks

            def extract_yeda_lines(lines: List[str]) -> List[Tuple[str, str, int]]:
                """Extract individual lines from YEDA annotation blocks."""
                import re
                results = []
                pattern = r'\[@\s*(.*?)\s*#([^\*]+)\*\]'

                for match in re.finditer(pattern, '\n'.join(lines), re.DOTALL):
                    content = match.group(1)
                    label = match.group(2).strip()

                    # Split content into lines
                    content_lines = content.split('\n')
                    for line_num, line in enumerate(content_lines):
                        if line or line_num < len(content_lines) - 1:
                            results.append((label, line, line_num))

                return results

            # UDF to extract lines
            extract_udf = udf(
                extract_yeda_lines,
                LineArrayType(LineStructType([
                    LineStructField("label", StringType(), False),
                    LineStructField("value", StringType(), False),
                    LineStructField("line_number", LineIntegerType(), False)
                ]))
            )

            # Extract lines
            grouped_df = (
                ann_df.groupBy("filename")
                .agg(collect_list("value").alias("lines"))
                .withColumn("line_data", explode(extract_udf(col("lines"))))
                .select(
                    "filename",
                    col("line_data.label").alias("label"),
                    col("line_data.value").alias("value"),
                    col("line_data.line_number")
                )
            )
        else:
            # Paragraph-level extraction (original behavior)
            extract_udf = udf(
                ParagraphExtractor.extract_annotated_paragraphs,
                ArrayType(StringType())
            )

            # Group and extract paragraphs
            grouped_df = (
                ann_df.groupBy("filename")
                .agg(collect_list("value").alias("lines"))
                .withColumn("value", explode(extract_udf(col("lines"))))
                .drop("lines")
            )

            # Extract labels
            label_pattern = r"#(\S+?)(?:\*)?]"
            lead_pattern = r"^\[@"
            trail_pattern = label_pattern + r"$"
            clean_pattern = lead_pattern + r"(.*)" + trail_pattern

            grouped_df = grouped_df.withColumn(
                "label", regexp_extract(col("value"), label_pattern, 1)
            ).withColumn(
                "value", regexp_extract(col("value"), clean_pattern, 1)
            )

        # Optionally collapse labels
        if collapse_labels:
            collapse_udf = udf(
                ParagraphExtractor.collapse_labels,
                StringType()
            )
            grouped_df = grouped_df.withColumn(
                "label", collapse_udf(col("label"))
            )

        return grouped_df

    def load_raw_data(self, file_paths: List[str]) -> DataFrame:
        """
        Load and preprocess raw text data.

        Args:
            file_paths: List of paths to raw text files

        Returns:
            Preprocessed DataFrame with paragraphs
        """
        # Read raw files
        df = self.spark.read.text(file_paths).withColumn(
            "filename", input_file_name()
        )

        # Define UDF for heuristic paragraph extraction
        heuristic_udf = udf(
            ParagraphExtractor.extract_heuristic_paragraphs,
            ArrayType(StringType())
        )

        # Window specification for ordering
        window_spec = Window.partitionBy("filename").orderBy("start_line_id")

        return (
            df.groupBy("filename") # pyright: ignore[reportUnknownMemberType]
            .agg(
                collect_list("value").alias("lines"),
                min("filename").alias("start_line_id"),
            )
            .withColumn("value", explode(heuristic_udf(col("lines"))))
            .drop("lines")
            .filter(trim(col("value")) != "")
            .withColumn("row_number", row_number().over(window_spec))
        ) # pyright: ignore[reportUnknownVariableType, reportUnknownVariableType]

    def build_feature_pipeline(
        self,
        use_suffixes: bool = True,
        min_doc_freq: int = 10
    ) -> Pipeline:
        """
        Build the feature extraction pipeline.

        Args:
            use_suffixes: Whether to include suffix features
            min_doc_freq: Minimum document frequency for IDF

        Returns:
            Pipeline object
        """
        # Tokenization
        tokenizer = Tokenizer(inputCol="value", outputCol="words")

        # Word TF-IDF
        word_count_vectorizer = CountVectorizer(
            inputCol="words", outputCol="word_tf"
        )
        word_idf = IDF(
            inputCol="word_tf", outputCol="word_idf", minDocFreq=min_doc_freq
        )

        stages = [tokenizer, word_count_vectorizer, word_idf]

        # Suffix TF-IDF (optional)
        if use_suffixes:
            suffixer = SuffixTransformer(inputCol="words", outputCol="suffixes")
            suffix_count_vectorizer = CountVectorizer(
                inputCol="suffixes", outputCol="suffix_tf"
            )
            suffix_idf = IDF(
                inputCol="suffix_tf", outputCol="suffix_idf", minDocFreq=min_doc_freq
            )
            idf_combiner = VectorAssembler(
                inputCols=["word_idf", "suffix_idf"], outputCol="combined_idf"
            )
            stages.extend([
                suffixer, suffix_count_vectorizer, suffix_idf, idf_combiner
            ])

        # Label indexing
        indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
        stages.append(indexer)

        return Pipeline(stages=stages)

    def fit_features(
        self,
        data: DataFrame,
        use_suffixes: bool = True,
        min_doc_freq: int = 10
    ) -> DataFrame:
        """
        Fit the feature extraction pipeline and transform data.

        Args:
            data: Input DataFrame with 'value' and 'label' columns
            use_suffixes: Whether to include suffix features
            min_doc_freq: Minimum document frequency for IDF

        Returns:
            Transformed DataFrame with features
        """
        pipeline = self.build_feature_pipeline(use_suffixes, min_doc_freq)
        self.pipeline_model = pipeline.fit(data)
        self.labels = self.pipeline_model.stages[-1].labels
        return self.pipeline_model.transform(data)

    def train_classifier(
        self,
        train_data: DataFrame,
        model_type: str = "logistic",
        features_col: str = "combined_idf",
        **model_params
    ) -> PipelineModel:
        """
        Train a classification model.

        Args:
            train_data: Training DataFrame with features
            model_type: Type of classifier ('logistic' or 'random_forest')
            features_col: Name of features column
            **model_params: Additional model parameters

        Returns:
            Fitted classifier pipeline model
        """
        if model_type == "logistic":
            classifier = LogisticRegression(
                family="multinomial",
                featuresCol=features_col,
                labelCol="label_indexed",
                maxIter=model_params.get("maxIter", 10),
                regParam=model_params.get("regParam", 0.01)
            )
        elif model_type == "random_forest":
            classifier = RandomForestClassifier(
                featuresCol=features_col,
                labelCol="label_indexed",
                numTrees=model_params.get("numTrees", 100),
                seed=model_params.get("seed", 42)
            )
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                "Choose 'logistic' or 'random_forest'."
            )

        pipeline = Pipeline(stages=[classifier])
        self.classifier_model = pipeline.fit(train_data)
        return self.classifier_model

    def predict(self, data: DataFrame) -> DataFrame:
        """
        Make predictions on data.

        Args:
            data: DataFrame with features

        Returns:
            DataFrame with predictions
        """
        if self.classifier_model is None:
            raise ValueError("No classifier model found. Train a model first.")
        return self.classifier_model.transform(data)

    def evaluate(
        self,
        predictions: DataFrame,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate predictions.

        Args:
            predictions: DataFrame with predictions
            verbose: Whether to print statistics

        Returns:
            Dictionary with evaluation metrics
        """
        return calculate_stats(predictions, verbose=verbose)

    def predict_raw_text(
        self,
        file_paths: List[str],
        output_format: str = "annotated"
    ) -> DataFrame:
        """
        Process and predict labels for raw text files.

        Args:
            file_paths: List of paths to raw text files
            output_format: Output format ('annotated' or 'simple')

        Returns:
            DataFrame with predictions
        """
        if self.pipeline_model is None or self.classifier_model is None:
            raise ValueError(
                "Models not trained. Call fit_features() and train_classifier() first."
            )

        # Load and preprocess raw data
        raw_df = self.load_raw_data(file_paths)

        # Extract features
        features = self.pipeline_model.transform(raw_df)

        # Predict
        predictions = self.classifier_model.transform(features)

        # Convert label indices to strings
        converter = IndexToString(
            inputCol="prediction",
            outputCol="predicted_label",
            labels=self.labels
        )
        labeled_predictions = converter.transform(predictions)

        # Format output
        if output_format == "annotated":
            labeled_predictions = labeled_predictions.withColumn(
                "annotated_value",
                concat(
                    lit("[@ "),
                    col("value"),
                    lit("#"),
                    col("predicted_label"),
                    lit("*]")
                )
            )

        return labeled_predictions

    def save_annotated_output(
        self,
        predictions: DataFrame,
        output_path: str
    ) -> None:
        """
        Save annotated predictions to disk.

        Args:
            predictions: DataFrame with annotated predictions
            output_path: Directory to save output files
        """
        # Aggregate paragraphs by file
        aggregated_df = (
            predictions.groupBy("filename")
            .agg(
                expr("sort_array(collect_list(struct(row_number, annotated_value))) AS sorted_list")
            )
            .withColumn("annotated_value_ordered", expr("transform(sorted_list, x -> x.annotated_value)"))
            .withColumn("final_aggregated_pg", expr("array_join(annotated_value_ordered, '\n')"))
            .select("filename", "final_aggregated_pg")
        )

        # Write to disk
        aggregated_df.write.partitionBy("filename").mode("overwrite").text(output_path)

    def fit(
        self,
        annotated_file_paths: List[str],
        model_type: str = "logistic",
        use_suffixes: bool = True,
        test_size: float = 0.2,
        line_level: bool = False,
        **model_params
    ) -> Dict[str, Any]:
        """
        Complete training pipeline: load data, extract features, train model, evaluate.

        Args:
            annotated_file_paths: Paths to annotated training files
            model_type: Type of classifier
            use_suffixes: Whether to use suffix features
            test_size: Proportion of data for testing
            line_level: If True, train on individual lines instead of paragraphs
            **model_params: Additional model parameters

        Returns:
            Dictionary with evaluation results
        """
        # Load annotated data
        annotated_df = self.load_annotated_data(
            annotated_file_paths,
            line_level=line_level
        )

        # Extract features
        features = self.fit_features(annotated_df, use_suffixes=use_suffixes)

        # Split data
        train_data, test_data = features.randomSplit(
            [1 - test_size, test_size], seed=42
        )

        # Determine features column
        features_col = "combined_idf" if use_suffixes else "word_idf"

        # Train classifier
        self.train_classifier(
            train_data, model_type=model_type,
            features_col=features_col, **model_params
        )

        # Evaluate
        predictions = self.predict(test_data)
        stats = self.evaluate(predictions)

        return {
            "train_size": train_data.count(),
            "test_size": test_data.count(),
            "model_type": model_type,
            "features_col": features_col,
            "line_level": line_level,
            **stats
        }

    def save_to_redis(self) -> bool:
        """
        Save the trained models to Redis.

        The models are saved to a temporary directory, then packaged and stored in Redis
        as a compressed binary blob along with metadata.

        Uses the Redis client and key configured in the constructor.

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If no models are trained or Redis client is not configured
        """
        if self.pipeline_model is None or self.classifier_model is None:
            raise ValueError(
                "No models to save. Train models using fit() or train_classifier() first."
            )

        if self.redis_client is None:
            raise ValueError(
                "No Redis client configured. Initialize classifier with redis_client."
            )

        temp_dir = None
        try:
            # Create temporary directory for model files
            temp_dir = tempfile.mkdtemp(prefix="skol_model_")
            temp_path = Path(temp_dir)

            # Save pipeline model
            pipeline_path = temp_path / "pipeline_model"
            self.pipeline_model.save(str(pipeline_path))

            # Save classifier model
            classifier_path = temp_path / "classifier_model"
            self.classifier_model.save(str(classifier_path))

            # Save metadata (labels and model info)
            metadata = {
                "labels": self.labels,
                "version": "0.0.1"
            }
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            # Create archive in memory
            import io
            import tarfile

            archive_buffer = io.BytesIO()
            with tarfile.open(fileobj=archive_buffer, mode='w:gz') as tar:
                tar.add(temp_path, arcname='.')

            # Get compressed data
            archive_data = archive_buffer.getvalue()

            # Save to Redis
            self.redis_client.set(self.redis_key, archive_data)

            return True

        except Exception as e:
            print(f"Error saving to Redis: {e}")
            return False

        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def load_from_redis(self) -> bool:
        """
        Load trained models from Redis.

        Uses the Redis client and key configured in the constructor.

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If Redis client is not configured or key doesn't exist
        """
        if self.redis_client is None:
            raise ValueError(
                "No Redis client configured. Initialize classifier with redis_client."
            )

        temp_dir = None
        try:
            # Retrieve from Redis
            archive_data = self.redis_client.get(self.redis_key)
            if archive_data is None:
                raise ValueError(f"No model found in Redis with key: {self.redis_key}")

            # Create temporary directory for extraction
            temp_dir = tempfile.mkdtemp(prefix="skol_model_load_")
            temp_path = Path(temp_dir)

            # Extract archive
            import io
            import tarfile

            archive_buffer = io.BytesIO(archive_data)
            with tarfile.open(fileobj=archive_buffer, mode='r:gz') as tar:
                tar.extractall(temp_path)

            # Load pipeline model
            pipeline_path = temp_path / "pipeline_model"
            self.pipeline_model = PipelineModel.load(str(pipeline_path))

            # Load classifier model
            classifier_path = temp_path / "classifier_model"
            self.classifier_model = PipelineModel.load(str(classifier_path))

            # Load metadata
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.labels = metadata.get("labels")

            return True

        except Exception as e:
            print(f"Error loading from Redis: {e}")
            return False

        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def save_to_disk(self, path: str) -> None:
        """
        Save the trained models to disk.

        Args:
            path: Directory path to save the models

        Raises:
            ValueError: If no models are trained
        """
        if self.pipeline_model is None or self.classifier_model is None:
            raise ValueError(
                "No models to save. Train models using fit() or train_classifier() first."
            )

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save pipeline model
        pipeline_path = save_path / "pipeline_model"
        self.pipeline_model.save(str(pipeline_path))

        # Save classifier model
        classifier_path = save_path / "classifier_model"
        self.classifier_model.save(str(classifier_path))

        # Save metadata
        metadata = {
            "labels": self.labels,
            "version": "0.0.1"
        }
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def load_from_disk(self, path: str) -> None:
        """
        Load trained models from disk.

        Args:
            path: Directory path containing the saved models

        Raises:
            ValueError: If path doesn't exist or models are not found
        """
        load_path = Path(path)
        if not load_path.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Load pipeline model
        pipeline_path = load_path / "pipeline_model"
        if not pipeline_path.exists():
            raise ValueError(f"Pipeline model not found at: {pipeline_path}")
        self.pipeline_model = PipelineModel.load(str(pipeline_path))

        # Load classifier model
        classifier_path = load_path / "classifier_model"
        if not classifier_path.exists():
            raise ValueError(f"Classifier model not found at: {classifier_path}")
        self.classifier_model = PipelineModel.load(str(classifier_path))

        # Load metadata
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.labels = metadata.get("labels")

    def load_from_couchdb(self, pattern: str = "*.txt") -> DataFrame:
        """
        Load raw text from CouchDB attachments using distributed UDFs.

        This method uses Spark UDFs to fetch attachments in parallel across workers,
        rather than loading all data on the driver.

        Uses the CouchDB configuration from the constructor.

        Args:
            pattern: Pattern for attachment names (default: "*.txt")

        Returns:
            DataFrame with columns: doc_id, attachment_name, value

        Raises:
            ValueError: If CouchDB is not configured
        """
        if self.couchdb_url is None or self.database is None:
            raise ValueError(
                "CouchDB not configured. Initialize classifier with couchdb_url and database."
            )

        conn = CouchDBConnection(
            self.couchdb_url, self.database, self.username, self.password
        )
        return conn.load_distributed(self.spark, pattern)


    def predict_from_couchdb(
        self,
        pattern: str = "*.txt",
        output_format: str = "annotated",
        line_level: bool = False
    ) -> DataFrame:
        """
        Load text from CouchDB, predict labels, and return predictions.

        Uses the CouchDB configuration from the constructor.

        Args:
            pattern: Pattern for attachment names
            output_format: Output format ('annotated' or 'simple')
            line_level: If True, process line-by-line instead of by paragraphs

        Returns:
            DataFrame with predictions, including doc_id and attachment_name

        Raises:
            ValueError: If models are not trained or CouchDB is not configured
        """
        if self.pipeline_model is None or self.classifier_model is None:
            raise ValueError(
                "Models not trained. Call fit_features() and train_classifier() first."
            )

        # Load data from CouchDB
        df = self.load_from_couchdb(pattern)

        if line_level:
            # Line-level processing: split content into lines

            # Split the content into individual lines
            lines_df = (
                df.withColumn("value", explode(split(col("value"), "\n")))
                .filter(trim(col("value")) != "")
            )

            # Add line numbers
            window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy(lit(1))
            processed_df = lines_df.withColumn("line_number", row_number().over(window_spec) - 1)

            # Add row number for ordering
            processed_df = processed_df.withColumn("row_number", row_number().over(window_spec))
        else:
            processed_df = self._extracted_from_predict_from_couchdb_48(df)
        # Extract features
        features = self.pipeline_model.transform(processed_df)

        # Predict
        predictions = self.classifier_model.transform(features)

        # Convert label indices to strings

        converter = IndexToString(
            inputCol="prediction",
            outputCol="predicted_label",
            labels=self.labels
        )
        labeled_predictions = converter.transform(predictions)

        # Format output
        if output_format == "annotated":
            labeled_predictions = labeled_predictions.withColumn(
                "annotated_value",
                concat(
                    lit("[@ "),
                    col("value"),
                    lit("#"),
                    col("predicted_label"),
                    lit("*]")
                )
            )

        return labeled_predictions

    # TODO Rename this here and in `predict_from_couchdb`
    def _extracted_from_predict_from_couchdb_48(self, df: DataFrame) -> DataFrame:
        # Paragraph-level processing

        # First, split content into lines
        lines_df = df.withColumn("value", explode(split(col("value"), "\n")))

        heuristic_udf = udf(
            ParagraphExtractor.extract_heuristic_paragraphs,
            ArrayType(StringType())
        )

        # Window specification for ordering
        window_spec = Window.partitionBy("doc_id", "attachment_name").orderBy("start_idx")

        return (
            lines_df.groupBy("doc_id", "attachment_name")
            .agg(
                collect_list("value").alias("lines"),
                min(lit(0)).alias("start_idx"),
            )
            .withColumn("value", explode(heuristic_udf(col("lines"))))
            .drop("lines")
            .filter(trim(col("value")) != "")
            .withColumn("row_number", row_number().over(window_spec))
        )

    def save_to_couchdb(
        self,
        predictions: DataFrame,
        suffix: str = ".ann",
        coalesce_labels: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Save annotated predictions back to CouchDB using distributed UDFs.

        This method uses Spark UDFs to save attachments in parallel across workers,
        distributing the write operations.

        Uses the CouchDB configuration from the constructor.

        Args:
            predictions: DataFrame with predictions
            suffix: Suffix to append to attachment names (default: ".ann")
            coalesce_labels: If True, coalesce consecutive lines with same label
                           into YEDA blocks (for line-level predictions)

        Returns:
            List of results from CouchDB operations

        Raises:
            ValueError: If CouchDB is not configured
        """
        if self.couchdb_url is None or self.database is None:
            raise ValueError(
                "CouchDB not configured. Initialize classifier with couchdb_url and database."
            )

        conn = CouchDBConnection(
            self.couchdb_url, self.database, self.username, self.password
        )

        if coalesce_labels:
            # For line-level predictions with coalescence
            # Create struct with line and label, collect by document
            aggregated_df = (
                predictions
                .select(
                    "doc_id",
                    "attachment_name",
                    col("line_number"),
                    col("value").alias("line"),
                    col("predicted_label").alias("label")
                )
                .groupBy("doc_id", "attachment_name")
                .agg(
                    expr(
                        "sort_array(collect_list(struct(line_number, line, label))) AS sorted_lines"
                    )
                )
            )

            # Define UDF to coalesce labels
            coalesce_udf = udf(self.coalesce_consecutive_labels, StringType())

            # Extract line/label pairs and coalesce
            final_df = (
                aggregated_df
                .withColumn(
                    "lines_data",
                    expr("transform(sorted_lines, x -> struct(x.line as line, x.label as label))")
                )
                .withColumn("final_aggregated_pg", coalesce_udf(col("lines_data")))
                .select("doc_id", "attachment_name", "final_aggregated_pg")
            )
        else:
            # Original paragraph-based aggregation
            aggregated_df = (
                predictions.groupBy("doc_id", "attachment_name")
                .agg(
                    expr("sort_array(collect_list(struct(row_number, annotated_value))) AS sorted_list")
                )
                .withColumn("annotated_value_ordered", expr("transform(sorted_list, x -> x.annotated_value)"))
                .withColumn("final_aggregated_pg", expr("array_join(annotated_value_ordered, '\n')"))
                .select("doc_id", "attachment_name", "final_aggregated_pg")
            )
            final_df = aggregated_df

        # Save to CouchDB using distributed UDF
        result_df = conn.save_distributed(final_df, suffix)

        # Collect results
        results = []
        for row in result_df.collect():
            results.append({
                'doc_id': row.doc_id,
                'attachment_name': f"{row.attachment_name}{suffix}",
                'success': row.success
            })

        return results

    def load_raw_data_lines(self, text_contents: List[str]) -> DataFrame:
        """
        Load raw text data as individual lines (not paragraphs).

        Args:
            text_contents: List of raw text strings

        Returns:
            DataFrame with individual lines
        """
        # Create DataFrame from raw text strings
        # Each string in the list is treated as a separate document
        data = []
        for doc_id, text in enumerate(text_contents):
            lines = text.split('\n')
            for line_num, line in enumerate(lines):
                data.append((f"doc_{doc_id}", line, line_num))

        df = self.spark.createDataFrame(
            data,
            ["filename", "value", "line_number"]
        )

        return df

    def predict_lines(
        self,
        text_contents: List[str],
        output_format: str = "yeda"
    ) -> DataFrame:
        """
        Process and predict labels for individual lines in raw text strings.

        Args:
            text_contents: List of raw text strings
            output_format: Output format ('yeda', 'annotated', or 'simple')

        Returns:
            DataFrame with line-level predictions

        Raises:
            ValueError: If models are not trained
        """
        if self.pipeline_model is None or self.classifier_model is None:
            raise ValueError(
                "Models not trained. Call fit_features() and train_classifier() first."
            )

        # Load lines (not paragraphs)
        raw_df = self.load_raw_data_lines(text_contents)

        # Extract features
        features = self.pipeline_model.transform(raw_df)

        # Predict
        predictions = self.classifier_model.transform(features)

        # Convert label indices to strings
        converter = IndexToString(
            inputCol="prediction",
            outputCol="predicted_label",
            labels=self.labels
        )
        labeled_predictions = converter.transform(predictions)

        # Format output based on requested format
        if output_format == "yeda":
            return labeled_predictions
        elif output_format == "annotated":
            labeled_predictions = labeled_predictions.withColumn(
                "annotated_line",
                concat(
                    lit("[@ "),
                    col("value"),
                    lit("\n#"),
                    col("predicted_label"),
                    lit("*]")
                )
            )
            return labeled_predictions
        else:
            return labeled_predictions

    @staticmethod
    def coalesce_consecutive_labels(lines_data: List[Dict[str, Any]]) -> str:
        """
        Coalesce consecutive lines with the same label into YEDA blocks.

        Args:
            lines_data: List of dicts with 'line' and 'label' keys,
                       sorted by line_number

        Returns:
            String with YEDA-formatted blocks
        """
        if not lines_data:
            return ""

        blocks = []
        current_label = None
        current_lines = []

        for item in lines_data:
            line = item['line']
            label = item['label']

            if label == current_label:
                # Same label, add to current block
                current_lines.append(line)
            else:
                # Label changed, finish previous block
                if current_lines and current_label:
                    blocks.append(
                        f"[@ {chr(10).join(current_lines)}\n#{current_label}*]"
                    )
                # Start new block
                current_label = label
                current_lines = [line]

        # Don't forget the last block
        if current_lines and current_label:
            blocks.append(
                f"[@ {chr(10).join(current_lines)}\n#{current_label}*]"
            )

        return "\n".join(blocks)

    def save_yeda_output(
        self,
        predictions: DataFrame,
        output_path: str
    ) -> None:
        """
        Save YEDA-formatted predictions to disk.

        Coalesces consecutive lines with the same label into YEDA blocks.

        Args:
            predictions: DataFrame with line-level predictions
            output_path: Directory to save output files
        """

        # Create struct with line and label, collect by filename
        aggregated_df = (
            predictions
            .select(
                "filename",
                "line_number",
                col("value").alias("line"),
                col("predicted_label").alias("label")
            )
            .groupBy("filename")
            .agg(
                expr(
                    "sort_array(collect_list(struct(line_number, line, label))) AS sorted_lines"
                )
            )
        )

        # Define UDF to coalesce labels
        coalesce_udf = udf(self.coalesce_consecutive_labels, StringType())

        # Extract line/label pairs and coalesce
        final_df = (
            aggregated_df
            .withColumn(
                "lines_data",
                expr("transform(sorted_lines, x -> struct(x.line as line, x.label as label))")
            )
            .withColumn("yeda_text", coalesce_udf(col("lines_data")))
            .select("filename", "yeda_text")
        )

        # Write to disk
        final_df.write.partitionBy("filename").mode("overwrite").text(output_path)