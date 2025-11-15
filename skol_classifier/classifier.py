"""
Main classifier module for SKOL text classification
"""

import pickle
import tempfile
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    input_file_name, collect_list, regexp_extract, col, udf,
    explode, trim, row_number, min, expr, concat, lit
)
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Tokenizer, CountVectorizer, IDF, StringIndexer, VectorAssembler, IndexToString
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

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
        redis_key: str = "skol_classifier_model"
    ):
        """
        Initialize the SKOL classifier.

        Args:
            spark: SparkSession (creates one if not provided)
            redis_client: Redis client connection (optional, for model persistence)
            redis_key: Key name to use in Redis for storing the model
        """
        if spark is None:
            import sparknlp
            self.spark = sparknlp.start()
        else:
            self.spark = spark

        self.pipeline_model: Optional[PipelineModel] = None
        self.classifier_model: Optional[PipelineModel] = None
        self.labels: Optional[List[str]] = None
        self.redis_client = redis_client
        self.redis_key = redis_key

    def load_annotated_data(
        self,
        file_paths: List[str],
        collapse_labels: bool = True
    ) -> DataFrame:
        """
        Load and preprocess annotated data.

        Args:
            file_paths: List of paths to annotated files
            collapse_labels: Whether to collapse labels to 3 main categories

        Returns:
            Preprocessed DataFrame with paragraphs and labels
        """
        # Read annotated files
        ann_df = self.spark.read.text(file_paths).withColumn(
            "filename", input_file_name()
        )

        # Define UDF for paragraph extraction
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

        # Group and extract paragraphs
        grouped_df = (
            df.groupBy("filename")
            .agg(
                collect_list("value").alias("lines"),
                min("filename").alias("start_line_id")
            )
            .withColumn("value", explode(heuristic_udf(col("lines"))))
            .drop("lines")
            .filter(trim(col("value")) != "")
            .withColumn("row_number", row_number().over(window_spec))
        )

        return grouped_df

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
                "annotated_pg",
                concat(
                    lit("[@ "),
                    col("value"),
                    lit("#"),
                    col("predicted_label"),
                    lit("]")
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
                expr("sort_array(collect_list(struct(row_number, annotated_pg))) AS sorted_list")
            )
            .withColumn("annotated_pg_ordered", expr("transform(sorted_list, x -> x.annotated_pg)"))
            .withColumn("final_aggregated_pg", expr("array_join(annotated_pg_ordered, '\n')"))
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
        **model_params
    ) -> Dict[str, Any]:
        """
        Complete training pipeline: load data, extract features, train model, evaluate.

        Args:
            annotated_file_paths: Paths to annotated training files
            model_type: Type of classifier
            use_suffixes: Whether to use suffix features
            test_size: Proportion of data for testing
            **model_params: Additional model parameters

        Returns:
            Dictionary with evaluation results
        """
        # Load annotated data
        annotated_df = self.load_annotated_data(annotated_file_paths)

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
            **stats
        }

    def save_to_redis(
        self,
        redis_client: Optional[Any] = None,
        redis_key: Optional[str] = None
    ) -> bool:
        """
        Save the trained models to Redis.

        The models are saved to a temporary directory, then packaged and stored in Redis
        as a compressed binary blob along with metadata.

        Args:
            redis_client: Redis client (uses self.redis_client if not provided)
            redis_key: Redis key name (uses self.redis_key if not provided)

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If no models are trained or Redis client is not available
        """
        if self.pipeline_model is None or self.classifier_model is None:
            raise ValueError(
                "No models to save. Train models using fit() or train_classifier() first."
            )

        client = redis_client or self.redis_client
        key = redis_key or self.redis_key

        if client is None:
            raise ValueError(
                "No Redis client available. Provide redis_client argument or "
                "initialize classifier with redis_client."
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
            client.set(key, archive_data)

            return True

        except Exception as e:
            print(f"Error saving to Redis: {e}")
            return False

        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)

    def load_from_redis(
        self,
        redis_client: Optional[Any] = None,
        redis_key: Optional[str] = None
    ) -> bool:
        """
        Load trained models from Redis.

        Args:
            redis_client: Redis client (uses self.redis_client if not provided)
            redis_key: Redis key name (uses self.redis_key if not provided)

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If Redis client is not available or key doesn't exist
        """
        client = redis_client or self.redis_client
        key = redis_key or self.redis_key

        if client is None:
            raise ValueError(
                "No Redis client available. Provide redis_client argument or "
                "initialize classifier with redis_client."
            )

        temp_dir = None
        try:
            # Retrieve from Redis
            archive_data = client.get(key)
            if archive_data is None:
                raise ValueError(f"No model found in Redis with key: {key}")

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