"""
Main classifier module for SKOL text classification
"""

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

    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the SKOL classifier.

        Args:
            spark: SparkSession (creates one if not provided)
        """
        if spark is None:
            import sparknlp
            self.spark = sparknlp.start()
        else:
            self.spark = spark

        self.pipeline_model: Optional[PipelineModel] = None
        self.classifier_model: Optional[PipelineModel] = None
        self.labels: Optional[List[str]] = None

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