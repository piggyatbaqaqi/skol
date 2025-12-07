"""
Feature extraction module for SKOL classifier.

This module provides the FeatureExtractor class for transforming text into
features suitable for machine learning classification.
"""

from typing import List, Optional
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Tokenizer, CountVectorizer, IDF, StringIndexer, VectorAssembler
)
from pyspark.sql import DataFrame

from .preprocessing import SuffixTransformer


class FeatureExtractor:
    """
    Extracts features from text data for classification.

    Supports word TF-IDF and optional suffix TF-IDF features.
    """

    def __init__(
        self,
        use_suffixes: bool = True,
        min_doc_freq: int = 2,
        input_col: str = "value",
        label_col: str = "label",
        word_vocab_size: int = 800,
        suffix_vocab_size: int = 200
    ):
        """
        Initialize the feature extractor.

        Args:
            use_suffixes: Whether to include suffix features
            min_doc_freq: Minimum document frequency for IDF
            input_col: Name of input text column
            label_col: Name of label column
        """
        self.use_suffixes = use_suffixes
        self.min_doc_freq = min_doc_freq
        self.input_col = input_col
        self.label_col = label_col
        self.pipeline_model: Optional[PipelineModel] = None
        self.labels: Optional[List[str]] = None
        self.word_vocab_size = word_vocab_size
        self.suffix_vocab_size = suffix_vocab_size

    def build_pipeline(self) -> Pipeline:
        """
        Build the feature extraction pipeline.

        Returns:
            Pipeline object
        """
        # Tokenization
        tokenizer = Tokenizer(inputCol=self.input_col, outputCol="words")

        # Word TF-IDF
        word_count_vectorizer = CountVectorizer(
            inputCol="words", outputCol="word_tf", vocabSize=self.word_vocab_size
        )
        word_idf = IDF(
            inputCol="word_tf", outputCol="word_idf", minDocFreq=self.min_doc_freq
        )

        stages = [tokenizer, word_count_vectorizer, word_idf]

        # Suffix TF-IDF (optional)
        if self.use_suffixes:
            suffixer = SuffixTransformer(inputCol="words", outputCol="suffixes")
            suffix_count_vectorizer = CountVectorizer(
                inputCol="suffixes", outputCol="suffix_tf",
                vocabSize=self.suffix_vocab_size
            )
            suffix_idf = IDF(
                inputCol="suffix_tf", outputCol="suffix_idf", minDocFreq=self.min_doc_freq
            )
            idf_combiner = VectorAssembler(
                inputCols=["word_idf", "suffix_idf"], outputCol="combined_idf"
            )
            stages.extend([
                suffixer, suffix_count_vectorizer, suffix_idf, idf_combiner
            ])

        # Label indexing
        indexer = StringIndexer(inputCol=self.label_col, outputCol="label_indexed")
        stages.append(indexer)

        return Pipeline(stages=stages)

    def fit_transform(self, data: DataFrame) -> DataFrame:
        """
        Fit the feature extraction pipeline and transform data.

        Args:
            data: Input DataFrame with text and label columns

        Returns:
            Transformed DataFrame with features
        """
        pipeline = self.build_pipeline()
        self.pipeline_model = pipeline.fit(data)
        self.labels = self.pipeline_model.stages[-1].labels
        return self.pipeline_model.transform(data)

    def transform(self, data: DataFrame) -> DataFrame:
        """
        Transform data using the fitted pipeline.

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame with features

        Raises:
            ValueError: If pipeline hasn't been fitted yet
        """
        if self.pipeline_model is None:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")
        return self.pipeline_model.transform(data)

    def get_pipeline(self) -> Optional[PipelineModel]:
        """Get the fitted pipeline model."""
        return self.pipeline_model

    def get_labels(self) -> Optional[List[str]]:
        """Get the label list."""
        return self.labels

    def get_label_mapping(self) -> Optional[List[str]]:
        """Get the label mapping (alias for get_labels)."""
        return self.labels

    def get_features_col(self) -> str:
        """
        Get the name of the features column.

        Returns:
            Name of the features column based on configuration
        """
        return "combined_idf" if self.use_suffixes else "word_idf"
