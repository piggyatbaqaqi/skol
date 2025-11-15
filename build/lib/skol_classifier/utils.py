"""
Utility functions for SKOL text classification
"""

import glob
from typing import List
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def get_file_list(folder: str, pattern: str = "**/*.txt*", exclude_pattern: str = "Sydowia") -> List[str]:
    """
    List all files matching the pattern in a folder.

    Args:
        folder: Directory path to search
        pattern: Glob pattern for file matching (default: "**/*.txt*")
        exclude_pattern: Pattern to exclude from results (default: "Sydowia")

    Returns:
        List of file paths matching the criteria

    Raises:
        FileNotFoundError: If the folder doesn't exist
        PermissionError: If access to the folder is denied
    """
    try:
        files = [
            file for file in glob.glob(f'{folder}/{pattern}', recursive=True)
            if exclude_pattern not in file
        ]
        return files
    except FileNotFoundError:
        print(f"Folder '{folder}' not found.")
        raise
    except PermissionError:
        print(f"Permission denied to access folder '{folder}'.")
        raise


def create_evaluators():
    """
    Create evaluation metrics for multiclass classification.

    Returns:
        Dictionary containing evaluators for accuracy, precision, recall, and F1 score
    """
    evaluators = {
        'accuracy': MulticlassClassificationEvaluator(
            labelCol="label_indexed",
            predictionCol="prediction",
            metricName="accuracy"
        ),
        'precision': MulticlassClassificationEvaluator(
            labelCol="label_indexed",
            predictionCol="prediction",
            metricName="precisionByLabel"
        ),
        'recall': MulticlassClassificationEvaluator(
            labelCol="label_indexed",
            predictionCol="prediction",
            metricName="recallByLabel"
        ),
        'f1': MulticlassClassificationEvaluator(
            labelCol="label_indexed",
            predictionCol="prediction",
            metricName="f1"
        )
    }
    return evaluators


def calculate_stats(predictions, evaluators=None, verbose=True):
    """
    Calculate and optionally print evaluation statistics.

    Args:
        predictions: PySpark DataFrame with predictions
        evaluators: Dictionary of evaluators (created if None)
        verbose: Whether to print statistics (default: True)

    Returns:
        Dictionary containing accuracy, precision, recall, and f1_score
    """
    if evaluators is None:
        evaluators = create_evaluators()

    stats = {
        'accuracy': evaluators['accuracy'].evaluate(predictions),
        'precision': evaluators['precision'].evaluate(predictions),
        'recall': evaluators['recall'].evaluate(predictions),
        'f1_score': evaluators['f1'].evaluate(predictions)
    }

    if verbose:
        print(f"Test Accuracy: {stats['accuracy']:.4f}")
        print(f"Test Precision: {stats['precision']:.4f}")
        print(f"Test Recall: {stats['recall']:.4f}")
        print(f"Test F1 Score: {stats['f1_score']:.4f}")

    return stats