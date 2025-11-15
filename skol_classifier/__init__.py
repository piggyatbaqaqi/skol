"""
SKOL II Text Classifier Module

A PySpark-based text classification pipeline for taxonomic literature.
Created by: Christopher Murphy, La Monte Yarroll, David Caspers
"""

from .classifier import SkolClassifier
from .preprocessing import ParagraphExtractor, SuffixTransformer
from .utils import get_file_list, create_evaluators, calculate_stats

__version__ = "0.1.0"
__all__ = [
    "SkolClassifier",
    "ParagraphExtractor",
    "SuffixTransformer",
    "get_file_list",
    "create_evaluators",
    "calculate_stats"
]