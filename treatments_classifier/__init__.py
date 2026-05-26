"""Taxa Classifiers using TF-IDF and Decision Tree.

This module provides tools for building classifiers that can identify
taxa based on their descriptions or structured JSON annotations using
TF-IDF text encoding and Decision Tree classification.

Classes:
    TreatmentsDecisionTreeClassifier: Classifies taxa based on description text
    TreatmentsJSONClassifier: Classifies taxa based on flattened JSON annotations
"""

from .treatments_decision_tree import TreatmentsDecisionTreeClassifier
from .treatments_json_classifier import TreatmentsJSONClassifier

__all__ = ['TreatmentsDecisionTreeClassifier', 'TreatmentsJSONClassifier']
