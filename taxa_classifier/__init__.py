"""Taxa Classifiers using TF-IDF and Decision Tree.

This module provides tools for building classifiers that can identify
taxa based on their descriptions or structured JSON annotations using
TF-IDF text encoding and Decision Tree classification.

Classes:
    TaxaDecisionTreeClassifier: Classifies taxa based on description text
    TaxaJsonClassifier: Classifies taxa based on flattened JSON annotations
"""

from .taxa_decision_tree import TaxaDecisionTreeClassifier
from .taxa_json_classifier import TaxaJsonClassifier

__all__ = ['TaxaDecisionTreeClassifier', 'TaxaJsonClassifier']
