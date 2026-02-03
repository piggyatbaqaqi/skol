"""Taxa Description Classifier using TF-IDF and Decision Tree.

This module provides tools for building classifiers that can identify
taxa based on their descriptions using TF-IDF text encoding and
Decision Tree classification.
"""

from .taxa_decision_tree import TaxaDecisionTreeClassifier

__all__ = ['TaxaDecisionTreeClassifier']
