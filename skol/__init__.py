"""
SKOL - Taxonomic text classification and extraction pipeline.

This package provides constrained decoding and ontology support for
structured extraction from mycological literature.
"""

from .constrained_decoder import ConstrainedDecoder, TaxonomySchema
from .ontology import OntologyRegistry, OntologyContextBuilder

__all__ = [
    'ConstrainedDecoder',
    'TaxonomySchema',
    'OntologyRegistry',
    'OntologyContextBuilder',
]
