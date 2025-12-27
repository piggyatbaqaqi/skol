"""
Ingesters package for ingesting web data into CouchDB.

This package provides base classes and specialized implementations for
ingesting data from various web sources.
"""

from .ingestor import Ingestor
from .ingenta import IngentaIngestor
from .local_ingenta import LocalIngentaIngestor

__all__ = ['Ingestor', 'IngentaIngestor', 'LocalIngentaIngestor']
