"""
Ingesters package for ingesting web data into CouchDB.

This package provides base classes and specialized implementations for
ingesting data from various web sources.
"""

from .ingestor import Ingestor
from .ingenta import IngentaIngestor
from .local_ingenta import LocalIngentaIngestor
from .local_mykoweb import LocalMykowebJournalsIngestor
from .local_mykoweb_literature import LocalMykowebLiteratureIngestor
from .mycosphere import MycosphereIngestor
from .publications import PublicationRegistry
from .rate_limited_client import RateLimitedHttpClient
from .timestamps import set_timestamps, get_iso_timestamp

__all__ = [
    'Ingestor',
    'IngentaIngestor',
    'LocalIngentaIngestor',
    'LocalMykowebJournalsIngestor',
    'LocalMykowebLiteratureIngestor',
    'MycosphereIngestor',
    'PublicationRegistry',
    'RateLimitedHttpClient',
    'set_timestamps',
    'get_iso_timestamp',
]
