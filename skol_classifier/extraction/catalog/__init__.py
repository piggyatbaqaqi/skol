"""Catalog infrastructure for the skol extraction pipeline.

Vendored from autonlab/ngautonml@main (Apache-2.0) — see catalog.py
for full attribution.  Provides:

  - ``Catalog[T]``: abstract generic catalog interface
  - ``MemoryCatalog[T]``: in-memory implementation with tag indexes +
    directory autoloader
  - ``CatalogElementMixin``: mixin supplying ``name`` and ``tags``
    properties for objects intended to live in a catalog

Skol's extraction pipeline (``skol_classifier/extraction/``) uses two
parameterised catalogs: ``InspectorCatalog`` and ``ComponentCatalog``.
"""

from .catalog import (
    Catalog,
    CatalogError,
    CatalogLookupError,
    CatalogNameError,
    CatalogValueError,
    upcast,
)
from .catalog_element_mixin import CatalogElementMixin
from .memory_catalog import (
    CatalogDuplicateError,
    CatalogNoRegister,
    MemoryCatalog,
)

__all__ = [
    "Catalog",
    "CatalogElementMixin",
    "CatalogError",
    "CatalogLookupError",
    "CatalogNameError",
    "CatalogNoRegister",
    "CatalogDuplicateError",
    "CatalogValueError",
    "MemoryCatalog",
    "upcast",
]
