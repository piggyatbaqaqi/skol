"""Widgets for testing the catalog module.

Vendored from ngautonml (Apache-2.0); skol-local import paths.
"""

from ...memory_catalog import MemoryCatalog
from ...catalog_element_mixin import CatalogElementMixin


class Widget():
    """This is the thing that is stored in WidgetCatalog."""


class NamedWidget(Widget, CatalogElementMixin):
    """Widget that inherits from CatalogElementMixin and so has a name."""


class WidgetCatalog(MemoryCatalog[Widget]):
    """This is the class we will exercise."""
