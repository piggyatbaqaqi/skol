"""Test object that registers itself in a WidgetCatalog.

Vendored from ngautonml (Apache-2.0); skol-local import paths.
"""

from typing import Optional

from skol_classifier.extraction.catalog.catalog import upcast
from skol_classifier.extraction.catalog.widgets.impl.widget_catalog import (
    NamedWidget,
    WidgetCatalog,
)


class DooDad(NamedWidget):
    """Test object to register in a Widget catalog."""

    tags = {
        'some_tag': ['test_dir'],
    }
    frob: Optional[str]
    quux: Optional[str]

    def __init__(self, frob: Optional[str] = None, quux: Optional[str] = None):
        super().__init__('DooDad')
        self.frob = frob
        self.quux = quux


def register(catalog: WidgetCatalog, *args, **kwargs) -> None:
    """Register all the objects in this file."""
    doodad = DooDad(*args, **kwargs)
    catalog.register(doodad, doodad.name, upcast(doodad.tags))
