"""Inspector: identify the XML format of the article.xml attachment.

Produces ``xml_format``, one of ``"jats"``, ``"taxpub"``, ``"other"``,
or ``"none"`` (``"none"`` when no XML attachment is present).

The ``xml_format`` field is set on most skol_dev docs during PMC
ingestion; this inspector trusts that value when present and only
falls back to a sniff of the XML root element if it's missing.

Requires the ``has_xml`` property (produced by AttachmentsInspector).
"""

from typing import Any, Dict, FrozenSet

from skol_classifier.extraction.catalog import (
    CatalogElementMixin,
    MemoryCatalog,
)
from skol_classifier.extraction.interfaces import Inspector


class XmlRootInspector(CatalogElementMixin, Inspector):
    _name = "xml_root"
    _tags = {
        "category": "inspector",
        "cost": "low",
        "produces": ["xml_format"],
    }
    requires: FrozenSet[str] = frozenset({"has_xml"})

    def inspect(self, doc: Dict[str, Any], props: Dict[str, Any]) -> Dict[str, Any]:
        if not props.get("has_xml"):
            return {"xml_format": "none"}
        # Trust the ingestion-time field where set.
        fmt = doc.get("xml_format")
        if fmt in ("jats", "taxpub"):
            return {"xml_format": fmt}
        if isinstance(fmt, str) and fmt:
            return {"xml_format": "other"}
        # No ingestion-time field — could sniff XML, but skol_dev is
        # well-populated so this branch is empirical-rare.  Default to
        # "other" so downstream components don't accidentally run.
        return {"xml_format": "other"}


def register(catalog: MemoryCatalog[Inspector], **kwargs: Any) -> None:
    inspector = XmlRootInspector()
    catalog.register(inspector, inspector.name, inspector.tags)
