"""Inspector: does the doc carry TaxPub treatment markup?

Produces ``has_taxpub_markup`` — ``True`` iff the article.xml
attachment contains at least one ``<*:taxon-treatment>`` element.
The check is a simple substring test (the same heuristic used by
``bin/jats_to_yedda._has_taxpub``); see :func:`_has_taxpub` below.

The ingestion-time ``is_taxpub`` field on skol_dev docs already
encodes this; we trust it when present and only re-scan the XML if
the field is missing.

Requires ``has_xml`` (so we don't try to read a missing attachment)
and ``xml_format`` (so we don't sniff non-JATS XML).
"""

from typing import Any, Dict, FrozenSet

from skol_classifier.extraction.catalog import (
    CatalogElementMixin,
    MemoryCatalog,
)
from skol_classifier.extraction.interfaces import Inspector


def _has_taxpub(xml_string: str) -> bool:
    """Return True iff *xml_string* contains taxon-treatment markup.

    Substring check mirroring the same-named helper in
    ``bin/jats_to_yedda.py``.  Cheap, intentional — taxon-treatment
    is a Plazi/TaxPub element name unlikely to appear in non-TaxPub
    XML.
    """
    return "taxon-treatment" in xml_string


class TaxpubMarkupInspector(CatalogElementMixin, Inspector):
    _name = "taxpub_markup"
    _tags = {
        "category": "inspector",
        "cost": "low",
        "produces": ["has_taxpub_markup"],
    }
    requires: FrozenSet[str] = frozenset({"has_xml", "xml_format"})

    def inspect(self, doc: Dict[str, Any], props: Dict[str, Any]) -> Dict[str, Any]:
        if not props.get("has_xml"):
            return {"has_taxpub_markup": False}
        if props.get("xml_format") not in ("jats", "taxpub"):
            return {"has_taxpub_markup": False}

        # Prefer the ingestion-time field.
        flag = doc.get("is_taxpub")
        if isinstance(flag, bool):
            return {"has_taxpub_markup": flag}

        # Fallback: sniff the XML.  Best to look at the in-doc bytes
        # rather than re-issue a CouchDB get_attachment call — we
        # rely on the dispatcher to have passed a doc with an
        # _attachments mapping that may hold pre-fetched bytes.
        atts = doc.get("_attachments") or {}
        entry = atts.get("article.xml")
        if isinstance(entry, (bytes, str)):
            xml = entry.decode("utf-8") if isinstance(entry, bytes) else entry
            return {"has_taxpub_markup": _has_taxpub(xml)}

        # No bytes available without a network call — be conservative.
        return {"has_taxpub_markup": False}


def register(catalog: MemoryCatalog[Inspector], **kwargs: Any) -> None:
    inspector = TaxpubMarkupInspector()
    catalog.register(inspector, inspector.name, inspector.tags)
