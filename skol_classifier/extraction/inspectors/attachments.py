"""Inspector: which input formats are attached to this doc?

Reads the doc's ``_attachments`` keys and produces four boolean
properties: ``has_xml``, ``has_pdf``, ``has_plaintext``, ``has_markdown``.

Per docs/extraction_pipeline.md the dispatcher uses these to decide
which components are eligible to run (e.g. the taxpub extractor
requires ``has_xml``; gnfinder requires ``has_plaintext``).
"""

from typing import Any, Dict

from skol_classifier.extraction.catalog import (
    CatalogElementMixin,
    MemoryCatalog,
)
from skol_classifier.extraction.interfaces import Inspector


class AttachmentsInspector(CatalogElementMixin, Inspector):
    """Read the doc's attachment-name dict; emit per-format flags."""

    _name = "attachments"
    _tags = {
        "category": "inspector",
        "cost": "low",
        "produces": [
            "has_xml", "has_pdf", "has_plaintext", "has_markdown",
            "has_yedda_ann",
        ],
    }

    def inspect(self, doc: Dict[str, Any], props: Dict[str, Any]) -> Dict[str, Any]:
        atts = doc.get("_attachments") or {}
        return {
            "has_xml": "article.xml" in atts,
            "has_pdf": "article.pdf" in atts,
            "has_plaintext": "article.txt" in atts,
            "has_markdown": "article.md" in atts,
            # True if any pre-existing YEDDA ``.ann`` attachment is
            # present — these are the outputs of an upstream
            # predict_classifier or jats_to_yedda run.  Used by the
            # ``classifier_logistic_v3`` component which reads that
            # attachment verbatim.
            "has_yedda_ann": any(
                k.endswith(".txt.ann") or k.endswith(".pdf.ann")
                for k in atts
            ),
        }


def register(catalog: MemoryCatalog[Inspector], **kwargs: Any) -> None:
    """Autoloader entry point."""
    inspector = AttachmentsInspector()
    catalog.register(inspector, inspector.name, inspector.tags)
