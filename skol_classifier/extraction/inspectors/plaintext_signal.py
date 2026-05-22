"""Inspector: does the plaintext show signs of taxonomic content?

Produces ``has_taxonomic_signal`` — ``True`` iff the article.txt
attachment contains at least one of the taxonomy abbreviations
listed in ``env_config['taxonomy_abbrevs']`` (sp., var., gen.,
nov., etc.).

Matches the pre-filter used in
``bin/predict_classifier.mark_taxonomy_documents()`` so the
dispatcher's decision aligns with the existing pipeline's.

Requires ``has_plaintext`` (so we don't read a missing attachment).
"""

from typing import Any, Dict, FrozenSet, List

from skol_classifier.extraction.catalog import (
    CatalogElementMixin,
    MemoryCatalog,
)
from skol_classifier.extraction.interfaces import Inspector


class PlaintextSignalInspector(CatalogElementMixin, Inspector):
    _name = "plaintext_signal"
    _tags = {
        "category": "inspector",
        "cost": "low",
        "produces": ["has_taxonomic_signal"],
    }
    requires: FrozenSet[str] = frozenset({"has_plaintext"})

    #: Default abbreviation list — kept in sync with
    #: ``env_config['taxonomy_abbrevs']``.  Overridable via the
    #: ``taxonomy_abbrevs`` key of the constructor kwargs (which the
    #: autoloader forwards from MemoryCatalog kwargs).
    DEFAULT_ABBREVS: List[str] = [
        "comb.", "fam.", "gen.", "nom.", "ined.", "var.", "subg.",
        "subsp.", "sp.", "f.", "syn.", "nov.", "spec.", "ssp.", "spp.",
        "sensu", "s.l.", "s.s.", "s.str.", "cf.", "aff.", "incertae",
        "sed.",
    ]

    def __init__(self, taxonomy_abbrevs: List[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._abbrevs = list(taxonomy_abbrevs) if taxonomy_abbrevs else list(
            self.DEFAULT_ABBREVS
        )

    def inspect(self, doc: Dict[str, Any], props: Dict[str, Any]) -> Dict[str, Any]:
        if not props.get("has_plaintext"):
            return {"has_taxonomic_signal": False}

        atts = doc.get("_attachments") or {}
        entry = atts.get("article.txt")
        if not isinstance(entry, (bytes, str)):
            # Real CouchDB metadata dict — without a get_attachment
            # call we can't sniff.  Be conservative.
            return {"has_taxonomic_signal": False}

        text = entry.decode("utf-8", errors="replace") if isinstance(entry, bytes) else entry
        for abbrev in self._abbrevs:
            if abbrev in text:
                return {"has_taxonomic_signal": True}
        return {"has_taxonomic_signal": False}


def register(catalog: MemoryCatalog[Inspector], **kwargs: Any) -> None:
    inspector = PlaintextSignalInspector(**kwargs)
    catalog.register(inspector, inspector.name, inspector.tags)
