"""Component: ``taxpub_treatment_extractor``.

Deterministic structural extraction for TaxPub-marked JATS XML.
Reads the doc's ``article.xml`` attachment, walks the
``<*:taxon-treatment>`` elements via
:func:`ingestors.jats_to_yedda.jats_xml_to_tagged_blocks`, and
contributes the resulting :class:`TaggedBlock` list to the pipeline
state at priority **10** — the highest deterministic-labeler
priority (per docs/extraction_pipeline.md §"Output merging").

Selected by the dispatcher whenever ``has_taxpub_markup`` holds.

The component is stateless; ``create_instance()`` returns a new
:class:`TaxpubTreatmentExtractorInstance` every time the dispatcher
asks for one.  The actual XML parsing happens inside
``Instance.run(state)``, lazily fetching the attachment via
:meth:`PipelineState.get_attachment`.
"""

from typing import Any, Dict, FrozenSet, Type

from ingestors.jats_to_yedda import jats_xml_to_tagged_blocks

from skol_classifier.extraction.catalog import (
    CatalogElementMixin,
    MemoryCatalog,
)
from skol_classifier.extraction.interfaces import (
    Component,
    ComponentInstance,
    SectionLabeler,
)
from skol_classifier.extraction.state import PipelineState


_PRIORITY = 10
_SOURCE = "taxpub_treatment_extractor"


class TaxpubTreatmentExtractorInstance(ComponentInstance):
    """Per-doc runtime: reads article.xml, contributes TaggedBlocks."""

    def run(self, state: PipelineState) -> None:
        xml_bytes = state.get_attachment("article.xml")
        xml = xml_bytes.decode("utf-8")
        blocks = jats_xml_to_tagged_blocks(xml)
        state.add_section_labels(
            source=_SOURCE,
            blocks=blocks,
            priority=_PRIORITY,
        )


class TaxpubTreatmentExtractor(CatalogElementMixin, SectionLabeler):
    """Catalog descriptor for the TaxPub deterministic extractor."""

    _name = _SOURCE
    _tags = {
        "category": "section_labeler",
        "cost": "low",
        "source": "skol_native",
        "produces": ["treatment_labels"],
        "requires_props": ["has_taxpub_markup"],
    }
    requires_props: FrozenSet[str] = frozenset({"has_taxpub_markup"})
    requires_outputs: FrozenSet[str] = frozenset()
    produces_outputs: FrozenSet[str] = frozenset({"treatment_labels"})
    instance_constructor: Type[ComponentInstance] = (
        TaxpubTreatmentExtractorInstance
    )

    def preconditions(self, props: Dict[str, Any]) -> bool:
        return bool(props.get("has_taxpub_markup"))


def register(catalog: MemoryCatalog[Component], **kwargs: Any) -> None:
    """Autoloader entry point."""
    descriptor = TaxpubTreatmentExtractor()
    catalog.register(descriptor, descriptor.name, descriptor.tags)
