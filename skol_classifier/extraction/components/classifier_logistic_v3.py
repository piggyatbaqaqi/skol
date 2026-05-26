"""Component: ``classifier_logistic_v3``.

Reads the doc's pre-predicted ``article.txt.ann`` attachment (the
output of ``bin/predict_classifier.py`` running the ``v3_hand``
model) and contributes the raw YEDDA text to the pipeline state at
priority **4** — the model-based-labeler priority.

The component does *not* re-run the classifier; it consumes the
existing attachment.  The attachment is expected to be present on
PDF/plaintext docs that have been through ``predict_classifier``;
the dispatcher's ``preconditions`` gate on ``has_plaintext`` so we
only attempt the read when the doc plausibly carries a `.ann`.

Selected as the fallback labeler for docs without taxpub markup.
Returns a ``LabelContribution`` of ``ann_text`` shape so the
assembler can pass the YEDDA verbatim through ``parse_annotated``
— no lossy round-trip via ``TaggedBlock``.
"""

from typing import Any, Dict, FrozenSet, Type

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


_PRIORITY = 4
_SOURCE = "classifier_logistic_v3"
# Attachment names predict_classifier may have written.  Plaintext-fed
# docs land under ``article.txt.ann``; PDF-fed docs (crossref / Ingenta /
# direct-PDF ingestors) under ``article.pdf.ann``.  We prefer the text
# variant when both are present (the OCR/plaintext-derived prediction
# is typically cleaner than the raw-PDF one).
#
# Pre-fix: this component hardcoded ``article.txt.ann`` and raised
# FileNotFoundError on every PDF-derived doc, producing zero treatments
# from named-journal ingest records (Mycotaxon, Journal of Fungi via
# crossref, Cryptogamie, etc.).  The bug only surfaced when the
# Ingestion Sources page audit showed all 14,032 v3_hand treatments
# bucketed into "Unknown" — see the user's report.
_ANN_ATTACHMENTS = ("article.txt.ann", "article.pdf.ann")


class ClassifierLogisticV3Instance(ComponentInstance):
    """Per-doc runtime: reads .ann attachment, contributes YEDDA text."""

    def run(self, state: PipelineState) -> None:
        atts = state.doc.get("_attachments") or {}
        attachment_name = next(
            (name for name in _ANN_ATTACHMENTS if name in atts), None,
        )
        if attachment_name is None:
            # Preconditions should have gated this, but be defensive:
            # let get_attachment raise the canonical FileNotFoundError
            # if neither variant is present.
            attachment_name = _ANN_ATTACHMENTS[0]
        ann_bytes = state.get_attachment(attachment_name)
        ann_text = ann_bytes.decode("utf-8")
        state.add_ann_text(
            source=_SOURCE,
            text=ann_text,
            priority=_PRIORITY,
        )


class ClassifierLogisticV3(CatalogElementMixin, SectionLabeler):
    """Catalog descriptor for the classifier-output labeler.

    Preconditions: ``has_yedda_ann`` — the .ann attachment must
    exist on the doc (an upstream ``predict_classifier`` run wrote
    it).  We deliberately gate on the .ann file rather than on
    ``has_plaintext`` because Spark partitions feed the dispatcher
    just the .ann content, not the underlying .txt.

    When both this component and ``taxpub_treatment_extractor``
    qualify, the priority-10 taxpub extractor wins on merge at the
    PipelineState level.
    """

    _name = _SOURCE
    _tags = {
        "category": "section_labeler",
        "cost": "low",
        "source": "model",
        "produces": ["treatment_labels"],
        "requires_props": ["has_yedda_ann"],
    }
    requires_props: FrozenSet[str] = frozenset({"has_yedda_ann"})
    requires_outputs: FrozenSet[str] = frozenset()
    produces_outputs: FrozenSet[str] = frozenset({"treatment_labels"})
    instance_constructor: Type[ComponentInstance] = (
        ClassifierLogisticV3Instance
    )

    def preconditions(self, props: Dict[str, Any]) -> bool:
        return bool(props.get("has_yedda_ann"))


def register(catalog: MemoryCatalog[Component], **kwargs: Any) -> None:
    """Autoloader entry point."""
    descriptor = ClassifierLogisticV3()
    catalog.register(descriptor, descriptor.name, descriptor.tags)
