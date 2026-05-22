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
_ANN_ATTACHMENT = "article.txt.ann"


class ClassifierLogisticV3Instance(ComponentInstance):
    """Per-doc runtime: reads .ann attachment, contributes YEDDA text."""

    def run(self, state: PipelineState) -> None:
        ann_bytes = state.get_attachment(_ANN_ATTACHMENT)
        ann_text = ann_bytes.decode("utf-8")
        state.add_ann_text(
            source=_SOURCE,
            text=ann_text,
            priority=_PRIORITY,
        )


class ClassifierLogisticV3(CatalogElementMixin, SectionLabeler):
    """Catalog descriptor for the classifier-output labeler.

    Preconditions: ``has_plaintext`` (the .ann attachment lives
    alongside the .txt the classifier read).  We additionally
    require ``not has_taxpub_markup`` to be implicit at dispatch
    time — when both gates hold we let the dispatcher run both
    components and the priority-10 taxpub extractor wins on merge.
    """

    _name = _SOURCE
    _tags = {
        "category": "section_labeler",
        "cost": "low",
        "source": "model",
        "produces": ["treatment_labels"],
        "requires_props": ["has_plaintext"],
    }
    requires_props: FrozenSet[str] = frozenset({"has_plaintext"})
    requires_outputs: FrozenSet[str] = frozenset()
    produces_outputs: FrozenSet[str] = frozenset({"treatment_labels"})
    instance_constructor: Type[ComponentInstance] = (
        ClassifierLogisticV3Instance
    )

    def preconditions(self, props: Dict[str, Any]) -> bool:
        return bool(props.get("has_plaintext"))


def register(catalog: MemoryCatalog[Component], **kwargs: Any) -> None:
    """Autoloader entry point."""
    descriptor = ClassifierLogisticV3()
    catalog.register(descriptor, descriptor.name, descriptor.tags)
