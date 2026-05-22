"""Component: ``treatment_assembler``.

The terminal assembler.  Reads the merged YEDDA text from
:meth:`PipelineState.merged_ann_text`, builds a
:class:`CouchDBFile` so the existing parse-and-group pipeline can
consume it unchanged, and produces ``Treatment`` records into
``state.treatments``.

Two important design choices:

1. **Reuses the existing Paragraph-based pipeline.** The
   pre-dispatcher flow used ``parse_annotated → remove_interstitials
   → group_paragraphs``.  Rather than rebuilding the Treatment
   assembly logic against TaggedBlock input, we serialise the
   merged contribution back to YEDDA text and feed it through
   ``CouchDBFile``.  This preserves field-equality with the
   pre-refactor pipeline (verified in A.10 of v3_buildout) at the
   cost of a serialisation hop on the taxpub path.

2. **Per-doc metadata sourced from ``state.doc``.**  Treatment's
   ``ingest`` field flows down from each ``Line``'s metadata.  We
   populate it by passing the original doc dict as the
   ``CouchDBFile`` ingest payload — the same field shape the
   Spark-partition path produced.

Selected by the dispatcher last (its preconditions check that at
least one labeler has contributed something to the state).
"""

from typing import Any, Dict, FrozenSet, Type

from couchdb_file import CouchDBFile
from finder import parse_annotated, remove_interstitials
from treatment import group_paragraphs

from skol_classifier.extraction.catalog import (
    CatalogElementMixin,
    MemoryCatalog,
)
from skol_classifier.extraction.interfaces import (
    Assembler,
    Component,
    ComponentInstance,
)
from skol_classifier.extraction.state import PipelineState


_SOURCE = "treatment_assembler"
_DEFAULT_ATTACHMENT_NAME = "article.txt.ann"


class TreatmentAssemblerInstance(ComponentInstance):
    """Per-doc runtime: turns merged YEDDA into Treatment objects."""

    def run(self, state: PipelineState) -> None:
        ann_text = state.merged_ann_text()
        if not ann_text:
            # No labeler contributed; nothing to assemble.
            return

        doc = state.doc
        doc_id = doc.get("_id", "unknown")
        # Match the row.value/row.ingest convention from
        # couchdb_file.read_couchdb_partition: pass the full doc as
        # the ingest payload so downstream Line/Paragraph/Treatment
        # objects can read ``url``, ``pdf_url``, and other ingest
        # fields straight from it.
        db_name = state.config.get("ingest_db_name", "")

        file_obj = CouchDBFile(
            content=ann_text,
            doc_id=doc_id,
            attachment_name=_DEFAULT_ATTACHMENT_NAME,
            db_name=db_name,
            human_url=doc.get("url"),
            pdf_url=doc.get("pdf_url"),
            ingest=doc,
        )

        lines = list(file_obj.read_line())
        paragraphs = parse_annotated(iter(lines))
        filtered = remove_interstitials(paragraphs)
        treatments = [
            t for t in group_paragraphs(iter(list(filtered)))
            if t.has_nomenclature()
        ]
        state.treatments = treatments


class TreatmentAssembler(CatalogElementMixin, Assembler):
    """Catalog descriptor for the terminal assembler.

    ``preconditions`` is True iff at least one labeler has put
    something into ``state._label_contributions`` — there's no
    point assembling when no contributor has spoken.

    Reads the labeler outputs via ``state.merged_ann_text`` (and
    so its ``requires_outputs`` declares ``treatment_labels``,
    which the dispatcher uses to topologically order this
    component last).
    """

    _name = _SOURCE
    _tags = {
        "category": "assembler",
        "cost": "low",
        "source": "skol_native",
        "produces": ["treatments"],
        "requires_outputs": ["treatment_labels"],
    }
    requires_props: FrozenSet[str] = frozenset()
    requires_outputs: FrozenSet[str] = frozenset({"treatment_labels"})
    produces_outputs: FrozenSet[str] = frozenset({"treatments"})
    instance_constructor: Type[ComponentInstance] = (
        TreatmentAssemblerInstance
    )

    def preconditions(self, props: Dict[str, Any]) -> bool:
        # The assembler always runs *if* at least one labeler has
        # populated state.  The dispatcher checks ``requires_outputs``
        # against what's actually available; this method is the
        # property-only gate (no property gates it).
        return True


def register(catalog: MemoryCatalog[Component], **kwargs: Any) -> None:
    """Autoloader entry point."""
    descriptor = TreatmentAssembler()
    catalog.register(descriptor, descriptor.name, descriptor.tags)
