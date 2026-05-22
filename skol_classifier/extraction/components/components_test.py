"""Tests for the three extraction-pipeline Commit-1 components.

Covers:
  - taxpub_treatment_extractor    (TaggedBlock contribution path)
  - classifier_logistic_v3        (ann_text contribution path)
  - treatment_assembler           (terminal — turns merged YEDDA into
                                   Treatment objects via existing
                                   parse_annotated → group_paragraphs)
  - autoloader integration         (MemoryCatalog.load picks up all
                                    three)
"""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from ingestors.yedda_tags import Tag, TaggedBlock

from ..catalog import MemoryCatalog
from ..interfaces import Component
from ..state import PipelineState

from .classifier_logistic_v3 import (
    ClassifierLogisticV3,
    ClassifierLogisticV3Instance,
)
from .taxpub_treatment_extractor import (
    TaxpubTreatmentExtractor,
    TaxpubTreatmentExtractorInstance,
)
from .treatment_assembler import (
    TreatmentAssembler,
    TreatmentAssemblerInstance,
)


# Minimal TaxPub fixture: one treatment with description + etymology.
_TAXPUB_XML = """<?xml version="1.0"?>
<article xmlns:tp="http://www.plazi.org/taxpub">
  <body>
    <tp:taxon-treatment>
      <tp:nomenclature>
        <tp:taxon-name><tp:taxon-name-part>Foo bar</tp:taxon-name-part></tp:taxon-name>
      </tp:nomenclature>
      <tp:treatment-sec sec-type="description">
        <p>Cap red.</p>
      </tp:treatment-sec>
      <tp:treatment-sec sec-type="etymology">
        <p>From Latin.</p>
      </tp:treatment-sec>
    </tp:taxon-treatment>
  </body>
</article>
"""

_YEDDA_ANN = (
    "[@Foo bar#Nomenclature*]\n\n"
    "[@Cap red.#Description*]\n\n"
    "[@From Latin.#Etymology*]\n"
)


class TestTaxpubTreatmentExtractor(TestCase):
    """Wraps jats_xml_to_tagged_blocks; emits TaggedBlock list."""

    def test_preconditions_gates_on_has_taxpub_markup(self) -> None:
        descriptor = TaxpubTreatmentExtractor()
        self.assertFalse(descriptor.preconditions({}))
        self.assertFalse(
            descriptor.preconditions({"has_taxpub_markup": False}),
        )
        self.assertTrue(
            descriptor.preconditions({"has_taxpub_markup": True}),
        )

    def test_run_populates_section_labels_at_priority_10(self) -> None:
        descriptor = TaxpubTreatmentExtractor()
        instance = descriptor.create_instance()
        state = PipelineState(
            doc={
                "_id": "x",
                "_attachments": {"article.xml": _TAXPUB_XML},
            },
        )
        instance.run(state)
        # Contributor present at priority 10.
        self.assertEqual(state.label_sources(), ["taxpub_treatment_extractor"])
        self.assertEqual(
            state._label_contributions[0].priority, 10,
        )
        # At least one block emitted with one of the expected tags.
        blocks = state._label_contributions[0].blocks
        self.assertIsNotNone(blocks)
        tags = {b.tag for b in blocks}
        self.assertIn(Tag.NOMENCLATURE, tags)


class TestClassifierLogisticV3(TestCase):
    """Reads article.txt.ann attachment; emits YEDDA-text contribution
    at priority 4 — lossless passthrough, no TaggedBlock round-trip."""

    def test_preconditions_gates_on_has_plaintext(self) -> None:
        descriptor = ClassifierLogisticV3()
        self.assertFalse(descriptor.preconditions({}))
        self.assertTrue(
            descriptor.preconditions({"has_plaintext": True}),
        )

    def test_run_passes_ann_text_verbatim(self) -> None:
        descriptor = ClassifierLogisticV3()
        instance = descriptor.create_instance()
        state = PipelineState(
            doc={
                "_id": "x",
                "_attachments": {"article.txt.ann": _YEDDA_ANN},
            },
        )
        instance.run(state)
        self.assertEqual(state.label_sources(), ["classifier_logistic_v3"])
        contrib = state._label_contributions[0]
        self.assertEqual(contrib.priority, 4)
        self.assertIsNone(contrib.blocks)
        self.assertEqual(contrib.ann_text, _YEDDA_ANN)


class TestTreatmentAssembler(TestCase):
    """Reads merged_ann_text → CouchDBFile → parse_annotated →
    group_paragraphs.  Field-equality with the pre-dispatcher flow is
    verified in A.10 against skol_golden_v2; here we just check
    that the assembler runs end-to-end and produces Treatments."""

    def test_no_labelers_yields_no_treatments(self) -> None:
        descriptor = TreatmentAssembler()
        instance = descriptor.create_instance()
        state = PipelineState(doc={"_id": "x"})
        instance.run(state)
        self.assertEqual(state.treatments, [])

    def test_taxpub_blocks_flow_through_to_treatment(self) -> None:
        """A taxpub-style blocks contribution gets serialised to YEDDA
        text, parsed back through parse_annotated, and assembled
        into a Treatment by group_paragraphs."""
        descriptor = TreatmentAssembler()
        instance = descriptor.create_instance()
        state = PipelineState(doc={"_id": "x", "url": "http://e.com"})
        state.add_section_labels(
            "taxpub_treatment_extractor",
            [
                TaggedBlock(text="Foo bar", tag=Tag.NOMENCLATURE),
                TaggedBlock(text="Cap red.", tag=Tag.DESCRIPTION),
            ],
            priority=10,
        )
        instance.run(state)
        # group_paragraphs filters to treatments with nomenclature —
        # ours has one Nomenclature paragraph, so we expect ≥1.
        self.assertGreaterEqual(len(state.treatments), 1)

    def test_ann_text_flow_through_to_treatment(self) -> None:
        """A classifier-style ann_text contribution is consumed verbatim
        by parse_annotated."""
        descriptor = TreatmentAssembler()
        instance = descriptor.create_instance()
        state = PipelineState(doc={"_id": "x", "url": "http://e.com"})
        state.add_ann_text(
            "classifier_logistic_v3", _YEDDA_ANN, priority=4,
        )
        instance.run(state)
        self.assertGreaterEqual(len(state.treatments), 1)

    def test_higher_priority_wins_into_assembler(self) -> None:
        """When both labelers contribute, the priority-10 blocks
        contribution beats the priority-4 ann_text contribution."""
        descriptor = TreatmentAssembler()
        instance = descriptor.create_instance()
        state = PipelineState(doc={"_id": "x"})
        # Loser at priority 4 — would assemble its own treatment if
        # it won.
        state.add_ann_text(
            "classifier_logistic_v3",
            "[@LOSER NAME#Nomenclature*]\n\n[@LOSER DESC#Description*]\n",
            priority=4,
        )
        # Winner at priority 10.
        state.add_section_labels(
            "taxpub_treatment_extractor",
            [
                TaggedBlock(text="WINNER NAME", tag=Tag.NOMENCLATURE),
                TaggedBlock(text="WINNER DESC", tag=Tag.DESCRIPTION),
            ],
            priority=10,
        )
        instance.run(state)
        # Pick the Treatment row and confirm we got the winner's
        # nomenclature text.  Treatment.as_row() exposes the
        # concatenated Nomenclature paragraphs under the
        # ``treatment`` key.
        self.assertGreaterEqual(len(state.treatments), 1)
        row = state.treatments[0].as_row()
        treatment_text = row.get("treatment", "") or ""
        self.assertIn("WINNER NAME", treatment_text)
        self.assertNotIn("LOSER NAME", treatment_text)


class TestComponentAutoload(TestCase):
    """``MemoryCatalog.load`` picks up all three components in this
    directory under their declared category tags."""

    def test_load_registers_all_three(self) -> None:
        catalog: MemoryCatalog[Component] = MemoryCatalog()
        catalog.load(Path(__file__).parent)
        names = {n for n, _ in catalog.items()}
        self.assertEqual(
            names,
            {
                "taxpub_treatment_extractor",
                "classifier_logistic_v3",
                "treatment_assembler",
            },
        )

    def test_two_section_labelers_one_assembler(self) -> None:
        catalog: MemoryCatalog[Component] = MemoryCatalog()
        catalog.load(Path(__file__).parent)
        labelers = catalog.lookup_by_tag_and(category="section_labeler")
        self.assertEqual(
            set(labelers.keys()),
            {"taxpub_treatment_extractor", "classifier_logistic_v3"},
        )
        assemblers = catalog.lookup_by_tag_and(category="assembler")
        self.assertEqual(set(assemblers.keys()), {"treatment_assembler"})
