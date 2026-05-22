"""Tests for the extraction-pipeline dispatcher.

Covers the four-phase dispatch (inspectors → selection → topological
sort → execution) using a mix of dummy components for controlled
unit tests and the live default-catalog factory for end-to-end
integration tests.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest import TestCase

from ingestors.yedda_tags import Tag, TaggedBlock

from .catalog import CatalogElementMixin, MemoryCatalog
from .dispatcher import Dispatcher, DispatcherError
from .interfaces import (
    Component,
    ComponentInstance,
    Inspector,
    SectionLabeler,
)


# ---------------------------------------------------------------------------
# Lightweight test doubles
# ---------------------------------------------------------------------------


class _DummyInspector(CatalogElementMixin, Inspector):
    """Adds a named property to the state."""

    def __init__(self, name: str, produces: str, value: Any = True):
        self._name = name
        self._tags = {"category": "inspector"}
        self._produces = produces
        self._value = value
        super().__init__()

    def inspect(self, doc, props):
        return {self._produces: self._value}


class _RecordingInstance(ComponentInstance):
    """A ComponentInstance that records its run order on a list."""

    def __init__(self, name: str, log: list):
        self._name = name
        self._log = log

    def run(self, state):
        self._log.append(self._name)


class _DummyLabeler(CatalogElementMixin, SectionLabeler):
    """A SectionLabeler with configurable preconditions + dependencies."""

    def __init__(self, name, runlog,
                 *, requires_props=frozenset(),
                 requires_outputs=frozenset(),
                 produces_outputs=frozenset({"treatment_labels"})):
        self._name = name
        self._tags = {"category": "section_labeler"}
        self._runlog = runlog
        self.requires_props = frozenset(requires_props)
        self.requires_outputs = frozenset(requires_outputs)
        self.produces_outputs = frozenset(produces_outputs)
        super().__init__()

    def preconditions(self, props):
        return self.requires_props.issubset(props.keys())

    def create_instance(self, **kwargs):
        return _RecordingInstance(self._name, self._runlog)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestInspectorOrdering(TestCase):
    """Phase 1: inspectors run when their ``requires`` is satisfied."""

    def test_independent_inspectors_all_run(self) -> None:
        ic: MemoryCatalog[Inspector] = MemoryCatalog()
        cc: MemoryCatalog[Component] = MemoryCatalog()
        a = _DummyInspector("a", "alpha")
        b = _DummyInspector("b", "beta", value=42)
        ic.register(a, a.name, a.tags)
        ic.register(b, b.name, b.tags)
        state = Dispatcher(ic, cc).extract_state({"_id": "x"})
        self.assertEqual(state.props, {"alpha": True, "beta": 42})

    def test_dependent_inspector_runs_after_producer(self) -> None:
        """An inspector that requires ``alpha`` is held back until the
        inspector that produces ``alpha`` runs."""
        class _BetaNeedingAlpha(_DummyInspector):
            requires = frozenset({"alpha"})

        ic: MemoryCatalog[Inspector] = MemoryCatalog()
        cc: MemoryCatalog[Component] = MemoryCatalog()
        a = _DummyInspector("a", "alpha")
        b = _BetaNeedingAlpha("b", "beta")
        # Register in opposite order to ensure the dispatcher's
        # fixed-point loop handles re-ordering.
        ic.register(b, b.name, b.tags)
        ic.register(a, a.name, a.tags)
        state = Dispatcher(ic, cc).extract_state({"_id": "x"})
        self.assertEqual(state.props, {"alpha": True, "beta": True})

    def test_inspector_with_unsatisfied_requires_does_not_run(self) -> None:
        class _NeedsImpossible(_DummyInspector):
            requires = frozenset({"never_produced"})

        ic: MemoryCatalog[Inspector] = MemoryCatalog()
        cc: MemoryCatalog[Component] = MemoryCatalog()
        n = _NeedsImpossible("n", "never_runs")
        ic.register(n, n.name, n.tags)
        state = Dispatcher(ic, cc).extract_state({"_id": "x"})
        self.assertNotIn("never_runs", state.props)


class TestComponentSelection(TestCase):
    """Phase 2: only components whose preconditions hold get selected."""

    def test_unsatisfied_precondition_skipped(self) -> None:
        ic: MemoryCatalog[Inspector] = MemoryCatalog()
        cc: MemoryCatalog[Component] = MemoryCatalog()
        runlog: list = []
        ok = _DummyLabeler("ok", runlog,
                            requires_props=frozenset({"prop"}))
        nope = _DummyLabeler("nope", runlog,
                             requires_props=frozenset({"absent"}))
        cc.register(ok, ok.name, ok.tags)
        cc.register(nope, nope.name, nope.tags)
        produces_prop = _DummyInspector("p", "prop")
        ic.register(produces_prop, produces_prop.name, produces_prop.tags)
        Dispatcher(ic, cc).extract({"_id": "x"})
        self.assertIn("ok", runlog)
        self.assertNotIn("nope", runlog)


class TestTopologicalOrder(TestCase):
    """Phase 3: components run in dependency order; same-stratum
    components have unspecified relative order."""

    def test_producer_runs_before_consumer(self) -> None:
        ic: MemoryCatalog[Inspector] = MemoryCatalog()
        cc: MemoryCatalog[Component] = MemoryCatalog()
        runlog: list = []
        producer = _DummyLabeler(
            "producer", runlog,
            produces_outputs=frozenset({"treatment_labels"}),
        )
        consumer = _DummyLabeler(
            "consumer", runlog,
            requires_outputs=frozenset({"treatment_labels"}),
            produces_outputs=frozenset({"final"}),
        )
        cc.register(producer, producer.name, producer.tags)
        cc.register(consumer, consumer.name, consumer.tags)
        Dispatcher(ic, cc).extract({"_id": "x"})
        self.assertEqual(
            runlog.index("producer"), runlog.index("consumer") - 1,
        )

    def test_cycle_raises(self) -> None:
        ic: MemoryCatalog[Inspector] = MemoryCatalog()
        cc: MemoryCatalog[Component] = MemoryCatalog()
        runlog: list = []
        # Two labelers depending on each other.
        a = _DummyLabeler(
            "a", runlog,
            requires_outputs=frozenset({"b_out"}),
            produces_outputs=frozenset({"a_out"}),
        )
        b = _DummyLabeler(
            "b", runlog,
            requires_outputs=frozenset({"a_out"}),
            produces_outputs=frozenset({"b_out"}),
        )
        cc.register(a, a.name, a.tags)
        cc.register(b, b.name, b.tags)
        with self.assertRaises(DispatcherError):
            Dispatcher(ic, cc).extract({"_id": "x"})


class TestEndToEnd(TestCase):
    """Phase 4 + 5: integration tests using the default catalogs."""

    def test_taxpub_doc_emits_treatment(self) -> None:
        """A pre-seeded TaxPub doc flows through inspectors → taxpub
        extractor → assembler → state.treatments."""
        taxpub_xml = (
            '<?xml version="1.0"?>'
            '<article xmlns:tp="http://www.plazi.org/taxpub">'
            '<body><tp:taxon-treatment>'
            '<tp:nomenclature><tp:taxon-name>'
            '<tp:taxon-name-part>Foo bar</tp:taxon-name-part>'
            '</tp:taxon-name></tp:nomenclature>'
            '<tp:treatment-sec sec-type="description">'
            '<p>Cap red.</p>'
            '</tp:treatment-sec>'
            '</tp:taxon-treatment></body></article>'
        )
        doc = {
            "_id": "tx",
            "xml_format": "taxpub",
            "is_taxpub": True,
            "_attachments": {"article.xml": taxpub_xml},
        }
        treatments = Dispatcher.from_default_catalogs().extract(doc)
        self.assertGreaterEqual(len(treatments), 1)
        row = treatments[0].as_row()
        self.assertIn("Foo bar", row.get("treatment", ""))

    def test_plaintext_doc_emits_treatment(self) -> None:
        """A doc with article.txt + article.txt.ann flows through
        classifier_logistic_v3 → assembler."""
        ann = (
            "[@Foo bar#Nomenclature*]\n\n"
            "[@Cap red.#Description*]\n"
        )
        doc = {
            "_id": "p",
            "_attachments": {
                "article.txt": "Foo bar sp. nov.",
                "article.txt.ann": ann,
            },
        }
        treatments = Dispatcher.from_default_catalogs().extract(doc)
        self.assertGreaterEqual(len(treatments), 1)
        row = treatments[0].as_row()
        self.assertIn("Foo bar", row.get("treatment", ""))

    def test_taxpub_beats_classifier_on_overlap(self) -> None:
        """A doc with BOTH taxpub markup and a pre-existing .ann
        attachment yields a Treatment from the taxpub extractor
        (priority 10) rather than the classifier (priority 4)."""
        taxpub_xml = (
            '<?xml version="1.0"?>'
            '<article xmlns:tp="http://www.plazi.org/taxpub">'
            '<body><tp:taxon-treatment>'
            '<tp:nomenclature><tp:taxon-name>'
            '<tp:taxon-name-part>WINNER</tp:taxon-name-part>'
            '</tp:taxon-name></tp:nomenclature>'
            '<tp:treatment-sec sec-type="description">'
            '<p>From XML.</p>'
            '</tp:treatment-sec>'
            '</tp:taxon-treatment></body></article>'
        )
        doc = {
            "_id": "both",
            "xml_format": "taxpub",
            "is_taxpub": True,
            "_attachments": {
                "article.xml": taxpub_xml,
                "article.txt": "Foo sp. nov.",
                "article.txt.ann": "[@LOSER#Nomenclature*]\n",
            },
        }
        treatments = Dispatcher.from_default_catalogs().extract(doc)
        self.assertGreaterEqual(len(treatments), 1)
        row = treatments[0].as_row()
        self.assertIn("WINNER", row.get("treatment", ""))
        self.assertNotIn("LOSER", row.get("treatment", ""))

    def test_doc_without_taxonomic_content_yields_no_treatments(self) -> None:
        """A doc with neither XML markup nor a .ann attachment falls
        through cleanly: no component is selected, no Treatment is
        produced."""
        doc = {"_id": "empty"}
        treatments = Dispatcher.from_default_catalogs().extract(doc)
        self.assertEqual(treatments, [])
