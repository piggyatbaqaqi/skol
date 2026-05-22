"""Tests for the extraction-pipeline interface ABCs."""

from __future__ import annotations

from typing import Any, Dict
from unittest import TestCase

from .catalog import (
    CatalogElementMixin,
    CatalogNameError,
    MemoryCatalog,
)
from .interfaces import (
    Assembler,
    Component,
    ComponentInstance,
    EntityDetector,
    Inspector,
    SectionLabeler,
    TextProducer,
)


# A minimal concrete inspector used by the registration tests.
class _DummyInspector(CatalogElementMixin, Inspector):
    _name = "dummy_inspector"
    _tags = {"category": "inspector", "cost": "low"}

    def inspect(self, doc, props):
        return {"dummy_prop": True}


# A minimal concrete component instance + descriptor used by the
# registration tests.
class _DummyInstance(ComponentInstance):
    def run(self, state):
        state.props["ran_dummy"] = True


class _DummySectionLabeler(CatalogElementMixin, SectionLabeler):
    _name = "dummy_labeler"
    _tags = {
        "category": "section_labeler",
        "cost": "low",
        "produces": ["treatment_labels"],
    }
    requires_props = frozenset({"dummy_prop"})
    produces_outputs = frozenset({"treatment_labels"})
    instance_constructor = _DummyInstance

    def preconditions(self, props: Dict[str, Any]) -> bool:
        return bool(props.get("dummy_prop"))


class TestInspectorAbc(TestCase):
    """Inspector contract: cannot instantiate ABC; concrete subclass
    works and registers cleanly into a catalog."""

    def test_abstract_class_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            Inspector()  # type: ignore[abstract]

    def test_concrete_inspector_inspects(self) -> None:
        ins = _DummyInspector()
        self.assertEqual(
            ins.inspect({"_id": "x"}, {}),
            {"dummy_prop": True},
        )

    def test_concrete_inspector_registers_in_catalog(self) -> None:
        catalog: MemoryCatalog[Inspector] = MemoryCatalog()
        ins = _DummyInspector()
        catalog.register(ins, ins.name, ins.tags)
        self.assertIs(catalog.lookup_by_name("dummy_inspector"), ins)

    def test_requires_defaults_to_empty(self) -> None:
        self.assertEqual(_DummyInspector.requires, frozenset())


class TestComponentAbc(TestCase):
    """Component contract: descriptor + ComponentInstance split,
    preconditions check, registration via tags."""

    def test_abstract_classes_cannot_be_instantiated(self) -> None:
        for cls in (Component, TextProducer, EntityDetector,
                    SectionLabeler, Assembler):
            with self.assertRaises(TypeError):
                cls()  # type: ignore[abstract]

    def test_concrete_component_registers_in_catalog(self) -> None:
        catalog: MemoryCatalog[Component] = MemoryCatalog()
        comp = _DummySectionLabeler()
        catalog.register(comp, comp.name, comp.tags)
        self.assertIs(catalog.lookup_by_name("dummy_labeler"), comp)
        # Filter by tag.
        section_labelers = catalog.lookup_by_tag_and(
            category="section_labeler",
        )
        self.assertEqual(set(section_labelers.keys()), {"dummy_labeler"})

    def test_preconditions_reads_props(self) -> None:
        comp = _DummySectionLabeler()
        self.assertFalse(comp.preconditions({}))
        self.assertTrue(comp.preconditions({"dummy_prop": True}))

    def test_create_instance_returns_componentinstance(self) -> None:
        comp = _DummySectionLabeler()
        inst = comp.create_instance()
        self.assertIsInstance(inst, ComponentInstance)
        self.assertIsInstance(inst, _DummyInstance)


class TestCatalogElementMixinIntegration(TestCase):
    """Sanity check: classes that forget to set ``_name`` raise when
    instantiated, even when they inherit through ``Inspector`` /
    ``Component`` ABCs."""

    def test_inspector_without_name_raises(self) -> None:
        class _NoNameInspector(CatalogElementMixin, Inspector):
            def inspect(self, doc, props):
                return {}

        with self.assertRaises(CatalogNameError):
            _NoNameInspector()

    def test_component_without_name_raises(self) -> None:
        class _NoNameComponent(CatalogElementMixin, SectionLabeler):
            instance_constructor = _DummyInstance

            def preconditions(self, props):
                return True

        with self.assertRaises(CatalogNameError):
            _NoNameComponent()
