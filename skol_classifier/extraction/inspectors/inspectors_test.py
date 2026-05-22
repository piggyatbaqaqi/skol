"""Tests for the four extraction-pipeline inspectors.

Each inspector gets a happy-path test (correct property emitted on
representative input) and a missing-precondition test (gracefully
returns conservative defaults when its ``requires`` set isn't
satisfied).  Plus an autoload integration test that confirms
``MemoryCatalog.load(inspectors_dir)`` picks up all four.
"""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from ..catalog import MemoryCatalog
from ..interfaces import Inspector

from .attachments import AttachmentsInspector
from .plaintext_signal import PlaintextSignalInspector
from .taxpub_markup import TaxpubMarkupInspector
from .xml_root import XmlRootInspector


class TestAttachmentsInspector(TestCase):
    """Emits four boolean flags based on the attachments dict keys."""

    def test_all_present(self) -> None:
        ins = AttachmentsInspector()
        doc = {"_attachments": {
            "article.xml": {},
            "article.pdf": {},
            "article.txt": {},
            "article.md": {},
        }}
        self.assertEqual(
            ins.inspect(doc, {}),
            {
                "has_xml": True,
                "has_pdf": True,
                "has_plaintext": True,
                "has_markdown": True,
            },
        )

    def test_none_present(self) -> None:
        ins = AttachmentsInspector()
        self.assertEqual(
            ins.inspect({}, {}),
            {
                "has_xml": False,
                "has_pdf": False,
                "has_plaintext": False,
                "has_markdown": False,
            },
        )


class TestXmlRootInspector(TestCase):
    """Trusts ingestion-time xml_format when present; otherwise falls
    back to a conservative 'other'."""

    def test_no_xml_returns_none(self) -> None:
        ins = XmlRootInspector()
        self.assertEqual(
            ins.inspect({}, {"has_xml": False}),
            {"xml_format": "none"},
        )

    def test_trusts_jats_field(self) -> None:
        ins = XmlRootInspector()
        self.assertEqual(
            ins.inspect({"xml_format": "jats"}, {"has_xml": True}),
            {"xml_format": "jats"},
        )

    def test_trusts_taxpub_field(self) -> None:
        ins = XmlRootInspector()
        self.assertEqual(
            ins.inspect({"xml_format": "taxpub"}, {"has_xml": True}),
            {"xml_format": "taxpub"},
        )

    def test_falls_back_to_other_when_unknown(self) -> None:
        ins = XmlRootInspector()
        self.assertEqual(
            ins.inspect({}, {"has_xml": True}),
            {"xml_format": "other"},
        )


class TestTaxpubMarkupInspector(TestCase):
    """Trusts the ingestion-time is_taxpub flag when present; falls
    back to scanning the XML bytes only when both are needed."""

    def test_no_xml_returns_false(self) -> None:
        ins = TaxpubMarkupInspector()
        self.assertEqual(
            ins.inspect({}, {"has_xml": False, "xml_format": "none"}),
            {"has_taxpub_markup": False},
        )

    def test_non_jats_xml_returns_false(self) -> None:
        ins = TaxpubMarkupInspector()
        self.assertEqual(
            ins.inspect(
                {"xml_format": "other"},
                {"has_xml": True, "xml_format": "other"},
            ),
            {"has_taxpub_markup": False},
        )

    def test_trusts_is_taxpub_true(self) -> None:
        ins = TaxpubMarkupInspector()
        self.assertEqual(
            ins.inspect(
                {"is_taxpub": True, "xml_format": "jats"},
                {"has_xml": True, "xml_format": "jats"},
            ),
            {"has_taxpub_markup": True},
        )

    def test_trusts_is_taxpub_false(self) -> None:
        ins = TaxpubMarkupInspector()
        self.assertEqual(
            ins.inspect(
                {"is_taxpub": False, "xml_format": "jats"},
                {"has_xml": True, "xml_format": "jats"},
            ),
            {"has_taxpub_markup": False},
        )

    def test_falls_back_to_xml_sniff(self) -> None:
        """When is_taxpub is missing and the doc has pre-seeded XML
        bytes, the inspector scans them."""
        ins = TaxpubMarkupInspector()
        xml = (
            '<article xmlns:tp="http://www.plazi.org/taxpub">'
            '<body><tp:taxon-treatment/></body></article>'
        )
        self.assertEqual(
            ins.inspect(
                {"xml_format": "jats", "_attachments": {"article.xml": xml}},
                {"has_xml": True, "xml_format": "jats"},
            ),
            {"has_taxpub_markup": True},
        )

    def test_xml_sniff_without_taxpub_returns_false(self) -> None:
        ins = TaxpubMarkupInspector()
        xml = "<article><body><sec><p>Hello</p></sec></body></article>"
        self.assertEqual(
            ins.inspect(
                {"xml_format": "jats", "_attachments": {"article.xml": xml}},
                {"has_xml": True, "xml_format": "jats"},
            ),
            {"has_taxpub_markup": False},
        )


class TestPlaintextSignalInspector(TestCase):
    """Detects taxonomy abbreviations in pre-seeded article.txt bytes."""

    def test_no_plaintext_returns_false(self) -> None:
        ins = PlaintextSignalInspector()
        self.assertEqual(
            ins.inspect({}, {"has_plaintext": False}),
            {"has_taxonomic_signal": False},
        )

    def test_sp_nov_triggers(self) -> None:
        ins = PlaintextSignalInspector()
        doc = {"_attachments": {
            "article.txt": "Hericium ophelieae sp. nov. described.",
        }}
        self.assertEqual(
            ins.inspect(doc, {"has_plaintext": True}),
            {"has_taxonomic_signal": True},
        )

    def test_no_abbrevs_returns_false(self) -> None:
        ins = PlaintextSignalInspector()
        doc = {"_attachments": {
            "article.txt": "An article about cake recipes.",
        }}
        self.assertEqual(
            ins.inspect(doc, {"has_plaintext": True}),
            {"has_taxonomic_signal": False},
        )

    def test_var_triggers(self) -> None:
        ins = PlaintextSignalInspector()
        doc = {"_attachments": {
            "article.txt": "Foo bar var. baz",
        }}
        self.assertEqual(
            ins.inspect(doc, {"has_plaintext": True}),
            {"has_taxonomic_signal": True},
        )

    def test_custom_abbrevs_override(self) -> None:
        ins = PlaintextSignalInspector(taxonomy_abbrevs=["xyzzy."])
        doc = {"_attachments": {"article.txt": "Foo xyzzy. bar"}}
        self.assertEqual(
            ins.inspect(doc, {"has_plaintext": True}),
            {"has_taxonomic_signal": True},
        )
        # The default abbrev list is no longer in effect.
        doc2 = {"_attachments": {"article.txt": "Foo sp. bar"}}
        self.assertEqual(
            ins.inspect(doc2, {"has_plaintext": True}),
            {"has_taxonomic_signal": False},
        )


class TestInspectorAutoload(TestCase):
    """The autoloader picks up all 4 inspectors in this directory."""

    def test_load_registers_all_four(self) -> None:
        catalog: MemoryCatalog[Inspector] = MemoryCatalog()
        catalog.load(Path(__file__).parent)
        names = {n for n, _ in catalog.items()}
        self.assertEqual(
            names,
            {"attachments", "xml_root", "taxpub_markup", "plaintext_signal"},
        )

    def test_loaded_inspectors_share_inspector_category(self) -> None:
        catalog: MemoryCatalog[Inspector] = MemoryCatalog()
        catalog.load(Path(__file__).parent)
        by_category = catalog.lookup_by_tag_and(category="inspector")
        self.assertEqual(
            set(by_category.keys()),
            {"attachments", "xml_root", "taxpub_markup", "plaintext_signal"},
        )
