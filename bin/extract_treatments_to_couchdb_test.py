"""Tests for extract_treatments_to_couchdb helper functions.

Covers generate_taxon_doc_id: hash stability, section sensitivity,
and None/empty equivalence.  The Spark-dependent pipeline code is
not tested here (integration tests live in tests/).

Run with: python -m pytest bin/extract_treatments_to_couchdb_test.py -v
"""

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_treatments_to_couchdb import (
    EXTRACT_SCHEMA, convert_taxa_to_rows, generate_taxon_doc_id,
    iter_taxpub_treatments,
)
from label import Label
from line import Line
from paragraph import Paragraph
from treatment import Treatment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_dict(**overrides):
    """Minimal taxon dict with all section fields absent (None)."""
    d = {
        'treatment': 'Amanita muscaria (L.) Lam.',
        'description': 'Cap convex, red with white warts.',
        'diagnosis': None,
        'etymology': None,
        'distribution': None,
        'materials_examined': None,
        'type_designation': None,
        'biology': None,
        'notes': None,
        'key': None,
        'figure_captions': None,
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Stability and format
# ---------------------------------------------------------------------------

class TestGenerateTaxonDocIdFormat(unittest.TestCase):
    """Output format is always 'taxon_<64-hex-chars>'."""

    def test_starts_with_taxon_prefix(self):
        doc_id = generate_taxon_doc_id(_base_dict())
        self.assertTrue(doc_id.startswith("taxon_"))

    def test_hex_suffix_is_64_chars(self):
        doc_id = generate_taxon_doc_id(_base_dict())
        hex_part = doc_id[len("taxon_"):]
        self.assertEqual(len(hex_part), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in hex_part))

    def test_deterministic_same_input(self):
        """Same dict always produces the same ID."""
        d = _base_dict()
        self.assertEqual(generate_taxon_doc_id(d), generate_taxon_doc_id(d))

    def test_deterministic_reconstructed_dict(self):
        """Independent construction of identical dicts gives identical IDs."""
        self.assertEqual(
            generate_taxon_doc_id(_base_dict()),
            generate_taxon_doc_id(_base_dict()),
        )


# ---------------------------------------------------------------------------
# Sensitivity to content changes
# ---------------------------------------------------------------------------

class TestGenerateTaxonDocIdSensitivity(unittest.TestCase):
    """Different content → different ID."""

    def test_different_taxon_text(self):
        a = generate_taxon_doc_id(_base_dict(treatment='Amanita muscaria'))
        b = generate_taxon_doc_id(_base_dict(treatment='Amanita phalloides'))
        self.assertNotEqual(a, b)

    def test_different_description(self):
        a = generate_taxon_doc_id(_base_dict(description='Cap red.'))
        b = generate_taxon_doc_id(_base_dict(description='Cap white.'))
        self.assertNotEqual(a, b)

    def test_different_diagnosis(self):
        a = generate_taxon_doc_id(_base_dict(diagnosis=None))
        b = generate_taxon_doc_id(_base_dict(diagnosis='Differs from A. phalloides.'))
        self.assertNotEqual(a, b)

    def test_different_etymology(self):
        a = generate_taxon_doc_id(_base_dict(etymology=None))
        b = generate_taxon_doc_id(_base_dict(etymology='From Latin muscarius.'))
        self.assertNotEqual(a, b)

    def test_different_distribution(self):
        a = generate_taxon_doc_id(_base_dict(distribution=None))
        b = generate_taxon_doc_id(_base_dict(distribution='Europe and North America.'))
        self.assertNotEqual(a, b)

    def test_different_materials_examined(self):
        a = generate_taxon_doc_id(_base_dict(materials_examined=None))
        b = generate_taxon_doc_id(_base_dict(materials_examined='NY 12345.'))
        self.assertNotEqual(a, b)

    def test_different_type_designation(self):
        a = generate_taxon_doc_id(_base_dict(type_designation=None))
        b = generate_taxon_doc_id(_base_dict(type_designation='Holotype: NY 12345.'))
        self.assertNotEqual(a, b)

    def test_different_biology(self):
        a = generate_taxon_doc_id(_base_dict(biology=None))
        b = generate_taxon_doc_id(_base_dict(biology='Saprotrophic on soil.'))
        self.assertNotEqual(a, b)

    def test_different_notes(self):
        a = generate_taxon_doc_id(_base_dict(notes=None))
        b = generate_taxon_doc_id(_base_dict(notes='See also A. muscaria var. formosa.'))
        self.assertNotEqual(a, b)

    def test_two_treatments_same_taxon_different_sections(self):
        """Two treatments with the same taxon but different section content differ."""
        a = generate_taxon_doc_id(_base_dict(description='Cap red.', diagnosis=None))
        b = generate_taxon_doc_id(_base_dict(description='Cap red.', diagnosis='Diag.'))
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# None / empty equivalence
# ---------------------------------------------------------------------------

class TestGenerateTaxonDocIdNoneHandling(unittest.TestCase):
    """None and empty string are treated identically for each field."""

    def test_none_and_empty_string_are_equivalent_for_diagnosis(self):
        a = generate_taxon_doc_id(_base_dict(diagnosis=None))
        b = generate_taxon_doc_id(_base_dict(diagnosis=''))
        self.assertEqual(a, b)

    def test_none_and_empty_string_are_equivalent_for_distribution(self):
        a = generate_taxon_doc_id(_base_dict(distribution=None))
        b = generate_taxon_doc_id(_base_dict(distribution=''))
        self.assertEqual(a, b)

    def test_none_and_empty_string_are_equivalent_for_biology(self):
        a = generate_taxon_doc_id(_base_dict(biology=None))
        b = generate_taxon_doc_id(_base_dict(biology=''))
        self.assertEqual(a, b)

    def test_whitespace_only_treated_as_empty(self):
        """Strip whitespace before hashing so '  ' ≡ '' ≡ None."""
        a = generate_taxon_doc_id(_base_dict(diagnosis=None))
        b = generate_taxon_doc_id(_base_dict(diagnosis='   '))
        self.assertEqual(a, b)

    def test_missing_key_treated_as_empty(self):
        """Dict without a section key is equivalent to that key being None."""
        full = _base_dict(diagnosis=None)
        partial = {k: v for k, v in full.items() if k != 'diagnosis'}
        self.assertEqual(
            generate_taxon_doc_id(full),
            generate_taxon_doc_id(partial),
        )


# ---------------------------------------------------------------------------
# Canonical ordering (field order is fixed, not dict-insertion order)
# ---------------------------------------------------------------------------

class TestGenerateTaxonDocIdCanonicalOrder(unittest.TestCase):
    """Hash uses a fixed canonical section order regardless of dict order."""

    def test_section_order_is_stable(self):
        """Diagnosis content does not collide with distribution content."""
        a = generate_taxon_doc_id(_base_dict(diagnosis='X', distribution=None))
        b = generate_taxon_doc_id(_base_dict(diagnosis=None, distribution='X'))
        self.assertNotEqual(a, b)

    def test_taxon_position_does_not_swap_with_description(self):
        """Swapping taxon and description values gives a different ID."""
        a = generate_taxon_doc_id(_base_dict(
            taxon='AAA', description='BBB',
        ))
        b = generate_taxon_doc_id(_base_dict(
            taxon='BBB', description='AAA',
        ))
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# Schema ↔ as_row() field-set contract
# ---------------------------------------------------------------------------

class TestExtractSchemaMatchesAsRow(unittest.TestCase):
    """EXTRACT_SCHEMA's field set must exactly match what gets fed into
    ``createDataFrame``: ``Treatment.as_row()`` keys plus the two keys
    added by ``convert_taxa_to_rows`` (``_id``, ``json_annotated``).

    Drift between the two sides triggers a Spark
    ``FIELD_STRUCT_LENGTH_MISMATCH`` at runtime — which the
    skol_golden_v2 verification path didn't catch because it compared
    as_row() dicts to each other, not against the Spark schema.
    """

    @staticmethod
    def _make_para(text: str, label_str: str, para_num: int = 1) -> Paragraph:
        line = Line(f"[@{text}#{label_str}*]")
        return Paragraph(
            labels=[Label(label_str)], lines=[line], paragraph_number=para_num,
        )

    def _minimal_treatment(self) -> Treatment:
        """A Treatment with one nomenclature and every section label,
        so as_row() exercises every section / span field."""
        t = Treatment()
        t.add_nomenclature(
            self._make_para("Amanita muscaria", "Nomenclature", para_num=1)
        )
        section_labels = [
            "Description", "Diagnosis", "Etymology", "Distribution",
            "Materials-examined", "Type-designation", "Biology", "Notes",
            "Key", "Figure-caption",
        ]
        for i, label_str in enumerate(section_labels, start=2):
            t.add_section(
                label_str, self._make_para(f"text-{label_str}", label_str, para_num=i)
            )
        return t

    def test_schema_field_set_matches_as_row_plus_caller_added(self):
        row = self._minimal_treatment().as_row()
        # convert_taxa_to_rows() adds these before createDataFrame:
        produced_keys = set(row.keys()) | {"_id", "json_annotated"}
        schema_keys = set(EXTRACT_SCHEMA.fieldNames())
        missing_in_schema = produced_keys - schema_keys
        missing_in_row = schema_keys - produced_keys
        self.assertEqual(
            missing_in_schema, set(),
            msg=(
                "Treatment.as_row() (+ caller-added keys) produces fields "
                f"that EXTRACT_SCHEMA does not declare: {missing_in_schema}. "
                "Add matching StructFields to EXTRACT_SCHEMA."
            ),
        )
        self.assertEqual(
            missing_in_row, set(),
            msg=(
                "EXTRACT_SCHEMA declares fields that Treatment.as_row() "
                f"(+ caller-added keys) never produces: {missing_in_row}. "
                "Either drop the StructFields or update as_row()."
            ),
        )

    def test_convert_taxa_to_rows_aligns_with_schema_positionally(self):
        """``createDataFrame(rdd, schema)`` matches Row fields by position,
        not by name.  ``Treatment.as_row()``'s dict insertion order does
        NOT match ``EXTRACT_SCHEMA.fieldNames()`` — without explicit
        reordering, the 15th value of the Row (``biology`` text in
        as_row order) would land in the schema's 15th slot (``pdf_page``,
        IntegerType), failing type validation on any non-empty Biology
        section.  This test locks the alignment by name and (because
        ``Row(**ordered_dict)`` preserves dict order) by position.
        """
        t = self._minimal_treatment()
        rows = list(convert_taxa_to_rows(iter([t])))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        schema_field_names = EXTRACT_SCHEMA.fieldNames()
        # Row.__fields__ is the kwarg order from Row(**dict).
        self.assertEqual(
            list(row.__fields__), schema_field_names,
            msg=(
                "Row field order must match EXTRACT_SCHEMA.fieldNames() "
                "exactly so positional matching in createDataFrame "
                "lands every value in the right column."
            ),
        )
        # Spot-check that values land in the right schema slots — not
        # shifted by the as_row()/schema ordering mismatch.
        row_dict = row.asDict()
        self.assertEqual(row_dict["pdf_page"], 0)  # int from Line default
        self.assertIsInstance(
            row_dict["biology"], str,
            msg="biology text must land in the biology field, not pdf_page",
        )
        self.assertIn("text-Biology", row_dict["biology"])


# ---------------------------------------------------------------------------
# Phase G.1: taxpub_treatment_extractor fork wiring
# ---------------------------------------------------------------------------

class TestIterTaxpubTreatments(unittest.TestCase):
    """``iter_taxpub_treatments`` is the non-Spark sweep that feeds
    ``is_taxpub=True`` docs through the dispatcher so the
    ``taxpub_treatment_extractor`` component actually fires — the gap
    documented in [v3_buildout.md §Phase G.1] where the Spark partition
    flow can't see ``article.xml`` bytes.
    """

    _TAXPUB_XML = b"""<?xml version="1.0"?>
<article xmlns:tp="http://www.plazi.org/taxpub">
  <body>
    <tp:taxon-treatment>
      <tp:nomenclature>
        <tp:taxon-name><tp:taxon-name-part>Foo bar</tp:taxon-name-part></tp:taxon-name>
      </tp:nomenclature>
      <tp:treatment-sec sec-type="description">
        <p>Cap red.</p>
      </tp:treatment-sec>
    </tp:taxon-treatment>
  </body>
</article>
"""

    def test_yields_treatment_for_taxpub_doc(self):
        doc = {"_id": "doc1", "is_taxpub": True, "xml_format": "jats"}
        treatments = list(iter_taxpub_treatments([(doc, self._TAXPUB_XML)]))
        self.assertEqual(len(treatments), 1)
        t = treatments[0]
        self.assertTrue(t.has_nomenclature())
        row = t.as_row()
        self.assertIn("Foo bar", row["treatment"])
        self.assertIn("Cap red", row["description"] or "")

    def test_pre_seeded_attachments_are_not_overwritten(self):
        """A doc that already has _attachments (e.g. CouchDB metadata
        stubs) must keep its other entries; only ``article.xml`` is
        injected by this helper."""
        doc = {
            "_id": "doc1",
            "is_taxpub": True,
            "xml_format": "jats",
            "_attachments": {"article.pdf": {"content_type": "application/pdf"}},
        }
        treatments = list(iter_taxpub_treatments([(doc, self._TAXPUB_XML)]))
        self.assertEqual(len(treatments), 1)
        # The input doc must not be mutated — taxpub iterator runs on a copy.
        self.assertNotIn("article.xml", doc["_attachments"])

    def test_empty_input_yields_nothing(self):
        self.assertEqual(list(iter_taxpub_treatments([])), [])


if __name__ == '__main__':
    unittest.main()
