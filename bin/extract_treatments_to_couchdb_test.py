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

from extract_treatments_to_couchdb import generate_taxon_doc_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_dict(**overrides):
    """Minimal taxon dict with all section fields absent (None)."""
    d = {
        'taxon': 'Amanita muscaria (L.) Lam.',
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
        a = generate_taxon_doc_id(_base_dict(taxon='Amanita muscaria'))
        b = generate_taxon_doc_id(_base_dict(taxon='Amanita phalloides'))
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


if __name__ == '__main__':
    unittest.main()
