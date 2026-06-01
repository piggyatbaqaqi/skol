"""Tests for rename_journals pure helpers.

Phase 2 of the publication-metadata consolidation
(docs/publications_metadata_consolidation.md) — walks skol_dev,
identifies docs whose ``journal`` field carries an old compound
name (``"Journal of Fungi (PMC)"`` etc.), and prepares a rewrite.

The CouchDB-touching ``apply_renames`` is exercised by hand
against the real skol_dev with ``--dry-run``; no fixture for
that here.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestors.publications import strip_publisher_suffix  # type: ignore[import]
from rename_journals import compute_renames  # type: ignore[import]


class TestComputeRenames(unittest.TestCase):
    """``compute_renames(docs, mapping_fn)`` walks a sequence of
    documents and returns ``{doc_id: (old_journal, new_journal)}``
    for those that would actually change.  Idempotent —
    ``mapping_fn(name) == name`` produces no entry."""

    def test_compound_name_detected(self):
        docs = [{'_id': 'x', 'journal': 'Journal of Fungi (PMC)'}]
        result = compute_renames(docs, strip_publisher_suffix)
        self.assertEqual(
            result,
            {'x': ('Journal of Fungi (PMC)', 'Journal of Fungi')},
        )

    def test_no_op_when_mapping_returns_same(self):
        """A canonical name passes through unchanged → no entry."""
        docs = [{'_id': 'x', 'journal': 'Mycotaxon'}]
        result = compute_renames(docs, strip_publisher_suffix)
        self.assertEqual(result, {})

    def test_skips_doc_without_journal_field(self):
        docs = [{'_id': 'x', 'meta': {'source': 'mykoweb'}}]
        result = compute_renames(docs, strip_publisher_suffix)
        self.assertEqual(result, {})

    def test_skips_empty_or_none_journal(self):
        docs = [
            {'_id': 'empty',   'journal': ''},
            {'_id': 'none',    'journal': None},
            {'_id': 'spaces',  'journal': '   '},
        ]
        result = compute_renames(docs, strip_publisher_suffix)
        self.assertEqual(result, {})

    def test_skips_design_docs(self):
        """CouchDB ``_design/`` docs aren't user data."""
        docs = [{'_id': '_design/foo', 'journal': 'X (PMC)'}]
        result = compute_renames(docs, strip_publisher_suffix)
        self.assertEqual(result, {})

    def test_multiple_docs_mixed(self):
        """Compound names get entries; canonical names don't."""
        docs = [
            {'_id': 'a', 'journal': 'Mycology (PMC)'},
            {'_id': 'b', 'journal': 'Mycology'},
            {'_id': 'c', 'journal': 'Journal of Fungi (PMC)'},
            {'_id': 'd', 'journal': 'Mycotaxon'},
        ]
        result = compute_renames(docs, strip_publisher_suffix)
        self.assertEqual(set(result.keys()), {'a', 'c'})
        self.assertEqual(result['a'][1], 'Mycology')

    def test_non_string_journal_skipped(self):
        """Defensive: a doc whose ``journal`` is mistyped as int /
        list / dict doesn't crash; just skipped."""
        docs = [
            {'_id': 'a', 'journal': 123},
            {'_id': 'b', 'journal': ['X', 'Y']},
            {'_id': 'c', 'journal': {'name': 'X'}},
        ]
        result = compute_renames(docs, strip_publisher_suffix)
        self.assertEqual(result, {})

    def test_arbitrary_mapping_fn(self):
        """The mapping_fn is pluggable.  Phase 3 will pass a
        different one (post-alias-migration normalize_journal_name);
        confirm the script doesn't bake strip_publisher_suffix into
        its contract."""
        def upcase(name: str) -> str:
            return name.upper()
        docs = [
            {'_id': 'a', 'journal': 'mycotaxon'},
            {'_id': 'b', 'journal': 'MYCOTAXON'},
        ]
        result = compute_renames(docs, upcase)
        self.assertEqual(
            result,
            {'a': ('mycotaxon', 'MYCOTAXON')},
        )


if __name__ == '__main__':
    unittest.main()
