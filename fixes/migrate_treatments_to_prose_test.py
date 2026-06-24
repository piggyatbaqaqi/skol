"""Tests for the pure helper in migrate_treatments_to_prose.

Covers `rename_experiment_databases` only — the CouchDB-touching
`rewrite_documents` and `migrate` are exercised end-to-end against
a real database when run by hand.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from migrate_treatments_to_prose import rename_experiment_databases


class TestRenameExperimentDatabases(unittest.TestCase):

    def test_legacy_key_renamed_to_canonical(self):
        """``treatments`` → ``treatments_prose``, no other changes."""
        doc = {
            '_id': 'production_v3_hand',
            'databases': {
                'ingest': 'skol_dev',
                'treatments': 'skol_treatments_v3_dev',
            },
        }
        new_doc, changed = rename_experiment_databases(doc)
        self.assertTrue(changed)
        self.assertEqual(
            new_doc['databases'],
            {
                'ingest': 'skol_dev',
                'treatments_prose': 'skol_treatments_v3_dev',
            },
        )

    def test_full_pair_renamed_to_structured(self):
        """``treatments_full`` → ``treatments_structured``."""
        doc = {
            'databases': {
                'treatments_full': 'skol_treatments_full_v3_dev',
            },
        }
        new_doc, changed = rename_experiment_databases(doc)
        self.assertTrue(changed)
        self.assertEqual(
            new_doc['databases']['treatments_structured'],
            'skol_treatments_full_v3_dev',
        )
        self.assertNotIn('treatments_full', new_doc['databases'])

    def test_both_pairs_renamed_together(self):
        """Real production_v4-style: both keys present, both rename."""
        doc = {
            'databases': {
                'ingest': 'skol_dev',
                'treatments': 'skol_treatments_v3_dev',
                'treatments_full': 'skol_treatments_full_v3_dev',
            },
        }
        new_doc, changed = rename_experiment_databases(doc)
        self.assertTrue(changed)
        self.assertEqual(
            new_doc['databases'],
            {
                'ingest': 'skol_dev',
                'treatments_prose': 'skol_treatments_v3_dev',
                'treatments_structured': 'skol_treatments_full_v3_dev',
            },
        )

    def test_idempotent_on_already_migrated(self):
        """Doc with only canonical keys is a no-op (changed=False)."""
        doc = {
            'databases': {
                'ingest': 'skol_dev',
                'treatments_prose':
                    'skol_exp_production_v4_02_00_treatments_prose',
                'treatments_structured':
                    'skol_exp_production_v4_03_00_treatments_structured',
            },
        }
        new_doc, changed = rename_experiment_databases(doc)
        self.assertFalse(changed)
        self.assertIs(new_doc, doc)

    def test_both_legacy_and_canonical_same_value_drops_legacy(self):
        """If a doc somehow has both keys pointing at the same DB
        name, the legacy key is dropped without complaint."""
        doc = {
            'databases': {
                'treatments': 'skol_treatments_v3_dev',
                'treatments_prose': 'skol_treatments_v3_dev',
            },
        }
        new_doc, changed = rename_experiment_databases(doc)
        self.assertTrue(changed)
        self.assertEqual(
            new_doc['databases'],
            {'treatments_prose': 'skol_treatments_v3_dev'},
        )

    def test_both_legacy_and_canonical_differing_values_raises(self):
        """Different values for legacy vs canonical is data divergence
        that needs a human — silently picking one would lose data."""
        doc = {
            '_id': 'production_v4',
            'databases': {
                'treatments': 'skol_treatments_v3_dev',
                'treatments_prose':
                    'skol_exp_production_v4_02_00_treatments_prose',
            },
        }
        with self.assertRaises(ValueError) as cm:
            rename_experiment_databases(doc)
        msg = str(cm.exception)
        self.assertIn('production_v4', msg)
        self.assertIn('treatments', msg)
        self.assertIn('treatments_prose', msg)

    def test_doc_without_databases_block_unchanged(self):
        doc = {'_id': 'misc', 'notes': 'hello'}
        new_doc, changed = rename_experiment_databases(doc)
        self.assertFalse(changed)
        self.assertIs(new_doc, doc)

    def test_doc_with_non_dict_databases_unchanged(self):
        """A doc with ``databases: None`` or a list shouldn't crash."""
        doc = {'databases': None}
        new_doc, changed = rename_experiment_databases(doc)
        self.assertFalse(changed)

        doc2 = {'databases': []}
        new_doc2, changed2 = rename_experiment_databases(doc2)
        self.assertFalse(changed2)

    def test_unrelated_keys_preserved(self):
        """``taxa`` / ``taxa_full`` / ``ingest`` / ``annotations`` etc.
        must not be touched — this script targets exactly the
        mid-migration `treatments` and `treatments_full` keys."""
        doc = {
            'databases': {
                'ingest': 'skol_dev',
                'taxa': 'skol_taxa_dev',
                'taxa_full': 'skol_taxa_full_dev',
                'annotations': 'skol_exp_foo_ann',
                'treatments': 'skol_treatments_v3_dev',
            },
        }
        new_doc, changed = rename_experiment_databases(doc)
        self.assertTrue(changed)
        # All non-target keys preserved verbatim
        self.assertEqual(new_doc['databases']['ingest'], 'skol_dev')
        self.assertEqual(new_doc['databases']['taxa'], 'skol_taxa_dev')
        self.assertEqual(new_doc['databases']['taxa_full'], 'skol_taxa_full_dev')
        self.assertEqual(
            new_doc['databases']['annotations'], 'skol_exp_foo_ann',
        )
        # treatments renamed
        self.assertEqual(
            new_doc['databases']['treatments_prose'], 'skol_treatments_v3_dev',
        )
        self.assertNotIn('treatments', new_doc['databases'])

    def test_does_not_mutate_input(self):
        """Pure helper: caller's doc is unchanged."""
        doc = {
            'databases': {
                'treatments': 'skol_treatments_v3_dev',
            },
        }
        rename_experiment_databases(doc)
        self.assertEqual(
            doc, {'databases': {'treatments': 'skol_treatments_v3_dev'}},
        )


if __name__ == '__main__':
    unittest.main()
