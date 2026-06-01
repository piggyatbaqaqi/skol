"""Tests for backfill_pmc_jats_metadata pure helpers.

The script's job is to walk skol_dev for PMC docs that have an
``article.xml`` attachment but are missing ``journal`` (and/or
``doi``), re-parse the attached JATS front matter, and backfill
the missing fields.  No network — just re-uses XML that's
already on disk.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_pmc_jats_metadata import (  # type: ignore[import]
    compute_jats_field_update,
)


class TestComputeJatsFieldUpdate(unittest.TestCase):
    """``compute_jats_field_update(doc, metadata)`` returns the dict
    of fields to write onto ``doc`` from a JATS metadata extraction.
    Only fields that are CURRENTLY missing on the doc AND PRESENT
    in the extraction get returned.  Empty dict means no-op."""

    def test_journal_added_when_doc_lacks_it(self):
        doc = {'pmcid': '123'}
        meta = {'journal': 'Persoonia', 'doi': '10.x/y'}
        update = compute_jats_field_update(doc, meta)
        self.assertEqual(update['journal'], 'Persoonia')
        self.assertEqual(update['doi'], '10.x/y')

    def test_does_not_overwrite_set_journal(self):
        """A doc that already has a journal value keeps it —
        we trust the existing value over a re-parse."""
        doc = {'pmcid': '123', 'journal': 'Already Set'}
        meta = {'journal': 'Different'}
        update = compute_jats_field_update(doc, meta)
        self.assertNotIn('journal', update)

    def test_does_not_overwrite_set_doi(self):
        doc = {'pmcid': '123', 'doi': '10.1234/already'}
        meta = {'doi': '10.5678/new'}
        update = compute_jats_field_update(doc, meta)
        self.assertNotIn('doi', update)

    def test_empty_strings_count_as_missing(self):
        """A doc carrying ``journal=''`` (some old ingest paths
        wrote empties) is treated as missing — the JATS value
        wins."""
        doc = {'pmcid': '123', 'journal': '', 'doi': '  '}
        meta = {'journal': 'Persoonia', 'doi': '10.x/y'}
        update = compute_jats_field_update(doc, meta)
        self.assertEqual(update['journal'], 'Persoonia')
        self.assertEqual(update['doi'], '10.x/y')

    def test_empty_metadata_no_update(self):
        """Extraction returned nothing useful — no update."""
        doc = {'pmcid': '123'}
        meta = {'journal': '', 'doi': ''}
        self.assertEqual(compute_jats_field_update(doc, meta), {})

    def test_only_missing_field_in_update(self):
        """Doc has DOI but no journal; metadata has both —
        only journal comes back."""
        doc = {'pmcid': '123', 'doi': '10.1234/foo'}
        meta = {'journal': 'Persoonia', 'doi': '10.5678/bar'}
        update = compute_jats_field_update(doc, meta)
        self.assertEqual(set(update.keys()), {'journal'})
        self.assertEqual(update['journal'], 'Persoonia')

    def test_normalizes_journal_name(self):
        """JATS returns the journal's long-form name (e.g.
        ``Persoonia - Molecular Phylogeny and Evolution of Fungi``);
        the helper normalises through PublicationRegistry so the
        doc gets the canonical short form (``Persoonia``)."""
        doc = {'pmcid': '123'}
        meta = {
            'journal':
                'Persoonia - Molecular Phylogeny and Evolution of Fungi',
        }
        update = compute_jats_field_update(doc, meta)
        self.assertEqual(update['journal'], 'Persoonia')


if __name__ == '__main__':
    unittest.main()
