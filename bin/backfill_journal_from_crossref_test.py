"""Tests for backfill_journal_from_crossref helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_journal_from_crossref import (
    extract_journal_from_crossref_work,
    needs_backfill,
)


class TestExtractJournalFromCrossrefWork(unittest.TestCase):
    """Pull the canonical journal name out of a Crossref ``works`` reply."""

    def test_container_title_with_one_entry(self):
        work = {'container-title': ['Mycotaxon']}
        self.assertEqual(extract_journal_from_crossref_work(work), 'Mycotaxon')

    def test_container_title_first_wins(self):
        """Some works have both long-form and short-form in
        ``container-title``; mirror the existing
        ``ingestors/crossref.py:221`` convention of taking the first."""
        work = {'container-title': [
            'Persoonia - Molecular Phylogeny and Evolution of Fungi',
            'Persoonia',
        ]}
        self.assertEqual(
            extract_journal_from_crossref_work(work),
            'Persoonia - Molecular Phylogeny and Evolution of Fungi',
        )

    def test_empty_container_title_returns_none(self):
        self.assertIsNone(extract_journal_from_crossref_work({'container-title': []}))

    def test_missing_container_title_returns_none(self):
        self.assertIsNone(extract_journal_from_crossref_work({}))

    def test_empty_string_treated_as_missing(self):
        self.assertIsNone(extract_journal_from_crossref_work({'container-title': ['']}))

    def test_whitespace_only_treated_as_missing(self):
        self.assertIsNone(extract_journal_from_crossref_work({'container-title': ['   \n']}))

    def test_strips_whitespace(self):
        self.assertEqual(
            extract_journal_from_crossref_work(
                {'container-title': ['  Mycotaxon  ']},
            ),
            'Mycotaxon',
        )

    def test_non_string_entry_returns_none(self):
        """Defensive: Crossref doesn't usually do this, but a non-string
        entry should not crash the caller."""
        self.assertIsNone(extract_journal_from_crossref_work({'container-title': [None]}))


class TestNeedsBackfill(unittest.TestCase):
    """Decide whether a skol_dev doc is eligible for the Crossref lookup:
    must have a DOI and must not already carry a journal."""

    def test_no_doi_skipped(self):
        self.assertFalse(needs_backfill({'_id': 'x'}))
        self.assertFalse(needs_backfill({'_id': 'x', 'doi': None}))
        self.assertFalse(needs_backfill({'_id': 'x', 'doi': ''}))

    def test_has_journal_skipped(self):
        self.assertFalse(needs_backfill({'doi': '10.x', 'journal': 'Mycotaxon'}))

    def test_empty_journal_eligible(self):
        """An empty-string journal counts as missing (some Crossref
        ingest paths fill the field with '' rather than leaving it
        absent)."""
        self.assertTrue(needs_backfill({'doi': '10.x', 'journal': ''}))
        self.assertTrue(needs_backfill({'doi': '10.x', 'journal': None}))

    def test_missing_journal_with_doi_eligible(self):
        self.assertTrue(needs_backfill({'doi': '10.1234/foo'}))


if __name__ == '__main__':
    unittest.main()
