"""Tests for backfill_journal helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_journal import (
    extract_journal_from_crossref_journal,
    extract_journal_from_crossref_work,
    needs_backfill,
    needs_issn_backfill,
    normalize_issn,
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


class TestExtractJournalFromCrossrefJournal(unittest.TestCase):
    """Pull the canonical journal name from a Crossref ``journals``
    reply (the ISSN-keyed endpoint, distinct from the ``works`` one)."""

    def test_title_present(self):
        msg = {'title': 'Studies in Mycology'}
        self.assertEqual(
            extract_journal_from_crossref_journal(msg),
            'Studies in Mycology',
        )

    def test_strips_whitespace(self):
        self.assertEqual(
            extract_journal_from_crossref_journal(
                {'title': '  Mycotaxon  '},
            ),
            'Mycotaxon',
        )

    def test_missing_title_returns_none(self):
        self.assertIsNone(extract_journal_from_crossref_journal({}))

    def test_empty_title_returns_none(self):
        self.assertIsNone(
            extract_journal_from_crossref_journal({'title': ''}),
        )
        self.assertIsNone(
            extract_journal_from_crossref_journal({'title': None}),
        )

    def test_whitespace_only_title_returns_none(self):
        self.assertIsNone(
            extract_journal_from_crossref_journal({'title': '   '}),
        )


class TestNormalizeIssn(unittest.TestCase):
    """Crossref expects ISSNs in the canonical ``NNNN-NNNN`` form;
    skol_dev has some malformed values (no hyphen, only 7 digits with
    a missing leading zero).  Normalize before sending."""

    def test_already_canonical(self):
        self.assertEqual(normalize_issn('0166-0616'), '0166-0616')

    def test_strips_whitespace(self):
        self.assertEqual(normalize_issn(' 0166-0616\n'), '0166-0616')

    def test_uppercase_x_check_digit(self):
        """The ISSN check digit can be 'X' — preserve case."""
        self.assertEqual(normalize_issn('1234-567X'), '1234-567X')
        self.assertEqual(normalize_issn('1234-567x'), '1234-567X')

    def test_inserts_missing_hyphen(self):
        """``10520368`` → ``1052-0368`` (Crossref needs the hyphen)."""
        self.assertEqual(normalize_issn('10520368'), '1052-0368')

    def test_pads_seven_digit_with_leading_zero(self):
        """A 7-digit value lost its leading zero; restore it before
        inserting the hyphen."""
        self.assertEqual(normalize_issn('1660616'), '0166-0616')

    def test_returns_none_on_garbage(self):
        self.assertIsNone(normalize_issn(''))
        self.assertIsNone(normalize_issn(None))
        self.assertIsNone(normalize_issn('not an issn'))
        self.assertIsNone(normalize_issn('123'))


class TestNeedsIssnBackfill(unittest.TestCase):
    """ISSN-pass eligibility: has an issn/eissn AND no journal.
    Docs with a DOI are handled by the work-API pass, so they're
    excluded here unless that pass failed to find a journal."""

    def test_has_journal_skipped(self):
        self.assertFalse(
            needs_issn_backfill({'issn': '0166-0616', 'journal': 'X'}),
        )

    def test_issn_no_journal_no_doi_eligible(self):
        self.assertTrue(needs_issn_backfill({'issn': '0166-0616'}))

    def test_eissn_no_journal_no_doi_eligible(self):
        """An ``eissn`` is acceptable too — Crossref's journals
        endpoint accepts either."""
        self.assertTrue(needs_issn_backfill({'eissn': '2154-8889'}))

    def test_no_issn_at_all_skipped(self):
        self.assertFalse(needs_issn_backfill({'_id': 'x'}))

    def test_doi_doc_falls_through_when_still_no_journal(self):
        """If the DOI pass left a doc with no journal (Crossref 404
        or no container-title), the ISSN pass should still try
        when an ISSN is available."""
        self.assertTrue(
            needs_issn_backfill({'doi': '10.x', 'issn': '0166-0616'}),
        )


if __name__ == '__main__':
    unittest.main()
