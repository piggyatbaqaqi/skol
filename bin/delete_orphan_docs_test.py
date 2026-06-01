"""Tests for delete_orphan_docs pure helpers.

The script's job is to identify and delete skol_dev docs that have
no identifying fields at all — no ``journal``, no ``doi``, no
``issn``, no ``pdf_url``, no ``pmcid``.  These are stubs left by
partial / failed ingests with no path to recovery; the user is
OK with deleting them so re-scrape with current code can re-create
proper ones.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from delete_orphan_docs import is_orphan  # type: ignore[import]


class TestIsOrphan(unittest.TestCase):
    """A doc is an orphan when ALL of ``journal``, ``doi``,
    ``issn``, ``pdf_url``, ``pmcid`` are missing / empty.  Any
    one non-empty value protects the doc."""

    def test_completely_empty_doc_is_orphan(self):
        self.assertTrue(is_orphan(
            {'_id': 'x', 'source': 'pmc',
             'create_time': '2024-01-01'},
        ))

    def test_journal_set_protects(self):
        self.assertFalse(is_orphan(
            {'journal': 'Mycotaxon'},
        ))

    def test_doi_set_protects(self):
        self.assertFalse(is_orphan({'doi': '10.1234/foo'}))

    def test_issn_set_protects(self):
        self.assertFalse(is_orphan({'issn': '0093-4666'}))

    def test_pdf_url_set_protects(self):
        self.assertFalse(is_orphan(
            {'pdf_url': 'https://example.com/x.pdf'},
        ))

    def test_pmcid_set_protects(self):
        """pmcid alone is enough to recover URLs via
        bin/backfill_pmc_urls; not an orphan."""
        self.assertFalse(is_orphan({'pmcid': '1234567'}))

    def test_empty_string_fields_are_missing(self):
        """Empty / whitespace-only values count as missing."""
        self.assertTrue(is_orphan({
            'journal': '', 'doi': '   ', 'issn': None,
            'pdf_url': '', 'pmcid': '',
        }))

    def test_non_string_value_protects_only_if_truthy(self):
        """Defensive — a doc carrying ``issn=12345678`` (mistyped
        as int) still has identifying info."""
        self.assertFalse(is_orphan({'issn': 12345678}))
        self.assertTrue(is_orphan({'issn': 0}))


if __name__ == '__main__':
    unittest.main()
