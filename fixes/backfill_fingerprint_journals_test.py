"""Tests for backfill_fingerprint_journals pure helper.

The script's job is to walk skol_dev for docs lacking a
``journal`` field and set one by URL- or DOI-fingerprint match
against JOURNALS entries' ``ingenta_path`` / ``doi`` fields.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_fingerprint_journals import (  # type: ignore[import]
    compute_fingerprint_update,
)


class TestComputeFingerprintUpdate(unittest.TestCase):
    """``compute_fingerprint_update(doc)`` returns ``{'journal':
    <canonical name>}`` when a fingerprint matches, ``{}`` else.
    Idempotent — never overwrites an existing journal."""

    def test_ingenta_persoonia_url(self):
        update = compute_fingerprint_update({
            'pdf_url': 'https://www.ingentaconnect.com/contentone/'
                       'wfbi/pimj/2007/00000019/00000002/art00008',
        })
        self.assertEqual(update, {'journal': 'Persoonia'})

    def test_ingenta_fuse_url(self):
        update = compute_fingerprint_update({
            'pdf_url': 'https://www.ingentaconnect.com/contentone/'
                       'wfbi/fuse/2023/00000011/00000001/art00001',
        })
        self.assertEqual(
            update,
            {'journal': 'Fungal Systematics and Evolution'},
        )

    def test_journal_doi_oajmms(self):
        """The 1 doi.org Unknown doc has ``doi='10.23880/oajmms'``
        matching the JOURNALS row's ``doi`` field."""
        update = compute_fingerprint_update({'doi': '10.23880/oajmms'})
        self.assertEqual(update['journal'],
                         'Open Access Journal of Mycology & '
                         'Mycological Sciences')

    def test_existing_journal_protected(self):
        """Doc already has a journal — skip even if a fingerprint
        would match."""
        update = compute_fingerprint_update({
            'journal': 'Some Other Journal',
            'pdf_url': 'https://www.ingentaconnect.com/contentone/'
                       'wfbi/pimj/2007/00000019/00000002/art00008',
        })
        self.assertEqual(update, {})

    def test_empty_string_journal_treated_as_missing(self):
        update = compute_fingerprint_update({
            'journal': '',
            'pdf_url': 'https://www.ingentaconnect.com/contentone/'
                       'wfbi/pimj/x',
        })
        self.assertEqual(update, {'journal': 'Persoonia'})

    def test_no_match_returns_empty(self):
        """Non-matching URL / non-matching DOI / no signals → no
        update."""
        self.assertEqual(
            compute_fingerprint_update({
                'pdf_url': 'https://example.com/x.pdf',
            }),
            {},
        )

    def test_no_pdf_url_or_doi_returns_empty(self):
        self.assertEqual(compute_fingerprint_update({}), {})

    def test_doi_takes_priority_over_url(self):
        """If both fingerprints would match different journals,
        DOI wins (more specific signal — identifies a journal
        directly via its own DOI)."""
        update = compute_fingerprint_update({
            'doi':     '10.23880/oajmms',
            'pdf_url': 'https://www.ingentaconnect.com/contentone/'
                       'wfbi/pimj/x',
        })
        self.assertIn(
            'Open Access Journal of Mycology',
            update['journal'],
        )


if __name__ == '__main__':
    unittest.main()
