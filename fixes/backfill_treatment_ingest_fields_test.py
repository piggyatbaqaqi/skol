"""Tests for backfill_treatment_ingest_fields pure helpers.

The script walks a treatment database, looks up each treatment's
parent ingest doc in skol_dev, and copies missing identifying
fields (currently ``doi`` and ``xml_url``) into the treatment's
nested ``ingest`` map.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_treatment_ingest_fields import (  # type: ignore[import]
    compute_ingest_update,
    parent_doc_id,
)


class TestParentDocId(unittest.TestCase):
    """``parent_doc_id(treatment_doc)`` returns the skol_dev ingest
    doc's ``_id`` from the treatment's nested ``ingest`` map, or
    ``None`` when absent / malformed."""

    def test_normal_treatment(self):
        self.assertEqual(
            parent_doc_id({'ingest': {'_id': 'abc123'}}),
            'abc123',
        )

    def test_missing_ingest(self):
        self.assertIsNone(parent_doc_id({}))

    def test_ingest_not_a_dict(self):
        """Defensive: ``ingest`` value of wrong type doesn't crash."""
        self.assertIsNone(parent_doc_id({'ingest': None}))
        self.assertIsNone(parent_doc_id({'ingest': 'string'}))

    def test_ingest_missing_id(self):
        self.assertIsNone(parent_doc_id({'ingest': {'url': 'x'}}))


class TestComputeIngestUpdate(unittest.TestCase):
    """``compute_ingest_update(treatment, parent)`` returns the
    fields to write into the treatment's ``ingest`` map.  Empty
    dict means no-op.  Idempotent — re-running is a no-op once
    fields are populated."""

    _PARENT = {
        '_id':     'doc1',
        'doi':     '10.1234/x',
        'xml_url': 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?...',
    }

    def test_treatment_missing_both_fields_gets_both(self):
        treatment = {'ingest': {'_id': 'doc1', 'pdf_url': '...'}}
        update = compute_ingest_update(treatment, self._PARENT)
        self.assertEqual(update['doi'], '10.1234/x')
        self.assertEqual(
            update['xml_url'],
            'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?...',
        )

    def test_treatment_already_has_doi_no_overwrite(self):
        treatment = {
            'ingest': {'_id': 'doc1', 'doi': '10.5678/already-set'},
        }
        update = compute_ingest_update(treatment, self._PARENT)
        self.assertNotIn('doi', update)
        self.assertIn('xml_url', update)

    def test_treatment_already_has_xml_url_no_overwrite(self):
        treatment = {
            'ingest': {'_id': 'doc1', 'xml_url': 'https://already-set/'},
        }
        update = compute_ingest_update(treatment, self._PARENT)
        self.assertNotIn('xml_url', update)
        self.assertIn('doi', update)

    def test_parent_lacks_doi_no_update(self):
        """If the parent has no DOI, don't write anything for that
        field (no inventing values)."""
        treatment = {'ingest': {'_id': 'doc1'}}
        parent = {'_id': 'doc1', 'xml_url': 'https://x/'}
        update = compute_ingest_update(treatment, parent)
        self.assertNotIn('doi', update)
        self.assertEqual(update['xml_url'], 'https://x/')

    def test_parent_lacks_both_no_update(self):
        treatment = {'ingest': {'_id': 'doc1'}}
        parent = {'_id': 'doc1'}
        self.assertEqual(compute_ingest_update(treatment, parent), {})

    def test_empty_string_in_parent_counts_as_missing(self):
        """``''`` / ``None`` on the parent → don't write."""
        treatment = {'ingest': {'_id': 'doc1'}}
        parent = {'_id': 'doc1', 'doi': '', 'xml_url': None}
        self.assertEqual(compute_ingest_update(treatment, parent), {})

    def test_empty_string_in_treatment_treated_as_missing(self):
        """A treatment whose ingest.doi is ``''`` should still be
        updated from a real parent value — empties don't protect."""
        treatment = {'ingest': {'_id': 'doc1', 'doi': '', 'xml_url': ''}}
        update = compute_ingest_update(treatment, self._PARENT)
        self.assertEqual(update['doi'], '10.1234/x')
        self.assertIn('xml_url', update)

    def test_treatment_already_complete_idempotent(self):
        """Re-running on an up-to-date treatment is a no-op."""
        treatment = {
            'ingest': {
                '_id':     'doc1',
                'doi':     '10.1234/x',
                'xml_url': 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?...',
            },
        }
        self.assertEqual(
            compute_ingest_update(treatment, self._PARENT),
            {},
        )


if __name__ == '__main__':
    unittest.main()
