"""Tests for backfill_pmc_urls pure helpers.

One-shot script that walks skol_dev for docs with ``pmcid`` set
but missing ``pdf_url`` / ``xml_url`` (everything ingested before
the PMC-URL fix) and populates the URL fields from the pmcid.
No network — derived from ``PMC_ARTICLE_URL_TEMPLATE`` and the
OAI-PMH GetRecord URL pattern.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_pmc_urls import compute_pmc_url_update  # type: ignore[import]


class TestComputePmcUrlUpdate(unittest.TestCase):
    """``compute_pmc_url_update(doc)`` returns the dict of URL
    fields that would be added to ``doc``, or an empty dict if the
    doc isn't a PMC backfill candidate or is already at the target
    state.  Idempotent."""

    def test_virgin_pmc_doc_gets_both_urls(self):
        """A doc with ``pmcid`` but no URL fields gets both."""
        update = compute_pmc_url_update({
            '_id': 'doc1', 'pmcid': '1234567', 'source': 'pmc',
        })
        self.assertEqual(
            update['pdf_url'],
            'https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/',
        )
        self.assertIn('verb=GetRecord', update['xml_url'])
        self.assertIn('oai:pubmedcentral.nih.gov:1234567',
                      update['xml_url'])

    def test_doc_without_pmcid_no_update(self):
        """A non-PMC doc isn't this script's domain."""
        self.assertEqual(
            compute_pmc_url_update({'_id': 'x', 'pdf_url': '...'}),
            {},
        )

    def test_doc_with_both_urls_already_set_no_update(self):
        """Idempotent — re-running is a no-op once URLs are populated."""
        doc = {
            'pmcid':   '1234567',
            'pdf_url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/',
            'xml_url': 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/'
                       '?verb=GetRecord'
                       '&identifier=oai:pubmedcentral.nih.gov:1234567'
                       '&metadataPrefix=pmc',
        }
        self.assertEqual(compute_pmc_url_update(doc), {})

    def test_doc_with_only_pdf_url_gets_xml_url(self):
        """Partial state — only the missing field comes back."""
        update = compute_pmc_url_update({
            'pmcid':   '1234567',
            'pdf_url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/',
        })
        self.assertNotIn('pdf_url', update)
        self.assertIn('xml_url', update)

    def test_doc_with_only_xml_url_gets_pdf_url(self):
        update = compute_pmc_url_update({
            'pmcid':   '1234567',
            'xml_url': 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/'
                       '?verb=GetRecord'
                       '&identifier=oai:pubmedcentral.nih.gov:1234567'
                       '&metadataPrefix=pmc',
        })
        self.assertIn('pdf_url', update)
        self.assertNotIn('xml_url', update)

    def test_doc_with_pmcid_and_empty_string_url_treated_as_missing(self):
        """``pdf_url=''`` (empty string) counts as missing, not
        as-set.  Defensive — some old ingests may have written
        empties."""
        update = compute_pmc_url_update({
            'pmcid':   '1234567',
            'pdf_url': '',
            'xml_url': '',
        })
        self.assertIn('pdf_url', update)
        self.assertIn('xml_url', update)


if __name__ == '__main__':
    unittest.main()
