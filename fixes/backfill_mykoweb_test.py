"""Tests for backfill_mykoweb pure helpers.

Network-touching ``main()`` is exercised end-to-end against a real
skol_dev by hand with ``--dry-run``; no fixture for that here.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_mykoweb import (  # type: ignore[import]
    compute_field_update,
    is_mykoweb_literature,
    pdf_url_to_lookup_key,
)


# ---------------------------------------------------------------------------
# pdf_url_to_lookup_key — mykoweb URL → JSON-index key
# ---------------------------------------------------------------------------


class TestPdfUrlToLookupKey(unittest.TestCase):
    """The systematics_pdf_metadata.json keys are site-relative
    (e.g. ``systematics/literature/foo.pdf``).  Existing skol_dev
    docs have ``pdf_url`` set to the full mykoweb URL.  Bridge the
    two so the lookup works."""

    def test_literature_url(self):
        self.assertEqual(
            pdf_url_to_lookup_key(
                'https://mykoweb.com/systematics/literature/A Book.pdf'
            ),
            'systematics/literature/A Book.pdf',
        )

    def test_journals_url(self):
        """The same routine handles journals URLs too — they live
        under ``systematics/journals/<journal>/<file>.pdf``."""
        self.assertEqual(
            pdf_url_to_lookup_key(
                'https://mykoweb.com/systematics/journals/Mycotaxon/'
                'Mycotaxon v001n1.pdf'
            ),
            'systematics/journals/Mycotaxon/Mycotaxon v001n1.pdf',
        )

    def test_non_mykoweb_url_returns_none(self):
        """A PDF URL from somewhere else (Crossref, Pensoft, etc.)
        is not a mykoweb backfill candidate."""
        self.assertIsNone(pdf_url_to_lookup_key(
            'https://example.com/some.pdf',
        ))

    def test_empty_or_none_returns_none(self):
        self.assertIsNone(pdf_url_to_lookup_key(''))
        self.assertIsNone(pdf_url_to_lookup_key(None))


# ---------------------------------------------------------------------------
# is_mykoweb_literature — eligibility for the backfill pass
# ---------------------------------------------------------------------------


class TestIsMykowebLiterature(unittest.TestCase):
    """The LocalMykowebLiteratureIngestor stamps each doc with
    ``meta = {'source': 'mykoweb', 'type': 'literature'}``.  The
    backfill targets exactly those rows — other mykoweb docs
    (journals, CAF, etc.) are handled separately or by the
    directory-encoded source-of-truth and don't need this pass."""

    def test_positive(self):
        doc = {'meta': {'source': 'mykoweb', 'type': 'literature'}}
        self.assertTrue(is_mykoweb_literature(doc))

    def test_wrong_type(self):
        doc = {'meta': {'source': 'mykoweb', 'type': 'journal'}}
        self.assertFalse(is_mykoweb_literature(doc))

    def test_wrong_source(self):
        doc = {'meta': {'source': 'crossref', 'type': 'literature'}}
        self.assertFalse(is_mykoweb_literature(doc))

    def test_missing_meta(self):
        """Defensive: an older doc without a ``meta`` block (or
        with a non-dict meta) is not eligible."""
        self.assertFalse(is_mykoweb_literature({}))
        self.assertFalse(is_mykoweb_literature({'meta': None}))
        self.assertFalse(is_mykoweb_literature({'meta': 'string'}))


# ---------------------------------------------------------------------------
# compute_field_update — what to write back to the doc
# ---------------------------------------------------------------------------


def _existing(**fields: Any) -> Dict[str, Any]:
    """Helper: an existing skol_dev doc with the pre-integration
    shape (filename-as-title / itemtype=book) plus the supplied
    overrides."""
    base: Dict[str, Any] = {
        'meta':     {'source': 'mykoweb', 'type': 'literature'},
        'pdf_url':  'https://mykoweb.com/systematics/literature/clavaria.pdf',
        'title':    'clavaria',
        'itemtype': 'book',
    }
    base.update(fields)
    return base


def _index_with_clavaria() -> Dict[str, Dict[str, Any]]:
    """Index containing the canonical journal-article test record."""
    return {
        'systematics/literature/clavaria.pdf': {
            'kind':            'journal_article',
            'title':           'The North American Species of Clavaria',
            'container_title': 'Annals of the Missouri Botanical Garden',
            'volume':          '9',
            'issue':           '1',
            'pages':           '1-78',
            'author':          'Burt, E.A.',
            'year':            1922,
            'source_html':     'systematics/literature.html',
        },
    }


class TestComputeFieldUpdate(unittest.TestCase):
    """The script merges *only* the fields that would actually
    change.  Returning an empty dict is the signal 'nothing to do
    for this doc' — drives idempotency, makes re-runs cheap."""

    def test_virgin_doc_full_update(self):
        """A pre-integration doc with no curated fields gets every
        field from the record applied."""
        update = compute_field_update(_existing(), _index_with_clavaria())
        self.assertEqual(update['itemtype'], 'article')
        self.assertEqual(update['journal'],
                         'Annals of the Missouri Botanical Garden')
        self.assertEqual(update['volume'], '9')
        self.assertEqual(update['issue'], '1')
        self.assertEqual(update['pages'], '1-78')
        self.assertEqual(update['title'],
                         'The North American Species of Clavaria')
        self.assertEqual(update['author'], 'Burt, E.A.')
        self.assertEqual(update['year'], 1922)

    def test_partial_overlap_only_diffs_returned(self):
        """If some fields already match (e.g. a previous partial
        backfill ran), only the *different* ones come back."""
        existing = _existing(
            itemtype='article',
            journal='Annals of the Missouri Botanical Garden',
            volume='9',
            issue='1',
            pages='1-78',
            year=1922,
            author='Burt, E.A.',
            # Title is still the old filename → must be updated
            title='clavaria',
        )
        update = compute_field_update(existing, _index_with_clavaria())
        self.assertEqual(set(update.keys()), {'title'})
        self.assertEqual(update['title'],
                         'The North American Species of Clavaria')

    def test_all_match_empty_update(self):
        """A doc already fully enriched produces no update — re-running
        the backfill must be a no-op."""
        existing = _existing(
            itemtype='article',
            journal='Annals of the Missouri Botanical Garden',
            volume='9',
            issue='1',
            pages='1-78',
            title='The North American Species of Clavaria',
            author='Burt, E.A.',
            year=1922,
        )
        update = compute_field_update(existing, _index_with_clavaria())
        self.assertEqual(update, {})

    def test_not_in_index_empty_update(self):
        """A doc whose pdf_url isn't in the metadata JSON (e.g. a
        PDF added after the last extraction run) gets no update —
        we don't invent fields."""
        existing = _existing(
            pdf_url='https://mykoweb.com/systematics/literature/missing.pdf',
        )
        update = compute_field_update(existing, _index_with_clavaria())
        self.assertEqual(update, {})

    def test_non_literature_record_empty_update(self):
        """If the index entry for this PDF has kind != journal_article /
        book (e.g. ``kind=key`` for a taxonomic-key PDF), no
        update."""
        existing = _existing(
            pdf_url='https://mykoweb.com/systematics/keys/foo.pdf',
        )
        index = {
            'systematics/keys/foo.pdf': {
                'kind':  'key',
                'title': 'Some key',
            },
        }
        update = compute_field_update(existing, index)
        self.assertEqual(update, {})

    def test_non_mykoweb_url_empty_update(self):
        """Defensive: a doc that somehow slipped through the
        eligibility filter with a non-mykoweb pdf_url returns no
        update (rather than crashing on the key lookup)."""
        existing = _existing(pdf_url='https://example.com/foo.pdf')
        update = compute_field_update(existing, _index_with_clavaria())
        self.assertEqual(update, {})


if __name__ == '__main__':
    unittest.main()
