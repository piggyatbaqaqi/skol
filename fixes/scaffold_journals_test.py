"""Tests for scaffold_journals pure helpers.

The script's job is to scaffold a draft ``JOURNALS`` dict by
walking the current ``SOURCES`` table and (optionally) enriching
each entry with Crossref ``/journals/{issn}`` lookups.  The
network-touching part is exercised by hand; this file pins the
pure helpers that decide slug names, strip publisher tags from
compound journal names, and shape the JournalEntry record.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scaffold_journals import (  # type: ignore[import]
    collect_unique_journals,
    extract_journal_fields_from_crossref,
    infer_slug_from_journal_name,
    slugify,
    strip_publisher_suffix,
)


class TestSlugify(unittest.TestCase):
    """Display name → URL-safe slug.  Lowercase, hyphens for spaces,
    punctuation removed.  The slug is the JOURNALS primary key, so
    these are *stable* identifiers — they don't change when the
    display name changes."""

    def test_simple_word(self):
        self.assertEqual(slugify('Sydowia'), 'sydowia')
        self.assertEqual(slugify('Mycotaxon'), 'mycotaxon')

    def test_multi_word(self):
        self.assertEqual(
            slugify('Journal of Fungi'),
            'journal-of-fungi',
        )

    def test_comma_punctuation(self):
        """``Cryptogamie, Mycologie`` — internal comma becomes the
        same separator as space."""
        self.assertEqual(
            slugify('Cryptogamie, Mycologie'),
            'cryptogamie-mycologie',
        )

    def test_colon_subtitle(self):
        """A colon-separated subtitle becomes part of the slug
        (we keep the full name to disambiguate, e.g.
        ``Mycology: An International Journal on Fungal Biology``
        from a generic ``Mycology``)."""
        self.assertEqual(
            slugify('Mycology: An International Journal on Fungal Biology'),
            'mycology-an-international-journal-on-fungal-biology',
        )

    def test_long_name(self):
        self.assertEqual(
            slugify('Annals of the Missouri Botanical Garden'),
            'annals-of-the-missouri-botanical-garden',
        )

    def test_idempotent_on_existing_slug(self):
        """A slug-shaped input round-trips unchanged.  Useful if a
        caller passes a slug expecting normalisation."""
        self.assertEqual(slugify('journal-of-fungi'), 'journal-of-fungi')

    def test_trims_whitespace(self):
        self.assertEqual(slugify('  Sydowia  '), 'sydowia')

    def test_collapses_repeated_separators(self):
        """Multiple spaces or punctuation runs collapse to a single
        hyphen — no ``foo--bar`` slugs."""
        self.assertEqual(slugify('Foo,  Bar'), 'foo-bar')

    def test_ampersand_handled(self):
        """``Journal of Fungi & Friends`` → ``journal-of-fungi-friends``."""
        self.assertEqual(
            slugify('Journal of Fungi & Friends'),
            'journal-of-fungi-friends',
        )


class TestStripPublisherSuffix(unittest.TestCase):
    """Strip ``" (Publisher)"`` parenthetical suffixes that current
    SOURCES uses to disambiguate the same journal ingested via
    multiple sources.  Idempotent and conservative — leaves a
    name alone if no recognised suffix is present."""

    def test_pmc_suffix(self):
        self.assertEqual(
            strip_publisher_suffix('Journal of Fungi (PMC)'),
            'Journal of Fungi',
        )
        self.assertEqual(
            strip_publisher_suffix('Mycology (PMC)'),
            'Mycology',
        )

    def test_taylor_francis_suffix(self):
        self.assertEqual(
            strip_publisher_suffix(
                'Mycology: An International Journal on Fungal Biology'
                ' (Taylor & Francis)',
            ),
            'Mycology: An International Journal on Fungal Biology',
        )

    def test_internet_archive_suffix(self):
        self.assertEqual(
            strip_publisher_suffix('Sydowia (Internet Archive)'),
            'Sydowia',
        )

    def test_no_suffix_passthrough(self):
        """A name with no recognised parenthetical is returned
        unchanged."""
        self.assertEqual(
            strip_publisher_suffix('Mycologia'),
            'Mycologia',
        )

    def test_parenthetical_that_isnt_a_publisher_preserved(self):
        """Only the publisher / source-name parentheticals get
        stripped; other parentheticals (rare) stay."""
        self.assertEqual(
            strip_publisher_suffix('Mycotaxon (n.s.)'),
            'Mycotaxon (n.s.)',
        )

    def test_idempotent(self):
        out = strip_publisher_suffix('Journal of Fungi (PMC)')
        self.assertEqual(strip_publisher_suffix(out), 'Journal of Fungi')

    def test_trims_whitespace_around_result(self):
        self.assertEqual(
            strip_publisher_suffix('  Mycology  (PMC)  '),
            'Mycology',
        )


class TestInferSlugFromJournalName(unittest.TestCase):
    """``infer_slug_from_journal_name`` composes
    ``strip_publisher_suffix`` and ``slugify`` — so a compound
    SOURCES entry like ``"Journal of Fungi (PMC)"`` resolves to
    the same slug ``journal-of-fungi`` as a clean
    ``"Journal of Fungi"``.  This is what causes multi-source
    journals to dedupe into one ``JOURNALS`` row."""

    def test_compound_pmc(self):
        self.assertEqual(
            infer_slug_from_journal_name('Journal of Fungi (PMC)'),
            'journal-of-fungi',
        )
        self.assertEqual(
            infer_slug_from_journal_name('Journal of Fungi'),
            'journal-of-fungi',
        )

    def test_compound_taylor_francis(self):
        self.assertEqual(
            infer_slug_from_journal_name(
                'Mycology: An International Journal on Fungal Biology'
                ' (Taylor & Francis)',
            ),
            'mycology-an-international-journal-on-fungal-biology',
        )


class TestCollectUniqueJournals(unittest.TestCase):
    """``collect_unique_journals(sources_dict)`` walks SOURCES and
    returns ``{slug: [source_keys_pointing_at_it]}``.  This is the
    dedup view that drives the scaffold output."""

    def test_dedupes_multi_source_journal(self):
        """Persoonia has multiple SOURCES entries; all collapse
        under a single ``persoonia`` slug."""
        sources = {
            'persoonia-pmc': {'journal': 'Persoonia (PMC)'},
            'persoonia-rss': {'journal': 'Persoonia'},
            'persoonia':     {'journal': 'Persoonia'},
        }
        result = collect_unique_journals(sources)
        self.assertEqual(
            set(result['persoonia']),
            {'persoonia-pmc', 'persoonia-rss', 'persoonia'},
        )

    def test_single_source_journal_one_key(self):
        sources = {'mycotaxon-rss': {'journal': 'Mycotaxon'}}
        result = collect_unique_journals(sources)
        self.assertEqual(result['mycotaxon'], ['mycotaxon-rss'])

    def test_sources_without_journal_field_skipped(self):
        """Local-mirror entries (mykoweb-caf, etc.) may have
        ``journal: None`` — skip them in the journal scaffold."""
        sources = {
            'mycotaxon-rss': {'journal': 'Mycotaxon'},
            'mykoweb-caf':   {'journal': None},
            'mykoweb-misc':  {},
        }
        result = collect_unique_journals(sources)
        self.assertEqual(set(result.keys()), {'mycotaxon'})

    def test_returns_sorted_source_keys(self):
        """Source keys per slug are returned in sorted order so the
        scaffold output diffs cleanly across re-runs."""
        sources = {
            'zzz-source':  {'journal': 'Mycology'},
            'aaa-source':  {'journal': 'Mycology'},
            'mmm-source':  {'journal': 'Mycology'},
        }
        result = collect_unique_journals(sources)
        self.assertEqual(result['mycology'],
                         ['aaa-source', 'mmm-source', 'zzz-source'])


class TestExtractJournalFieldsFromCrossref(unittest.TestCase):
    """``extract_journal_fields_from_crossref(response)`` pulls
    title / ISSN(s) / publisher / abbrev out of a Crossref
    ``/journals/{issn}`` reply.  Returns a dict with the
    JournalEntry-shaped fields the scaffold can drop straight into
    its draft."""

    def test_full_response(self):
        """A complete Crossref reply has all four fields."""
        msg = {
            'title':     'Persoonia',
            'publisher': 'Naturalis Biodiversity Center',
            'ISSN':      ['0031-5850', '1878-9080'],
            'short-container-title': ['Persoonia'],
        }
        out = extract_journal_fields_from_crossref(msg)
        self.assertEqual(out['name'], 'Persoonia')
        self.assertEqual(out['publisher'], 'Naturalis Biodiversity Center')
        # The first ISSN is the print one; the rest are e-versions
        # by Crossref convention.
        self.assertEqual(out['issn'], '0031-5850')
        self.assertEqual(out['eissn'], '1878-9080')

    def test_single_issn_no_eissn(self):
        """Older or print-only journals have a single ISSN."""
        msg = {
            'title':     'Sydowia',
            'publisher': 'Verlag Ferdinand Berger & Söhne',
            'ISSN':      ['0082-0598'],
        }
        out = extract_journal_fields_from_crossref(msg)
        self.assertEqual(out['issn'], '0082-0598')
        self.assertNotIn('eissn', out)

    def test_missing_publisher_omitted(self):
        msg = {'title': 'Sydowia', 'ISSN': ['0082-0598']}
        out = extract_journal_fields_from_crossref(msg)
        self.assertNotIn('publisher', out)
        self.assertEqual(out['name'], 'Sydowia')

    def test_empty_response_returns_empty_record(self):
        """Crossref 404 (no body parsed): defensive — produces an
        empty record rather than crashing.  The scaffold inserts
        a ``# TODO`` placeholder when fields are missing."""
        self.assertEqual(extract_journal_fields_from_crossref({}), {})

    def test_title_whitespace_normalised(self):
        msg = {'title': '  Persoonia  ', 'ISSN': ['0031-5850']}
        out = extract_journal_fields_from_crossref(msg)
        self.assertEqual(out['name'], 'Persoonia')


if __name__ == '__main__':
    unittest.main()
