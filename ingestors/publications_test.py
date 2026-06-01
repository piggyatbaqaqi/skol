"""Tests for ingestors/publications.py — phase 1B of the JOURNALS+SOURCES
consolidation (see docs/publications_metadata_consolidation.md).

Covers:
- structural invariants on the JOURNALS dict and the SOURCES → JOURNALS
  foreign-key integrity;
- the new classmethods (``get_journal``, ``find_journal_by_issn`` /
  _by_doi / _by_isbn);
- back-compat on the existing classmethods (``get_by_journal``,
  ``normalize_journal_name``) after the SOURCES rewrite.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.publications import (  # type: ignore[import]
    PublicationRegistry,
)


SLUG_RE = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')


class TestJournalsRegistryShape(unittest.TestCase):
    """Structural invariants on the new JOURNALS dict and the
    SOURCES → JOURNALS foreign-key integrity."""

    def test_journals_dict_populated(self):
        """Phase-1B scaffold produces ~20 entries; allow for growth
        but fail loud if someone empties it."""
        self.assertGreater(len(PublicationRegistry.JOURNALS), 15)

    def test_every_key_is_a_slug(self):
        """JOURNALS keys are stable opaque slugs — lowercase,
        hyphen-separated, no spaces / punctuation / uppercase."""
        for slug in PublicationRegistry.JOURNALS:
            with self.subTest(slug=slug):
                self.assertRegex(slug, SLUG_RE)

    def test_every_entry_has_a_name(self):
        """``name`` is the only required field on JournalEntry."""
        for slug, entry in PublicationRegistry.JOURNALS.items():
            with self.subTest(slug=slug):
                self.assertIn('name', entry)
                self.assertIsInstance(entry['name'], str)
                self.assertTrue(entry['name'].strip())

    def test_sources_journal_is_an_fk_into_journals(self):
        """Every ``SOURCES[*].journal`` value, when set, must be a
        key in JOURNALS — that's the FK invariant the refactor
        introduces.  Local-mirror entries with ``journal=None``
        (mykoweb-caf etc.) are exempt."""
        slugs = set(PublicationRegistry.JOURNALS.keys())
        for source_key, cfg in PublicationRegistry.SOURCES.items():
            j = cfg.get('journal')
            if not j:
                continue
            with self.subTest(source_key=source_key, journal=j):
                self.assertIn(
                    j, slugs,
                    f'SOURCES[{source_key!r}].journal={j!r} is not a '
                    f'JOURNALS slug',
                )


class TestGetJournal(unittest.TestCase):
    """``get_journal(slug_or_name)`` resolves both ways — direct
    slug lookup OR display-name lookup (case-sensitive on the
    name; aliases come via normalize_journal_name)."""

    def test_lookup_by_slug(self):
        entry = PublicationRegistry.get_journal('sydowia')
        assert entry is not None
        self.assertEqual(entry['name'], 'Sydowia')

    def test_lookup_by_display_name(self):
        entry = PublicationRegistry.get_journal('Sydowia')
        assert entry is not None
        self.assertEqual(entry['name'], 'Sydowia')

    def test_lookup_by_legacy_alias_resolves(self):
        """``'Sydowia Beih.'`` is in JOURNAL_NAME_ALIASES → ``'Sydowia'``
        for phase 1, so get_journal must still find the right
        entry through normalize_journal_name's alias resolution."""
        entry = PublicationRegistry.get_journal('Sydowia Beih.')
        assert entry is not None
        self.assertEqual(entry['name'], 'Sydowia')

    def test_unknown_returns_none(self):
        self.assertIsNone(
            PublicationRegistry.get_journal('does-not-exist'),
        )
        self.assertIsNone(
            PublicationRegistry.get_journal('Nonexistent Journal'),
        )


class TestFindJournalByIssn(unittest.TestCase):
    """``find_journal_by_issn(issn)`` scans JOURNALS for an entry
    whose ``issn`` or ``eissn`` matches; returns the slug.  This
    is what surfaces journal-less skol_dev docs under the right
    Sources-page bucket once Phase 4 wires it into
    ``resolve_source_name``."""

    def test_issn_primary_match(self):
        """Sydowia's ISSN matches its ``issn`` field."""
        self.assertEqual(
            PublicationRegistry.find_journal_by_issn('0082-0598'),
            'sydowia',
        )

    def test_eissn_match(self):
        """Persoonia's eissn matches via the ``eissn`` field."""
        self.assertEqual(
            PublicationRegistry.find_journal_by_issn('1878-9080'),
            'persoonia',
        )

    def test_unknown_issn_returns_none(self):
        self.assertIsNone(
            PublicationRegistry.find_journal_by_issn('9999-9999'),
        )

    def test_empty_or_none_returns_none(self):
        self.assertIsNone(PublicationRegistry.find_journal_by_issn(''))
        self.assertIsNone(PublicationRegistry.find_journal_by_issn(None))


class TestFindJournalByDoiAndIsbn(unittest.TestCase):
    """The DOI / ISBN attribute scanners follow the same shape as
    ``find_journal_by_issn``.  Phase 1 doesn't populate any
    journal-DOIs or ISBNs (Crossref doesn't return journal-DOIs
    and none of the current 20 journals are book-shaped), so the
    main coverage here is "doesn't crash on absent data" and
    "returns None for unknown values"."""

    def test_unknown_doi_returns_none(self):
        self.assertIsNone(
            PublicationRegistry.find_journal_by_doi('10.9999/nope'),
        )

    def test_unknown_isbn_returns_none(self):
        self.assertIsNone(
            PublicationRegistry.find_journal_by_isbn(
                '978-0-00-000000-0',
            ),
        )

    def test_empty_doi_returns_none(self):
        self.assertIsNone(PublicationRegistry.find_journal_by_doi(''))
        self.assertIsNone(PublicationRegistry.find_journal_by_doi(None))


class TestNormalizeJournalNameStripsPublisherSuffix(unittest.TestCase):
    """``normalize_journal_name`` gains a step: strip a known
    publisher-disambiguating parenthetical suffix
    (``" (PMC)"``, etc.) BEFORE consulting the legacy
    JOURNAL_NAME_ALIASES dict.  Legacy alias resolution still
    works for the entries already in that dict (phase 3 migrates
    them into JOURNALS[*].aliases)."""

    def test_pmc_suffix_stripped(self):
        self.assertEqual(
            PublicationRegistry.normalize_journal_name(
                'Journal of Fungi (PMC)',
            ),
            'Journal of Fungi',
        )

    def test_legacy_alias_still_resolves(self):
        """``'Persoonia - Molecular Phylogeny and Evolution of Fungi'``
        is in JOURNAL_NAME_ALIASES → ``'Persoonia'``."""
        self.assertEqual(
            PublicationRegistry.normalize_journal_name(
                'Persoonia - Molecular Phylogeny and Evolution of Fungi',
            ),
            'Persoonia',
        )

    def test_compound_name_with_alias(self):
        """The strip-suffix step runs first, then alias resolution
        — order matters only when both apply.  Sydowia Beih.
        becomes Sydowia (alias); not a publisher suffix case."""
        self.assertEqual(
            PublicationRegistry.normalize_journal_name('Sydowia Beih.'),
            'Sydowia',
        )

    def test_unknown_name_passthrough(self):
        self.assertEqual(
            PublicationRegistry.normalize_journal_name('Mycotaxon'),
            'Mycotaxon',
        )


class TestGetByJournalBackCompat(unittest.TestCase):
    """``get_by_journal(journal_name)`` returned a SOURCES entry
    matching the display name before the refactor.  After the
    refactor it still does — but internally it resolves
    name → canonical → slug → SOURCES entry."""

    def test_canonical_name_still_works(self):
        cfg = PublicationRegistry.get_by_journal('Mycotaxon')
        assert cfg is not None
        # Multiple Mycotaxon SOURCES entries; any of them is fine.
        self.assertIn(cfg['key'], ('mycotaxon', 'mycotaxon-rss'))

    def test_compound_name_resolves_via_strip(self):
        """A caller passing the old compound ``'Journal of Fungi
        (PMC)'`` (e.g. from a skol_dev doc whose journal field
        hasn't been rewritten yet — that's phase 2's job) still
        gets back a usable SOURCES entry."""
        cfg = PublicationRegistry.get_by_journal('Journal of Fungi (PMC)')
        assert cfg is not None
        self.assertTrue(cfg['key'].startswith('jof'))

    def test_legacy_alias_resolves(self):
        cfg = PublicationRegistry.get_by_journal(
            'Persoonia - Molecular Phylogeny and Evolution of Fungi',
        )
        assert cfg is not None
        self.assertTrue(cfg['key'].startswith('persoonia'))

    def test_unknown_returns_none(self):
        self.assertIsNone(
            PublicationRegistry.get_by_journal('Nonexistent Journal'),
        )


if __name__ == '__main__':
    unittest.main()
