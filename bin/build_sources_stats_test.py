"""Tests for build_sources_stats helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_sources_stats import (
    count_new_taxon_acts,
    count_sanctioned_markers,
    redis_key_for_experiment,
    resolve_source_name,
)


class TestCountNewTaxonActs(unittest.TestCase):
    """Detect nomenclatural-act markers in Treatment text."""

    def test_sp_nov_canonical(self):
        self.assertEqual(count_new_taxon_acts('Foo bar sp. nov.'), 1)

    def test_gen_nov(self):
        self.assertEqual(count_new_taxon_acts('Foonidae gen. nov.'), 1)

    def test_comb_nov(self):
        self.assertEqual(count_new_taxon_acts('Bar quux comb. nov.'), 1)

    def test_nom_nov(self):
        self.assertEqual(count_new_taxon_acts('Foo baz nom. nov.'), 1)

    def test_no_trailing_period(self):
        self.assertEqual(count_new_taxon_acts('Foo bar sp nov'), 1)

    def test_case_insensitive(self):
        self.assertEqual(count_new_taxon_acts('Foo bar SP. NOV.'), 1)

    def test_multiple_in_one_text(self):
        text = 'Foo sp. nov.  Bar comb. nov.  Baz sp. nov.'
        self.assertEqual(count_new_taxon_acts(text), 3)

    def test_no_match(self):
        self.assertEqual(count_new_taxon_acts('Foo bar Linnaeus 1753'), 0)

    def test_must_be_word_boundary(self):
        """``sp.novelty`` is not a nomenclatural act."""
        self.assertEqual(count_new_taxon_acts('sp.novelty in fungi'), 0)

    def test_empty_input(self):
        self.assertEqual(count_new_taxon_acts(''), 0)
        self.assertEqual(count_new_taxon_acts(None), 0)


class TestCountSanctionedMarkers(unittest.TestCase):
    """Detect Fries / Persoon sanctioning-author citations."""

    def test_fries_colon_fr(self):
        self.assertEqual(
            count_sanctioned_markers('Lentinus tigrinus : Fr.'),
            1,
        )

    def test_fries_paren(self):
        self.assertEqual(
            count_sanctioned_markers('Polyporus (Fr.) Murrill 1903'),
            1,
        )

    def test_persoon_colon(self):
        self.assertEqual(
            count_sanctioned_markers('Bovista plumbea : Pers.'),
            1,
        )

    def test_persoon_paren(self):
        self.assertEqual(
            count_sanctioned_markers('Some species (Pers.) Modern Author'),
            1,
        )

    def test_ex_fries(self):
        self.assertEqual(
            count_sanctioned_markers('Boletus edulis Bull. ex Fries 1821'),
            1,
        )

    def test_ex_persoon(self):
        self.assertEqual(
            count_sanctioned_markers('Some species ex Persoon 1801'),
            1,
        )

    def test_multiple_in_one_text(self):
        text = 'Lentinus tigrinus : Fr.  Bovista plumbea : Pers.'
        self.assertEqual(count_sanctioned_markers(text), 2)

    def test_no_match(self):
        self.assertEqual(
            count_sanctioned_markers('Amanita muscaria (L.) Lam.'),
            0,
        )

    def test_empty_input(self):
        self.assertEqual(count_sanctioned_markers(''), 0)
        self.assertEqual(count_sanctioned_markers(None), 0)


class TestRedisKeyForExperiment(unittest.TestCase):
    """``redis_key_for_experiment`` builds the per-experiment Redis key
    so the Django Ingestion Sources page can show experiment-scoped
    stats.  Default falls back to ``skol:sources:stats`` for the v1
    pipeline / anonymous users (back-compat with the existing cron job
    and Django fast-path Redis read)."""

    def test_no_experiment_uses_default_key(self):
        self.assertEqual(redis_key_for_experiment(None), 'skol:sources:stats')
        self.assertEqual(redis_key_for_experiment(''), 'skol:sources:stats')

    def test_named_experiment_appends_suffix(self):
        self.assertEqual(
            redis_key_for_experiment('production_v3_hand'),
            'skol:sources:stats:production_v3_hand',
        )

    def test_other_experiment_names(self):
        for name in (
            'production', 'jats_v1', 'production_v3_jats', 'production_v3_full',
        ):
            self.assertEqual(
                redis_key_for_experiment(name),
                f'skol:sources:stats:{name}',
            )


class TestResolveSourceName(unittest.TestCase):
    """Group docs by ``journal`` first; if the journal field is
    empty, fall back to identifying the source by other signals so
    docs don't collapse into a single opaque ``Unknown`` bucket
    when we know they came from mykoweb but lack a curated
    container_title."""

    def test_journal_wins(self):
        """Whenever the journal field is set, return it verbatim —
        the fallback logic only kicks in for empty journal."""
        self.assertEqual(
            resolve_source_name({'journal': 'Mycotaxon'}),
            'Mycotaxon',
        )

    def test_journal_set_with_mykoweb_url_still_wins(self):
        """A doc with both journal set AND mykoweb pdf_url stays
        on the journal — the mykoweb fallback only applies when
        journal is missing."""
        self.assertEqual(
            resolve_source_name({
                'journal': 'Mycotaxon',
                'pdf_url': 'https://mykoweb.com/systematics/x.pdf',
            }),
            'Mycotaxon',
        )

    def test_empty_journal_mykoweb_https_url_falls_back(self):
        self.assertEqual(
            resolve_source_name({
                'pdf_url': 'https://mykoweb.com/CAF/PDF/foo.pdf',
            }),
            'mykoweb',
        )

    def test_empty_journal_mykoweb_http_url_falls_back(self):
        """``http://`` (non-TLS) URLs are also mykoweb."""
        self.assertEqual(
            resolve_source_name({
                'pdf_url': 'http://mykoweb.com/misc/foo.pdf',
            }),
            'mykoweb',
        )

    def test_empty_journal_meta_source_mykoweb_falls_back(self):
        """No pdf_url, but meta.source identifies the doc as
        mykoweb-sourced."""
        self.assertEqual(
            resolve_source_name({
                'meta': {'source': 'mykoweb', 'type': 'literature'},
            }),
            'mykoweb',
        )

    def test_empty_journal_meta_source_mykoweb_caf_falls_back(self):
        """meta.source is one of the per-publication mykoweb-* keys
        (mykoweb-caf, mykoweb-crepidotus, etc.) — also mykoweb."""
        self.assertEqual(
            resolve_source_name({'meta': {'source': 'mykoweb-caf'}}),
            'mykoweb',
        )

    def test_empty_journal_archive_org_stays_unknown(self):
        """A non-mykoweb pdf_url (e.g. archive.org/Sydowia) doesn't
        match the fallback — falls through to Unknown."""
        self.assertEqual(
            resolve_source_name({
                'pdf_url': 'https://archive.org/download/sydowia_1925.pdf',
            }),
            'Unknown',
        )

    def test_empty_journal_no_signals_unknown(self):
        """A doc with no journal, no pdf_url, no meta — stays
        Unknown."""
        self.assertEqual(resolve_source_name({}), 'Unknown')

    def test_whitespace_only_journal_falls_through(self):
        """Empty / whitespace-only journal must trigger the fallback
        the same way a missing one does."""
        self.assertEqual(
            resolve_source_name({
                'journal': '   ',
                'pdf_url': 'https://mykoweb.com/x.pdf',
            }),
            'mykoweb',
        )


class TestResolveSourceNameIssnFallback(unittest.TestCase):
    """Phase-4 ISSN fallback: when ``journal`` is empty, consult
    ``PublicationRegistry.find_journal_by_issn`` before falling
    through to the mykoweb / Unknown buckets.  Surfaces the 249
    archive.org Sydowia docs under ``"Sydowia"`` instead of the
    Unknown bucket — they all carry ``issn=0082-0598``."""

    def test_empty_journal_issn_matches_known_journal(self):
        """ISSN 0082-0598 → Sydowia (the JOURNALS row's canonical
        ``name``)."""
        self.assertEqual(
            resolve_source_name({'issn': '0082-0598'}),
            'Sydowia',
        )

    def test_empty_journal_eissn_matches_known_journal(self):
        """eissn is consulted the same way as issn — Persoonia's
        electronic ISSN."""
        self.assertEqual(
            resolve_source_name({'eissn': '1878-9080'}),
            'Persoonia',
        )

    def test_archive_org_sydowia_doc_now_resolves(self):
        """The exact shape of the 249 archive.org Sydowia docs:
        no journal, pdf_url under archive.org, issn 0082-0598.
        Previously stayed in Unknown; phase 4 surfaces them."""
        self.assertEqual(
            resolve_source_name({
                'pdf_url': 'https://archive.org/download/sydowia_1925.pdf',
                'issn':    '0082-0598',
                'title':   'Annales Mycologici 1925: Vol 23 1-2',
            }),
            'Sydowia',
        )

    def test_issn_takes_priority_over_mykoweb_fallback(self):
        """When both signals are present, the ISSN (more specific
        — identifies a journal directly) wins over the mykoweb
        catch-all."""
        self.assertEqual(
            resolve_source_name({
                'issn':    '0082-0598',
                'pdf_url': 'https://mykoweb.com/x.pdf',
            }),
            'Sydowia',
        )

    def test_unknown_issn_falls_through_to_next_fallback(self):
        """An ISSN that isn't in JOURNALS doesn't short-circuit;
        the mykoweb / Unknown fallbacks still get a chance."""
        self.assertEqual(
            resolve_source_name({
                'issn':    '9999-9999',
                'pdf_url': 'https://mykoweb.com/x.pdf',
            }),
            'mykoweb',
        )
        self.assertEqual(
            resolve_source_name({'issn': '9999-9999'}),
            'Unknown',
        )


if __name__ == "__main__":
    unittest.main()
