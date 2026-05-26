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


if __name__ == '__main__':
    unittest.main()
