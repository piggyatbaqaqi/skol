"""Tests for build_sources_stats helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_sources_stats import redis_key_for_experiment


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
