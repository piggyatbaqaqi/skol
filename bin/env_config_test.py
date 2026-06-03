"""Tests for bin/env_config.py.

Focused on the experiment-doc → config mapping introduced by Step 1.B
of the golden-v2 plan: ``databases.golden`` → ``golden_db_name`` and
``databases.golden_ann`` → ``golden_ann_db_name``.
"""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import (  # type: ignore[import]  # noqa: E402
    _apply_experiment,
    _parse_embedding_expire,
)


def _starter_config(**overrides: Any) -> Dict[str, Any]:
    """Return a config dict with v1 defaults for the golden keys.

    Mirrors the defaults the production code will set in `get_env_config()`
    once Step 1.B lands.  Tests only need to assert what
    ``_apply_experiment`` does on top of these starters.
    """
    config: Dict[str, Any] = {
        'golden_db_name':     'skol_golden',
        'golden_ann_db_name': 'skol_golden_ann_hand',
    }
    config.update(overrides)
    return config


class TestApplyExperimentGoldenMapping:
    """The new mapping rows pull databases.golden / databases.golden_ann
    out of the experiment doc and write them into the config dict."""

    def test_golden_field_propagates(self) -> None:
        config = _starter_config()
        exp = {'databases': {'golden': 'skol_golden_v2'}}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['golden_db_name'] == 'skol_golden_v2'

    def test_golden_ann_field_propagates(self) -> None:
        config = _starter_config()
        exp = {'databases': {'golden_ann': 'skol_golden_ann_hand_v2'}}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['golden_ann_db_name'] == 'skol_golden_ann_hand_v2'

    def test_both_fields_together(self) -> None:
        config = _starter_config()
        exp = {'databases': {
            'golden':     'skol_golden_v2',
            'golden_ann': 'skol_golden_ann_jats_v2',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['golden_db_name'] == 'skol_golden_v2'
        assert config['golden_ann_db_name'] == 'skol_golden_ann_jats_v2'

    def test_missing_fields_leave_defaults(self) -> None:
        """An experiment doc that omits the new fields must NOT clobber the
        starter defaults (v1 names).  This is the backward-compat guarantee."""
        config = _starter_config()
        exp = {'databases': {'ingest': 'skol_dev'}}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['golden_db_name'] == 'skol_golden'
        assert config['golden_ann_db_name'] == 'skol_golden_ann_hand'

    def test_empty_string_value_is_ignored(self) -> None:
        """databases.golden_ann == '' (explicit empty, as production carries
        for databases.annotations) should be treated as "no override"
        rather than overwriting the starter default with empty string."""
        config = _starter_config()
        exp = {'databases': {'golden_ann': ''}}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['golden_ann_db_name'] == 'skol_golden_ann_hand'

    def test_cli_explicit_keys_block_override(self) -> None:
        """If the user already passed --golden-db on the CLI, the experiment
        doc's value must NOT win."""
        config = _starter_config(golden_db_name='cli_value')
        exp = {'databases': {'golden': 'skol_golden_v2'}}
        _apply_experiment(
            config, exp, cli_explicit_keys={'golden_db_name'}
        )
        assert config['golden_db_name'] == 'cli_value'

    def test_other_databases_unaffected(self) -> None:
        """A no-op safety test: pre-existing mapping rows (ingest, training,
        treatments, etc.) keep working after the new rows are added."""
        config = _starter_config(
            ingest_db_name='default_ingest',
            training_database='default_training',
        )
        exp = {'databases': {
            'ingest':   'skol_dev',
            'training': 'skol_training_v2',
            'golden':   'skol_golden_v2',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['ingest_db_name'] == 'skol_dev'
        assert config['training_database'] == 'skol_training_v2'
        assert config['golden_db_name'] == 'skol_golden_v2'


# ---------------------------------------------------------------------------
# Post-Step-3 doc-field-name mapping (treatments / treatments_full)
# ---------------------------------------------------------------------------


class TestApplyExperimentTreatmentsMapping:
    """The Step-3 data migration renamed databases.taxa → databases.treatments
    in every experiment doc, but env_config kept looking for the old field
    name.  These tests pin the post-migration mapping rows in place."""

    def test_treatments_field_propagates(self) -> None:
        config = {'treatments_db_name': 'skol_treatments_dev'}
        exp = {'databases': {
            'treatments': 'skol_treatments_taxpub_v1_dev',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['treatments_db_name'] == 'skol_treatments_taxpub_v1_dev'

    def test_treatments_full_field_propagates(self) -> None:
        config = {'dest_db': 'skol_treatments_full_dev'}
        exp = {'databases': {
            'treatments_full': 'skol_exp_x_treatments_full',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['dest_db'] == 'skol_exp_x_treatments_full'

    def test_treatments_also_sets_source_db(self) -> None:
        """databases.treatments writes to BOTH treatments_db_name and
        source_db — same shape as the pre-rename ('taxa', [...]) row."""
        config = {
            'treatments_db_name': 'skol_treatments_dev',
            'source_db':          'skol_treatments_dev',
        }
        exp = {'databases': {'treatments': 'skol_treatments_my_exp'}}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['treatments_db_name'] == 'skol_treatments_my_exp'
        assert config['source_db'] == 'skol_treatments_my_exp'

    def test_legacy_taxa_field_still_accepted(self) -> None:
        """An unmigrated doc with the legacy databases.taxa field still
        works (backward-compat fallback)."""
        config = {'treatments_db_name': 'skol_treatments_dev'}
        exp = {'databases': {'taxa': 'skol_taxa_legacy_exp'}}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['treatments_db_name'] == 'skol_taxa_legacy_exp'

    def test_canonical_wins_when_both_present(self) -> None:
        """A doc carrying both new and old field names (mid-migration) lets
        the canonical 'treatments' value win — the mapping list orders
        canonical before fallback."""
        config = {'treatments_db_name': 'skol_treatments_dev'}
        exp = {'databases': {
            'treatments': 'new_value',
            'taxa':       'old_value',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['treatments_db_name'] == 'new_value'


class TestParseEmbeddingExpire:
    """Embedding TTLs default to *no* expiry — only set one when the
    caller explicitly asks for a positive integer.  This is the policy
    that fixes the v3_hand-embed silent-loss bug where the previous
    2-day default would expire embeddings before the next successful
    nightly refresh.
    """

    def test_unset_means_no_expiry(self) -> None:
        assert _parse_embedding_expire('') is None

    def test_whitespace_means_no_expiry(self) -> None:
        assert _parse_embedding_expire('   ') is None

    def test_literal_none_string_means_no_expiry(self) -> None:
        assert _parse_embedding_expire('None') is None
        assert _parse_embedding_expire('none') is None
        assert _parse_embedding_expire('NONE') is None

    def test_zero_means_no_expiry(self) -> None:
        """The CLI already documents ``--expire 0`` as 'never expire';
        env-var path mirrors that."""
        assert _parse_embedding_expire('0') is None

    def test_positive_integer_passes_through(self) -> None:
        assert _parse_embedding_expire('172800') == 172800
        assert _parse_embedding_expire('7') == 7

    def test_negative_integer_means_no_expiry(self) -> None:
        """Defensive: a negative value isn't a meaningful TTL.
        Treat as 'no expiration' rather than crashing on it."""
        assert _parse_embedding_expire('-1') is None

    def test_garbage_means_no_expiry(self) -> None:
        """A typo'd env var (e.g. ``2d`` instead of ``172800``) must
        not silently become a 2-day TTL via implicit conversion —
        better to disable expiry than to inherit a wrong value."""
        assert _parse_embedding_expire('two days') is None
        assert _parse_embedding_expire('2d') is None

    def test_none_argument(self) -> None:
        """Passed ``None`` directly (not the string) — defensive."""
        assert _parse_embedding_expire(None) is None


class TestGnservicesUrls:
    """gnfinder_url / gnparser_url default to the local services on
    ports 9080 / 9081 (v4 plan §1.A).  CLAUDE.md rule 11 precedence:
    CLI flag > env var > hardcoded default."""

    def test_default_localhost(self, monkeypatch: Any) -> None:
        """With no env vars and no CLI flags, both URLs point at
        localhost with the same subpaths the public defaults use
        (gnfinder /api/v1/find, gnparser /api/v1)."""
        monkeypatch.delenv('GNFINDER_URL', raising=False)
        monkeypatch.delenv('GNPARSER_URL', raising=False)
        monkeypatch.setattr('sys.argv', ['envconfig_test'])
        from env_config import get_env_config  # type: ignore[import]
        cfg = get_env_config()
        assert cfg['gnfinder_url'] == 'http://localhost:9080/api/v1/find'
        assert cfg['gnparser_url'] == 'http://localhost:9081/api/v1'

    def test_env_var_overrides_default(self, monkeypatch: Any) -> None:
        """GNFINDER_URL / GNPARSER_URL env vars override the localhost
        defaults — the env-var tier of rule 11."""
        monkeypatch.setenv(
            'GNFINDER_URL', 'http://prod-finder.example/api/v1/find',
        )
        monkeypatch.setenv(
            'GNPARSER_URL', 'http://prod-parser.example/api/v1',
        )
        monkeypatch.setattr('sys.argv', ['envconfig_test'])
        from env_config import get_env_config  # type: ignore[import]
        cfg = get_env_config()
        assert cfg['gnfinder_url'] == 'http://prod-finder.example/api/v1/find'
        assert cfg['gnparser_url'] == 'http://prod-parser.example/api/v1'

    def test_cli_overrides_env_var(self, monkeypatch: Any) -> None:
        """``--gnfinder-url`` / ``--gnparser-url`` on the CLI override
        the env vars — the CLI tier of rule 11."""
        monkeypatch.setenv(
            'GNFINDER_URL', 'http://env-finder.example/api/v1/find',
        )
        monkeypatch.setenv(
            'GNPARSER_URL', 'http://env-parser.example/api/v1',
        )
        monkeypatch.setattr('sys.argv', [
            'envconfig_test',
            '--gnfinder-url', 'http://cli-finder.example/api/v1/find',
            '--gnparser-url', 'http://cli-parser.example/api/v1',
        ])
        from env_config import get_env_config  # type: ignore[import]
        cfg = get_env_config()
        assert cfg['gnfinder_url'] == 'http://cli-finder.example/api/v1/find'
        assert cfg['gnparser_url'] == 'http://cli-parser.example/api/v1'
