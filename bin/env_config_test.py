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

    def test_v4_pass1_redis_key_propagates(self) -> None:
        """redis_keys.classifier_model_pass1 → classifier_model_key_pass1.
        Drives Step 5's two-CRF predictor."""
        config = _starter_config(classifier_model_key_pass1='')
        exp = {'redis_keys': {
            'classifier_model_pass1': 'skol:custom:v4_layout_hand',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert (
            config['classifier_model_key_pass1']
            == 'skol:custom:v4_layout_hand'
        )

    def test_v4_pass2_redis_key_propagates(self) -> None:
        config = _starter_config(classifier_model_key_pass2='')
        exp = {'redis_keys': {
            'classifier_model_pass2': 'skol:custom:v4_pass2_combined',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert (
            config['classifier_model_key_pass2']
            == 'skol:custom:v4_pass2_combined'
        )

    def test_v4_both_redis_keys_together(self) -> None:
        config = _starter_config(
            classifier_model_key_pass1='',
            classifier_model_key_pass2='',
        )
        exp = {'redis_keys': {
            'classifier_model_pass1': 'skol:k:pass1',
            'classifier_model_pass2': 'skol:k:pass2',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['classifier_model_key_pass1'] == 'skol:k:pass1'
        assert config['classifier_model_key_pass2'] == 'skol:k:pass2'

    def test_v4_single_redis_key_propagates(self) -> None:
        """redis_keys.classifier_model_single → classifier_model_key_single.
        Drives the post-Step-7 production cutover: when this field is
        set on an experiment doc, predict_v4 defaults to single-CRF
        mode against that key."""
        config = _starter_config(classifier_model_key_single='')
        exp = {'redis_keys': {
            'classifier_model_single': 'skol:custom:v4_single_combined',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert (
            config['classifier_model_key_single']
            == 'skol:custom:v4_single_combined'
        )

    def test_v4_all_three_redis_keys_together(self) -> None:
        """A v4 experiment can carry pass1 + pass2 + single
        simultaneously; predict_v4 chooses which to use at dispatch
        time (single wins by default; CLI flags override).  This
        test only asserts that env_config faithfully propagates all
        three — the dispatch rule lives in predict_v4."""
        config = _starter_config(
            classifier_model_key_pass1='',
            classifier_model_key_pass2='',
            classifier_model_key_single='',
        )
        exp = {'redis_keys': {
            'classifier_model_pass1': 'skol:k:pass1',
            'classifier_model_pass2': 'skol:k:pass2',
            'classifier_model_single': 'skol:k:single',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['classifier_model_key_pass1'] == 'skol:k:pass1'
        assert config['classifier_model_key_pass2'] == 'skol:k:pass2'
        assert config['classifier_model_key_single'] == 'skol:k:single'

    def test_v4_single_redis_key_default_empty_string(self) -> None:
        """An experiment doc that omits ``classifier_model_single``
        leaves the starter empty string in place — no spurious
        non-empty default that would auto-flip dispatch."""
        config = _starter_config(
            classifier_model_key_single='__sentinel__',
        )
        exp = {'redis_keys': {
            'classifier_model_pass1': 'skol:k:pass1',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        # Starter value untouched.
        assert config['classifier_model_key_single'] == '__sentinel__'

    def test_pipeline_field_propagates(self) -> None:
        """experiment.pipeline → config['pipeline'].  The
        manage_experiment dispatcher reads this and loads the
        matching ``bin/pipelines/<name>.py`` module."""
        config = _starter_config(pipeline='')
        exp = {'pipeline': 'v4_crf'}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['pipeline'] == 'v4_crf'

    def test_pipeline_field_omitted_leaves_default(self) -> None:
        """An experiment doc without ``pipeline`` field leaves the
        starter empty string in place — the manage_experiment
        runtime catches that with a clear migration message."""
        config = _starter_config(pipeline='')
        exp = {'redis_keys': {}}   # no pipeline field
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['pipeline'] == ''

    def test_pipeline_field_cli_explicit_blocks_override(self) -> None:
        """If the operator passed ``--pipeline`` on the CLI, the
        experiment doc value MUST NOT win."""
        config = _starter_config(pipeline='cli_value')
        exp = {'pipeline': 'v4_crf'}
        _apply_experiment(
            config, exp, cli_explicit_keys={'pipeline'},
        )
        assert config['pipeline'] == 'cli_value'

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


class TestApplyExperimentTreatmentsProseStructuredMapping:
    """Post-2026-06-10 rename: databases.treatments → treatments_prose,
    databases.treatments_full → treatments_structured.  The new
    fields populate BOTH the legacy script-level keys
    (treatments_db_name / dest_db) AND the new role-named keys."""

    def test_treatments_prose_field_propagates(self) -> None:
        config = {
            'treatments_db_name':      'skol_treatments_dev',
            'source_db':               'skol_treatments_dev',
            'treatments_prose_db_name': '',
        }
        exp = {'databases': {
            'treatments_prose': 'skol_exp_x_02_00_treatments_prose',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        # Legacy keys still get populated so existing script consumers
        # work unchanged.
        assert (config['treatments_db_name']
                == 'skol_exp_x_02_00_treatments_prose')
        assert (config['source_db']
                == 'skol_exp_x_02_00_treatments_prose')
        # New role-named key for downstream consumers that opt in.
        assert (config['treatments_prose_db_name']
                == 'skol_exp_x_02_00_treatments_prose')

    def test_treatments_structured_field_propagates(self) -> None:
        config = {
            'dest_db':                       'skol_treatments_full_dev',
            'treatments_structured_db_name': '',
        }
        exp = {'databases': {
            'treatments_structured': 'skol_exp_x_03_00_treatments_structured',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert (config['dest_db']
                == 'skol_exp_x_03_00_treatments_structured')
        assert (config['treatments_structured_db_name']
                == 'skol_exp_x_03_00_treatments_structured')

    def test_new_role_names_win_when_all_present(self) -> None:
        """Doc carrying treatments_prose AND legacy treatments AND
        legacy taxa — the new role name wins because the mapping
        list places it last (later assignment overwrites)."""
        config = {'treatments_db_name': ''}
        exp = {'databases': {
            'treatments_prose': 'new_prose_db',
            'treatments':       'mid_treatments_db',
            'taxa':             'legacy_taxa_db',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert config['treatments_db_name'] == 'new_prose_db'

    def test_eval_annotations_db_field_propagates(self) -> None:
        """databases.annotations_eval lets operators override the
        synthesised ``{annotations_db}_eval`` default in
        build_variables."""
        config = {'eval_annotations_db_name': ''}
        exp = {'databases': {
            'annotations_eval': 'skol_exp_x_01_00_ann_eval_custom',
        }}
        _apply_experiment(config, exp, cli_explicit_keys=set())
        assert (config['eval_annotations_db_name']
                == 'skol_exp_x_01_00_ann_eval_custom')


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


class TestRedisClusterMode:
    """REDIS_CLUSTER_MODE env var → config['redis_cluster_mode'] boolean.

    This is the toggle that determines whether create_redis_client returns
    a redis.Redis() or a redis.cluster.RedisCluster() (Phase 2 step 4 of
    the skol→tsqali Redis migration).  Default off so dev environments
    and non-cluster prod boxes keep using single-node Redis until they
    opt in by setting REDIS_CLUSTER_MODE=yes in /home/skol/.skol_env.
    """

    def _isolate(self, monkeypatch: Any) -> None:
        """Strip REDIS_CLUSTER_MODE from os.environ and stub out
        _load_skol_env so host-specific .skol_env files can't leak
        into the test outcome."""
        monkeypatch.delenv('REDIS_CLUSTER_MODE', raising=False)
        monkeypatch.setattr('env_config._load_skol_env', lambda: {})
        monkeypatch.setattr('sys.argv', ['envconfig_test'])

    def test_default_false(self, monkeypatch: Any) -> None:
        """With no env var set, default is False (single-node Redis)."""
        self._isolate(monkeypatch)
        from env_config import get_env_config  # type: ignore[import]
        assert get_env_config()['redis_cluster_mode'] is False

    def test_truthy_values_enable(self, monkeypatch: Any) -> None:
        """'yes', 'true', '1' (case-insensitive) flip cluster mode on.
        Matches the existing REDIS_TLS parsing convention."""
        for truthy in ('yes', 'YES', 'true', 'True', '1'):
            self._isolate(monkeypatch)
            monkeypatch.setenv('REDIS_CLUSTER_MODE', truthy)
            from env_config import get_env_config  # type: ignore[import]
            assert get_env_config()['redis_cluster_mode'] is True, (
                f'{truthy!r} should enable cluster mode'
            )

    def test_falsy_values_leave_disabled(self, monkeypatch: Any) -> None:
        """Anything else — including 'no', 'false', '0', and typos —
        leaves cluster mode off.  Defensive: we don't want a typo'd
        env var to silently route traffic at a non-existent cluster."""
        for falsy in ('', 'no', 'false', '0', 'off', 'cluster', 'on'):
            self._isolate(monkeypatch)
            monkeypatch.setenv('REDIS_CLUSTER_MODE', falsy)
            from env_config import get_env_config  # type: ignore[import]
            assert get_env_config()['redis_cluster_mode'] is False, (
                f'{falsy!r} should NOT enable cluster mode'
            )

    def test_propagates_to_get_redis_config(self, monkeypatch: Any) -> None:
        """get_redis_config() surfaces cluster_mode alongside the other
        connection fields — that's how the Django factory will read it
        in Phase 2 step 4."""
        self._isolate(monkeypatch)
        monkeypatch.setenv('REDIS_CLUSTER_MODE', 'yes')
        from env_config import get_redis_config  # type: ignore[import]
        assert get_redis_config()['cluster_mode'] is True
