"""Tests for bin/manage_experiment.py.

Post-restructure: per-family pipeline modules in ``bin/pipelines/``
own step-command construction.  This file pins the manage_experiment
glue: the required ``pipeline`` field, the lazy-repair logic, the
``--pipeline`` CLI flags on create + update, and the redis-key
flags that were there pre-restructure.
"""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manage_experiment import (  # type: ignore[import]  # noqa: E402
    _ensure_pipeline,
    _render_pipeline_step,
    cmd_approve,
    cmd_create,
    cmd_deploy,
    cmd_unapprove,
    cmd_update,
)


def _config(**overrides: Any) -> Dict[str, Any]:
    """Minimal resolved-config dict carrying the golden keys."""
    base: Dict[str, Any] = {
        'golden_db_name':     'skol_golden',
        'golden_ann_db_name': 'skol_golden_ann_hand',
    }
    base.update(overrides)
    return base


def _flatten(cmds):
    """Return the union of every arg across every command in the list."""
    out = []
    for cmd in cmds:
        out.extend(cmd)
    return out


# ---------------------------------------------------------------------------
# Pipeline-field requirement + render glue
# ---------------------------------------------------------------------------


class TestRequiresPipelineField:
    """``_ensure_pipeline`` is the gatekeeper that ensures every
    experiment doc names a known pipeline before any step builder
    runs.  The error message guides operators to the migration
    command."""

    def test_missing_field_raises_with_migration_hint(self) -> None:
        import pytest
        doc: Dict[str, Any] = {'_id': 'production'}
        with pytest.raises(ValueError) as exc:
            _ensure_pipeline(doc)
        msg = str(exc.value)
        assert 'pipeline' in msg.lower()
        assert 'update' in msg
        assert 'production' in msg

    def test_unknown_pipeline_name_raises_with_known_names(self) -> None:
        import pytest
        doc: Dict[str, Any] = {
            '_id': 'production', 'pipeline': 'not_a_real_pipeline',
        }
        with pytest.raises(ValueError) as exc:
            _ensure_pipeline(doc)
        msg = str(exc.value)
        assert 'v3_logistic' in msg
        assert 'v4_crf' in msg


class TestEnsurePipelineRepair:
    """Lazy repair against the family's canonical step list — the
    same behaviour as pre-restructure ``_ensure_pipeline`` but
    sourced from the pipeline module instead of a global list."""

    def test_repair_adds_missing_steps_as_pending(self) -> None:
        """Doc has only the first v4 step recorded under
        ``pipeline_state.steps``; repair fills in the rest from
        the v4_crf canonical list with ``status='pending'`` while
        preserving any pre-existing entries.

        Field shape: post-restructure the experiment doc carries
        ``pipeline`` (str, the family name) and ``pipeline_state``
        (the per-step status records).  The legacy field name
        ``pipeline`` (dict) is replaced — see the migration in
        docs/experiments.md."""
        doc: Dict[str, Any] = {
            '_id': 'production_v4',
            'pipeline': 'v4_crf',
            'pipeline_state': {
                'current_step': 0,
                'steps': [
                    {'name': 'annotate', 'status': 'completed',
                     'started_at': None, 'completed_at': None},
                ],
            },
        }
        _ensure_pipeline(doc)
        names = [s['name'] for s in doc['pipeline_state']['steps']]
        assert 'annotate' in names
        assert 'embed_lines' in names
        assert 'predict' in names

    def test_repair_preserves_existing_step_status(self) -> None:
        doc = {
            '_id': 'production_v4',
            'pipeline': 'v4_crf',
            'pipeline_state': {
                'current_step': 0,
                'steps': [
                    {'name': 'annotate', 'status': 'completed',
                     'started_at': '2026-06-01', 'completed_at': '2026-06-01'},
                ],
            },
        }
        _ensure_pipeline(doc)
        annotate = next(
            s for s in doc['pipeline_state']['steps']
            if s['name'] == 'annotate'
        )
        assert annotate['status'] == 'completed'
        assert annotate['started_at'] == '2026-06-01'

    def test_repair_initialises_from_scratch_when_state_absent(self) -> None:
        doc = {'_id': 'production_v4', 'pipeline': 'v4_crf'}
        _ensure_pipeline(doc)
        assert 'pipeline_state' in doc
        names = [s['name'] for s in doc['pipeline_state']['steps']]
        assert names[0] == 'annotate'   # v4 prereq goes first
        assert all(
            s['status'] == 'pending'
            for s in doc['pipeline_state']['steps']
        )


# ---------------------------------------------------------------------------
# Render glue — manage_experiment.py's thin wrapper around
# pipelines.base.render_step
# ---------------------------------------------------------------------------


class TestRenderPipelineStep:
    """The dispatcher just (a) loads the family, (b) finds the
    step by name, (c) builds the variable dict, (d) calls
    pipelines.base.render_step.  These tests pin the integration
    boundary."""

    def test_v3_predict_step_renders_predict_classifier(self) -> None:
        config = {'pipeline': 'v3_logistic'}
        cmd = _render_pipeline_step(
            'predict', 'production', force=False, config=config,
        )
        assert any('predict_classifier' in t for t in cmd)
        assert '--experiment' in cmd
        assert 'production' in cmd

    def test_v4_predict_step_renders_predict_v4_with_source_db(self) -> None:
        config = {
            'pipeline': 'v4_crf',
            'ingest_db_name': 'skol_dev',
        }
        cmd = _render_pipeline_step(
            'predict', 'production_v4', force=False, config=config,
        )
        assert any('predict_v4' in t for t in cmd)
        assert '--source-db' in cmd
        sd_idx = cmd.index('--source-db')
        assert cmd[sd_idx + 1] == 'skol_dev'

    def test_unknown_step_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            _render_pipeline_step(
                'nonexistent_step', 'production',
                force=False,
                config={'pipeline': 'v3_logistic'},
            )

    def test_force_propagates_to_render(self) -> None:
        config = {'pipeline': 'v4_crf', 'ingest_db_name': 'skol_dev'}
        cmd = _render_pipeline_step(
            'predict', 'production_v4', force=True, config=config,
        )
        # predict has --skip-existing in args; force flips it.
        assert '--force' in cmd
        assert '--skip-existing' not in cmd


# cmd_create + new --redis-key-pass1 / --redis-key-pass2 flags
# ---------------------------------------------------------------------------


class _FakeExperimentsDb:
    """Minimal stand-in for the couchdb.Database used by cmd_create.
    Supports __contains__, get, save, and view (for _all_docs)."""

    def __init__(self) -> None:
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.saves: int = 0

    def __contains__(self, key: str) -> bool:
        return key in self.docs

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.docs[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.docs.get(key, default)

    def save(self, doc: Dict[str, Any]) -> Any:
        self.saves += 1
        doc.setdefault('_id', doc.get('_id', doc.get('name', '')))
        doc.setdefault('_rev', '1-fake')
        self.docs[doc['_id']] = doc
        return doc['_id'], doc['_rev']

    def view(self, _view_name: str, **_kwargs):
        return iter([])


def _create_args(**overrides: Any) -> Any:
    """Build the argparse.Namespace shape cmd_create reads."""
    import argparse
    defaults = {
        'name': 'test_exp', 'notes': None, 'comments': None,
        'pipeline': 'v3_logistic',
        'model_name': None,
        'training_db': None, 'ingest_db': None, 'annotations_db': None,
        'features_status_db': None, 'features_hand_db': None,
        'redis_key_pass1': None, 'redis_key_pass2': None,
        'redis_key_single': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _update_args(**overrides: Any) -> Any:
    """Build the argparse.Namespace shape cmd_update reads."""
    import argparse
    defaults = {
        'name': 'test_exp', 'notes': None, 'comments': None,
        'status': None,
        'pipeline': None,
        'model_name': None,
        'training_db': None, 'ingest_db': None, 'annotations_db': None,
        'features_status_db': None, 'features_hand_db': None,
        'discover_databases': False, 'force': False,
        'redis_key_pass1': None, 'redis_key_pass2': None,
        'redis_key_single': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCmdCreateV4RedisKeys:
    """v6 Step 6.0: cmd_create must accept --redis-key-pass1 /
    --redis-key-pass2 and write them into ``redis_keys`` so the
    v4 two-CRF predictor can resolve its Pass-1/Pass-2 bundles
    from the experiment doc."""

    def test_create_writes_pass1_redis_key(self) -> None:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4',
            model_name='v4_crf',
            redis_key_pass1='skol:custom:v4_layout_hand',
        ))
        doc = db.docs['production_v4']
        assert (
            doc['redis_keys']['classifier_model_pass1']
            == 'skol:custom:v4_layout_hand'
        )
        assert doc['model_name'] == 'v4_crf'

    def test_create_writes_pass2_redis_key(self) -> None:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4',
            redis_key_pass2='skol:custom:v4_pass2_combined',
        ))
        doc = db.docs['production_v4']
        assert (
            doc['redis_keys']['classifier_model_pass2']
            == 'skol:custom:v4_pass2_combined'
        )

    def test_create_writes_both_pass_keys(self) -> None:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4',
            model_name='v4_crf',
            redis_key_pass1='skol:k:p1',
            redis_key_pass2='skol:k:p2',
        ))
        rk = db.docs['production_v4']['redis_keys']
        assert rk['classifier_model_pass1'] == 'skol:k:p1'
        assert rk['classifier_model_pass2'] == 'skol:k:p2'

    def test_create_does_not_write_pass_keys_when_omitted(self) -> None:
        """v3 experiments (no v4 flags) keep their default redis_keys
        without any spurious pass1/pass2 entries."""
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='production_v3'))
        rk = db.docs['production_v3']['redis_keys']
        assert 'classifier_model_pass1' not in rk
        assert 'classifier_model_pass2' not in rk
        assert 'classifier_model_single' not in rk

    def test_create_writes_single_redis_key(self) -> None:
        """Post-Step-7: cmd_create accepts --redis-key-single and
        writes redis_keys.classifier_model_single.  When this field
        is set on production_v4, predict_v4 defaults to single-CRF."""
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4',
            model_name='v4_crf',
            redis_key_single='skol:classifier:model:v4_single_combined',
        ))
        doc = db.docs['production_v4']
        assert (
            doc['redis_keys']['classifier_model_single']
            == 'skol:classifier:model:v4_single_combined'
        )
        assert doc['model_name'] == 'v4_crf'

    def test_create_writes_all_three_pass_keys_together(self) -> None:
        """A v4 experiment can hold pass1 + pass2 + single side by
        side — operators set whichever the dispatch hierarchy needs.
        """
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4',
            model_name='v4_crf',
            redis_key_pass1='skol:k:p1',
            redis_key_pass2='skol:k:p2',
            redis_key_single='skol:k:single',
        ))
        rk = db.docs['production_v4']['redis_keys']
        assert rk['classifier_model_pass1'] == 'skol:k:p1'
        assert rk['classifier_model_pass2'] == 'skol:k:p2'
        assert rk['classifier_model_single'] == 'skol:k:single'


class TestCmdUpdateV4RedisKeys:
    """The production cutover lands via ``manage_experiment update``,
    not ``create``.  These tests pin the update-path semantics for
    every v4 redis-key flag — cmd_update's pass1/pass2 path was
    previously untested."""

    def _seeded_db(self) -> _FakeExperimentsDb:
        """Db pre-populated with a v4 experiment that has the doc
        shape produced by cmd_create."""
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4', model_name='v4_crf',
            redis_key_pass1='skol:k:original_p1',
            redis_key_pass2='skol:k:original_p2',
        ))
        return db

    def test_update_writes_pass1_redis_key(self) -> None:
        db = self._seeded_db()
        cmd_update(db, _update_args(
            name='production_v4',
            redis_key_pass1='skol:k:new_p1',
        ))
        rk = db.docs['production_v4']['redis_keys']
        assert rk['classifier_model_pass1'] == 'skol:k:new_p1'
        # Unaffected siblings stay put.
        assert rk['classifier_model_pass2'] == 'skol:k:original_p2'

    def test_update_writes_pass2_redis_key(self) -> None:
        db = self._seeded_db()
        cmd_update(db, _update_args(
            name='production_v4',
            redis_key_pass2='skol:k:new_p2',
        ))
        rk = db.docs['production_v4']['redis_keys']
        assert rk['classifier_model_pass2'] == 'skol:k:new_p2'
        assert rk['classifier_model_pass1'] == 'skol:k:original_p1'

    def test_update_writes_single_redis_key(self) -> None:
        """The actual cutover command's effect: setting
        classifier_model_single on a previously two-pass experiment
        doc — leaves pass1/pass2 alone so they remain available as
        fallbacks for explicit-CLI two-pass invocations."""
        db = self._seeded_db()
        cmd_update(db, _update_args(
            name='production_v4',
            redis_key_single='skol:classifier:model:v4_single_combined',
        ))
        rk = db.docs['production_v4']['redis_keys']
        assert (
            rk['classifier_model_single']
            == 'skol:classifier:model:v4_single_combined'
        )
        # Pass-1 + Pass-2 still present so explicit --pass1-key /
        # --pass2-key two-pass invocations still resolve.
        assert rk['classifier_model_pass1'] == 'skol:k:original_p1'
        assert rk['classifier_model_pass2'] == 'skol:k:original_p2'

    def test_update_does_not_write_single_when_omitted(self) -> None:
        """Omitting --redis-key-single from an update call leaves
        the field untouched (defensive against accidental clobbering
        by other update operations like --notes)."""
        db = self._seeded_db()
        cmd_update(db, _update_args(
            name='production_v4',
            notes='just touching notes',
        ))
        rk = db.docs['production_v4']['redis_keys']
        assert 'classifier_model_single' not in rk


# ---------------------------------------------------------------------------
# --pipeline flag on create + update
# ---------------------------------------------------------------------------


class TestCmdCreatePipelineField:
    """``bin/manage_experiment create --pipeline X`` must write
    the family name into ``doc['pipeline']`` so future ``runnext``
    invocations can dispatch the right step list."""

    def test_create_writes_pipeline_field(self) -> None:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='new_v4', pipeline='v4_crf'))
        doc = db.docs['new_v4']
        assert doc['pipeline'] == 'v4_crf'

    def test_create_writes_default_pipeline_when_omitted(self) -> None:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='new_legacy'))
        doc = db.docs['new_legacy']
        # _create_args defaults to v3_logistic, matching the
        # operator-facing default on the CLI flag.
        assert doc['pipeline'] == 'v3_logistic'

    def test_create_initialises_pipeline_state_from_family(self) -> None:
        """A fresh v4_crf experiment gets pipeline_state.steps
        populated from the family's canonical list — `annotate`
        first, `embed_lines` second."""
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='new_v4', pipeline='v4_crf'))
        names = [
            s['name']
            for s in db.docs['new_v4']['pipeline_state']['steps']
        ]
        assert names[0] == 'annotate'
        assert names[1] == 'embed_lines'

    def test_create_rejects_unknown_pipeline_with_clear_error(self) -> None:
        import pytest
        db = _FakeExperimentsDb()
        with pytest.raises(ValueError) as exc:
            cmd_create(db, _create_args(
                name='new_x', pipeline='not_a_real_pipeline',
            ))
        assert 'v3_logistic' in str(exc.value)
        assert 'v4_crf' in str(exc.value)


class TestCmdUpdatePipelineField:
    """``bin/manage_experiment update --pipeline X`` is the
    one-shot migration command for legacy experiment docs."""

    def _seeded_legacy_db(self) -> _FakeExperimentsDb:
        """A db with a legacy-shape doc: ``pipeline`` is the dict
        (current_step + steps), no ``pipeline_state`` field."""
        db = _FakeExperimentsDb()
        db.docs['legacy_v3'] = {
            '_id': 'legacy_v3',
            'model_name': 'logistic_sections',
            'pipeline': {
                'current_step': 2,
                'steps': [
                    {'name': 'train', 'status': 'completed',
                     'started_at': None, 'completed_at': None},
                ],
            },
        }
        return db

    def test_update_writes_pipeline_field_string(self) -> None:
        db = self._seeded_legacy_db()
        cmd_update(db, _update_args(
            name='legacy_v3', pipeline='v3_logistic',
        ))
        assert db.docs['legacy_v3']['pipeline'] == 'v3_logistic'

    def test_update_migrates_legacy_pipeline_dict_to_state(self) -> None:
        """The legacy ``pipeline`` dict becomes ``pipeline_state``;
        any per-step status records the legacy doc had are
        preserved."""
        db = self._seeded_legacy_db()
        cmd_update(db, _update_args(
            name='legacy_v3', pipeline='v3_logistic',
        ))
        state = db.docs['legacy_v3']['pipeline_state']
        assert state['current_step'] == 2
        train_step = next(
            s for s in state['steps'] if s['name'] == 'train'
        )
        assert train_step['status'] == 'completed'

    def test_update_rejects_unknown_pipeline_with_clear_error(self) -> None:
        import pytest
        db = self._seeded_legacy_db()
        with pytest.raises(ValueError) as exc:
            cmd_update(db, _update_args(
                name='legacy_v3', pipeline='bogus',
            ))
        assert 'v3_logistic' in str(exc.value)


# ---------------------------------------------------------------------------
# Features DB flags + discover_databases
# ---------------------------------------------------------------------------


class TestCmdCreateFeaturesDbs:
    """cmd_create accepts --features-status-db and --features-hand-db
    so the naming-convention fallback warnings in
    resolve_status_db_name / resolve_hand_db_name never fire on
    freshly-created experiments."""

    def test_create_writes_features_status_db(self) -> None:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4',
            features_status_db=(
                'skol_exp_production_v4_02_50_features_status'
            ),
        ))
        doc = db.docs['production_v4']
        assert (
            doc['databases']['features_status']
            == 'skol_exp_production_v4_02_50_features_status'
        )

    def test_create_writes_features_hand_db(self) -> None:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4',
            features_hand_db=(
                'skol_exp_production_v4_02_55_features_hand'
            ),
        ))
        doc = db.docs['production_v4']
        assert (
            doc['databases']['features_hand']
            == 'skol_exp_production_v4_02_55_features_hand'
        )

    def test_create_omits_features_dbs_when_not_passed(self) -> None:
        """Backwards-compat: existing v3 create invocations that
        don't pass the new flags don't grow spurious empty keys."""
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='production_v3'))
        databases = db.docs['production_v3']['databases']
        assert 'features_status' not in databases
        assert 'features_hand' not in databases


class TestCmdUpdateFeaturesDbs:
    """cmd_update accepts the same two flags for post-hoc
    canonicalization of legacy docs that predate the
    features_status / features_hand fields."""

    def _seeded_db(self) -> _FakeExperimentsDb:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4', pipeline='v4_crf',
        ))
        return db

    def test_update_writes_features_status_db(self) -> None:
        db = self._seeded_db()
        cmd_update(db, _update_args(
            name='production_v4',
            features_status_db=(
                'skol_exp_production_v4_02_50_features_status'
            ),
        ))
        databases = db.docs['production_v4']['databases']
        assert (
            databases['features_status']
            == 'skol_exp_production_v4_02_50_features_status'
        )

    def test_update_writes_features_hand_db(self) -> None:
        db = self._seeded_db()
        cmd_update(db, _update_args(
            name='production_v4',
            features_hand_db=(
                'skol_exp_production_v4_02_55_features_hand'
            ),
        ))
        databases = db.docs['production_v4']['databases']
        assert (
            databases['features_hand']
            == 'skol_exp_production_v4_02_55_features_hand'
        )

    def test_update_omits_features_dbs_when_not_passed(self) -> None:
        db = self._seeded_db()
        cmd_update(db, _update_args(name='production_v4'))
        databases = db.docs['production_v4']['databases']
        assert 'features_status' not in databases
        assert 'features_hand' not in databases


class TestDiscoverDatabases:
    """Pure-function sweep of the canonical naming convention.

    Non-mutating — returns the {added, kept, missing} triage
    the caller applies to ``doc['databases']``.
    """

    def _existing_all_v4_dbs(self) -> set:
        """The full set of canonical DB names for the
        production_v4 experiment.  Substitutes for a server
        iteration in a live run."""
        return {
            'skol_exp_production_v4_01_00_ann',
            'skol_exp_production_v4_02_00_treatments_prose',
            'skol_exp_production_v4_02_50_features_candidate',
            'skol_exp_production_v4_02_50_features_status',
            'skol_exp_production_v4_02_55_features_hand',
            'skol_exp_production_v4_03_00_treatments_structured',
        }

    def test_adds_missing_roles(self) -> None:
        from manage_experiment import (  # type: ignore[import]
            discover_databases,
        )
        result = discover_databases(
            self._existing_all_v4_dbs(),
            'production_v4',
            current_dbs={},
        )
        assert (
            result['added']['features_status']
            == 'skol_exp_production_v4_02_50_features_status'
        )
        assert (
            result['added']['features_hand']
            == 'skol_exp_production_v4_02_55_features_hand'
        )
        # eval-suffixed DBs don't exist in our fake server →
        # they should appear in 'missing', not 'added'.
        assert 'annotations_eval' in result['missing']

    def test_preserves_existing_by_default(self) -> None:
        """When databases.features_hand is already set to a
        non-canonical name, discovery keeps it — operators
        may have deliberate overrides."""
        from manage_experiment import (  # type: ignore[import]
            discover_databases,
        )
        current = {
            'features_hand': 'custom_review_pool',
        }
        result = discover_databases(
            self._existing_all_v4_dbs(),
            'production_v4',
            current_dbs=current,
        )
        assert 'features_hand' not in result['added']
        assert result['kept']['features_hand'] == 'custom_review_pool'

    def test_force_overwrites_existing(self) -> None:
        """--force flips the preserve behaviour: the canonical
        name replaces the operator's existing entry when the
        canonical DB actually exists on the server."""
        from manage_experiment import (  # type: ignore[import]
            discover_databases,
        )
        current = {
            'features_hand': 'custom_review_pool',
        }
        result = discover_databases(
            self._existing_all_v4_dbs(),
            'production_v4',
            current_dbs=current,
            force=True,
        )
        assert (
            result['added']['features_hand']
            == 'skol_exp_production_v4_02_55_features_hand'
        )

    def test_force_keeps_custom_when_canonical_absent(self) -> None:
        """--force with a missing canonical DB should NOT null
        out the operator's existing entry — refusing to point at
        a nonexistent DB is safer than clearing the field."""
        from manage_experiment import (  # type: ignore[import]
            discover_databases,
        )
        current = {'features_hand': 'custom_review_pool'}
        # Server has NO features_hand DB at the canonical name:
        existing_no_hand = self._existing_all_v4_dbs() - {
            'skol_exp_production_v4_02_55_features_hand',
        }
        result = discover_databases(
            existing_no_hand,
            'production_v4',
            current_dbs=current,
            force=True,
        )
        assert result['kept']['features_hand'] == 'custom_review_pool'
        assert 'features_hand' not in result['added']

    def test_reports_missing_roles(self) -> None:
        """A role whose canonical DB doesn't exist and isn't
        in the doc appears in 'missing' so the operator can
        see which stages haven't been provisioned."""
        from manage_experiment import (  # type: ignore[import]
            discover_databases,
        )
        # Server has NOTHING for this experiment:
        result = discover_databases(
            set(), 'brand_new_exp', current_dbs={},
        )
        assert result['added'] == {}
        assert 'features_status' in result['missing']
        assert 'features_hand' in result['missing']

    def test_does_not_mutate_input(self) -> None:
        from manage_experiment import (  # type: ignore[import]
            discover_databases,
        )
        current = {'features_hand': 'custom'}
        before = dict(current)
        discover_databases(
            self._existing_all_v4_dbs(),
            'production_v4',
            current_dbs=current,
        )
        assert current == before


class TestCmdUpdateDiscoverDatabases:
    """cmd_update --discover-databases end-to-end: writes newly
    discovered DBs into the experiment doc, preserves existing
    entries by default."""

    class _FakeServer:
        """Iterable-of-db-names stand-in for couchdb.Server."""

        def __init__(self, names) -> None:
            self._names = set(names)

        def __contains__(self, name: str) -> bool:
            return name in self._names

    def _seeded_db(self) -> _FakeExperimentsDb:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4', pipeline='v4_crf',
        ))
        return db

    def test_discover_writes_features_status_and_hand(self) -> None:
        db = self._seeded_db()
        server = self._FakeServer([
            'skol_exp_production_v4_02_50_features_status',
            'skol_exp_production_v4_02_55_features_hand',
        ])
        cmd_update(
            db,
            _update_args(
                name='production_v4', discover_databases=True,
            ),
            server=server,
        )
        databases = db.docs['production_v4']['databases']
        assert (
            databases['features_status']
            == 'skol_exp_production_v4_02_50_features_status'
        )
        assert (
            databases['features_hand']
            == 'skol_exp_production_v4_02_55_features_hand'
        )

    def test_discover_without_server_errors(self) -> None:
        """Guardrail: running --discover-databases from a
        server-less test context must raise a clear error rather
        than silently no-op."""
        import pytest
        db = self._seeded_db()
        with pytest.raises(SystemExit):
            cmd_update(
                db,
                _update_args(
                    name='production_v4', discover_databases=True,
                ),
                # server=None (default) — the flag needs one
            )


# ---------------------------------------------------------------------------
# --log convenience: stdout/stderr → /var/log/skol/manage-experiment-<step>.log
# ---------------------------------------------------------------------------


class TestLogPathResolution:
    """``runnext`` / ``runstep --log`` redirect output to a
    per-step log file under SKOL_LOG_DIR (default
    ``/var/log/skol``).  Operators no longer have to remember
    the ``> ${LOGDIR}/...`` redirect in shell history or cron."""

    def test_default_path(self, monkeypatch: Any) -> None:
        from manage_experiment import _log_path_for_step
        monkeypatch.delenv('SKOL_LOG_DIR', raising=False)
        assert _log_path_for_step('predict') == (
            '/var/log/skol/manage-experiment-predict.log'
        )

    def test_env_overrides_dir(self, monkeypatch: Any) -> None:
        from manage_experiment import _log_path_for_step
        monkeypatch.setenv('SKOL_LOG_DIR', '/tmp/skol-logs')
        assert _log_path_for_step('predict') == (
            '/tmp/skol-logs/manage-experiment-predict.log'
        )

    def test_step_name_propagates(self, monkeypatch: Any) -> None:
        from manage_experiment import _log_path_for_step
        monkeypatch.setenv('SKOL_LOG_DIR', '/var/log/skol')
        assert _log_path_for_step('extract_treatments') == (
            '/var/log/skol/manage-experiment-extract_treatments.log'
        )


# ---------------------------------------------------------------------------
# `--` passthrough — forward args after `--` to the subprocess script
# ---------------------------------------------------------------------------


class TestSplitPassthroughArgs:
    """``manage_experiment runstep|runnext`` accepts ``--`` as a
    Unix-convention separator between its own argparse args and
    extra args to forward verbatim to the underlying script.  E.g.

      bin/manage_experiment runstep production_v4 embed_lines --log \\
          -- --verbosity 2 --batch-size 192

    The ``--verbosity 2 --batch-size 192`` portion lands on the
    embed_lines.py argv after the pipeline-template-rendered args.
    """

    def test_split_returns_main_only_when_no_separator(self) -> None:
        from manage_experiment import _split_passthrough_args
        main, passthrough = _split_passthrough_args(
            ['runstep', 'production_v4', 'embed_lines', '--log'],
        )
        assert main == ['runstep', 'production_v4', 'embed_lines', '--log']
        assert passthrough == []

    def test_split_at_separator(self) -> None:
        from manage_experiment import _split_passthrough_args
        main, passthrough = _split_passthrough_args(
            ['runstep', 'production_v4', 'embed_lines', '--log',
             '--', '--verbosity', '2', '--batch-size', '192'],
        )
        assert main == ['runstep', 'production_v4', 'embed_lines', '--log']
        assert passthrough == [
            '--verbosity', '2', '--batch-size', '192',
        ]

    def test_split_empty_passthrough_after_separator(self) -> None:
        """``-- `` with nothing after is degenerate but valid; the
        passthrough is empty, not an error."""
        from manage_experiment import _split_passthrough_args
        main, passthrough = _split_passthrough_args(
            ['runstep', 'production_v4', 'embed_lines', '--'],
        )
        assert main == ['runstep', 'production_v4', 'embed_lines']
        assert passthrough == []

    def test_split_only_first_separator_is_the_split(self) -> None:
        """If the passthrough itself contains ``--`` (e.g. forwarding
        to a script that uses ``--`` for its own purposes), the
        FIRST ``--`` is the manage_experiment-vs-subprocess split;
        any later ``--`` rides along to the subprocess."""
        from manage_experiment import _split_passthrough_args
        main, passthrough = _split_passthrough_args(
            ['runstep', 'production_v4', 'embed_lines',
             '--', '--foo', '--', '--bar'],
        )
        assert main == ['runstep', 'production_v4', 'embed_lines']
        assert passthrough == ['--foo', '--', '--bar']


class TestRunStepExtraArgs:
    """``_run_step`` accepts an ``extra_args`` kwarg that gets
    appended to the rendered subprocess command, after the args
    that ``render_step`` produces from the pipeline template."""

    def _seeded_db(self) -> _FakeExperimentsDb:
        """Doc with a v3_logistic pipeline so the steps resolve."""
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(
            name='production_v4',
            pipeline='v3_logistic',
        ))
        return db

    def test_extra_args_appended_to_subprocess_command(self) -> None:
        """``extra_args`` lands at the end of the argv that
        ``subprocess.run`` receives."""
        from unittest import mock
        from manage_experiment import _run_step
        db = self._seeded_db()
        doc = db.docs['production_v4']
        # The first pipeline step in v3_logistic is `train`.
        with mock.patch(
            'manage_experiment.subprocess.run',
        ) as mock_run, mock.patch(
            'manage_experiment._render_pipeline_step',
            return_value=['python', 'bin/train_classifier.py',
                          '--experiment', 'production_v4'],
        ):
            mock_run.return_value = mock.MagicMock(returncode=0)
            _run_step(
                db, doc, 0, 'production_v4',
                verbosity=0, force=False,
                extra_args=['--verbosity', '2', '--batch-size', '192'],
            )
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == [
            'python', 'bin/train_classifier.py',
            '--experiment', 'production_v4',
            '--verbosity', '2', '--batch-size', '192',
        ]

    def test_no_extra_args_leaves_command_unchanged(self) -> None:
        """Default behaviour: when no ``extra_args``, the rendered
        command is what subprocess.run gets."""
        from unittest import mock
        from manage_experiment import _run_step
        db = self._seeded_db()
        doc = db.docs['production_v4']
        with mock.patch(
            'manage_experiment.subprocess.run',
        ) as mock_run, mock.patch(
            'manage_experiment._render_pipeline_step',
            return_value=['python', 'bin/train_classifier.py',
                          '--experiment', 'production_v4'],
        ):
            mock_run.return_value = mock.MagicMock(returncode=0)
            _run_step(
                db, doc, 0, 'production_v4',
                verbosity=0, force=False,
                # extra_args=None — default
            )
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == [
            'python', 'bin/train_classifier.py',
            '--experiment', 'production_v4',
        ]

    def test_empty_extra_args_leaves_command_unchanged(self) -> None:
        """An empty list of extra args is equivalent to no extra
        args — important for the ``-- `` (separator with nothing
        after) edge case."""
        from unittest import mock
        from manage_experiment import _run_step
        db = self._seeded_db()
        doc = db.docs['production_v4']
        with mock.patch(
            'manage_experiment.subprocess.run',
        ) as mock_run, mock.patch(
            'manage_experiment._render_pipeline_step',
            return_value=['python', 'bin/train_classifier.py',
                          '--experiment', 'production_v4'],
        ):
            mock_run.return_value = mock.MagicMock(returncode=0)
            _run_step(
                db, doc, 0, 'production_v4',
                verbosity=0, force=False,
                extra_args=[],
            )
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd == [
            'python', 'bin/train_classifier.py',
            '--experiment', 'production_v4',
        ]


# ---------------------------------------------------------------------------
# Production-experiment seed removal — Step 5 of the DB-naming cleanup
# (2026-06-13).  No more hardcoded ``_production_experiment`` seed with
# legacy ``taxa`` / ``taxa_full`` field names.  Operators create the
# production doc explicitly via ``manage_experiment create``.
# ---------------------------------------------------------------------------


class TestProductionExperimentSeedRemoved:
    """Regression guard: the ``_production_experiment`` helper is
    gone.  Re-adding it would re-introduce the legacy field names
    (``taxa`` / ``taxa_full``) on every fresh CouchDB
    initialization.  See docs/skol-db-naming-cleanup.md."""

    def test_production_experiment_helper_no_longer_exists(self) -> None:
        import manage_experiment
        assert not hasattr(manage_experiment, '_production_experiment'), (
            'The _production_experiment seed was deliberately removed '
            'as part of Step 5 of the DB-naming cleanup (2026-06-13). '
            'Operators now create the production doc explicitly via '
            '`manage_experiment create --name production '
            '--pipeline v3_logistic` after a fresh setup.'
        )


class TestCmdDeployRequiresExplicitProduction:
    """``cmd_deploy`` no longer silently creates a 'production' doc
    when one doesn't exist — the operator must create it
    explicitly first.  Eliminates the legacy-field-names-on-seed
    leak (see TestProductionExperimentSeedRemoved)."""

    def test_deploy_errors_when_production_doc_missing(self) -> None:
        """Without a pre-existing 'production' doc, cmd_deploy
        exits non-zero with an operator-actionable message rather
        than silently auto-creating one."""
        import argparse
        import io
        import pytest
        from contextlib import redirect_stderr
        db = _FakeExperimentsDb()
        # Seed a non-production source experiment to deploy from.
        cmd_create(db, _create_args(
            name='my_exp', pipeline='v3_logistic',
        ))
        captured = io.StringIO()
        with redirect_stderr(captured), pytest.raises(SystemExit) as exc:
            cmd_deploy(db, argparse.Namespace(name='my_exp'))
        assert exc.value.code != 0
        # The error message must surface the migration hint so an
        # operator knows the fix.
        err = captured.getvalue()
        assert 'production' in err.lower()
        assert 'create' in err.lower()


class TestCmdApprove:
    """``approve <name>`` sets approved=True on the experiment doc.
    Used by the web UI's default-visible filter."""

    def test_approve_sets_flag_to_true(self) -> None:
        import argparse
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='exp1', pipeline='v3_logistic'))
        cmd_approve(db, argparse.Namespace(name='exp1'))
        assert db.docs['exp1']['approved'] is True

    def test_approve_updates_updated_at(self) -> None:
        import argparse
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='exp1', pipeline='v3_logistic'))
        prior = db.docs['exp1'].get('updated_at', '')
        cmd_approve(db, argparse.Namespace(name='exp1'))
        assert db.docs['exp1']['updated_at'] != ''
        # We don't assert > prior because both timestamps may be the
        # same wall-clock second; just that the field is rewritten.

    def test_approve_idempotent_when_already_approved(self) -> None:
        """Re-approving an already-approved experiment is a no-op
        (prints "already approved", does not bump _rev)."""
        import argparse
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='exp1', pipeline='v3_logistic'))
        cmd_approve(db, argparse.Namespace(name='exp1'))
        saves_after_first = db.saves
        cmd_approve(db, argparse.Namespace(name='exp1'))
        assert db.saves == saves_after_first

    def test_approve_missing_experiment_exits_nonzero(self) -> None:
        import argparse
        import pytest
        db = _FakeExperimentsDb()
        with pytest.raises(SystemExit) as exc:
            cmd_approve(db, argparse.Namespace(name='does_not_exist'))
        assert exc.value.code != 0


class TestCmdUnapprove:
    """``unapprove <name>`` sets approved=False on the experiment doc."""

    def test_unapprove_sets_flag_to_false(self) -> None:
        import argparse
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='exp1', pipeline='v3_logistic'))
        cmd_approve(db, argparse.Namespace(name='exp1'))
        cmd_unapprove(db, argparse.Namespace(name='exp1'))
        assert db.docs['exp1']['approved'] is False

    def test_unapprove_idempotent_when_not_approved(self) -> None:
        """Unapproving an experiment that was never approved is a
        no-op — doesn't write approved=False just to make a point."""
        import argparse
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='exp1', pipeline='v3_logistic'))
        saves_before = db.saves
        cmd_unapprove(db, argparse.Namespace(name='exp1'))
        assert db.saves == saves_before
        assert 'approved' not in db.docs['exp1']

    def test_unapprove_missing_experiment_exits_nonzero(self) -> None:
        import argparse
        import pytest
        db = _FakeExperimentsDb()
        with pytest.raises(SystemExit) as exc:
            cmd_unapprove(db, argparse.Namespace(name='does_not_exist'))
        assert exc.value.code != 0


# ---------------------------------------------------------------------------
# §10.4 deliverable 4.5 — features_candidate DB key on the experiment doc
# ---------------------------------------------------------------------------


class TestCmdCreateFeaturesCandidate:
    """``cmd_create`` writes ``databases.features_candidate`` via the
    naming convention (no flag — derived from the experiment name).

    Convention: ``skol_exp_<name>_02_50_features_candidate`` — the
    02_50 slot sorts between 02_00_treatments_prose extraction and
    03_00_treatments_structured SLM output, per
    docs/schema_constrained_pipeline.md §10.3 and
    docs/skol-db-naming-cleanup.md.
    """

    def test_create_writes_features_candidate(self) -> None:
        db = _FakeExperimentsDb()
        cmd_create(db, _create_args(name='production_v4'))
        doc = db.docs['production_v4']
        assert doc['databases']['features_candidate'] == (
            'skol_exp_production_v4_02_50_features_candidate'
        )

    def test_create_does_not_emit_features_candidate_flag(self) -> None:
        """The flag is deliberately absent: option (b) — convention
        only, derived from the experiment name.  If a future change
        adds a flag, this test catches the regression."""
        # _create_args's default Namespace has no
        # 'features_candidate_db' attribute (per option b).  If
        # cmd_create starts reading one, that'd be a deviation.
        args = _create_args(name='production_v4')
        assert not hasattr(args, 'features_candidate_db')


class TestCmdUpdateFeaturesCandidateSelfRepair:
    """``cmd_update`` fills in ``databases.features_candidate`` if
    missing (lazy repair for legacy experiment docs).  Doesn't
    overwrite if already set — preserves an operator's custom value.
    """

    def test_update_backfills_missing_field(self) -> None:
        """Legacy doc with no features_candidate field gets it
        populated on the next bare ``update <name>``."""
        import argparse
        db = _FakeExperimentsDb()
        # Hand-craft a legacy doc shape (no features_candidate)
        db.docs['legacy'] = {
            '_id': 'legacy',
            'pipeline': 'v3_logistic',
            'pipeline_state': {'current_step': 0, 'steps': []},
            'databases': {
                'ingest': 'skol_dev',
                'treatments_prose': 'skol_treatments_dev',
            },
        }
        cmd_update(db, _update_args(name='legacy'))
        assert db.docs['legacy']['databases'][
            'features_candidate'
        ] == 'skol_exp_legacy_02_50_features_candidate'

    def test_update_preserves_existing_custom_value(self) -> None:
        """Operator-set custom name is left alone — repair only
        when missing."""
        import argparse
        db = _FakeExperimentsDb()
        db.docs['custom'] = {
            '_id': 'custom',
            'pipeline': 'v3_logistic',
            'pipeline_state': {'current_step': 0, 'steps': []},
            'databases': {
                'ingest': 'skol_dev',
                'features_candidate': 'my_special_features_db',
            },
        }
        saves_before = db.saves
        cmd_update(db, _update_args(name='custom'))
        assert db.docs['custom']['databases'][
            'features_candidate'
        ] == 'my_special_features_db'
        # No-op because the field was already present.
        assert db.saves == saves_before

    def test_update_creates_databases_block_if_absent(self) -> None:
        """A doc with no 'databases' key at all should still get
        the field populated (defensive setdefault)."""
        import argparse
        db = _FakeExperimentsDb()
        db.docs['bare'] = {
            '_id': 'bare',
            'pipeline': 'v3_logistic',
            'pipeline_state': {'current_step': 0, 'steps': []},
        }
        cmd_update(db, _update_args(name='bare'))
        assert db.docs['bare']['databases'][
            'features_candidate'
        ] == 'skol_exp_bare_02_50_features_candidate'
