"""Tests for bin/pipelines/base.py — PipelineStep dataclass + render_step.

These pin the contract every per-family pipeline module relies on:
the variable-substitution syntax, the ``--force`` rewrite rule, and
the fail-fast behaviour on an unknown variable name.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bin.pipelines.base import (  # noqa: E402
    PipelineStep,
    build_variables,
    render_step,
)


# ---------------------------------------------------------------------------
# Variable substitution
# ---------------------------------------------------------------------------


class TestRenderStepSubstitution(unittest.TestCase):

    def _vars(self, **overrides):
        base = {
            'experiment':       'production',
            'input_db':         'skol_dev',
            'training_db':      'skol_training',
            'annotations_db':   'skol_ann',
            'golden_db':        'skol_golden',
            'golden_ann_db':    'skol_golden_ann_hand',
            'eval_ann_db':      'skol_ann',
            'model_key_single': 'skol:k:single',
            'model_key_pass1':  'skol:k:p1',
            'model_key_pass2':  'skol:k:p2',
            'sbert_model':      'mpnet',
        }
        base.update(overrides)
        return base

    def test_substitutes_known_variables(self):
        step = PipelineStep(
            name='predict', script='predict_v4',
            args=('--experiment', '{experiment}',
                  '--source-db', '{input_db}'),
        )
        cmd = render_step(step, self._vars(), force=False)
        # The first 2 tokens are [python, /abs/path/bin/predict_v4.py];
        # we don't assert on them here (script-resolution test lives below).
        self.assertEqual(
            cmd[2:],
            ['--experiment', 'production',
             '--source-db', 'skol_dev'],
        )

    def test_raises_on_unknown_variable(self):
        step = PipelineStep(
            name='broken', script='nowhere',
            args=('--something', '{not_a_real_var}'),
        )
        with self.assertRaises(KeyError) as cm:
            render_step(step, self._vars(), force=False)
        self.assertIn('not_a_real_var', str(cm.exception))

    def test_tokens_without_braces_pass_through(self):
        step = PipelineStep(
            name='build_vocab', script='build_vocab_tree',
            args=('--experiment', '{experiment}'),
            sequential=False,
        )
        cmd = render_step(step, self._vars(), force=False)
        self.assertEqual(cmd[2:], ['--experiment', 'production'])

    def test_token_with_no_format_tokens_at_all(self):
        """A step like ``build_sources_stats`` includes
        ``--verbosity 2`` — no format strings.  Pass-through."""
        step = PipelineStep(
            name='stats', script='build_sources_stats',
            args=('--verbosity', '2'),
            sequential=False,
        )
        cmd = render_step(step, self._vars(), force=False)
        self.assertEqual(cmd[2:], ['--verbosity', '2'])

    def test_repeated_substitution_inside_one_step(self):
        """A step can reference the same variable in multiple
        positions (e.g. golden_db appears as both --plaintext-db
        and --golden-db in score_golden)."""
        step = PipelineStep(
            name='score_golden', script='evaluate_golden',
            args=('--plaintext-db', '{golden_db}',
                  '--predicted-db', '{annotations_db}',
                  '--golden-db', '{golden_ann_db}'),
            sequential=False,
        )
        cmd = render_step(step, self._vars(), force=False)
        self.assertEqual(
            cmd[2:],
            ['--plaintext-db', 'skol_golden',
             '--predicted-db', 'skol_ann',
             '--golden-db', 'skol_golden_ann_hand'],
        )


# ---------------------------------------------------------------------------
# --force rewriting
# ---------------------------------------------------------------------------


class TestForceFlagRewrite(unittest.TestCase):
    """Operator passes ``--force`` to runstep / runnext: any
    ``--skip-existing`` in the rendered argv flips to ``--force``,
    so the underlying script re-does work it would otherwise skip."""

    def _vars(self):
        return {'experiment': 'production', 'input_db': 'skol_dev'}

    def test_force_rewrites_skip_existing(self):
        step = PipelineStep(
            name='predict', script='predict_v4',
            args=('--experiment', '{experiment}',
                  '--source-db', '{input_db}',
                  '--skip-existing'),
        )
        cmd = render_step(step, self._vars(), force=True)
        self.assertIn('--force', cmd)
        self.assertNotIn('--skip-existing', cmd)

    def test_default_keeps_skip_existing(self):
        step = PipelineStep(
            name='predict', script='predict_v4',
            args=('--experiment', '{experiment}',
                  '--source-db', '{input_db}',
                  '--skip-existing'),
        )
        cmd = render_step(step, self._vars(), force=False)
        self.assertIn('--skip-existing', cmd)
        self.assertNotIn('--force', cmd)

    def test_force_does_not_touch_unrelated_flags(self):
        """--incremental, --output-database, etc. stay put."""
        step = PipelineStep(
            name='predict', script='predict_v4',
            args=('--experiment', '{experiment}',
                  '--incremental', '--skip-existing'),
        )
        cmd = render_step(step, self._vars(), force=True)
        self.assertIn('--incremental', cmd)

    def test_force_does_not_double_inject(self):
        """A step whose args already include --force (the v3 train
        step does) doesn't end up with two --force tokens."""
        step = PipelineStep(
            name='train', script='train_classifier',
            args=('--experiment', '{experiment}', '--force'),
        )
        cmd = render_step(step, self._vars(), force=True)
        self.assertEqual(cmd.count('--force'), 1)


# ---------------------------------------------------------------------------
# Script path resolution
# ---------------------------------------------------------------------------


class TestScriptResolution(unittest.TestCase):

    def test_first_two_tokens_are_python_and_script_path(self):
        step = PipelineStep(
            name='predict', script='predict_v4',
            args=('--experiment', '{experiment}'),
        )
        cmd = render_step(
            step, {'experiment': 'production'}, force=False,
        )
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].endswith('bin/predict_v4.py'))


# ---------------------------------------------------------------------------
# build_variables — the variable dict factory the pipeline runner uses
# ---------------------------------------------------------------------------


class TestBuildVariables(unittest.TestCase):
    """Variables flow from env_config / experiment doc into the
    substitution dict.  This pins which keys are exposed and which
    have safe defaults when the config doesn't override."""

    def test_includes_all_canonical_variables(self):
        vars_ = build_variables('production_v4', {})
        for key in (
            'experiment', 'input_db', 'training_db',
            'annotations_db', 'golden_db', 'golden_ann_db',
            'eval_ann_db',
            'model_key_single', 'model_key_pass1', 'model_key_pass2',
            'sbert_model',
        ):
            self.assertIn(key, vars_, f'missing variable {key}')
        self.assertEqual(vars_['experiment'], 'production_v4')

    def test_eval_ann_db_defaults_to_eval_suffixed_sibling(self):
        """Decision 2026-06-09: eval predictions go to a sibling
        DB by default with the ``_eval`` suffix.  Sorts directly
        next to the production DB in ``_all_dbs`` AND prevents
        eval runs from poisoning the production data the search
        UI reads."""
        vars_ = build_variables('x', {'annotations_db_name': 'shared_ann'})
        self.assertEqual(vars_['eval_ann_db'], 'shared_ann_eval')

    def test_eval_ann_db_override_wins(self):
        """Operators wanting the legacy shared-DB behaviour or a
        custom DB path set ``eval_annotations_db_name`` on the
        experiment doc — the explicit value wins."""
        vars_ = build_variables('x', {
            'annotations_db_name': 'prod_ann',
            'eval_annotations_db_name': 'eval_ann',
        })
        self.assertEqual(vars_['eval_ann_db'], 'eval_ann')

    def test_eval_ann_db_empty_when_no_annotations(self):
        """Defensive: with no annotations DB resolved, the eval
        sibling stays empty (rather than a bare ``_eval`` string
        that would otherwise fall out of f-string formatting)."""
        vars_ = build_variables('x', {})
        self.assertEqual(vars_['eval_ann_db'], '')

    def test_treatments_prose_and_structured_db_variables(self):
        """Decision 2026-06-09: the per-experiment treatment tiers
        get their own variable slots so pipeline steps can address
        them explicitly without falling through env_config in
        every consumer.  Defaults are empty (no opinion) — the
        per-experiment env_config resolution populates them at
        runtime."""
        vars_ = build_variables('x', {
            'treatments_prose_db_name': 'skol_exp_x_02_00_treatments_prose',
            'treatments_structured_db_name':
                'skol_exp_x_03_00_treatments_structured',
        })
        self.assertEqual(
            vars_['treatments_prose_db'],
            'skol_exp_x_02_00_treatments_prose',
        )
        self.assertEqual(
            vars_['treatments_structured_db'],
            'skol_exp_x_03_00_treatments_structured',
        )

    def test_safe_defaults_when_config_silent(self):
        """If env_config didn't pick up overrides, we don't crash —
        the canonical defaults from Step 6 / Step 7 apply.  Lets a
        fresh experiment work without exhaustively setting every
        knob."""
        vars_ = build_variables('production_v4', {})
        self.assertEqual(vars_['input_db'], 'skol_dev')
        self.assertEqual(vars_['golden_db'], 'skol_golden')
        self.assertEqual(vars_['sbert_model'], 'mpnet')


if __name__ == '__main__':
    unittest.main()
