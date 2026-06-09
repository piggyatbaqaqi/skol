"""Tests for bin/pipelines/v4_crf.py.

Pins the post-Step-7 cutover invariants that today's
``production_v4`` depends on:

* annotate + embed_lines are first-class prerequisite steps, not
  out-of-band manual setup.
* ``predict`` uses ``{input_db}`` (the ingest DB), NOT the
  ``{golden_db}`` — the Step-7-cutover invariant that ``b276fbd``
  patched at the dispatcher level and this file now bakes into
  the per-family pipeline.
* Evaluate is split into ``predict_golden`` + ``score_golden``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bin.pipelines import load  # noqa: E402
from bin.pipelines.base import build_variables, render_step  # noqa: E402


class TestV4CRFPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = load('v4_crf')

    def test_step_names_include_annotate_and_embed_lines_as_prereqs(self):
        names = tuple(s.name for s in self.pipeline)
        # First two are the v4-only prereqs Step 6.PRE used to run
        # out-of-band; everything else is the shared shape.
        self.assertEqual(names[:2], ('annotate', 'embed_lines'))
        self.assertIn('train', names)
        self.assertIn('predict', names)
        self.assertIn('predict_golden', names)
        self.assertIn('score_golden', names)
        self.assertNotIn('evaluate', names)   # split

    def test_annotate_step_renders_to_annotate_v4_database_input_db(self):
        step = next(s for s in self.pipeline if s.name == 'annotate')
        cmd = render_step(
            step,
            build_variables('production_v4', {'ingest_db_name': 'skol_dev'}),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/annotate_v4.py'))
        self.assertIn('--database', cmd)
        db_idx = cmd.index('--database')
        self.assertEqual(cmd[db_idx + 1], 'skol_dev')

    def test_embed_lines_step_uses_input_db_and_sbert_model(self):
        step = next(s for s in self.pipeline if s.name == 'embed_lines')
        cmd = render_step(
            step,
            build_variables('production_v4', {'ingest_db_name': 'skol_dev'}),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/embed_lines.py'))
        self.assertIn('--source-db', cmd)
        sd_idx = cmd.index('--source-db')
        self.assertEqual(cmd[sd_idx + 1], 'skol_dev')
        self.assertIn('--sbert-model', cmd)
        sm_idx = cmd.index('--sbert-model')
        self.assertEqual(cmd[sm_idx + 1], 'mpnet')

    def test_train_step_renders_to_train_crf_single_with_source_db(self):
        step = next(s for s in self.pipeline if s.name == 'train')
        cmd = render_step(
            step,
            build_variables('production_v4', {
                'training_database': 'skol_training_v3_combined_no_golden',
                'classifier_model_key_single':
                    'skol:classifier:model:v4_single_combined',
            }),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/train_crf_single.py'))
        self.assertIn('--source-db', cmd)
        sd_idx = cmd.index('--source-db')
        self.assertEqual(
            cmd[sd_idx + 1], 'skol_training_v3_combined_no_golden',
        )
        self.assertIn('--redis-key', cmd)
        rk_idx = cmd.index('--redis-key')
        self.assertEqual(
            cmd[rk_idx + 1], 'skol:classifier:model:v4_single_combined',
        )

    def test_predict_uses_ingest_db_not_golden(self):
        """Step-7-cutover invariant: ``predict`` runs over the full
        production corpus, not the 105-doc golden set.  Regressing
        this is the bug that started the pipeline restructure."""
        step = next(s for s in self.pipeline if s.name == 'predict')
        cmd = render_step(
            step,
            build_variables('production_v4', {
                'ingest_db_name': 'skol_dev',
                'golden_db_name': 'skol_golden_v2',
            }),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/predict_v4.py'))
        self.assertIn('--source-db', cmd)
        sd_idx = cmd.index('--source-db')
        self.assertEqual(cmd[sd_idx + 1], 'skol_dev')   # NOT golden
        self.assertNotIn('skol_golden_v2', cmd)

    def test_predict_golden_uses_eval_ann_db_for_output(self):
        step = next(s for s in self.pipeline if s.name == 'predict_golden')
        self.assertFalse(step.sequential)
        cmd = render_step(
            step,
            build_variables('production_v4', {
                'golden_db_name': 'skol_golden_v2',
                'annotations_db_name': 'skol_exp_production_v4_ann_combined',
            }),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/predict_v4.py'))
        self.assertIn('--source-db', cmd)
        sd_idx = cmd.index('--source-db')
        self.assertEqual(cmd[sd_idx + 1], 'skol_golden_v2')
        # Output DB defaults to the production annotations DB so
        # the 105 golden predictions land alongside production —
        # operators who want a separate eval DB set
        # eval_annotations_db_name on the experiment doc.
        self.assertIn('--output-database', cmd)
        od_idx = cmd.index('--output-database')
        self.assertEqual(
            cmd[od_idx + 1], 'skol_exp_production_v4_ann_combined',
        )

    def test_score_golden_uses_golden_ann_and_plaintext_dbs(self):
        step = next(s for s in self.pipeline if s.name == 'score_golden')
        self.assertFalse(step.sequential)
        cmd = render_step(
            step,
            build_variables('production_v4', {
                'golden_db_name': 'skol_golden_v2',
                'golden_ann_db_name': 'skol_golden_ann_hand_v2',
            }),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/evaluate_golden.py'))
        self.assertIn('--plaintext-db', cmd)
        self.assertIn('skol_golden_v2', cmd)
        self.assertIn('--golden-db', cmd)
        self.assertIn('skol_golden_ann_hand_v2', cmd)
        self.assertIn('--save-to-experiment', cmd)

    def test_sequential_block_includes_annotate_and_embed_lines(self):
        """Both v4 prereqs go in the sequential block — they need
        to complete before predict can run."""
        sequential = tuple(s.name for s in self.pipeline if s.sequential)
        self.assertIn('annotate', sequential)
        self.assertIn('embed_lines', sequential)
        self.assertIn('predict', sequential)


if __name__ == '__main__':
    unittest.main()
