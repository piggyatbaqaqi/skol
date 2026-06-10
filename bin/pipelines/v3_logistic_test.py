"""Tests for bin/pipelines/v3_logistic.py.

Asserts the pipeline matches the post-evaluate-split shape today's
v3 experiments need: same step names as the legacy
``_PIPELINE_STEPS`` list except ``evaluate`` splits into
``predict_golden`` + ``score_golden``.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bin.pipelines import load  # noqa: E402
from bin.pipelines.base import build_variables, render_step  # noqa: E402


class TestV3LogisticPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = load('v3_logistic')

    def test_step_names_match_current_v3_pipeline(self):
        """The legacy list was: train, predict, annotate_jats,
        extract_treatments, embed, treatments_to_json, annotate_spans,
        evaluate, build_vocab, build_sources_stats.  Evaluate
        splits into predict_golden + score_golden — everything
        else stays."""
        names = tuple(s.name for s in self.pipeline)
        self.assertEqual(names, (
            'train', 'predict', 'annotate_jats', 'extract_treatments',
            'embed', 'treatments_to_json', 'annotate_spans',
            'predict_golden', 'score_golden',
            'build_vocab', 'build_sources_stats',
        ))

    def test_sequential_steps_form_the_linear_block(self):
        """train → annotate_spans are sequential; the trailing
        four (predict_golden, score_golden, build_vocab,
        build_sources_stats) can run after the sequential block
        in any order."""
        sequential = tuple(s.name for s in self.pipeline if s.sequential)
        self.assertEqual(sequential, (
            'train', 'predict', 'annotate_jats', 'extract_treatments',
            'embed', 'treatments_to_json', 'annotate_spans',
        ))

    def test_train_step_renders_to_train_classifier_with_force(self):
        step = next(s for s in self.pipeline if s.name == 'train')
        cmd = render_step(
            step,
            build_variables('production', {}),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/train_classifier.py'))
        self.assertIn('--experiment', cmd)
        self.assertIn('production', cmd)
        self.assertIn('--force', cmd)

    def test_predict_step_renders_with_incremental_and_skip_existing(self):
        step = next(s for s in self.pipeline if s.name == 'predict')
        cmd = render_step(
            step,
            build_variables('production', {}),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/predict_classifier.py'))
        self.assertIn('--incremental', cmd)
        self.assertIn('--skip-existing', cmd)

    def test_predict_golden_renders_with_golden_db(self):
        step = next(
            s for s in self.pipeline if s.name == 'predict_golden'
        )
        self.assertFalse(step.sequential)
        cmd = render_step(
            step,
            build_variables('production', {
                'golden_db_name': 'skol_golden_v2',
            }),
            force=False,
        )
        self.assertTrue(cmd[1].endswith('bin/predict_classifier.py'))
        self.assertIn('--golden-db', cmd)
        gd_idx = cmd.index('--golden-db')
        self.assertEqual(cmd[gd_idx + 1], 'skol_golden_v2')

    def test_score_golden_renders_evaluate_golden_with_dual_dbs(self):
        step = next(s for s in self.pipeline if s.name == 'score_golden')
        self.assertFalse(step.sequential)
        cmd = render_step(
            step,
            build_variables('production', {
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


if __name__ == '__main__':
    unittest.main()
