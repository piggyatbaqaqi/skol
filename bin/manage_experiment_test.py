"""Tests for bin/manage_experiment.py — the evaluate-step builder.

Focused on Step 1.C of docs/golden_v2_plan.md: ``_build_step_commands``
must no longer hardcode "skol_golden" / "skol_golden_ann_hand"; it must
read the values from the resolved config (which flowed in from the
experiment doc via ``_apply_experiment`` in env_config).
"""

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manage_experiment import (  # type: ignore[import]  # noqa: E402
    _build_step_commands,
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


class TestEvaluateStepGoldenWiring:
    """The evaluate step builds two commands (predict-golden, evaluate);
    both must use the per-experiment golden DB names from config."""

    def test_uses_v1_defaults_when_unset(self) -> None:
        """With v1-default config the resulting commands match the
        pre-rewire behaviour — backward compatibility guarantee."""
        cfg = _config()
        cmds = _build_step_commands(
            "evaluate", "production", force=False, config=cfg,
        )
        args = _flatten(cmds)
        assert "skol_golden" in args
        assert "skol_golden_ann_hand" in args

    def test_uses_v2_values_when_config_overrides(self) -> None:
        """A v2 experiment whose doc points at the v2 golden DBs results in
        the v2 names showing up in the commands."""
        cfg = _config(
            golden_db_name='skol_golden_v2',
            golden_ann_db_name='skol_golden_ann_hand_v2',
        )
        cmds = _build_step_commands(
            "evaluate", "production_v2", force=False, config=cfg,
        )
        args = _flatten(cmds)
        assert "skol_golden_v2" in args
        assert "skol_golden_ann_hand_v2" in args
        # Old hardcoded literals must not survive in the rewired version.
        assert "skol_golden_ann_hand" not in args  # superseded by v2 value

    def test_jats_experiment_scores_against_jats_silver(self) -> None:
        """jats_v1's experiment doc carries databases.golden_ann =
        skol_golden_ann_jats; the evaluate step must pick that up
        instead of the hand standard the hardcoded literal used."""
        cfg = _config(golden_ann_db_name='skol_golden_ann_jats')
        cmds = _build_step_commands(
            "evaluate", "jats_v1", force=False, config=cfg,
        )
        args = _flatten(cmds)
        assert "skol_golden_ann_jats" in args
        # The plaintext-db arg still uses golden_db_name (plaintext source
        # is the same DB across hand/silver tracks).
        assert "skol_golden" in args

    def test_predict_command_uses_golden_db_name(self) -> None:
        """The predict_classifier.py invocation carries --golden-db pointing
        at the plaintext DB (databases.golden), not the answer-key DB."""
        cfg = _config(
            golden_db_name='skol_golden_v2',
            golden_ann_db_name='skol_golden_ann_hand_v2',
        )
        cmds = _build_step_commands(
            "evaluate", "production_v2", force=False, config=cfg,
        )
        predict_cmd = next(
            c for c in cmds if any('predict_classifier' in a for a in c)
        )
        # Find the --golden-db value in the predict command.
        i = predict_cmd.index('--golden-db')
        assert predict_cmd[i + 1] == 'skol_golden_v2'

    def test_evaluate_command_separates_golden_and_plaintext(self) -> None:
        """evaluate_golden.py gets --golden-db = answer key,
        --plaintext-db = source plaintext DB."""
        cfg = _config(
            golden_db_name='skol_golden_v2',
            golden_ann_db_name='skol_golden_ann_jats_v2',
        )
        cmds = _build_step_commands(
            "evaluate", "jats_v2", force=False, config=cfg,
        )
        eval_cmd = next(
            c for c in cmds if any('evaluate_golden' in a for a in c)
        )
        # answer-key DB (--golden-db on evaluate_golden) = golden_ann
        gi = eval_cmd.index('--golden-db')
        assert eval_cmd[gi + 1] == 'skol_golden_ann_jats_v2'
        # plaintext DB (--plaintext-db on evaluate_golden) = golden
        pi = eval_cmd.index('--plaintext-db')
        assert eval_cmd[pi + 1] == 'skol_golden_v2'


class TestEvaluateStepForceFlag:
    """Sanity check that the --force semantics still work after the rewire."""

    def test_force_replaces_skip_existing(self) -> None:
        cfg = _config()
        cmds = _build_step_commands(
            "evaluate", "production", force=True, config=cfg,
        )
        args = _flatten(cmds)
        assert "--force" in args
        assert "--skip-existing" not in args


class TestOtherStepsUnaffected:
    """Non-evaluate steps don't need golden DB names — they should build
    successfully even when config doesn't carry the new keys."""

    def test_train_step_does_not_require_golden_keys(self) -> None:
        cmds = _build_step_commands(
            "train", "production", force=False, config={},
        )
        # Just verifying it runs without KeyError.
        assert len(cmds) == 1
        assert any('train_classifier' in a for a in cmds[0])

    def test_embed_step_does_not_require_golden_keys(self) -> None:
        cmds = _build_step_commands(
            "embed", "production", force=False, config={},
        )
        assert len(cmds) == 1
        assert any('embed_treatments' in a for a in cmds[0])


class TestPredictStepV4Dispatch:
    """v5 Step 5.C: the 'predict' step dispatches on model_name —
    v4_crf → predict_v4.py; everything else → predict_classifier.py.
    Both v3 and v4 experiments use the same step name so the
    pipeline shape stays uniform."""

    def test_predict_step_dispatches_to_predict_classifier_for_v3(
        self,
    ) -> None:
        cmds = _build_step_commands(
            "predict", "production", force=False,
            config={'model_name': 'logistic_sections'},
        )
        assert len(cmds) == 1
        args = cmds[0]
        assert any('predict_classifier' in a for a in args)
        assert not any('predict_v4' in a for a in args)

    def test_predict_step_dispatches_to_predict_v4_for_v4_crf(
        self,
    ) -> None:
        cmds = _build_step_commands(
            "predict", "production_v4", force=False,
            config={'model_name': 'v4_crf'},
        )
        assert len(cmds) == 1
        args = cmds[0]
        assert any('predict_v4' in a for a in args)
        assert not any('predict_classifier' in a for a in args)

    def test_predict_step_defaults_to_v3_when_model_name_absent(
        self,
    ) -> None:
        """An experiment doc without model_name (legacy / unset) keeps
        the v3 path — no surprise upgrades."""
        cmds = _build_step_commands(
            "predict", "production", force=False, config={},
        )
        assert len(cmds) == 1
        assert any('predict_classifier' in a for a in cmds[0])

    def test_evaluate_step_predict_golden_also_dispatches(self) -> None:
        """The evaluate step's first command (predict_golden) must
        respect the same dispatch — otherwise v4 experiments would
        evaluate against v3 predictions."""
        cmds = _build_step_commands(
            "evaluate", "production_v4", force=False,
            config={
                'model_name': 'v4_crf',
                'golden_db_name': 'skol_golden_v2',
                'golden_ann_db_name': 'skol_golden_ann_hand_v2',
            },
        )
        # evaluate emits two commands: predict_golden then evaluate.
        assert len(cmds) == 2
        predict_golden_args = cmds[0]
        assert any('predict_v4' in a for a in predict_golden_args)
        assert not any(
            'predict_classifier' in a for a in predict_golden_args
        )
