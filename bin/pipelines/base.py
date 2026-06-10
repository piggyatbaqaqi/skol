"""Pipeline step model + rendering helpers.

Each per-family pipeline module (``bin/pipelines/v3_logistic.py``,
``bin/pipelines/v4_crf.py``, …) exports a tuple of
:class:`PipelineStep` records describing the canonical ordered
step list for that family.  ``bin/manage_experiment.py`` looks up
the module via ``bin/pipelines/__init__.py:load(name)`` based on
the experiment doc's ``pipeline`` field, then calls
:func:`render_step` to turn a step + a config dict into an argv
list ready for ``subprocess``.

This file is the contract between the dispatcher and the family
modules.  Keep it small.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStep:
    """One canonical step in a family's pipeline.

    Fields
    ------
    name : str
        Pipeline-step identifier matching the experiment doc's
        ``pipeline.steps[].name``.  Cron + operator commands refer
        to this string.
    script : str
        Script in ``bin/`` (no ``.py``, no path) to invoke.  E.g.
        ``'predict_v4'`` → ``bin/predict_v4.py``.
    args : tuple[str, ...]
        Argv tokens after the script.  Tokens containing ``'{var}'``
        get ``.format(**variables)``-substituted at render time;
        unknown variable names raise ``KeyError`` so config drift
        fails fast instead of producing literal-brace argv.
    sequential : bool, default True
        Whether the step is in the linear pipeline block (default)
        or one of the trailing steps (evaluate / build_vocab /
        build_sources_stats) that may run after the sequential
        block in any order.
    """

    name: str
    script: str
    args: Tuple[str, ...]
    sequential: bool = True


# ---------------------------------------------------------------------------
# Variable factory
# ---------------------------------------------------------------------------


def build_variables(
    experiment_name: str, config: Mapping[str, Any],
) -> Dict[str, str]:
    """Compose the substitution-variable dict for one experiment.

    The whitelist is curated — adding a new variable is a one-line
    addition here.  Auto-injection of arbitrary config keys is
    intentionally avoided so secrets and other surprises stay out
    of argv.

    Defaults are chosen to match Step 6 / Step 7's
    operational state so a freshly-created experiment doc that
    omits most fields still produces sensible commands.
    """
    annotations = config.get('annotations_db_name', '')
    treatments_prose = config.get('treatments_prose_db_name', '')
    treatments_structured = config.get('treatments_structured_db_name', '')
    # Eval predictions go to a sibling DB by default — the
    # ``_eval`` suffix keeps them sorted directly next to their
    # production counterparts in ``_all_dbs`` while preventing
    # eval runs from poisoning the production data the search
    # UI reads.  Operators wanting the legacy shared-DB behaviour
    # set ``eval_annotations_db_name`` on the doc.  Convention
    # settled 2026-06-09; see docs/skol-db-naming-cleanup.md.
    default_eval_ann = f'{annotations}_eval' if annotations else ''
    return {
        'experiment':       experiment_name,
        'input_db':         config.get('ingest_db_name', 'skol_dev'),
        'training_db':      config.get(
            'training_database',
            'skol_training_v3_combined_no_golden',
        ),
        'annotations_db':           annotations,
        'treatments_prose_db':      treatments_prose,
        'treatments_structured_db': treatments_structured,
        'golden_db':        config.get('golden_db_name', 'skol_golden'),
        'golden_ann_db':    config.get(
            'golden_ann_db_name', 'skol_golden_ann_hand',
        ),
        'eval_ann_db':      config.get(
            'eval_annotations_db_name', default_eval_ann,
        ),
        'model_key_single': config.get(
            'classifier_model_key_single',
            'skol:classifier:model:v4_single_combined',
        ),
        'model_key_pass1':  config.get(
            'classifier_model_key_pass1',
            'skol:classifier:model:v4_layout',
        ),
        'model_key_pass2':  config.get(
            'classifier_model_key_pass2',
            'skol:classifier:model:v4_pass2_combined',
        ),
        'sbert_model':      'mpnet',
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


_BIN_DIR = Path(__file__).resolve().parent.parent


def render_step(
    step: PipelineStep,
    variables: Mapping[str, str],
    *,
    force: bool = False,
) -> List[str]:
    """Turn a :class:`PipelineStep` into a ready-to-Popen argv list.

    ``[sys.executable, str(_BIN_DIR / f'{step.script}.py'),
       *substituted_args]``.

    When ``force`` is True, every standalone ``--skip-existing``
    token flips to ``--force``.  Steps that already include
    ``--force`` are unaffected.
    """
    cmd: List[str] = [
        sys.executable,
        str(_BIN_DIR / f'{step.script}.py'),
    ]
    for token in step.args:
        # ``{var}`` substitution is opt-in via the brace syntax;
        # bare tokens pass through.  We let ``.format`` decide on
        # the keys present in ``variables`` — unknown names raise
        # ``KeyError`` which bubbles up to ``render_step``'s caller.
        if '{' in token:
            cmd.append(token.format(**variables))
        else:
            cmd.append(token)
    if force:
        cmd = ['--force' if t == '--skip-existing' else t for t in cmd]
    return cmd
