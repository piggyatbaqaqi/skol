"""v3 logistic-classifier pipeline.

The legacy default pipeline that production / production_v3_hand /
production_v3_jats use.  Step list matches the pre-refactor
``_PIPELINE_STEPS`` constant, with the single legacy ``evaluate``
step split into ``predict_golden`` + ``score_golden`` per the
restructure plan.
"""
from __future__ import annotations

from bin.pipelines.base import PipelineStep


PIPELINE = (
    PipelineStep(
        name='train',
        script='train_classifier',
        args=('--experiment', '{experiment}', '--force'),
    ),
    PipelineStep(
        name='predict',
        script='predict_classifier',
        args=(
            '--experiment', '{experiment}',
            '--incremental', '--skip-existing',
        ),
    ),
    PipelineStep(
        name='annotate_jats',
        script='jats_to_yedda',
        args=(
            '--experiment', '{experiment}',
            '--all', '--output-to', 'couchdb',
            '--skip-existing',
        ),
    ),
    PipelineStep(
        name='extract_treatments',
        script='extract_treatments_to_couchdb',
        args=('--experiment', '{experiment}', '--skip-existing'),
    ),
    PipelineStep(
        name='embed',
        script='embed_treatments',
        # Belt-and-suspenders with env_config: --expire None forces
        # persistent embeddings even when EMBEDDING_EXPIRE is set
        # in the cron environment.  Matches the v1 cron's behaviour.
        args=(
            '--experiment', '{experiment}',
            '--force', '--expire', 'None',
        ),
    ),
    PipelineStep(
        name='treatments_to_json',
        script='treatments_to_json',
        args=(
            '--experiment', '{experiment}',
            '--incremental', '--skip-existing',
            '--use-constrained-decoding', '--graceful-degradation',
            '--timeout', '1200',
        ),
    ),
    PipelineStep(
        name='annotate_spans',
        script='annotate_spans',
        args=('--experiment', '{experiment}', '--skip-existing'),
    ),
    PipelineStep(
        name='predict_golden',
        script='predict_classifier',
        sequential=False,
        args=(
            '--experiment', '{experiment}',
            '--golden-db', '{golden_db}',
            '--skip-existing',
        ),
    ),
    PipelineStep(
        name='score_golden',
        script='evaluate_golden',
        sequential=False,
        args=(
            '--experiment', '{experiment}',
            '--golden-db', '{golden_ann_db}',
            '--plaintext-db', '{golden_db}',
            '--save-to-experiment',
        ),
    ),
    PipelineStep(
        name='build_vocab',
        script='build_vocab_tree',
        sequential=False,
        args=('--experiment', '{experiment}'),
    ),
    PipelineStep(
        name='build_sources_stats',
        script='build_sources_stats',
        sequential=False,
        args=('--experiment', '{experiment}', '--verbosity', '2'),
    ),
)
