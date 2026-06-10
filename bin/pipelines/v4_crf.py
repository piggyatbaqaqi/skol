"""v4 SBERT + CRF pipeline.

Two more steps than ``v3_logistic`` at the start of the sequential
block — ``annotate`` (Step 1.D's ``annotate_v4`` script — writes
``article.spans.v4.json`` and ``article.page-headers.json``
attachments) and ``embed_lines`` (Step 0's SBERT cache populator).
Both must complete before ``predict_v4`` can compute features.
Step 6.PRE used to run these out-of-band; making them first-class
pipeline steps prevents the recurring "missing spans /
page-headers — re-run annotate_v4" failure mode.

Otherwise the same shape as ``v3_logistic`` with evaluate split.
``train`` invokes ``train_crf_single`` against the combined corpus
(Step 7's production-pinned single-CRF model); ``predict`` runs
``predict_v4`` against the ingest DB explicitly so the Step-7
cutover's full-corpus invariant is baked into the family file.
"""
from __future__ import annotations

from bin.pipelines.base import PipelineStep


PIPELINE = (
    # --- v4-specific prereqs (Step 6.PRE), now first-class steps ---
    PipelineStep(
        name='annotate',
        script='annotate_v4',
        args=('--database', '{input_db}'),
    ),
    PipelineStep(
        name='embed_lines',
        script='embed_lines',
        args=(
            '--source-db', '{input_db}',
            '--sbert-model', '{sbert_model}',
        ),
    ),

    # --- shared sequential block ---
    PipelineStep(
        name='train',
        script='train_crf_single',
        args=(
            '--source-db', '{training_db}',
            '--redis-key', '{model_key_single}',
        ),
    ),
    PipelineStep(
        name='predict',
        script='predict_v4',
        args=(
            '--experiment', '{experiment}',
            '--source-db', '{input_db}',     # Step-7-cutover invariant
            '--skip-existing',
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

    # --- trailing steps (sequential=False) ---
    PipelineStep(
        name='predict_golden',
        script='predict_v4',
        sequential=False,
        args=(
            '--experiment', '{experiment}',
            '--source-db', '{golden_db}',
            '--output-database', '{eval_ann_db}',
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
