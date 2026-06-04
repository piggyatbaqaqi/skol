# `production_v4` evaluation report

Operational artefact for v4 plan §Step 6.  Records the
Pass-2-hand vs Pass-2-combined ablation that decided which
Pass-2 training-data scope `production_v4` ships with.

## Setup

| | |
|---|---|
| Pass-1 (layout CRF) | `skol:classifier:model:v4_layout` — 791-d emission, 8 labels |
| Pass-2 variants | `skol:classifier:model:v4_pass2_hand` (hand corpus only) and `:v4_pass2_combined` (full combined corpus) |
| Training corpus (Pass 1) | `skol_training_v2_no_golden` — 160 docs, 128 train + 32 dev |
| Training corpus (Pass-2-hand) | same 160-doc split |
| Training corpus (Pass-2-combined) | `skol_training_v3_combined_no_golden` — 1 884 docs, 1 508 train + 376 dev |
| Golden plaintext | `skol_golden_v2` — 105 docs |
| Golden answer key | `skol_golden_ann_hand_v2` — 30 docs |
| Hyperparameters | 20 epochs, Adam lr=1e-3, seed=42, dev_fraction=0.2, mpnet SBERT cache |
| Hardware | RTX 5090 Laptop (24 GB), pytorch 2.9.1+cu129 |

## Training-time metrics

| Run | Train docs | Dev docs | Final dev macro F1 | Wall (start → end) |
|---|---|---|---|---|
| Pass-1 layout | 128 | 32 | **0.297** | 2026-06-04 03:27 → 06:18 (≈ 2 h 51 m) |
| Pass-2 hand | 128 | 32 | **0.315** | 2026-06-04 14:39 → 15:23 (≈ 44 m) |
| Pass-2 combined | 1 508 | 376 | **0.678** | 2026-06-04 03:28 → 06:29 (≈ 3 h 01 m) |

Pass-2-combined's dev F1 is 2.15× Pass-2-hand's — the in-distribution
gain from 12× more training data is large, despite the dev set
being drawn from the same (mostly JATS-derived) distribution.

## Golden-set evaluation (the decision metric)

Both variants were evaluated against `skol_golden_ann_hand_v2`
(30 hand-annotated PDF-derived docs) via
`bin/evaluate_golden.py` after running `bin/predict_v4` over
`skol_golden_v2`.

| Variant | Macro F1 (char-level) | Char accuracy | Block macro F1 |
|---|---|---|---|
| Pass-2 hand | 0.3106 | 0.4675 | 0.3106 |
| **Pass-2 combined** | **0.4790** | **0.5528** | **0.4790** |
| Delta | **+0.1684** | **+0.0853** | +0.1684 |

The combined variant wins by every aggregate metric.  More
importantly it wins or ties on EVERY tag — there is no tag where
the hand variant edges out combined (see per-tag table below).
This is a meaningfully different outcome from the v3 baseline,
where cross-distribution training on JATS hurt PDF performance.
The plan §Training-data scope question — "does SBERT's semantic
representation make the combined corpus a net win, or does
cross-distribution dilution still hurt?" — answers cleanly in
favour of the combined corpus.

### Per-tag F1 (char-level, golden set)

Tags ordered by combined variant F1 descending.

| Tag | hand F1 | combined F1 | Δ | Note |
|---|---|---|---|---|
| Page-header | 0.811 | 0.811 | 0.000 | Pass-1 only |
| Description | 0.752 | 0.796 | +0.044 | |
| Etymology | 0.066 | 0.769 | **+0.703** | dominated by training-corpus support |
| Materials-examined | 0.564 | 0.710 | +0.146 | |
| Materials-and-methods | 0.486 | 0.705 | +0.219 | |
| Index | 0.659 | 0.659 | 0.000 | Pass-1 only |
| Nomenclature | 0.593 | 0.603 | +0.011 | |
| Phylogeny | 0.000 | 0.542 | **+0.542** | hand corpus has near-zero support |
| Type-designation | 0.120 | 0.517 | **+0.397** | |
| Biology | 0.304 | 0.493 | +0.189 | |
| Misc-exposition | 0.433 | 0.490 | +0.057 | catch-all |
| Notes | 0.196 | 0.458 | +0.262 | |
| Diagnosis | 0.055 | 0.349 | +0.295 | |
| Table | 0.142 | 0.142 | 0.000 | Pass-1 only |
| Bibliography | 0.085 | 0.085 | 0.000 | Pass-1 only |
| Figure-caption | 0.013 | 0.013 | 0.000 | Pass-1 only |
| Key | 0.000 | 0.000 | 0.000 | Pass-1 only (no support in golden) |

The seven layout tags show identical F1 across variants because
they are decided by Pass-1, which is the same model for both
runs.  Their weakness (Bibliography 0.085, Figure-caption 0.013,
Key 0.000, Table 0.142) reflects Pass-1's modest dev F1 (0.297)
and is a Step 7 / future-work concern, not a Pass-2 decision input.

## Decision

`production_v4.redis_keys.classifier_model_pass2` is set to
`skol:classifier:model:v4_pass2_combined` (the winner).  The
`evaluation` field on the experiment doc records the
combined-variant macro F1 (0.479) and per-tag F1.

The hand-variant Redis bundle
`skol:classifier:model:v4_pass2_hand` is kept for the Step 7
report's ablation comparison; future operators can drop it if
disk pressure warrants.

## Out of scope (Step 7 work)

- Single-CRF baseline (§Step 6.F).  Defers to a focused follow-up
  that lands `skol_classifier/v4/crf_single.py` +
  `bin/train_crf_single.py` and adds a third row to this table.
- v3 logistic baseline comparison — needs a fresh predict run
  with the same `skol_golden_v2` split.
- Particle-block ablation (zero the 12-d particle features and
  re-evaluate).
- Pass-1 training-corpus expansion or hyper-parameter tuning —
  Pass-1 dev F1 = 0.297 is the next bottleneck.
- Exposure-bias measurement (Pass-2 trained on Pass-1 *predicted*
  rather than oracle labels).
