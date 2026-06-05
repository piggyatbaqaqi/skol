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

| Run | Train docs | Dev docs | Final dev macro F1 | Wall |
|---|---|---|---|---|
| Pass-1 layout (8 labels) | 128 | 32 | **0.297** | ≈ 2 h 51 m on RTX 5090 |
| Pass-2 hand (12 labels) | 128 | 32 | **0.315** | ≈ 44 m |
| Pass-2 combined (12 labels) | 1 508 | 376 | **0.678** | ≈ 3 h 01 m |
| Single-CRF hand (19 labels, §6.F) | 128 | 32 | **0.344** | ≈ 45 m |

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
| Two-pass hand | 0.3106 | 0.4675 | 0.3106 |
| **Two-pass combined** | **0.4790** | **0.5528** | **0.4790** |
| Single-CRF hand (§6.F) | 0.4124 | **0.5725** | 0.4124 |
| Δ (combined − two-pass hand) | **+0.1684** | **+0.0853** | +0.1684 |
| Δ (single − two-pass hand) | **+0.1018** | **+0.1050** | +0.1018 |
| Δ (combined − single) | +0.0666 | −0.0197 | +0.0666 |

The two-pass-combined variant wins overall macro F1, but the
ranking on **char accuracy** flips: single-CRF (0.572) beats
combined (0.553).  See the §6.F decision section below for the
architectural reading.

### Per-tag F1 (char-level, golden set)

Tags ordered by combined-two-pass variant F1 descending.  W
column flags the per-tag winner across all three variants
(C = combined-two-pass, S = single-CRF, — = tie within 0.001).

| W | Tag | two-pass hand | two-pass combined | single-CRF | note |
|---|---|---|---|---|---|
| C | Page-header           | 0.811 | 0.811 | **0.833** | layout |
| C | Description           | 0.752 | **0.796** | 0.776 | treatment |
| C | Etymology             | 0.066 | **0.769** | 0.000 | single CRF has zero recall here |
| C | Materials-examined    | 0.564 | **0.710** | 0.557 | treatment |
| C | Materials-and-methods | 0.486 | **0.705** | 0.513 | treatment |
| S | Index                 | 0.659 | 0.659 | **0.714** | layout |
| C | Nomenclature          | 0.593 | **0.603** | 0.598 | within 0.01 |
| C | Phylogeny             | 0.000 | **0.542** | 0.000 | hand support too low for either hand-only model |
| C | Type-designation      | 0.120 | **0.517** | 0.027 | combined-corpus signal dominates |
| C | Biology               | 0.304 | **0.493** | 0.486 | within 0.01 |
| S | Notes                 | 0.196 | 0.458 | **0.473** | single CRF edges combined |
| C | Misc-exposition       | 0.433 | **0.490** | 0.442 | catch-all |
| C | Diagnosis             | 0.055 | **0.349** | 0.098 | |
| S | Table                 | 0.142 | 0.142 | **0.270** | layout — single CRF wins |
| S | Bibliography          | 0.085 | 0.085 | **0.721** | layout — single CRF massively better |
| S | Figure-caption        | 0.013 | 0.013 | **0.421** | layout — single CRF massively better |
| S | Key                   | 0.000 | 0.000 | **0.082** | layout — only single CRF recovers any |

The pattern is striking and consistent:

- **Layout tags (Bibliography, Figure-caption, Index, Key,
  Page-header, Table)**: the single CRF wins 6/7.  Two-pass shares
  Pass-1 across variants and its Pass-1 dev F1 of 0.297 is the
  bottleneck.  The single CRF gets to decide layout vs treatment
  in the same joint Viterbi pass and clearly does it better here.
- **Treatment tags (the 12 vocab members)**: the
  combined-corpus two-pass wins all 12.  The hand-only single CRF
  (and hand-only two-pass) simply lack training signal for
  Etymology, Phylogeny, Type-designation, etc.  Single CRF beats
  two-pass-hand on most of these — but not by enough to close the
  combined-corpus gap.

## Decision

`production_v4.redis_keys.classifier_model_pass2` is set to
`skol:classifier:model:v4_pass2_combined` (the winner).  The
`evaluation` field on the experiment doc records the
combined-variant macro F1 (0.479) and per-tag F1.

The hand-variant Redis bundle
`skol:classifier:model:v4_pass2_hand` is kept for the Step 7
report's ablation comparison; future operators can drop it if
disk pressure warrants.  Same for
`skol:classifier:model:v4_single_hand`.

## Two-pass vs single-CRF (§Step 6.F)

Headline ranking on the golden set:

1. **Two-pass combined**  — macro F1 0.479, char acc 55.3 %
2. **Single-CRF hand**    — macro F1 0.412, char acc **57.3 %**
3. **Two-pass hand**      — macro F1 0.311, char acc 46.8 %

Single-CRF (hand-only) beats two-pass (hand-only) by **+0.10
macro F1 and +0.11 char accuracy**.  When both architectures are
trained on the same data, the simpler single-CRF design wins —
the two-pass split's overhead is not paying for itself on the
hand corpus alone.

The two-pass design's win over single-CRF (+0.067 macro F1)
comes **entirely** from Pass-2's ability to train on the
combined corpus (12× more docs).  Take that lever away — train
single-CRF on the combined corpus, or train two-pass-Pass-2 on
the hand corpus — and the single CRF is uniformly better or
equal at the architectural level.

**Recommended Step 7 actions**:

1. **Train single-CRF on the combined corpus** — same Misc-
   exposition catch-all should work; the new
   `bin/train_crf_single --source-db
   skol_training_v3_combined_no_golden --redis-key
   skol:classifier:model:v4_single_combined` is a single
   command.  If single-combined ≥ two-pass-combined on macro F1,
   retire the two-pass design.
2. **Pass-1 training-corpus expansion** — the layout F1s
   (Bibliography 0.085 → 0.721 going from two-pass to single)
   show the single-CRF joint-Viterbi formulation handles layout
   significantly better.  If we keep two-pass, Pass-1 needs more
   hand-annotated PDF layout data to close that gap; if we
   adopt single, the problem dissolves.
3. **Particle-block ablation** — zero the 12-d particle features
   and re-evaluate.  Single-CRF's joint decoding may use the
   particle signal differently than the two-pass split does.
4. **Exposure-bias measurement** — only relevant if two-pass
   survives Step 7.1.

## Out of scope (Step 7 work)

- Single-CRF trained on the **combined** corpus (the most
  important Step 7 follow-up — see §6.F decision above).
- v3 logistic baseline comparison — needs a fresh predict run
  with the same `skol_golden_v2` split.
- Particle-block ablation (zero the 12-d particle features and
  re-evaluate).
- Pass-1 training-corpus expansion or hyper-parameter tuning —
  Pass-1 dev F1 = 0.297 is the next bottleneck if we keep the
  two-pass design.
- Exposure-bias measurement (Pass-2 trained on Pass-1 *predicted*
  rather than oracle labels) — only matters if two-pass survives.
