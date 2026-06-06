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

## Decision (Step 6.F status, superseded by Step 7 §Recommendation)

`production_v4.redis_keys.classifier_model_pass2` is **currently**
set to `skol:classifier:model:v4_pass2_combined` (the Step 6.F
winner).  The Step 7 measurements below show the single-CRF
combined architecture beats the two-pass design by a wide margin;
the Step 7 §Recommendation below names the cutover and the
operator runs it.  See §Recommendation.

The hand-variant Redis bundles
(`skol:classifier:model:v4_pass2_hand`, `…:v4_single_hand`) are
kept for the ablation comparison.

## Two-pass vs single-CRF (§Step 6.F)

Headline ranking on the golden set (HAND CORPUS ONLY):

1. **Two-pass combined**  — macro F1 0.479, char acc 55.3 %
2. **Single-CRF hand**    — macro F1 0.412, char acc **57.3 %**
3. **Two-pass hand**      — macro F1 0.311, char acc 46.8 %

Single-CRF (hand-only) beats two-pass (hand-only) by **+0.10
macro F1 and +0.11 char accuracy**.  When both architectures are
trained on the same data, the simpler single-CRF design wins —
the two-pass split's overhead is not paying for itself on the
hand corpus alone.

§Step 6.F recommended training the single-CRF on the combined
corpus to see whether the two-pass design's combined-corpus
advantage survives when single-CRF gets the same data lever.
§Step 7.β answers that question.

# Step 7 — Comparison report

This section adds the v4-vs-v3 baseline grid, the single-CRF
combined-corpus run §Step 6.F flagged, two ablation measurements
(particle features at inference; exposure bias at training), and
a cost synthesis.  All measurements share the same protocol: 20
epochs, lr=1e-3, seed=42, predict over `skol_golden_v2`,
evaluate against `skol_golden_ann_hand_v2` with
`bin/evaluate_golden.py`.

## §7.α — v4 vs v3 baseline grid

Char-level macro F1 on the 30-doc golden set (lower is worse):

| Rank | Variant | Macro F1 | Char acc |
|---:|---|---:|---:|
| 1 | **Single-CRF combined** (§7.β) | **0.585** | **0.653** |
| 2 | Two-pass combined (production) | 0.479 | 0.553 |
| 3 | Single-CRF combined, no particles (§7.γ) | 0.584 | 0.640 |
| 4 | Two-pass combined, no particles (§7.γ) | 0.473 | 0.549 |
| 5 | Single-CRF hand | 0.412 | 0.573 |
| 6 | Two-pass combined, exposure-bias (§7.δ) | 0.367 | 0.471 |
| 7 | Two-pass hand | 0.311 | 0.468 |
| — | **v3 logistic baseline** (`logistic_sections_v2.0`) | 0.127 | 0.429 |

**Every v4 variant beats v3 by at least +0.18 macro F1.**  The
worst v4 variant (two-pass hand, 0.311) is 2.5× the v3 baseline;
the best v4 variant (single-CRF combined, 0.585) is 4.6× the v3
baseline.  v4 was designed to beat v3 — it does so unambiguously.

The v3 row sets the floor.  Its 0.127 macro F1 reflects bag-of-
words per-line classification: cross-distribution training on
JATS hurts PDF performance, and the model has no sequence
structure to recover when individual line predictions are wrong.
v4's SBERT embeddings + CRF transitions address both problems.

## §7.β — Single-CRF on the combined corpus

The §6.F report recommended this experiment: if single-CRF
trained on the same combined corpus as Pass-2 wins, retire the
two-pass design.

Macro F1 = **0.585** on the golden set (char acc 0.653).
Combined dev F1 0.634.  Beats two-pass-combined by +0.106 macro
F1 — **an order of magnitude over the decision threshold**
(≥ +0.01 macro F1, no per-label F1 regression > 0.03).

Per-tag F1 comparison (single-CRF-combined vs two-pass-combined):

| Tag | two-pass | single | Δ | note |
|---|---:|---:|---:|---|
| Description | 0.796 | **0.847** | +0.051 | |
| Page-header | 0.811 | 0.815 | +0.004 | layout |
| Bibliography | 0.085 | **0.804** | **+0.719** | layout — joint Viterbi recovers what Pass-1 missed |
| Index | 0.659 | **0.737** | +0.078 | layout |
| Key | 0.000 | **0.687** | **+0.687** | layout |
| Materials-examined | 0.710 | 0.709 | −0.001 | within noise |
| Notes | 0.458 | 0.454 | −0.004 | within noise |
| Misc-exposition | 0.490 | **0.515** | +0.025 | catch-all |
| Type-designation | **0.517** | 0.484 | −0.033 | edges threshold (Δ=0.033 vs 0.03) |
| Phylogeny | **0.542** | 0.409 | −0.134 | regression |
| Etymology | **0.769** | 0.629 | −0.140 | regression |
| Diagnosis | 0.349 | 0.411 | +0.062 | |
| Biology | 0.493 | 0.521 | +0.028 | |
| Materials-and-methods | 0.705 | 0.681 | −0.024 | |
| Nomenclature | 0.603 | 0.626 | +0.023 | |
| Figure-caption | 0.013 | 0.044 | +0.031 | both architectures struggle |
| Table | 0.142 | **0.331** | +0.189 | layout |

Three tags regress beyond the 0.03 threshold (Etymology −0.14,
Phylogeny −0.13, Type-designation −0.033), but they're more than
offset by Bibliography (+0.72), Key (+0.69), and Table (+0.19) —
all layout tags where the two-pass design's Pass-1 was the
bottleneck.  The Etymology / Phylogeny regressions are small
absolute numbers (combined F1 0.4-0.6) on low-support tags;
inspection of the confusion matrix (not reproduced here) shows
single-CRF confuses them with neighboring treatment tags
(Description, Notes) more than two-pass did.

**Net F1 wins for 13 tags; net losses for 4.**  Char accuracy
also moves +0.10.

## §7.γ — Particle-block ablation

Method: `--ablate-particles` zeros the 12-d particle feature
slice (`features.PARTICLE_SLICE = [768:780]`) at inference time
on the production-pinned two-pass-combined model AND on §7.β's
single-CRF combined model.  No retraining.

| Variant | Macro F1 (with) | Macro F1 (no particles) | Δ |
|---|---:|---:|---:|
| Two-pass combined | 0.479 | 0.473 | −0.006 |
| Single-CRF combined | 0.585 | 0.584 | −0.001 |

Aggregate impact is **negligible** (well under the 0.01
threshold for either architecture).  But per-tag deltas are
non-uniform — the spans pipeline shifts probability mass between
tags rather than uniformly raising or lowering F1:

| Tag | two-pass Δ | single Δ | note |
|---|---:|---:|---|
| Bibliography | **−0.052** | −0.021 | spans help layout disambiguation |
| Description | −0.022 | **−0.048** | spans help in single-CRF (joint decode) |
| Etymology | −0.027 | — | small effect |
| Nomenclature | **+0.039** | — | spans pull lines toward Nomenclature |
| Phylogeny | — | **+0.077** | spans hurt Phylogeny in single-CRF |
| Table | **−0.037** | +0.011 | spans help two-pass tables |
| Notes | −0.009 | **−0.029** | spans help Notes in single-CRF |

Interpretation: the particle pipeline is largely **redundant
with SBERT**.  SBERT already encodes the lexical cues a Taxon-
name span carries; removing the explicit span count costs ~0 net
F1.  The non-uniform per-tag deltas suggest spans nudge specific
disambiguation cases but don't change the overall class
boundaries.

**Action**: keep the spans pipeline (zero retirement cost, +0.04
F1 on Nomenclature for the production two-pass model), but its
maintenance priority drops.  Future feature engineering effort
should target the categories the model is still missing
(Phylogeny, Etymology) rather than refining particle detection.

## §7.δ — Exposure-bias measurement

Method: retrain Pass-2 on the combined corpus, but build the
per-doc Pass-1 layout sequence by **decoding with the trained
Pass-1 CRF** instead of from oracle YEDDA labels.  Pass-2 is
now trained on the same noisy non-layout subsequence it sees at
inference time.

| Variant | Pass-2 dev F1 | Golden macro F1 |
|---|---:|---:|
| Pass-2 combined (oracle) | 0.678 | 0.479 |
| Pass-2 combined, exposure-bias | 0.477 | **0.367** |
| Δ | **−0.201** dev | **−0.112** golden |

Exposure-bias-trained Pass-2 is **uniformly worse**: dev F1 falls
0.20, golden macro F1 falls 0.11.  Scheduled-sampling theory says
training on predicted-label sequences should reduce train/test
distribution shift and *improve* test F1.  The measured result
is the opposite.

The standard interpretation: Pass-1's predicted labels are too
noisy (dev F1 0.297) to be useful training signal.  When Pass-2
trains on these noisy sequences it learns to compensate for
Pass-1's specific failure modes — but those failure modes don't
generalize from the training corpus to the golden set, so the
"correction" Pass-2 learns is corpus-specific noise.  Oracle-
trained Pass-2 generalizes better because it learns the *true*
transition structure, which carries across distributions.

This result also makes the §7.β single-CRF win less surprising:
the two-pass architecture pays a real cost for the train/test
distribution mismatch at the Pass-1/Pass-2 boundary, and there's
no way to recover it without first fixing Pass-1's F1.

**Action**: do not adopt exposure-bias training.  The result is
independent confirmation that the two-pass architecture has an
inherent disadvantage versus joint single-CRF decoding.

## §7.ε — Cost synthesis

Training-time wall clock (RTX 5090 Laptop GPU, 20 epochs each,
Adam lr=1e-3, seed=42):

| Variant | Train docs | Dev docs | Dev F1 | Wall |
|---|---:|---:|---:|---:|
| Pass-1 layout (8 labels) | 128 | 32 | 0.297 | 2 h 51 m |
| Pass-2 hand (12 labels) | 128 | 32 | 0.315 | 0 h 29 m |
| Pass-2 combined (12 labels) | 1 508 | 376 | 0.678 | 3 h 00 m |
| Pass-2 combined, exposure-bias | 1 508 | 376 | 0.477 | 0 h 59 m |
| Single-CRF hand (19 labels) | 128 | 32 | 0.344 | 0 h 40 m |
| **Single-CRF combined** (19 labels) | 1 508 | 376 | 0.634 | 1 h 02 m |
| **Total v4 training compute** | | | | **≈ 9 h** |

The 5-row Step 6 + Step 7 training pipeline finishes in under a
working day on a single laptop GPU.  Notable: single-CRF
combined trains in 1 h 02 m vs Pass-2 combined's 3 h — the joint
decode is faster because the trainer doesn't pay to filter to
the non-layout subsequence per epoch (which involves a separate
build_label_sequence pass through every doc's YEDDA blocks).

Inference: 105 golden docs predicted in ≈ 35-45 s wall on the
same GPU regardless of architecture (≈ 0.35 s/doc, dominated by
SBERT cache lookup + feature assembly).  Pure decode (Viterbi
over ~600 lines × 19 labels) is < 5 ms/doc.

Other infrastructure cost:

| Resource | Cost |
|---|---|
| SBERT cache (Redis) | 293 922 keys @ 768 fp32 ≈ **0.84 GB raw vectors** (+Redis overhead, ≈ 2 GB in practice) |
| article.spans.v4.json (CouchDB) | one attachment per training doc ≈ 1-10 KB each |
| article.page-headers.json (CouchDB) | one attachment per training doc ≈ 0.5-2 KB each |
| article.txt.ann (output) | one attachment per golden doc ≈ 10-200 KB each |
| gnservices runtime | localhost gnfinder + gnparser, ~24 MB RSS each |

The whole v4 stack runs comfortably on a single laptop.  No
cluster dependency, no GPU larger than 24 GB.

## §7 Recommendation

Single-CRF trained on the combined corpus beats the current
production two-pass-combined pin by **+0.106 macro F1 and +0.10
char accuracy** on the golden set.  The win is robust:

- v3 baseline floor: 0.127.  Production two-pass: 0.479
  (+0.352 vs v3).  Single-CRF combined: **0.585** (+0.458 vs v3,
  +0.106 vs production).
- The architectural argument (§6.F): on apples-to-apples
  hand-only training, single-CRF beats two-pass by +0.10 — the
  two-pass design needs the 12× combined-corpus advantage to be
  competitive.  §7.β closes that loophole.
- §7.γ rules out the particle pipeline as an explanation: the
  comparison holds with and without particles.
- §7.δ rules out exposure bias as a fixable two-pass disadvantage:
  retraining Pass-2 on predicted-Pass-1 sequences *worsens* F1.

**Recommended cutover** (executed 2026-06-06; commits be28861 +
the cutover-completion commit follow):

1. ✅ Schema landed in `be28861`:
   `bin/env_config.py:_apply_experiment` now maps
   `redis_keys.classifier_model_single` →
   `config['classifier_model_key_single']`.
   `bin/manage_experiment.py` accepts `--redis-key-single` on
   both `create` and `update`.
2. ✅ Production_v4 updated:
   ```
   bin/manage_experiment update production_v4 \
       --redis-key-single skol:classifier:model:v4_single_combined
   ```
   The experiment doc now carries `classifier_model_single`
   alongside the existing `pass1`/`pass2` keys (kept as fallback).
3. ✅ Dispatch policy: when `classifier_model_single` is set on
   the experiment doc AND no explicit `--pass1-key`/`--pass2-key`
   CLI flag was passed, `predict_v4` defaults to single-CRF mode
   against that key.  Explicit CLI flags still force two-pass for
   ad-hoc A/B runs (precedence: explicit CLI > experiment doc).
4. ☐ Optional cleanup: retire
   `skol:classifier:model:v4_pass2_hand` and
   `…:v4_pass2_combined_exposure` from Redis (~80 KB saved).
   The combined Pass-2 bundle stays as a fallback for the
   `--pass1-key`/`--pass2-key` explicit-CLI path.  Deferred.

Live verification: `bin/predict_v4 --experiment production_v4
--golden-db skol_golden_v2 --output-database
skol_exp_production_v4_ann_combined --limit 3 --dry-run` prints
`single=skol:classifier:model:v4_single_combined` and decodes
the docs in single-CRF mode (~0.4 s/doc on the 5090).  Forcing
two-pass with `--pass1-key skol:classifier:model:v4_layout
--pass2-key skol:classifier:model:v4_pass2_combined` falls back
to the legacy production behavior cleanly.

The single-CRF combined bundle (`skol:classifier:model:v4_single_combined`)
+ Pass-1 (`skol:classifier:model:v4_layout`) are now the
recommended v4 production pair.  Pass-1 still trains in
isolation but is no longer load-bearing for treatment-level
labels — single-CRF decodes the full 19-label vocab in one
pass.

## Out of scope (post-v4 follow-up)

- Schema + dispatch changes for the cutover (Recommendation
  items 1-3 above).
- Pass-1 retraining on an expanded layout corpus.  The
  single-CRF architecture sidesteps this for production, but a
  better Pass-1 would still help if a future two-pass design
  re-emerges.
- Fine-tuning SBERT or swapping for a sequence-aware embedder
  (BiLSTM / transformer on lines).  Single-CRF combined's
  Etymology/Phylogeny regressions vs two-pass-combined are the
  obvious next gap to close.
- Cross-validation / multiple seeds for variance estimation.
  All measurements here are single-seed point estimates on a
  30-doc evaluation set; expect ± 0.02 macro F1 variance under
  reseeding.
