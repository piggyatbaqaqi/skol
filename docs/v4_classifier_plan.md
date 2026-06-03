# v4 classifier — SBERT + two-pass CRF

## Background

The v3 baselines ([production_v3_plan.md](production_v3_plan.md)) establish
a 19-class logistic-regression floor. v4 is the first model designed to
*beat* that floor by using line-level semantic embeddings (SBERT) and
explicit sequence structure (linear-chain CRF) rather than per-line
independent classification.

Two architectural choices distinguish v4:

1. **Two-pass CRF.** The first pass identifies *layout* categories —
   page headers, figure captions, tables, keys, bibliographies, indices,
   ToC entries — that physically interrupt narrative treatment text in
   PDF-derived documents. Those lines are removed; a second pass over
   the cleaned sequence labels the remaining lines with treatment-level
   tags (Nomenclature, Description, Diagnosis, Etymology, …). The
   motivation is that layout artifacts are structurally simpler to
   recognize than treatment boundaries, and removing them lets the
   treatment CRF see a more contiguous label sequence.

2. **Heuristic + learned features fused early.** gnfinder / gnparser /
   particle_detector (per
   [treatment_architecture.md §Phase 2](treatment_architecture.md)) and
   page-header detection (per
   [page-header-detection.md](page-header-detection.md)) run *before*
   the CRFs and contribute features to each line. The reasoning: an
   MB-number on a line is a near-deterministic Nomenclature signal;
   making the CRF rediscover that pattern from raw text wastes capacity.

The model is trained against the v3 no-golden corpora
(`skol_training_v2_no_golden`, `skol_training_taxpub_v2_no_golden`,
`skol_training_v3_combined_no_golden`) and evaluated against the v2
golden universe — directly comparable to v3 numbers.

## Goals

1. Beat v3 baseline macro F1 on `skol_golden_ann_hand_v2`: **v3_hand
   0.459** (the hand-trained baseline; see
   [production_v3_report.md](production_v3_report.md)). The
   JATS-trained baseline (v3_jats macro F1 0.132) is the floor v4 must
   clear if Pass 2's combined-corpus variant is to be considered viable
   (see "Pass-2 training-corpus ablation" below).
2. Demonstrate measurable improvement on the layout-rich PDF docs in
   `skol_golden_ann_hand_v2`. The hand gold has 17 distinct semantic
   tags including Page-header, Index, Bibliography, Table — categories
   the v3 logistic regression treats per-line. v4's sequence structure
   should help disambiguate runs of these layout artefacts.
3. Establish a reusable feature pipeline (SBERT + particle features +
   layout features + page-header signals) that subsequent models
   (transformer, BiLSTM-CRF, ensemble) can also consume.

## Non-goals

- **Not fine-tuning SBERT.** Frozen, used as a feature extractor only.
  Fine-tuning multiplies training cost and isn't justified until v4
  beats v3 with frozen embeddings.
- **Not introducing a new tag set.** v4 emits `ACTIVE_TAGS_19` (Step 1
  of v3 plan); the partition into Pass-1 and Pass-2 label spaces below
  is an internal model concern.
- **Not changing the YEDDA output format or the evaluator.** v4 emits
  `article.txt.ann` in the same format as v3; `evaluate_golden.py`
  scores it identically.

## Label-space partition

ACTIVE_TAGS_19 splits cleanly into two pass-specific spaces:

| Space | Labels | Why |
|---|---|---|
| **Pass 1 — layout (7 from ACTIVE_TAGS_19 + synthetic `Other`)** | Page-header, Figure-caption, Table, Key, Bibliography, Index, ToC-entry, `Other` (sentinel) | All physically inserted into narrative text; share structural cues (short lines, repetition, all-caps, isolated numbers); never carry treatment content. |
| **Pass 2 — treatment (12)** | Nomenclature, Description, Diagnosis, Etymology, Materials-examined, Materials-and-methods, Type-designation, Biology, Phylogeny, New-combinations, Notes, Misc-exposition | The treatment universe. Misc-exposition is the catch-all keeping the treatment-assembly state machine happy. |

Union covers every ACTIVE_TAGS_19 tag exactly once. `*Other*` is a
Pass-1-internal label; the final emitter rewrites Pass-1 `Other` lines
to whatever Pass 2 produces.

## Architecture

```
plaintext
   │
   ├─► gnfinder + gnparser + particle_detector ──► article.spans.json
   │                                                  (Layer 2 spans)
   ├─► page_header_detector ──► article.page-headers.json
   │
   ├─► section_header_detector ──► spans in .spans.json
   │
   ├─► sbert(line) per unique line ──► Redis cache  skol:sbert:<sha256(line)>
   │
   ▼
per-line feature vector  =  concat(
      sbert_emb[768],
      particle_features[~12],   # span-density counts derived from .spans.json
      layout_features[~8],      # line length, indent, all-caps fraction, ...
      page_header_score[2],     # heuristic confidence + binary flag
      section_header_flag[1]    # is this line a section header?
   )  ≈ 791-dim
   │
   ├─► CRF Pass 1 ──► layout labels per line  (Page-header / Figure-caption /
   │                                            Table / Key / Bibliography /
   │                                            Index / ToC-entry / Other)
   ▼
remove lines with Pass-1 layout labels
   │
   ▼
CRF Pass 2 ──► treatment labels per surviving line
   │
   ▼
merge_passes ──► per-line label = Pass-1 layout label OR Pass-2 treatment label
   │
   ▼
coalesce_consecutive_labels ──► YEDDA blocks ──► article.txt.ann
```

## Feature engineering

### Per-line feature vector

```python
@dataclass
class LineFeatures:
    sbert: np.ndarray              # (768,) frozen SBERT cls embedding
    particles: np.ndarray          # (12,)  one count per Layer-2 span label,
                                   #        plus flag for SP_NOV annotation
    layout: np.ndarray             # (8,)   length, indent_pct, allcaps_pct,
                                   #        digit_pct, trailing_digit_flag,
                                   #        is_short, blank_before, blank_after
    page_header_score: np.ndarray  # (2,)   heuristic confidence ∈ [0,1] and
                                   #        binary flag from the
                                   #        page_header_detector
    section_header_flag: np.ndarray  # (1,)  1.0 if any section-header span
                                     #       overlaps this line, else 0.0
```

Total dim ≈ 791. Concatenated horizontally; both passes consume the
same vector. (Item 2(b) from the design discussion: the page-header
heuristic feeds the CRF as a confidence-weighted feature rather than a
hard label, so the CRF can learn to override the heuristic when other
signals contradict it.)

### What gets trained vs not

| Component | Trainable? | Notes |
|---|---|---|
| SBERT | No (frozen) | One forward pass per *unique* line text, cached. |
| gnfinder / gnparser / particle_detector | No | Rule-based + REST services. Cached as `.spans.json` per doc. |
| page_header_detector | No | Pure heuristic from [page-header-detection.md](page-header-detection.md). Cached as `.page-headers.json`. |
| section_header_detector | No | Regex. Output merged into `.spans.json`. |
| **CRF Pass 1** `(W_1, A_1)` | **Yes** | Emission projection W_1 (8 × 791) + transition matrix A_1 (8 × 8). |
| **CRF Pass 2** `(W_2, A_2)` | **Yes** | Emission projection W_2 (12 × 791) + transition matrix A_2 (12 × 12). |

The two CRFs have *no shared parameters*. They share input
representations (SBERT cache, span cache, page-header cache) but each
fits independently.

### SBERT compute scope

SBERT is computed once per unique line text across the entire combined
corpus (1 884 docs ≈ 1 M lines, with substantial duplication of
boilerplate). Cache key: `skol:sbert:<sha256(line_text)>`. Both passes
read from the same cache.

The Pass-1 / Pass-2 corpus split (next section) is purely a row-filter
over the trained CRFs' inputs — it does not change the SBERT cache.

## Training data scope

| Pass | Training corpus | Doc count | Why |
|---|---|---|---|
| Pass 1 (layout) | `skol_training_v2_no_golden` | 160 | Hand-annotated PDF-derived docs are the *only* source of true positives for Page-header / Index / Page-header / Table / etc. — JATS-derived docs (XML source) have zero of these by construction. Training Pass 1 on the combined corpus would dilute the signal with 1 724 docs of all-Other lines. |
| Pass 2 (treatment) | **OPEN** — see ablation note below | 160 or 1 884 | The v3 negative result ([extraction_pipeline.md](extraction_pipeline.md) §Background) showed cross-distribution training data does *not* help under bag-of-words representations. Whether SBERT's semantic embeddings make the transfer work is the open empirical question. Step 6 schedules an explicit ablation. |

**Pass-2 training-corpus ablation.** The combined corpus is 9:1 JATS-
dominated, and the v3 experiments showed JATS-derived labels at best
fail to help and at worst degrade PDF-classification metrics
(`v3_jats` macro F1 0.132 vs `v3_hand` 0.459 on the same hand gold).
That negative was at the level of tf-idf features; SBERT may or may
not erase the gap by abstracting surface form away.

Train two Pass-2 variants in Step 6 and compare:

- `v4_pass2_hand` — Pass 2 trained on `skol_training_v2_no_golden`
  alone, mirroring Pass 1's training scope.
- `v4_pass2_combined` — Pass 2 trained on
  `skol_training_v3_combined_no_golden` (after dropping
  layout-labelled lines).

Both share the same SBERT cache and the same Pass 1 model. If
`hand` ≥ `combined` on hand gold, the v3 negative generalises and we
drop the combined option (Pass 2 trains on hand-only). If
`combined` > `hand`, SBERT is doing what we hoped and JATS data is a
useful augmentation despite the surface-form gap. The cost is roughly
2× training time; cheap relative to the certainty it buys.

### Train/eval/dev split

Carve **20% of hand-corpus docs** (32 docs) as a dev set for CRF
regularization tuning. Pass 1's 160-doc training corpus becomes 128
docs. The v2 golden (`skol_golden_ann_hand_v2`, 30 docs;
`skol_golden_ann_jats_v2`, 75 docs) remains the *evaluation* set and is
queried *once* at the end, not for tuning.

The dev set is sampled stratified by document length so the held-out
docs span the layout-artefact distribution.

## Exposure bias (known limitation)

Pass 2 trains on ground-truth-cleaned line sequences but at inference
sees *Pass-1-predicted-cleaned* sequences. If Pass 1 mislabels a
Description line as Bibliography (false positive), Pass 2 never trained
on that "now-Description-shaped-gap" scenario.

We accept this for v4 and measure it. Mitigations if needed in a
follow-on:

1. **Scheduled sampling.** At Pass 2 training time, use Pass 1's
   *predictions* (with some noise budget) instead of ground truth to
   filter lines. Standard sequence-model fix; adds training-loop
   complexity.
2. **Joint single-CRF baseline.** One CRF over the full 19-label space;
   no removal step. Simpler; we ablate against this in Step 6.

## Step-by-step

### Step 0 — SBERT cache build-out

| # | Description | Status |
|---|---|---|
| 0.A | Choose SBERT variant (`all-MiniLM-L6-v2` 384-dim vs `all-mpnet-base-v2` 768-dim). Lean: mpnet for accuracy; revisit if compute budget breaks. | ✅ Both supported via `--sbert-model {mpnet,minilm}`; separate key namespaces (`skol:sbert:mpnet:*` / `skol:sbert:minilm:*`) so we can A/B without recomputing. mpnet used for the initial pass. |
| 0.B | Per-line embedder: `bin/embed_lines.py` reads a CouchDB DB, splits each `article.txt` into lines, computes SBERT per *unique* line (hash de-dup), writes to Redis `skol:sbert:<model>:<sha256>`. Idempotent on re-run. | ✅ Lands in commit `1c3ecc6` (with `e06dfcf` skip-existing fix). 3-path plaintext fallback chain (article.txt → article.pdf via `plaintext_from_pdf` → article.txt.ann via `plaintext_from_yedda`). CLI: `--sbert-model`, `--source-db`, `--batch-size`, `--force`, env_config flags. |
| 0.C | Compute for `skol_training_v3_combined_no_golden` (1 884 docs) + `skol_golden_v2` (105 docs). Approx 1 M unique lines. | ✅ Actual numbers: **293,922 unique lines** (≪ the 1 M estimate; the JATS-converted majority dedupe internally to ~177 unique lines/doc). Total wall time **~53 min** on puchpuchobs's RTX 5090 Laptop (~6 ms/line at batch 64). Training pass alone: 48 m 25 s for 236,743 keys; golden pass: 4 m 25 s for +57,179 keys. Redis memory: 1.95 GB. |
| 0.D | TDD: tests covering hash-key derivation, idempotency, empty-line handling. | ✅ 21 tests in `bin/embed_lines_test.py` across 9 classes (4 pure-helper for key derivation, 4 for line iteration, 4 for the plaintext fallback chain, 3 for skip-existing semantics, 6 for live-Redis integration). FakeRedis stub dropped in favour of live Redis with namespaced `skol:sbert:test:<uuid>:` keys, per the project's existing convention. |

### Step 1 — Particle / span / page-header pipeline

| # | Description | Status |
|---|---|---|
| 1.A | Implement (or reuse from [treatment_architecture.md §Phase 2](treatment_architecture.md)) `ingestors/gnfinder_client.py`, `gnparser_client.py`, `particle_detector.py`, `spans.py`. If those don't exist yet, write them per Phase 2's spec. **Precondition:** the existing clients default to `https://finder.globalnames.org` / `https://parser.globalnames.org`. v4 callers must pass `server_url='http://localhost:9080'` / `'http://localhost:9081'` (or equivalent env-config plumbing) to hit the local services whose cost is reported in §Span-layer cost. Don't rely on client defaults. | ⬜ |
| 1.B | New: `ingestors/page_header_detector.py` implementing the heuristic from [page-header-detection.md](page-header-detection.md). Sub-steps 1.B.1–1.B.5 below mirror the five algorithm stages in that doc; each is its own TDD-able unit. Final output: `article.page-headers.json` listing line indices and per-line confidence (the score consumed as the `page_header_score[2]` feature in Step 2). | ⬜ |
| 1.B.1 | **Candidate collection** ([§Step 1](page-header-detection.md)). `collect_candidates(lines) -> List[PageNumCandidate]` with fields `(line_index, position ∈ {'start','end'}, value: int, raw_token: str)`. Filters: 1–4 decimal digits; exclude tokens matching `(19\|20)\d{2}` at line-end (years); exclude tokens ≥5 digits (accession numbers, specimen IDs). TDD: hand-crafted line fixtures hitting each filter rule. | ⬜ |
| 1.B.2 | **Sequence fitting** ([§Step 2](page-header-detection.md)). `fit_sequence(candidates) -> SequenceFit` returning a RANSAC-style fit `page_number ≈ a × doc_position + b` plus the gap-histogram quality score (sharp peak at 1 or 2 ⇒ confident; flat ⇒ noise). Accepts gap ≥ 2 for journals that omit numbers on some pages. TDD: synthetic sequences with planted OCR-substitution residuals to confirm RANSAC rejects them. | ⬜ |
| 1.B.3 | **Recto/verso alternation** ([§Step 3](page-header-detection.md)). `partition_alternation(candidates) -> (verso_fit, recto_fit, alternation_score)`. Fit odd and even candidate sets independently and verify the two subsequences interleave cleanly; clean interleave raises confidence sharply. TDD: one synthetic doc with strict alternation, one with mixed placement (no alternation). | ⬜ |
| 1.B.4 | **Journal-name clustering** ([§Step 4](page-header-detection.md)). `cluster_header_text(confirmed_lines) -> List[TokenCluster]` over the non-numeric portion of confirmed header lines. Approximate matching (edit distance or token overlap) tolerant of OCR substitutions, abbreviation drift, and CAPS-vs-title-case variation. TDD: lines from two real PDF docs in `skol_training_v2_no_golden` exercising both cluster styles (journal-name cluster + author/title cluster). | ⬜ |
| 1.B.5 | **Two-pass block recovery** ([§Step 5](page-header-detection.md)). `recover_header_block(lines, confirmed_anchors) -> List[HeaderRegion]`. Pass 2 uses confirmed sequence members as anchors to extend the marked region to adjacent unnumbered lines (volume/issue strings, blank separators). Emits the final `article.page-headers.json` attachment. TDD: fixture doc with header + adjacent non-header lines, assert region boundaries. | ⬜ |
| 1.C | New: `ingestors/section_header_detector.py`. Regex over short, title-case-or-all-caps lines matching known section names ("Taxonomy", "Systematics", "Introduction", "Materials and methods", etc.). Emit spans into `.spans.json` with `label="section-header"`. | ⬜ |
| 1.D | `bin/annotate_v4.py`: orchestrator that runs the four detectors for every doc in a target DB, writes `.spans.json` and `.page-headers.json` attachments. Idempotent. | ⬜ |
| 1.E | TDD per module; integration test on 3 sample docs from `skol_training_v2_no_golden`. | ⬜ |

### Step 2 — Line-feature assembler

| # | Description | Status |
|---|---|---|
| 2.A | New module: `skol_classifier/v4/features.py`. Public function `build_line_features(line_text, line_index, doc_lines, sbert_cache, spans, page_headers) -> LineFeatures`. | ⬜ |
| 2.B | TDD: unit tests for each feature group (SBERT lookup, particle counts, layout, page-header score, section-header flag). | ⬜ |
| 2.C | Train/inference parity test: feed a fixture doc through twice, assert identical feature vectors. | ⬜ |

### Step 3 — CRF Pass 1 (layout)

| # | Description | Status |
|---|---|---|
| 3.A | Adopt `sklearn-crfsuite` or `pytorch-crf` for the CRF implementation. Lean: `pytorch-crf` because it integrates cleanly with PyTorch tensors (`W · x_t + b` is a `nn.Linear`). | ⬜ |
| 3.B | New module: `skol_classifier/v4/crf_layout.py`. `class LayoutCRF(nn.Module)` with `forward(features) -> loss` and `decode(features) -> labels` (Viterbi). | ⬜ |
| 3.C | New trainer: `bin/train_crf_layout.py`. Reads `skol_training_v2_no_golden` (minus 32-doc dev split), assembles features, fits the CRF. Saves to Redis `skol:classifier:model:v4_layout`. | ⬜ |
| 3.D | TDD: synthetic-data test (handcrafted 10-line sequence) confirming Viterbi decodes correctly; idempotent training (same seed → same weights). | ⬜ |

### Step 4 — CRF Pass 2 (treatment)

| # | Description | Status |
|---|---|---|
| 4.A | New module: `skol_classifier/v4/crf_treatment.py`. Same shape as `LayoutCRF` but over 12 treatment labels. | ⬜ |
| 4.B | New trainer: `bin/train_crf_treatment.py`. Reads `skol_training_v3_combined_no_golden`, filters out lines whose label is in the Pass-1 layout set, fits the CRF. Saves to Redis `skol:classifier:model:v4_treatment`. | ⬜ |
| 4.C | TDD: parallel to Step 3.D. | ⬜ |

### Step 5 — End-to-end predictor + YEDDA emitter

| # | Description | Status |
|---|---|---|
| 5.A | New module: `skol_classifier/v4/predictor.py`. `predict_doc(text)` runs feature assembly → Pass 1 → strip → Pass 2 → merge → coalesce consecutive labels → emit YEDDA-formatted .ann text. | ⬜ |
| 5.B | New CLI: `bin/predict_v4.py`. Required flags: `--experiment <experiment_doc_id>` (to look up Pass-1/Pass-2 model keys in the experiment doc), `--golden-db <db_name>` (input docs), `--output-database <db_name>` (where to write `.ann` attachments). Env-var controls for `SKIP_EXISTING`, `FORCE`, `DRY_RUN` matching `predict_classifier.py`. Reuse env_config for COUCHDB_URL / credentials. (The v3 script's `--model` flag is replaced by `--experiment` because v4's two-CRF design is keyed by the experiment doc, not a single model name.) | ⬜ |
| 5.C | Wire into `bin/manage_experiment.py` — new `predict_v4` step (or reuse `predict` with experiment doc `model_name: "v4_crf"`). Lean: new explicit step name so v3 and v4 can coexist per-experiment. | ⬜ |
| 5.D | TDD: round-trip test — input plaintext → predict → assert output is parseable YEDDA. | ⬜ |

### Step 6 — Experiment doc + pipeline runs

| # | Description | Status |
|---|---|---|
| 6.A | New experiment doc `production_v4` in `skol_experiments`. Fields per the v3 pattern; `model_name: "v4_crf"`. Three training-database variants are not needed — v4 uses the combined corpus only (Pass 1 internally restricts to hand). | ⬜ |
| 6.B | Run `train_crf_layout` (Pass 1). | ⬜ |
| 6.C | Train **two** Pass-2 variants: `v4_pass2_hand` (hand-only) and `v4_pass2_combined` (full combined corpus). See "Training data scope" above. | ⬜ |
| 6.D | Run `predict_v4` twice over `skol_golden_v2` — once with each Pass 2 variant. | ⬜ |
| 6.E | Run `evaluate_golden.py` for each Pass 2 variant against `skol_golden_ann_hand_v2`. The variant with the higher macro F1 becomes `production_v4`'s real Pass 2. | ⬜ |
| 6.F | Ablation: train a *single-CRF* baseline (no Pass 1 / Pass 2 split, full 19-label space, hand-only training) and compare. If the two-pass design doesn't beat the single-CRF baseline, that's a Step 7 decision to abandon the two-pass complexity. | ⬜ |

### Step 7 — Comparison report

`docs/production_v4_report.md` covering:

- **v4 vs v3 baseline grid** — macro F1 and per-tag F1 on both golden
  DBs.
- **Two-pass vs single-CRF ablation** (from 6.F). If the gap is small,
  recommend the single CRF for simplicity.
- **Per-tag improvements** — which categories did SBERT + CRF help most
  on? Likely candidates: Page-header (sequence structure helps), Notes
  (semantic disambiguation), Diagnosis (CRF transitions encode "after
  Description, often comes Diagnosis").
- **Particle feature contribution** — ablate by zeroing the particle
  feature block; report the F1 delta. This tells us whether the spans
  pipeline is pulling its weight.
- **Exposure bias measurement** — re-train Pass 2 using Pass 1's
  predicted labels (scheduled sampling, single iteration); report
  whether the Pass-2 metrics shift.
- **Cost** — total training time, prediction-throughput
  (docs/second), Redis memory for SBERT cache.

## Intermediate-data storage policy

| Artefact | Cost to recompute | Storage |
|---|---|---|
| SBERT line embeddings | GPU minutes per doc, hours for the corpus | **Redis** `skol:sbert:<sha256>` keyed by line hash; shared across docs. ~768 floats × 1 M unique lines ≈ 3 GB if float32, 1.5 GB if float16. |
| `.spans.json` (gnfinder / gnparser / particle / section_header) | Medium — REST calls to gnfinder / gnparser plus regex | **CouchDB attachment** per doc; matches the existing [treatment_architecture.md §Phase 2](treatment_architecture.md) layer-2 spec. |
| `.page-headers.json` | Cheap — pure-Python heuristic | **CouchDB attachment** per doc. Keeps the per-document calibration auditable. |
| Layout features | Trivial — line-length / indent / all-caps counters | **In-memory only**. Recomputed on every train + inference pass. |
| Pass-1 predictions (intermediate) | Cheap once SBERT is cached | **In-memory only** between Pass 1 and Pass 2 of the same inference invocation. |
| Final per-line labels + coalesced blocks | Cheap | **CouchDB attachment** `article.txt.ann` (final v4 output). |

## Reproducibility pin (lives in `production_v4` experiment doc)

- SBERT model name + version + revision + quantisation flag (`mpnet` / `MiniLM`, fp32 / int8 / ONNX backend)
- gnfinder service URL + version reported by `GET /api/v1/version`
- gnparser version
- particle_detector regex pattern source-file sha256
- Fungarium-codes Redis snapshot timestamp + sha256 of the snapshot
- CRF library + version (`pytorch-crf 0.7.x`)
- Training-doc set hashes (skol_training_v2_no_golden, _taxpub_v2_, _combined_) and random-seed-fixed dev-doc list

So a future re-eval against today's numbers matches exactly.

## Span-layer cost (measured 2026-05-22)

A 15-doc sample (5k-220k chars each, 676 k total chars, 1 299
TaxonNames detected) was timed against:

| Service | Per-doc avg | 30 k-doc projection (sequential) | 30 k-doc projection (16-parallel) |
|---|---:|---:|---:|
| gnfinder local (`-p 9080`)  | 0.027 s | 14 min  | 1 min  |
| gnparser local (`-p 9081`)  | 0.17 s  | 84 min  | 5 min  |
| **Combined**                | **0.20 s** | **98 min** | **6 min** |

Public-API equivalents were ~10× slower and gnparser was outright
unreachable due to a stale endpoint convention in the client (GET vs
POST mismatch; fixed in `gnparser_client.py`).  Local services
remove the network bottleneck and make the span layer essentially
free at corpus scale.

**Operational work items unblocked by this finding:**

- ⬜ **gnservices packaging.**  Decide between bundling the two
  binaries in the main `skol.deb` (Option C — fastest) and shipping
  a separate `skol-gnservices.deb` (Option B — the right
  long-term answer; independent release cadence).  Either way, add
  systemd units to start `gnfinder -p 9080` and `gnparser -j 8 -p 9081`
  on boot.  Track as a Phase-D-blocker — needs to land before
  v4 prod deployment.
- ⬜ **Span cache.**  ``sha256(text) → spans`` Redis keyed by line
  content.  Cuts incremental-sweep runtime once we re-process the
  corpus on a future ingest cycle.  Defer to after first run.

- **Linear-chain vs BiLSTM-CRF?** Lean: start linear-chain over SBERT
  vectors. SBERT already encodes per-line semantics; the CRF only
  needs label-transition structure. Upgrade to BiLSTM-CRF only if
  metrics demand it (Step 7 decision).
- **Class weights for the CRF.** Reuse the Step 3.B inverse-frequency
  weights from
  [production_v3_plan.md](production_v3_plan.md). pytorch-crf accepts
  per-class weights in its NLL loss.
- **Inference cost.** Each doc needs: lookup SBERT for every line
  (Redis hits, microseconds), assemble features, run two Viterbi
  decodes. Expected throughput: 100s of docs/second on CPU once SBERT
  is cached. Worth measuring in 6.F.
- **Line-boundary cases.** A physical line that straddles a Nomenclature
  → Description boundary forces a single label choice. Measure
  frequency on hand gold first; if >5%, consider sub-line tokenisation
  as a Step-7 follow-up.
- **Pass-1 evaluation surface.** The v2 golden has all 17 tags; we can
  evaluate Pass 1's per-tag F1 directly against the layout tags in
  the gold. No new evaluator code needed.

## Workflow / sequencing

1. **Step 0 first** (SBERT cache) — every downstream step needs it.
2. **Step 1** (detectors) in parallel with Step 0 once their tests are
   green.
3. **Step 2** (feature assembler) — depends on 0 + 1.
4. **Step 3 / Step 4** can run in parallel (separate CRFs, no shared
   weights).
5. **Step 5** end-to-end predictor wires it together.
6. **Step 6** is the model run; **Step 7** is the writeup.

## Rollback

- **Step 0:** delete the Redis SBERT cache. v3 unaffected.
- **Step 1:** drop the new attachments (`.page-headers.json`,
  any new `.spans.json` files). v3 unaffected.
- **Steps 2–5:** revert the v4 modules. They live under
  `skol_classifier/v4/` and `bin/{predict_v4,train_crf_*}.py` and
  don't touch v3 code paths.
- **Step 6:** remove the `production_v4` experiment doc; delete its
  Redis classifier keys (`skol:classifier:model:v4_layout`,
  `:v4_treatment`).

## Progress

To be filled in as sub-steps land.
