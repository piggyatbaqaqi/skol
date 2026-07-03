# production_v5 execution plan

*Living document.  Started 2026-07-03.  Milestone-driven, not
time-boxed.  The operator has a paid-work commitment coming that
may compress available time — the plan explicitly accommodates
that by making every milestone shippable independently.*

## Context

The 2026-07-01 → 2026-07-03 triage-CSV review pass generated
~30 memo commits in [docs/data_quality_production_v4_model.md](../data_quality_production_v4_model.md)
cataloguing failure modes in production_v4.  This plan
converts that catalogue into an execution sequence.  Six
top-level concerns motivated the plan:

0. Faster structured-form creation than the Mistral-all-at-once
   model.
1. Vocabulary sampling curve is early — several × more
   training material needed for coverage; more still for
   per-feature support.  (Heaps' Law analysis, see
   `jupyter/heaps_law_analysis.ipynb`.)
2. Terminology equivalence dictionary
   (`docs/feature_label_canonicalization.json`) needs 1+
   human passes.
3. Treatment extractor problems — merged treatments (§6 in
   the memo, 12+ documented cases).
4. Section classifier problems — clipping (§10), inadequate
   non-Description component identification (§12).
5. Feature clause labeling is somewhat independent of
   treatment creation.  Feature-label signals could inform
   both treatment extraction AND section classification —
   a positive-feedback opportunity.
6. Other considerations from the memo not otherwise covered.

## Guiding principles

* **Milestone-driven, not weekly.**  Each milestone ships
  independently — you can pause after any of them without
  stranding half-done work.
* **Deliver the next smallest business-value increment.**
  Post-hoc audit tooling before pipeline changes;
  pipeline changes before productization.
* **Continuous tracks alongside milestone gates.**  Two
  streams run in parallel with the milestone sequence and
  don't gate it:
    - Continuous track A: hand-review of triage-flagged
      treatments.  Golden data compounds.
    - Continuous track B: terminology-equivalence dict
      human pass.  Runs whenever the operator has bandwidth.
* **Bootstrap before verified.**  Where a milestone needs
  training data, start with noisy Claude-candidate
  annotations; retrain on the reviewer-verified subset as
  it grows.  Don't gate on "waiting for 200 more reviewed
  treatments" — the vocabulary curve rewards volume more
  than perfection at the segment-classifier training
  stage.

## Continuous tracks (parallel to milestones)

### Track A: Hand-review flow

* **Cadence**: as time permits.  Recent burst rate 45
  treatments/pass; sustainable rate probably 10-20/week
  under paid-work pressure.
* **Target**: ~200-250 reviewer-verified treatments to
  satisfy the Heaps' Law vocabulary-coverage threshold.
* **Compounding value**: each week's reviews improve
  golden-data quality AND provide fresh training data for
  the segment classifier (Milestone 4).
* **Tooling upgrades feed into this**: Milestones 1 and 2
  make the triage tools better, so each review pass
  yields more targets and cleaner precision/recall
  statistics.

### Track B: Terminology-equivalence dictionary pass

* **Owner**: operator (human judgment call).
* **Current state**: 26 drift entries in
  `docs/feature_label_canonicalization.json`, curated
  ad-hoc during review passes.
* **Target**: comprehensive pass identifying every
  Claude-emitted label variant that maps to a canonical
  form.  Reduces vocabulary drift, improves training
  signal quality.
* **Independent of everything else**: can happen any
  week the operator has an hour to spare.  No gate.

## Milestones

### M1: Detector upgrades on v4 (post-hoc audit)

**Scope**: implement four documented detectors, rerun
triage CSV against production_v4.

* gnfinder / gnparser integration for §6 idea #2
  (formally-cited authored binomials in Description).
  Local services already running at
  `localhost:9080/9081`.
* E→L→E order-aware Latin-block detector
  (§6 idea #1(b)).  Extends existing `latin_block_count`.
* Tail-clip detector (`[a-z]-\s*$` mid-word hyphen;
  "description doesn't end with sentence-final period"
  heuristic).
* Diagnosis-field head-clip detector — apply existing
  `desc_starts_mid_sentence` rule to the diagnosis
  field, gated on `diag_length > 0`.

**Deliverable**: rerun triage CSV surfacing ~20+ more
merges from the same production_v4 corpus.

**Business value**: expanded triage queue for continued
review; catches previously-missed merges without waiting
for pipeline changes.

**No experiment change needed.**  Stays on v4.

### M2: Detector-suite consolidation

**Scope**: implement the remaining lower-priority
detector ideas from the memo.

* Header-keyword watchlist expansion
  (`Illustration:`, `Description and illustration:`,
  `Cultural characteristics`, `Colonies on`, `Etymology:`,
  `Habitat:`, `Type:`, `Holotype:`) in `count_*_headers`.
* Structural-anatomy-doubling detector — curated word
  list (Basidiocarps, Ascomata, Asci, Paraphyses,
  Conidiomata, Conidia, Pileus, Stipe, …), fire on
  count ≥ 2 or ≥ 3.
* Period-form `Description.` header regex extension.
* Roman-numeral couplet-line support in
  `_COUPLET_LINE_RE`.
* U+FFFD-tolerant header regex extension.
* Mid-body `.\s+[a-z]` transition detector for the
  head-only-clip → new-species-start pattern
  (taxon_9ecad903).

**Deliverable**: rerun triage CSV, additional flags
surfaced.

**No experiment change needed.**  Stays on v4.

### M3: Bootstrap-candidate-based segment classifier v0

**Scope**: train a first segment classifier using the
candidate-annotation DB
(`skol_exp_production_v4_02_50_features_candidate`) as
training data.

Rationale for using candidate annotations (not
reviewer-verified):
* Volume: candidate DB has spans from ~7500 non-skipped
  treatments (`triage_treatments --include-skipped`
  reference number).
* Bootstrap validity: Claude's per-span labels are
  noisy but usable as weak supervision.
* Retraining path: replace training set with
  reviewer-verified spans as the verified corpus grows
  past a stability threshold.

**Model**: small classifier (BERT-base fine-tune or a
CRF over SBERT embeddings — whichever prototypes fastest).

**Deliverable**: segment-classifier v0 with per-line
Anatomical / Nomenclature / Diagnosis / Habitat / Type /
Distribution / Key / Other predictions.  Evaluated on the
56 reviewed treatments (as a small held-out set).

**Prerequisite for M4.**  **This is where production_v5
gets created** — the segment classifier's output becomes
a new column in the treatments_prose pipeline, requiring
a fresh experiment container.

### M4: production_v5 — segment-classifier-informed section CRF

**Scope**: retrain the v4 layout / section CRF with
segment-classifier v0 predictions as input features
alongside the existing SBERT embeddings.

Hypothesis: adding a per-line section-shape signal
reduces the §10 (clipping) and §12 (mis-routing)
failures documented in the memo.

* Create `production_v5` experiment via
  `bin/manage_experiment create --name production_v5 --pipeline v4_crf`
* Copy training set from v4; add the segment-classifier
  predictions to the feature extraction.
* Retrain; predict on the 45k-treatment corpus.
* Re-extract treatments_prose from the new predictions.
* Bootstrap-annotate a sample of ~50 v5 treatments with
  the existing Claude prompt for side-by-side comparison
  against v4.

**Deliverable**: production_v5 experiment DB populated;
comparison metrics on the sample.  Visible quality
improvement or clear evidence the hypothesis was wrong
(useful either way).

**This is the v5 flagship.**  Success criterion:
measurable reduction in the §6/§10/§12 failure rates on
the sample vs v4 baseline.

### M5: v5 iteration — retrain segment classifier on
verified data

**Scope**: as Track A's reviewer-verified treatment count
grows past 200, retrain segment classifier on the
verified subset.  Push retrained predictions back through
the section CRF.

**Deliverable**: v5 segment-classifier + section CRF at
higher quality than M4's bootstrap version.
Corpus-wide re-extraction as quality justifies.

### M6: Structured-form productization
(Mistral-alternative)

**Scope**: replace the current Mistral-all-at-once
structured-form generator with a
segment-classifier-informed pipeline.

If the segment classifier is doing anatomical entity
identification well, the "structured form" step becomes
cheaper — the entity boundaries are already known; only
attribute extraction remains.

**Prerequisite**: v5 stable per M4/M5.  Not started
before v5 is validated.

**Deliverable**: faster / cheaper structured-form
generation, ready for search-product integration.

## v5 timing summary

* **Not now**: no pipeline change is ready to ship yet.
  Weeks 1-3 of milestone work stays on v4 (post-hoc
  audit).
* **Trigger**: v5 is created at M3, when the first
  change that requires a fresh pipeline container is
  ready — the segment-classifier as an additional
  feature source for the section CRF.
* **Not later**: waiting past M3 delays the compound
  gains of Milestone 4 without benefit.

## Heaps' Law dependency

`jupyter/heaps_law_analysis.ipynb` estimates the
vocabulary-coverage growth: hand-annotated treatments
so far (~56) put us early on the curve.  Analysis
suggests ~200-250 verified treatments are needed for
"relatively complete coverage," and more for solid
per-feature support.

**How this interacts with the milestones**:

* Milestones 1, 2, 3 don't gate on hand-annotation
  volume — they use existing data.
* Milestone 4 (v5 pilot) doesn't gate on hand-annotation
  either, because M3 trains on Claude-candidate data
  (higher volume, noisier).
* **Milestone 5 does gate on Track A** — retraining the
  segment classifier on verified data requires enough
  verified data to matter (>200 treatments).
* Milestone 6 (productization) depends on M5's quality,
  which depends on the Heaps'-Law-driven Track A
  volume.

So the vocabulary-coverage curve is the pacing
constraint on the *quality* of v5, not on its
*existence*.  We ship the noisy v5 first (M3+M4) and
improve it as verified data flows in (M5).

## What we're explicitly NOT doing first

* **Mistral-alternative structured-form work (concern 0)**.
  Deferred to M6.  No compounding value until extraction
  quality is fixed.  Building it earlier would sink
  engineering time into a productization step whose
  inputs are still known-broken.
* **Full extractor rewrite**.  Too big for a milestone;
  attacked incrementally via segment-classifier signals
  feeding into the current extractor.
* **More vocabulary sampling before terminology dict
  pass**.  Track B first; new samples without it just
  replay the drift.
* **Detector tightening that risks poster-child
  regressions**.  Any change flagging the six §0.5
  poster-children breaks the "clean single-species"
  contract.  Regression check on all §0.5 entries is a
  gate for every detector change.

## Adjustment triggers

Reasons to revise this plan:

* **Hand-annotation velocity turns out much lower than
  expected under paid-work pressure** — may push M5
  further out; M4's bootstrap version becomes the
  operational quality bar for longer.
* **M3 segment classifier fails to train usefully on
  candidate data** — fall back to a smaller supervised
  bootstrap using just the 56 reviewed treatments; M4
  starts later.
* **v4 extractor failures compound in a way M4 can't
  reach** — may require a more direct extractor
  intervention (segment-classifier signals feeding the
  treatment-grouper, not just the section CRF).
* **Business-value pressure shifts** — search-product
  integration deadline, funding milestone, external
  demo.  Re-rank milestones as needed; the plan is not
  a contract.

## Cross-references

* [docs/data_quality_production_v4_model.md](../data_quality_production_v4_model.md)
  — the failure-mode catalogue this plan responds to.
* [docs/feature_label_canonicalization.json](../feature_label_canonicalization.json)
  — Track B's target artifact.
* [jupyter/heaps_law_analysis.ipynb](../../jupyter/heaps_law_analysis.ipynb)
  — vocabulary-coverage curve; drives Track A's target
  volume.
* [docs/schema_constrained_pipeline.md](../schema_constrained_pipeline.md)
  — Phase 1 context for the bootstrap annotator.

## Change log

* **2026-07-03** — initial draft created after the
  triage-CSV review pass wrap-up.  Six top-level
  concerns converted to milestones + two continuous
  tracks.  Milestone-driven cadence chosen over weekly
  due to expected paid-work commitment.  v5 timing
  fixed to M3 trigger.
