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

> **Corrected 2026-08-23.**  Track A's target was justified by
> the Heaps' Law vocabulary curve.  That was wrong:
> `jupyter/heaps_law_analysis.ipynb` computes its curve from
> **`features_candidate` alone** (cell 4,
> `load_candidate_annotations`); `features_hand` appears in one
> separate comparison cell.  **Vocabulary coverage is bought with
> API volume, not operator hours.**  200-250 remains a sensible
> *M5 training-set* target — retraining the segment classifier on
> verified data does need verified data — but that is a
> training-set argument, and the two were conflated.  See
> [annotation-activity-split.md](annotation-activity-split.md).

* **Cadence**: as time permits.  Recent burst rate 45
  treatments/pass; sustainable rate probably 10-20/week
  under paid-work pressure.
* **Target**: ~200-250 reviewer-verified treatments **as M5
  training data**.  *Not* a prerequisite for the vocabulary
  curve.
* **Label validation is a separate, much smaller activity.**
  Measured 2026-08-23 on the only random sample (round 3):
  precision 100 %, recall 99 %.  A 50-treatment random review
  gives precision to ±1.1 pp and retires the question; more
  hand review buys training data, not label confidence.
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
* ~~Mid-body `.\s+[a-z]` transition detector for the
  head-only-clip → new-species-start pattern
  (taxon_9ecad903).~~  **TRIED AND REJECTED (M2 Group
  C, 2026-07-05)**.  Fixture regression revealed a
  fatal FP class — some legit single-species
  treatments use lowercase-continuation paragraph
  style (fires 3× on taxon_b9a6232, a fixture-tracked
  regression target).  Detector reverted; see the
  taxon_9ecad903 memo entry for the writeup.  Robust
  mid-body boundary detection requires
  paragraph-level section classification (M3
  dependency), not regex.

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

> ### ⚠️ M4 now competes with Experiment 6 — see the evidence below
>
> **Added 2026-09-01.**  The round-5 dossier review generated
> §12.3.1–12.3.42 of
> [the v4 memo](../data_quality_production_v4_model.md), and much of
> it bears on whether M4's *incremental* bet is the right one.
>
> M4's hypothesis is that **adding a per-line section-shape signal**
> reduces the §10 and §12 failures.  The review found repeatedly that
> those failures are **not per-line and not semantic**:
>
> | finding | why a per-line feature cannot reach it |
> |---|---|
> | §12.3.15 — `Table` tracks **mean line length**, not content; `Nomenclature` is the nearest content label to it | the signal is page geometry, absent from the text |
> | §12.3.6 — rogue `Key` is a **document**-level property | per-line features see one line |
> | §12.3.31 — a citation and a description opening **share a line** | the label boundary is *inside* the unit being labelled |
> | §12.3.13 / §12.3.34 — lost newlines and 2-character micro-fragments | the line itself is the corrupted unit |
> | §12.3.23 — 18 % of treatments start **before Materials-and-methods** | requires intra-article position |
> | §12.3.35 — every failure is *"inferring layout from text after
>   layout was discarded"* | the discarded information is not recoverable per line |
>
> **[experiment_6_design_and_implementation_plan.md](experiment_6_design_and_implementation_plan.md)
> proposes exactly the decomposition these findings argue for**, and its
> phasing maps onto them:
>
> | Exp-6 phase | review evidence |
> |---|---|
> | 1. Layout filter | §12.3.35, §12.3.15, §12.3.13 |
> | 3. Span grouper | §12.3.11 (25 % of `Misc-exposition` steals a boundary), §12.3.27, §12.3.34 |
> | 4. Article segmenter | §12.3.9, §12.3.23, §12.3.40 |
>
> **This is not a decision to abandon M4.**  Two things genuinely
> favour it: it is far cheaper, and §12.3.41's finding that defects
> **concentrate in a document minority** (top 10 % of documents hold
> 57 % of boundary theft) means a modest improvement plus document
> triage may capture much of the value.  **But the "not later" argument
> in the timing summary below assumed M4's gains compound; if the
> failures are architectural, they do not**, and M4's cost is then spent
> against a ceiling.
>
> **Recorded so the choice is made deliberately rather than by
> default.**

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

#### Candidate v5 change, independent of the segment classifier

**A document-level "is this a taxonomic article?" gate.**
Measured 2026-08-23: **49.9 % of source documents produce
*only* empty-description treatments** — 8 797 documents,
14 143 treatments.  Sampling 400 of them, just **7.5 %** have
a taxonomic keyword in the title, against **31.5 %** for
documents that do yield descriptions; the all-empty set is
62 % *Journal of Fungi* (broad applied mycology) while the
all-full set is dominated by MycoKeys and Mycotaxon.  Titles
are unambiguous — soil fungal community ecology, calcineurin
inhibitors, laccase production.

These are non-taxonomic papers that happen to contain
binomials, and the grouper built treatments around them.
**54.7 %** of empty treatments carry
`synthetic_nomenclature: true`.

A journal + title-keyword gate would drop ~14 000 spurious
treatments *before* extraction, at no modelling cost.  Worth
sequencing ahead of the CRF work, since it shrinks the corpus
the CRF has to be right about.  Note the scale: *Journal of
Fungi* is the corpus's largest source at 8 817 ingest
documents.

The **other** 64.1 % of empty-description treatments come
from documents that *do* produce descriptions — real
taxonomic papers emitting empty treatments alongside good
ones.  That half is a grouper/CRF defect, is the same suspect
as §6/§12, and is what M4 targets.  The two halves must not
be pooled.

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

**Schema settled 2026-08-24** — see
[structured-form-schema.md](../structured-form-schema.md).  The
two-phase split is kept, for a reason beyond the original one:
phase 1 has an evaluation harness and phase 2 has none, so
collapsing them would discard the only measurement in the
pipeline.  Four decisions recorded there, each with its
evidence: values are verbatim source phrases rather than
parsed; output is a **list of blocks**, not an object keyed by
feature name (39 % of repeated-label groups carry conflicting
measurements, so merging at extraction corrupts ~9.6 % of
blocks — merge downstream, where it can refuse); keys are flat
and lowercase; and **the model recognises measurements while a
regex parses them**, because a language model asked to do
arithmetic fails quietly.

Two premises still to test before any fine-tuning: the slot
vocabulary does not exist yet (322 labels, 54 % singletons), and
"Mistral did poorly" was measured on *entire descriptions* —
the problem phase 1 removes.  Re-run it on labelled spans first.

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
vocabulary-coverage growth.

**Corrected 2026-08-23 — two claims were conflated here.**

* **The vocabulary curve does not gate on hand review.**  It is
  computed from `features_candidate` alone, so it is bought with
  **API volume**.  The plan to establish it — a 1 000-treatment
  random draw, annotated but *not* reviewed — is
  [annotation-activity-split.md](annotation-activity-split.md).
* **M5 does gate on verified volume**, because retraining the
  segment classifier on verified data needs verified data.  That
  is the surviving half of the original claim.

Two further cautions on reading the curve, both verified
2026-08-23 and fixed as part of the companion plan:

* The notebook orders treatments by `created_at`, which
  `bin/llm_annotate_features` stamps **after** the API call
  returns.  With 5 workers that is completion order ≈ latency ≈
  output length, so label-poor treatments cluster at the head and
  **β is overestimated** — the analysis would demand more samples
  than are actually needed.  Use round-file order plus a
  permutation-averaged band.
* The notebook loads the whole candidate DB, so a new draw would
  be pooled with the biased rounds 1/2/4.  It needs a round filter.

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
  which depends on Track A volume.

So Track A volume is the pacing constraint on the *quality* of
v5, not on its *existence*.  We ship the noisy v5 first (M3+M4)
and improve it as verified data flows in (M5).

## What we're explicitly NOT doing first

* **Mistral-alternative structured-form work (concern 0)**.
  Deferred to M6.  No compounding value until extraction
  quality is fixed.  Building it earlier would sink
  engineering time into a productization step whose
  inputs are still known-broken.
* **Full extractor rewrite**.  Too big for a milestone;
  attacked incrementally via segment-classifier signals
  feeding into the current extractor.
* ~~**More vocabulary sampling before terminology dict
  pass**.  Track B first; new samples without it just
  replay the drift.~~  **Relaxed 2026-08-23.**  The
  notebook canonicalizes **post-hoc** via
  `feature_label_canonicalization.json` and plots raw and
  canonical curves side by side, so sampling first is safe:
  the map can grow later and the canonical curve be
  recomputed **without re-annotating**.  The ordering now
  runs the other way — take the baseline on the current
  prompt, then fix the label schema and re-measure the same
  sample.  Drift is real (318 distinct labels over 1 582
  annotations, **54 % singletons**, plus a systematic
  base+context family: `Asci` / `Asci protologue` /
  `Asci in culture MEA`), but it is measured *from* the
  baseline rather than blocking it.
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

* [../structured-form-schema.md](../structured-form-schema.md)
  — the phase-2 JSON schema and the reasoning behind it.
* [annotation-activity-split.md](annotation-activity-split.md)
  — the companion execution plan that corrects Track A's
  premise, establishes the Heaps baseline, and splits label
  validation from pathology detection.
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

* **2026-09-01** — **M4's premise flagged as contested.**  The
  round-5 dossier review (memo §12.3.1–12.3.42) found the
  §10/§12 failures to be architectural rather than
  per-line-tunable: `Table` and `Key` track page geometry
  rather than content, label boundaries fall *inside* lines,
  and 18 % of treatments are harvested from article front
  matter.  Experiment 6 proposes the matching decomposition.
  M4 is not withdrawn — it is much cheaper, and §12.3.41's
  document-concentration finding may let it plus triage
  capture most of the value — but the "gains compound"
  argument for doing it before Exp 6 no longer holds
  unexamined.  Cross-reference added above M4.  Also recorded:
  `production_v4_1` was created rather than `production_v5`,
  since re-extraction is a v4 re-**grouping** and does not
  trigger the M3 container.
* **2026-08-23** — **Track A's premise corrected.**  Its
  200-250 target was justified by the Heaps' Law vocabulary
  curve, but that curve is computed from `features_candidate`
  alone — vocabulary coverage is bought with API volume, not
  operator hours.  200-250 survives as an M5 *training-set*
  target.  Consequences: the "Heaps' Law dependency" section
  now separates the two claims; the "no vocabulary sampling
  before Track B" prohibition is relaxed, because the notebook
  canonicalizes post-hoc and sampling first costs nothing; M4
  gains a candidate change (a document-level taxonomic-article
  gate) after measuring that 49.9 % of source documents produce
  only empty-description treatments and are overwhelmingly
  non-taxonomic papers.  Two notebook defects recorded that
  would have biased the curve — latency-ordered x-axis and an
  unfiltered whole-DB load.  Full rationale and the execution
  sequence in
  [annotation-activity-split.md](annotation-activity-split.md).
* **2026-07-03** — initial draft created after the
  triage-CSV review pass wrap-up.  Six top-level
  concerns converted to milestones + two continuous
  tracks.  Milestone-driven cadence chosen over weekly
  due to expected paid-work commitment.  v5 timing
  fixed to M3 trigger.
