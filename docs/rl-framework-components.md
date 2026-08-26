# Components for an RL framework

*Started 2026-08-26. A catalogue of signals this project already
produces, or could cheaply produce, that might serve as **reward or
diagnostic signal** for training the next layout/segmentation model.*

## What this document is, and is not

It is a **parts list with load ratings**. Each component says what it
measures, what it costs, what it needs, and — most importantly — **how
it fails if you optimise against it**.

It is **not** a proposed reward function. Every signal below is a
measurement first; turning a measurement into a reward is a separate
decision, and the place where the measurement's assumptions become
exploitable rather than merely wrong.

**The organising caution.** A diagnostic can be biased and still
useful, because a human reads it and discounts. A reward cannot: the
model will find the bias and live there. So each component here is
graded twice — once as a diagnostic, once as a reward — and several
that are excellent as the first are actively dangerous as the second.

---

## 1. Confusion-like matrices: how well are two labels differentiated?

The question these answer: **for labels *i* and *j*, how reliably does
the system tell them apart?**

Four routes, in increasing distance from ground truth.

### 1.1 Confusion against hand-labelled data

**What it measures.** The real thing: predicted label against a human's
label, block by block.

**Status: already built.** `bin/evaluate_golden.py:499`
(`compute_confusion_matrix`) aligns predicted blocks to
`skol_golden_ann_hand` and prints the matrix at `-v -v`.
`skol_classifier/base_model.py` computes one over Spark predictions as
well. Nothing new is needed to start; what is missing is *reading* the
existing output as a pairwise-differentiation question rather than as
per-class accuracy.

**Cost.** Near zero.

**Caveats.**

* **Bounded by the golden set.** Pairs that are rare in the golden data
  get cells estimated from a handful of blocks, and the interval on
  those is far wider than the point estimate suggests.
* **The golden set is not a random sample of the corpus.** It was
  curated, so genre coverage is whatever curation happened to include —
  and §5.5, §5.5.1 and §5.7 all show genre is the dominant failure axis.

**As a reward: degenerates into overfitting the golden set.** With a
few thousand labelled blocks and a model of any capacity, optimising
this directly memorises them. Usable as a *held-out gate*, not as the
objective.

### 1.2 From the model's own posteriors

**What it measures.** For each block, the model's probability
distribution over labels. Averaging the mass placed on *j* over blocks
where *i* won gives a confusion-shaped matrix **with no ground truth at
all**.

**Feasible.** `skol_classifier/base_model.py` already carries
per-class probabilities in its prediction frame; the plumbing exists.

**Cost.** One inference pass over `ann_combined` — ~21 000 documents.

**Caveats — and this is the important one.**

* **Posteriors measure model uncertainty, not truth.** A confidently
  wrong model reports excellent differentiation. This is not a
  hypothetical here: §5.5.1 found 3 396 structurally identical blocks
  in one document assigned seventeen different labels, which means the
  model was making *confident, arbitrary* calls. Posterior-based
  confusion would have called that document well-differentiated.
* **Calibration is unmeasured.** Before trusting posteriors for
  anything, check calibration against 1.1 on the golden set. If they
  disagree, the posteriors are decoration.

**As a reward: degenerates immediately.** "Be more certain" is trivially
satisfiable by sharpening the output distribution without changing a
single argmax. Any reward built on model-reported confidence rewards
overconfidence. **Do not use this as an objective.** It is a diagnostic
for finding *where* the model is unsure, nothing more.

### 1.3 Feature-space separability

**What it measures.** Something different and arguably more useful:
**can these two labels be separated at all**, given the features
available? Fisher ratio or silhouette between per-label centroids.

**Cost.** Low. No model, no labels beyond the existing ones.

**Why it earns a place.** It distinguishes **model error** from
**schema error**. If `Notes` and `Diagnosis` occupy the same region of
feature space, no amount of training separates them — the answer is to
merge the labels, add a disambiguating feature, or accept the
ambiguity. Routes 1.1 and 1.2 cannot tell those cases apart; they just
report a hot cell either way.

**Caveats.**

* Answers a question about *features*, not about the *labels'*
  legitimacy. Two labels can be genuinely distinct concepts that the
  current feature set happens not to capture.
* Sensitive to feature scaling in a way the others are not.

**As a reward: not applicable, and that is fine.** This is a design
instrument — it tells you what to change about the *problem* before you
start optimising.

### 1.4 Label entropy within shape-clusters

**What it measures.** Cluster blocks by surface form, then measure the
entropy of the label distribution inside each cluster. High entropy
means structurally identical inputs are getting different labels.

**Cost.** Lowest of the four. **No model access and no labelled data** —
only `ann_combined`, which is already on disk.

**Precedent.** This is the ad-hoc form that produced §5.5.1's headline:
3 396 blocks matching `Genus Author (N)` in one document, seventeen
labels, none above 32 %. That single number did more to characterise
the failure than any per-class accuracy figure in the memo.

**Caveats.**

* **"Same shape" is a modelling choice**, and the result is only as
  good as the clustering. The `(N)` regex above was hand-picked because
  the genre was already understood.
* **Legitimate variation exists.** Two identically-shaped blocks *can*
  deserve different labels from context — a bare binomial is
  nomenclature in one position and a cross-reference in another.
  Entropy counts that as failure.

**As a reward: degenerates into label collapse.** A model that assigns
`Misc-exposition` to everything scores perfect consistency. If used at
all it must be paired with a term that rewards *using* the label space
— and at that point you are hand-designing a balance, which is where
reward hacking lives. **Consistency is not correctness.**

### 1.5 Two properties any such metric must have

**It must be asymmetric.** P(pred = `Notes` | true = `Diagnosis`) is
not P(pred = `Diagnosis` | true = `Notes`), and both directions occur
here: D19 measured 32.3 % of *trailing* diagnoses being misrouted
`Notes` blocks, while `taxon_47c3b37d` (2026-08-26) is a
diagnosis-like Notes that stayed `Notes`. A symmetric "confusability"
score averages away the thing worth seeing.

**It must be paired with a per-label sink measure.** A *missing class*
does not appear as a hot pair. §5.7 found the schema has no `Abstract`
label at all, so abstract blocks scatter 70/10/9/5/3/2 % across six
labels — diffuse mediocrity, not a confusable pair. The complement to a
pairwise matrix is the **entropy of what lands in each label**, which
is what surfaces sinks: `Misc-exposition` at 35.4 % of all blocks
(§12.2) is a sink, not a confusion.

**But a sink is not automatically worth splitting**, and this is where
a metric stops being able to advise you. §5.7 read the scattered
abstracts as an argument for adding an `Abstract` class; narrowed
2026-08-26, it mostly is not. The segmenter is two-stage, structural
matter is Pass 1's business, and Pass 1's dominant error runs the other
way — **~18 000 blocks of real content discarded as artefact against
~840 of abstract leaking in**. High entropy in a sink tells you the
label is doing many jobs; it does not tell you whether any of those
jobs matter downstream. **Pair every sink measurement with the cost of
what lands there.**

### 1.6 Where to start

**1.4, then 1.1.** 1.4 needs nothing that is not already on disk,
directly measures the failure the round-5 review kept hitting, and its
degenerate solution is irrelevant while it is used as a diagnostic.
1.1 then grounds it against real labels wherever the golden set has
coverage.

1.2 is worth computing **only** to check calibration against 1.1. 1.3
is worth computing before any retraining decision, because it is the
one that can say "stop, this is a schema problem."

---

## 2. Pathology filters as signal *(to be written)*

The detectors accumulated in
`docs/data_quality_production_v4_model.md` — D1 through D19, plus the
§12.2 block-level rules — are candidate reward or curriculum signals,
and several are cheap enough to run every step.

To fold in, with the same two-column grading:

* **Self-labelling blocks** (§12.2) — a block whose first line reads
  `Etymology –` is an etymology whatever the layout pass called it.
  ~15 400 corpus-wide, and there is no inference step to get wrong.
* **Registry identifiers** (§12.2) — 74.2 % mislabelled, the highest
  rate measured; a lower bound, since the measurement missed bare `MB`.
* **`Key` blocks with no numbered couplets** (§12.2) — only about half
  of legible `Key` blocks are keys.
* **Span-count signals over term repetition** (§6.1, §5.5.1) —
  `>1 diagnosis span` catches merges `merge_metric` cannot see, and
  `merge_metric` itself measured 51.7 % precision.
* **`annotation_count == 0` with prose present** (§5.6) — 10/10
  precision as a "this is not a treatment" signal.
* **Detectors that failed, and why** — gap-block density (§12.2), the
  vacuous gap test in `recover_bands`, the `Asexual morph` contradiction
  (7/7 false positives). A parts list is more useful with the rejected
  parts still on it.

**The caution to carry into that section**: most of these detect
*extraction* faults, not *classification* faults. Rewarding a model for
not tripping them optimises the wrong stage unless the model is the one
producing the labels they read.

---

## Cross-references

* [data_quality_production_v4_model.md](data_quality_production_v4_model.md)
  — §5.5–§5.7 (genre failures), §6.1 (merge-metric precision),
  §12.1–§12.2 (label schema and layout measurements), D1–D19
* [plans/annotation-activity-split.md](plans/annotation-activity-split.md)
  — T3b's label inventory, T4's estimator work
* [structured-form-schema.md](structured-form-schema.md) — §4, the same
  argument in a different domain: do not spend model capability on what
  a deterministic tool does better
