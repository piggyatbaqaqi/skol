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

> **2026-09-10 — read, then corrected 2026-09-10 after operator
> review.**  `skol_exp_production_v4_03_00_treatments_structured` holds
> **1 285 records at 100 % `json_valid`**, produced by
> `bin/treatments_to_json.py`.  A representative output:
>
> > `Ascomata are solitary and immersed, brown to black, coriaceous texture, and subglobose to depressed globose, ostiolate and clypeate`
>
> ```json
> {"Ascomata": {"Form": ["Solitary","Immersed"], "Color": ["Brown","Black"],
>               "Texture": ["Coriaceous"], "Shape": ["Subglobose","Depressed Globose"],
>               "Aperture": {"Type": ["Ostiolate"], "Accessory": ["Clypeate"]}}}
> ```
>
> **These are NOT the per-span re-run this paragraph asks for.**  Per
> the operator: stock Mistral, run on **whole descriptions** — the very
> configuration the paragraph above says was measured.  The re-run on
> labelled spans **has still not been done**, and it remains the next
> step.  *(An earlier version of this note claimed otherwise.  The
> structured database records no `model` field, so nothing on the
> documents distinguishes stock from fine-tuned or span from whole
> description; the operator's recollection is the only provenance.
> **Record a `model` and `input_granularity` field on the next run** —
> this ambiguity cost a wrong reading.)*
>
> **What the 1 285 records do settle: the objection was speed, not
> quality.**  "Mistral did poorly" is not supported by its own output —
> 100 % parseable, and the tree above is the shape the schema calls
> for.  The path was abandoned because whole-description inference was
> **prohibitively slow** for the application.  That reframes the
> re-run: per-phrase with constraints attacks throughput directly
> (short inputs, constrained decoding, batching), and it is being tried
> for speed with quality already evidenced, rather than the other way
> round.
>
> **`json_valid` means parseable, not consistent.**  Top-level keys
> over 400 valid records: `Species` 41 / `species` 27, `Genus` 39 /
> `genus` 13, `Family` 27, and the taxonomy ranks (`Kingdom`, `Phylum`,
> `Class`, `Order`, 59 between them) level with `Conidia` and
> `Conidiophores`.  Case drift on the commonest keys, no separation of
> rank from morphology — and this is exactly what a decode-time
> vocabulary constraint fixes.  **The vocabulary work is not
> misdirected; it is currently applied post-hoc where it needs to be
> applied as a constraint.**
> `docs/schema_constrained_pipeline.md` already designs that: Pass A
> induction → freeze the canonical schema → Pass B constrained
> extraction, with 25 882 canonical and 2 396 hand-verified annotations
> available to build the constraint from.

### M6 R1 result: per-clause is 2.9x SLOWER — 2026-09-11

**Measured.**  20 treatments, seed 20260910, stock
`mistralai/Mistral-7B-Instruct-v0.3`, 4-bit, RTX 5090 Laptop.
`bin/benchmark_json_granularity.py`; report in the session scratchpad.

| arm | s/treatment | speedup | calls | cap-hits |
|---|---:|---:|---:|---:|
| `whole` @1024 *(incumbent)* | 54.46 | 1.00x | 20 | 0.30 |
| `whole_512` | **37.88** | **1.44x** | 20 | 0.60 |
| `clause_seq` @384 | 157.02 | **0.35x** | 171 | 0.13 |
| `clause_batch` @384, batch 8 | 238.66 | **0.23x** | 29 | 0.13 |

**The per-clause premise fails, and not narrowly** — the kill criterion
was ≥3x faster and the result is 2.9x *slower*.

**Batching made it worse, which is the finding worth keeping.**  A
batch decodes until its *longest* member finishes; `model.generate` has
no per-sequence early exit.  Cap-hits is 0.135, so clauses individually
stop early — but in a batch of 8 every sequence pays for the slowest.
Per call: `clause_batch` 164 s for ~6 items against `clause_seq` 18.4 s
for 1.  That penalty more than cancels the dequantisation amortised
across the batch.  **Arm B is what made this visible**; with only the
two arms the card proposed, the conclusion would have been "clauses are
slow" and the reason would have been missed.

**The parameter hypothesis is half right and the right half is not
free.**  `whole_512` is genuinely 1.44x.  But cap-hits goes 0.30 →
0.60: at 1 024 a third of generations run to the cap, at 512 two
thirds.  That speed is bought by truncating more outputs.  The free
version is **a stop condition on the closing brace** — 30 % of
generations run to the full 1 024 while the extracted JSON is ~294
tokens, so the model emits its JSON and then keeps going.

#### What this actually says: the runtime is the problem, not the granularity

54.46 s/treatment x 81 527 treatments = **51 days**.  `whole_512` is 36
days.  Neither ships, so re-cutting the input was never going to be
enough.  `model.generate` on a 4-bit HF model — no continuous batching,
no early exit, no paged attention — is the wrong engine.  **The next
attempt belongs in the runtime** (vLLM or llama.cpp), and R5's
constrained decoding should be built against whatever that turns out to
be rather than against `transformers`.

`outlines` being already installed argued for staying on
`transformers`; this result outweighs that.

#### CAVEAT ADDED 2026-09-14: no arm used an instruction prompt

**Found while investigating a Qwen anomaly, and it is a defect in the
instrument.**  `_build_prompt` in `treatments_json_translator.py` wraps
the pipeline's task prompt (`DEFAULT_PROMPT`) in Mistral's
`<s>[INST]…[/INST]` format.  **`bin/benchmark_json_granularity.py`
never calls it.**  `build_workloads` puts the raw `description` string
in the `whole` arm and a bare `CLAUSE_PROMPT` in the clause arms, and
`batched_generate` tokenizes those directly — so a bare species
description was handed to an instruction-tuned model with no
instruction and no chat template.

**What survives.**  All four arms were treated identically and both
models were mis-prompted identically, so every *relative* comparison
holds: per-clause loses, batching hurts on Mistral, and Qwen-1.5B is
2.0x Mistral-7B.

**What does not.**  The **51-day figure was never the pipeline's
cost** — the real pipeline prepends a long instruction prompt, which
changes prefill and stopping behaviour in ways this run cannot
predict.  Treat it as "the incumbent is far too slow", not as a
number.  And **every quality figure is uninterpretable**: a model told
nothing has no stable behaviour to diverge from, which is why Qwen's
`seq_vs_batch_json` shows 85 of 159 differing against Mistral's 5 of
19.  The cap-hit comparison against the stored 1 285 records was also
apples to oranges for the same reason.

**Unexplained and not asserted.**  `clause_seq` and `clause_batch` run
identical prompts, yet Qwen's cap-hit rates are 0.053 and 0.626.  Too
large for padding artifacts, but not worth a claim from a mis-prompted
run.

**Ruled out, not assumed:** the EOS hypothesis.  Qwen's `eos_token` is
`<|im_end|>` (151645) and matches its config, so stopping was
configured correctly; the fault is upstream, and
`add_eos_token=True` — correct for the fine-tuning setup it was copied
from, wrong for generation — leaves Qwen at a turn boundary with no
assistant cue.

#### What a grammar can and cannot fix — 2026-09-15

Operator, reading the model outputs: *"If a model gives `thin, whitish
or yellow, hyphal or subiculum-like` instead of breaking those into
separate features `{"thickness": "thin", "color": ["whitish",
"yellow"], "composition": ["hyphal", "subiculum-like"]}` — is that
something a grammar could fix?  Or is it a model issue?"*

**Both halves have different answers, and the binding constraint is
neither.**

**Structure: a grammar fixes it outright.**  If the schema declares
`Subiculum → {thickness, color: [], composition: []}`, constrained
decoding makes emitting one blob *impossible* — the token sequence
cannot match the grammar.  A hard guarantee, not a nudge.

**Assignment: the grammar cannot decide it.**  Whether `thin` lands in
`thickness` is semantic.  But forcing the slots changes the task from
"describe this" to "what is the thickness?", and models are markedly
better at slot-filling than at open description, so assignment tends to
improve as a side effect — indirectly, not by construction.

**What actually blocks it: there are no slots to enforce.**  Measured
on `…_02_57_features_canonical`:

| | |
|---|---:|
| canonical spans | 25 882 |
| `attribute_path` **depth 1** | **25 373 (98.0 %)** |
| depth 2 | 509 |
| spans containing a comma | 19 946 (77 %) |
| spans containing " or " | 5 257 (20 %) |

Phase 1 labels `Subiculum` and stops.  There is no
`Subiculum → {thickness, color, composition}` anywhere to compile a
grammar *from*, while the decomposable structure sits in the source
text unlabelled — 77 % of spans carry a comma.  The operator's example
is real rather than hypothetical; here is a live span:

> `Context` — `"thin, rigid context, reddish brown color"`

**Point a grammar at today's schema and it will faithfully enforce
`Subiculum → one string` — the blob — forever.**

**So the ordering is R5a → R5b → model, and R5a carries the weight.**
If Pass A induces the slots, the grammar enforces them mechanically.
If it does not, R5b compiles a grammar that blesses the current
behaviour.

**A trap to design for now, not later.**  Required slots are what force
decomposition; optional slots let the model blob again.  But a required
`color` on a clause that never mentions colour invites invention.  The
primitive is already settled in this project: **absence is a *value*
(`presence: absent`), not a suppression.**  An explicit "not stated"
makes required slots safe.

Decomposition is compatible with
[structured-form-schema.md](../structured-form-schema.md)'s
verbatim-values decision — `thin`, `whitish`, `yellow`, `hyphal`,
`subiculum-like` are all verbatim substrings.  It is splitting, not
paraphrasing.

**Same problem as T6, seen from the other side.**  T6 of the annotation
plan requires partitioning labels into organ-names and property-names
before trusting the singleton rate.  `thickness` and `color` are
properties with nowhere to live, so they are absorbed into the organ's
text — which is why the vocabulary looks flat and why the singleton
rate is hard to read.

#### R7: the stop condition is a no-op, because R2 already won it — 2026-09-15

Mistral-7B bf16, 20 treatments, seed 20260910, `--stop-on-json` the
only difference between the two runs.

| arm | flag off | flag on |
|---|---:|---:|
| `whole` | 56.00 | 56.12 |
| `whole_512` | 38.91 | 39.01 |
| `clause_seq` | 69.79 | 68.80 |
| `clause_batch` | 35.66 | 35.62 |

Cap-hits identical to three decimals in every arm; **18 of 20 `whole`
outputs byte-identical**, the other two differing by ~4 characters,
which is kernel nondeterminism rather than the criterion.

**Not a bug — there is nothing to cut.**  Binary-searching each
persisted output for the point where its JSON first parses:

| arm | n | never closes | JSON closes at (median) | tokens after the close |
|---|---:|---:|---:|---:|
| `whole` | 20 | 7 | **100 %** of the output | 0 % median, 0 % max |
| `clause_seq` | 171 | 6 | **100 %** | 0 % median, 83 % max |

**The model closes its JSON at the very end of what it generates.**

**The premise was invalidated by the prompt fix, and I did not
re-derive it.**  R7 was justified by R1's *"30 % of generations run to
the full 1 024-token cap while the extracted JSON is ~294 tokens — the
model emits its JSON and then keeps talking"*.  That came from the
**mis-prompted** run: a model given a bare description with no
instruction has no reason to stop.  Given a real instruction it stops
on its own.  The corroborating figure is in the same reports —
**clause-arm cap-hits fell from 0.13 to 0.006**.

The caveat recorded on 2026-09-14 said in terms that R1's absolutes did
not survive R2.  R7 was then built on one of those absolutes without
re-checking it against R2, which was one query over data already on
disk.  Cost: ~2 cu and two hours of GPU.

**What survives:** `json_is_complete` is sound and tested; the flag is
off by default and harmless; it may still earn its place under vLLM,
where per-sequence stop strings are native, or on a model that does
ramble.

**A quality finding falls out of the same table and is NOT a speed
issue: 7 of 20 whole-description outputs (35 %) never produce a
complete JSON object at all.**  Worth carrying into R4.

#### R2: the re-run with proper prompting OVERTURNS R1 — 2026-09-14

Same 20 treatments, seed 20260910, now with each model's own chat
template and the pipeline's `DEFAULT_PROMPT` on the `whole` arm.

| model | quant | whole | whole_512 | clause_seq | clause_batch | cap-hits |
|---|---|---:|---:|---:|---:|---:|
| Mistral-7B | 4-bit | 64.25 | 42.50 | 79.46 | 141.04 | 0.20 |
| Mistral-7B | **bf16** | 56.00 | 38.92 | 69.77 | **35.65** | 0.20 |
| Qwen2.5-1.5B | bf16 | **17.10** | **13.04** | 30.61 | 16.47 | 0.10 |
| Qwen2.5-7B | 4-bit | 54.79 | 38.60 | 78.88 | 121.58 | 0.20 |
| Phi-3.5-mini | bf16 | — | — | — | — | **20/20 failed** |

**1. The incumbent is slower than R1 said.**  54.46 → **64.25
s/treatment**, because `DEFAULT_PROMPT` adds real prefill that the bare
description did not.  The honest baseline is **60.6 days**, not 51.
R1's figure flattered the incumbent.

**2. Per-clause batched WINS on bf16 — R1's headline is reversed.**
35.65 against 56.00 for `whole`, **1.57x faster**.  On 4-bit the same
arm is catastrophic (141.04).  So *"batching hurts"* was never a
property of batching; it was a property of **bnb-4bit**.  R1 measured
one quantisation and generalised.

**3. bf16 beats 4-bit — R8 answered.**  1.15x on `whole`, **4x** on
`clause_batch`.  The 17 GB left idle was costing throughput, exactly as
the near-idle-GPU measurement suggested.

**4. Size beats family.**  Qwen2.5-7B (54.79) ≈ Mistral-7B (64.25) at
the same quantisation, while Qwen2.5-1.5B is **3.8x the incumbent**.
Best configuration measured so far is **Qwen2.5-1.5B bf16 at
`whole_512`: 13.04 s/treatment ≈ 12 days** — still not shippable, but
five times better than where R1 left it.

**Independent evidence the prompt fix worked:** cap-hits fell from
0.30–0.65 to **0.10–0.20**.  Models now stop on their own instead of
running to the cap, which is what an instruction-tuned model does when
it is actually given an instruction.

**Phi-3.5-mini failed all 20 generations** and is excluded pending
diagnosis.  The harness swallowed the exceptions by design — a bad
prompt must not kill a run — but it records `failures` without the
*reason*, and that gap cost a whole model's data.  Worth fixing.

#### R9a: vLLM works on this Blackwell card — 2026-09-14

Built from source into `skol-vllm312`: **vllm 0.29.1rc1.dev91**, torch
2.13.0+cu130, CUDA 13.1, sm_120, conda-forge g++-14 as CUDA host
compiler.  Loads Mistral-7B in bf16, captures CUDA graphs, generates.
Load 501 s (mostly one-time graph capture), generate 2.29 s.

Four blockers on the way, each of which would have been a silent trap:
the torch pin (2.13.0 against `skol3.14`'s 2.12.0, which would have
been swapped under a running benchmark); a user-global pip config
pointing at a dead `pypi.ngc.nvidia.com` and a **cu129** index on a
cu130 box; and the host compiler — CUDA 12.9 refuses gcc > 14 outright,
CUDA 13.1 with gcc-15 hits an `rsqrt` exception-specification clash in
the glibc headers, and the distro ships `gcc-14` with **no matching
`g++-14`**.

*A fourth was misdiagnosed and is recorded for honesty: I attributed an
early failure to Python 3.14 via `mistral_common`, reading pip's
backtracking through versions 1.6–1.8 as the cause.  `mistral_common`
is at 1.11.7 with `requires_python <3.15`, and vLLM requires >= 1.11.6,
so 3.14 was never excluded.  Both failures were the compiler.  The
separate env is still right — because of **torch**, not Python.*

**Note for R9b:** a hand-formatted `<s>[INST]…` string gets a second
BOS from vLLM.  Formatting belongs to the server's chat endpoint, not
to us.

#### The GPU is nearly idle at batch 1 — 2026-09-14

Operator, watching the Mistral run: *"the mistral-7B model is only
using 7G of RAM on the GPU. Is there a way to take advantage of the
other 2/3rds?  I'm assuming the answer is no because we're keeping the
GPU utilization at 100%."*

**`utilization.gpu` does not mean what it appears to.**  It reports
that *at least one kernel was resident* during the sample, not that the
card is working.  Measured with `nvidia-smi dmon` over 8 samples on the
live run:

| | |
|---|---|
| SM "utilization" | **99 %** |
| memory controller active | **11 %** |
| SM clock | **225 MHz** — the idle clock; this part boosts past 2 000 |
| power draw | **82 W** |
| VRAM | 7.4 of 24.5 GB |

A card running flat out sits near its boost clock and near its power
limit.  **This one is at its idle clock drawing 82 W.**

**Why:** batch-1 autoregressive decode is *latency*-bound.  Each token
depends on the one before it, so there is no parallelism to exploit;
the kernels are small, they serialise on a dependency chain, and the
card never has enough concurrent work to justify ramping clocks.  99 %
utilisation at an idle clock is the signature of exactly that.

**This is the strongest argument yet for the runtime change.**  The
spare memory is only usable as *concurrent sequences*, and
`model.generate` cannot exploit them because a batch runs until its
longest member finishes — which is the penalty R1 measured directly.
Continuous batching and a paged KV cache exist to fix precisely that.

**Second, cheaper lever, queued 2026-09-14:** stop quantising the 7Bs.
bnb-4bit dequantises to bf16 on every matmul and its kernels are not
fast; the pipeline uses it presumably for a smaller card.  With 17 GB
spare there is no reason to pay that tax.  Which way it goes is
genuinely open — 4-bit reads a quarter as many weight bytes, bf16 skips
the dequant — so a bf16 Mistral-7B run is queued behind the four-model
comparison, same 20 treatments and seed, isolating quantisation as a
single variable.

#### Model candidates are licence-constrained — 2026-09-14

**SKOL will not accept the Llama licence.**  Operator's decision, and a
project constraint rather than a preference: it is not OSI-approved and
its **geographic restrictions would exclude a European SKOL
instance**, which is inconsistent with a permanently free-access,
GPL-3.0 project ([structure_ocr.md](../structure_ocr.md) §1).

**Licences are not uniform within a family, so check per size.**  I
proposed `Qwen2.5-3B-Instruct` as the third model and it is
disqualified on the same grounds — `license: other`, the Qwen Research
licence, non-commercial — while `0.5B`, `1.5B` and `7B` are all
Apache-2.0.  Verified from HF model-card tags:

| licence-clean | |
|---|---|
| Apache-2.0 | `Mistral-7B-Instruct-v0.3`, `Qwen2.5-0.5B/1.5B/7B-Instruct`, `SmolLM2-1.7B-Instruct`, `granite-3.1-2b-instruct` |
| MIT | `Phi-3.5-mini-instruct` |
| **disqualified** | `Qwen2.5-3B` (Qwen Research), `Llama-3.2-*` (Meta; 403 `GatedRepoError`), `gemma-2-*` (Gemma, gated) |

**Open weights vs OSAID is unresolved** and does not block anything.
The practical consequence: **OSAID 1.0 requires "data information"** —
enough detail about training data to recreate a comparable system — and
none of the models above meet it, Apache-2.0 included.  A strict OSAID
filter selects nothing runnable today; the weights licence is the
filter that actually selects, and is what the list above applies.

#### Qwen2.5-1.5B against Mistral-7B — 2026-09-14

Same 20 treatments and seed, Qwen in bf16 (bnb-4bit is a memory
optimisation; at 1.5 B the dequantisation kernel would dominate).
**Subject to the caveat above.**

| arm | Qwen 1.5B bf16 | Mistral 7B 4-bit | ratio |
|---|---:|---:|---:|
| `whole` | **27.07** | 54.46 | **2.01x** |
| `whole_512` | 14.93 | 37.88 | 2.54x |
| `clause_seq` | 41.01 | 157.02 | 3.83x |
| `clause_batch` | 33.42 | 238.66 | 7.14x |

25 days over the corpus against 51.  Better, still not shippable.

**Batching reverses between the models, which is why the batched arms
are kept.**  On Mistral, `clause_batch` is *slower* than `clause_seq`
(238.66 against 157.02).  On Qwen it is **faster** (33.42 against
41.01).  The penalty mechanism — a batch decodes until its longest
member finishes — scales with output-length variance, and
mis-prompting inflates exactly that.  So "batching hurts" is not a
settled property; it is a Mistral-at-this-prompt property.

**The untested configuration** is batching *whole descriptions across
treatments*.  `whole` carries one item per treatment at batch size 1,
so the incumbent has never been batched at all.

#### Two caveats on the instrument, both recorded rather than buried

* **Batching changes the ANSWER, not just the timing — 5 of 19.**
  Measured 2026-09-11 after the instrument was fixed, on a 3-treatment
  run: comparing *extracted* JSON rather than raw generations gives
  n = 19, 14 exact, **5 genuinely different**, 4 incomparable (no JSON
  on one side).  Every raw difference that yielded parseable JSON on
  both sides was a real content difference — 5 for 5.  The raw
  comparison had said **0 differ**, because everything non-identical
  fell into `unparseable`; had the instrument not been fixed this would
  have been recorded as clean.

  **No change to the generation path follows** — the batched arm is
  already dead on speed, so nothing depends on it.  What follows is a
  **provenance requirement**: the same model and prompt give different
  answers at different batch sizes, so `batch_size` joins `model` and
  `input_granularity` as a field R6 must record, and **R4's gold set
  and R4c's scorer must fix and record the batch configuration** — a
  score obtained at one batch size does not transfer to another.  The
  production config has to be the config that was scored.

  It also **does not generalise to a new runtime by assumption**.  vLLM
  and llama.cpp batch differently; if either is adopted, this has to be
  re-measured, not inherited.

* **`agree_rate` 0.655 is a lower bound and the comparison was
  flawed.**  `batched_generate` returns the raw continuation, but the
  pipeline extracts JSON with `_extract_json_object`.  Comparing raw
  generations made `json.loads` fail on everything not byte-identical,
  so `structural` is 0 and all 59 disagreements landed in
  `unparseable`.  Real agreement is probably much higher.  It cannot be
  recomputed: the report does not persist outputs.  **112 of 171
  clauses were byte-identical between the sequential and batched arms,
  with 0 genuine disagreements**, which is the reassuring direction.
* **The right-padding warnings are a false positive**, verified by
  re-encoding and inspecting the mask: padding *is* left.  HF's
  detector tests whether the last token equals `pad_token_id`, and
  `add_eos_token=True` with `pad_token = eos_token` trips it on every
  sequence.  Both arms share the tokenizer, so the comparison stands.

**Scope note:** 20 treatments, not the 50 the card specified.  At 54
s/treatment across four arms, 50 would have been a 5.5-hour benchmark,
and 20 separates a 3x effect comfortably.

### M6a: phrase labelling has no local path — 2026-09-10

Raised by the operator: *"how do we do the phrase labelling locally
that we've been using Claude API for?"*  **We do not.**  All **25 742**
candidate annotations in `…_02_50_features_candidate` carry
`model: claude-opus-4-7`.  `bin/llm_annotate_features.py` constructs
`anthropic.Anthropic(...)` and calls `client.messages.create` directly;
there is no local implementation and no interface behind which to put
one.

And: *"I don't remember any work on a per-phrase application of a local
SLM."*  Confirmed — there is none.  The only SLM code in the tree is
`bin/treatments_to_json.py` (whole descriptions) and
`bin/embed_lines.py` (SBERT, an encoder, not generative).

**The seam already exists, which makes this cheaper than it looks.**
`treatments_to_structured/llm_annotate.py` is already pure:
`build_user_prompt(synth_txt, seed) -> str` and
`parse_claude_response(text, model_name) -> annotations`.  Only the two
client call sites in `bin/llm_annotate_features.py`
(`client.messages.create`, `client.messages.count_tokens`) are
Anthropic-specific.  A local backend is: define the interface those two
calls imply, keep prompt-building and parsing untouched.

**Why it matters beyond cost.**  The platform is permanently
free-access (`docs/structure_ocr.md` §1), so a per-treatment paid API
call in the production path is a standing liability.  It also blocks
the fine-tuning question from ever being answered locally.

**Not scheduled here.**  It shares its target with the per-phrase
Mistral re-run above — both are "run a small local model on a short
span" — and should be decided together, not separately.

**M6 is the "full JSON representations of the descriptions" goal**
named in the 2026-09-10 planning session.  It is one of two active
goals; the other is Experiment 6, planned in
[experiment_6_design_and_implementation_plan.md](experiment_6_design_and_implementation_plan.md).
The two do **not** serialise: M6 runs on v4 data with today's
pipeline and needs nothing from Experiment 6.

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

**Measured 2026-09-01 on round 5.  The curve is no longer a
dependency to be waited on — it has been fitted, cross-checked and
read.**  What follows replaces the estimate this section used to
describe; the notebook defects it warned about are fixed.

### The fit, and why it is trustworthy

Round 5 alone — 1 000 drawn, 877 producing labels, 7 486 annotations —
with `permutation_band` over 200 permutations of the full drawn set.
Rounds 1, 2 and 4 are selection-biased toward suspected problems and
**must not be pooled** into this.

*(The notebook reports 7 480 annotations, V = 957 and 555 hapax
against the 7 486 / 961 / 559 quoted here: it also drops the
hand-picked canary `taxon_46ff7dde`, which carries the round-5 stamp
but was not drawn.  β is 0.601 either way.)*

| | |
|---|---|
| distinct labels `V` at n=1 000 | **961** |
| **β**, fitted on n ∈ [200, 1 000] | **0.601** (band 0.54–0.66) |
| `K` | 15.2 |
| β fitted from n=1 | 0.645 — head-inflated |

Two independent cross-checks say this is a real power law rather than
a curve that merely fits:

* **Hapax fraction.**  559 of 961 types (58.2 %) appear in exactly one
  treatment.  For a Zipf/Heaps process the singleton fraction
  converges to β: observed **0.582** against fitted **0.601**.
* **Good–Turing against held-out measurement.**  Singleton labels
  carry 621 of 7 486 instances = **8.3 %** missing mass.  Measured by
  permutation, an unseen treatment has **91.7 %** of its annotation
  instances already covered — 8.3 % uncovered.  Two estimators, no
  shared assumptions, the same number.

### `V(n)` is the wrong target; coverage is the right one

β = 0.6 means **each doubling of the vocabulary costs 3.17× the
samples**: 1.25× V needs n≈1 450, 1.5× needs 1 960, 2× needs 3 170,
and annotating all 38 303 eligible treatments reaches V≈8 600 while
still adding 0.13 labels per treatment.  The vocabulary never
saturates, so "how many for a complete vocabulary" has no answer.

The measured coverage curve does, and assumes no functional form:

| after n | distinct labels of an unseen treatment already known | its annotation instances |
|---:|---:|---:|
| 100 | 79.6 % | 80.3 % |
| 250 | 85.9 % | 86.4 % |
| 500 | 88.1 % | 88.3 % |
| 1 000 | 91.5 % | **91.7 %** |

Roughly **+2–3 points per doubling**.  95 % costs another ~2 000–3 000
treatments; 98 % is out of reach of the entire eligible population.

**Recommendation: one further round of 1 000–2 000**, targeting
~94–95 % instance coverage and roughly doubling the df ≥ 5 support
set (183 labels at n=1 000; df ≥ 10 gives 90, df ≥ 20 gives 47).
Past that the marginal label costs more than it is worth.

*(Executed as round 6, 2026-09-01 — coverage landed at 94.1 %, df ≥ 5
at 288.  See "Round 6" below for the out-of-sample validation.)*

### The growth is corpus breadth, not annotator drift

The obvious worry — that an LLM emitting free text inflates β with
invented phrasings — was tested and does not hold.

* The canonicalization map collapses **20 of 961** forms; raw and
  canonicalized curves are indistinguishable (β 0.601 vs 0.604).
* Of the 559 hapax labels, **58.5 % have no string neighbour at all**
  among the 183 labels with df ≥ 5, and inspection shows real
  structures from taxonomically distant groups — `Interfacial
  envelope`, `Mycelial pellicle`, `Base of endoperidial body`,
  `Capitate hyphopodia`, `Spot Tests`.  The corpus spans lichens,
  myxomycetes, hyphomycetes and coelomycetes, each with its own organ
  vocabulary.
* **String similarity is not a classifier here** and must never be
  used as one.  35.6 % of hapax labels get a "loose" match (0.70–0.88)
  and the matches are mostly false: `Pycnidiospores → conidiophores`,
  `Apothecium → hypothecium`, `Mitochondria → microconidia`,
  `Asterocystidia → pleurocystidia`.  Mycological morphology is built
  from a small set of morphemes, so string proximity says little about
  biological identity.  This is the hazard
  `docs/feature_label_non_synonyms.md` exists to record; a similarity
  sweep is a candidate *generator*, never a decider.
* OCR damage reaches the spans (`Pyenidiospores`, `Lo wER SURFACE`)
  but **zero** hapax labels look OCR-damaged — the annotator repairs
  OCR silently when emitting a label.

**Do not extrapolate the support-thresholded curves.**  df ≥ 2, 3, 5,
10 and 20 fit β of 0.69–0.87 at n ≤ 1 000, *higher* than V₁'s.  That is
a small-n transient — those curves are still fed by the singleton pool
and must asymptotically share V₁'s exponent.  Only the β = 0.601
estimate, corroborated by the hapax fraction, is safe to project.

### Round 6: the model was tested out of sample and held

Drawn and annotated 2026-09-01, 1 000 treatments, seed 20260901.
**958 success, 34 partial, 8 error; 8 145 annotations; $28.64; 25
minutes at 5 workers** (round 5: $26.79).

**The headline is the out-of-sample test.**  The 91.7 % coverage
figure above was an *in-sample* permutation estimate — it asked what
round 5's own treatments looked like to a vocabulary built from the
rest of round 5.  Round 6 is 1 000 treatments that vocabulary has
never seen:

| | predicted (in-sample, round 5) | observed (out-of-sample, round 6) |
|---|---:|---:|
| distinct labels already known | 91.5 % | **90.9 %** |
| annotation instances already known | 91.7 % | **91.3 %** (90.6 % pooled) |

**The estimator was optimistic by well under a point.**  That is the
validation the whole method needed, and it arrived despite round 6
drawing from a slightly *broader* population (see below), which
should if anything have depressed coverage.

**The power law extrapolated correctly across a doubling.**

| | round 5 alone | rounds 5+6 |
|---|---:|---:|
| β (fit on n ≥ 200) | 0.604 | **0.595** |
| hapax fraction (should track β) | 0.586 | 0.577 |
| V, canonical | 938 | **1 480** — the fit predicted 1 429 |
| Good–Turing missing mass | 8.3 % | **6.1 %** |

Coverage rose **91.6 % → 94.1 %** over the doubling, **+2.4 points**,
inside the +2–3 predicted.  Round 6 contributed 542 labels new to
round 5's 938.

**The support set is where the round paid off most:**

| | n=1 000 | n=2 000 |
|---|---:|---:|
| labels with df ≥ 2 | 402 | 626 |
| df ≥ 5 | 183 | **288** |
| df ≥ 10 | 90 | **163** |
| df ≥ 20 | 47 | **92** |

#### The population is not identical to round 5's, deliberately

Round 5 ran the merge filter at `--merge-threshold 10`; the default is
now **15** (`2612ab9`), after 10 measured at **51.7 % precision**
against 30 hand verdicts — wrongly excluding roughly 3 111 of the
7 632 treatments it flagged.  Round 6 was left at 15 rather than
forced back to 10, for two reasons: 10 is the worse filter, and a
threshold-10 run would write `features_status` skip docs for ~2 112
legitimate treatments, decisions that every later draw *trusts as
prior*.  A bad filter's mistakes would have become sticky.

**Consequence:** round 6's pool is 39 539 against round 5's 38 303, so
about 5 % of round 6's pool is treatments round 5 excluded.  A naively
pooled Heaps curve is therefore contestable — which is why the
headline measurement above is the **out-of-sample coverage test**, not
a pooled curve extension.  The pooled figures are reported because
they agree, not because they are load-bearing.

#### What a third thousand would buy

Projecting the missing mass (which scales as n^(β−1)): 6.1 % at
n=2 000 becomes ~5.2 % at n=3 000, so coverage ~94.8 % — **about +0.9
points for ~$29**.  **On coverage grounds the case is now weak**: the
curve has visibly flattened and 98 % remains out of reach of the whole
eligible population.

**Audited 2026-09-02 — see
[feature_label_singletons.md](../feature_label_singletons.md).**  Of a
50-label random sample of the 854 singletons, **56 % are genuine rare
traits and ~44 % are consolidatable** (260–500 of the tail, ±14
points at n=50).  A quarter of the tail matches five mechanical
shapes, and **39 singletons were minted from spans saying the
structure is absent** — a prompt defect that manufactures both
singletons and compounds.  Consolidating first makes the next
thousand worth more; it does not make it urgent.

**The support set is the better reason to keep going, if there is
one.**  Labels with df ≥ 20 went 47 → 92 across this doubling; another
thousand would put it near 130.  Whether that matters depends on what
M5 needs to train and validate against — a coverage target does not
answer it.  **Decide against the support curve, not the coverage
curve.**

#### Two operational findings

* **The dropped-span recovery script does not exist.**  Every run
  prints *"run an offline-recovery script against …features_status to
  retry dropped spans"*, and `llm_annotate_features` records the
  `dropped_spans` payload for exactly that purpose, but nothing in
  `fixes/` or `bin/` reads it.  **89 spans are waiting** (round 6: 48,
  round 5: 37, rounds 2 and 4: 4), 83 of them the recoverable kind —
  *text not found in synthetic doc* — which the recorded payload was
  designed to retry with NFKD and dash normalisation **at no API
  cost**.  The other 6 crossed a field boundary and need a human.
* **No new drift appeared.**  Of round 6's 1 041 raw label forms, 16
  were moved by the hand map and **0 by the rules** — a thousand fresh
  treatments produced no new `sexual`/`asexual` head-noun variant.
  The rules remain preventive, as designed.

### Notebook status

`jupyter/heaps_law_analysis.ipynb` **no longer carries the two defects
this section used to warn about.**  Both were fixed by extracting the
logic into `treatments_to_structured/heaps.py` with tests:

* **F1, the latency-ordered x-axis** — the estimator is now
  `permutation_band` over 200 random orders; completion order survives
  only as a second panel explicitly labelled "the ordering artefact,
  not the curve".
* **F2, the unfiltered whole-DB load** — cell 4 filters on the stamped
  `round` field (`ROUND = 5`, 9 068 → 7 480 annotations) and plots
  against the *drawn* population from the round file, so treatments
  that yielded nothing still advance the x-axis.
* A third, latent defect is guarded rather than assumed away: the old
  curve keyed on `created_at`, so a shared timestamp both
  mis-attributed and double-counted.  `timestamp_collisions` asserts
  zero on every run.

**One live defect remains.**  The notebook's `heaps_beta` fits over
the whole range including n < 200, which reports 0.645 where the
tail-fitted estimate is 0.601 — the head, where nearly every label is
new, inflates the slope.  Fit the upper range, or report both and say
which is the estimate.  A second, smaller improvement: cell 8 carries
its own copy of the canonicalization loader, so it misses the
rule-shaped forms `treatments_to_structured.feature_label_rules`
handles; `labels_by_treatment` would need to accept a callable as well
as a mapping.

### How this interacts with the milestones

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

* **2026-09-11** — **R1 ran: per-clause is 2.9x slower, and the runtime
  is the real constraint.**  Full table in the M6 R1 result section
  above.  Three things change because of it.  **R5's constrained
  decoding should target the new runtime**, not `transformers` —
  `outlines` being pre-installed argued for staying put and this
  outweighs it.  To be precise about what changed: **R1 measured
  nothing about `outlines` itself.**  Constrained decoding is not an
  independent choice — it is provided *by* a runtime (`outlines` for
  `transformers`, guided decoding for vLLM, GBNF for llama.cpp) — so
  "we already have `outlines`" was only ever an argument for staying on
  `transformers`, and R1 removed the grounds for staying.  **R3/R4 are unaffected**: the measurement parser and
  the gold set are runtime-independent and still pay.  And **a stop
  condition on the closing brace is now the cheapest available win** —
  30 % of generations run to the full 1 024-token cap while the
  extracted JSON is ~294 tokens.

* **2026-09-10** — **Planning session: two active goals, and the
  relationship between v5 and Experiment 6 stated.**  The operator's
  two goals are M6 (full JSON descriptions) and Experiment 6 (pipeline
  rewrite).  There is no experiment 5; `production_v5` in this document
  is the next *extraction run* of the current architecture, while
  Experiment 6 is a *replacement* architecture — different axes, not
  successive versions.  **M4/M5 are therefore at risk of being
  overtaken**: if Experiment 6's layered pipeline lands, a v5 run of
  the single-CRF architecture buys little.  Not cancelled here, but
  flagged — the decision belongs with the Experiment 6 phasing.
  M1–M3 (detector work on v4) are unaffected and still pay.

  **What was recorded in M6 above**: the Mistral re-run this plan asked
  for has already happened — 1 285 records, 100 % `json_valid` — so the
  fine-tuning question turns on key *consistency*, not capability.

  **Two increments identified that turn §12.3 into user-visible value**,
  neither needing a pipeline change and neither previously scheduled:
  surfacing `predicted_issues` in the Django UI, and harvesting the
  ~2 757 documents from §12.3.61 that named new taxa and produced no
  named treatment because the act landed in `Misc-exposition`.  The
  second is corpus growth on v4 with existing infrastructure and is the
  largest business-value increment available without the rewrite.

* **2026-09-09** — **Round 7 read: the support projection held, the
  coverage argument stayed marginal, and the round is richer than the
  two before it.**  Round 7 ran 2026-09-03 and sat unread for six days.
  Reading it required rebuilding `features_canonical` first — the
  09-03 build ran before the round finished writing and was 49 records
  short, leaving 30 round-7 annotations with no canonical record.  The
  rebuild moved round 7 only (V 901 → 909); rounds 5 and 6 were
  untouched, so the baseline comparison is clean.

  **The support curve, which is why the round was drawn.**  Measured on
  the drawn population, canonical labels, permutation estimator:

  | | n=1 000 (r5) | n=2 000 (r5+6) | n=3 000 (r5+6+7) |
  |---|---:|---:|---:|
  | df ≥ 5 | 169 | 274 | 347 |
  | df ≥ 10 | 85 | 162 | **214** |
  | df ≥ 20 | 50 | 91 | 127 |

  The round-7 commit projected **~210 at n = 3 000** for df ≥ 10.  It
  landed at **214**.  Gains per thousand are decelerating but not
  flattening: df ≥ 10 added 77 in the second thousand and 52 in the
  third, df ≥ 20 added 41 then 36.

  **Coverage behaved as predicted, and the estimator is optimistic in
  the same direction twice.**  Vocabulary from rounds 5+6 (1 199
  canonical labels) against held-out round 7: predicted 95.3 %
  (Good–Turing), observed **94.4 % pooled / 94.6 % mean**.  Optimistic
  by 0.7 points, against 0.3 points on the round 5 → round 6 test.  Two
  consecutive out-of-sample validations; the model is usable and
  slightly generous, and the generosity grows as the held-out round
  gets richer.

  **Round 7 is measurably richer than 5 and 6, and that is the finding
  worth keeping.**  V 909 against 795 and 835; hapax 60.5 % against
  55.6 % and 53.3 %; β 0.610 against 0.575 and 0.560; missing mass
  7.5 % against 6.6 % and 6.2 %.  It contributed **392 labels unseen in
  2 000 prior treatments**.  Two explanations were tested and both
  fail: the document mix is identical (1.26 treatments per source
  document, the same as round 6), and the novelty is **diffuse** — 202
  documents carry it and the top 10 % hold only 31 % of the mass,
  against the 57 % concentration §12.3.41 found for boundary theft.
  So the pooled curve understates how much vocabulary is still
  arriving, and **β for the pooled 3 000 (0.569) is above the pooled
  2 000 (0.559)** rather than below it.

  **The five canonicalization rules generalize.**  They were derived
  from rounds 5+6; on round 7 they fire at 1.13× (`map`) to 1.69×
  (`condition`) the per-1 000-annotation rate they achieve on their
  own derivation set, with `sub_attribute` at 1.62× and `case_fold` at
  1.53×.  They are structural, not fitted to the strings they were
  built from.

  **A latent selector defect surfaced and is fixed.**
  `--exclude-annotated` read only the candidate DB — "treatments with
  at least one annotation" — so a treatment the annotator ran and got
  *nothing* from left no candidate document and stayed drawable
  forever.  Round 7 re-drew **six** such treatments from rounds 5 and 6.
  No API money was lost (the annotator's own status filter skipped
  them) but the round file over-claims: realized n is 994.  The leaking
  population is exactly the empty-description class — about 11 % of
  every round, 334 treatments at round 7, growing ~110 per round — so
  the leak is biased toward the treatments worth sampling least, and
  its rate grows.  Round 6 drew 0 against an expectation of ~3; round 7
  drew 6 against ~5.8, which is the expected rate, not bad luck.
  `fetch_attempted_treatment_ids` now consults `features_status`;
  exclusion goes 1 871 → 3 111.  Merge-suspect skips are deliberately
  excluded from *this* filter so lowering `--merge-threshold` can still
  reach them.

  **The notebook was reporting a vocabulary a fifth too large.**  Cell 8
  claimed to canonicalize "through the same entry point the writers
  use", but called `canonicalize(label, map)` context-free.  Four of
  the five rules need the `known` and `established` indices
  `build_canonical_labels` supplies, so only the hand map fired: 1 480
  labels for rounds 5+6 where the pipeline says 1 199, and 418 labels
  over-counted at n = 3 000, with β inflated to match.  The canonical
  curve now reads `features_canonical` directly.  Two consequences
  recorded in the cells: this is *hindsight* canonicalization (the
  builder's indices saw the whole corpus, so early points know their
  future — `heaps.prequential_band` is the causal counterpart, and the
  notebook still does not use it), and cell 12's drift heuristic
  changed meaning, since drift now measures what the rules **removed**
  rather than what is outstanding.  Its replacement reports the
  residual canonical singleton fraction: **550/909, 61 % for round 7**.

  **Round 8 is not drawn, and the reason is Track A, not the curve.**
  A fourth thousand buys perhaps +40–50 labels at df ≥ 10 and under a
  point of coverage for ~$29 — still positive, and round 7's richness
  argues sampling has not saturated.  But 3 000 treatments of candidate
  annotation now stand against **147 hand-reviewed treatments**, and
  **rounds 6 and 7 are entirely unreviewed** — 16 674 candidate
  annotations whose precision has never been checked, and none of it
  tested since the canonicalization rules landed.  The binding
  constraint has moved from vocabulary volume to knowing what the
  volume is worth.  See the amendment to the annotation-activity split.

* **2026-09-02** — **The singleton tail audited; the label schema, not
  the sampling, is the open question.**  854 of 1 480 canonical labels
  appear once.  A 50-label sample classifies 56 % as genuine rare
  traits (rusts, smuts, slime moulds, lichens — rare because the taxa
  are), 20 % as collapsible drift and 24 % as not-a-trait
  (property-of-trait, compounds, artifacts).  Counting shapes across
  the whole tail: 73 are `<established> + property`, 59 compounds, 46
  base+growth-condition, 39 minted from *absence* statements, 36
  case-only duplicates — 221 distinct, 26 % of the tail, reachable
  without judgment.  Ranked follow-ups in
  [feature_label_singletons.md](../feature_label_singletons.md); the
  prompt fix for absence statements is first because it is the only
  one that stops the tail growing.  Also landed: the out-of-sample
  coverage measure moved into `treatments_to_structured.heaps` with
  tests and a notebook cell, so the round-6 validation is reproducible
  rather than a scratchpad result.

* **2026-09-01 (later)** — **Round 6 run; the coverage model was
  tested out of sample and held.**  1 000 treatments, $28.64, 25
  minutes.  A vocabulary built from round 5 covered **91.3 %** of a
  fresh round-6 treatment's annotation instances against the 91.7 %
  in-sample prediction — optimistic by well under a point, and
  measured on a slightly broader population.  β moved 0.604 → 0.595
  across the doubling and V landed at 1 480 against a predicted 1 429,
  so the power law extrapolates.  Coverage 91.6 % → 94.1 %, missing
  mass 8.3 % → 6.1 %, df ≥ 20 support 47 → 92.  Round 6 kept the
  current `--merge-threshold 15` rather than reproducing round 5's 10,
  so the populations differ by ~5 % and the out-of-sample test, not a
  pooled curve, is the load-bearing measurement.  **A third thousand
  buys ~+0.9 coverage points for ~$29 — decide it on the support
  curve, not the coverage curve.**  Two operational findings: the
  dropped-span recovery script every run advertises does not exist
  (89 spans waiting, 83 recoverable at no API cost), and a thousand
  fresh treatments produced no new drift the rules would catch.

* **2026-09-01** — **The Heaps' Law curve was measured, and the
  section rewritten from estimate to result.**  β = 0.601 on
  n ∈ [200, 1 000], cross-checked two ways: the hapax fraction
  implies 0.582, and Good–Turing missing mass (8.3 %) matches the
  measured held-out coverage (91.7 %) exactly.  The decision-relevant
  reading is the coverage curve, not `V(n)`: +2–3 points per
  doubling, so one further round of 1 000–2 000 buys ~94–95 % and
  98 % is unreachable.  The drift worry was tested and dismissed —
  the canonicalization map moves 20 of 961 forms, 58.5 % of hapax
  labels have no string neighbour at all, and the singletons are real
  structures from taxonomically distant clades.  Recorded with it:
  string similarity must never decide a merge here (35.6 % of hapax
  get a loose match and the matches are mostly false), and the
  support-thresholded curves are small-n transients that must not be
  extrapolated.  **Both notebook defects this section warned about
  are fixed** — `permutation_band` replaced the latency-ordered
  x-axis and cell 4 filters on the stamped `round` — leaving one
  live defect (`heaps_beta` fits from n=1 and reports 0.645) and one
  improvement (cell 8's private canonicalization loader misses the
  rule-shaped forms).

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
