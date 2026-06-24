# Schema-Constrained Two-Pass Pipeline for Prose → Structured JSON

A design document for converting prose taxonomic descriptions into structured,
hierarchical JSON. Written to be **taxon-agnostic**: fungi are the first target,
but every design choice below is meant to generalize to plants, insects, lichens,
or any descriptive-taxonomy corpus.

---

## 0. Design goals

1. **Consistency over cleverness.** The same feature should always land in the
   same place in the JSON, across documents and across languages. Variance is the
   enemy; we remove degrees of freedom rather than hoping the model behaves.
2. **Speed by decomposition.** Replace one hard prose→deep-JSON jump with several
   easy steps, most of which are cheap (CRF, regex, embedding lookup) and only one
   of which is a constrained LLM call on short spans.
3. **Taxon portability.** Nothing in the *machinery* is fungus-specific. Only the
   **schema** and the **gazetteers** change per taxon group. Swapping clades means
   swapping data files, not rewriting code.

---

## 1. Architecture overview

The core idea is two passes separated by a one-time aggregation step:

- **Pass A (induction, run once per corpus):** discover the candidate structure —
  what features, subfeatures, and value types actually occur in the literature.
  Output is a *canonical schema*.
- **Aggregation (run once):** merge, deduplicate, prune, and freeze Pass A's
  output into a single fixed schema + controlled vocabularies.
- **Pass B (extraction, run per document):** fill the frozen schema using
  schema-constrained decoding. This is the fast, repeatable production pass.

```
                          ┌─────────────────────────────┐
   corpus sample ───────► │  PASS A: structure induction │
                          │  (LLM, ungrounded, sampled)  │
                          └──────────────┬──────────────┘
                                         │ candidate triples
                                         ▼
                          ┌─────────────────────────────┐
                          │  AGGREGATION (once)          │
                          │  merge · dedup · prune ·     │
                          │  canonicalize · freeze       │
                          └──────────────┬──────────────┘
                                         │ canonical schema + vocabularies
                                         ▼
   full corpus ─► [segment] ─► [classify] ─► PASS B: schema-constrained fill ─► JSON
                   CRF/SBERT     feature      (grammar/JSON-schema decoding)
```

The per-document path (bottom row) is what runs in production. Pass A and
Aggregation are paid once and amortized.

---

## 2. Pass A — structure induction (run once)

**Purpose:** learn the shape of the data instead of imposing a possibly-sparse
external ontology. This directly addresses the "existing ontologies are too sparse
and full of dead intermediate layers" problem — an induced schema only contains
structure your corpus actually motivates.

**Inputs:** a representative *sample* of descriptions (not the whole corpus —
a few hundred to a few thousand spans is usually enough to saturate the feature
inventory for a given taxon group).

**Method:**

1. Segment each sampled description into feature-bearing spans (reuse the Pass B
   segmenter, below).
2. For each span, ask the LLM to emit candidate `(feature, subfeature, value_type,
   example_value)` tuples — *ungrounded*, no fixed schema yet. Let it over-generate.
3. Collect all tuples across the sample.

**Output:** a raw, noisy, redundant list of candidate structural paths plus
example values. This is deliberately permissive; cleanup happens in aggregation.

**Prompt note:** ask for the *type* of each value (categorical / ordinal /
measurement-with-unit / count / range / free-text) alongside the value. Value
typing is what later lets you attach units and ranges cleanly, and it is fully
taxon-general.

---

## 3. Aggregation — freeze the canonical schema (run once)

This is where the two structural complaints get fixed mechanically.

1. **Canonicalize terms.** Cluster surface variants ("pileus" / "cap" / "pileo")
   to a single canonical key using your multilingual SBERT embeddings. The
   cross-lingual model is doing real work here: Latin/vernacular/translation
   variants collapse to one node.
2. **Merge paths.** Identical or near-identical `feature → subfeature` paths
   become one. Frequency counts come for free and tell you what's core vs. rare.
3. **Prune dead layers.** Remove any intermediate node that has exactly one child
   and carries no values of its own — this is the "uninteresting intermediate
   layer" killer. A node earns its place only if it branches or holds values.
4. **Type and constrain values.** For categorical features, the observed value set
   *becomes* a controlled vocabulary (an enum). For measurements, record unit and
   plausible range. These become hard constraints in Pass B.
5. **Freeze.** Emit a single canonical schema (JSON Schema) + a set of controlled
   vocabularies / gazetteers. This is the contract Pass B fills.

**Taxon portability:** the *procedure* in this section is identical for any group.
Only the resulting schema file and vocabularies differ between fungi, plants, etc.
Keep schemas in a registry keyed by clade so the same code loads the right one.

---

## 4. Pass B — per-document extraction (production)

Three sub-steps. Only the last touches the LLM, and only on short spans.

### 4.1 Segment
Split the prose into feature-bearing spans. Your existing **line-level SBERT + CRF**
machinery already does most of this. The CRF's sequential context handles
interrupting blocks (page headers, figure captions) you've already given
first-class labels to. No LLM needed.

### 4.2 Classify
Map each span to a **feature type** from the frozen schema. Options, cheapest first:
- gazetteer / regex anchors for reliable terms (MycoBank-style ID anchoring
  generalizes to any registry: IPNI for plants, ZooBank for animals);
- SBERT nearest-centroid against the canonical feature keys;
- the CRF head if you fold feature type into the label set.

This step decides *which part of the schema* a span feeds, so the LLM in 4.3 is
only ever asked to fill one known feature on a short, relevant span.

### 4.3 Schema-constrained fill
For each classified span, call the LLM with **constrained decoding** so it
*cannot* emit off-schema structure. The model fills values into a fixed shape
rather than inventing one.

Tooling options (all enforce a grammar/schema at decode time):
- **GBNF grammars** in llama.cpp;
- **Outlines** (regex/JSON-schema constrained);
- **XGrammar** (fast structured decoding);
- native **JSON-schema / structured-output** modes where available.

Because decoding is constrained:
- variance collapses (no malformed or off-schema JSON to re-parse);
- categorical values are restricted to the controlled vocabulary;
- measurements are forced into the typed value shape;
- it is typically **faster**, because no tokens are wasted on invalid structure.

---

## 5. How this addresses the original problems

| Problem | Where it's solved |
|---|---|
| Slow Mistral run (weeks) | §4: LLM only runs on short spans, after CRF/regex do the heavy lifting; §6: batched inference |
| Variable feature/subfeature/value quality | §3 freeze + §4.3 constrained decoding remove degrees of freedom |
| Ontologies too sparse | §2–3: induce structure from the corpus itself |
| Dead intermediate layers | §3 step 3: mechanical pruning of single-child, value-less nodes |
| Works only for fungi | §0/§3: machinery is generic; only schema + gazetteers are per-taxon |

---

## 6. Speed levers (apply before buying GPUs)

1. **Batched / continuous-batching inference** (vLLM, TGI) instead of sequential
   calls — often 5–20× throughput on the *same* hardware.
2. **Constrained decoding** (§4.3) — fewer wasted tokens, no re-generation of
   malformed output.
3. **Shrink the LLM's job** — segmentation/classification done by CRF/regex means
   the LLM only does value extraction on short spans.
4. **Cache** Pass A and aggregation results; they're paid once.
5. *Then*, if still compute-bound, add GPU parallelism — last lever, not first.

---

## 7. Taxon-portability checklist

When moving to a new group (plants, insects, lichens):
- [ ] swap the **gazetteers / registry anchors** (IPNI, ZooBank, Index Fungorum…);
- [ ] re-run **Pass A** on a sample of that group's literature;
- [ ] re-run **Aggregation** to freeze a new canonical schema;
- [ ] keep **all pipeline code** (segment, classify, constrained-fill) unchanged;
- [ ] register the new schema in the clade-keyed schema registry.

The promise: new taxon group = new *data files*, not new code.

---

## 8. Suggested build order

1. Stand up the segmenter from existing SBERT+CRF as a clean stage interface.
2. Hand-write a small frozen schema for one well-understood feature (e.g. pileus)
   to validate the §4.3 constrained-fill loop end to end.
3. Add batched inference (§6.1) and measure throughput delta.
4. Build Pass A induction + aggregation to grow the schema from the corpus.
5. Generalize gazetteers/registry anchoring to make the taxon swap concrete with a
   second clade.

---

## ⏭️ Reminder: second project — automatic ontology building

**Come back to the ontology-learning track as a separate effort.** It overlaps
with Pass A here but is its own project: building a richer, reusable ontology from
a body of literature (Hearst patterns, distributional subsumption, hyperbolic /
Poincaré embeddings for clean tree recovery, LLM-based induction à la
OntoGPT/SPIRES). The pipeline above deliberately sidesteps the need for a perfect
ontology by *inducing-then-freezing* a task schema — but the standalone ontology
would feed back into this pipeline as a stronger prior and is worth pursuing once
the extraction pipeline is stable.

---

## 9. XP Planning Game estimate

**Calibration unit:** 1 point = the work to build a PDF scraper for a new
journal (subclass `Ingestor`, map the journal's URL structure, handle its
PDF flow, tests; ~1–2 days of focused work for someone familiar with the
codebase).

**Total: ~19 points** (with ±5 variance — most of it concentrated in Pass A).

| Step (from §8) | Item | Pts | Notes |
|---|---|---|---|
| 1 | Segmenter stage interface from existing SBERT+CRF | 1 | Existing code, refactor + thin wrapper. |
| 2 | Hand-written schema + constrained-fill loop (one feature, end-to-end) | 3 | `outlines` is already a dep; new infra but well-trodden. |
| 3 | Batched inference (vLLM / TGI) + throughput measurement | 2 | Standard work; lots of recipes available. |
| 4 | Pass A induction + Aggregation (LLM tuple-gen, embedding cluster, path merge, dead-layer prune, type detect, schema emit) | **8** | **The high-variance bit.** Could be 4 if prompts converge quickly; could be 15 if cluster quality, distance thresholds, or type detection misbehave and you iterate on prompts + dedup rules for a few weeks. |
| 5 | Second-clade generalization (swap gazetteers, re-run, validate abstraction) | 3 | First clade is the hard one; second mostly proves the seams. |
| —  | Integration with existing `treatments_to_json` + tests (per CLAUDE.md) + docs | 2 | Stable but real. |
|    | **Total** | **19** | |

**What pushes it bigger than its line count suggests:**
- §4 (Pass A) is research-shaped work.  "Discover the candidate structure" is
  a phrase that's easy to write and very hard to estimate; convergence on a
  stable schema after aggregation is the gate.
- §3 step 1 ("canonicalize terms" with SBERT clustering) is one line in the
  doc but several days of distance-threshold + cluster-validation tuning in
  practice.
- Schema-driven pipelines tend to have a long tail of "this corner case broke
  decoding" — the real cost is the third and fourth feature you add, not the
  first.

**What makes it *not* enormous:**
- Most of the Pass B machinery already exists (CRF segmenter, SBERT
  embeddings, gazetteer pattern from the gnfinder integration, `outlines`
  in deps).
- The taxon-portability story is mostly about *not* hard-coding clade
  specifics — easier to validate than to build.

**Cheapest variance-reducer:** §8 step 2 in isolation.  Hand-write a schema
for one well-understood feature (pileus) and get a single constrained-fill
call working end-to-end.  1–2 days of work that de-risks the next 17
points by proving the constrained-decoding loop is real.
