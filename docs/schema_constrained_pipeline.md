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

### 4.4 Feature grouping (open design point — deferred)

The Pass A bootstrap annotator emits a flat list of annotations.
Biologically-related ones often belong to a shared parent feature
that Pass A doesn't represent. Concrete examples from the live
2026-06-28 sample:

- *Amanita magniverrucata* (`taxon_841d5cbed…`) — one Universal
  veil with four annotations: `Universal veil on pileus`,
  `Universal veil on stipe`, `Universal veil (microscopic, on
  pileus)`, `Universal veil (microscopic, on stipe base)`.  Plus
  Partial veil × 2 (macro + microscopic).  Plus Pileus context
  × 2, Stipe context × 2.
- *Aureoboletus* (`taxon_0029f1413f…`) — Cystidia subdivides into
  sibling Cheilocystidia and Pleurocystidia.  Are these peers
  or children of a Cystidia parent?

Three places this grouping *could* happen:

1. **At annotation time** (in the Pass A prompt) — Claude emits
   a `parent_feature_label` per annotation.  Rejected for Phase 1:
   couples the open-ended labelling rule to a hierarchy we don't
   yet know, and the grouping rule is taxon-specific (each
   kingdom would need seed-equivalent hints).
2. **Post-review, pre-Pass-B** — algorithmic grouping by
   canonical parent name on the candidate / golden annotations.
   A standalone step the reviewer audits.
3. **In Pass B itself** — Pass B's schema-constrained fill
   (§4.3) targets hierarchical schemas (e.g., Universal veil →
   {macro: {…}, microscopic: {pileus: {…}, stipe: {…}}}).  The
   classifier (§4.2) routes each Pass A annotation to the right
   slot of the right parent; Pass B accumulates across siblings.

**This document picks (3) when Pass B is designed in detail.**
The cleanest design coupling is at the point where structured
output actually needs the hierarchy — the structured schema
defines the parents, Pass B's classifier maps annotations to
schema slots, and Pass A stays a flat-list producer that doesn't
need to know the eventual hierarchy.

Two prerequisites surface ahead of that work:

- **Label canonicalization is the bigger problem first.**  Even
  within "Universal veil" Claude varied between `microstructure`
  and `(microscopic)` across runs of the same prompt.  Any
  grouping logic has to solve this regardless; a normalization
  step early in Pass B (or as a small post-Pass-A pass) cleans
  up the label vocabulary and is reusable across grouping
  strategies.
- **Optional `parent_feature_label` field** on the candidate
  annotation schema (default null) plants a flag without
  committing to any specific grouping logic.  Future Pass B
  code populates it; current code keeps working unchanged.

Decision deferred until Pass B is being designed in detail —
that's where the biological model and the data model converge
and we'll know what shape the parents should take.

### 4.5 Conjunction labels (open design point — deferred)

Inverse of §4.4: where §4.4 is about MERGING related annotations
into one parent feature, this is about SPLITTING one annotation
into multiple features.  Pass B has to handle both.

Concrete example from the 2026-06-29 live sample (*Calonectria
pentaseptata*, `taxon_2114314b…`):

```
T18  Megaconidia_and_microconidia  Megaconidia and microconidia not seen.
```

The single annotation carries two features sharing the same
value.  Structured form is:

```json
{
  "Megaconidia": ["not seen"],
  "Microconidia": ["not seen"]
}
```

Claude's `"Megaconidia and microconidia"` label is OK as Pass A
output — the conjunction is honest about what's in the source
text — but Pass B has to split it.  Common conjunction patterns
worth handling:

- `"X and Y"` — two features, same value
- `"X or Y"` — uncertainty between two features (rarer; usually
  reflects taxonomist's hedge, not a Pass-A error)
- `"X, Y, and Z"` — three or more features, same value (e.g.,
  `"Pleurocystidia, cheilocystidia, and caulocystidia absent"`)
- `"X but not Y"` — distinct value per feature (`"Lamellae
  adnate but not decurrent"` → Lamellae attachment = adnate; not
  a Microconidia=not-seen-style split)

Where this could happen in the pipeline:

1. **At annotation time** — prompt Claude to split conjunctions
   into separate spans.  Rejected for same reasons as §4.4
   (couples Pass A to Pass B's schema; taxon-specific rules).
2. **In Pass B's classifier (§4.2)** — when routing an
   annotation to a schema slot, detect conjunction patterns in
   the feature_label and emit multiple (slot, value) pairs from
   one annotation.  Lives naturally next to the canonicalization
   step (both transform labels before structured fill).
3. **As a dedicated splitting pass between §4.2 and §4.3** —
   keeps the classifier simple; the splitter has its own rule
   set.  Likely cleanest if conjunction patterns grow beyond
   a handful.

**This document picks (2) or (3) — to be decided when Pass B is
designed in detail.**  The choice depends on how many distinct
conjunction patterns we see across kingdoms.  If just `"X and
Y"`-style same-value-multiple-features dominates, (2) is enough.
If we need different value-merging strategies per pattern (e.g.,
`"X but not Y"` produces opposite values), (3) is cleaner.

Worth noting that the conjunction pattern is BIOLOGICAL
shorthand, not a Pass A failure.  The taxonomist's prose
"Megaconidia and microconidia not seen" packs negative-evidence
about two features into one statement because that's how
mycologists write.  Pass B's job is to expand the shorthand,
not to penalize Claude for preserving it.

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

## 10. Phase 1 plan — bootstrap to editable training data

**Scope contract:** Phase 1 ends the moment a reviewer can open brat
on a synthetic per-treatment document, see Claude-API-generated
feature annotations on the `description` and `diagnosis` prose,
correct them, and have the corrections land in CouchDB as
training-quality ground truth.  No production extractor, no Pass A
induction, no schema freeze — just enough scaffolding to start
editing.

Out of scope for Phase 1 (deliberately): production §4.3
constrained-fill loop, batched inference, Pass A schema induction,
second-clade generalization, the §3 aggregation pipeline.  These
are all later phases against the doc above.

### 10.1 Module layout

New top-level package `treatments_to_structured/` — sibling to
`skol_classifier/` and `treatments_classifier/`.  Named
symmetrically with `bin/treatments_to_json` to make the "v5 starts
where v4 ends" relationship visible in directory listings.

```
treatments_to_structured/
    __init__.py
    complexity.py             # complexity_score(treatment_doc) -> float
    schemas/
        pileus.json           # JSON Schema for one feature (seed)
    brat_render.py            # Treatment → (.txt, .ann) synthesis +
                              # annotation ↔ (field-relative,
                              # source-plaintext) coordinate maps
    storage.py                # CouchDB read/write for annotations
    storage_test.py
    complexity_test.py
    brat_render_test.py

bin/
    select_for_annotation.py  # CLI: complexity-scored sampler
    llm_annotate_features.py  # CLI: Claude-API bootstrap pass
    brat_ingest.py            # CLI: read brat .ann back to CouchDB
    promote_to_golden.py      # CLI: candidate → golden + set marker
```

`treatments_to_structured/` and `bin/bin.*` go in the wheel
include lists (`pyproject.toml` + `setup.py`) at the same time as
the package is created — avoids the repeated "missing package on
prod" bug we hit during the 0.9.0 cycle.  Per `CLAUDE.md`: a
missing package on production is a packaging error.

### 10.2 Synthetic brat document layout

One brat document per Treatment.  Constructed by concatenating the
Treatment's top-level `description` and `diagnosis` string fields
(each may be `null` if the source had no content for that field;
either may be present alone).  Section markers make the field
boundary visible to both the reviewer and the classifier:

```
=== description ===
A small mushroom with a brown pileus 3–5 cm wide.  Lamellae cream
when young, turning ochre.  Stipe 4 cm long, cylindrical, smooth.

=== diagnosis ===
Differs from M. brevicaulis by the absence of a partial veil and
the consistently smaller stipe diameter.
```

If a field is `null` or empty, its section header is omitted —
brat doesn't render a `=== diagnosis ===` block when the
treatment has no diagnosis.  Between-field joiner is
`\n\n=== <field_name> ===\n\n`.

The `description_spans` / `diagnosis_spans` lists on the
Treatment carry source-plaintext char offsets, not text — they're
used only by the annotation writer to populate `source_spans` on
each annotation (the durable backref to source-plaintext
coordinates).  The brat-visible text comes from the top-level
prose fields.

### 10.3 Storage shape

**Per-annotation document** (one CouchDB doc per (treatment_id,
feature) annotation):

```json
{
  "_id": "<treatment_id>:<feature_label>:<offset_hash>",
  "treatment_id": "taxon_xxx",
  "doc_id": "source_plaintext_doc_yyy",
  "field": "description",                // or "diagnosis"
  "start": 142,
  "end": 256,
  "source_spans": [
    {"start": 5842, "end": 5900},
    {"start": 6010, "end": 6056}
  ],
  "source_text": "the brown cap ... edges flaring",
  "feature_label": "Pileus",
  "model": "claude-opus-4-7",
  "created_at": "2026-06-26T..."
}
```

Invariant: source plaintext joined from `source_spans` in order
equals `source_text`, equals `concatenated_field_text[start:end]`.
Validated at every write; mismatch raises rather than silently
storing drift.

**Two databases**, same shape:

- `skol_exp_<name>_02_50_features_candidate` — Claude API output,
  pre-review.  Per-experiment because the prompt and seed schema
  evolve per-experiment.  The ``02_50`` slot sorts between
  ``02_00_treatments_prose`` (CRF extraction output) and
  ``03_00_treatments_structured`` (SLM field-extraction output),
  per the sort-in-pipeline-order convention in
  ``docs/skol-db-naming-cleanup.md``.
- `skol_golden_features` — post-review ground truth.  Global,
  mirrors the `skol_golden_ann_hand` convention (no ``XX_YY``
  prefix — globals don't carry pipeline order).

**Marker on source Treatment** in `skol_exp_<name>_treatments_prose`:
field `in_golden_features: true` flags a treatment whose
annotations have been promoted.  Train-set generators filter
these out; eval scorers operate on the complement.

Per `CLAUDE.md`: add both new DBs to `docs/couchdbs.md` as part of
this work.

### 10.4 Deliverables (and order)

1. **Complexity scorer** — `treatments_to_structured/complexity.py`.
   Pure function over a Treatment doc returning a float.  First-cut
   definition: weighted combo of (a) total prose word count across
   description + diagnosis, (b) feature-keyword hits from a small
   seed gazetteer (pileus, lamellae, stipe, spores, …), (c)
   measurement-pattern count (`\d+(\.\d+)?\s*(mm|cm|µm|µ|nm)`).
   Tunable; comparative semantics — we'll calibrate by inspecting
   scored samples, not by hitting an absolute threshold.

2. **Sample selector CLI** — `bin/select_for_annotation.py`.  Reads
   the experiment's treatments_prose DB, scores each treatment via
   (1), emits N treatment IDs split across complexity bands.
   Example: `bin/select_for_annotation --experiment production_v4
   --n 100 --bands low:25,mid:50,high:25` → 100 IDs printed.

   **How banding actually works, and why not to omit it.**
   `treatments_to_structured.select.select_treatments` sorts the
   scored population ascending and cuts it into **equal-size slices,
   one per band entry** — quantiles, not score thresholds.  The band
   *names* are arbitrary labels; the **order** is what binds each
   quota to a slice, so the first entry always draws from the
   lowest-scoring slice.

   Omitting `--bands` is not "no preference" — `_resolve_band_specs`
   returns a single `('all', n)` band, i.e. one flat random draw
   across the whole scored population.  **Don't do that**: a uniform
   draw spends most of the reviewer's time on treatments with little
   prose to annotate, which is the opposite of what the round is for.
   Weight toward the top instead; the 25/50/25 above is an
   illustrative example, not a calibrated recommendation.

   Two protections are automatic and don't need a flag:
   treatments scoring 0 are filtered out entirely (they lack the
   prose we'd annotate — this is the ~44%-of-corpus cohort §5 of
   `data_quality_production_v4_model.md` attributes to
   `synthetic_nomenclature` false positives), and
   `--exclude-suspected-merges` defaults **on** at
   `--merge-threshold 10`, calibrated 2026-07-01.

   For any round after the first, pass `--exclude-annotated` (needs
   `--experiment`) so the N are N *new* treatments rather than
   repeats, and `--seed` so the selection is reproducible.  Round 4
   (2026-08-14) used:

   ```
   bin/select_for_annotation --experiment production_v4 --n 50 \
       --bands low:5,mid:15,high:30 --exclude-annotated --seed 4
   ```

   81,527 scored → 46,045 non-zero → 62 already-annotated dropped →
   7,632 suspected merges dropped → 38,351 sampled from.  Verified
   median complexity by band: 61 / 203 / 400.

   Caveat: `--dry-run` is inherited from `common_parser()` and shows
   up in `--help`, but `select_for_annotation.py` never reads it —
   the merge filter's `features_status` skip-doc writes happen
   regardless.  There is no preview mode.

3. **Tiny hand-written schema** — `treatments_to_structured/schemas/pileus.json`.
   One feature, JSON Schema, used as a structural prompt
   ingredient.  Pass A induction is later; for Phase 1 the schema
   is hand-authored from operator knowledge.

4. **brat-render module** — `treatments_to_structured/brat_render.py`.
   Functions:
   - `render(treatment) -> (txt: str, span_map: SpanMap)` — builds
     the synthetic .txt and a map for (field-relative ↔ source-
     plaintext) coordinate translation.
   - `annotations_to_brat(annotations, span_map) -> str` — produces
     the `.ann` file body (brat T-entity lines).
   - `parse_brat_ann(ann_text, span_map) -> annotations` —
     round-trip for `brat_ingest.py`.

4.5. **Experiment-schema integration** — cross-cutting between (4)
   and (5).  Required because deliverable (5) writes to a
   per-experiment candidate database, and the experiment doc's
   `databases.*` block is where that location is canonically
   recorded.

   * **`bin/manage_experiment` (cmd_create / cmd_update)**: write
     `databases.features_candidate` per the naming convention
     `skol_exp_<name>_02_50_features_candidate`.  Derived from the
     experiment name; no explicit `--features-candidate-db` flag.
     Matches the existing `treatments_prose` /
     `treatments_structured` pattern.  Tests cover
     create-writes-it and update-rewrites-it.
   * **`bin/replicate_experiment`**: no code change expected —
     `databases_for_experiment` already enumerates any string
     value in the `databases` block, so a new key replicates for
     free.  Add a test that confirms `features_candidate` shows up
     in the replicate list when the source experiment doc carries
     it.
   * **`bin/llm_annotate_features` (deliverable 5)**: resolve the
     target DB by reading `experiment.databases.features_candidate`
     rather than computing the name inline.  Falls back to the
     naming convention if the field is missing on legacy docs
     (operator-actionable warning, not a crash).
   * **Global golden DB**: `skol_golden_features` does NOT belong
     in `experiment.databases` (it's global, not per-experiment —
     matches `skol_golden_ann_hand` precedent).  Declare the name
     as a constant in `treatments_to_structured/storage.py` or
     similar so promotion / eval tooling references it from one
     place.

5. **Claude-API bootstrap annotator** — `bin/llm_annotate_features.py`.
   Reads treatment IDs from stdin (pipe from selector), for each:
   renders the synthetic .txt, sends to Claude with the seed schema,
   parses Claude's JSON response into the annotation schema, validates
   the source-text invariant, writes to candidate DB.  Idempotent on
   `_id` (re-runs overwrite).  Model + created_at recorded mechanically.

6. **brat ingestion CLI** — `bin/brat_ingest.py`.  Reads a brat
   working directory (`.txt` + `.ann` files), translates back to
   annotation docs, writes to candidate DB.  Same shape as (5)'s
   output but originating from a human-edited `.ann`.

7. **Promotion to golden** — `bin/promote_to_golden.py`.  Per-
   treatment: reads from candidate, writes to `skol_golden_features`,
   sets `in_golden_features: true` on the source Treatment doc.  Per
   `manage_experiment` conventions: hard-fail on unknown treatment
   IDs.

8. **Docs** — `docs/couchdbs.md` entries for the two new DBs;
   a brief operator runbook in `docs/treatments_to_structured.md`
   covering the select → annotate → review-in-brat → promote loop.

The first three (complexity, selector, schema) deliver no end-user
value on their own but are the dependencies for (5).  Item (5) is
the first moment a reviewer can see candidate output.  Items (6)
and (7) close the editing loop.

### 10.5 LLM choice for the bootstrap pass

**Claude API**.  Top-quality on the seed data, no GPU infra
needed, per-token cost is bounded by the sample size from (2).
Operationally: Anthropic Python SDK, model selection via env
var (`SKOL_BOOTSTRAP_LLM_MODEL`, default `claude-opus-4-7`),
API key from `~/.skol_env` (`ANTHROPIC_API_KEY`).  Local-Mistral
production §4.3 path stays open; this is bootstrap-only.

### 10.6 First failing test (TDD entry point)

`treatments_to_structured/complexity_test.py`:

```python
import pytest
from treatments_to_structured.complexity import complexity_score


def _make_treatment(description=None, diagnosis=None):
    """Minimal Treatment-doc fixture for complexity_score().

    Matches the production Treatment shape: top-level ``description``
    and ``diagnosis`` are STRING fields holding the prose; either
    may be ``None`` (CouchDB null) for treatments whose prose lives
    elsewhere (notes, biology) — those are out of scope for Phase 1.
    """
    return {
        '_id': 'taxon_test',
        'description': description,
        'diagnosis': diagnosis,
    }


class TestComplexityScore:
    """Comparative semantics: richer prose → higher score.  We
    don't bake in absolute weights here — the calibration step
    (Phase 1 deliverable 1) tunes those by inspection."""

    def test_returns_float(self):
        assert isinstance(
            complexity_score(_make_treatment(description='hi.')), float,
        )

    def test_empty_treatment_scores_zero(self):
        assert complexity_score(_make_treatment()) == 0.0

    def test_richer_description_scores_higher_than_minimal(self):
        minimal = _make_treatment(description='A small fungus.')
        rich = _make_treatment(
            description=(
                'Pileus brown, 3-5 cm wide.  Lamellae cream-colored '
                'when young, ochre at maturity.  Stipe 4 cm long, '
                'cylindrical, smooth.'
            ),
        )
        assert complexity_score(rich) > complexity_score(minimal)

    def test_measurement_density_raises_score(self):
        """Two descriptions of similar word count, one with
        measurements, one without — measurements should win."""
        bland = _make_treatment(
            description=(
                'Pileus brown.  Lamellae cream.  Stipe long and '
                'cylindrical and smooth and pale.'
            ),
        )
        measured = _make_treatment(
            description=(
                'Pileus brown 3 cm.  Lamellae cream 5 mm.  Stipe '
                '4 cm long 8 mm wide cylindrical smooth.'
            ),
        )
        assert complexity_score(measured) > complexity_score(bland)

    def test_diagnosis_contributes_to_score(self):
        """Both fields count; a treatment with both Description AND
        Diagnosis prose scores higher than one with Description alone
        (controlling for description content)."""
        desc_only = _make_treatment(
            description='Pileus brown, 3 cm wide.',
        )
        both = _make_treatment(
            description='Pileus brown, 3 cm wide.',
            diagnosis='Differs from M. brevicaulis by the absent veil.',
        )
        assert complexity_score(both) > complexity_score(desc_only)
```

These define `complexity_score`'s contract behaviorally (comparative,
no absolute thresholds).  Next session: make them pass with the
weighted-combo implementation in `complexity.py`.  When the first
test is green, deliverable (1) is done and (2) can start.

### 10.7 Phase-1 done means

- A reviewer has opened brat on a candidate-DB treatment, edited
  the annotations, and seen the edits land in CouchDB via
  `brat_ingest`.
- At least one Treatment has been promoted to
  `skol_golden_features` and carries `in_golden_features: true`.
- The complexity scorer, sample selector, render/ingest module,
  and Claude annotator all have green tests.

What this leaves for Phase 2+: Pass A schema induction, the
production §4.3 constrained-fill loop, batched inference, second-
clade generalization.  All deferred but unlocked by the editable
training data Phase 1 produces.

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
