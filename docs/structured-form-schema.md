# Structured-form schema: design decisions

*Started 2026-08-24, from a working session on one Ascospores segment.
Settles the shape of the JSON that phase 2 produces, and — as much as
the shape itself — records **why**, so the reasoning survives.*

## Context

The pipeline converts a treatment into nested
`feature → subfeature → value` JSON in **two phases**:

1. **Span labelling** — Claude API assigns a `feature_label` to each
   clause. Measured 2026-08-23 on the only random sample: **precision
   100 %, recall 99 %**.
2. **Structuring** — an SLM turns a labelled span into nested JSON.

The split exists because Mistral did poorly at breaking down *entire
descriptions*. Phase 1 removes that problem by handing it one labelled
clause at a time.

**The split holds**, for a reason beyond the original one: phase 1 has
an evaluation harness (precision/recall against hand review) and phase 2
has none. Collapsing them would discard the only measurement in the
pipeline. See `docs/plans/production-v5-execution.md` M6.

## The worked example

Source (`taxon_cdcba8db`, *Hypoderma subiculatum*, span T3):

> `Ascospores 22–27(–30) × 3–4 µm, cylindrical, straight, tapering
> slightly towards base, ends rounded, aseptate, hyaline, surrounded by
> a 1 µm thick gelatinous sheath.`

Target:

```json
[
  {
    "feature": "Ascospores",
    "provenance": {"field": "description", "start": 1126, "end": 1291},
    "dimensions": ["22–27(–30) × 3–4 µm"],
    "shape":      ["cylindrical"],
    "curvature":  ["straight"],
    "taper":      ["tapering slightly towards base"],
    "ends":       ["rounded"],
    "septation":  ["aseptate"],
    "colour":     ["hyaline"],
    "sheath": {
      "thickness": ["1 µm"],
      "texture":   ["gelatinous"]
    }
  }
]
```

## Decisions

### 1. Values are verbatim source phrases, in lists

`"taper": ["tapering slightly towards base"]`, **not**
`{"degree": "slight", "toward": "base"}`.

A normalised parse invents vocabulary — *slightly* becomes *slight* —
and commits to an interpretation. **Verbatim is checkable against the
source; a parse is not, and its errors are silent.** For output that
will train a model, that difference outweighs elegance.

Lists everywhere, even for single values, so the arity never changes.

### 2. A list of blocks, not an object keyed by feature name

`[{"feature": "Ascospores", …}]`, **not** `{"Ascospores": {…}}`.

The keyed form implies merging repeated features into one node with
more branches. That is the right **final** view and the wrong
**extraction** form.

**Measured 2026-08-24** over 241 repeated-label groups in
`features_candidate`:

| | | |
|---|---:|---|
| no measurements | 100 | 41 % — merging plausible |
| identical measurements | 48 | 20 % — genuine restatement, safe |
| **different measurements** | **93** | **39 % — merging corrupts** |

That is **9.6 % of all feature blocks** in a group that would merge
unsafely. Spot-checked, the conflicts are real, not different aspects
of one structure:

* `taxon_2a9d07e6` — `Conidia … 0–5-transversely euseptate` beside
  `Conidia solitary, aseptate`. Merged, that yields
  `"septation": ["0–5-transversely euseptate", "aseptate"]` — a
  contradiction stated as a fact about one organism.
* `taxon_22346900a8` — two spore descriptions with different
  dimensions, from different taxa.

Four further reasons, any one of which is sufficient:

* **Merging is irreversible; not-merging is not.** Merge downstream
  when you know it is safe. You can never un-merge.
* **The precondition is not checkable from the merged output.**
  "Valid if the treatment is really one name" has to be established
  *before* the merge — and if you are checking beforehand, keeping the
  blocks costs nothing.
* **It fights the label-schema fix.** The vocabulary already holds
  `Asci in culture MEA`, `Asci in culture V8` and `Asci protologue` —
  three observations of one feature in **one taxon**, differing by
  medium. Merging discards which medium produced which, which is
  exactly the base+context information that fix exists to preserve.
* **Provenance needs a home.** Every block records where it came from,
  for the same reason `*_spans` do. A merged node cannot carry two
  source locations without a list — at which point the list of blocks
  has been reinvented, minus the ability to say which value came from
  where.

**Merging is a downstream view**, and as a function over this list it
can *refuse* when blocks conflict — behaviour the baked-in form cannot
offer.

### 3. Flat and consistent, lowercase keys

`shape`, `curvature`, `taper`, `ends` as siblings rather than under a
`geometry` layer. Shallow hierarchies are easier for humans to hold,
and **consistency matters more than depth**. A `geometry` grouping is a
view over flat keys and can be added later without migrating anything.

Keys are lowercase throughout. `feature` carries the canonical label
from the vocabulary, so it keeps that vocabulary's capitalisation.

### 4. The model recognises measurements; a regex parses them

`"dimensions": ["22–27(–30) × 3–4 µm"]` — the model's job is to
**recognise** the clause as a measurement and put it in the right slot.
Splitting it is deterministic work for a regex.

**Do not spend model capability on something a regex does better.** A
language model asked to do arithmetic fails quietly.

The parse runs as a separate deterministic pass writing a sibling key,
leaving the model's output untouched:

```json
"dimensions": ["22–27(–30) × 3–4 µm"],
"dimensions_parsed": [{
  "length": {"min": 22, "typical_max": 27, "extreme_max": 30, "unit": "µm"},
  "width":  {"min": 3,  "max": 4, "unit": "µm"}
}]
```

Keeping them separate means the parse is re-runnable and independently
testable without re-invoking the model, and the model's output stays a
stable artifact to evaluate against.

**The notation needs a three-point range.** `22–27(–30)` means "usually
to 27, occasionally to 30". Two-point `min`/`max` cannot hold it, and it
is common in mycological description — hence `typical_max` and
`extreme_max`.

## Open

* **The slot vocabulary is not fixed.** 322 distinct feature labels,
  **54 % singletons**. Before either phase can target a schema, the
  slots have to be enumerated — and Claude is already indicating them:
  **21 % of labels are `modifier + an existing label`** (`excipulum` ←
  ectal/medullary/outer/proper; `hyphae` ← generative/skeletal/fertile),
  and span nesting expresses part-of directly (`Subiculum` inside
  `Ascomata`, memo §0.1). Harvest those for **schema induction** before
  generating any training data.

  **The missing slot vocabulary is actively corrupting the feature
  vocabulary**, which raises the priority. Measured 2026-08-25 on
  `taxon_fa7f4de6`: the reviewer added `Squamules` for *"The majority of
  squamules are sterile."* — a **fertility property** of a structure
  already described under `Thallus`, not a new organ. The annotator is
  offered `feature_label` and nothing else, so a property with no
  structural home has to be expressed as an organ-shaped label. It wants
  to be `{"feature": "Thallus", "fertility": [...]}`.

  Two consequences:

  * **The singleton rate is not a clean vocabulary measurement.** The
    construction occurs in 13 of 42 096 descriptions (0.03 %), so
    `Squamules` lands permanently in the 54 % singleton tail while
    describing nothing new. Partition the 322 labels into organ-names
    and property-names **before** reading saturation off the curve.
  * **It is the same defect as `Asci in culture MEA`.** There the
    qualifier is a medium, here it is a property; both are welded into
    the label string because the schema has no slot. Fixing base+context
    without fixing this fixes half a problem. Memo §12.1.
* **Whether the SLM needs fine-tuning at all.** The "Mistral did poorly"
  finding was measured on *entire descriptions*, which is the problem
  phase 1 removes. Re-run it on labelled spans before assuming a
  fine-tune is needed.
* **No evaluation harness for phase 2.** Harder to build than the
  extraction, and nothing works without it.

## Watch for

**Newlines must become spaces, not disappear.** The description field
holds `tapering slightly\ntowards base`; a prompt builder that joins
lines naïvely produces `slightlytowards`, which reaches the model as an
unrecognised token and becomes a silent wrong value rather than an
error. Verified 2026-08-24: the source `.ann` has `slightly \n` and the
trailing space is dropped on the way to the `description` field, so the
newline is the only separator left.

## Cross-references

* [plans/production-v5-execution.md](plans/production-v5-execution.md) — M6
* [data_quality_production_v4_model.md](data_quality_production_v4_model.md)
  — §0.1 (nested annotations), D15 (repeated features)
* [feature_label_non_synonyms.md](feature_label_non_synonyms.md)
  — what may and may not be collapsed
