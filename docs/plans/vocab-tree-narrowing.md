# Narrowing the Vocabulary tree

*Planning doc.  Started 2026-11-16.  A living document: revise it as
the proposals are argued out.  Related to Trello R5a ("freeze the
canonical schema", `docs/schema_constrained_pipeline.md` §3), which
may be revised to match.*

## How the Vocabulary selector works today

1. **Build.**  `bin/build_vocab_tree.py` reads `json_annotated` from
   every document in the experiment's `treatments_structured`
   database.  JSON keys at depth *n* become menu level *n*; leaf
   values become the last menu.  The tree is stored in Redis under
   the experiment record's `redis_keys.menus`.
2. **Folding.**  `normalize_key` merges case, wrapper characters,
   `_`/`-`/space, simple English plurals (sparing words ending in
   `ss`, `us`, `is`, `ies`, `ous`) and eight Latin plural pairs.  Each
   group is displayed under its most frequent spelling.  Values are
   lowercased, and yes/true/present → `present`, no/false/absent →
   `absent`.  All of this happens only in the menu build.
3. **Pruning.**  `prune_top_level_singletons` drops top-level keys
   seen in at most one document.  Nothing deeper is pruned
   (`docs/api-vocab-tree.md`, "Singleton pruning").
4. **UI.**  `django/frontend/src/VocabTreeWidget.jsx` shows cascading
   react-select dropdowns fed by `GET /api/vocab-tree/children/`.
   Children are sorted alphabetically; the tree stores no per-node
   counts, so it cannot sort any other way.  "Add to Description"
   inserts the chosen path as text (`pileus color brown; `), and that
   text is what search embeds.

`json_annotated` has a second consumer:
`treatments_classifier/treatments_json_classifier.py` flattens it to
tokens such as `pileus_color=brown` for TF-IDF and a decision tree.

## Measurements (production, 2026-11-16)

Read-only, over the 43,534 documents with `json_annotated` in
`skol_exp_production_v4_1_03_00_treatments_structured`, using the
same key folding as the build.  "Nodes" counts distinct normalized
paths at that depth, leaf values included; a node's support is the
number of documents containing its path.

| Level | Nodes | Support 1 | Support ≤5 | Support ≥1% of docs |
|---|---:|---:|---:|---:|
| 1 | 30,101 | 80.6% | 94.4% | 60 |
| 2 | 206,054 | 82.8% | 95.3% | 138 |
| 3 | 818,840 | 88.9% | 98.2% | 26 |
| 4 | 571,169 | 92.6% | 99.2% | 0 |
| 5 | 260,251 | 95.8% | 99.8% | 0 |

Nodes remaining under a minimum-support threshold applied at **every**
level.  Support never exceeds the parent's, so a threshold removes
whole subtrees and leaves no orphans.

| Min support | L1 | L2 | L3 | L4 | L5 | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (all) | 30,101 | 206,054 | 818,840 | 571,169 | 260,251 | 1,886,415 |
| 2 | 5,837 | 35,491 | 91,172 | 42,188 | 11,001 | 185,689 |
| 3 | 3,369 | 20,230 | 40,214 | 15,043 | 2,655 | 81,511 |
| 5 | 1,967 | 11,470 | 19,180 | 6,126 | 886 | 39,629 |
| 10 | 1,100 | 5,895 | 7,921 | 2,003 | 250 | 17,169 |

Today's top level corresponds to the min-2 row's L1 (5,837 entries);
every deeper level corresponds to the first row.

**Concentration.**  The top 50 top-level keys cover 50% of top-level
occurrences; the top 200 cover 66.5%.  The most common: species
(9,847 documents), conidium, conidiophore, culture characteristic,
conidiogenous cell, ascus, spore, pileus, stipe, basidium, mycelium,
genus.

**The long tail is mostly debris**, not rare morphology: citations
(`phukhamsakda et al. 2020`, `fig. 4`), taxon names used as keys
(`pseudonectria`, `l. leoninus`), OCR damage (`i iyphae`), run-together
words (`stereomicroscopicfeature`), and name-plus-qualifier one-offs
(`thallus (second specimen)`, `conidium (hormonema state)`).

**Migration candidates.**  7,284 of the 30,101 top-level keys also
occur as a child of a more common top-level key: `stipe` and
`lamellae` under `pileus`, `hypha` under `mycelium`, `conidiomata`
under `asexual morph`, `ascomata` under `sexual morph`, `colonies` and
`colony` under `culture characteristic`.  Several of those parents
(`species`, `morphology`, `asexual morph`, `sexual morph`) are
container layers rather than features.

**Missed merges.**  `colony` (1,265) vs `colonies` (1,293) — `-ies` is
a spared ending; `pileus color` (12) vs `pileus colour` (2); `cap`
(114) vs `pileus` (3,097), a true synonym pair no string rule finds.

## Proposal 1 — per-node document counts (accepted 2026-11-16)

Alphabetical order is arbitrary; frequency is what a user browsing a
key wants.

* Store each node's support in the tree.
* `get_children` sorts by descending support, ties alphabetical.
* Apply a minimum support at every level, replacing
  `prune_top_level_singletons`.  The threshold is a CLI flag with an
  environment variable, per the parameter-priority rule; default to be
  decided (2 reproduces today's top level and cuts the whole tree to
  ~10%).
* The children endpoint returns the count; the widget shows it
  (`pileus (3,097)`).  Update `docs/api-reference.md` and
  `docs/api-vocab-tree.md`.
* Menu-only change, no merge risk.  TDD as usual.

## Proposal 2 — fold synonyms in the JSON, with a visible record

The user's position (2026-11-16): folding belongs in the JSON, not
only in the menu, as long as the supplanted synonym stays easy to see.
That holds up:

* **Two consumers already disagree.**  The menu folds `conidia` into
  `conidium`; the JSON classifier sees two tokens.  `cap_color=brown`
  and `pileus_color=brown` are different features to it.  Folding once,
  in the data, makes every consumer agree.
* **The case for menu-only folding is reversibility and fidelity** —
  the JSON should say what the source said, and a bad merge is close to
  invisible once applied (`docs/feature_label_non_synonyms.md`).  Both
  are met by **not overwriting** `json_annotated`:

### Design

* **`json_annotated` stays as extracted.**  It is the faithful record
  of what the model read in the prose.
* **New derived field `json_canonical`** in the same document, plus
  `canonical_renames`, one entry per rewritten key:

  ```json
  "canonical_map_version": "<hash of the map file>",
  "canonical_renames": [
    {"path": ["cap", "color"], "from": "cap", "to": "pileus",
     "rule": "synonym"},
    {"path": ["conidia"], "from": "conidia", "to": "conidium",
     "rule": "latin_plural"}
  ]
  ```

  The renames list is the per-document record: look at any treatment
  and see exactly which of its keys were changed, and from what.
* **The map is a git-tracked file**, e.g. `docs/vocab_key_synonyms.json`,
  each entry carrying its evidence and date:

  ```json
  {"from": "cap", "to": "pileus", "rule": "synonym",
   "support": {"from": 114, "to": 3097}, "added": "2026-11-16"}
  ```

  Rule-based folds (plurals, case, separators — today's
  `normalize_key`) are recorded by rule name, not enumerated.  Rejected
  pairs go in `docs/feature_label_non_synonyms.md` or a sibling, so the
  record of what was *not* merged is as visible as what was.
* **Undo is cheap**: remove the map entry and re-run canonicalization.
  No model re-run, because the raw JSON is untouched.
* **Consumers switch to `json_canonical`**: `build_vocab_tree` and the
  JSON classifier.  The menu can show the record too:
  `pileus (3,097; also: cap)`.
* **Where it runs**: a separate step after `treatments_to_json`, not
  inside the menu build, so every consumer reads the same result.
  Needs a `bin/` program, a cron entry, and a `docs/couchdbs.md` field
  note (no new database).

### Synonymy is not hyponymy

The user's hunch was that many keys are "very specific cases of more
common values".  Those are **hyponyms**, and they must not be renamed:
`docs/feature_label_non_synonyms.md` already records `Pycnidia` as a
*kind of* conidioma and the position-specific cystidia as kinds of
cystidium — renaming loses the specific term.  A synonym (`cap`/
`pileus`) gets renamed; a hyponym gets **re-parented** (Proposal 3) or
left alone.

### Open questions

* Collisions: a document with both `cap` and `pileus` — merge children,
  and when leaf values conflict, keep both as a list?  Record it in
  `canonical_renames` either way.
* Is the phase-1 `docs/feature_label_canonicalization.json` the same
  map or a sibling?  Its labels are title-case annotation labels; the
  JSON keys are lowercase model output.  Probably a sibling sharing the
  2026-08-17 rule (prefer the more idiomatic form, counted in canonical
  space).
* How candidates are found: the `-ies` fold and spelling variants
  (`colour`) by rule; true synonyms by an SBERT similarity sweep
  reviewed by a human, never applied automatically.

## Proposal 3 — re-parent keys and flatten container layers

Where a top-level key usually lives under another (`stipe` under
`pileus`?  More likely both under a fruiting-body layer), move it; and
drop container layers (`species`, `morphology`) that only wrap organs.
This is the "merge paths" and "prune dead layers" step of
`docs/schema_constrained_pipeline.md` §3, i.e. R5a.  Doing it as a
separate menu heuristic would leave the menu and the extraction schema
disagreeing about structure, so it waits for R5a — but it should use
the same `json_canonical` + renames mechanism, with a `reparent` rule,
so the record is uniform.

## Disposition log

* 2026-11-16: doc created with the measurements, Proposals 1–3.
  Proposal 1 accepted; Proposal 2 direction (fold in the JSON with a
  record) set by the user; design above is a draft.
