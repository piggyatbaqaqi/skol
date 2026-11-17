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
occur as a child of a more common top-level key: `hypha` under
`mycelium`, `conidiomata` under `asexual morph`, `ascomata` under
`sexual morph`, `colonies` and `colony` under `culture
characteristic`.  This count is an **upper bound**: it matches by key
name alone, and the same name under different parents can be
different structures (see Proposal 3, the `stipe` case).  Some of
those parents (`species`, `morphology`, `description`) are container
layers; `asexual morph` and `sexual morph` are not (Proposal 3).

**Missed merges.**  `colony` (1,265) vs `colonies` (1,293) — `-ies` is
a spared ending; `pileus color` (12) vs `pileus colour` (2); `cap`
(114) vs `pileus` (3,097), a true synonym pair no string rule finds.

## Proposal 1 — per-node document counts (accepted 2026-11-16)

Trello: https://trello.com/c/sa8yg1K4 (prose→JSON list).  Estimate:
3 units (user).

* **Main cost: where the counts live.**  Storing counts inside each
  node changes the Redis tree format.  That format is walked by
  `build_vocab_tree`, the Django tree view and its `_limit_depth`, and
  `django/search/tests/test_vocab_menus.py`.  A parallel path→count
  map leaves the format alone and would bring the cost toward 2.
* **Side benefit:** the Django tree view parses the whole tree JSON
  from Redis on every request.  At minimum support 2 that JSON is a
  tenth of today's size.  Not checked: whether `VocabTreeChildrenView`
  parses the whole tree the same way.

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

### Split into three cards (2026-11-16)

| Card | Scope | Estimate |
|---|---|---|
| **P2a** https://trello.com/c/AZeJMCR3 | The mechanism: canonicalizer; language rules moved from `normalize_key` (adding the `-ies` and `colour` folds); map loader with a seed entry; `json_canonical` + `canonical_edits`; pinning the version in the record; the `bin/` step and its cron; `build_vocab_tree` reads `json_canonical`; the production backfill, run by the user | 5 (user) |
| **P2b** https://trello.com/c/RdV2gXqD | Curation: candidate sweep (spelling, SBERT, a rare key whose children mostly match a common key's), human review, accepted pairs with evidence, the non-synonyms record, and hyponyms noted as R5a moves | 2 for the first pass, then ongoing (Claude's suggestion) |
| **P2c** https://trello.com/c/ifsDJ1U3 | The JSON classifier reads `json_canonical`; retrain and compare; the model records its map version; settle `redis_keys.classifier_model` | 1–2 (Claude's suggestion) |

Order: P2a, then P2b, then P2c.  P2a depends on the `data/` move
card in Infrastruction and Productionization (https://trello.com/c/EMbuJc0r).
That card is unestimated; Claude's suggestion is 2 units, with the
risk in the postinst and `skol-switch-version` symlink handling.

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
  `canonical_edits` (renamed from `canonical_renames`; it records the
  structural edits of Proposal 3 as well), one entry per edit:

  ```json
  "canonical_map_version": "<git version of the map>",
  "canonical_edits": [
    {"path": ["cap", "color"], "from": "cap", "to": "pileus",
     "rule": "rename"},
    {"path": ["conidia"], "from": "conidia", "to": "conidium",
     "rule": "latin_plural"}
  ]
  ```

  The edits list is the per-document record: look at any treatment
  and see exactly which of its keys were changed, and from what.
* **The map is a git-tracked file in `data/`**, e.g.
  `data/vocab_key_synonyms.json`, each entry carrying its evidence and
  date:

  ```json
  {"from": "cap", "to": "pileus", "rule": "synonym",
   "support": {"from": 114, "to": 3097}, "added": "2026-11-16"}
  ```

  Rule-based folds (plurals, case, separators — today's
  `normalize_key`) are recorded by rule name, not enumerated.  Rejected
  pairs go in a non-synonyms record beside it in `data/`, so the record
  of what was *not* merged is as visible as what was.  Structural
  rulings (which layers drop, which keys move) are in
  [`docs/vocab_tree_layer_rulings.md`](../vocab_tree_layer_rulings.md).
* **Undo is cheap**: remove the map entry and re-run canonicalization.
  No model re-run, because the raw JSON is untouched.
* **Consumers switch to `json_canonical`**: `build_vocab_tree` and the
  JSON classifier.  The menu can show the record too:
  `pileus (3,097; also: cap)`.
* **Where it runs**: a separate step after `treatments_to_json`, not
  inside the menu build, so every consumer reads the same result.
  Needs a `bin/` program, a cron entry, and a `docs/couchdbs.md` field
  note (no new database).

### Where the map lives: source code or an experiment-specific table

Raised 2026-11-16.  The map holds three kinds of knowledge, and they
belong in different places:

| Kind | Example | Varies by experiment? |
|---|---|---|
| Language rules | plurals, `colour`/`color`, case, separators | No |
| Domain synonyms | `cap` = `pileus` | No: it is a fact about mycological usage |
| Evidence and pinning | support counts; which map version an experiment's `json_canonical` used | Yes |

**Source code (or package data in git).**
* *For:* it is reviewed as a diff, and the same `cap` = `pileus`
  decision is made once for every experiment.
* *For:* it is testable: a test can assert that no entry merges a pair
  listed in `docs/feature_label_non_synonyms.md`.
* *For:* it is the "easily seen record": `git log -p` on the map is
  the history of every merge and its reason.
* *Against:* changing the map needs a release (wheel + deb).  Synonym
  changes should be rare and reviewed, so this is acceptable.

**Experiment-specific table** (e.g. a CouchDB database named in the
record's `databases`).
* *For:* it can change without a deploy.
* *For:* an experiment's extraction model may invent its own variant
  keys, which another experiment never sees.
* *Against:* it duplicates facts that are not experiment-specific, so
  two experiments drift apart on whether `cap` is `pileus`.
* *Against:* edits are not diffs; nobody reviews them, and nothing
  shows what changed.  This defeats the visible record.

**Decision (2026-11-16): language rules in code, domain synonyms as
git-tracked data, and the experiment record pins the git version.**

* **Language rules** are today's `normalize_key`.  They stay code,
  with tests.
* **The synonym map and its non-synonyms record live in `data/`**
  (the user's call, 2026-11-16), not in `docs/` and not as package
  data.  The draft path `docs/vocab_key_synonyms.json` is withdrawn.
* **The experiment record names the git version of the map** it
  applies, and each document's `canonical_map_version` must match it.
  This follows `docs/plans/experiment-record-canonical.md`: the record
  is the only description of what production serves.
  - Open detail: production has no git checkout.  So the version is
    either stamped at build time (the commit that last changed the
    map), or is the file's git blob hash.  Production can compute a
    blob hash from the installed file alone, and it matches
    `git hash-object`.
* **Overlays:** if an experiment really needs entries of its own, add
  a git-tracked overlay file in `data/`, named by the record.  Do not
  use a database table.
* **Support counts** are evidence from one experiment.  A map entry
  names the experiment its counts came from, or the counts live in a
  generated report instead of the map.  The `support` field in the
  draft entry above needs this.

**Packaging requirement.**
* `data/` reaches neither the wheel nor the deb, except
  `data/ontologies/*.obo`, which `build-deb.sh` copies to
  `/usr/share/skol/ontologies/`.  The synonym map and the
  non-synonyms record must be shipped the same way, and the code must
  find them where they are installed.
* A test in the style of `tests/wheel_modules_test.py` should check
  that every `data/` file the shipped code opens is actually shipped.
  A missing map on production is a packaging error.
* An existing case of the same hazard:
  `treatments_to_structured/feature_label_rules.py` loads
  `docs/feature_label_canonicalization.json` relative to the source
  tree, a path that doesn't exist once the package is installed.
  - *Correction:* an earlier version of this note also flagged
    `bin/treatments_to_json.py` for looking in `/opt/skol/data/ontologies`
    while the deb stages `/usr/share/skol/ontologies`.  That is not a
    bug: `debian/postinst.template` copies the staged files into
    `/opt/skol/data/ontologies`.
* **The work is Trello card "Move the feature-label synonym map and
  non-synonyms record to data/, and ship them"**
  (https://trello.com/c/EMbuJc0r), in Infrastruction and
  Productionization.  It also settles the install location.  The
  recommendation there is `/opt/skol/share/`, a symlink to
  `/opt/skol/versions/<ver>/share/` switched with `/opt/skol/bin`, so a
  version rollback rolls back the map that the experiment record pins.
  `/opt/skol/data/` is ruled out: it is the writable `SKOL_DATA`
  directory.

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
  `canonical_edits` either way.
* Do the phase-1 `docs/feature_label_canonicalization.json` and
  `docs/feature_label_non_synonyms.md` move to `data/` too?  The
  rationale for putting the new pair in `data/` applies to them, but
  moving them touches the code that loads the JSON (above) and about
  ten references (code, tests, fixtures, docs).
* Is the phase-1 map the same map as the new one, or a sibling?  Its labels are title-case annotation labels; the
  JSON keys are lowercase model output.  Probably a sibling sharing the
  2026-08-17 rule (prefer the more idiomatic form, counted in canonical
  space).
* What format is the non-synonyms record in `data/`?  A JSON list of
  `{a, b, why}` would let a test assert that no map entry merges a
  rejected pair.  The phase-1 record is Markdown, and its test copies
  the pairs out by hand.
* How candidates are found: the `-ies` fold and spelling variants
  (`colour`) by rule; true synonyms by an SBERT similarity sweep
  reviewed by a human, never applied automatically.

## Proposal 3 — structural edits are separate decisions

Revised 2026-11-16.  The first draft lumped "re-parent keys" with
"drop container layers".  They are different operations decided on
different evidence: a drop is a test on one layer, while a move is a
part-of claim about one (parent, key) pair.

**The operations, the tests and the per-layer rulings are kept in
[`docs/vocab_tree_layer_rulings.md`](../vocab_tree_layer_rulings.md).**
That is the standing record; this section is the plan.  The ruling
tables, with evidence, live there and not here.

Rulings so far (2026-11-16):
* **Drop:** `species`.
* **Drop, pending the collision test:** `morphology` and
  `description`.
* **Keep:** `asexual morph` and `sexual morph`, with `key_to_value`
  on `sexual morph`'s answer children (`undetermined`, `not
  observed`, …).
* **Keep:** `micromorphology`.  All 76 of its `stipe` children are
  conidiophore stipes, so dropping it would file them with mushroom
  stipes.
* **Undecided:** `macromorphology` (17 documents).

Sequencing:
* **Drop layers first, then measure again.**  Dropping `species` and
  `morphology` makes 1,233 `conidium` placements coincide with the
  two main ones, which shrinks the real placement question.
* **Dropping layers and key to value can go ahead of R5a.**  Moves and
  inserted layers need a schema: R5a
  (`docs/schema_constrained_pipeline.md` §3, "merge paths").  Doing
  them as a menu heuristic would leave the menu and the extraction
  schema disagreeing about structure.

### Recording

Every operation is recorded the same way as a rename.  The
per-document list becomes `canonical_edits`, renamed from
`canonical_renames`, with rules `rename`, `drop_layer`, `move`,
`insert_layer` and `key_to_value`.  Each entry gives the old path and
the new one, so any edit can be traced and reverted from the record.

## Disposition log

* 2026-11-16: doc created with the measurements, Proposals 1–3.
  Proposal 1 accepted; Proposal 2 direction (fold in the JSON with a
  record) set by the user; design above is a draft.
* 2026-11-16: Proposal 2 gains "Where the map lives": code for
  language rules, git-tracked package data for synonyms, and the
  experiment record pins the version.  This is a recommendation and
  is not yet decided.  The draft `docs/` path for the map is withdrawn
  (`docs/` is not in the wheel).  Proposal 3 is rewritten as separate
  operations, with measurements of the layers.  Two first-draft claims
  are corrected: `asexual morph`/`sexual morph` are not containers,
  and `stipe` under `pileus` is not a valid move (homonyms).  Key to
  value is added as an operation.
* 2026-11-16: The user accepted the map recommendation and set the
  location: the synonym map and its non-synonyms record go in `data/`,
  and the experiment record names the map's git version.  The
  packaging requirement and two existing instances of the hazard are
  noted.  Proposal 3's operations and layer rulings move to
  `docs/vocab_tree_layer_rulings.md`.  New ruling there:
  `micromorphology` is kept, because it fails a new homonym-collision
  test (76/76 of its `stipe` documents are conidiophore stipes).
  `canonical_renames` becomes `canonical_edits` throughout.
* 2026-11-16: The `treatments_to_json` ontology-path hazard is
  retracted (postinst copies the files).  The phase-1 move is now a
  Trello card, with the install-location recommendation.  R5a has a
  comment summarizing this plan.
* 2026-11-16: Trello cards created for Proposal 1 (sa8yg1K4) and
  Proposal 2 (AZeJMCR3).
* 2026-11-16: Trello list renamed to "prose→JSON" by the user.  The
  user estimated P1 at 3 units and P2 at 5.  P2 is now three cards:
  P2a (the mechanism, keeping the 5-unit estimate), P2b (curation)
  and P2c (the classifier switch).  P1 gains the note on where counts
  are stored and the side benefit for the Django tree view.
