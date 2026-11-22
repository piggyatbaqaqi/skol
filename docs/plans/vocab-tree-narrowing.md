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

## Implementation order (approved 2026-11-17)

1. **P1** (https://trello.com/c/sa8yg1K4).  It is independent and
   changes only the menu.  When P2a lands, the counts come from
   `json_canonical` instead, so none of P1 is thrown away.
2. **The `data/` move card** (https://trello.com/c/EMbuJc0r).  P2a
   needs the map shipped and findable on production.
3. **P2a, then P2b, then P2c.**
4. **Proposal 3's `drop_layer` and `key_to_value`**, reusing P2a's
   `canonical_edits`.  Moves and inserted layers wait for R5a.

All of it is TDD: tests approved, committed with strict xfail, then
implemented.

## Proposal 1 — per-node document counts (accepted 2026-11-16)

Trello: https://trello.com/c/sa8yg1K4 (prose→JSON list).  Estimate:
3 units (user).

Decisions (user, 2026-11-17):
* **The counts live in a separate path→count map**, not inside the
  tree nodes, so the tree format is unchanged.
* **The default minimum support is 2, and each user can change it
  from the Settings menu.**  The threshold is therefore applied when
  children are served, using the user's setting, and not only when the
  tree is built.

* **Main cost was where the counts live.**  Storing counts inside each
  node would have changed the Redis tree format.  That format is
  walked by `build_vocab_tree`, the Django tree view and its
  `_limit_depth`, and `django/search/tests/test_vocab_menus.py`.  The
  separate map leaves the format alone.
* **The counts key must not break the versions listing.**
  `VocabTreeVersionsView` scans `skol:ui:menus_*` and calls `GET` on
  every match.  A Redis hash stored under that pattern (e.g.
  `<menus key>:counts`) would make `GET` raise `WRONGTYPE`, which the
  view does not catch, so the whole listing would fail.  The view must
  skip the counts key, or the key must sit outside the pattern.
* **Side benefit:** the Django tree view parses the whole tree JSON
  from Redis on every request.  At minimum support 2 that JSON is a
  tenth of today's size.  Not checked: whether `VocabTreeChildrenView`
  parses the whole tree the same way.

Alphabetical order is arbitrary; frequency is what a user browsing a
key wants.

* Store each node's support in the tree.
* `get_children` sorts by descending support, ties alphabetical.
* Apply a minimum support at every level, replacing
  `prune_top_level_singletons`.  There are two thresholds:
  - **the build floor**, a CLI flag with an environment variable per
    the parameter-priority rule.  It sets what the tree contains.  At 1
    the tree is 1,886,415 nodes; today's tree (top-level singletons
    pruned) is 1,603,885; at 2 it is 185,689.  The Django views parse
    the whole tree on every request, so the floor sets their cost.
  - **the user's setting** (default 2), applied when children are
    served.  A user cannot go below the build floor.
  - **The build floor is 2** (user, 2026-11-17): `--min-support`,
    `$VOCAB_MIN_SUPPORT`, default 2.  Requests get about 9× cheaper;
    the support-1 entries a floor of 1 would add are mostly debris.
* The children endpoint returns the count; the widget shows it
  (`pileus (3,097)`).  Update `docs/api-reference.md` and
  `docs/api-vocab-tree.md`.
* **A tree without its counts hash is an error, not an old format.**
  The user rejected a silent fallback to alphabetical, unfiltered
  children (2026-11-17): after the first rebuild the tree and counts
  are written in one transaction, so a missing hash means something
  broke, and a fallback would hide it.  Instead:
  - the children view returns a 500 naming `{<menus key>}:counts` and
    saying to rebuild, and logs an error;
  - `BuildVocabTreeView` treats the tree as not built (GET
    `not_found`, POST rebuilds), so the widget's existing
    build-and-retry path repairs it.
* **The widget must wait for a build it started.**  `POST
  /api/vocab-tree/build/` answers 202 with status `building` when it
  starts a build, but `VocabTreeWidget.jsx`'s `triggerBuild` accepts
  only `complete` or `exists`, so it shows "Build failed" while the
  rebuild runs.  The tree is repaired by the next page load, but the
  user sees an error first.  This already happens today for a missing
  tree; with 10b it would also happen for a missing counts hash.  The
  fix, part of P1: on `building`, poll `GET /api/vocab-tree/build/`
  every 2 s until it reports `exists`, then reload the top level.  It
  gives up after 3 minutes (user, 2026-11-17) with a notice, not an
  error ("still being rebuilt; try again in a few minutes") and a Retry
  button.  The wait does not block the page, only the widget.  A
  read-only dry run of the production build against the local replica
  took 54 s for 43,534 documents and kept 181,709 nodes at minimum
  support 2.  The estimate above (185,689) did not apply the build's
  `_is_valid_term` filter.
* **Frontend tests.**  `django/frontend/` had no test runner; nobody
  set one up after the first PDF-viewer draft.  Adding Jest is a P1
  prerequisite (https://trello.com/c/RVVWoO9s, prose→JSON), so the
  widget fix and the count labels get real tests.  The Settings input
  lives in `django/templates/index.html`'s inline scripts, which Jest
  cannot import; it stays a hand check until those move into `src/`
  (https://trello.com/c/vpyl9B9L, Infrastruction and
  Productionization).  `PDFViewer.jsx` needs pdf.js worker and canvas
  mocks (https://trello.com/c/YCvPT9f2, Backlog); not on P1's path.
* **Deploy step:** rebuild the vocabulary tree right after deploying
  P1.  The production tree predates the counts hash, so otherwise the
  first visitor's menu triggers the rebuild and waits for it.
* Tests: `bin/build_vocab_tree_test.py` (classes after "Proposal 1")
  and `django/search/tests/test_vocab_counts.py`.  Widget tests follow
  once the Jest card lands (above); the Settings input is checked by
  hand.  Implementing P1 deletes
  `TestPruneTopLevelSingletons` and `TestTopLevelDocCounts` with the
  method they test.
* Menu-only change, no merge risk.  TDD as usual.

## Proposal 2 — fold synonyms in the JSON, with a visible record

### Split into three cards (2026-11-16)

| Card | Scope | Estimate |
|---|---|---|
| **P2a** https://trello.com/c/AZeJMCR3 | The mechanism: canonicalizer; language rules moved from `normalize_key` (adding the `-ies` and `colour` folds); map loader with a seed entry; `json_canonical` + `canonical_edits`; pinning the version in the record; the `bin/` step and its cron; `build_vocab_tree` reads `json_canonical`; the production backfill, run by the user | 5 (user) |
| **P2b** https://trello.com/c/RdV2gXqD | Curation: candidate sweep (spelling, SBERT, a rare key whose children mostly match a common key's), human review, accepted pairs with evidence, the non-synonyms record, and hyponyms noted as R5a moves | 2 for the first pass, then ongoing (user) |
| **P2c** https://trello.com/c/ifsDJ1U3 | The JSON classifier reads `json_canonical`; retrain and compare; the model records its map version; settle `redis_keys.classifier_model` | set by the user from the 1–2 suggestion; the value is in Trello |

Order: P2a, then P2b, then P2c.  P2a depends on the `data/` move
card in Infrastruction and Productionization (https://trello.com/c/EMbuJc0r).
That card is estimated at 2 units (user).  Its risk is in the postinst
and `skol-switch-version` symlink handling.

The user's position (2026-11-16): folding belongs in the JSON, not
only in the menu, as long as the supplanted synonym stays easy to see.
That holds up:

* **Two consumers already disagree.**  The menu folds `conidia` into
  `conidium`; the JSON classifier sees two tokens.  `cap_color=brown`
  and `pileus_color=brown` are different features to it.  Folding once,
  in the data, makes every consumer agree.
* **The case for menu-only folding is reversibility and fidelity** —
  the JSON should say what the source said, and a bad merge is close to
  invisible once applied (`data/feature_label_non_synonyms.md`).  Both
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
  listed in `data/feature_label_non_synonyms.md`.
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
  `treatments_to_structured/feature_label_rules.py` loaded
  `docs/feature_label_canonicalization.json` relative to the source
  tree, a path that doesn't exist once the package is installed.
  *Fixed 2026-11-18 (EMbuJc0r):* the map is in `data/` and found
  through `skol_share` (`docs/shared-data.md`).
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
`data/feature_label_non_synonyms.md` already records `Pycnidia` as a
*kind of* conidioma and the position-specific cystidia as kinds of
cystidium — renaming loses the specific term.  A synonym (`cap`/
`pileus`) gets renamed; a hyponym gets **re-parented** (Proposal 3) or
left alone.

### Decisions (user, 2026-11-17)

These were open questions; each was settled as recommended.
* **Collisions:** when a document has both `cap` and `pileus`, merge
  their children.  When leaf values conflict, keep both as a list.
  Record the merge in `canonical_edits` either way.
* **Map version:** the git blob hash of the installed map file
  (`git hash-object`), which production can compute from the file
  alone.
* **Non-synonyms record format:** a JSON list of `{a, b, why}`, so a
  test can check that no map entry merges a rejected pair.
* **Phase-1 map:** `data/feature_label_canonicalization.json` stays a
  sibling of the new map, not the same file.  The two share the
  2026-08-17 rule: prefer the more idiomatic form, counted in canonical
  space.

### Decisions for P2a (user, 2026-11-18)

Measured on 44,211 local documents: the idiomatic key forms are
capitalized and usually plural (`Conidia` 5,716 against `Conidium`
339, `conidia` 265, `conidium` 18; `Asci` 2,927; `Culture
characteristics` 1,963 against `Culture Characteristics` 1,782).
`normalize_key`'s grouping keys (`conidium`, `culture
characteristic`) are spelled the way no description is.

* **Spelling:** `json_canonical` writes each key group in its most
  frequent spelling, counted in canonical space (the 2026-08-17 rule).
  The spellings come from a generated, git-tracked table,
  `data/vocab_key_spellings.json`, made by a `bin/` tool from corpus
  counts and shipped through `SHARE_FILES`.  A document's
  `json_canonical` depends only on the document, the synonym map and
  the table.  Edits are recorded only for keys that change.  A key
  whose group is not in the table is left as written, with no edit.
* **Stale documents:** a document is stale when it has no
  `json_canonical`, its `canonical_map_version` differs from the
  record's pin, or `json_annotated` changed since.  `build_vocab_tree`
  refuses to write the tree while any document is stale, naming the
  count and the command to run.  The `build_vocab` cron line runs the
  canonicalize step first.
* **Pinning:** the record's `vocab_canonical` is
  `{"synonyms": <blob hash>, "spellings": <blob hash>}`, written only
  by an explicit `--pin` run of the step.  A normal run refuses when
  the installed files do not match the pin.  Each document stores
  `canonical_source`, a hash of the `json_annotated` it was made from.
* **Smaller choices:**
  - the `-ies` fold spares `species` and `series`;
  - British spellings `colour` and `odour` fold now; P2b may add more;
  - leaf values stay as written, and the menu keeps `normalize_value`;
  - the menu shows synonym renames only: `Pileus (3,097; also: cap)`;
  - an edit's `rule` is `synonym`, `spelling` or `merge`, and `folds`
    names the language rules (`wrapper`, `case`, `separator`,
    `plural`, `latin_plural`, `british`);
  - the seed synonym map is `cap` = `Pileus`;
  - the non-synonyms record is `data/vocab_key_non_synonyms.json`, a
    JSON list of `{a, b, why}`, seeded with one or two entries.

### Open questions

* ~~Do the phase-1 canonicalization map and non-synonyms record move
  from `docs/` to `data/` as part of the move card?~~  *Resolved
  2026-11-18:* yes; they moved with EMbuJc0r (see the disposition log).
* How candidates are found: the `-ies` fold and spelling variants
  (`colour`) by rule; true synonyms by an SBERT similarity sweep
  reviewed by a human, never applied automatically.  *Designed
  2026-11-22:* see "P2b design" below.

### P2b design (proposed 2026-11-22, awaiting the user's decisions)

Card: https://trello.com/c/RdV2gXqD.  P2b fills the map that P2a reads.
Nothing a sweep finds is applied until a human has ruled on it.

#### What a probe of local production_v4_1 showed

Read-only, over the 44,211 documents' `json_annotated`, grouped by
`group_key` (so `Cap` is still separate from `Pileus`).  The 1,954
top-level groups found in at least 5 documents were compared pairwise.

* **Name similarity alone misses the best synonyms.**  SBERT
  (`all-MiniLM-L6-v2`) gives `Cap`/`Pileus` a cosine of 0.35, rank
  112,984 of 1.9 M pairs.  `Gills`/`Lamellae` and `Stem`/`Stipe` are
  as low.  Lay and technical names do not look alike.
* **Shared children find them.**  Compare the keys found under each
  group, each weighted by IDF so that generic children (`Shape`,
  `Size`, `Color`) count for little.  Restrict to a rare key against a
  common one (at most 20% of its support).  On that measure the top 40
  pairs include `Cap` → `Pileus`, `Gills` → `Lamellae`, `Stem` →
  `Stipe`, `Anamorph` → `Asexual morph`, `Ascomatal Wall` → `Peridium`
  and `Taxonomic Classification` → `Taxonomy`.  The noise is sibling
  structures with the same attributes: the cystidia family, and
  `Subtribe` → `Order`.
* **Name similarity finds what the rules miss.**  Ranked by an equal
  mix of name cosine and child overlap, 107 pairs score at least 0.85.
  Most are gaps in the language rules, not synonyms:

  | Candidate rule | Pairs (≥2 docs) | Docs on the minority side | Examples |
  |---|---:|---:|---|
  | Joined words (`ClampConnections`, `culturecharacteristic`) | 135 | 1,109 | `Conidiogenous cells` 3,630 / `ConidiogenousCells` 118 |
  | `-oma` / `-omata` | 17 | 666 | `Basidiomata` 1,446 / `Basidioma` 128; `Ascomata` / `Ascoma`; `Stromata` / `Stroma` |
  | More Latin plurals (`-ia`/`-ium`, `-ae`/`-a`) | 36 | 295 | `Pycnidia` / `Pycnidium`; `Lamellae` 1,823 / `Lamella` 10; `Columella` / `Columellae` |
  | `&` for `and` | 6 | 32 | `Ecology & distribution` / `Ecology and distribution` |
  | Accents | 7 | 19 | `Sétules` / `Setules`; `Hyménium` / `Hymenium` |

  Among the high-scoring pairs that are true synonyms: `Odor`/`Smell`,
  `Flavour`/`Taste`, `Disc`/`Disk`, `Fruit body`/`Fruiting
  body`/`Fruitbody`, `Teleutospores`/`Teliospores`, `Clamps`/`Clamp
  Connections`, `Pileal Trama`/`Pileus Trama`.  Clear hyponyms or
  compounds also score high: `Alpha conidia` ~ `Conidia`,
  `Pleuromacrocystidia` ~ `Pleurocystidia`, `Mature spores` ~ `Spores`,
  `Smell and Taste` ~ `Smell`.
* **Co-occurrence is a weak filter.**  True synonyms seldom appear in
  one description (`Cap` with `Pileus`: 2 of 114).  But most rare keys
  seldom co-occur with anything, so the filter keeps 89% of the top
  pairs.  It is useful for ruling pairs *out* (more than 5%), not in.
* **A gap in P2a.**  The language rules are code, and the record pins
  only the two data files.  Changing a rule changes `json_canonical`
  without making any document stale.  (A rule change usually changes
  the regenerated spelling table too, so the pin often moves anyway,
  but not always.)

#### The sweep

`bin/sweep_vocab_synonyms --experiment <X>` reads the experiment's
`json_canonical` (so pairs already merged do not reappear).  It writes
a review file and changes nothing else.  Each candidate is tagged with
the lanes that proposed it:

1. **Rules lane:** pairs that an untried language rule would merge.
   These are proposals for `vocab_canonical`'s rules and word lists,
   not map entries.
2. **Structure lane:** IDF-weighted child overlap, rare against
   common.
3. **Name lane:** SBERT cosine on the shown spellings, mixed with child
   overlap.
4. **Phase-1 lane:** pairs in `data/feature_label_canonicalization.json`
   (already human-reviewed feature-label merges) whose sides both occur
   as keys.

Every candidate carries its evidence: both supports, both spellings,
the scores, the co-occurrence rate, the shared children, and two
example document ids.  The sweep drops pairs already in the synonym map
or the non-synonyms record, in either direction.  It **groups
candidates into clusters** (`Basidioma`, `Basidiomata`, `Basidiome`,
`Basidiocarps`), so each cluster is ruled on once.  The map refuses
chains, so a cluster resolves to one target.

The first pass covers top-level keys.  Second-level keys are compared
only with siblings under the same parent (`Asci → Number of Spores` ~
`Asci → Spore Count`), as a second pass.

#### Review and apply

The reviewer marks each candidate with one of:
* `accept`: a synonym, from the rarer to the more idiomatic form
  (2026-08-17 rule, counted in canonical space);
* `flip`: a synonym, but in the other direction;
* `reject`, with a reason;
* `hyponym`: a kind of the other term; noted for a later `move`;
* `rule`: fold it with a language rule;
* `skip`: decide later.

`bin/apply_vocab_review <file>` validates the file and refuses on:
* a missing verdict, or a `reject` without a reason;
* an accepted pair that is in the non-synonyms record;
* a chain or conflict once each cluster is resolved;
* a review file made from another map version.

It then writes `data/vocab_key_synonyms.json` and
`data/vocab_key_non_synonyms.json` deterministically, with each entry's
support, experiment and date taken from the review file.  `hyponym`
verdicts go into the non-synonyms record, marked `"relation":
"hyponym"`, so the next sweep skips them and Proposal 3 can find them.
`rule` verdicts are listed for a code change, which goes through TDD as
usual.

Each accepted batch is one commit and a new map version.  The
experiment is re-pinned and backfilled, and `build_vocab` rebuilt (the
P2a order; the user runs it on production).

#### Decisions for the user

1. **Rules version in the pin.**  Recommended: `vocab_canonical`
   gets a `RULES_VERSION` constant, included in `installed_version`
   and the record's pin.  A test fails when the rule tables change
   without a bump.
2. **New language rules from the probe.**  Recommended: take joined
   CamelCase (`ClampConnections`), `&`/`and` and accents as rules.
   Grow the Latin-plural word list through the review (`-oma`/`-omata`,
   `-ia`/`-ium`, `-ae`/`-a` pairs as `rule` candidates), not as general
   suffix rules: `-ae`/`-a` and `-ia` would also hit words that are not
   plurals.  Lowercase joins (`culturecharacteristic`) cannot be split
   by rule; they go to the map.
3. **Review file format and home.**  Recommended: a TSV, one row per
   candidate with a `verdict` and a `why` column, editable in a
   spreadsheet.  Committed under `docs/vocab_reviews/<date>.tsv` as the
   provenance of each batch, and not shipped.
4. **Draft verdicts.**  Recommended: Claude pre-fills a suggested
   verdict and reason in separate `suggested_*` columns.  The user's
   `verdict` column is the only one that counts.
5. **First-pass size.**  Recommended: top-level keys, support of at
   least 5 documents, about 150 candidates across the lanes.
6. **Embedding model.**  `all-MiniLM-L6-v2`, already cached and used
   by `skol/constrained_decoder.py`.  The card's "existing embeddings"
   are of whole descriptions and do not cover keys, so key names are
   embedded fresh (seconds for ~2,000 keys).
7. **Phase-1 non-synonyms.**  `data/feature_label_non_synonyms.md` is
   prose about feature labels.  Recommended: carry its pairs that occur
   as vocab keys into `data/vocab_key_non_synonyms.json` in the first
   batch, reviewed like any candidate.
8. **Leaf values** stay out of P2b (the 2026-11-22 observations); only
   key pairs are swept.

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

## Observations after P2a (user, 2026-11-22) — future work

The user's verdict on the P2a tree: "MUCH better now".  Four issues are
recorded for later; none is scheduled.  Counts are from the local
production_v4_1 tree after the P2a backfill, at the default support
floor of 2.

### Leaf values need synonymy too

P2a folds keys only; leaf values stay as written (a P2a decision).
`Asci → Spore Count` has 32 values, most of them one answer:
`8-spored` 736, `eight-spored` 44, `Octosporous` 25, `8spored` 21,
`8-spori` 13, `8–spored` 10 (en dash), `octospori` 10, `Eight` 6.

* Rule-shaped folds come first and need no model: dash variants (`–`
  vs `-`), a missing hyphen (`8spored`), and Latin and English number
  words (`octo-`, `eight`, `8`).  Like the key folds, they would go in
  `vocab_canonical` as named rules, with tests.
* For larger value sets, the user asks whether SBERT can cluster them.
  It fits the plan's rule for synonym candidates (Proposal 2, Open
  questions): an SBERT similarity sweep proposes, a human reviews, and
  nothing is applied automatically.  Ranges (`4–8-spored`) are not
  synonyms of their endpoints and must stay separate.
* Priority is low, by the user's judgement: the wording in one
  description should barely move the SBERT embedding of the whole
  description, so search is little affected.  The gain is a shorter,
  clearer menu.

### Values filed under the wrong attribute

`Asci → Shape` holds spore counts (`8-spored` 15, `eight-spored` 4,
`Octosporous` 2) and wall types (`Bitunicate` 52, `Unitunicate` 34).
"Technically correct" in the source, but they belong under
`Asci → Spore Count` and a wall or type attribute.

* This is Proposal 3's `move`, applied to values rather than keys: re-home
  a value to the sibling attribute it describes, decided per
  (part, attribute, value) and recorded like the other layer rulings
  (`docs/vocab_tree_layer_rulings.md`).
* Related key synonymy, for P2b: `Asci → Number of Spores` (60) beside
  `Asci → Spore Count` (1,084).

### Generic attributes

`Asci → Type` (378 documents, 53 values) mixes kinds of answer:

* wall types: `Bitunicate` 173, `Unitunicate` 89, `Fissitunicate` 72;
* shapes: `cylindrical` 14, `clavate` 11;
* spore counts: `8-spored` 13;
* habit: `Subsessile` 9, `Pedicellate` 5, `Stipitate` 4;
* other: `Fasciculate` 16, `Evanescent` 10, `Aparaphysate` 9,
  `Hyaline` 6.

The user would concede that the named ascus types belong under `Type`:
`lecanora-type` 10, `bacidia-type` 4, `aspicilia-type` 3,
`lecidea-type` 2, `lichina-type` 2, `rhizocarpon-type` 2.  The rest
would be re-homed by value as above.  So a generic attribute is not
dropped: its values are sorted into the specific attributes beside it.

### The depth limit is in extraction, not in the tree

The user remembered a tree that seemed to be "the upper 4 levels of a
tree with no list of values at the leaves", and asked whether 4 levels
is a hard limit.

* Yes, for keys.  `bin/treatments_to_json.py` constrains decoding to
  `--schema-max-depth`, default 4 (range 2–6), and no pipeline
  overrides it.  In the local data no key is deeper than level 4
  (179,312 keys at level 4, none at level 5), and every leaf value is a
  string under 2–4 levels of keys.  The menu therefore has at most 5
  levels: up to 4 of keys, then values.
* Neither `build_vocab_tree` nor the widget limits depth.
* The "no values" tree did not reproduce.  No level-4 key holds an empty
  value, and every value is a string, which the tree keeps.  At the
  default floor, no key has all its children filtered out.  A user with
  a higher `vocab_min_support` setting can see a key whose values all
  fall below their floor.  That is the likeliest cause; the next
  sighting should note the path and the setting.
* If deeper structure is wanted, `--schema-max-depth 5` is a
  re-extraction, a change of the experiment's model output, and would
  need its own measurement.

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
* 2026-11-16: The user set the estimates in Trello's points extension,
  which the Trello adapter can't read: the `data/` move card 2, P2b 2,
  P2c from the 1–2 suggestion (the value is in Trello).
* 2026-11-17: The user approved the implementation order (new
  section), and decided P1's open points: counts go in a separate map,
  and the minimum support defaults to 2 and is settable per user from
  Settings.  P1 now has two thresholds, the build floor and the user's
  setting, and notes the versions-listing hazard of a counts key under
  `skol:ui:menus_*`.  P2a's open questions on collisions, map version,
  the non-synonyms format and the phase-1 sibling were decided as
  recommended (new "Decisions" subsection).  Still open: whether the
  phase-1 files move to `data/`.
* 2026-11-17: P1 build floor set to 2 by the user.  The proposed
  missing-counts fallback was withdrawn at the user's challenge and
  replaced by an error plus rebuild (10a/10b); P1 gains a deploy step.
  P1 tests approved and committed as expected failures.
* 2026-11-17: Added to P1: the widget polls a build it started instead
  of reporting `building` as a failure (a bug that predates P1, found
  while checking 10b).
* 2026-11-17: Frontend test cards created: Jest runner (P1
  prerequisite), moving `index.html`'s inline scripts to `src/`, and
  `PDFViewer.jsx` mocks (Backlog).
* 2026-11-17: P1 widget tests approved and committed as `test.failing`
  (`django/frontend/src/VocabTreeWidget.test.jsx`): build polling,
  the 3-minute limit, count labels, counts kept out of the description.
  Measured build: 54 s, 181,709 nodes.
* 2026-11-17: P1 implemented: `bin/build_vocab_tree.py` (`a23588a`),
  Django views, setting and migration 0024 (`5e94341`), the Settings
  input (`2165eba`), the Jest runner (`cd6adea`) and the widget
  (`284d596`).  Remaining: hand checks of the Settings input and the
  live widget, then the production deploy and the vocabulary-tree
  rebuild, which the user runs.
* 2026-11-17: The user's hand checks passed on the local dev server.
  They found two bugs older than P1, both fixed in `d450eae`: the
  widget's build POST had no CSRF token (403 for every signed-in user),
  and the experiment selector went blank after choosing an unapproved
  experiment.  They also exposed Django tests writing to the local
  CouchDB (Trello KosomUsu, fix; 9ZkM5Wuq, cleanup).  P1 remaining:
  deploy, then rebuild the vocabulary tree.
* 2026-11-17: P1 deployed to production by the user (skol and
  skol-django).  The first page load showed "Vocabulary counts
  missing" because the browser ran the cached pre-P1 bundle (static
  URLs carry no content hash); a forced reload loaded the new widget.
  Every deploy that changes a bundle needs that reload; the user
  decided against cache-busting static files (2026-11-17).  *Correction:* this entry first said the
  widget "then worked".  It did not: its rebuilds failed on the
  cluster (the hotfix entry below).
* 2026-11-17: Hotfix.  Production's Redis is a cluster, and the build's
  tree-plus-counts transaction failed there with
  `CrossSlotTransactionError`: the tree key and `<key>:counts` hash to
  different slots.  No production tree had counts, so the menus were
  down.  The counts key is now `{<menus key>}:counts`, hash-tagged into
  the tree key's slot; `rebuild_redis` matches it.  The local tests use
  a single-node Redis, which allows cross-slot transactions, so they
  could not show this; a fake cluster pipeline now does.
* 2026-11-17: The user deployed the cluster hotfix and the follow-up,
  hard-reloaded, and disabled both prose→JSON cron jobs on production
  (they held its GPU; Trello OWazYxed moves the work to Modal).
  Cache-busting is dropped.  `rebuild_redis` building the tree without
  `--experiment` is Trello dbQuh6Jc.
* 2026-11-17: **P1 complete.**  After the hard reload the widget
  rebuilt the production tree and the menus came back with counts
  (confirmed by the user).  Trello sa8yg1K4 marked complete.  Next in
  the implementation order: the `data/` move (EMbuJc0r).
* 2026-11-18: **The `data/` move is complete** (Trello EMbuJc0r).
  Recorded in `docs/shared-data.md`; tests `ebc3d2e`, implementation
  `715f2dc`, preexisting lint and type fixes `0184de3`.
  - `data/feature_label_canonicalization.json` and
    `data/feature_label_non_synonyms.md` moved from `docs/`; every
    reference follows, except quoted text, which stays verbatim.
  - `skol_share.py` resolves shipped data: `--share-dir` (in
    `common_parser()`), then `SKOL_SHARE`, then the checkout's
    `data/`, then `/opt/skol/share`.
  - `build-deb.sh` stages `SHARE_FILES` under
    `/opt/skol/versions/<ver>/share/`.  postinst links
    `/opt/skol/share` beside `/opt/skol/bin`, and
    `skol-switch-version` switches both (user decision, 2026-11-18).
  - Scope widened by the user to every `data/` file that shipped code
    opens.  Production had no `site-packages/data` (confirmed by the
    user), so the following were all missing there:
    * the canonicalization map (`bin/build_canonical_labels` would
      have raised);
    * `ocr_damage`'s corpus and botanical Latin lists (skipped
      silently, leaving `/usr/share/dict` alone);
    * `particle_detector`'s `personal_fungaria.json`;
    * the nomenclature tokenizers' LTWA and Index Fungorum authors.
  - Not shipped: `species/Export.csv` (95 MB, read only by the
    nomenclature tokenizers, which nothing imports; user ruling).
  - The new vocab-tree key map (P2) uses the same mechanism: add it
    to `SHARE_FILES`.
  - Production OCR-damage flags computed before this deploy used the
    smaller vocabulary.  Whether to recompute them is not decided.
* 2026-11-18: **P2a implemented; not yet deployed** (Trello AZeJMCR3).
  Tests `05fe2c3`, implementation `c0fab65`; design decisions
  `eb934c9` (Decisions for P2a, above).
  - `treatments_to_structured/vocab_canonical.py`;
    `bin/canonicalize_json.py`, the pipeline step after
    `treatments_to_json` in v4_crf and v3_logistic;
    `bin/build_vocab_spellings.py`.
  - Data: `data/vocab_key_synonyms.json` (seed `cap` = `Pileus`),
    `data/vocab_key_non_synonyms.json` (Sexual/Asexual morph,
    Conidiomata/Pycnidia) and `data/vocab_key_spellings.json` (26,923
    of 128,564 groups, generated from local production_v4_1,
    min support 2).  All three are in `SHARE_FILES`.
  - Measured across all depths, two majorities differ from the
    top-level counts in the decision: `Culture Characteristics` and
    `Asexual Morph` win.  The table follows the rule as decided.
  - Found while implementing: the children view lowercased every path
    part, so a case-kept tree was a 404 below the top level.  Fixed
    (exact match, else ignoring case), with a test added.  Also an
    approved test's regex could not step over the cron line's `2>&1`;
    fixed.
  - The table includes noise keys (measurement strings) that map to
    themselves.  They change nothing; raising `--min-support` would
    shrink the file.  Not done; the user decides.
  - Order of operations on any host: deploy, `canonicalize_json
    --experiment <X> --pin`, run the step (the backfill), then
    `build_vocab`.  Until the backfill, a build (including the widget's
    rebuild button) refuses and the existing tree stays.
- 2026-11-22 — P2a hand check on puchpuchobs (local CouchDB and
  Redis, production_v4_1).
  - Pinned to synonyms `58db06c5`, spellings `b315bf4c`, matching git's
    blob hashes for the committed files.
  - Backfill (`runstep production_v4_1 canonicalize_json`): 44,211
    documents updated, none unusable, 22 s.  A second run found all
    44,211 fresh.  Edits: 291,829 spelling, 487 synonym (every depth),
    348 merge.
  - `runstep production_v4_1 build_vocab`: 5,791 top-level terms,
    181,437 counted paths, 56 paths with `also` entries.  The children
    view answers `Pileus` (3,209; also: Cap), up from 3,097, and
    lowercase paths (`pileus`) still resolve.
  - Found: `build_vocab_tree.py` and `rebuild_redis.py` could not import
    `env_config` from a checkout, because the root was put on `sys.path`
    after it.  Installed packages were unaffected.  Fixed in `3d192db`.
  - Cosmetic, not recorded elsewhere: under `Pileus.Color`, a few
    annotations use a colour as a key (`Yellow`), so it shows in the
    key's capitalised form beside lowercase values.  It is one entry
    with no lowercase duplicate.
- 2026-11-22 — P2a complete on production (run by the user).
  - The release carrying the packaging fix `00c4d1d` is deployed:
    `canonicalize_json` now runs as a command.
  - Pinned `production` to synonyms `58db06c5`, spellings `b315bf4c`.
  - Backfill: 44,211 documents updated, none unusable.
  - `build_vocab`: identical to the puchpuchobs hand check.  181,555
    nodes, 5,791 top-level terms, 181,437 counted paths, 56 paths with
    `also` entries.
  - From now on the cron runs `canonicalize_json` before `build_vocab`
    (`debian/skol.cron`).  Its first scheduled run will show the
    nightly chain working, with every document already fresh.
  - Next in the order: P2b (https://trello.com/c/RdV2gXqD).
