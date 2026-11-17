# Vocabulary tree: structural edits and layer rulings

The standing record of how the keys in `json_annotated` may be
restructured when `json_canonical` is derived, and the rulings made
on individual layers.  It is the structural counterpart of the
synonym map (renames) and its non-synonyms record, both in `data/`.
Each ruling carries its evidence, so the next person to look at
`micromorphology` does not have to re-derive why it stays.

The plan this came from, and its measurements, are in
[plans/vocab-tree-narrowing.md](plans/vocab-tree-narrowing.md).

## The operations

Five operations, each decided on different evidence.  Every
application is recorded per document in `canonical_edits` under the
rule name in the first column, giving the old and new paths.

| Rule | Example | Effect on depth | Decided by |
|---|---|---|---|
| `rename` | `cap` → `pileus` | none | Is it the same concept?  Synonymy only — a narrower term (hyponym) is never renamed to its broader one. |
| `drop_layer` | `species → conidium → shape` ⇒ `conidium → shape` | the layer's whole subtree rises one level | Does **this layer** carry information?  A test on one node, plus the collision test below. |
| `move` | `conidiophore → stipe` ⇒ `synnema → stipe` | deeper or shallower, by the difference in parent depth | Where does **this concept** belong?  A part-of claim, decided per (parent, key) pair. |
| `insert_layer` | `pileus`, `stipe`, `lamellae` ⇒ `basidiome → {…}` | the grouped subtrees sink one level | The inverse of dropping; needs the R5a schema. |
| `key_to_value` | `sexual morph → undetermined` ⇒ `sexual morph = undetermined` | the false level disappears | Is the "key" really an answer? |

### A layer may be dropped only if it passes both tests

1. **It adds nothing to the path.**  It has no leaf values of its
   own, and its name does not distinguish its children from the same
   keys elsewhere.
2. **Dropping it creates no homonym collision.**  When a child rises
   to the grandparent, it must mean the same thing as the key already
   there with that name.  If it doesn't, the layer is carrying
   information by disambiguation, even if its name sounds generic.

### Moves are per (parent, key) pair, never per key

The parent can change what a key means.  `stipe` under `ascus` or
`conidiophore` is not the stipe of a mushroom.  "Move every `stipe`
under its most common parent" would merge different structures.

## Layer rulings

Production `production_v4_1`, 43,534 documents with `json_annotated`,
measured 2026-11-16.

| Layer | Docs | Ruling | Evidence |
|---|---:|---|---|
| `species` | 9,847 | **drop** | 5,901 distinct children, no values.  Every treatment describes one species, so the name adds nothing.  Children (`name` 7,662, `description`, `asexual morph`, `culture characteristic`, `conidium`, …) are the same keys found at the top level. |
| `morphology` | 1,576 | **drop, pending the collision test** | 1,704 children, no values: `conidium`, `conidiophore`, `spore`, `ascus`, `stipe` (168), `pileus`.  Its `stipe` children have not been checked for the conidiophore-stipe homonym that blocks `micromorphology`. |
| `description` | 756 | **drop, pending the collision test** | 1,277 children, no values: `conidium`, `morphology`, `conidiophore`, `stipe` (66), `pileus`.  Same caveat. |
| `asexual morph` | 2,550 | **keep** | States which morph a structure belongs to.  `docs/feature_label_non_synonyms.md` records sexual/asexual morph as opposites; dropping it would pool the two morphs' hyphae, mycelium and cultures. |
| `sexual morph` | 2,358 | **keep; `key_to_value` on its answer children** | Same reasoning as `asexual morph`.  Its children include `observed` (288), `undetermined` (207), `unknown` (139) and `not observed` (107).  These are answers stored as keys ("Sexual morph: Undetermined"). |
| `micromorphology` | 276 | **keep** (fails the collision test) | 188 children: `conidiophore` 158, `conidium` 151, `ascomata` 121, `phialide` 119, `vesicle` 77, `stipe` 76, `metulae` 71.  **All 76 documents with `micromorphology → stipe` also have conidiophore-type siblings** (`vesicle`, `metulae`, `phialide`, `conidial head` or `conidiophore`), so that `stipe` is the aspergilloid conidiophore stipe.  Dropping the layer would file it with the mushroom stipe (2,968 documents at the top level).  The right fix is `move` (`micromorphology → stipe` ⇒ `conidiophore → stipe`); only after that could the layer be re-tested for dropping. |
| `macromorphology` | 17 | **undecided** (too rare to rule on) | 39 distinct children, none in more than 7 documents (`pileus` 7).  It shares only 7 keys with `micromorphology` (`stipe`, `lamellae`, `colony`, `peridium`, `size`, `growth`, `shape`).  Its frequency is below any threshold the menu would show. |

Other morphology-named top-level keys, not yet examined:
`microscopic feature` (142), `colony morphology` (133),
`morphological characteristic` (65), `cell morphology` (59),
`morphological feature` (43), `asexual morphology` (39),
`microscopic character` (21), `morphological character` (17).
`morphological difference` (37) and `morphological similarity` (25)
are comparisons with other taxa, not containers.

## Change log

* **2026-11-16** — Created from Proposal 3 of the vocab-tree plan:
  - the five operations;
  - the drop and collision tests;
  - rulings on `species`, `morphology`, `description`, `asexual morph`,
    `sexual morph`, `micromorphology` and `macromorphology`.
