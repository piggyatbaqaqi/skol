# Vocabulary as a learned concept graph (P2b design B)

Status: **proposed 2026-11-22, awaiting the user's decisions.**  Card:
https://trello.com/c/RdV2gXqD.  Design A, a per-pair review sweep, is
frozen in [`vocab-tree-narrowing.md`](vocab-tree-narrowing.md) ("P2b
design A").  Either design's features may be cherry-picked later.

## The constraints this design answers to

Stated by the user, 2026-11-22, for SKOL as a whole:

* **Taxon-agnostic.**  Every component must work as well for plants,
  insects or diatoms as for fungi.  A hand-written list of mycological
  words is not a component; it is data for one taxon.
* **Training material over audit results.**  The methods must scale to
  all of biology.  Hand review is worth its cost only when its output
  trains or seeds something that produces further vocabulary on its
  own.

Design A fails the second constraint: every synonym is one human
decision, and the work grows with the vocabulary.  Design B treats each
human decision as a **labelled example**, and the map as the output of
a model trained on those labels.

## The proposal, as suggested

Google Search AI Mode suggested this, 2026-11-22 (pasted by the user):

1. **Sense induction.**  Embed each occurrence of a term in its
   context with a biomedical encoder (BioLinkBERT, PubMedBERT, SciBERT).
   Cluster the occurrences per term, so that a polysemous term such as
   `stipe` (mushroom stem; conidiophore stalk) splits into senses.
2. **Candidates.**  Pair the sense centroids by cosine (≥ 0.75).  Add
   lexical edges: compound suffixes (`pleurocystidium` → `cystidium`)
   and definitional patterns (`X (or Y)`, `X, i.e. Y`, `Y such as X`).
3. **Relation classification.**  Label each pair, with prototype
   sentences, as synonym, hyponym, co-hyponym or unrelated.  This uses
   directional entailment or a local LLM with structured output.
4. **A SKOS concept graph.**  One `skos:Concept` per sense, with
   `prefLabel` and `altLabel` for synonyms, `broader` for hyponyms and
   parts, and `exactMatch` to OBO terms (FAO, PATO).  The menu shows
   `prefLabel`s, collapses chains without branches, and groups
   co-hyponyms.

## What SKOL's data says about it

Read-only probes of local production_v4_1 (44,211 treatments),
2026-11-22.

* **Context is available.**  Each structured treatment keeps its
  `description` text beside `json_annotated`, so a key's occurrences
  can be found in the prose.  Keys are written by the extraction model,
  though, and need not appear verbatim (`Culture characteristics` is
  often a heading the model invented).  The alignment rate is unmeasured.
* **The JSON is context too.**  A key's parent path, sibling keys and
  values say which sense it has: `Micromorphology → Stipe` beside
  `Vesicle` and `Metulae` is the aspergilloid stipe
  (`docs/vocab_tree_layer_rulings.md`).  That signal works for any
  taxon and needs no alignment.
* **Definitional patterns are rare in treatments.**  Treatments are
  telegraphic.
  - `X, i.e. Y` and `Y such as X` linked two vocabulary keys **zero**
    times; `X (= Y)` 8 times.
  - `X (Y)` linked keys 2,483 times, mostly abbreviations: `agar (PDA)`
    322, `agar (MEA)` 232.  That is an abbreviation lexicon, useful in
    itself, not synonymy.
  - `X or Y` (7,760) is mostly alternatives: `odor or taste`, `brown or
    black`, `cylindrical or clavate`.  Those are **co-hyponym and
    contrast examples**, which a relation classifier needs as much as
    synonyms.
  - Design B should not depend on these patterns for synonyms; they
    are a source of the other labels.
* **Ontologies give labels, but noisy ones.**  `data/ontologies/`
  already ships FAO and PATO.
  - FAO is small: 114 terms, 27 EXACT synonyms, 124 `is_a`, 23
    `part_of`.  PATO has 2,785 terms and 645 EXACT synonyms, for
    qualities (shapes, colours, textures), which are our leaf values.
  - FAO calls `cap` and `stem` only RELATED to `pileus` and `stipe`, and
    lists `fruitbody` both as a term and as an EXACT synonym of
    `sporophore`.  Ontology labels need the same scrutiny as any other
    weak label.
  - Other OBO ontologies (Plant Ontology, UBERON, the Hymenoptera
    Anatomy Ontology) supply the same kinds of label for other taxa.
    This is the part that scales to all of biology.
* **Name similarity is a weak signal for the important pairs.**  From
  the design A probe: SBERT scores `Cap`/`Pileus` at 0.35.  A fixed
  cosine threshold of 0.75 would never propose it.  The suggested
  thresholds (0.22 for sense splits, 0.75 for pairs) are guesses until
  they are tuned on labels.
* **The suggested encoders are biomedical**, trained mostly on human
  medicine.  Whether they beat general SBERT on Latin and Greek
  morphological terms is a question for a labelled set, not an
  assumption.
* **Compound suffixes are a hyponym signal across biology**
  (`basidiospore` → `spore`, `pleurocystidium` → `cystidium`).  But a
  multiword key whose head comes last is often a *part*, not a kind:
  `Pileus trama` is part of the pileus, not a kind of trama.  Its
  precision needs measuring.
* **`broader` must be typed.**  The tree mixes kind-of (`cystidium` →
  `cheilocystidium`) and part-of (`basidiome` → `pileus`).  Proposal
  3's `move` operations are part-of claims.  SKOS's untyped `broader`
  loses that.  ISO 25964's extension of SKOS distinguishes
  `broaderGeneric` (kind-of) from `broaderPartitive` (part-of); use it.

## Design B

Five stages.  Each one produces either labelled examples or a
generated artifact checked against held-out labels.

### B1. A labelled relation set (the training material)

A git-tracked file of **labelled pairs in context**:
`(term A + context, term B + context, relation, source, confidence)`.
The relations are:
* `exact_synonym`;
* `kind_of` (A is a kind of B);
* `part_of` (A is part of B);
* `co_hyponym` (siblings; do not merge);
* `abbreviation`;
* `unrelated`.

Sources, each kept apart so its reliability can be measured:

* **Ontologies.**  OBO EXACT synonyms, `is_a` and `part_of` among
  terms that occur as keys or values.  Any taxon, any ontology.
* **Decisions already made.**  `cap` = `Pileus`; the non-synonyms
  records; the phase-1 feature-label map; the layer rulings.
* **Text patterns.**  `X (ABBR)` for abbreviations, and `X or Y` for
  co-hyponyms and contrasts.
* **A teacher model.**  An LLM labels sampled pairs, with prototype
  contexts.  A human audits a random sample of its labels; the audit
  rate is the teacher's measured precision.
* **Human review.**  Only of the queue in B4, and every verdict joins
  the set.

### B2. Senses from context

A sense is a cluster of a term's occurrences.  Start with the cheapest
taxon-agnostic context, the JSON: parent path, sibling keys and
values.  Measure how often a term splits.  Add contextual embeddings of
the aligned description span only where the JSON leaves a term
ambiguous, and only after measuring the alignment rate.

### B3. Candidate generators, judged by recall

Each generator proposes pairs; each is scored by how many held-out
labelled pairs it recovers.  Candidates include design A's lanes
(language rules, IDF-weighted child overlap, name cosine), compound
suffixes, and sense-centroid similarity under each encoder (MiniLM,
mpnet, PubMedBERT, BioLinkBERT, SciBERT).  The comparison decides which
encoders and thresholds to keep.

### B4. A relation classifier, with an active-learning queue

Train a classifier (a cross-encoder over the two terms and their
contexts) on B1's labels to predict the relation and a confidence.
Report precision and recall per relation on held-out labels.

* Pairs it is sure of become graph edges without review, at a
  confidence threshold set from the held-out precision.
* Pairs where it is unsure, and that matter most (most documents
  affected), go to a human queue.  Each verdict is a new label; retrain
  and repeat.  The human's time goes where it improves the model most.
  This is the only hand review, and it generates further vocabulary.

### B5. The concept graph, pinned and applied

A git-tracked, generated graph file in SKOS terms:
* one concept per sense, with a stable id;
* a `prefLabel` chosen by the 2026-08-17 rule (most idiomatic, counted
  in canonical space);
* `altLabel`s, typed `broaderGeneric` and `broaderPartitive` edges,
  and `exactMatch` to OBO ids;
* the provenance of each edge (label source, or classifier and
  confidence).

It replaces `data/vocab_key_synonyms.json` as the pinned artifact.
P2a's mechanism carries over unchanged in shape: the record pins the
graph's blob hash, `canonicalize_json` renames by sense (so `stipe`
under a conidiophore is not merged with the mushroom stipe), the edits
are recorded per document, and stale documents block the menu build.
`altLabel`s become the menu's "also:".  `broader` edges feed Proposal
3's moves and the menu's chain collapsing.  Exports to RDF or JSON-LD
come for free, for reuse by others.

## Cost and risk

* **Larger than design A.**  It needs a labelled set before anything
  reaches the menu, and a trained model before review gets cheaper.
  Design A would ship synonyms sooner; it would not scale.
* **The cross-taxon claim is untested.**  The corpus is fungal.  Until
  non-fungal treatments exist, the test of generality is held-out
  ontology labels and keeping taxon words out of the code.  The
  `taxon-agnostic` rule can be enforced in review, and partly by tests:
  no fungal word lists in code.
* **The teacher model's labels are weak labels.**  They count only with
  a measured audit precision, and never as ground truth.
* **Polysemy changes the canonical form.**  `json_canonical` keys would
  carry sense ids or remain labels with a sense recorded in
  `canonical_edits`.  That is a decision below.

## Decisions for the user

1. **Adopt design B's direction**: labels as training material, a
   relation classifier, and a generated concept graph.
2. **Relations**: the six above, with `broader` typed as generic and
   partitive.
3. **Label sources**: which to use, and the human audit rate for the
   teacher model's labels.  Which model is the teacher (a local model,
   or Claude through the API), given cost.
4. **Ontologies**: add Plant Ontology and UBERON (or others) to
   `data/ontologies/` now as label sources for other taxa, or start
   with FAO and PATO.
5. **Senses**: start with JSON-context senses, and add description
   embeddings only where measured to be needed.
6. **`json_canonical` and senses**: do keys become sense ids, or stay
   labels with the sense in `canonical_edits`?
7. **First milestone**: B1 and B3 only.  That means a labelled set, plus
   measured recall for each candidate generator and encoder, before any
   classifier or menu change.
8. **What to cherry-pick from design A now**, if anything.  Its new
   language rules are taxon-agnostic morphology (joined words, `&`,
   accents), and it found the `RULES_VERSION` gap in P2a's pin, which
   either design needs fixed.
