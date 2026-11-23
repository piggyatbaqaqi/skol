# Vocabulary as a learned concept graph (P2b design B)

Status: **direction accepted 2026-11-22; first milestone B1 + B3.**
Rulings and the questions still open are under "Decisions" at the end.
Card:
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

A sense is a cluster of a term's occurrences.  An **occurrence** is one
use of a key in one treatment.  Its context can come from two places:

* **The treatment's own JSON** (`json_annotated`, after P2a's
  renaming): the key's parent path, its sibling keys, and its children
  and values, all in that one document.  This is per document, so it
  keeps co-occurrence.  The Vocabulary tool's tree is the sum of these
  paths over all documents and loses it: its `Micromorphology → Stipe`
  node mixes the children of every stipe filed there.  In one document,
  a conidiophore stipe sits beside `Vesicle` and `Metulae`; a mushroom
  stipe sits beside `Pileus` and `Lamellae`.  Senses come from the
  per-document context, not from the tree.
* **The description text**: the sentence of the treatment's
  `description` where the term is written.  A **description
  embedding** is a vector from running a sentence encoder (the
  `all-MiniLM-L6-v2` model SKOL already uses, or PubMedBERT and the
  like) over that sentence.  We compute them ourselves from the 44,211
  stored descriptions; nothing is downloaded but the model.  A GPU
  embeds hundreds of thousands of sentences in minutes.  The cost is
  alignment: the key must be found in the prose, and the extraction
  model often invents keys (`Culture characteristics`) that the text
  does not contain.

Start with the JSON context: it is cheaper, works for any taxon, and
needs no alignment.  Measure how often a term splits.  Add description
embeddings only for terms the JSON leaves ambiguous, after measuring
the alignment rate.

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
**The graph is the source; `json_canonical` is derived from it**, as
P2a derives it today from `json_annotated` and the map, and is never
the definitive record of anything.  Its keys are the concept's
`prefLabel`, which the graph requires to be unique.  SKOS itself does
not; this is SKOL's own rule.  Where two senses share a word, the
labels are qualified, for example `Stipe (conidiophore)`.  The concept
id of each renamed key is recorded in `canonical_edits`.  Keys stay
readable and carry the sense, with no opaque ids in the JSON.

P2a's mechanism carries over unchanged in shape: the record pins the
graph's blob hash, `canonicalize_json` renames by sense (so `stipe`
under a conidiophore is not merged with the mushroom stipe), the edits
are recorded per document, and stale documents block the menu build.
`altLabel`s become the menu's "also:".  `broader` edges feed Proposal
3's moves and the menu's chain collapsing.  Exports to RDF or JSON-LD
come for free, for reuse by others.

A test validates the graph against the W3C SKOS integrity rules:
* one `prefLabel` per language per concept;
* `prefLabel` and `altLabel` disjoint;
* no cycles in `broader`.

SKOL adds one rule of its own: `prefLabel`s are unique.  The test uses
`rdflib` or `pyshacl`, which would be new dependencies and must be
packaged.

### What the menu could do with the graph (later; suggested)

Google Search AI Mode suggested this, 2026-11-22 (pasted by the user):

* top-level menu entries from `prefLabel`s, grouped by collections;
* `altLabel`s hidden from the lists but found by autocomplete;
* **character axes**: under an organ, only the qualities that apply to
  it (`Pileus` → Shape, Surface texture, Color), so the same quality
  subtrees are not repeated under every organ;
* modified states (`subglobose`) tucked under their base state
  (`globose`).

This needs to change before it meets SKOL's constraints:
* **Axes must be generated, not written.**  Which qualities occur under
  which organ, and how often, is a count over `json_canonical`, and
  works for any taxon.
* **No fungal names in the structure.**  The suggestion uses a `myco:`
  namespace and Macroscopic/Microscopic Features collections.  The
  namespace would be generic, and any collections would be derived
  from the data.

The character axes would also change how the Vocabulary tool behaves,
not refine it.  Under the tool rule (`docs/plans/result-splitting-features.md`),
that makes it a new tool beside the current one.  It is out of scope
until B5 ships.

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
* **Polysemy changes the canonical form.**  A key that splits into
  senses gets a qualified `prefLabel` in `json_canonical` (B5), so menu
  counts and the classifier's inputs change for those keys.

## Cherry-picked from design A (adopted 2026-11-22)

Both are mechanism, not curation, and either design needs them.  They
can land before B1, through TDD as usual.

1. **A rules version in the pin.**  The language rules are code, and
   P2a's record pins only the two data files.  Changing a rule changes
   `json_canonical` without making any document stale.  (A rule change
   usually changes the regenerated spelling table too, so the pin often
   moves anyway, but not always.)  The fix:
   * `vocab_canonical` gets a `RULES_VERSION` constant;
   * it is included in `installed_version` and in the record's pin;
   * a test fails when the rule tables change without a bump.

   Under B5 the pin becomes {graph, spellings, rules version}.
2. **Three new language rules**, all morphology that holds in any
   taxon's literature:
   * joined CamelCase words (`ConidiogenousCells` → `Conidiogenous
     cells`: 135 pairs, 1,109 documents on the minority side);
   * `&` for `and` (6 pairs, 32 documents);
   * accents folded (`Sétules` / `Setules`: 7 pairs, 19 documents).

   **Not taken as rules:**
   * More Latin plurals (`-oma`/`-omata`, `-ia`/`-ium`, `-ae`/`-a`).  As
     general suffix rules, `-ae`/`-a` and `-ia` would also hit words that
     are not plurals.  In design B they are candidate generators (B3),
     and their pairs are labelled like any other.
   * Lowercase joins (`culturecharacteristic`), which no rule can split.
     They become `altLabel`s in the graph.

Counts are from the design A probe, local production_v4_1
(`vocab-tree-narrowing.md`, "P2b design A").

## The teacher model's cost

The user has had good results labelling through the Claude API.  The
choice of model waits for a dollar estimate.

**Method.**  Each pair is one request.
* A cached system prompt of about 3,000 tokens: the six relations and
  their definitions, plus worked examples.
* About 400 tokens per pair: the two terms, their JSON contexts, and a
  description sentence each.
* About 100 tokens of structured output: the relation, the confidence
  and a short reason.
* Thinking: Claude Opus 5.5 cannot turn thinking off, so allow about
  300 more output tokens at low effort.  Claude Sonnet 5.5 can turn it
  off.

All requests go through the Message Batches API, which charges 50% of
list price.  The list prices are from the claude-api skill's table,
cached 2026-10-06; check the current prices before the pilot.

| Teacher | Per 10,000 pairs, cache hits | No cache hits (worst case) |
|---|---:|---:|
| Claude Opus 5.5 ($4 / $20 per MTok; cache reads $0.20) | ≈ $55 | ≈ $110 |
| Claude Sonnet 5.5 ($2 / $10; cache reads $0.20), thinking off | ≈ $15 | ≈ $40 |
| Claude Sonnet 5.5, thinking on | ≈ $30 | ≈ $55 |
| Claude Haiku 5.5 ($0.10 / $0.50) | ≈ $2 | ≈ $3 |

Prompt caching inside a batch is best effort, which is why there is a
worst-case column.  **Scale:** the first milestone needs a few thousand
teacher labels, which is tens of dollars on Opus.  Every round of the
B4 loop adds a few thousand more.  The whole project is likely a few
hundred dollars at most.

**Recommended:** before choosing, run a pilot of 200 pairs on each
candidate model, about $1–2 each.  The pilot measures:
* the real token counts;
* the precision each model gets on the human audit.

Choose by **cost per correct label**, not per request.  A cheaper
teacher that labels worse costs audit time, and its mistakes become
training data.

## Ontologies

Ruled 2026-11-22: start with FAO and PATO.  Other taxa are a couple of
years away, so Plant Ontology, UBERON and the like wait.

**The case for the Ascomycete Phenotype Ontology (APO)**, measured
2026-11-22:
* The probe ran over the `releases/2026-10-30` file (CC BY 4.0).  Its
  names and synonyms were matched to top-level and second-level keys by
  `group_key`, over local production_v4_1.
* **Size:** 309 live terms, 33 EXACT synonyms, 306 `is_a` edges, and no
  `part_of`.
* **Namespaces:** observable 246, experiment_type 63, mutant_type 18,
  qualifier 15.

What it offers:
* **It names an axis that FAO and PATO do not: cultures.**  61 terms
  name a key, 16 of them in at least 50 documents.  The hits are mostly
  observables of colonies and cells:

  | Key | Documents |
  |---|---:|
  | colony size | 605 |
  | colony color | 486 |
  | colony appearance | 483 |
  | colony shape | 385 |
  | cell size | 224 |
  | cell shape | 200 |
  | sporulation | 729 |
  | conidiation | 80 |

  FAO's 69 hits are all structures (conidium 11,771, conidiophore
  9,216, …).  Culture descriptions are a large part of ascomycete and
  asexual-morph treatments.
* **Its shape fits the character-axis idea.**  An APO phenotype is an
  observable plus a qualifier, which is close to our key → value.

What counts against it:
* **It was built for yeast mutant phenotypes.**  The experiment_type and
  mutant_type namespaces don't apply to treatments.
* **It has false friends.**  APO's `PCA` is a protein-fragment
  complementation assay; ours (146 documents) is potato carrot agar.
* **It has few synonyms**, so it adds little to the `exact_synonym`
  labels.  Its `is_a` edges are its value.
* **It is ascomycete-only**, so it doesn't help the cross-taxon goal.

**Recommendation:** worth adding as a third label source, limited to
the observable namespace.  It is small (106 KB) and covers what FAO
misses.  Its labels would be held to the same audit as the others.
Not added; that is the user's decision.

## Scale

**Months**: roughly two to three of elapsed time, at the pace P2a
went.  Each stage below is roughly one to three weeks of work.

| Work | Scale | Note |
|---|---|---|
| Cherry-picks (rules version, three rules) | days | Before B1; TDD |
| B1 labelled set: ontology and existing-decision importers, the pattern miner, the teacher pilot and its audit | 1–2 weeks | Includes the user's audit time |
| B3 generators and the recall harness | 1–2 weeks | First milestone ends here |
| B2 senses from JSON context | about 1 week | |
| B4 classifier and the first review rounds | 2–3 weeks, then ongoing | Rounds are paced by review |
| B5 graph, pin, canonicalize by sense, backfill | 2–3 weeks | Production steps run by the user |

The first milestone (cherry-picks, B1, B3) is about three to four
weeks.  Its recall numbers are the first point where the rest can be
estimated with any confidence, so the later rows should be estimated
again then.

## Decisions

Ruled by the user, 2026-11-22:

1. **Direction**: design B goes ahead.  It is more robust than design A
   and slower to realize.  Trello cards come once the plan has work
   products whose size can be estimated.
2. **Relations**: the six, with `broader` typed as generic
   (`broaderGeneric`) and partitive (`broaderPartitive`).
3. **Label sources**: the teacher is Claude through the API, pending a
   dollar estimate ("The teacher model's cost", above).  The model and
   the audit rate are still open.
4. **Ontologies**: FAO and PATO first.  The case for APO is above; it
   has not been decided.
5. **Senses**: explained under B2.  The recommendation stands: start
   with per-document JSON context, and add description embeddings only
   where they are measured to be needed.  It has not been ruled on.
6. **`json_canonical` and senses**: the graph is the source and
   `json_canonical` is derived (B5).  Keys become unique, readable
   `prefLabel`s, and concept ids are recorded in `canonical_edits`.
   This follows the user's point that `json_canonical` is derived and
   should not be the source for anything.  The details have not been
   ruled on.
7. **First milestone**: B1 and B3 only.
8. **Cherry-picks from design A**: adopted as recommended (above).

Still open:
* the teacher model and audit rate, after the pilot;
* APO;
* decision 5 and the details of decision 6;
* the Trello layout (below).

### Trello layout (proposed)

The SKOL board's other lists are themed projects (Plazi integration,
LOTE support, Tsugae field guide).  Description tools holds the cards
for user-facing tools.

* **Proposed:** a new list, **Vocabulary concept graph**, for this
  project's back-end cards: cherry-picks, B1–B5, and the teacher pilot.
* Cards that change what users see go in Description tools and link
  back to the project list: a menu built from the graph, autocomplete
  over `altLabel`s, and the character-axis tool.  That list stays a view
  of tool work.
* Card RdV2gXqD (P2b) moves to the new list as the umbrella card.  Its
  description is rewritten to point at this document.
