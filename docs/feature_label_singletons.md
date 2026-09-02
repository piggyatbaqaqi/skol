# The singleton tail: what 854 once-seen labels actually are

Third companion to
[`docs/feature_label_canonicalization.json`](feature_label_canonicalization.json)
(pairs we collapse) and
[`docs/feature_label_non_synonyms.md`](feature_label_non_synonyms.md)
(pairs we refuse to collapse). This one is about the labels that have
no pair at all.

**The question.** After rounds 5 and 6, the canonical feature
vocabulary is 1 480 labels and **854 of them (58 %) appear in exactly
one treatment.** M3 trains a segment classifier on this vocabulary.
Before buying more annotation, someone had to establish whether that
tail is a real ontology — rare traits from taxonomically distant
groups — or schema noise that more annotation would only lengthen.

Sampled and classified 2026-09-02. **The answer is "about half and
half", and the noise half has shape.**

## Method, and its limits

Fifty singletons sampled at random (seed 20260902) from the canonical
rounds-5+6 vocabulary. For each: the label, the field, the
`source_text` it annotated, and the three nearest **established**
labels (df ≥ 5, 288 of them) ranked by TF-IDF cosine over their
*span* text.

**Ranking by span, not by label string, is deliberate.** An earlier
sweep over the same tail showed label-string similarity proposing
`Pycnidiospores → conidiophores`, `Apothecium → hypothecium` and
`Mitochondria → microconidia`; mycological morphology is built from a
few morphemes, so string proximity says almost nothing about
biological identity.

**The candidate generator was checked against both control sets
before use**, which is the rule this project already follows:

| control | result |
|---|---|
| **positive** — a known drift form's span should rank its canonical target | **#1 in 71 %** (32/45), top-3 in 93 % |
| **negative** — no recorded non-synonym pair may collapse | **0 collapses** across `Sexual`/`Asexual morph`, `Macro`/`Microconidia`, `Cystidia`/`Cheilocystidia`, `Conidiomata`/`Pycnidia`, `Hymenium`/`Subhymenium` |

So it is a usable *candidate generator* and nothing more. **The 50
classifications below are my calls, not verified ones**, made with the
span in view; they are recorded individually so they can be
re-adjudicated rather than taken on trust.

## The result

| class | n | share |
|---|---:|---:|
| **(a) genuine rare trait** — keep | 28 | 56 % |
| **(b) drift, collapsible into an existing label** | 10 | 20 % |
| **(c) not a trait** — an attribute, a compound, or an artifact | 12 | 24 % |

**Roughly 44 % of the tail is consolidatable.** At n=50 the 95 %
interval on that is about ±14 points, so **260–500 of the 854**, best
estimate ~375. That is wide, and it is enough: it separates "the tail
is real, leave it alone" from "the tail is half noise", which was the
decision it had to inform.

### (a) genuine rare traits — 28

`Venae externae`, `Cytokinesis`, `Replicative conidia`, `Supporting
cells`, `Aerial phialides`, `Basal membrane`, `Zygotes`, `Perigynia`,
`Germlings`, `Rostral hyphae`, `Macrospores`, `Stromatal wall`,
`Macroconidiogenous cells`, `Perithecial setae`, `Conidial
propagules`, `Merosporangia`, `Ustilospores`, `Primary appendages`,
`Paraphysoids`, `Blisters`, `Conidial locules`, `Hydrogen sulfide
production`, `Helminthosporioid hyphae`, `Trophic amoebae`,
`Secondary phialides`, `Extracellular deposits`, `Infection hyphae`,
`Teliospore wall`.

These are rare because the *taxa* are rare — slime moulds, rusts,
smuts, lichens, anamorphic ascomycetes, one mycovirus paper. Several
are position- or order-qualified in exactly the way
`feature_label_non_synonyms.md` protects (`Aerial phialides` beside
`Sporodochial phialides`, `Primary appendages` beside `Apical
appendages`), and collapsing them would repeat the `Cheilocystidia`
mistake that document exists to prevent.

Two carry a caveat rather than a reclassification: `Perigynia`'s span
reads as host-plant vocabulary, and `Infection hyphae`'s span is a
figure legend (`IH_18h, Infection hyphae at 18 hpi`) — the label is
real, the span is a caption.

### (b) collapsible drift — 10

| label | collapses toward | shape |
|---|---|---|
| `Lower Surface` | `Lower surface` | case only |
| `Secondary Conidiophores` | `Secondary conidiophores` | case only |
| `Sterile Hyphae` | `Sterile hyphae` | case only |
| `Aecidia` | `Aecia` | historical synonym |
| `Gonytrichum-type Conidiogenous cells` | `Conidiogenous cells` | genus-type qualifier |
| `Colony on oatmeal agar` | `Colony` + context `OA` | growth condition |
| `Stem lesions` | `Lesions` | host-organ qualifier |
| `Stalk of ascomata` | `Stalk` | part-of phrasing |
| `Pseudoparenchymatous layer cells` | `Pseudoparenchymatous layer` | part-of phrasing |
| `Conidiophore cells` | `Conidiophores` | part-of phrasing |

### (c) not a trait — 12

* **Attribute of an established trait** (5): `Ascomata height`,
  `Sporangial dimensions`, `Pileipellis pigmentation`, `Sporangial
  vacuoles`, `Conidiophore cells`\*. The label names a *property* of
  something already labelled, not a structure.
* **Compound of two or three labels** (3): `Gamma and beta conidia`,
  `Micro- and macropycnidia`, `Microconidia chlamydospores and sexual
  morph`.
* **Artifact or out of schema** (4): `Anatomy` (a section heading),
  `Exsiccatum` (a specimen reference), `Head cells` (OCR wreckage —
  span reads `he. ovate, entire, 11—15x 10—13 u.`), `Stipe base`
  (span is the single Latin word `bulboso`), `Virion` (a mycovirus
  particle, not a fungal character).

\* `Conidiophore cells` is arguably (b); counted once, in (b).

## The tail has shapes, and a quarter of it is mechanically detectable

The hand sample suggested the noise is not 854 unique problems.
Counting the shapes across **all 854** singletons rather than the
sample:

| shape | n | example |
|---|---:|---|
| `<established label>` + a property word | **73** | `Ascomata height`, `Annulus microstructure` |
| compound — contains "and"/"or" | **59** | `Basidia and cheilocystidia`, `Ascomata and Pycnidia co-occurrence` |
| base + growth condition | **46** | `Chlamydospores in culture`, `Colonies in culture` |
| **minted from an absence statement** | **39** | span reads *not observed* / *absent* / *none seen* |
| case-only duplicate of a label that already exists | **36** | `Aerial Phialides`, `Colony Reverse`, `Conidia in Culture` |
| **union (shapes overlap)** | **221** | **26 % of the tail** |

So a quarter of the tail is reachable without any judgment call at
all, and the hand sample says roughly another 18 % needs a human.

### The absence-statement class is a prompt defect, not a data defect

**39 singleton labels were minted from spans saying the structure is
absent** — `Micro- or macropycnidia not seen`, `gamma and beta conidia
are not observed`, `Microconidiation, chlamydospores and sexual morph
not observed`. The annotator is labelling the *mention* of a
structure, and a negation mentions it.

This is the cheapest fix on the list and the only one that stops the
tail growing: a prompt instruction that absence statements do not mint
labels. It also explains the compound class — an absence sentence is
where several structures get named at once, which is why `Gamma and
beta conidia` exists and `Gamma conidia` (df 5+) already did.

## What follows

1. **Fix the prompt first.** Absence statements minting labels
   accounts for 39 singletons and manufactures compounds. It costs a
   prompt edit and it changes every future round.
2. **Case folding is free.** 36 singletons are case-only duplicates of
   labels that already exist, the same class the hand map already
   handles one entry at a time. This is rule-shaped, like the
   sexual/asexual family in `feature_label_rules`.
3. **The growth-condition family already has its mechanism** — the
   `context` field, built and backfilled 2026-09-01. 46 singletons are
   waiting for a consumer to use it.
4. **Property-of-trait and compound labels need a schema decision**,
   not a rule: is `Ascomata height` a label, an attribute of
   `Ascomata`, or nothing? 132 singletons hang on that answer, and it
   is the same base+context conflation the non-synonyms doc identified
   for media.
5. **Then decide on more annotation.** The coverage curve says a third
   thousand buys ~+0.9 points. This audit says the vocabulary it would
   extend is ~44 % consolidatable. Consolidating first makes the next
   thousand worth more; it does not make it urgent.
