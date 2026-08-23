# Feature labels that are NOT synonyms

Companion to
[`docs/feature_label_canonicalization.json`](feature_label_canonicalization.json).

That file records label pairs we **do** collapse.  This one records
pairs that look collapsible — they score high on string similarity,
or differ by one morpheme — and must be left alone.  Without it the
next person to run a similarity sweep rediscovers the same candidates
and has to re-derive why each was rejected, which is exactly how a
wrong merge gets made.

A wrong merge is close to invisible once applied: the annotations
still validate, the detector suite still passes, and the two
biological concepts are simply gone from the vocabulary.

## The canonicalization rule

Established 2026-08-17: **prefer the more idiomatic form, settled by
counting occurrences in the corpus.**

Two qualifications learned while applying it:

* **Count in canonical space, not raw.**  `Cultural characteristics`
  occurs **zero** times as a literal label — it is a target the map
  itself invented, carrying 32 annotations via four spellings.  Scored
  on raw strings it loses to `CultureCharacteristics` (3) and the
  established canonical would be overturned by an artifact.
* **A thin margin is not idiom.**  `Neck` (2) vs `Necks` (3) is a
  one-annotation difference.  Left unmapped deliberately; revisit when
  the corpus is larger.

Where counting was silent — two exact ties, `Generative
Hyphae`/`Generative hyphae` (1–1) and `Growth
Temperature`/`Growth temperature` (2–2) — the tie was broken toward
lowercase, the majority pattern across the other ~300 labels.

The rule replaced an earlier "prefer singular" proposal, which was
abandoned because it would have renamed the most frequent labels in
the corpus (`Spores` 192, `Lamellae` 124, `Conidia` 108, `Asci` 81,
`Basidia` 74) to forms no mycologist writes.  It also overrode a
"prefer `-carp` for family consistency" proposal for the fruiting-body
group; see below.

## Not synonyms — do not merge

### Paired opposites

| a | b | why they differ |
|---|---|---|
| `Sexual morph` (17) | `Asexual morph` (16) | Opposite states of the same fungus.  0.96 string similarity — the highest scoring pair in the corpus, and the most dangerous. |
| `Macroconidia` (3) | `Microconidia` (1) | Two conidium size classes, often both present in one species. |
| `Endoperidium` (1) | `Exoperidium` (1) | Inner vs outer peridium layer. |
| `Generative hyphae` (2) | `Vegetative hyphae` (4) | Distinct hyphal roles in a hyphal system. |
| `Primary branches` (2) | `Tertiary branches` (2) | Branching order. |

### Whole vs part, or layer vs layer

| a | b | why they differ |
|---|---|---|
| `Apothecium`→`Apothecia` (42) | `Hypothecium` (13) | Fruiting body vs the tissue layer beneath its hymenium.  Similar spelling, unrelated structures. |
| `Hymenium` (19) | `Subhymenium` (18), `Epihymenium` (3) | The spore-bearing layer vs the layers below and above it. |
| `Peridium` (20) | `Exoperidium` (1) | Whole vs outer layer. |
| `Cystidia` (33) | `Cheilocystidia` (16), `Pleurocystidia` (13), `Pileocystidia` (4), `Gloeocystidia` (12) | Position- and type-specific cystidia.  The general term is not a synonym for any specific one; collapsing loses where the cystidium sits. |
| `Cheilocystidia` (16) | `Cheilolamprocystidia` (2) | Different cystidium type at the same position. |
| `Pleurocystidia` (13) | `Pleurolamprocystidia` (2), `Pleuropseudocystidia` (2) | Likewise. |
| `Conidiomata` (50) | `Pycnidia` (15) | A pycnidium is a *specific kind* of conidioma.  Preserve the more specific term when the treatment uses it. |

### Same feature, different substrate or medium

`Culture on CMA` / `MEA` / `OA` / `PCA` / `PDA` / `DG18`,
`Colony on MEA` / `OA` / `PDA`, `Colony neotype on MEA` / `V8`,
`Asci in culture MEA` / `V8`, `Ascospores in culture MEA` / `V8`,
`Pseudothecium in culture MEA` / `V8`.

These score 0.83–0.93 against each other because only the medium
abbreviation differs — and the medium is the entire point of the
observation.  Never collapse this family.

### Merely similar

| a | b | why they differ |
|---|---|---|
| `Habit` (6) | `Habitat` (10) | Growth form vs where it grows. |
| `Basidioles` (2) | `Basidiomata` (37) | Immature basidia vs the fruiting body. |
| `Otosporoid spores` (1) | `Tricisporoid spores` (1) | Two spore ornamentation types. |
| `Outer excipulum` (1) | `Proper excipulum` (2) | Two excipulum regions. |
| `Perithecial wall` (1) | `Pseudothecial wall` (2) | Walls of two different ascoma types. |

## Deliberate asymmetries

* **`Conidiomata` (50) is left unmapped** while `Basidiocarp`/
  `Basidiome` collapse to `Basidiomata` and `Ascocarp` to `Ascomata`.
  There is no standard *conidiocarp* to collapse toward, and
  `conidioma`/`conidiomata` is already the accepted term.  The result
  is that all three families now sit on the `-mata` form, which is
  self-consistent even though it was reached by different routes.
* **The `-carp` forms lost.**  An earlier decision collapsed the group
  toward `Basidiocarp`/`Ascocarp`/`Sporocarp` on family-consistency
  grounds.  Counting reversed all three: `Basidiomata` (22) beats
  `Basidiocarp` (10), `Ascomata` (21) beats `Ascocarp` (2), and
  `Sporophore` (10) beats `Sporocarp` (8).  The reversal also agrees
  with the separate instruction to preserve whichever term the
  treatment itself uses rather than generalise the clade — the
  `-mata` forms are the clade-specific ones.
* **`Ascomatal wall` (4) and `Conidiomatal wall` (2) are untouched.**
  They are distinct anatomical features, not synonyms of anything, so
  they do not belong in the canonicalization map.  They are named off
  `ascoma`/`conidioma` and remain readable now that the parent labels
  are `Ascomata`/`Conidiomata`.
* **`Wall` (4), `Upper wall` (1), `Lower wall` (1)** look
  under-specified rather than drifted — the treatment presumably named
  a wall whose parent structure the annotator dropped.  A labelling
  question, not a synonymy one.

## Terms still undecided

* `Hypoderm` (4) vs `Hypodermium` (12) — plausibly the same layer, not
  yet confirmed against the source treatments.
* `Chemical reaction` (3) vs `Microchemical reactions` (2) — the
  micro- prefix may be a real methodological distinction.
* `Culture characters` (1) vs `Cultural characteristics` (32) — almost
  certainly the same thing, held back only because a single occurrence
  is thin evidence.

## What does NOT get a label at all

The rule above decides which labels collapse.  This section
records candidate labels that were **considered and declined**, so
the question is not reopened every round.

### Bibliographic citations — declined 2026-08-23

Inline literature citations (`Ju and Rogers (1996)`, `(Kornerup &
Wanscher 1978)`, `(de Hoog et al. 2000a, Najafzadeh et al. 2010a,
b)`) appear in **13.6 %** of descriptions and proved diagnostically
useful during round-4 review — they are the discriminator that
separates a compilatory genus entry from a genuine two-species
merge.  The operator asked whether they should get their own
annotation label.

**They should not.**  Four reasons, in order of weight:

1. **A regex finds them at least as well as a reviewer.**  Measured
   over 15 540 descriptions: clean matches, and **zero false
   positives** on ten adversarial anatomical probes —
   `(3–)3.5–5.5(–6) × (2–)2.5–4 µm`, `(av. 4.4 × 4.9 µm)`,
   `(n = 60/2)`, `(Fig. 117)`, `(holotype CBS H-8155)`,
   `(sub-)globose`.  Annotation time should go to what only human
   or LLM judgement can supply.
2. **Presence is not the signal; position is.**  A citation in a
   description is often perfectly legitimate — `(Kornerup &
   Wanscher 1978)` is the colour chart, and taxon_343eec40's
   `(in collection De Kesel 1979)` is an odour comparison in a
   *poster child*.  What discriminates is a citation *terminating a
   repeated-label group*.  A label would capture presence, which is
   the half that carries no information.
3. **It is not an anatomical feature.**  The features DB describes
   the organism; a citation describes the literature.
4. **Mid-round label changes are expensive.**  Rounds 1–3 (62
   treatments) and round 4 (47) are annotated against the current
   set.  Adding a label now splits the vocabulary across rounds and
   perturbs both the canonicalization map and the Heaps'-law
   analysis while three synonym pairs are still undecided.

The detection work belongs in `treatments_to_structured/triage_signals.py`
instead — see **D13** in
[`docs/data_quality_production_v4_model.md`](data_quality_production_v4_model.md).

**This would flip** if the goal were ever to *train* a model to
recognise citations rather than to detect them.  It is not; the
regex suffices.

## Maintenance

Rerun the similarity sweep after each bootstrap round.  Candidates
land in one of three places: the canonicalization map if they are
drift, this document if they are distinct, or the undecided list above
if the source treatments have not been checked.  An entry that stays
undecided across two rounds should be resolved rather than carried.
