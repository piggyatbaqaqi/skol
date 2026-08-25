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

### Clade-specific spore terms

Settled 2026-08-23 by an operator correction in round 4:
`taxon_7dfd35bd`'s spore span was labelled `Spores` over text
reading *Basidiospores*, and was corrected to `Basidiospores`.

| a | b | why they differ |
|---|---|---|
| `Spores` (192) | `Basidiospores`, `Ascospores` | The clade-specific term states how the spore is borne — on a basidium, inside an ascus.  Labelling either as `Spores` discards that, and it cannot be recovered from the label later. |

**The rule is the same one already applied to fruiting bodies**:
follow the term the treatment uses.  `Basidiocarp`/`Basidiome`
collapse to `Basidiomata` and `Ascocarp` to `Ascomata` — within a
clade — but `Basidiomata` and `Ascomata` were deliberately kept
apart.  Spores follow: normalise spelling variants within a term,
never across clades.

`Spores` remains correct where the treatment itself says *spores*,
which is common in older literature and in groups where the spore
type is not in question.  It is the most frequent label in the
corpus at 192, so this is not a rare correction — **five further
round-4 treatments still carry it over clade-specific text**:
`taxon_fd50457a` and `taxon_4b89d160` (*Ascospores*),
`taxon_d2d620ae`, `taxon_b673586a` and `taxon_5fe9223f`
(*Basidiospores*).

No canonicalization entry is involved: the map has no
`Basidiospores` key and `Spores` is not a target, so this was the
annotator choosing the general term directly.  The fix is in the
annotation, not the map.

#### Tie-break: when the heading and the content disagree, follow the content

Added 2026-08-24.  The five cases are **not** alike, and checking them
showed why.  Four carry no section heading at all — the text simply
reads `Basidiospores allantoid…` or `Ascospores…`, so labelling them
`Spores` generalises with no warrant from the source.

**`taxon_d2d620ae` is different.**  Its source carries a literal
section heading:

> `…smaller than basidia.Spores. Basidiospores allantoid, hyaline,
> thin-walled, smooth…`

So the annotator was **following the treatment's own heading**, which is
what the rule above asks for. The treatment uses *both* terms.

**The content term wins.**  Three reasons:

* a heading is a structural marker, not a statement about the spore;
* the content term is what the author reached for when actually
  describing the object;
* the rule exists to preserve clade information that *cannot be
  recovered from the label later* — and `Basidiospores` is right there
  in the text.

So `taxon_d2d620ae` still corrects to `Basidiospores`, but for a
different reason from the other four, and anyone re-deriving the rule
from that treatment alone would reasonably conclude the opposite.

*(Note the heading is only visible on inspection: the §15 element join
renders it `basidia.Spores.` with no space, so in brat it reads as
part of the preceding sentence.)*

### Clade-specific hyphal terms

Raised 2026-08-24 from `taxon_a3308621`, where `Fertile hyphae` was
relabelled `Generative hyphae` on the reasonable grounds that the
former is unfamiliar.  **Declined — they are not synonyms**, and the
corpus shows why.

| a | b | why they differ |
|---|---|---|
| `Fertile hyphae` (2) | `Generative hyphae` (2) | *Generative* is one member of the **monomitic/dimitic/trimitic hyphal-system scheme**, defined by contrast with *skeletal* and *binding* hyphae.  *Fertile* means "bearing conidiogenous cells" and belongs to anamorph morphology.  Different concepts from different clades' vocabularies. |

**Every corpus use bears this out.**  `Generative hyphae` and
`Generative Hyphae` occur only in basidiomycetes, always inside the
hyphal-system classification — *Tubulicrinis indicus* ("Generative
hyphae ≤4 µm wide, branched, septate, clamped" beside "Hyphal system
monomitic"), *Vararia lincangensis* ("Hyphal system dimitic,
generative hyphae…" beside a `Skeletal hyphae` label), and the same
pairing in *Podoscypha*, *Cyanosporus*, *Lyomyces*, *Nigroporus* and
*Mycorrhaphium*.  `Fertile hyphae` occurs only in anamorphic
ascomycetes, and in `taxon_a3308621` the fertile hyphae are the ones
"provided with a single, rarely two, lateral conidiogenous openings"
— i.e. they bear the phialides.

**This is the same shape as the spore correction above**, run the
other way: there the general term discarded clade information; here a
clade-specific term from the *wrong* clade would import information
that is not in the source.  Collapsing them would assert that an
anamorphic ascomycete has a hyphal system in the basidiomycete sense.

**If `Fertile hyphae` needs a canonical target, it is
`Conidiophores`** (27 treatments), which is what the structure does
here.  But there is a real argument for leaving it alone: the text
describes *sinuous, undifferentiated* hyphae with lateral openings,
and some authors use *fertile hyphae* precisely to avoid implying a
differentiated conidiophore.  The treatment already carries a separate
`Phialides` label, so the distinction is being made.  **Left undecided
— see "Terms still undecided".**

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

**Nor should the family be grown.**  Asked 2026-08-24 whether
`taxon_a3308621`'s `Colony` should become
`Cultural_characteristics_on_MEA`: **no**, for three reasons.

* **The medium is already captured.**  That span's text reads
  *"Colonies spreading rapidly, reaching 60–80 mm diam in 3 days at
  25 °C **on 2 % MEA**"*.  `source_text` holds it; putting it in the
  label duplicates data already stored, in the one place that cannot
  be queried structurally.
* **`Colony` is the canonical form** — the map already carries
  `Colonies` → `Colony`.  A medium-specific variant moves *off*
  canonical, into the family this section says never to collapse,
  which means it can never be normalised later either.
* **It is the base+context conflation that the label-schema work
  exists to fix.**  Six `Colony on …` and six `Culture on …` variants
  already exist because the label string carries both the feature and
  the observation condition.  The fix is a separate `context` field,
  not a longer label.

Keep the base term; let the medium live in the span text until the
schema has somewhere structured to put it.

**Worked example, and it is a clean treatment.**  `taxon_f7c117ed`
(*Diaporthe isoberliniae*, poster child
`species-emendation-correctly-attached`) carries **three** colony
accounts in one description — PDA, MEA and OA — with genuinely
different content: *grayed yellow (160C) ring* on PDA, *grayed yellow
(161A) with a white ring* on MEA, *grayed white (156A) with grayed
yellow margins* on OA.  Collapsing those to one `Colony` node loses
which medium produced which, and the media are the point of running
three plates.  The same treatment shows the pattern once more with
`Alpha` / `Beta` / `Gamma_conidia`.

This is what the family looks like when nothing has gone wrong: the
repetition is real, informative, and must survive into the structured
form as **context on one feature**, not as three features.

### Merely similar

| a | b | why they differ |
|---|---|---|
| `Habit` (6) | `Habitat` (10) | Growth form vs where it grows. |
| `Basidioles` (2) | `Basidiomata` (37) | Immature basidia vs the fruiting body. |
| `Otosporoid spores` (1) | `Tricisporoid spores` (1) | Two spore ornamentation types. |
| `Outer excipulum` (1) | `Proper excipulum` (2) | Two excipulum regions. |
| `Perithecial wall` (1) | `Pseudothecial wall` (2) | Walls of two different ascoma types. |

## Morph terms: technical collapses to plain

Settled 2026-08-23.  `Anamorph` → `Asexual morph` and
`Teleomorph` → `Sexual morph` are now in the canonicalization map.

**They are genuine synonyms**, with one nuance that does not block
the merge: `anamorph`/`teleomorph` also carry a *nomenclatural*
sense from pre-2011 dual nomenclature, where the asexual state
could be a separately named form-taxon (*Fusarium* as anamorph of
*Gibberella*).  `Asexual morph` is purely a morphological state.
For a **feature label** the distinction is immaterial — the label
marks "this span describes the asexual state", and the
nomenclatural claim lives in the span text.  Our one
`Anamorph`-labelled treatment reads `Anamorph: unnamed, Coremiella
or Oidiodendron like`; "unnamed" is nomenclatural, the label is
not.

**Direction settled by counting, both spaces agreeing:**

| | label space (candidate + hand) | text (20 474 treatments) |
|---|---:|---:|
| `Sexual morph` | 22 | 2 699 treatments |
| `Asexual morph` | 17 | 2 744 treatments |
| `Anamorph` | 6 | 940 treatments |
| `Teleomorph` | **0** | 645 treatments |

The plain forms lead ~3–4× in the literature and `Teleomorph` has
never once been emitted as a label here.

Two things to be honest about in that evidence:

* **The hand DB's zero for `Anamorph` is not a rejection.**  All
  six come from one treatment, `taxon_8d815304`, whose
  `reviewer_action` is `None` — never reviewed either way.
* **`Teleomorph` is a defensive key with zero occurrences.**  It
  violates the map's usual "keys are labels Claude has emitted"
  contract, added deliberately because the term is frequent in the
  source text (1 254 occurrences) and will eventually surface as a
  label.

**The operator's stated preference was for the technical forms**
and counting overrode it, as the rule requires.  Recorded so the
decision is not silently re-litigated.

**Spelling variants were *not* added.**  The source text carries
`teleomorphs`, `teleomorphic`, `teleomorphosis`, even a
`telemorphosis` typo — but none of these has ever appeared as a
label, and speculative keys are what the map's contract excludes.
Add them if and when the annotator emits them.

## Deliberate asymmetries

* **`Synanamorph` and `Holomorph` stay technical.**  They have
  **no plain-language equivalent**.  A synanamorph is a *second*
  asexual state of the same fungus — collapsing it to `Asexual
  morph` would erase the distinction that names it.  A holomorph is
  the whole fungus, both states together.  Neither has ever been
  emitted as a label, but both are common in the text
  (`synanamorph` in 121 treatments, `holomorph` in 64), so the
  question will arise.  Collapsing `Anamorph`/`Teleomorph` to the
  plain forms therefore does **not** retire the technical
  vocabulary wholesale.
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

* **`Fertile hyphae` → `Conidiophores`?**  Raised 2026-08-24.  Same
  structure functionally, but *fertile hyphae* may be a deliberate
  choice for undifferentiated fertile mycelium.  2 occurrences, both
  anamorphic ascomycetes.  **Not** `Generative hyphae` — see above.

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
