# Data quality observations — production_v4 model

Notes from a Phase 1 bootstrap-annotation sample of 5 treatments selected
via `bin/select_for_annotation --experiment production_v4 --n 5
--bands low:1,mid:2,high:2 --seed 1` on 2026-06-28.  Four of the five
exhibited issues serious enough to flag for later attention; this file
captures the categories with concrete evidence so future fix work
doesn't start from scratch.

Tracking: see the corresponding Trello item.

**Pairing convention (added 2026-07-05)**: every taxon referenced in
this memo should have a corresponding entry in
[`tests/fixtures/pathologies.json`](../tests/fixtures/pathologies.json)
capturing the actual treatment content plus labelled detector output.
That file is the machine-readable pathology catalog; this memo is the
narrative reasoning about it.  See
[`tests/fixtures/README.md`](../tests/fixtures/README.md) for the
schema and the walkthrough for adding new entries.

## Sample treatments

Source database:
`skol_exp_production_v4_02_00_treatments_prose` on puchpuchobs as of
2026-06-28.  Ingest-doc lookups for `pdf_url` cross-checks against
`skol_dev` performed in the same session.

| Short tag | Treatment `_id` | Synthetic? | Article |
|-----------|----------------|------------|---------|
| **T1** | `taxon_ba964a8b803eaf40672ba3561a79866d14054fe9ef993b4032161a8e05d3d55e` | yes | Ambele et al. 2020, *Journal of Fungi* — EPF biocontrol of termites |
| **T2** | `taxon_2114314b6d1bf58aa91b2b99bb30442e7dc30c5fb9bc4a17b107586482e983fd` | no | Fungal Planet 132 (Persoonia 2012) — *Calonectria pentaseptata* sp. nov. |
| **T3** | `taxon_22346900a8a1da8533cf8eed86a4ec07619320aa690696132a6c8514094320c2` | yes | Murrill, California Fungi protologue — *Gymnopilus laeticolor* + *G. ornatulus* |
| **T4** | `taxon_841d5cbed697b1882ba6b0f044556d801ae2df2f698fcc72c7a52bcb2349ce44` | yes | Tulloss et al. — *Amanita magniverrucata* revision |
| **T5** | `taxon_2b793602153da2c98370528e7950159efd9fec7a49d8a4fb79b35f678c3cf6a9` | no | Murrill, NA Flora — *Laccaria striatula* + Melanoleuca key |

The "synthetic?" column tracks `treatment.synthetic_nomenclature` —
True means the layout CRF found section paragraphs without a
preceding Nomenclature heading and the treatment-grouper inserted a
`"Nomen ignotum"` stub.  3 of 5 are synthetic.  T4 is especially
striking: it is unambiguously a real *Amanita magniverrucata*
treatment but its species heading was never extracted.

## Issues identified by operator

### 1. Taxonomic citation in the `description` field

**Symptom**: the bibliographic / nomenclatural heading text lands in
`description` rather than in `nomenclature` or its own slot.

**Semantic distinction — Description vs Diagnosis**: A
`description` field should describe the properties of ONE
specimen (this taxon's own anatomy).  It should NOT contain
formal taxonomic citations for other taxa — those are
extractor errors and reliable merge signals.

A `diagnosis` field, by contrast, is a comparative statement
about how this taxon differs from related taxa, and CAN
legitimately contain citations to related taxa for comparison
purposes (e.g., "differs from *X. yz* Author (Journal 42:
17) in having smaller spores").  A citation in `diagnosis`
is not necessarily an error; a citation in `description` is.

**Evidence**:

* **T3 / `taxon_22346900...`** — `description` begins:
  ```
  I. Gymnopilus laeticolor sp. nov.
  Pileus convex or somen-hat conic to subexpanded ...
  ```
  The numbered species heading (`I. Gymnopilus laeticolor sp. nov.`)
  should be Nomenclature.  Mid-`description` the text also contains:
  ```
  3. Gymnopilus ornatulus sp. nov.
  ```
  i.e. a SECOND species heading buried inside what's labelled as a
  single Description block.  **Follow-up 2026-07-03** (operator):
  in fact there are **3 (possibly complete) descriptions** —
  counting Pileus clauses gives three, though only positions 1
  (`I.`) and 3 have `sp. nov.` citations, suggesting species 2
  may be a redescription of an existing species.  Numbering
  itself is **mixed Roman (`I.`) and Arabic (`3.`)** — a
  detector-gap observation: my
  `_COUPLET_LINE_RE = ^\s*\d+[a-z]?[.)]\s+[A-Z]` matches the
  Arabic `3.` but slips past the Roman `I.` (n_key_couplets =
  1 not 2 for this treatment).  Refinement: allow either
  Arabic digits OR Roman numerals at the couplet-line start
  (`^\s*(?:\d+[a-z]?|[IVXLC]+)[.)]\s+[A-Z]`).
  **Round-1 review data**: piggy@puchpuchobs kept all 13
  Claude annotations, added 0, deleted 0 — Claude covered
  species 1 cleanly per §0 rule 3.  Perfect signal from both
  ends.
* **`taxon_2a9d07e6...`** — discovered 2026-07-01 in the
  50-treatment run.  Nomenclature is
  `Teratosphaeria dunnii Crous & Carnegie, Persoonia 42: 327.`
  (a real, non-synthetic citation).  Mid-`description` sits a
  full second citation:
  ```
  Teratosphaeria obscuris (P.A. Barber & T.I. Burgess)
      P.A. Barber & T.I. Burgess, Persoonia 23: 115. 2009.
  Diagnosis: Leaf spots primarily epiphyllous ...
  ```
  Followed by another full anatomical description block.
  Co-occurs with §6 (multi-species merge — same treatment
  contains both *T. dunnii* and *T. obscuris* descriptions).
  This citation would have parsed cleanly via gnparser
  (`http://localhost:9081`) as author+year+journal+page —
  strong signal for automatic detection.

* **`taxon_2f276bfa...`** — noted 2026-07-02.  **Sub-shape:
  full citation at the HEAD of Description, not
  mid-body.**  First line is a complete authored citation:
  `Mycovellosiel/a micranlhae (Muller & Chupp) Dianese &
  Furlanetto, …` (single OCR error: `l/a` for `lla` in the
  genus).  The description proper starts AFTER the
  citation, with a capital-letter opener and ends with a
  period — so once the head citation is stripped, the
  description content is clean and well-formed.  This is
  the "Nomenclature line landed at the top of Description"
  variant, distinct from the mid-body citation of
  taxon_2a9d07e6 and T3.  Detector implications: neither
  §10 `desc_starts_mid_sentence` (starts with a capital)
  nor `mid_body_description_header` (no `Description:`
  header) fires.  gnfinder would NOT catch this specific case — 2026-07-02
  testing confirmed the `/` mid-genus (`Mycovellosiel/a`)
  defeats both gnfinder and gnparser even with fuzzy
  options enabled.  This treatment falls in the "mid-word
  character-substitution" gap in the §6 idea #2 OCR
  tolerance measurements.  Cheaper regex signal that would
  catch it: "description first-line matches the shape
  `Genus[-alpha-] species[-alpha-] (Author) Author, …`" —
  a shape-based regex tolerating slashes/dots/digits
  inside the token positions where letters should be.

**Affected treatments**: T3, `taxon_2a9d07e6...`,
`taxon_2f276bfa...`.

**Likely stage** (best guess, not investigated): the layout CRF
labelled these short numbered heading lines as `Description`
continuations, OR the treatment-grouper failed to split on them.
Either way the symptom is downstream — the heading text never made
it to a Nomenclature slot.

**Cascade effect**: T3 was also flagged with a synthetic Nomenclature
stub, presumably because the first paragraph the grouper saw
already had `Description` label rather than `Nomenclature`.
`taxon_2a9d07e6` did NOT — the initial `Teratosphaeria dunnii`
citation reached Nomenclature correctly.  Only the second
species's citation slipped into Description.

**Detection idea**: scan every treatment's `description` field
with gnfinder (`http://localhost:9080`) and gnparser
(`http://localhost:9081`).  Any hit on an authored binomial
inside a Description is either an error (§1 case) or a
comparison mention worth flagging.  gnparser distinguishes bare
names ("Pileus lorem ipsum") from cited names ("*X.* Author,
Year"), so the false-positive rate should be low.  Complementary
to the merge-metric approach — catches the exact taxon_2a9d07e6
case that the metric missed (metric = 0).

### 2. Taxonomic citation not extracted at all

**Symptom**: the species name / formal author citation exists in the
source plaintext but appears in no semantically correct field of the
Treatment doc.

**Evidence**:

* **T1** — non-taxonomic article (EPF biocontrol study), no taxonomic
  citations exist; `synthetic_nomenclature: true` is correctly
  applied to a stray fragment but the Treatment shouldn't exist at
  all.  See §6 (false-positive treatments).
* **T2** — real *Calonectria pentaseptata* protologue.  The formal
  citation `Calonectria pentaseptata L. Lombard, M.J. Wingf., P.Q.
  Thu & Crous, sp. nov.` is in **`figure_captions`** rather than
  `nomenclature`.  `nomenclature` does contain `Calonectria
  pentaseptata` but it's a short bare name; the authority + nov.
  status went elsewhere.
* **T3** — see §1; species headings live in `description`, never
  reach Nomenclature.
* **T4** — clearly an *Amanita magniverrucata* treatment but
  `synthetic_nomenclature: true` and Nomenclature is the
  `Nomen ignotum` stub.  The species name "Amanita magniverrucata"
  appears throughout the text and in `figure_captions` ("Amanita
  magniverrucata, habit, mature specimens") but never made it to
  the Nomenclature field.
* **`taxon_acd88732...`** — discovered during 2026-07-01
  hand-inspection.  Real Perithecia-bearing treatment (probably
  Colletotrichum), 2 valid annotations (Perithecia + Spores)
  correctly produced by Claude.  `synthetic_nomenclature: true`,
  Nomenclature is `Nomen ignotum`; the species name / citation
  never reached the Nomenclature field.  Co-occurs with §10
  (description starts mid-sentence) — the description was
  clipped at the top, which is likely where the citation lived
  in the source plaintext.

**Nomenclature-vs-synth-nomen inconsistency subclass**
(2026-07-07): four treatments in batch-2 exhibit a
distinct data-quality bug — the `nomenclature` field is
EMPTY but `synthetic_nomenclature = False`.  Expected
behavior: synth flag should be True when the extractor
couldn't identify a nomenclature.  Observed cases:

  * `taxon_9b787247...` — Rhizogene Syd. gen. nov. from
    a 1920 German paper.  Nomenclature line clearly
    present in source ("Rhizogene Syd. nov. gen.") but
    not extracted to the `nomenclature` field.
  * `taxon_c9181340...` — Materials_examined-leak case.
    Nomenclature clearly present in source (complete
    conidial fungus treatment) but not extracted.
  * `taxon_fd4323fb...` — smut-fungus treatment with
    both-ends-clipped Description and Differential
    Diagnosis in Diagnosis field.  Nomenclature not
    extracted.
  * `taxon_3e98d44d...` — Gaillardinia gen. nov. (yeast
    genus).  Description is a clean anatomical
    paragraph, but Nomenclature "Gaillardinia Q.M. Wang,
    Yurkov, Boekhout & F.Y. Bai, gen. nov."  absent.

**Four occurrences in the first 7 batch-2 treatments
reviewed** (57% incidence) confirms this is systematic,
not isolated.  Warrants its own Trello card for the
extractor-side fix.  Distinct from the §2 primary cases
where synth flag CORRECTLY fires (T4,
taxon_acd88732).

**Affected treatments**: T1 (vacuously), T2, T3, T4,
`taxon_acd88732...`.  Nomenclature-vs-synth inconsistency
subclass: `taxon_9b787247...`, `taxon_c9181340...`,
`taxon_fd4323fb...`, `taxon_3e98d44d...`.

**Likely stage**: layout CRF likely labels formal-citation paragraphs
as `Figure-caption` (T2) or misses them entirely (T4).  Where the
species heading IS labelled, the treatment-grouper's
Nomenclature-recognition rule may be too narrow (T3, T4).

**Severity**: high — without correct Nomenclature, downstream taxon
identification, name-resolution lookups, and per-species aggregation
all fail or fall back to `synthetic_nomenclature`-flagged stubs.

### 2.5. Old-orthography citations are invisible to gnfinder

**Observation** (operator, 2026-08-23, on taxon_66c1e6e3):
its citations use an abbreviated genus and the pre-1960s
convention of **capitalizing species epithets derived from
personal names** — `D. Ellisii`, `D. Harperi`.

Tested against the live gnfinder, the capitalized epithet is
the part that breaks detection, and the abbreviation
compounds it:

| form | gnfinder returns |
|---|---|
| `D. subochraceus` | `D. subochraceus` ✅ |
| `Dacryomyces ellisii` | `Dacryomyces ellisii` ✅ |
| `Dacryomyces Ellisii` | `Dacryomyces` — **genus only** |
| `D. Ellisii` | **nothing** |

So an abbreviated genus alone is fine.  A capitalized
epithet costs the species half.  Both together are
completely invisible.

**This is not a gnfinder defect.**  `D. Ellisii` is
orthographically identical to an author's initial plus
surname — the corpus is full of `P. Chaverri`, `Y. Chai`,
`N. Maryani` — so the form is genuinely ambiguous and
rejecting it is defensible.  Any detector that wants these
names has to disambiguate with context gnfinder does not
have, the obvious one being *the same genus spelled out
elsewhere in the same treatment*.

**Scale is uncertain and should not be quoted precisely.**
A regex for `Genus Epithetii` over 25 000 treatments yields
1 826 distinct candidate strings, but roughly half are
author names and journal titles (`Index Fungorum`, `Sylloge
Fungorum`, `Annales Mycologici`).  Classifying a 120-string
sample by lowercasing and re-querying gnfinder splits about
48 / 52 epithet vs other — but that test is itself leaky
(`Cuban Kungi` lowercases into a "name"), so treat it as an
order-of-magnitude signal only.  Verified real cases include
`Linospora Tremulae`, `Orbilia Cunninghamii`, `Linobolus
Ramosii`, `Mycosphaerella Oxyacanthae`.

**The `-ii` ending is diagnostic; `-i` is not** (operator,
2026-08-23: "'Ellisii' can be recognized as a probable
species epithet due to the grammatical ending.
Unfortunately, this does not work for Harperi — one needs to
know that Harper is an English surname and Harperi is not an
Italian surname.").  Measured over 25 000 treatments, taking
capitalized candidates and classifying each by lowercasing
and re-querying gnfinder:

| ending | distinct candidates | resolve as a binomial |
|---|---:|---:|
| **`-ii`** | 297 | **94 %** (66/70 sampled) |
| `-i` | 978 | 44 % (31/70 sampled) |

The round-trip test is leaky, so the absolute figures are
soft — but both buckets were measured the same leaky way, so
the **contrast** is trustworthy.  `-ii` is the Latin genitive
of a consonant-final name (*Ellis* → *Ellisii*), an ending
that is rare in ordinary Latin and rarer still in surnames.
`-i` collides with Italian and Japanese surnames, and the
misses in that bucket are exactly that: `Y. Otani`,
`Jphialoplwm Borelli`.

The sharpest case is *Petri*, which is simultaneously an
Italian surname and the Latin genitive of *Petrus*:

```
L. Petri        -> []          Lionello Petri -> []
Boletus Petri   -> []          Boletus petri  -> Boletus petri
```

**Is a full Latin suffix list worth compiling?  No — three
suffixes cover it** (operator, 2026-08-23).  Measured the
same way over 25 000 treatments:

| suffix | distinct capitalized | resolve as binomial | verdict |
|---|---:|---:|---|
| **`-ii`** | 277 | **94 %** | use |
| **`-ian*`** (`-ianus/-iana/-ianum`) | 74 | **88 %** | use |
| **`-iae`** | 99 | **85 %** | use |
| `-ae` | 248 | 60 % | reject |
| `-i` | 1 048 | 44 % | needs context |
| `-orum` | 33 | 27 % | reject |

All three keepers are **personal-name forms** — `-ii`
masculine genitive, `-iae` feminine genitive (the feminine
counterpart recalled by the operator), `-ian*` adjectival.
That is not a coincidence: the old convention capitalized
epithets derived from *persons* and *former generic names*,
not from anything else.

**And that is why the locality suffix does not need
detecting.**  `-ensis` is indeed almost never a surname, but
it is moot here: only **1.2 %** of `-ensis` epithets in the
corpus are capitalized at all (26 against 2 169 lowercase),
and `-ense` only 0.6 %.  Geographic epithets were not
capitalized under the convention, so they never enter this
problem.

The rejected suffixes fail for instructive reasons, each a
different collision:

* `-ae` is the ordinary first-declension plural, so it
  matches anatomy on every page — `Sporae`, `Lamellae`,
  `Hyphae` — as well as `Academiae`.
* `-ian*` misses are **toponyms**: `Louisiana`, `British
  Guiana`, `Reggiana`.  Still 88 %, but the failure mode is
  geography rather than surnames.
* `-iae` misses are Latin genitives of non-persons:
  `Lithuaniae`, `Academiae`, and subsection names like
  `Villadiniae`.
* `-orum` is dominated by book titles — `Index Herbariorum`,
  `Prosyllabus Tracheophytorum`.

**Practical rule for D10 and D12**: treat a capitalized
`-ii`, `-iae` or `-ian*` epithet as a probable binomial
half; treat `-i` as unresolved and fall back to context —
most usefully, whether the abbreviated genus is spelled out
in full elsewhere in the same treatment.  The three reliable
endings together cover about **450 of the ~1 826**
capitalized candidates, so this recovers roughly a quarter
of them cheaply; the `-i` bucket is the bulk and needs the
genus-context check regardless.

### Recovering these names: lowercase round-trip + genus expansion

Operator, 2026-08-23: "If we see capitalized words with any
of these endings can we make them lowercase and throw
gnfinder at them?"  Yes — measured, it works, **but not on
its own**.

**Step 1, the round-trip, recovers the names.**

```
D. Ellisii          -> []            lowered: D. ellisii          ✅
D. Harperi          -> []            lowered: D. harperi          ✅
Dacryomyces Ellisii -> Dacryomyces   lowered: Dacryomyces ellisii ✅
```

Both of taxon_66c1e6e3's citations come back, `-i` included
— so this recovers more than the suffix rule alone predicts.

**Step 2 is mandatory, because step 1 alone manufactures
false binomials from author names.**  gnfinder cannot
validate a single-letter genus, so almost any `Initial.
word` is accepted once lowercased:

```
Y. Otani  -> Y. otani     L. Petri   -> L. petri
A. Borelli-> A. borelli   M. Rossii  -> M. rossii
```

Only `P. Chaverri` refused.  Note **`M. Rossii` ends in
`-ii` and still false-positives** — so the suffix filter,
which is 94 % precise on full genera, does *not* protect the
abbreviated case.  A raw round-trip would hand D10 a
hallucinated binomial to compare against, which is worse
than the blindness it fixes.

**The discriminator is genus expansion**: require the
initial to expand to a full genus named somewhere in the
same treatment.  Verified on taxon_66c1e6e3, where gnfinder
finds *Coryne, Dacryomitra, Dacryomyces, Dacryopsis,
Ditiola, Exidia, Tremella* elsewhere in the document:

| candidate | lowered resolves | `X.` expands to | verdict |
|---|---|---|---|
| `D. Ellisii` | ✅ | Dacryomitra, Dacryomyces, Dacryopsis, Ditiola | **accept** |
| `D. Harperi` | ✅ | (same) | **accept** |
| `Y. State` | ✗ | nothing | reject |

**Two honest limits.**

* The expansion is often **ambiguous** — `D.` matches four
  genera here.  So the output is "a binomial in one of
  these", not a resolved name.  That is still enough for
  D10, which only needs to know whether the description's
  genus *can* be the nomenclature's.
* It is **not airtight**: in a treatment about *Mycena*, an
  author cited as `M. Rossii` would find `M.` → *Mycena* and
  be wrongly accepted.  Combining both filters — an `-ii`,
  `-iae` or `-ian*` ending **and** a genus expansion —
  narrows that, but a paper about *Mycena* citing
  M. Someoneii remains a false positive.  Expect precision,
  not certainty.

**This belongs outside the mycology code.**  Nothing in it
is fungal: it is nomenclatural orthography plus gnfinder, so
it fits `treatments_to_structured/gn_client.py` or the
`gnservices` component earmarked for extraction in
`docs/skol-repo-split-and-packaging.md`, and would be useful
to anyone parsing pre-1960s botanical or zoological
literature.

**Consequences.**

* **D10 is blinder than its entry claims.**  It compares
  genera named in the description against the nomenclature,
  and several fixture cases were dismissed as "names no
  binomial".  In old literature the description may name
  binomials that gnfinder cannot see, so D10 will
  under-report on exactly the pre-1960s material where §6
  merges are most common.
* **D12 must not lean on gnfinder alone** to recognise a
  nomenclature-shaped span, for the same reason.
* This compounds §15 and the §9 OCR modes: the older the
  source, the more ways its names are unreadable.

### 3. Biology and Materials-examined confusion

**Symptom**: content that should be `materials_examined` (specimen
collection records — date, locality, collector, herbarium accession)
lands in `biology` (habitat / distribution context), or vice versa.

**Evidence in the sample is weaker than expected; what we *did* see**:

* **T1** — `biology` contains ~5,000 characters of the article's
  Results+Discussion sections (termite-mortality assays, citations
  to [8] Rath 2000, etc.).  This is article body, NOT taxonomic
  biology.  Closest match to the symptom: "biology" being used as
  a catch-all for unclassified prose.
* **T5** — `materials_examined` contains a genus-level taxonomic
  citation: `MELANOLEUCA Pat. Tax. Hymen. 159. 1900.`  This is a
  generic-name citation belonging in Nomenclature for a Melanoleuca
  treatment, not in the *Laccaria striatula* treatment's
  materials_examined.  See §7 (multi-species merge).
* **T4** — `biology` correctly contains "ECOLOGY: Solitary to
  gregarious, ... coastal forests, with *Pinus muricata* ..." which
  is the right kind of content.  `materials_examined` correctly
  contains "MATERIAL EXAMINED: USA: CALIFORNIA — Alameda Co. ...".
  T4 shows the fields working as designed.

**Affected treatments**: T1, T5 (each in different failure modes —
not the clean swap originally hypothesized).

**Reframe**: the cleaner statement of this issue is that
`biology` / `materials_examined` / `notes` / `key` accumulate
*overflow content* the layout CRF couldn't classify confidently.
T1's "biology" really is "prose the model didn't recognize as
non-treatment article body."  Worth treating these four catch-all
fields as a single issue category.

**Likely stage** (best guess): layout CRF section-label confusion at
section-heading paragraphs that don't have visually-distinctive
formatting, plus a tendency to over-emit `biology`/`notes` labels
for ambiguous prose in non-treatment papers.

### 4. `pdf_url` null in Treatment, set correctly in skol_dev

**Symptom**: `treatment.ingest.pdf_url` is `null` even though the
corresponding `skol_dev` ingest document carries a non-null
`pdf_url`.

**Evidence — confirmed across all 3 we cross-checked**:

| Treatment | `treatment.ingest.pdf_url` | `skol_dev.pdf_url` |
|-----------|---------------------------|--------------------|
| T1 | `null` | `https://doi.org/10.3390/jof6030126` |
| T2 | `null` | `https://www.ingentaconnect.com/contentone/wfbi/pimj/2012/00000029/00000001/art00011?crawler=true` |
| T3 | `null` | `https://mykoweb.com/CAF/protologue/Gumnopilus_decoratus.pdf` |

Same pattern holds for `url`, `xml_url`, `doi`, `human_url` —
all `null` in the Treatment's `ingest` sub-doc, all populated in
`skol_dev`.

**Affected treatments**: T1, T2, T3 confirmed; T4 and T5 not
cross-checked but `ingest.pdf_url: null` is present in those too,
so the pattern is universal across this sample.

**Likely stage**: the ingest-doc projection inside
`bin/extract_treatments_to_couchdb.py` — the projection seems to
copy only the `_id` and `db_name` fields and leaves the URL/DOI
fields at their null defaults.  Probably mechanical and fixable
in one place.

**Severity**: low-to-medium — operators looking at a Treatment
can't follow back to the source PDF directly; need a separate
`skol_dev` lookup.  Doesn't affect ML training but hurts review
workflows (including the Phase 1 brat-review workflow this sample
was selected for).

## Additional issues surfaced by the sample

These weren't in the operator's original list but are obvious in
the data.

### 5. False-positive treatments from non-taxonomic papers

> **Scope note, 2026-08-24.**  The heading understates it: a false
> positive does **not** require a non-taxonomic paper.
> `taxon_9446b102` is a synthetic treatment assembled from the front
> matter of a paper that validly describes three new species — and it
> sits alongside a *correct* treatment extracted from the same
> document.  A document-level "is this a taxonomic paper?" gate, the
> obvious fix for the cases below, would not catch it.  See the
> sub-section at the end of this section.


**Symptom**: a Treatment record is created for a paper that contains
no actual taxonomic treatments.

**Evidence**:

* **T1** — Ambele et al. 2020 is a biocontrol study of
  entomopathogenic fungi against termites attacking cocoa.  It
  contains no `sp. nov.` descriptions, no formal type designations.
  But the extraction pipeline emitted a Treatment with a 187-char
  fragment from the Results section as `description`, ~5,000
  characters of Discussion as `biology`, and bibliography
  fragments in `figure_captions`.  The Treatment is genuinely a
  false positive.
* **`taxon_fb7bd18d...`** — noted 2026-07-02.  Source paper:
  "Paraphysoderma sedebokerense GlnS III Is Essential for the
  Infection of Its Host Haematococcus lacustris" (*Journal of
  Fungi*, DOI 10.3390/jof8060561).  Experimental gene-function
  / infection study, not a species description — no
  `sp. nov.`.  The extracted `description` is verbatim
  Results/Discussion prose: `is clear from the comparison of
  the control, featuring a totally collapsed algal culture
  (Figure 2C) compared with the culture treated with
  glufosinate remaining green ...`.
  **Self-quarantining behaviour confirmed**: Claude returned
  0 annotations on the non-taxonomic prose (correct behaviour),
  and `bin/brat_export` iterates the candidate DB, so the
  treatment doesn't appear in the review directory at all.
  §5 false positives silently drop out of the reviewer queue
  when Claude finds no anatomical features — same mechanism
  that filters the §9 taxon_cda95f9f case.  Detection: the
  `annotation_count = 0` field in the status doc is a
  post-hoc signal for §5 candidates; combined with
  `synthetic_nomenclature = True` (§2 flag) it's a strong
  post-bootstrap marker for corpus-cleanup work.
  **Conditional, not general** — see taxon_0a8c1077 below,
  where the orphan paragraph *is* morphological, draws 5
  annotations, and reaches the reviewer queue.

* **`taxon_0a8c1077...`** — noted 2026-08-21 from round-4.
  **Breaks both assumptions above.**  Source: Haelewaters
  *et al.* 2020, "Red yeasts from leaf surfaces and other
  habitats: three new species and a new combination of
  *Symmetrospora*", *Fungal Syst. Evol.* 5: 187–196
  (doi 10.3114/fuse.2020.05.12).  The extracted
  `description` is the Introduction's summary paragraph
  about seven **previously-described** species: `All seven
  species mentioned above form smooth, butyrous, somewhat
  shiny colonies on agar medium. The colonies produce
  entire margins and colony color varies from pink to
  brick-red. None of these species have been observed to
  form hyphae or pseudohyphae, but most of them do form
  ballistoconidia (Hamamoto et al. 2011, Sampaio 2011).
  These characters are ` — front matter, at article.txt
  lines 60–65 of 1348, char 3154 of 55 036 (~6 % in).
  Operator: "a truncated summary Notes section … that
  discusses 7 related species.  This should not have been
  extracted as a description."

  1. **The paper is a legitimate taxonomic paper.**  It
     formally describes three new species and a new
     combination, so it has real Nomenclature headings
     throughout.  The fix angle proposed below — gate stub
     creation on the source paper having at least one real
     Nomenclature heading anywhere — **would not catch
     this**.  Nor would the `skol_dev.taxonomy` flag.  The
     orphan paragraph is in a taxonomic paper; it just
     isn't a treatment.
  2. **It did not self-quarantine.**  taxon_fb7bd18d drew
     0 annotations and so never reached the review
     directory.  This one drew **5** — `Colony`,
     `Colony_color`, `Colony_margin`, `Hyphae`,
     `Ballistoconidia` — because a summary of seven
     species' colony morphology is written in perfectly
     genuine morphological language.  So
     `annotation_count = 0` is **not** a general post-hoc
     §5 marker; it works only where the orphan paragraph
     is non-morphological.  This one landed in a reviewer's
     queue and had to be caught by eye.

  **Tail-clip mechanism identified.**  `These characters
  are ` is cut at the bottom of page 1's column, and the
  next line of `article.txt` is the article's own masthead
  (title / authors / affiliations / abstract, lines 66–89),
  which the linearizer emitted *after* the body column.
  `article.page-headers.json` does not cover lines 66–73 —
  its nearest regions are 97–98 and 106–110 — so the
  header-stripping pass missed the masthead entirely.  The
  sentence's continuation appears nowhere in the extracted
  text (`These characters are` occurs exactly once), so
  this is real content loss at a page boundary, not merely
  reordering.

  **Flags**: `§2:synth_nomen` (treatment is `Nomen
  ignotum`), `§10:tail_clip`, `§13:no_source_anchor`
  (`source_anchors` is `[]`; `nomenclature_spans` holds a
  degenerate all-zero span).  **None of the three names the
  actual defect.**  Multi-species summary prose posing as a
  description has no detector.  The plural, anaphoric
  subject is the strongest available signal — "All seven
  species", "None of these species", "most of them" —
  and nothing looks for it.  **Proposed detector**: flag a
  description whose first sentence has a plural or
  anaphoric subject and contains no binomial, weighted
  higher when `synthetic_nomenclature` is True.  Tracked as
  **D1** in the Detector backlog, where the reference set
  rules out the plural-subject and no-binomial formulations
  and leaves anaphora as the usable signal.

**Affected treatments**: T1, `taxon_fb7bd18d...`,
`taxon_0a8c1077...`.

**Likely stage**: the treatment-grouper's
`synthetic_nomenclature` fallback (`treatment.py:360-366`) creates
a "Nomen ignotum" stub for any orphan section paragraph in
`Look for Nomenclatures` state.  This was added to capture
nameless treatments in legitimate taxonomic papers — it
over-fires on non-taxonomic papers where stray
`Description`/`Biology`-labelled paragraphs appear in the body.

**Severity**: medium — pollutes the corpus with noise treatments
that will land in any random sample (1 of 5 here is a startling
hit rate).  Probably the single biggest contributor to the
complexity-score=0 cohort the operator noted earlier.

**Possible fix angle**: gate `synthetic_nomenclature` stub
creation on a stronger signal (e.g., the source paper has at least
one *real* Nomenclature heading anywhere; OR the article-level
`skol_dev.taxonomy: true` field is set — though that flag is
itself set by an upstream heuristic).

**Insufficient on its own**: taxon_0a8c1077's source paper
describes three new species and carries real Nomenclature
headings throughout, so both proposed gates pass and the stub
is still created.  Catching that case needs a signal about the
*paragraph*, not the *paper* — see the proposed plural-subject
detector in its entry above.

#### 5.1 A front-matter treatment beside a correct one

`taxon_9446b102`, from round 4 (operator: *"truncated above and below
and does not look like an actual description"* — it is not a
description at all).

Source: **Persoonia 27 (2011)**, *"Stem cankers on sunflower
(Helianthus annuus) in Australia reveal a complex of pathogenic
Diaporthe"*, doi `10.3767/003158511X617110` — a real taxonomic paper
validly describing *D. gulyae*, *D. kochmanii* and *D. kongii* sp.
nov.

Every field holds a different piece of the paper's apparatus:

| field | paragraph | what it actually is |
|---|---|---|
| `description[0]` | 5 | introduction morphology, clipped both ends — ends mid-citation at `(Wehmeyer ` |
| `description[1]` | 63 | **table footnotes + `Table 1` caption** |
| `diagnosis` | 21 | the paper's **abstract**, head-clipped at `novel species.` |
| `biology` | 9, 13 | the paper's **introduction** |
| `notes` | 227 | more table footnotes + `Table 2` caption |

**The same document also yields a correct treatment.** Three come out
of it: this one (synthetic, paras 5–227), `taxon_bcebed1d` (synthetic,
para 261), and **`taxon_bab7c442`** — `synthetic_nomenclature: false`,
a 6 243-char description, paras 265–285 — the genuine *Diaporthe*
treatment. So this is the **"mixed source document"** half of the
empty-description finding: 64.1 % of empty-description treatments come
from documents that *do* produce descriptions. The paper is not the
problem; **the grouper opened a treatment before the first
nomenclature.**

**Four structural tells, none needing any text analysis:**

* `nomenclature_spans` is `[(para 5, char 0, char 0)]` — a
  **zero-length span at offset 0**. There is no nomenclature; the
  anchor is degenerate.
* `line_number` is `0` while the content spans paragraphs 5 → 227.
* `synthetic_nomenclature` is true and `treatment` is `Nomen ignotum`.
* it spans **222 paragraphs** — essentially the whole paper — where its
  well-formed sibling spans 20.

**The detectors are not blind here**, which is the interesting part.
Five flags fire — `§2:synth_nomen`, `§10:mid_sentence`,
`§10:diag_head_clip`, `§12:desc_span_gap`, `§13:no_source_anchor` — at
`merge_metric` 1. What is missing is anything that acts on the
*combination*: synthetic nomenclature **plus** a zero-length
nomenclature span **plus** a 222-paragraph spread is not a treatment
worth annotating, and it reached a reviewer regardless. That is a
cheaper gate than any new text detector, and it belongs upstream of the
annotator rather than in triage.

#### 5.2 Monographic books, and a correction to the source-class story

`taxon_a5efbd0b`, round 4 (operator: *"truncated almost to nothing …
the labels are fine, but the extraction is useless"*).  The whole
treatment is a **35-character description** — `is strongly rugulose or
papillate.` — plus a 103-char diagnosis.

The prose is not merely clipped, it is **shredded into alternating
labels**:

| offset | label | text |
|---|---|---|
| 424 499 | `Description` | `is strongly rugulose or papillate.` |
| 424 551 | `Misc-exposition` | `Two species occur in Java and have been well illustrated…` |
| 424 749 | `Table` | `Fig. 33 JANSIA ELEGANS…` |
| 424 908 | `Misc-exposition` | `phalloid, which is common in Java.` |
| 424 964 | `Diagnosis` | `The short gleba-bearing portion…` |
| 425 082 | `Misc-exposition` | `…the only species of Jansia that is com…` |

**The taxon is recoverable only from the dropped blocks.** *Jansia* is
a **phalloid**, not a puffball — `gleba` plus a distinct stipe reads
gasteroid, but the surrounding `Misc-exposition` says *"phalloid, which
is common in Java"* and the `Table` names *Jansia elegans*. The
extraction discarded precisely the context needed to identify what it
had. It also carries the **zero-length nomenclature span at offset 0**
seen on `taxon_9446b102` — two instances now, so it is a reliable tell.

The source is **C.G. Lloyd's *Mycological writings* Vol. III
1909–1912** — a book: title and year only, no journal, volume or DOI.
It yields **120 treatments, 90 % synthetic, 61 % with empty
descriptions**.

##### The correction

Measured 2026-08-24, ingest documents classify cleanly by metadata, and
the result does **not** support "whole-volume ingests are the problem
class":

| class | docs | treatments | synthetic | empty desc |
|---|---:|---:|---:|---:|
| per-article (journal + title/doi) | 30 484 | 62 427 | **39.6 %** | **49.2 %** |
| book-like (title, no journal) | 365 | 10 656 | 31.6 % | 51.1 % |
| whole-volume (journal, no title/doi) | 235 | 6 709 | 34.8 % | **33.2 %** |

Whole-volume is the **best** of the three on empty descriptions and
below per-article on synthetic rate. The earlier §5 framing implied
that class was the culprit; on these numbers it is not, and #404's case
rests on OCR quality and per-article structure rather than on synthetic
rates.

**The real signal is per-document, not per-class.** Across 242
book-like documents yielding ≥ 10 treatments the synthetic rate has
median 27.3 % and p90 56.2 % — but a small cluster is catastrophic:

| synthetic | n | document |
|---:|---:|---|
| 97.7 % | 44 | *Our Edible Toadstools and Mushrooms* |
| 96.9 % | 259 | *The Agaricaceae of Michigan* |
| 96.7 % | 60 | *Researches on Fungi, Volume 1* |
| 90.5 % | 63 | *One Thousand American Fungi* |
| **90.0 %** | **120** | ***Mycological writings Vol. III*** |
| 88.3 % | 128 | *Mycological writings Vol. II* |
| 84.8 % | 138 | *Mycological writings Vol. IV* |

Three Lloyd volumes at 84–90 %, 386 treatments between them, plus
Kauffman, Buller and McIlvaine. These are **early-20th-century
monographic and popular books** whose prose does not follow modern
treatment structure — and one modern review (*"The emerging role of
Fungi in sustainable farming"*, 88.9 %) sits in the same band, so the
common factor is **non-treatment-structured prose**, not age.

**Consequence for sequencing.** A per-document gate is cheaper and
better targeted than a per-class one: the top eight documents alone
account for ~715 treatments at ~90 % synthetic. Excluding a named
handful of books would remove more noise than any metadata-class rule,
and it is a filter, not a model.

### 5.3 Nomenclature quality — an axis nobody had measured

Raised by the operator 2026-08-25: *"we haven't really been
concentrating on Nomenclature tags."*  Correct, and measuring it
produced one correction and one finding.

#### Correction: the zero-length nomenclature span is not a signal

Five entries recorded during round-4 review note "the *N*th instance of
the zero-length nomenclature span at offset 0", as though it were a
rare diagnostic tell. **It is not.**

| | count | share |
|---|---:|---:|
| treatments | 81 527 | — |
| `synthetic_nomenclature: true` | 30 520 | 37.4 % |
| zero-length nomenclature span | **30 520** | **37.4 %** |

The two are **exactly coextensive**. A zero-length span at offset 0 is
simply *how* a synthetic nomenclature is represented — there is no
source text to point at — so it carries no information beyond
`§2:synth_nomen`, which already fires. Those five fixture notes should
be read as "this treatment has synthetic nomenclature", nothing more.

#### Finding: 27 % of real nomenclature fields do not begin with a name

Of the **51 007** treatments whose nomenclature is *not* synthetic,
**13 965 (27.4 %)** do not start with anything name-shaped, after
allowing for leading enumeration like `57. `:

| class | count |
|---|---:|
| other (largely head-clipped — see below) | 8 154 |
| **starts lowercase** — head-clipped outright | 4 612 |
| very long (> 400 chars) | 965 |
| running head / journal line | 179 |
| figure caption | 48 |
| collection date | 7 |

**Head-clipping dominates.** The characteristic shape keeps the
authorship and the nomenclatural act while losing the name itself:

* `Crous, sp. nov. MycoBank MB500690. Fig. 15A–W. Anamorph: Phaeoacremonium…`
* `Zamora, comb. nov. MycoBank MB 844755. Figs 2A, 3. Basionym: Dacryopsis…`

Both are unusable as names while looking superficially like citations.
Other classes are simply the wrong content — `Keywords: Trichophyton
spp; Microsporum spp;…`, `Exsicc. M. March. 1928,—Krieg. F. Sax. 234.`,
`Typus. C. cinnamomeus (L.: Fr.) Gray.` — and one field runs to
**63 590 characters**.

**Why this matters more than the count suggests.** Every downstream
join is on the name: gnfinder resolution, D10's genus comparison, D14's
rank check, deduplication, and the search product itself. A
head-clipped nomenclature is not a cosmetic loss — it removes the key.

**And it is detectable cheaply.** "Does the nomenclature start with a
capitalised binomial, a uninomial with a rank-bearing suffix, or an
all-caps genus?" is a regex over a field already extracted, with the
enumeration prefix stripped first. The measurement above *is* the
detector; what it needs is calibration against the OCR-damaged true
positives (`ThueJDeaella hirsuta`, a real name mangled) that currently
inflate the rate.

### 5.4 What the 35 482 empty treatments actually are (T3d)

Measured 2026-08-25. **p2b** — the treatments with `complexity_score`
0, i.e. no `description` and no `diagnosis` — is 35 482 of 81 527, and
until now nothing had characterised it.

Classifying every one into exactly one bucket, priority-ordered so the
partition is total. The sum is the check, and it holds:

| class | all-empty docs | mixed docs | total | degenerate anchor |
|---|---:|---:|---:|---:|
| A — no prose at all | 0 | 0 | **0** | 0 |
| B — `key` only | 5 741 | 1 664 | 7 405 | 0 |
| C — only `biology` / `figure_captions` | 2 747 | 5 588 | 8 335 | **5 594** |
| D — specimens + a real name | 554 | 5 512 | 6 066 | 0 |
| E — synthetic or unnamed | 2 781 | 4 801 | 7 582 | **7 433** |
| F — other | 618 | 5 476 | 6 094 | 0 |
| | | | **35 482** | 13 027 |

**Class A is empty, which was not expected.** The predicted
"extraction failure — all prose fields null, trace it to the layout
CRF" class **does not occur**: every one of the 35 482 carries *some*
prose. Whatever is wrong, it is not that the extractor produced
nothing.

**Roughly 45 % are legitimate.** `key`-only entries (7 405) are
dichotomous keys and specimens-plus-a-real-name (6 066) are
nomenclature-only entries — both are correct extractions of material
that simply has no description to find.

**The degenerate anchor is real, large, and perfectly segregated.**
`line_number == 0` while the treatment's spans start well into the
document is **13 027 treatments (36.7 %)** — and it occurs in classes
C and E and **nowhere else**, at 67 % and 98 % of those classes
respectively. Zero in B, D and F. A signal that clean on a partition
built from unrelated fields is worth building on: it separates the two
pathology classes from the three benign ones by itself.

#### F4's test is vacuous, and the hypothesis fails on a valid one

The plan proposed a free test: *compute `n_terms_above_5` over all
35 482; if p2b's merge rate is elevated versus p1, merges are causing
description loss, which unifies p2a and p2b into one root cause.*

**It cannot work.** `treatment_merge_metric` reads `description` and
`diagnosis` and nothing else — deliberately, per its own docstring —
and p2b is *defined* as having neither. The measured "0.00 % versus
p1's 16.60 %" is a tautology, not a refutation. Recorded because the
number looks like a decisive result and is not one, the same shape as
the vacuous gap test in `recover_bands` and the discarded
`Asexual morph` detector.

**The hypothesis is testable by adjacency instead**, and it fails.
If a neighbour ate the description, that neighbour should look
*more* merged than average:

| group | n | median | ≥ 10 |
|---|---:|---:|---:|
| p1 in all-full documents (control) | 12 282 | 1 | 16.64 % |
| p1 in mixed documents | 33 763 | 1 | 16.58 % |
| …nearest neighbour of a p2b treatment | 23 041 | **0** | **9.50 %** |

The control matters: at 16.64 % against 16.58 % there is **no
document-class confound**, so the neighbour figure can be trusted. And
it runs the *wrong way* — the p1 treatment closest to an empty one is
barely half as merge-prone as a random one in the same document.

**So p2a and p2b do not unify.** Descriptions are not being eaten by
their neighbours; if anything, treatments beside empty ones are
cleaner than average. The picture consistent with classes B and C is
different and simpler: **boundaries are being generated inside
stretches of non-descriptive material** — keys, captions, specimen
lists — where there was never a description to lose. That extends the
operator's non-taxonomic-article theory (§5) from whole documents down
to *regions within* taxonomic ones.

**Consequence for v5 sequencing**: the grouper fix that p2a needs is
not the fix p2b needs, and work aimed at one should not be expected to
move the other.

### 5.5.1 The classification outline — 3 396 identical blocks, 17 labels

`taxon_3888d38f`, another of §5.6's correct refusals. The operator:
*"a list of fully cited higher taxa… both layers of label are pretty
meaningless. I don't know what the parenthesized numbers are. This
could be a table of contents?"*

**It is not a table of contents; it is a classification outline**, and
**the parenthesised number is the species count in that genus.** The
shape is unmistakable once several are seen together:

> `Nosematidae Tokarev, Huang, Solter, Malysh, Becnel & Vossbrinck`
> `Nosema Nägeli (20)`  `Vairimorpha Pilley (15)`
> `Encephalitozoonidae Voronin`
> `Encephalitozoon Levaditi, Nicolau & Schoen (12)`

Family, then its genera with how many species each holds. The source
is *Outline of Fungi and fungus-like taxa*; three such documents in the
corpus produce **565 treatments**, 47 % of them with no prose at all.

**"Both layers of label are meaningless" is measurable, and it is
worse than it sounds.** Within that one document, **3 396 blocks
containing a `(N)` species count** — structurally identical entries —
received **seventeen different labels**:

| label | share |
|---|---:|
| `Key` | 31.9 % |
| `Table` | 31.7 % |
| `Nomenclature` | 13.6 % |
| `Misc-exposition` | 12.6 % |
| `Bibliography` | 4.0 % |
| `Notes` | 3.7 % |
| eleven others | < 1 % each |

No label reaches a third. **On this genre the classifier is
effectively assigning at random**, and that is the cleanest evidence
in this memo that the failure is genre-level rather than a matter of
tuning: identical inputs, seventeen outputs.

It also partly explains §12.2's finding that `Key` is a second
catch-all. **This single document contributes 1 085 `Key`-labelled
blocks that are not keys** — roughly 3.5 % of the corpus-wide estimate,
from one source.

**Same conclusion as §5.5, reached from a second genre.** A checklist
and a classification outline are both documents made of names rather
than treatments. Neither should be parsed as a monograph, both carry
real nomenclatural data, and both need a different parser rather than
exclusion.

#### The 2024 Outline is a hybrid, and it merges genus entries wholesale

`taxon_395517e7` comes from **The 2024 Outline of Fungi and fungus-like
taxa** — same series as `taxon_3888d38f` above, different structure.
The 2024 edition interleaves the bare name lists with **numbered genus
notes** carrying real descriptions:

> `Note 722 Statesia` … `Entry by Maoqiang He, State Key Laboratory of
> Mycology, Institute of Microbiology, Chinese …`

The operator: *"a list of genera with remarks including descriptions.
Most of the descriptions were identified as diagnosis or description,
but by no means all."* Across the document's **306 treatments** the
spans divide **722 `notes` / 319 `diagnosis` / 221 `description`** —
so the "by no means all" is genus descriptions landing in `notes`.

**The merge damage is the headline.** `taxon_395517e7` alone holds
**four `diagnosis` spans and two `description` spans** — several genus
entries in one treatment. Across the document:

| | |
|---|---|
| treatments with > 1 `diagnosis` or `description` span | **104 of 306 (34 %)** |
| `merge_metric` median | **0.0** |
| flagged at the threshold of 15 | **6 (2 %)** |

**A third of the document's treatments merge genus entries and the
merge detector finds six of them.** The reason is §5.5's: each genus
entry is short and they are *different* genera, so there are almost no
repeated terms for a repetition metric to count. §6.1 measured that
metric at 51.7 % precision on the population it does flag; this is the
complementary failure, on a population it cannot flag at all.

`>1 diagnosis span` would have caught 104 of them, needs no text
analysis, and is already stored.

### 5.5.2 The clade-organised revision — a rank the schema has no room for

`taxon_4b567381`, from **"Phytophthora: taxonomic and phylogenetic
revision of the genus"** (*Studies in Mycology* 106, 2023). The
operator: *"mostly a table of species names… interspersed with
cladistic commentary… the 'treatments' to be extracted here would be
**the clade characterizations**."*

That is the right reading, and it is a different failure from §5.5 and
§5.5.1.

**The document is organised by clade, not by species.** Each clade gets
a membership list, a shared-character description, and ecological
commentary:

> `description` — *"Almost all species produce non-papillate,
> non-caducous sporangia on unbranched sporangiophores (P. estuarina,
> P. macrochlamydospora and…)"*
> a gap — *"Subclade 1c contains Phytophthora infestans and closely
> related species P. andina, P. betacei, P. ipomoeae…"*

**What the extractor made of it.** One treatment spanning **56 338
chars** over 22 spans — 15 `biology`, 4 `description`, 2 `notes` — with
`nomenclature` reading `(Fig. 3).` Across the whole revision, **32
treatments**, one of them **368 356 chars**. The natural units are
roughly a dozen clades and a couple of hundred species; it produced
neither.

**The operator's other observation checks out**: the cladistic and
environmental commentary *is* correctly landing in `biology`. The
labels are not the problem here.

**The 42 blocks naming a numbered clade scatter as usual** —
`Phylogeny` 33 %, `Biology` 29 %, `Bibliography` 17 %,
`Misc-exposition` 12 %, `Table` 7 % — no label above a third, the
§5.5.1 signature again.

#### The synthesis: the treatment abstraction assumes a single rank

Three findings recorded separately are one finding:

| where | what happened |
|---|---|
| §6.1 | "rank cascade" merges — `Sistotremastrales ord. nov.` carrying order, family, genus **and** species descriptions in one treatment |
| §5.5.1 | the Outline's family → genus → species-count hierarchy flattened into undifferentiated blocks |
| **§5.5.2** | **clade characterisations with nowhere to go at all** |

**Taxonomic literature is inherently multi-rank, and a `Treatment` is
implicitly one taxon at one rank.** Everything above species — genus
diagnoses, family circumscriptions, clade characterisations — is
either fused into a species treatment, scattered, or lost. §6.1's
verdict that some "merges" are really rank cascades, and that splitting
them requires knowing *which rank each description belongs to*, is the
same requirement seen from a third angle.

**This one is not a parser problem.** A checklist needs a different
parser; a clade revision needs a **different target schema** — a unit
that can hold "these species share these characters" without pretending
to be a species. That is a v5 data-model question, not a
classification one, and it should be settled before more effort goes
into making the segmenter label these documents better.

### 5.6 A treatment the annotator declines is usually not a treatment

Round 5's first 50 contained **ten with prose and zero annotations**.
They were exported anyway, via `brat_export --allow-unannotated`,
because hiding them would have made recall look better than it is.

**The operator read all ten, and Claude was right about every one.**
Precision of "returned no spans" is **10/10** on this sample. What it
declined to annotate:

| treatment | what the "description" actually is |
|---|---|
| `22292f50` | **CLSI M02 disk-diffusion protocol** — *"test a maximum of 12 disks on a 150-mm plate"* |
| `65cf0058` | **a drug packaging insert** — *"BREXAFEMME (ibrexafungerp tablets) are purple, oval, biconvex shaped tablets debossed with 150"* |
| `fc47df1e` | clinical-trial demographics — *"mean age was 27.2 years (SD 8.8)"* |
| `f94b9c84` | clinical diagnostics prose |
| `3011c747` | French enzymology, alcohol dehydrogenase electrophoresis |
| `4cb3fcb6` | methods — *"Growth rate and conidiation were detected as previously described [25]"* |
| `e4150d1a`, `62ffeff0` | **bibliographic pointers** — *"Descriptions. Mathiesen-Käärik (1950, p. 298); Hunt (1956…)"* |
| `3888d38f` | an **Outline of Fungi** entry: a name and an author, no description |
| `a21ae068` | a **molecular-only diagnosis** — *"Distinguishable … based on a diagnostic nucleotide signature in LSU D3"*, no morphology |

**This inverts how a zero-annotation treatment should be read.** It
looks like a recall failure and is almost always the opposite: evidence
that the *treatment generator* produced something that is not a
treatment. Six of the ten are non-taxonomic text of the §5 kind, two
are citations of descriptions published elsewhere, one is an outline
entry, and one is a real diagnosis carrying no morphology to label.

`65cf0058` deserves its own line: the extractor found morphology — of
a **tablet**. Purple, oval, biconvex, debossed. That is the §5 failure
in its purest form.

**Scale.** Across round 5's full draw, **124 of 1 000 (12.4 %)** have
prose and zero annotations, extrapolating to roughly **5 000
treatments** in p1. Every one was genuinely asked, so this is the
annotator declining rather than being skipped.

**Consequences.**

* **The recall denominator is contaminated.** A treatment with nothing
  to annotate cannot contribute a missed label, so including these in
  a recall statistic understates the annotator by construction. T5's
  precision figure is unaffected — they contribute no candidates
  either.
* **`annotation_count == 0` with `complexity_score > 0` is a cheap,
  high-precision detector for §5's document gate** — 10/10 here — and
  it needs no new computation, only a join already available.
* **It costs money to find.** Each of these was a real API call at full
  input price. ~5 000 of them is roughly **$25** of the round's
  spend on text that should never have reached the annotator.

### 5.7 There is no `Abstract` label, so abstracts go somewhere else

`taxon_3011c747` — one of §5.6's ten correct refusals — turned out to
carry a second finding. The operator: *"an abstract in French and then
English. The French abstract starts in the prior Misc-exposition
block."*

**The 22-tag schema has no `Abstract`.** `Nomenclature`, `Description`,
`Diagnosis`, `Etymology`, `Materials-examined`,
`Materials-and-methods`, `Type-designation`, `Biology`, `Phylogeny`,
`New-combinations`, `Notes`, `Key`, `Figure-caption`, `Bibliography`,
`Table`, `Index`, `ToC-entry`, `Misc-exposition`, `FIX`,
`Page-header`, `Holotype`, `Distribution` — and every paper has an
abstract.

Measured over 400 documents, blocks opening on `Abstract`, `Résumé`,
`Summary`, `Zusammenfassung`, `Riassunto` or `Resumen`: **58 blocks,
~3 000 corpus-wide**, distributed across six labels:

| landed in | share |
|---|---:|
| `Misc-exposition` | 70.7 % |
| **`Diagnosis`** | **10.3 %** |
| **`Notes`** | **8.6 %** |
| **`Biology`** | 5.2 % |
| **`Materials-examined`** | 3.4 % |
| `Key` | 1.7 % |
| `Description` | **0** |

**A hypothesis this refutes.** The obvious reading of
`taxon_3011c747` — whose `description` field holds French abstract
prose — is that abstracts become descriptions. **They do not: zero of
58.** What happens instead is the §12.2 asymmetry again. The block
*carrying the `Résumé` header* is recognisable and goes to
`Misc-exposition`; the **headerless continuation** is just prose, and
that is what becomes `Description`. The operator's phrasing was exact:
the abstract *starts* in the prior block.

**But 27.6 % do reach content fields.** Sixteen of 58 land in
`Diagnosis`, `Notes`, `Biology` or `Materials-examined` — roughly
**840 abstract blocks corpus-wide entering treatments as though they
were data**. An abstract routed to `Diagnosis` is particularly bad: it
is discursive summary in the field a consumer trusts most for
differential characters.

#### ~~Adding an `Abstract` tag is the fix~~ — narrowed 2026-08-26

That is what this section said, and it was too broad. The operator:
*"abstract is not part of a taxonomic treatment, but it IS a major part
of an academic paper. The same can be said for author information,
introductory and concluding remarks… Does it actually help us trying to
detect treatments?"*

**Mostly no, and the recommendation was ambiguous about which label
space it meant.** The v4 segmenter is two-stage:

* **Pass 1, `LayoutCRF`** — a linear-chain CRF over the whole document,
  deciding which lines are layout artefacts.
* **Pass 2, `crf_treatment`** — labels only the **non-layout
  subsequence** with the 12 treatment categories.

Abstract, author block, introduction and acknowledgements are all
**Pass 1's** business and never reach Pass 2. Adding `Abstract` to
Pass 2's twelve would be plainly wrong: it is not a treatment category.

**And the direction of the dominant error is the opposite one.**
Ranked by what has actually been measured:

| Pass-1 error | direction | ≈ blocks |
|---|---|---:|
| self-labelling content → `Misc-exposition` (§12.2) | **content called artefact** | ~15 400 |
| registry identifiers → non-content (§12.2) | **content called artefact** | ~2 600 |
| abstracts → `Diagnosis` / `Notes` / `Biology` | artefact called content | ~840 |
| abstracts → `Misc-exposition` | artefact correctly filtered | ~2 100 |

**~18 000 blocks of real content are being discarded as artefact,
against ~840 of abstract leaking in.** An `Abstract` class does not
touch the large number, and the 2 100 that land in `Misc-exposition`
are already being filtered correctly — they cost nothing.

**The one argument that survives is about transitions, not naming.**
Pass 1 is a linear chain, so it learns P(next state | current state).
With everything before the taxonomy collapsed into one undifferentiated
`Misc-exposition`, the model cannot represent *"the taxonomic section
begins after Methods."* Distinct structural states would let it learn
P(`Description` | `Abstract`) ≈ 0 — and boundaries are precisely what
§12.2 keeps finding broken. Against that: more states means sparser
data per state, and the canonical order fails for exactly the genres
that fail worst — a checklist (§5.5) or an Outline (§5.5.1) has no
Methods section.

**Narrowed recommendation.** Add structural classes to **Pass 1 only**,
and only if a prior test comes back positive: **do treatment-boundary
errors cluster near paper-structure blocks?** If a treatment's first
and last spans go wrong disproportionately often when adjacent to an
abstract, introduction or acknowledgements, the transition argument
holds. If boundary errors are spread uniformly through the taxonomy
section, structural labels buy nothing and the effort belongs on the
~18 000.

That test needs only `ann_combined` and the stored spans. **No
retraining, and it should precede any schema change.**

**Also visible in this treatment: the title is shredded.** `ÉTUDE` in
`Misc-exposition`, `DE` in **`Table`**, `QUALITATIVE.'` back in
`Misc-exposition` — one title across three blocks and two labels, in a
1985 *Cryptogamie, Mycologie* scan. Ordinary §9 damage, noted because
it explains why the surrounding routing is so poor.

### 5.5 The annotated-checklist genre (T5)

Found 2026-08-26 during T5 review. The operator's verdict on
`taxon_1702e95b` was *"includes pieces of several different treatments,
nearly every label is wrong"* — correct, and the cause is **genre
rather than a per-treatment fault**.

The source is **"The lichens of the Alps – an annotated checklist"**
(*MycoKeys* 2018), and a checklist entry is not a treatment. It is a
line of abbreviated codes:

| field | contents |
|---|---|
| `description` | `Ge: OB. Sw: BE, GR, SZ, TI, UR, UW, VS. Fr: AMa.` — **distribution codes** |
| a gap | `L – Subs.: cor, xyl – Alt.: 2–3 – Note: a mainly cool-temperate species…` — **coded ecology** |
| a gap | `Cliostomum griffithii (Sm.) Coppins` — **a second species entirely** |

`merge_metric` reads **0**: checklist entries are short and share no
repeated terms, so the repetition metric cannot see a document that is
nothing but adjacent entries. **That one document produced 293
treatments.**

**Measured corpus-wide**, taking documents whose *title* says
checklist, check-list, catalogue or census — **126 documents, 1 157
treatments**:

| | n | share |
|---|---:|---:|
| no prose at all (p2b) | 928 | **80 %** |
| synthetic nomenclature | 191 | 17 % |
| coded `Subs.:` / `Alt.:` description | 79 | 7 % |

**80 % against a corpus-wide p2b rate of 43.5 %** — 1.8× enriched.
That is the same conclusion §5.4 reached from the other end:
boundaries are generated inside stretches of non-descriptive material.
A checklist is that condition holding for a whole document.

**A second cheap signal for the §5 document gate.** §5 proposed a
document-level "is this a taxonomic article" test using journal and
title keywords, and estimated it would remove ~14 000 spurious
treatments. This adds a sharper variant: *the title says checklist*.
1 157 treatments is small beside 14 000, but the precision is far
higher — a checklist genuinely is taxonomic, it simply does not
contain treatments — and the rule is one regex over a field already
stored.

**Not proposed: deleting them.** A checklist carries real
nomenclature, distribution and substrate data. It needs a *different
parser*, not exclusion, and conflating "cannot be parsed as a
treatment" with "not worth having" is how the Persoonia whole-volume
scans were nearly written off in §5.2.

### 6. Multiple species merged into one treatment

**Symptom**: one Treatment doc contains descriptive content for
two or more distinct species.

**Evidence**:

* **`taxon_3d0a3c69...`** — noted 2026-08-21 from round-4.
  **Extreme fragment scatter across multiple genera, with
  vascular-plant nomenclature.**  Source: "Contributions to
  the Botany of the State of New York", *Bulletin of the New
  York State Museum* — a 19th-century bulletin covering
  vascular plants **and** fungi in one document.  Operator:
  "a sequence of description fragments probably from more
  than one species as we have two disjoint Lamellae sections
  and three disjoint Pileus sections.  Many of the fragments
  have no capital letter at the start of a sentence."  The
  counts are higher than that — the annotation set has
  **five** `Pileus` spans and **five** `Lamellae` spans.

  **Scale**: 10 `description_spans` with 9 gaps, char 53 601
  → 69 434 — **15 833 characters of source**, paragraphs 463
  → 577, 114 paragraphs apart — yielding a 1684-char
  description.  Content mixes at least three genera:
  *Cantharellus* (`In Cantharellus (proper) the pileus is
  fleshy, glabrous`), *Leptocantharellus* (OCR'd
  `Leptocanthakellus`), and *Paxillus*.  Description span #1
  (53 601–53 825) **crosses a species boundary** — the
  heading `Paxillus involutus Fr. / Involute Paxillus.`
  falls inside its char range.  The `key` field separately
  holds a real *Paxillus* key.

  **Wrong-kingdom nomenclature.**  The `treatment` field
  holds `Potamogeton pauciflorus Pursh.` and `Juncus
  Canadensis var. coarctatus Engelm.` — a pondweed and a
  rush, both **vascular plants** — plus a bibliographic
  fragment, across 3 `nomenclature_spans`.  The
  treatment-grouper picked up plant headings from a
  mixed-kingdom source.  No detector looks at kingdom.

  **What fires**: `§10:mid_sentence` (opens `behind,
  distinct from the hymenophorum`), `§10:diag_head_clip`,
  `§12:desc_span_gap`.

  **What doesn't fire is the point.**  merge_metric reads
  **1** on maximally-merged content and
  `n_repeated_structural_anatomy` reads **0**, for two
  independent reasons:

  1. `Pileus` and `Stipe` are *deliberately excluded* from
     `_STRUCTURAL_ANATOMY_WATCHLIST` over false-positive
     risk, and `Lamellae` was never on it — so the two
     most-repeated nouns here can never fire it.
  2. Even if they were listed, the detector anchors on
     paragraph start (text start, or after a blank line),
     and these fragments are joined by **single** newlines
     from OCR line-wrapping.  Only 1 paragraph-start
     `Pileus` and 1 paragraph-start `Lamellae` are visible,
     against a threshold of 2.

  That second point is the operator's "no capital letter at
  the start of a sentence" observation cashed out: the worse
  the fragmentation, the less paragraph structure survives,
  and the less likely the §6 anatomy detector is to fire.
  **Detectability is anti-correlated with severity.**

  Meanwhile Claude's annotations found the merge immediately
  — 5 `Pileus` + 5 `Lamellae`.  The signal is already in
  data we store, which is what **D7** in the Detector
  backlog proposes to use.  Between taxon_adcb2fcc (2 spans,
  15-line gap) and taxon_2b793602 (the §8 flora slice,
  `Pileus` 46 / `Lamellae` 40) in severity, and the clearest
  case of the gap.
* **`taxon_3d9f50f8...`** — noted 2026-08-21 from round-4.
  **Two species, and every strong signal is in a field no
  detector reads.**  Nomenclature names only *Vexillomyces
  fraxinicola* S. Bien, S. Peters & G. Langer, sp. nov.  The
  description holds two unrelated descriptions: (A) an
  in-culture asexual morph (`Sexual morph not observed.
  Asexual morph on synthetic nutrient-poor agar (SNA).` →
  Vegetative hyphae, phialidic/adelophialidic conidiogenous
  cells with collarettes, conidia aseptate allantoid
  (2.5–)3–5(–7) × 1–1.5(–2) μm, `Conidiomata not
  observed.`); and (B) an in-situ cercosporoid leaf-spot
  description (Leaf spots amphigenous, Caespituli,
  `Conidiomata sporodochial, intraepidermal, erumpent`,
  holoblastic sympodial cells, conidia **55–90 × 3–6 μm,
  (0–)3–7-septate**).  The conidia differ by more than an
  order of magnitude and disagree on septation and
  conidiogenesis.

  Evidence, strongest first:

  1. **Two `Etymology.` blocks** — `Named after its host
     genus, Fraxinus` and `Named after Ivy May Hassard (née
     Pearce) (1914–1998)`.  Two species' etymologies; the
     second species' name appears nowhere in the doc.
  2. **`description_spans` is 4 spans in two clusters** —
     paragraphs 1175+1177 (chars 225 035–226 803) and
     1205+1207 (chars 231 450–232 482) — separated by
     **4 647 characters and 28 paragraphs**.  The source in
     that gap holds the Typus and Notes for a *third*
     species, *Valsonectria robiniae*: this is a
     Fungal-Planet-style sheet run sliced across
     consecutive species.
  3. **Two `Culture characteristics` blocks with different
     protocols** — OA and SNA read at 4 wk, versus PDA at
     25 °C, 14 d, in darkness.

  **Only `§12:desc_span_gap` fires**, and each miss has an
  identified cause:

  * `Culture characteristics` is **deliberately excluded**
    from `_SECTION_HEADER_WATCHLIST` as a substrate-specific
    subtype, for taxon_b9a6232 false-positive prevention.
    That guard is exactly what blinds it here, where the two
    blocks belong to different species.  Same shape as
    taxon_3d0a3c69's deliberate `Pileus`/`Stipe` exclusion —
    **a false-positive guard becoming a blind spot on a real
    merge.**  Two independent instances now; worth treating
    as a pattern rather than two accidents.
  * The two `Etymology.` blocks **would** fire
    `§6:multi_section_header` —
    `count_repeated_section_headers` returns 1 when run on
    this doc's `etymology` field — but the detector only
    ever sees `description` and `diagnosis`.  The strongest
    single signal in the treatment sits in a field nothing
    reads.
  * merge_metric reads **4**, below threshold.
  * **D7 would fire cleanly**: the annotation set repeats
    `Conidia` ×2, `Conidiogenous_cells` ×2, `Conidiophores`
    ×2, `Conidiomata` ×2 and `Culture_characteristics` ×2.
    Second case after taxon_3d0a3c69 where the annotations
    catch a merge the text detectors miss.
* **T3** — `description` contains both `I. Gymnopilus laeticolor
  sp. nov.` and `3. Gymnopilus ornatulus sp. nov.` blocks; the
  treatment-grouper failed to split between them.  `materials_examined`
  similarly has specimen records for both species concatenated.
* **T5** — `description` contains three distinct Pileus-block
  paragraphs that read like three different *Laccaria* species'
  descriptions stuck together.  `biology` has three TYPE LOCALITY
  / HABITAT / DISTRIBUTION triples (one per species).
* **`taxon_592128a8...`** — discovered during the 2026-06-29
  50-treatment intermediate run as a 161-annotation outlier
  (vs. median ~9).  From a multi-species reference book; many
  species treatments in quick succession got concatenated under
  a single Nomenclature.  Initial 2026-06-29 assessment (that
  each constituent species had intact internal sections) was
  **revised 2026-07-02 on closer inspection** — the constituent
  sections are themselves fragmented, and the failure is
  distributed across the whole treatment, not confined to
  inter-species boundaries.  Observed:
    - **Description opens with the tail fragment of a
      taxonomic citation** — the extraction clipped the start
      of the run's first Nomenclature and the citation-tail
      leaked into the Description field.  See §10 for the
      general pattern.
    - **Multiple fractional taxonomic citations** embedded in
      the description, several containing `sp. nov.` or close
      variants.  These are per-species Nomenclatures that
      should have anchored per-species treatments; instead
      they landed as loose fragments inside a single
      Description.
    - **Multiple Latin blocks** in the description, matching
      the `taxon_572d470e` alternation pattern (below) — each
      Latin↔English cycle marks another species.
    - **Three separate `Observations:` headers** in the
      diagnosis.  Analogous to the `Diagnosis:` count signal
      that caught `taxon_2a9d07e6` (below): a well-formed
      single-species treatment has at most one Observations
      block; three is a strong multi-species signal.
    - Sixteen `Pileus` clauses (previously noted) also match.
  The revised picture: the layout CRF DID identify per-species
  section boundaries, but the treatment-grouper collapsed them
  into flat fields (Description, Diagnosis) without preserving
  the segment labels, so the fragments now interleave.  See
  §12 for the strategic response.
* **`taxon_173204...`** — discovered 2026-07-01, revisited
  2026-07-03 (round-2 review by piggy@puchpuchobs).  Real
  nomenclature (`Setiferotheca nipponica Matsush.`), then a
  description field containing TWO similar-species descriptions
  concatenated.  Label distribution shows the tell-tale
  doubling: Asci × 2, Ascomata × 2, Ascospores × 2, Peridium
  × 2, plus singletons (Chlamydospores, Mycelium, Necks,
  Subiculum).  Only 12 annotations total — well within the
  single-species range for a rich ascomycete.  **Slipped past
  the merge-metric filter** (metric value 2, threshold 10)
  because the two species share anatomical vocabulary and
  each term appears only ~2 times, below the k=5 count
  threshold.  Compact 2-species merges where species are
  similar (congenerics, same family) are a documented blind
  spot of the current metric — see 'Merge-metric limitations'
  below.
  **2026-07-03 revisit — boundary marker present**: the
  operator identifies the two species as separated by a
  **complete taxonomic citation for the genus
  Syspastospora** — a formally-cited GENUS name between the
  two species descriptions.  This is a strong §6 idea #2
  (gnfinder) target — a formally-cited authored name inside
  a Description is a merge signal (§1 rule).  The genus
  citation for Syspastospora would parse cleanly through
  gnfinder; the fact that gnfinder detection isn't
  implemented yet is the reason this compact-congenerics
  case remains a blind spot.  Round-2 reviewer data:
  kept 12, added 2, deleted 0 — Claude covered species 1
  cleanly; reviewer added 2 features per §0 rule 3.
* **`taxon_2a9d07e6...`** — discovered 2026-07-01, extended
  2026-07-03.  Nomenclature `Teratosphaeria dunnii Crous &
  Carnegie` correctly parsed; description contains a SECOND
  full species treatment (`Teratosphaeria obscuris` with its
  own formal citation).  **Three structural markers** would
  have caught the merge:
    (1) the `description` field contains the literal string
    `Diagnosis:` twice (once at the top for T. dunnii, once
    mid-body for T. obscuris) — a properly-single-species
    description has one such header at most;
    (2) the second citation would parse cleanly via
    gnfinder / gnparser as an authored binomial with clean
    OCR (operator-confirmed 2026-07-03), and no legitimate
    `description` field should contain a formal citation
    (see §1's Description-vs-Diagnosis distinction);
    (3) each of the two diagnostic-traits clauses
    (§13 sense — Diagnostic Characters, per Cifelli &
    Kielan-Jaworowska 2005) ends with a `Description and
    illustration:` citation.  This is a NEW header keyword
    pattern — the paper's format is "brief diagnoses with
    external references to full descriptions," a legitimate
    publication style for keys and revisions.  Adds
    `Description and illustration:` to the §6 idea #3
    header watchlist.  Two occurrences = two species'
    references = independent merge signal beyond the
    `Diagnosis:` count.
  **Slipped past the merge-metric filter with metric = 0**
  — 7 annotations total across 6 labels; both species are
  compact enough that no term reaches k=5.  Worst of the
  observed blind spots for the term-frequency metric alone,
  but multiple orthogonal header signals catch it.
* **`taxon_572d470e...`** — discovered 2026-07-01 in the
  50-treatment run.  Nomenclature is
  `Saccobolus sphaerosporus Brumm., spec. nov.`; description
  contains multiple species descriptions from a pre-2012
  mycological monograph.  Structural signal noted by the
  operator: **alternating Latin and English blocks**.  The
  standard format is ONE Latin diagnosis + ONE English
  description per species; the description here has multiple
  Latin↔English cycles (e.g., `Apothecia angustata parva
  sessilia ...` [Latin, S. sphaerosporus] → `Apothecia
  solitary or closely crowded ...` [English, same species]
  → discussion / comparison → `Saccobolus purpureus Brumm.,
  spec. nov.` [heading] → `Apothecia sessilia, 0.15-0.50 mm
  diam. Receptaculum ...` [Latin, S. purpureus] ...).  Two
  Latin blocks in one description is a very strong merge
  signal.  gnfinder / gnparser would help here IN PRINCIPLE
  but the binomials in this treatment have heavy OCR
  corruption (`Brumm., spec. llOU.` for `Brumm., spec.
  nov.`) so parser-based detection would return low
  confidence.  Language-alternation is more robust to OCR
  noise since Latin's morphology signal survives typos.
  **Caught by the merge-metric filter** (metric = 66) but
  the Latin/English signal is more interpretable and
  survives compact 2-species cases the term-frequency
  approach misses.

  **Follow-up 2026-07-02**: closer inspection surfaces three
  additional details that recontextualize the case.
    - **Diagnosis-block leak into Description**: most of what
      SHOULD have been in the `diagnosis` field is present in
      `description` instead — another instance of the
      §12 assembly-loses-labels pattern.  The layout CRF
      likely tagged the block `Diagnosis`; the grouper
      flattened it into Description.
    - **Heterogeneous OCR quality within one treatment**: the
      previously-noted `Brumm., spec. llOU.` looked like the
      worst case, but 2026-07-02 curl testing against local
      gnfinder revealed the picture is more favourable than
      assumed.  **The binomial `Saccobolus sphaerosporus`
      matches cleanly** even followed by the garbage
      `Brumm., spec. llOU.` — gnfinder tolerates
      authority-suffix corruption.  What still fails is
      mid-word character substitution inside the binomial
      itself: `Mycostigtna` (the same treatment's genus
      name) doesn't match at all.  So gnfinder catches
      more of this treatment's citations than expected —
      just not the `gm . nov.` fragment whose genus is
      itself corrupted.  Argues gnfinder detection (§6
      idea #2) is worth trying even on high-OCR-noise
      treatments,
      not written off wholesale.
    - **§11 hierarchical pattern is also present** —
      `Mycostigtna … gn. nov.` is a new genus (§11) buried
      inside this multi-species merge.  This treatment
      compounds §6 (multi-species merge), §11 (new genus),
      §10 (Nomenclature-tail leak into Description), and
      §12 (Diagnosis leaked into Description).  Useful as
      a compound-failure exemplar for regression testing —
      a fixer that clears one class shouldn't regress the
      others.
* **`taxon_95dbdfb9...`** — noted 2026-07-02.  Compound
  §6 + §9 + §12 case with a **detector-topology observation**:
  the merge-metric filter returned 7 (**below** the
  threshold of 10) — MISSED — but the header-count
  detector counted 3 `Description:` headers and flagged
  it correctly (`§6:multi_description`).  First observed
  case where §6 idea #3 catches a merge that the
  term-frequency metric doesn't.  Argues header-count
  should be promoted to a first-class filter, not just an
  idea in the follow-up list.
    - **3 `Illustration:` + `Description:` header pairs**
      — one per constituent species, in the illustrated-
      monograph format ("Illustration: Braun et al. …"
      then a Description block).  Adds `Illustration:` to
      the §6 idea #3 keyword watchlist.
    - **U+FFFD noise runs embedded interstitially** —
      several long strings of `�` characters within the
      description.  Text before and after the noise
      sometimes flows together (OCR dropped a word or two)
      and sometimes reads like something bigger got lost.
      Distinct from taxon_cda95f9f's §9 case (which was
      whole-block corruption that killed Claude entirely);
      here Claude produced 19 annotations, so the
      interstitial noise is partial rather than fatal.
      See §9 for the extended pattern.
    - **Holotype designation at end of Description** —
      Type/Holotype content that should have been under a
      Type block landed in Description.  Same
      assembly-drops-labels shape as taxon_572d470e's
      Diagnosis-in-Description and taxon_f00f8353's
      Materials_examined-in-Description leaks.  Another
      §12 instance.
  Reviewer treatment: 3-species merge; apply the §0
  first-species-only rule.  The interstitial U+FFFD noise
  may force skipping some anatomical clauses per §0 rule
  2 if the noise obscures the feature value.

* **`taxon_876c18ec...`** — noted 2026-07-02.  Description
  opens with **4 `Colonies (Fig. 1)` blocks** — one per
  constituent species — plus 4 downstream lowercase
  `colonies` continuation sentences (8 mentions total).
  **Several blocks include the literal binomial inline**
  (rather than a properly-labelled Nomenclature header).
  Caught by the merge-metric filter (metric = 12, barely
  above threshold) AND by `synthetic_nomenclature = True`
  (§2) — the missing per-species Nomenclature block
  triggered the synth fallback.
    - **Fig. 1 mystery, resolved**: all 4 species reference
      "Fig. 1".  Not from different articles.  Single ingest
      source (`skol_dev/845a45f89f9e...`); all 7
      description spans in one contiguous line range
      (29118-29377).  This is a single monograph with a
      **composite Fig. 1** showing colony habits for all
      new species side-by-side — a common taxonomic
      convention for multi-new-species papers.  The
      shared-figure reference is itself a diagnostic
      signal: 4 species pointing at the same figure means
      1 paper, not 4.
    - **Assembly failure**: the source has per-species
      Nomenclatures (binomials appear in the description
      body); the layout CRF or grouper lost them.  Yet
      another §12 instance — labels present in the source
      but discarded at assembly.
    - **Labeling decision (2026-07-02, revised
      2026-07-02 by taxon_9e68c26b)**: `Colony` and
      `Cultural characteristics` are distinct anatomical
      entities and MUST NOT be collapsed onto each other.
      Evidence: taxon_9e68c26b differentiates colonies on
      the natural substrate (ecological/observational)
      from colonies on PDA (experimental / laboratory
      culture) within a single treatment.  A single
      canonical label would erase that distinction.
      Current mapping:
        * `Colonies` → `Colony` (plural → singular
          normalization; the anatomical entity in natural
          context)
        * `Culture_characteristics` /
          `Culture characteristics` /
          `Cultural_characteristics` → `Cultural
          characteristics` (spelling/underscore
          normalization; laboratory-culture behaviour)
      The earlier 2026-07-02 decision folding both onto
      `Colony` (rationale: "consistent with anatomical-
      entity naming pattern") was too aggressive — the
      operator's taxon_9e68c26b review revealed the
      hidden semantic distinction.  Drift-map in
      `docs/feature_label_canonicalization.json` updated
      accordingly.
* **`taxon_f00f8353...`** — noted 2026-07-02.  Compound
  §6 + §10 case:
    - **Description opens with `thick.`** — a canonical
      clipped-anatomical-dimension opening.  The source line
      was almost certainly `Apothecia 200 µm thick.` or
      similar; only the tail survived.  See §10 for the
      general pattern.
    - **Two separate diagnosis blocks** in the description,
      but **the triage detector missed them** —
      `n_diagnosis_headers` = 0 despite the operator seeing
      two distinct diagnoses on inspection.  The
      diagnosis-header regex in `triage_signals.py` is
      literal `\bDiagnosis:`; this treatment uses a different
      boundary marker (likely the layout CRF labelled the
      blocks `Diagnosis` internally but the header text in
      the source is something else — `Ascomata:`,
      un-headered, or a species-heading transition).  Argues
      §6 idea #3 needs the extended keyword list from the
      taxon_e74d89b1 note AND a "diagnosis-shaped block
      without a literal header" secondary detector — likely
      an assembly-time signal (§12) rather than a regex.
    - **5 `Apothecia → Margin` clause pairs** in the
      description — same repetition-count signal class as
      taxon_e6402cd3's 7 Conidiomata sections and
      taxon_592128a8's 16 Pileus clauses, this time on a
      compound feature pair.
    - **Habitat clause** and **short Materials_examined
      block** embedded in Description — same
      Diagnosis-block-in-Description assembly failure noted
      for taxon_572d470e (see §12).
  Caught by BOTH the merge-metric filter (value = 26) AND
  the §10 mid-sentence detector.  A cleanly-worked
  compound-detection case: two of the current detectors
  flag it independently.
* **`taxon_e74d89b1...`** — noted 2026-07-02.  Description
  contains **many `Cultural characteristics` segments** — one
  per constituent species, each describing colony growth on a
  different lab medium (PCA, PDA, CMA).  Same class of
  repetition signal as taxon_592128a8's Pileus clauses and
  taxon_e6402cd3's Conidiomata sections, but on a section-
  header keyword the current detectors don't watch for.
  Broadens the §6-idea-#3 header-count list beyond
  `Diagnosis:` / `Description:` — the general principle is
  "count any section-header keyword repetition in the raw
  description."  Culturally-heavy treatments (asexual moulds,
  yeasts, plant pathogens) are the natural home of this
  pattern.  Caught by the merge-metric filter (metric = 35);
  triage CSV first-line = "On Potato carrot agar (PCA),
  mycelium consisting of pale".

**Merge-metric false positives** (2026-07-02):

* **`taxon_b9a62329...`** — flagged by the merge-metric
  filter (value = 27, threshold = 10) but **on inspection
  is a single species with an unusually detailed
  culture-morphology description**.  Description length
  7385 chars, no diagnosis field; describes one plant-
  pathogen species in exhaustive detail including its
  behaviour on multiple culture media.  The metric caught
  it because anatomical terms (`mycelium`, `hyphae`,
  `conidia`) repeat many times across the multi-medium
  culture description — but they refer to the SAME
  organism observed under different growth conditions,
  not different organisms.  Triage CSV first-line:
  "Mycelium internal, hyphae (4)-6-(10) jlm in diameter,
  forming lesions on".

  **Distinguishing signal from taxon_e74d89b1 (true
  positive)**: e74d89b1's `Cultural characteristics`
  sections each describe a DIFFERENT species on a
  different medium; b9a6232's culture sections describe
  the SAME species on different media.  Both trip the
  raw header-count heuristic; the distinguishing feature
  is whether each cultural section is anchored to its own
  Nomenclature/citation or all reference the same taxon.
  This argues §6 idea #3 needs a refinement: a header
  repetition inside one Nomenclature scope is normal
  detailed description; a header repetition ACROSS
  multiple Nomenclatures / citations is a merge.

  **Reviewer treatment**: this is a legitimate single-
  species treatment.  Annotate normally per §0
  conventions.  Consider flagging it as a false-positive
  regression target so future merge-detector work knows
  not to drop it.

* **`taxon_841d5cbe...`** — flagged by the merge-metric
  filter (value = 33, well above threshold=10) but
  **on inspection is a single species with a very
  complete description** — 8933 chars, opens `PILEUS:
  60–156 mm wide, white to whitish at first, darkening
  slightly when …`.  Round-1 reviewer
  (piggy@puchpuchobs) kept all 20 Claude annotations
  AND added 6 more the model missed — a high-recall
  reviewer pass, confirming the treatment is
  legitimate rich content, not a merge.  Diagnosis
  is a proper Differential Diagnosis (§13 sense) that
  matches the description.

  **New false-positive sub-cause**: the description is
  simply thorough.  Distinct from taxon_b9a6232
  (single species × multiple media) and taxon_9e048013
  (short description + long comparative diagnosis) —
  this one has a normal single-species format but
  covers each anatomical feature exhaustively.  Any
  term (pileus, lamellae, spores, stipe) appears
  many times across a rich 8933-char description
  because the treatment enumerates every colour,
  size range, developmental stage, etc.

  Consolidates a pattern across the three
  false-positive cases: **`merge_metric` is measuring
  content richness, not multi-species content**.  A
  very thorough single-species description trips it
  the same way a multi-species merge does.  Refines
  the earlier taxon_9e048013 refinement suggestion:
  scoping to Description-only helps with the diagnosis-
  inflation false positive but doesn't help with this
  case (Description was where the richness lived).
  The deeper fix has to be the structural-boundary
  approach (§6 idea #3 headers, §6 idea #2 gnfinder
  binomials, §6 idea #1(b) E→L→E ordering) — signals
  that distinguish "many mentions with a boundary
  between them" from "many mentions of a rich
  single-species description."

  Also fired `§2:synth_nomen` correctly — the
  Nomenclature field wasn't parsed cleanly.
  Independent of the §6 misfire.

* **`taxon_9e048013...`** — flagged by the merge-metric
  filter (value = 10, right at threshold) but **on
  inspection is essentially perfectly extracted**.
  Description = 439 chars, diagnosis = 2045 chars —
  diagnosis is nearly 5× the length of the description.

  **Root cause of the false positive**: the diagnosis
  legitimately compares the target species to several
  closely-related species — that's what a diagnosis IS.
  Comparative diagnoses recycle anatomy vocabulary
  (`pileus`, `stipe`, `spores`) across each comparison,
  which pushes term-frequency counts above threshold when
  the metric aggregates over Description + Diagnosis.
  `merge_metric` currently computes over both fields
  joined; on this treatment the description alone would
  score near zero.

  **Description also has legitimate anatomical
  repetition**: pileus texture in one sentence, pileus
  color in a later sentence; stipe texture + odor
  together, then stipe color + color change; spore shape
  separate from spore dimensions.  This multi-pass
  literary style is common and does not signal a merge.
  Not the cause of the false-positive flag here (the
  description is too short to matter), but a general
  reminder that raw anatomy-word repetition is a weaker
  signal than structural markers.

  **Concrete refinement**: change `treatment_merge_metric`
  to scan `description` only, not `description +
  diagnosis`.  A Differential Diagnosis (see §13
  polysemy note) is by construction multi-species-name-
  heavy; folding it into the term count guarantees this
  class of false positive.  Any
  merge signal worth detecting will surface in
  Description alone.  Would eliminate taxon_9e048013
  from the flagged set; needs measurement on the
  true-positive corpus (taxon_592128a8, taxon_572d470e,
  taxon_e6402cd3, taxon_e74d89b1) to confirm they still
  score above threshold on Description-only.

  **Reviewer treatment**: single-species and cleanly
  extracted; annotate normally.  Add to the false-
  positive regression list.
* **`taxon_e6402cd3...`** — noted 2026-07-02.  Fits the same
  compound-merge shape as taxon_572d470e and taxon_592128a8:
    - **Seven `Conidiomata` sections** in the description —
      the same repetition-count signal as taxon_592128a8's
      16 Pileus clauses, on a smaller scale.
    - **Alternating English↔Latin↔English↔Latin** blocks in
      the description — matches taxon_572d470e's language-
      alternation pattern.  Continues to argue Latin-block
      count (§6 detection idea #1) is broadly applicable.
    - **A cleanly-OCR'd complete taxonomic citation embedded
      in the description**: `(13) Septoria lycopersici Speg.,
      Anales Soc. Ci. Argent. 12: 115 (1882)`.  Author,
      journal, volume, page, year all present and readable.
      Confirms §6 detection idea #2 (gnfinder / gnparser)
      works reliably on clean-OCR treatments — the noisy
      taxon_572d470e case is the exception, not the rule.
      This is a strong single-signal detection: no
      legitimate `description` field should contain a
      complete authored citation.  Caught by the merge-
      metric filter (metric = 41).

**Affected treatments**: T3, T5, `taxon_592128a8...`, `taxon_2b793602...`
(three species + a partial genus description + the genus key in one
`description`; written up in §8 because the key half is the novel part).

**Likely stage**: treatment-grouper's species-boundary detection.
In T3 the boundary lines are `I.` / `3.` numbered headings which
the layout CRF didn't tag as Nomenclature.  In T5 the boundaries
may be even more subtle (just blank lines + a Pileus paragraph).
The `taxon_592128a8` case suggests a different sub-failure:
section labels were preserved per-species but the grouper missed
the per-species reset signal.  Related to §1 in T3 / T5
(mis-labeled headings); independent failure mode in `taxon_592128a8`.

**Severity**: high — multi-species treatments break per-species
aggregation, dataset statistics, and the bootstrap annotator's
single-taxon assumption.  Phase 1 review work explicitly needs
single-species treatments.

**Bootstrap-cost signal**: when a multi-species treatment goes
through Phase 1 annotation, the API spend scales with the merged
treatment's size (161-annotation case used proportionally more
output tokens).  A pre-bootstrap "is this suspiciously large?"
filter on `bin/select_for_annotation` could quarantine likely
multi-species treatments for the grouper-fix queue rather than
spending API budget on them.  Implemented 2026-07-01 as
``--exclude-suspected-merges`` — but see the merge-metric
limitations below.

**Merge-metric limitations** (as of 2026-07-01 calibration):

The default metric (`n_terms_above_k` with k=5, threshold 10)
catches ~16.5% of the corpus (7568 of 45871 scored treatments
on production_v4).  Empirically it hits the T3/T5 pattern and
the `taxon_592128a8` mass-merge cleanly.  But it misses:

  * **Compact 2-species merges of similar species**
    (`taxon_173204` case).  When two congeneric or same-family
    species are described together, shared anatomical vocabulary
    appears only ~2 times per term.  Each term stays below k=5,
    so the metric returns a low value (2 in the taxon_173204
    case) and the merge slips through the filter.
  * **Short merges regardless of similarity**.  If two species
    descriptions are terse (~200 words each), no single term
    hits the k=5 count and the metric misses the merge.

Neither pattern is rare — 2-species compact merges seem common
in fungal-focused papers where a new species is described
alongside a close relative for comparison.

Ideas for a better metric that would catch these:

  1. **Detect alternating Latin ↔ English blocks in the
     description**.  In pre-2012 taxonomic literature (still
     the majority of the ingested corpus), the standard
     format is ONE Latin diagnosis + ONE English description
     per species — Latin comes FIRST, followed by its
     English translation or the English description proper.
     Two order-independent merge signals:
       (a) **More than one Latin block** anywhere in the
           description (`latin_block_count >= 2`).  Already
           implemented in `triage_signals.latin_block_count`;
           fires §6:latin_alt when count >= 2.  Would have
           caught taxon_572d470e cleanly.
       (b) **A single Latin block sandwiched between two
           English blocks** (E → L → E ordering).
           Operator note 2026-07-03 (taxon_9ecad903): a
           mid-description Latin block with English on
           BOTH sides is a pathology even when
           `latin_block_count == 1`.  Normal structure
           puts Latin first (or lets the two languages
           live in separate labelled sections); Latin
           sandwiched by English means the assembler
           collapsed adjacent species' content across a
           Latin diagnosis that should have anchored one
           of them.  Detector requires an order-aware
           pass (not just a count): score each paragraph
           as Latin or English, then flag when a Latin
           paragraph is neither at the start nor at the
           end of the run.
     Detection is robust to OCR corruption (Latin
     morphology — endings `-us`, `-a`, `-um`, `-orum`,
     `-arum`, `-ibus`; vocabulary `apothecia`, `sessilia`,
     `ascosporae` — survives typos that break binomial
     parsing).  Cheap to compute paragraph-by-paragraph
     via langdetect / pycld3 / a Latin-suffix heuristic.
     Pre-bootstrap; no API spend needed.
  2. **Parse `description` for taxonomic citations via gnfinder /
     gnparser** (`http://localhost:9080` / `9081`).  A
     `description` field should describe ONE specimen and
     should contain NO formal citations of other taxa (see
     §1's Description-vs-Diagnosis distinction).  gnparser
     distinguishes bare mentions ("similar to X") from formally-
     cited names ("X. yz Author, Year") — the latter in a
     Description is a near-certain merge signal.  Would have
     caught the taxon_2a9d07e6 case (which the term-frequency
     metric missed with a value of 0).  Pre-bootstrap; no API
     spend needed.  Compatible with the existing local
     gnfinder/gnparser install.
     **OCR tolerance — measured 2026-07-02** (curl tests
     against `http://localhost:9080/9081`):
       * Clean binomials matched: `Trichaptum perrottetii
         (Lév.) Ryvarden` (taxon_83e36037) → match.
       * Binomials with clean genus/species tokens but
         corrupt AUTHORITY tokens matched: `Saccobolus
         sphaerosporus Brumm., spec. llOU.`
         (taxon_572d470e) → binomial matched cleanly; the
         `spec. llOU.` garbage was ignored.  So the
         earlier "gnfinder defeated by OCR" caveat was too
         pessimistic — authority-suffix corruption is
         tolerated.
       * Binomials with mid-word character substitution
         DID fail: `Mycovellosiel/a micranlhae`
         (taxon_2f276bfa; `/` substituting for `l` in
         genus) → no match, even with `allMatches=True`
         and `oddsDetails=True`.  Similarly
         `Mycostigtna` (taxon_572d470e's `gm . nov.`
         fragment).
     Net: gnfinder covers more of the corpus than we
     initially thought; only mid-word character substitution
     defeats it.  Installing `gnverifier` (fuzzy match
     against a reference database — not currently running
     locally) would extend coverage further at the cost of
     network / storage / one more service.
  3. **Count section-header keyword repetitions in the raw
     description**.  A single-species treatment has each
     section-header keyword appearing at most once; two or
     more of the same header is a strong merge signal.
     Concrete headers to watch — `Diagnosis:`, `Description:`,
     `Description and illustration:`, `Observations:`,
     `Illustration:`, `Cultural characteristics`,
     `Culture characteristics`, `Colonies on`,
     `Etymology:`, `Habitat:`, `Type:`, `Holotype:`.
     `taxon_2a9d07e6` had two `Diagnosis:` headers AND
     two `Description and illustration:` citations (each
     diagnostic-traits clause terminates with an
     external-reference citation — the "brief diagnoses
     with external descriptions" paper style used in
     revisions and keys); `taxon_592128a8` had three
     `Observations:` headers; `taxon_e74d89b1` had many
     `Cultural characteristics` sections; `taxon_95dbdfb9`
     had 3 `Illustration:` + 3 `Description:` pairs
     (illustrated monograph format).
     **Position refinement (taxon_a21a83f4)**: a header
     appearing at ANY offset > 0 inside the raw description
     field is a merge signal, not just count ≥ 2.  The
     `description` field IS the description; the ONLY
     legitimate place for a `Description:` header inside
     it is offset 0 (if any).  A mid-body `Description:`
     marks the start of a second species even when there
     is no repeat.  taxon_a21a83f4 had exactly one
     `Description:` (below the current ≥ 2 threshold) but
     it was clearly a species boundary; the current
     detector missed the merge.  Refinement: fire on
     `first_mid_body_offset > 0 OR count >= 2`.
     Pre-bootstrap;
     cheap case-insensitive regex scan.  Culturally-heavy
     treatments (asexual moulds, yeasts, plant pathogens)
     require the extended keyword list — the
     `Diagnosis:`/`Description:` pair alone misses them.
  4. **Count `sp. nov.` / `nov. sp.` / numbered species-heading
     occurrences** in the raw description.  Complementary to
     the term-frequency approach; catches compact merges.
  5. **Count section-header repetitions in the candidate DB
     annotations**: `Asci: 2` in the annotation output is a
     strong signal even at low per-term counts.  Requires
     post-bootstrap analysis (uses Claude's output), not
     pre-bootstrap filtering — so useful for retroactive audit
     rather than avoiding API spend.
  6. **Watch for repeated `Basionym:` / `Type:` / `Holotype:`
     entries** in the description or materials_examined.
  7. **Compare description length distribution** — a treatment
     with an unusually long description for its per-annotation
     count is a warning sign.

Priority ordering for follow-up work: #1, #2, #3 are cheap
pre-bootstrap filters that would catch the observed
blind-spot cases (taxon_173204, taxon_2a9d07e6) plus the
Latin-alternation-detectable cases (taxon_572d470e) that the
term-frequency metric missed.  Worth implementing before the
next big bootstrap run.

For the 2026-07-01 review round, treatments in these
blind-spot patterns must be caught by the reviewer (§0 rule 3:
annotate first species only) rather than automatically
quarantined.

### 6.1 `n_terms_above_5` measured against 30 hand verdicts (T3a)

The merge filter has excluded **7 632 treatments** from p1 since
2026-07-01 on a threshold calibrated against a 56-treatment sample.
Measured properly 2026-08-26: a stratified draw of 30, read by the
operator through `bin/treatment_dossier`, verdict `merge` / `single` /
`unsure` per treatment.

| band | population | n | merge | single | precision | est. false positives |
|---|---:|---:|---:|---:|---|---:|
| 10–14 | 2 112 | 15 | 4 | 10 | **28.6 %** [11.7, 54.6] | ~1 509 |
| 15–50 | 4 005 | 10 | 6 | 4 | 60.0 % [31.3, 83.2] | ~1 602 |
| > 50 | 1 515 | 5 | 5 | 0 | **100 %** [56.6, 100] | 0 |
| pooled | 7 632 | 30 | 15 | 14 | 51.7 % [34.4, 68.6] | **~3 111** |

Wilson intervals; one `unsure` excluded.

**The threshold is wrong, and p1 has been too small all along.**
Roughly **3 111 of the 7 632** excluded treatments are not merges —
so p1 is about **41 400, not 38 303**, and every round drawn since
2026-07-01 sampled from a frame ~7.5 % smaller than it should have
been. The draws stay valid (they were uniform over what they saw); the
*population* they describe was mis-stated.

**Threshold 15 is the F1 optimum**, on the same 29 labelled cases:

| rule | precision | recall | F1 |
|---|---:|---:|---:|
| `>= 10` (current) | 51.7 % | 100 % | 68.2 |
| **`>= 15`** | **73.3 %** | **73.3 %** | **73.3** |
| `>= 20` | 72.7 % | 53.3 % | 61.5 |
| `>= 50` | 100 % | 33.3 % | 50.0 |

#### A claim of mine the data refuted

The review index told the operator that the count of
`nomenclature_spans` was "close to decisive: two names means two
treatments." **It is the worst predictor tested** — precision 50 %,
recall **6.7 %**, F1 11.8.

The operator's own notes say why, repeatedly:

> *"The nomenclature for the second one (Arnium hirtum) was lost to a
> notes section."*
> *"Taxonomic citations have been consistently absorbed by
> Misc-exposition, Key, and Table blocks."*
> *"The gap (Misc-exposition) between the first two nomenclatures
> should have also been nomenclature."*

**The merge and the swallowed name are the same event.** A treatment
absorbs its neighbour precisely *because* the heading that would have
separated them was labelled `Misc-exposition`, `Key` or `Table` — so
the second name never becomes a second span, and counting spans cannot
see it. That couples D12 to merge detection directly: **fix D12 and
`nomenclature_spans` becomes the merge detector it looks like it should
already be.**

Every alternative tested, for the record:

| rule | precision | recall | F1 |
|---|---:|---:|---:|
| `>= 4 description_spans` | 71.4 % | 66.7 % | **69.0** |
| `>= 2 materials_examined_spans` | 71.4 % | 33.3 % | 45.5 |
| `>= 2 type_designation_spans` | **100 %** | 13.3 % | 23.5 |
| `>= 2 nomenclature_spans` | 50.0 % | 6.7 % | 11.8 |
| `>=2 materials` OR `score >= 40` | 80.0 % | 53.3 % | 64.0 |

`>= 2 type_designation_spans` is the only clean *rule-in*: two holotype
designations means two taxa, no exceptions in this sample. Low recall,
but a treatment it flags needs no second opinion.

#### What the false positives actually are

**The metric is a damage detector wearing a merge detector's name.**
Almost every `single` verdict still describes a real defect — just not
a merge:

> *"Pieces are missing. The first Misc-exposition is the last line of
> the etymology and the type designation."*
> *"Part of the taxonomic citation … was consumed by a Figure-caption
> block."*
> *"Misc-exposition blocks have absorbed parts of adjacent blocks."*

Only three of the fourteen singles are clean treatments. So excluding
these from p1 was not arbitrary — they *are* unusual — but they are
annotatable, and the annotator was never given the chance.

**And some `merge` verdicts are a different class again: rank
cascades.** `taxon_a33e8dcb` (`Sistotremastrales ord. nov.`) carries
descriptions of the new order, its type family, type genus and type
species; `taxon_710585339` is a genus redescription plus its unnamed
type species. Taxonomically these belong together; for extraction they
are separate treatments. A detector tuned on "two unrelated species"
will not find them, and a fix that splits them must know which rank
each description belongs to — which is §12's label-aware assembly
again.

#### Recommendation

1. **Raise `--merge-threshold` from 10 to 15.** Recovers ~2 112
   treatments of which ~71 % are annotatable, and takes precision from
   51.7 % to 73.3 % at no recall cost worth the name.
2. **Do not chase precision past that with this metric.** Every
   threshold above 20 trades more recall than it gains, and the
   population it is trying to describe is not really "merges."
3. **The real fix is D12.** The swallowed heading causes the merge;
   recover it and both the merge and its detector fall out.

#### Executed 2026-08-26

The threshold is now **configuration, defined once**:
`treatments_to_structured.merge_metric.DEFAULT_MERGE_THRESHOLD = 15`,
imported by `bin/env_config` as the last tier of CLI →
`MERGE_THRESHOLD` → config file → constant. The literal `10` had been
written out in **four** scripts; none declares its own flag now.

`fixes/retire_merge_skips.py` deleted the stale skip docs: **2 112
retired, 5 520 kept, 0 refused**, and a re-run retires nothing. p1 is
**38 413 → 40 525**.

**Why 2 112 and not the ~3 111 estimated above.** Those are different
numbers and both are right. ~3 111 is how many of the 7 632 are *truly*
not merges, summed across all three bands; 2 112 is what raising the
threshold to 15 actually recovers, because the 15–50 band stays
excluded despite its measured 40 % false-positive rate. That is the
deliberate trade: recovering the other ~1 602 would cost more precision
than it buys, and the honest route to them is D12, not a lower
threshold.

**Round 5 is unaffected and still reproducible.** It was drawn at
threshold 10; `--merge-threshold 10` reproduces that population, and
its sidecar records the value it used.

### 7. `key` field contains wrong-genus content

**Symptom**: a Treatment for genus A contains a dichotomous key for
genus B in its `key` field.

**Evidence**:

* **T5** — The *Laccaria striatula* Treatment's `key` field is
  ~2,500 lines of dichotomous-key prose, but the key is for
  *Melanoleuca* species ("II. SPECIES OCCURRING ON THE PACIFIC
  COAST", "III. SPECIES OCCURRING IN TROPICAL NORTH AMERICA",
  with entries `4. M. melaleuca`, `88. M. tenuipes`, etc.).  The
  Melanoleuca genus citation also leaked into T5's
  `materials_examined` (see §3).  The whole *Laccaria* treatment
  appears to be a slice of an NA-Flora chapter that bleeds into
  the next genus (*Melanoleuca*) without a clean boundary.

**Affected treatments**: T5.

**Likely stage**: same root cause as §6 — treatment-grouper
boundary detection fails at chapter/genus transitions in
flora-style documents.

**Severity**: medium — key fields aren't currently part of any
training-data pipeline but corrupt the per-treatment view.

### 8. Key-body content in the `description` field

**Symptom**: a Treatment's `description` contains dichotomous-key
couplet text (telegraphic feature/value pairs like "Pileus white
... 2", "Pileus brown ... 3") instead of (or alongside) the
specimen's actual prose description.  Inverse failure mode of §7
— there, key text lands in the right field but for the wrong
genus; here, key text lands in the wrong field entirely.

**Evidence**:

* Discovered 2026-06-29 during hand-review of the production_v4
  sample.  (Treatment ID to be filled in by the reviewer who
  encountered it.)  The description contains key-style couplet
  pairs rather than (or in addition to) the specimen's
  descriptive prose.
* **`taxon_5b0a8ce7...`** — discovered 2026-07-01.
  Nomenclature is `Amanita chlorinosma.` (real, non-synthetic).
  Description is 406 chars — the ENTIRE description consists
  of two numbered dichotomous-key couplets:
  ```
  15. Basal bulb may be elongate, almost always doglegged;
      clamps present or absent; universal veil with
      occasional to plentiful hyphae; with or without strong
      odor.

  16. Basal bulb ovoid to ventricose; clamps absent at bases
      of basidia; odorless or with slight odor of a tide pool
      or the seashore; spores (7.8-) 9.8 - 14.0 (-21) x (3.9-)
      4.9 - 6.3 (-9.8) jam, with Q « (1.85-) L.04.- 2:48
      (2.50)... coronas
  ```
  No real descriptive prose; the treatment's description
  content is purely key text.  Operator note: key detection
  is weaker than expected — the numbered `15.` / `16.`
  prefixes should be a strong signal for the layout CRF to
  label these paragraphs as `Key`, not `Description`.
  **Follow-up 2026-07-03**: operator suspects these are
  **terminal couplets** — the end-of-key steps that each
  point at a specific species.  Couplet 15 distinguishes
  Amanita chlorinosma from the alternative at 16 (elongate
  vs. ovoid basal bulb; clamp presence; odor).  Couplet 16
  also appears **tail-clipped**: ends `... coronas` after
  what looks like a truncated spore-shape clause.  Extra
  detail: terminal-couplet content extracted as if it
  were the treatment's own description is a subtly
  distinct failure from "key content leaked from
  elsewhere in the paper" — the key text HAPPENS to be
  about this species, it's just not the species' own
  descriptive prose.  Treatment-grouper likely landed on
  the terminal couplet because it mentions the target
  species by name.  Suggests a fix angle: when the layout
  CRF sees numbered-couplet text pointing at a species
  name, route it to `key`, NOT `description` — even when
  the target species matches the treatment's Nomenclature.
* **`taxon_2b793602...`** — discovered 2026-08-15.  The
  **Likely stage** hypothesis below, confirmed by example: this
  is a slice of a flora chapter carrying, in ONE 7875-char
  `description` field, three species descriptions (paragraphs
  0–2, each opening `Pileus ...`), a leaked authored citation
  (`Agaricus (Clitocybe) glaucipes Berk. & Curt.`), a partial
  genus description, and the genus key.  `merge_metric` is 39
  against a threshold of 10, so §6 fires — but the key content
  is invisible to §8 twice over:

    * **The key leads carry no couplet numbering.**  They read
      `Lamellae yellow, unchanging.` / `Pileus
      pale-honey-yellow; spores 8-9X5.5-6.5 p.` — feature/value
      pairs with no `15.` / `16.` prefix.  `n_key_couplets` is
      0, and **detection idea 1 below (numbered-couplet prefix
      at line start) would not catch this treatment either.**
      Unnumbered keys need a different signature — telegraphic
      fragment density, or absence of a finite verb, rather
      than a line-start numeral.
    * **`§8:key_content_short` requires `desc_length < 500`.**
      This description is 7875 chars, so even a working couplet
      count would not raise the flag.  That heuristic catches
      key text which has *displaced* the real description, not
      key text *appended* to it.

  Also worth recording: the three `Pileus`-opening paragraphs
  do NOT fire `§6:multi_structural_anatomy`, because
  `_STRUCTURAL_ANATOMY_WATCHLIST` deliberately excludes
  `Pileus` / `Stipe` for false-positive risk (see
  `treatments_to_structured/triage_signals.py`).  That
  exclusion is defensible for one description mentioning a
  pileus once — but three *paragraph-start* `Pileus` openers in
  a single field is the exact merge signature the watchlist
  exists to catch.  Worth weighing as a counter-example when
  the watchlist migrates to the corpus-derived form in
  `docs/plans/clade-agnostic-detectors.md`.

  The treatment also has a separately populated 5790-char `key`
  field, so the key content was extracted to the right place
  **and** left in the description — duplication, not
  misrouting.  Fixture class:
  `§8-flora-chapter-slice-unnumbered-key`.

**Affected treatments**: `taxon_5b0a8ce7...`, `taxon_2b793602...`; almost certainly
others — the fact that a treatment with `Amanita chlorinosma`
as its nomenclature ends up with ZERO real description content
suggests the pattern isn't rare.

**Detection ideas** (for a future extraction-audit script):

  1. **Numbered-couplet prefix at line start**: `^[0-9]+[a-z]?\.?\s+`
     followed by a short anatomical clause is a dichotomous-key
     signature.  Two or more such lines in a Description field is
     a near-certain sign of key content leakage.
     `taxon_5b0a8ce7` would be caught: it has `15.` and `16.`
     as line starts, both followed by anatomical prose.
  2. **Description-length-vs-content-density anomaly**:
     `taxon_5b0a8ce7`'s description is 406 chars and mentions
     multiple anatomical features (basal bulb, clamps, universal
     veil, odor, spores).  Real single-species descriptions of
     comparable feature coverage are typically 1500+ chars.
     Very short descriptions covering many features are key-like.
  3. **Contrastive/hedging language patterns**: "may be X ...",
     "with or without Y", "present or absent" — key couplets
     use conditionals to describe the choice space, real
     descriptions assert.

**Likely stage** (best guess): treatment-grouper boundary
detection fails to recognize the transition from in-treatment
description prose to in-document key prose.  Possibly related
to §6 (multi-species merge) — when a treatment is sliced from
a flora chapter that contains both a species description AND
the genus-level key, the slice may include both without a
boundary signal.  Alternatively: the layout CRF may be
mis-labeling numbered key couplets as Description continuations
because the couplets DO contain anatomical vocabulary.

**Severity**: medium — pollutes the bootstrap input with
non-treatment text.  The bootstrap annotator wastes API budget
on key couplets; the human reviewer wastes review time deciding
whether to annotate them; the resulting golden set risks
absorbing a different prose genre than what Pass B is being
trained for.

**Reviewer action** (until the extraction is fixed — see §0
below): leave key-body content unannotated.  Reasons:

  * Key couplets are biologically valid statements about
    anatomy ("Pileus white" IS a Pileus feature), but they're
    *contrastive choices*, not *asserted properties of this
    specimen*.  Annotating them mixes two prose genres in the
    training data.
  * Surface form differs: key couplets are telegraphic
    ("Pileus white ... 2") while descriptions are detailed
    ("Pileus 60–156 mm wide, white to whitish at first ...").
    Training on the union risks teaching the SLM to favor
    brief, decontextualized patterns.
  * Pass B's structured extraction will treat key couplets
    differently from descriptions anyway (one expresses
    eligibility, the other expresses values).  Leaving them
    unannotated in Phase 1 keeps the golden set genre-pure.
  * If the misplaced-key problem turns out to be common, a
    future Phase 2 enhancement could annotate key content with
    a distinct `field: "key_leaked_into_description"` marker
    so the SLM treats it as a separate genre — but that's not
    Phase 1 scope.

## §0. Hand-review conventions

These rules apply to the reviewer's brat work on bootstrap
candidate annotations.  They consolidate the per-issue
"Reviewer action" notes above into one place.

1. **Don't annotate misplaced content.**  When a treatment
   field contains content that obviously belongs in a different
   field (taxonomic citation in description, key body in
   description, wrong-genus key in key field, article-body prose
   in biology), leave it unmarked.  An unannotated span is
   visible to future reviewers as "passed over intentionally"
   and to future automated cleanup as "available for moving."

2. **Don't promote false-positive treatments to golden.**  If
   the treatment is from a non-taxonomic paper (§5), skip the
   whole treatment rather than annotating the few stray fragments
   that happen to look feature-like.  Phase 1 golden should
   only contain real treatments.

3. **Multi-species treatments**: annotate the FIRST species's
   features only.  Subsequent species in the same treatment doc
   (§6) are a separate failure mode; mixing their features into
   one treatment's golden record creates training noise.

4. **Reviewer-only labels are welcome.**  If Claude missed a
   feature the description clearly names (e.g., a hymenophore
   in a bolete treatment), add it.  The diff (in
   `features_hand`) will flag these as `reviewer_action: added`
   — useful signal for tuning the bootstrap prompt.

5. **When in doubt about a label, leave the annotation in but
   note your hesitation.**  Brat supports `AnnotatorNotes`
   (`#N\tAnnotatorNotes T<n>\tnote text`) — the ingest path
   doesn't strip these, so future canonicalization can use them.

### §0.1 Nested annotations are permitted, and they round-trip

Asked by the operator 2026-08-24 on `taxon_cdcba8db`, seeing a
`Subiculum` span inside an `Ascomata` span: *"correct, but I didn't
know we could do that."*

**We can, and it works** — but it had never been exercised.

* **Claude produced it, not the reviewer.** The nesting is already in
  `features_candidate`: `Subiculum` [153:273] inside `Ascomata`
  [21:274]. And it is correct — the subiculum is described *within*
  the ascomata sentence (`groups of ascomata often surrounded by
  woolly, white subiculum`).
* **It is the only one in the corpus.** One nested pair in **1 588**
  candidate annotations across 110 treatments; `features_hand` had
  **zero**, because this treatment had not yet been ingested.
* **It survives ingest.** A dry run reports `kept=13 added=0
  deleted=0` against a 13-annotation `.ann` — every span preserved,
  nesting included.

**Why it works, and why that was fragile.** `annotation_key` is
`(feature_label, field, start, end)` and `annotation_doc_id` is
`<treatment_id>:<label>:<start>`, so a nested pair never collides.
Nesting is therefore supported *by consequence*, not by design —
nothing asserted it, and a later overlap-resolution or
span-normalisation step could have dropped the inner span silently.

Pinned by `TestNestedSpansSurviveTheDiff` in
`treatments_to_structured/brat_ingest_test.py`, covering kept, added
alone, deleted alone, and the limiting case of two co-extensive spans
with different labels.

**Practical guidance for review:** nest when the anatomy genuinely
nests. A structure described inside another structure's sentence
should carry its own label rather than being left out because the
outer span already covers the text.

## §0.5. Poster-child treatments (reference)

The rest of this memo catalogs what goes wrong.  For
balance and calibration, this section records
extraction-poster-child treatments the operator has
called out — what a clean, correctly-extracted single-
species treatment looks like.  Useful when explaining
"what right looks like" to future reviewers, and as
regression targets: any future detector or assembler
tightening MUST NOT surface these as false positives.

* **`taxon_7dfd35bd...`** — noted 2026-08-23 from round-4.
  Lepiotoid agaric (*Lepiota ochraceosquamea* J.F. Liang &
  Zhu L. Yang, sp. nov., *MycoKeys* 123).  1585-char
  description in a single span, 14 annotations covering it
  contiguously, merge_metric = 2, every detector silent, no
  §15 markers.

  **Overlap declared**: the third agaric, after
  taxon_e78904cb (*Cortinarius*) and taxon_d2a4c584
  (*Mycena*).  `Pileus`, `Lamellae`, `Cheilocystidia`,
  `Pleurocystidia`, trichoderm and dextrinoid are all
  already covered.  Four things are not:

  1. **An annulus on an agaric.**  Neither existing agaric
     has one — *Cortinarius* has a cortina, the *Mycena* is
     a 1.5–3 mm bare-stemmed species.  The set's only other
     annulus is on taxon_343eec40's veiled **bolete**, so
     anything keying annulus to boletes misreads this.
  2. **Lamellula-tier notation** — `Lamellae L = 40–60,
     l = 1–2`.  Capital *L* counts lamellae, lowercase *l*
     counts tiers of lamellulae.  Nowhere else in the
     fixture, and precisely the string a measurement parser
     mishandles: `L = 40–60` looks like a dimension and is
     not.
  3. **Chemical spore-reaction suite** — `congophilous`,
     `not metachromatic in cresyl blue`.  Dextrinoid alone
     appears elsewhere; this register does not.
  4. **Spore-sample notation** `[67/3/3]` — 67 spores, 3
     basidiomata, 3 collections.

  **Fourth `jats_section` clean-text case** (after
  taxon_343eec40, taxon_66b43429, against taxon_5fe9223f).

  **Negative-data spans are now a named pattern.**  This
  treatment alone has three — `Pleurocystidia absent.`,
  `Smell not distinct`, `taste not recorded` — joining
  `Spore print … not obtained` (taxon_343eec40) and `Sexual
  morph: Undetermined.` (taxon_38992c86).  **Anything
  treating a feature span as an assertion of presence is
  wrong on all five.**  Note too that `Odour` and `Taste`
  are split from one sentence at its semicolon, finer than a
  sentence-level splitter would give.

  **Both labelling directions appeared in one treatment, and
  the operator settled one of them** on 2026-08-23: the
  spore span, labelled `Spores` over text reading
  *Basidiospores*, was **corrected to `Basidiospores`**.
  Clade-specific terms are not generalised — the same rule
  already applied to fruiting bodies, where `Basidiocarp`
  and `Basidiome` collapse to `Basidiomata` but
  `Basidiomata` and `Ascomata` stay apart.  Recorded in
  [`docs/feature_label_non_synonyms.md`](feature_label_non_synonyms.md).

  The other direction stands: `Pileipellis` labels text
  reading *Pileus covering*, normalising vernacular to
  technical as *Peristome*/*Mouth* does on taxon_66b43429 —
  that adds precision rather than removing it, so the two
  are not the same question after all.

  **Five round-4 treatments still carry the uncorrected
  form**: `taxon_fd50457a` and `taxon_4b89d160`
  (*Ascospores*), `taxon_d2d620ae`, `taxon_b673586a` and
  `taxon_5fe9223f` (*Basidiospores*).
* **`taxon_66b43429...`** — noted 2026-08-22 from round-4.
  Gasteroid stalked puffball (*Tulostoma dunense* Finy,
  Jeppson, L. Albert, Ölvedi, Dima & V. Papp, sp. nov.,
  *MycoKeys* 100).  Operator: "looks like a complete
  description of a stalked puffball.  poster child?"  Yes —
  and it is the largest single coverage gain the set has
  had.  1006-char description in a **single span**, 9
  annotations covering it contiguously with no interior or
  tail gaps, merge_metric = 0, every detector silent.

  **An entirely new body plan.**  Every anatomical term it
  uses is absent from every other poster child:
  `Spore-sac`, `Exoperidium`, `Endoperidium`, `Peristome`,
  `Socket`, `Gleba`, `Capillitium`, `pseudorhiza`.  The set
  held agarics, boletes, ascomycetes, asexual morphs, a
  bird's-nest fungus and a lichen — nothing gasteroid.  A
  clade-agnostic detector built from the earlier entries has
  no priors for a spore-sac on a stalk, and no reason to
  expect a description that never mentions a pileus,
  lamellae, hymenium or basidia.

  **Third data point for the §15 split.**  Its
  `source_anchors` carry 4 × `plazi` + `arpha` +
  **`jats_section`** + `mycobank`, and the description has
  **zero** element-join markers.  Set beside taxon_343eec40
  (also `jats_section`, also clean) and taxon_5fe9223f (six
  `plazi`-only anchors, riddled with joins), the pattern
  §15 measured at 96.1 % vs 0.4 % now has three individually
  inspected cases behind it.

  **Annotation note** for the deferred vocabulary pass: two
  spans use the technical term where the text uses the
  vernacular — `Peristome` for "Mouth", `Stipe` for "Stem".
  That is the *opposite* direction from the
  `Basidiospores`-labelled-`Spores` generalisation noted at
  taxon_09b97d5f, which erased clade information.
  Normalising *Mouth* to *Peristome* adds precision rather
  than removing it — worth settling as one rule rather than
  two.
* **`taxon_38992c86...`** — noted 2026-08-21 from round-4.
  Asexual morph of a dematiaceous hyphomycete, described
  *in situ* on the natural substrate plus culture
  (*Sporidesmium aquaticivaginatum* J. Yang & K.D. Hyde,
  *Fungal Divers.* 80: 217, 2016).  1383-char description,
  8 annotations, merge_metric = 1, all detectors silent,
  text clean of §15 markers.

  **Overlap declared.**  This is close in shape to
  taxon_e534e6a9 (`ascomycete-both-morphs`): both open
  `Saprobic on … decaying wood`, both use the explicit
  morph pair with one morph Undetermined, and both run
  Mycelium → Conidiophores → Conidiogenous cells → Conidia
  off macronematous mononematous conidiophores.  It is kept
  for the deltas, which are the kind of thing detectors
  trip on — not because the clade differs.

  1. **Morph markers take the colon form** — `Asexual
     morph:`, `Sexual morph: Undetermined.` — where
     taxon_e534e6a9 has them bare (`Sexual morph
     Undetermined. Asexual morph Colonies…`), **and the
     order is inverted**: asexual first, sexual last.  Any
     morph-marker regex needs both forms and both orders.
  2. **`Sexual morph: Undetermined.` is annotated** as a
     `Sexual_morph` span — explicitly-*absent* data labelled
     as a feature.  Same pattern as the `Spore print and
     macrochemical reactions not obtained.` span in
     taxon_343eec40.  Worth knowing before anyone treats
     feature spans as presence assertions.
  3. **`Culture_characteristics` carries germination
     behaviour** (`Conidia germinating on PDA within 24 h,
     and germ tubes produced at the apex`) rather than only
     colony morphology, which is what taxon_d65547ed and
     taxon_06594607 carry.
  4. **Not a protologue.**  An existing species redescribed
     as a new geographic record, so the diagnosis field
     holds a *phylogenetic* identification block (`our
     isolate clustered with the holotype … with 100 %
     ML/1.00 PP support (Fig. 59) … first time to report
     this species from China`) rather than a differential
     diagnosis.  Distinct from taxon_09b97d5f, whose
     diagnosis is comparative *morphology*.  Note this is a
     `Notes` block routed into `diagnosis` again — the
     field mapping recorded at taxon_09b97d5f.
  5. Conidiogenesis differs in detail: monoblastic,
     integrated, terminal, determinate, with distoseptate
     conidia carrying a mucilaginous apical sheath, against
     taxon_e534e6a9's polytretic proliferating cells and
     catenate conidia.

  **Annotation detail**: the only uncovered text in the
  description is the 16-character marker `Asexual morph: `
  itself, while `Sexual morph: Undetermined.` *is*
  annotated.  Harmless — the asexual content is fully
  covered by the individual feature spans — but a coverage
  metric that counts marker text will read this treatment
  as 99 % rather than 100 %.
* **`taxon_343eec40...`** — noted 2026-08-21 from round-4.
  Veiled bolete (*Pulveroboletus sokponianus* sp. nov.,
  *MycoKeys* 43, doi 10.3897/mycokeys.43.30776).  2636-char
  description, 17 Claude annotations, merge_metric = 5, all
  triage detectors silent.  Operator: "another poster child
  bolete."  Three things make it non-redundant against the
  other three boletes.

  **First veiled bolete in the reference set.**  Opens
  `Basidiomata medium-sized, wrapped in a greenish-yellow
  (1A2–3) general veil when young.` and carries `Annulus`,
  `Partial_veil`, `Basal_mycelium` and `Spore_print` spans.
  None of taxon_0cfe582f, taxon_0029f141 or taxon_09b97d5f
  contains the words *veil*, *annulus* or *ring* at all;
  among poster children only the agaric taxon_e78904cb
  does.  Anything treating veil structures as an
  agaric-only signal misclassifies this.

  **§15 clean control.**  Richest `source_anchors` set in
  the fixture — 2 × `plazi` + `arpha` + `jats_section` +
  `mycobank` — and the description carries **zero**
  `[a-z]\.[A-Z]` element-join markers.  Direct contrast
  with taxon_30d8d8d4, which is `plazi`-only and in the
  96 %-affected population.  Same journal family, same Plazi
  involvement, clean text — evidence that the
  `jats_section`-carrying path is the healthy one and §15
  is specific to plazi-only ingest.  Also the best
  available exemplar for the Trello #401 polymorphic-anchor
  work.

  **Legitimate in-description citation.**  `Odour fungoid,
  when fresh like Lepista nuda (in collection De Kesel
  1979).`  A bibliographic citation sitting inside
  description prose that is *not* the §1 pathology — it is
  an odour comparison, not a taxonomic-authority citation.
  Control case for any §1 tightening.  Note also the
  negative-data span: `Spore print and macrochemical
  reactions not obtained.` is annotated `Spore_print`, an
  explicit statement of *absent* data labelled as a
  feature.

  **Annotation miss — recorded so the entry doesn't read as
  flawless.**  The `Spores` sentence at offsets 1701–1901
  is the only uncovered prose in the description, and it is
  unannotated.  Probable cause: the preceding `Spore_print`
  span ends at 1701 and the annotator appears to have
  treated the spore material as already covered.  Surveyed
  across all 50 round-4 treatments, 15 have uncovered
  in-description runs of 60+ chars — but the other 14 are
  OCR garbage, `TYPE LOCALITY` / `MATERIAL STUDIED` /
  habitat lines, chemistry, or discussion prose, all
  legitimately skipped.  **This is the only clean
  anatomical-character sentence Claude missed in the entire
  round.**  Extraction is unaffected, which is why the
  entry stays in `poster_children`.
* **`taxon_09b97d5f...`** — noted 2026-08-21 from round-4.
  Bolete (*Butyriboletus parachinarensis* sp. nov.,
  Persoonia).  1166-char description, 9 Claude annotations
  covering it contiguously with no uncovered prose,
  merge_metric = 2, all triage detectors silent.  Operator:
  "another basidiomycete poster child."

  **Boletoid control for the anatomical-noun-clip pattern.**
  It opens directly at `Pileus 9.7–10.5 cm broad, convex to
  plano-convex, …` with **no `Basidiomata` umbrella
  sentence** — where taxon_0cfe582f opens `Basidiomata
  medium large sized, boletoid.` and taxon_0029f141 opens
  `Description: Basidiomata small to medium-sized.`  A §10
  head-clip detector keying on "bolete descriptions start at
  Basidiomata" would false-fire here.  This is the boletoid
  counterpart to what taxon_38b5b1c6 does for ascomycetes
  (see taxon_7fbc71a8).  Its micro suite (Basidiospores,
  Basidia, Cheilocystidia, Pileipellis, `Clamp connections
  absent`) duplicates taxon_0cfe582f's and is *not* the
  reason for the entry.

  **True-negative control for taxon_9e048013.**  The
  diagnosis field holds a 916-char comparative Notes block
  dense with authored binomials — `Butyriboletus sanicibus`,
  `B. yicibus`, `B. parachinarensis`, with BLAST identities
  and Q-value comparisons.  `authored_binomial_in_text` on
  the *diagnosis* returns True; on the *description* it
  returns False, and merge_metric stays at 2.  taxon_9e048013
  is the §6 false positive where exactly this content shape
  inflated merge_metric to 10.  Keeping both means a §6
  precision fix has a matched pair to work against.

  **Field-mapping observation** (not a defect of this
  treatment): the source's `Notes — ` paragraph landed in the
  `diagnosis` field, with `notes` left empty.  This is
  systematic rather than one-off — **1 884** production_v4
  treatments move Notes into an otherwise-empty diagnosis,
  and a further **932** populate both fields (of 18 787 with
  any diagnosis).  Defensible for `sp. nov.` treatments that
  carry no formal diagnosis, since the Notes paragraph is
  doing the differential work.  Recorded because any future
  consumer that treats `diagnosis` as protologue-diagnosis
  will be wrong about roughly 15 % of the populated ones.
* **`taxon_06594607...`** — noted 2026-08-21 from round-4.
  Synnematous asexual morph (*Myxospora aptrootii* sp. nov.,
  Persoonia).  1029-char description in two registers:
  in-situ anatomy (`Conidiomata synnematous, solitary,
  125–200 μm high, …`, then Stroma, Conidiogenous cells,
  Conidia) followed by a `Culture characteristics — Colonies
  on PDA, OA and CMA …` paragraph.  Real Nomenclature, no
  diagnosis field, merge_metric = 0, all triage detectors
  silent.  5 Claude annotations — exactly one per anatomical
  sentence group, contiguous, no uncovered prose.  Operator:
  "looks like a textbook description of a conidial form."

  **Pairs with taxon_d65547ed** to bracket the asexual-morph
  shape: that one is culture-ONLY, with conidiophores arising
  directly from the mycelium and no conidiomata at all; this
  one has fruiting structures described in situ *and* culture
  data.  A detector that treats "describes a colony on agar"
  as evidence of the culture-only shape must not fire here.

  **True-negative control for `§12:desc_span_gap`.**  The
  description is assembled from two `description_spans`
  (lines 7457–7468 and 7470–7474), split by the sub-header
  paragraph break — a legitimate 2-line gap.  The detector
  correctly stays silent.  Contrast taxon_adcb2fcc, whose
  15-line gap between fragments is the real §12 failure.
  This is the entry that keeps a gap-threshold tightening
  honest.

  **Typographic note**: the sub-header is `Culture
  characteristics — ` — em-dash delimited with
  NARROW NO-BREAK SPACE (U+202F) on both sides, the Persoonia
  house form, which also shows up in this treatment's `Notes`
  and in the nomenclature line.  No `Description`/`Diagnosis`
  header appears anywhere in this treatment, so the header
  regexes noted at taxon_d65547ed aren't exercised here.
  They would cope if one did: `_DESC_HEADER_RE` /
  `_DIAG_HEADER_RE` spell the gap as `\s*`, and Python's
  `\s` matches U+202F (verified).  Recorded as a
  don't-regress note — the period-terminated-header fix
  proposed at taxon_d65547ed must keep `\s*` rather than
  hard-coding an ASCII space, or it would miss the entire
  Persoonia corpus.
* **`taxon_d7ffc349...`** — noted 2026-07-07 from batch-2.
  **First lichen poster-child** (endolithic lichen — grows
  inside sandstone).  1780-char description + 440-char
  diagnosis opening `Description. Thallus endolithic up to
  0.2 mm deep, up to 5 cm wide, algal cells ca. 5-10 µm
  wide…`.  Period-form `Description.` header (same shape
  as taxon_d65547ed and taxon_62a712ab).  Real Nomenclature;
  18 Claude annotations, status = success, merge_metric = 1.
  All triage detectors correctly silent.  Adds a distinct
  clade shape to the §0.5 reference set — Thallus + algal
  cells + apothecia is anatomically different from the
  basidio/asco/asexual-mould shapes previously covered.
* **`taxon_d2a4c584...`** — noted 2026-07-07 from
  batch-2.  **Latin diagnosis + matching English
  description** — a legitimate single-species convention
  (`Basidiomata solitaria. Pileus 1.5-3 mm latus,
  campanulatus…` in Latin, followed by matching
  `Basidiomata solitary. Pileus 1.5-3 mm broad,
  campanulate…` in English).  2667-char description, 32
  Claude annotations, status = success, merge_metric = 2.
  **Detector-refinement moment**: this treatment exposed
  a false-positive class in `count_repeated_structural_
  anatomy` (M2 Group B).  Before 2026-07-07: `Basidiomata`
  at para 0 (Latin) + para 2 (English) fired
  §6:multi_structural_anatomy — the detector counted
  cross-language repetition as a species boundary.
  **Refined 2026-07-07 to be LANGUAGE-AWARE** (per
  operator note): score each paragraph as Latin or
  English via `_latin_ratio` and count repetitions
  independently within each language.  Fires when either
  Latin or English has ≥ 2 paragraph-start mentions.
  Cross-language pair (1 Latin + 1 English) doesn't fire.
  **Not English-only**: two separate Latin descriptions
  IS a merge signal and would have been silently dropped
  by an English-only filter — this hasn't been observed
  yet but is plausible.  taxon_572d470e (documented true
  positive) unaffected: its `Apothecia` repetitions are
  both English.  Fixture-tracked as a poster-child; new
  regression bar against structural-anatomy detectors
  over-flagging Latin+English paired treatments.
* **`taxon_e78904cb...`** — noted 2026-07-07 from the
  first M2-detector-suite batch (batch-2, 10 new
  treatments selected via
  `bin/select_for_annotation --n 10 --exclude-annotated`).
  Agaric (gilled mushroom, not a bolete —
  distinguishes from taxon_0cfe582f which is a
  boletoid).  1420-char clean single-species
  description opening `Pileus 4-8 cm broad,
  hemispherical to convex, then planoconvex,
  fibrillose, white to ochraceous white when
  young…`.  Real Nomenclature; no diagnosis field.
  merge_metric = 3; all triage detectors correctly
  silent.  13 Claude annotations, status = success.
  Operator: "another poster child."  Adds an agaric-
  clade shape to the §0.5 reference set, complementing
  the boletoid taxon_0cfe582f.
* **`taxon_62a712ab...`** — round-2 reviewed by
  piggy@puchpuchobs, noted 2026-07-03 as the last
  treatment inspected in the triage-CSV review pass.
  Plant-pathogen ascomycete on cherry (Prunus
  cerasus), 1402-char description opening
  `Description. Saprobic on decaying branches of
  Prunus cerasus. Sexual morph: Str…`.  Note the
  period-form `Description.` header (same as
  taxon_d65547ed) — still not detected by the
  current `_DESC_HEADER_RE`, filed under the
  detector-gap for period-terminated header form.
  **Reviewer data**: kept 8, added 2, deleted 0.
  Second review-confirmed poster-child (after
  taxon_0029f141's 16/0/0).  20% add ratio — normal
  for a clean single-species treatment where Claude
  gets most features and reviewer adds a couple.
* **`taxon_0029f141...`** — round-1 reviewed by
  piggy@puchpuchobs, noted 2026-07-03.  **First
  review-confirmed poster-child**: 16 Claude
  annotations, all 16 kept, 0 added, 0 deleted —
  perfect signal from both the annotator and the
  reviewer.  2037-char basidiomycete description
  opening `Description: Basidiomata small to medium-
  sized. Pileus 2.6–8 cm`, with a 331-char Diagnosis
  (Differential Diagnosis).  Single `Description:`
  header at offset 0 (correctly does NOT fire
  multi_description).  merge_metric = 9, just below
  the 10-threshold — validates the threshold
  calibration.  Stronger regression target than the
  detector-only poster-children: any future
  tightening that surfaces this treatment as a false
  positive would be regressing against a round-1
  review confirmation.
* **`taxon_0cfe582f...`** — noted 2026-07-02.  A
  substantial single-species bolete description (1948
  chars, 16 annotations).  Clean anatomical opening:
  `Basidiomata medium large sized, boletoid. Pileus
  30–80 mm diam, initially hemispheric …`.  All triage
  detectors correctly silent.  Real Nomenclature; no
  diagnosis field, which is fine (not every species has
  a separate diagnosis block).  Structured anatomy
  proceeds through the expected sections without
  clipping, header repetition, mid-body Nomenclature
  fragments, or OCR corruption.  Operator called it
  "a poster child — this is what an extract description
  should look like."
* **`taxon_9f0c4134...`** — noted 2026-07-07 from batch-2.
  Xylariales-type stromatic ascomycete (`Saprobic on
  fallen leaves of an unknown plant. Sexual morph:
  Stromata 1-4.2 cm total length, solitary, upright or
  prostrate, cylindrical, unbranched…`).  1405-char
  description, real Nomenclature, 7 Claude annotations,
  merge_metric = 0.  Period-form `Description.` header.
  Distinct clade shape from the other §0.5 ascomycetes
  (taxon_38b5b1c6 is plant-pathogen leaf-immersed;
  taxon_e534e6a9 is both-morphs; this one is stromatic
  Xylariales-type).  Operator: "another description
  poster child."
* **`taxon_38b5b1c6...`** — noted 2026-07-03.
  Plant-pathogen ascomycete opening with the proper
  anatomical noun preserved: `Ascomata separate,
  immersed in leaf tissue, becoming erumpent, …`.
  920-char description, 6 annotations, all triage
  detectors silent.  Useful as the **control case
  for the anatomical-noun-clip pattern** noted at
  taxon_7fbc71a8 and taxon_418bf6b7: this is what
  those two would look like if the extractor had
  preserved the anatomical noun.  Opens with `Ascomata`
  as the sentence subject rather than dropping to the
  adjective-only continuation (`immersed, scattered
  or in groups. Venter …`).  Operator: "another
  poster child."
* **`taxon_e534e6a9...`** — noted 2026-07-03.
  Ascomycete-with-both-morphs shape.  1304-char
  description opens `Saprobic on decaying wood. Sexual
  morph Undetermined. Asexual morph Colonies on …`.
  The explicit "Sexual morph / Asexual morph" pair
  (with one marked Undetermined) is a convention for
  ascomycetes that have alternating sexual and asexual
  reproductive stages — the treatment describes the
  known one and marks the unknown one explicitly, which
  is more informative than silently omitting.  8
  annotations from a modest description length; all
  triage detectors correctly silent.  Operator: "looks
  perfect."
* **`taxon_d65547ed...`** — noted 2026-07-03.  A clean
  asexual-mould / plant-pathogen shape (contrast with
  taxon_0cfe582f's bolete).  750-char description
  covering ONLY cultural characteristics — legitimate
  for taxa where the anatomy is essentially the colony
  (asexual moulds, yeasts, some plant pathogens).
  Opens `Description. Colonies on PDA approx. 6−7 cm
  diam. after 7 d at 25 °C, surface f…`.  Real
  Nomenclature; no diagnosis field.  Natural start and
  finish per operator.  All triage detectors correctly
  silent, though see the detector-gap note below.
  **Detector-gap observation**: description starts with
  the literal string `Description.` — with a PERIOD as
  the header terminator, not a colon or em-dash.  My
  `_DESC_HEADER_RE` regex matches only `\bDescription\s*
  [-–—:]` (colon, hyphen, en-dash, em-dash); a period
  slips past.  Not a bug in this true-negative case
  (offset-0 header is legitimate and doesn't need to
  fire the multi_description merge flag) but the same
  gap would let a mid-body `Description.` marker at
  offset > 0 through the `mid_body_description_header`
  detector.  Consider extending the regex to include
  `.` when followed by whitespace-then-capital-letter
  (period-terminated header form) — a straightforward
  addition to `treatments_to_structured/triage_signals.py`.
  Tracked as **D2** in the Detector backlog; blocked on a
  fixture for the mid-body case.
* **`taxon_fa7f4de6...`** — noted 2026-08-25 from round-4.
  *Acarospora indistincta* K. Knudsen, Hodková & Kocourk.,
  sp. nov. (*MycoKeys* 112).  A **squamulose** lichen, and
  the second *Acarospora* in the set.

  **Overlap declared**: taxon_d7ffc349 (`endolithic-lichen`)
  is the same genus by the same lead author, but the
  **opposite thallus habit**, and that is the whole point of
  keeping both.  An endolithic lichen lives *inside* the
  rock and has no upper cortex to describe; this one is a
  cushion of squamules sitting on top of basalt, so
  `Hypothallus`, `Upper surface`, `Lower surface`,
  `Epicortex`, `Cortex` and `Medulla` all exist here and
  cannot exist there.  Anything that learns "lichen ⇒
  endolithic vocabulary" from one exemplar is wrong, and
  this is the counterexample.

  Two registers not otherwise in the fixture:

  1. **Negative observations as first-class features** —
     `Pycnidia not observed.` and `Chemistry: not producing
     secondary metabolites.`  Both are annotated features
     whose content is an absence.  A slot-filler that
     expects a measurement or a colour gets a negation, and
     "not producing secondary metabolites" must not
     normalise to a chemistry *value*.
  2. **An internal inconsistency in the published source.**
     The diagnosis says the cortex is `(50–)90–100` µm; the
     description four paragraphs later says `(60–)90–100`.
     Ours is a faithful extraction of both — the
     disagreement is the paper's.  Worth knowing before
     someone builds a cross-field consistency check and
     reports it as an extraction defect.

  **The reason it is here rather than in the pathology
  half.**  At **21 annotations** it is the densest treatment
  reviewed that is not a merge, and I predicted before
  reading it that "the only one likely to be a multi-taxon
  case at that density."  That prediction was wrong, and
  the correction is load-bearing for D7 — see the
  confounder recorded there.  Everything else reads clean:
  seven prose fields all correctly routed, contiguous odd
  paragraphs 27–43, a genuine differential diagnosis,
  `merge_metric` 0, no triage flags, `synthetic_nomenclature`
  false, and `OcrDamage` silent on all three modes.

  It is also the round's only **new vocabulary**: the
  reviewer added `Squamules` for *"The majority of squamules
  are sterile."*, a label absent from both the candidate and
  hand databases.  See §12.1 — it is arguably a *slot* on
  `Thallus` rather than a feature of its own.

* **`taxon_3b7a80bc...`** — noted 2026-08-26 from round-5 T5
  review.  *Fomitiporia roseo-bubalina* (*MycoKeys* 118).  Operator:
  *"looks perfect to me, even the diagnosis is correct.  The figure
  captions were correctly removed from the flow.  More like this one,
  please."*

  Six spans, every one correctly labelled — nomenclature, diagnosis,
  type_designation, etymology, description, notes — `merge_metric` 0,
  no flags, 14 well-formed annotations.  The diagnosis is genuinely
  differential, comparing to *F. ovoidospora* with discriminating
  measurements rather than a Notes block in the diagnosis slot.

  **The first poroid polypore in the set**, and it brings vocabulary no
  other poster child has:

  1. **The dimitic hyphal-system triad** — `Hyphal_system` (*"dimitic,
     generative hyphae simple septate"*) with `Generative_hyphae` and
     `Skeletal_hyphae` as separate features.  Nothing else in the
     fixture has this, and it is the central diagnostic axis for
     polypores.
  2. **`Rot_type`** — *"Type of rot. White rot."*  A feature that is
     ecological rather than morphological, and unique here.
  3. **`Hymenial_setae` recorded as ABSENT.**  In Hymenochaetaceae the
     *absence* of setae is diagnostic, so this is a labelled negative
     carrying real information — the same register as
     `taxon_fa7f4de6`'s `Pycnidia not observed`.
  4. **`Basidioles` distinguished from `Basidia`.**

  **It is also the set's positive control for `Figure-caption`.** Both
  of its gaps are figure captions, correctly kept out of the treatment.
  Against §12.2's many cases where that same label swallowed a
  nomenclature heading or a severed Notes, a case where it did its job
  is worth pinning: the label is not broken, its *boundary decisions*
  are.

## Notes for fix sequencing

These issues are deferred — not blocking Phase 1 bootstrap-annotation
work in `treatments_to_structured/`.  Suggested triage order when
the work is picked up:

> **Detector work has moved.**  This list is the original
> memo pass and covers pipeline/extractor fixes.  Every
> *detector* proposal in this memo is now consolidated in
> **Detector backlog** below, with its gating fixture
> entries.  Item 2 below has since been shown insufficient —
> see the correction in that section.

1. **`pdf_url` etc. (§4)** — likely a one-line fix in the
   `extract_treatments_to_couchdb.py` ingest projection.  Quick
   win, no model retraining, immediately improves the Phase 1
   review workflow.
2. **False-positive treatments (§5)** — gate
   `synthetic_nomenclature` stub creation on a stronger signal
   (paper-level taxonomy heuristic, or "at least one real
   Nomenclature elsewhere in the doc").  Likely highest corpus-wide
   payoff per unit work; explains the ~half-with-complexity-score-0
   phenomenon the operator flagged.
3. **Multi-species merge (§6)** + **citation misses (§1, §2)** —
   both reduce to layout-CRF Nomenclature-heading recall.  Probably
   need to retrain with labelled examples of:
     * numbered-list species headings (`I.`, `1.`, `3.`)
     * formal-author-citation paragraphs that don't begin with the
       bare species name
   Biggest lift; plan alongside the next v4 model refresh.
4. **`key` contamination (§7)** + **biology/materials overflow
   (§3)** — downstream symptoms of treatment-grouper boundary
   failures.  Likely resolve naturally once §2 and §6 are fixed
   (better Nomenclature recall → better boundary signals).

## Detector backlog (consolidated)

Written 2026-08-21.  Until now the detector proposals in this
memo lived as prose scattered across §0.5, §5, §10, §12 and
§13, and the fix-sequencing list above — written in the
memo's first pass — referenced none of them.  This section
is the single list; §15 (added the same day) is written
against it from the start.  **Nothing here is implemented.**  Each
item names the fixture entries that gate it, so the work can
start from a red test rather than from prose.

Implementation is deferred until round-4 annotation is
finished.  The doc pass is deliberately ahead of the code so
that the gating cases are settled before anyone writes a
regex.

### Non-detector items deferred to the same point

Not detectors, but blocked on the same round-4 boundary and
worth carrying in one list rather than three.

**U1 — Merge Django's `.ann` fallback into `span_resolver`.**
`django/search/views.py` resolves span attachments through its
own `_collect_ann_db_candidates()` probe order, while
`span_resolver` is the single path everything else uses.  Two
implementations of one rule is exactly the
synchronised-by-comment arrangement both sides currently
carry a note about.

The gap that justified keeping them separate has since
closed: `span_resolver` gained the attachment-name fallback
(`FALLBACK_ATTACHMENTS`) that v3_hand needs, so it now
handles the case Django's chain existed for.  What remains
different is the **database** probe — Django also tries the
ingest DB and the experiment's `databases.annotations` for
older taxa docs, where `span_resolver` deliberately refuses.

Before merging, settle that difference on evidence rather
than taste:

* Both endorsed experiments carry `annotations_db` on
  **every** document — production_v4 0 of 81 527 missing,
  production_v3_hand 0 of 73 139 — so the strict rule costs
  nothing on supported data.  (Measured 2026-08-21; the
  archived `production` experiment was the only endorsed
  exception, at 25 420 of 31 319 missing.)
* So the question is only which *unendorsed* experiments the
  Django views must keep serving.  If none, the probe order
  can go entirely.

**Verification is already built**: run `bin/verify_spans`
against both endorsed experiments before and after, and
require the rates not to drop — 100 % for production_v4,
≥ 90 % for production_v3_hand with its documented gap.

**U2 — Put the D-items back in numeric order.**  ✅ **Done
2026-08-24.**  The file order had drifted to

```
D1 D2 D3 D4 D5 D6 D7 D8 D10 D12 D13 D14 D15 D11 D9
```

because items were appended by inserting before whichever
heading was convenient at the time, so later additions
landed ahead of earlier ones.  It was a live editing hazard,
not an aesthetic complaint: on 2026-08-23 an edit that
spliced on a `### D6` end anchor computed an end index
*before* its start index and silently duplicated the entire
D7 section, caught only by `grep -c` afterwards.

Reordered as a pure text permutation — verified by asserting
the block's character count, line count and multiset of lines
were all unchanged, then confirming the whole file's word
count held at 34 561.

**The standing rule survives the reorder**, because
anchor-based edits remain how this file is maintained and it
is now past 3 000 lines:

> **After any scripted edit to this memo, check heading
> uniqueness** — `grep -c '^### D<n> '` must be 1 for every
> item.

The reorder does not prevent the failure; it only removes the
surprise that made it plausible. A future `sed`/Python splice
can still compute crossed indices. The `grep -c` check is what
actually catches it.

**U3 — No deduplication of volumes, articles or treatments
(Trello #405).**  Raised by the operator 2026-08-23.  It is
not a latent risk; it is already the corpus's largest
structural defect by volume, and it **gates #404**.

*Measured 2026-08-23 against `production_v4` / `skol_dev`:*

| | count | share |
|---|---:|---:|
| ingest documents | 31 084 | — |
| …carrying a DOI | 19 918 | 64.1 % |
| …sharing a DOI with another document | **11 401** | **36.7 %** |
| distinct DOIs appearing more than once | 5 647 | — |
| treatments from a multiply-ingested article | **33 551** | **41.2 %** |

**The duplication is systematic, not incidental.**  5 552 of
the 5 647 duplicate-DOI groups — **98.3 %** — are exactly a
`crossref` copy and a `pmc` copy of the same article: the
publisher PDF and the PMC JATS XML, ingested independently.
The attachment sets confirm it, `article.pdf` against
`article.xml`.  So for thousands of articles we hold a
PDF-defective extraction *and* a JATS-defective extraction
of the same text — the §15 element-join defect on one side,
page headers and OCR on the other.

**The content hash is not a dedup mechanism, and it fails in
the worst direction.**  `taxon_id` is
`sha256` over the prose fields only
(`bin/extract_treatments_to_couchdb.py`), with **no source
identity in the hash**.  Consequences:

* *Byte-identical* treatments from two ingests collapse to
  one document — dedup we did not need, and it **silently
  overwrites provenance**: whichever ingest writes last owns
  the `ingest` pointer, so the treatment claims one source
  when two produced it.
* *Near-identical* treatments do not collapse at all.  A
  better scan, a different extractor, one changed character
  — different hash, genuine duplicate.

That is precisely backwards for both #404 and #405.

**How much is provably redundant.**  3 688 duplicated DOIs
have treatments from **both** copies, giving a lower bound of
**7 389 redundant treatments (9.1 %)**.  Normalising
descriptions (lowercase, strip non-alphanumerics) and hashing
finds only **2 792 (3.4 %)** — so **normalisation-based
dedup catches under 40 % of even the provable cases**, and
under 10 % of the at-risk population.  The two extractions
differ by more than punctuation.

**Dedup must choose, not merely drop.**  No source dominates:

| copy carrying more description text | articles | share |
|---|---:|---:|
| `?` (neither field set) | 1 647 | 44.7 % |
| `crossref` | 1 287 | 34.9 % |
| `pmc` | 754 | 20.4 % |

A static "prefer PMC" rule would discard the better
extraction on **79.6 %** of duplicated articles.  The
selection needs a per-article quality signal, not a source
ranking.

*(Context, so the empty-description figures below are not
misread: **48.4 % of all treatments corpus-wide have an
empty `description`**, median length 141.  Within duplicated
articles the rate is 70.2 % for `crossref` and 56.2 % for
`pmc` — elevated — but 36.3 % for `?`, which is **better**
than baseline.  Empty descriptions are a pre-existing
corpus-wide condition, not something duplication created.)*

**The gating interaction with #404.**  DOI is the obvious
dedup key, and it is **blind to exactly the population #404
targets**: the whole-volume *Persoonia* documents are
defined by having *no title and no DOI*.  So

* a DOI-keyed dedup cannot see them — and the existing
  DOI-keyed dedup on the Plazi backfill path already cannot;
* ingesting *Persoonia* from Naturalis without either
  landing #405 first or explicitly retiring the old
  documents adds a **third** copy of volumes 1–19 rather
  than replacing them.

#405 therefore needs at least two keys — DOI where present,
and a content/title/volume key where it is not — or #404
must carry its own explicit retirement step.  Sequence the
two deliberately; they do not compose by themselves.

### What already fires

`treatments_to_structured/triage_signals.py` emits 18 flags
today: `§2:synth_nomen`; `§6:` `multi_diagnosis`,
`multi_description`, `multi_section_header`,
`multi_structural_anatomy`, `mid_body_desc`, `multi_sp_nov`,
`latin_alt`, `latin_ele`, `authored_binomial`,
`merge_metric=<N>`; `§8:` `key_content_short`,
`key_couplets`; `§10:` `mid_sentence`, `tail_clip`,
`diag_head_clip`; `§12:desc_span_gap`;
`§13:no_source_anchor`.

The regression bar is `tests/fixtures/pathologies.json` (13
poster children that must fire nothing, 28 pathologies that
must fire exactly their labelled set) driven by
`tests/pathologies_test.py`.

### D1 — Anaphoric-subject description (§5)

**Catches**: prose *about* taxa mis-extracted as prose
*describing* a taxon.  Introduction summaries, discussion
paragraphs, comparative recaps.

**Gating fixtures**: must fire on `taxon_0a8c1077`
(`§5-front-matter-summary-in-taxonomic-paper`, anaphora) and
on `taxon_76679aa3`
(`§2-nomenclature-is-host-plant-fragment`, authorial voice).
Must stay silent on all 16 poster children.

**The obvious formulations are all disqualified by the
reference set** — this is the reason to write the item down
rather than just implement it:

* *Plural subject* fails.  6 of 13 poster children open
  their first sentence with a plural anatomical noun:
  `Basidiomata medium large sized…` (taxon_0cfe582f),
  `Basidiomata small to medium-sized` (taxon_0029f141),
  `Basidiomata solitaria` (taxon_d2a4c584), `Ascomata
  separate, immersed…` (taxon_38b5b1c6), `Conidiomata
  synnematous…` (taxon_06594607), `Colonies on PDA…`
  (taxon_d65547ed).  Grammatical number carries no signal
  here.
* *No binomial in the description* fails.
  `authored_binomial_in_desc` is False for **13 of 13**
  poster children.  It is not a discriminator at all.
* *`synthetic_nomenclature = True`* is clean against the
  reference set (**0 of 13**) but is just §2 re-fired, and
  §2 is true of 1 884+ corpus treatments on its own — it
  can weight the signal, not be it.

**What is left is register** — two markers, both measured
clean against the reference set.

*Anaphora*: back-reference to material outside the
treatment.  `All seven species **mentioned above**`,
`**None of these** species`, `most of **them**`.  That, not
plurality, is what separates taxon_0a8c1077 from every
poster child.

*Authorial voice and attribution*, added 2026-08-23 from
`taxon_76679aa3`: a handbook commentary opening `In
Melampsora, all morphological basic types of spores and sori
are formed **that we also know** from other rust fungi.
**After Cummins & Hiratsuka (2003)**, spermogonia are
subcuticular …`.  It carries real anatomy — nine
annotations — but describes the genus in narrative rather
than diagnostic register.  Anaphora would not catch it; the
first person and the attributive citation do.

Measured across the fixture, first-person and attributive
markers (`we`/`our`/`us`, `After <Author>`, `viz.`) appear
in **0 of 16 poster children** and **5 of 39 pathologies**.
A clean description never uses them.

Formulate as: the first sentence refers to an antecedent
outside the treatment, **or** the passage speaks in the
authors' voice rather than describing a specimen.

**Depends on**: nothing.  Cheapest item here and the only
one with a live case that reached a reviewer's queue.

### D2 — Period-terminated section headers (§0.5)

**Catches**: `Description.` and `Diagnosis.` used as headers
with a period terminator.  `_DESC_HEADER_RE` /
`_DIAG_HEADER_RE` currently match only `[-–—:]`.

**Gating fixtures**: must stay silent on `taxon_d65547ed`
(`asexual-mould-culture-only`) and `taxon_62a712ab`
(`ascomycete-review-confirmed`) — both open with a
legitimate offset-0 `Description.` that must not fire
`§6:multi_description`.

**Blocked on a missing fixture.**  The case the widening
exists to catch — a **mid-body** `Description.` at offset
> 0, which should fire `§6:mid_body_desc` — has no entry.
Capture one before implementing, or the change ships with
only negative tests.

**Don't regress the delimiter class.**  Both regexes spell
the gap as `\s*`, and Python's `\s` matches U+202F NARROW
NO-BREAK SPACE, which is the Persoonia house form
(taxon_06594607, taxon_09b97d5f).  A rewrite that
hard-codes an ASCII space would silently drop the entire
Persoonia corpus.

**Depends on**: one new fixture capture.

### D3 — Comparative language in the description tail (§12)

**Catches**: a Diagnosis block leaked into the end of
`description`.  Apply the gnfinder / comparative-language
signals from §6 idea #2 (`differs from …`, `similar to …`,
authored binomials) to the **last N characters** of
`description` only.

**Gating fixtures**: must fire on `taxon_d2d26d25` — which
**has no fixture entry yet** (see §12; currently no detector
fires on it at all: merge_metric = 3, diagnosis field empty,
description does not start mid-sentence).  Capture it first.

Must stay silent on `taxon_b9a623297`
(`§6-false-positive-detailed-single`) and must not add flags
to `taxon_9e048013`
(`§6-false-positive-comparative-diagnosis`), which is
already over-flagged.  `taxon_09b97d5f`
(`bolete-Pileus-opener`) is the poster-child control: its
*diagnosis* is dense with authored binomials and BLAST
comparisons while its *description* is clean, so a detector
that reads the wrong field will show up there.

**Note**: the target fragment is a **Differential
Diagnosis** in English, not Latin.  The Latin-morphology
heuristics (`§6:latin_alt`, `§6:latin_ele`) do not apply.

**Depends on**: one new fixture capture.

### D4 — Mid-body species boundaries (§6/§10)

**Catches**: a second species starting inside `description`
with no section header.

**Status: a regex attempt was written and reverted.**  It
fired 3× on `taxon_b9a623297` — on `phragrnosporous`, `a
diameter of…`, `submembranaccous`, all legitimate
continuation prose within one species.  taxon_b9a623297 is
the standing disqualifier: **any detector that fires on it
is rejected.**  The reverted attempt also would not have
caught `taxon_9ecad903`, its motivating case, because that
species boundary is a trailing hyphen rather than a period.

**Conclusion already reached in §10**: this needs
paragraph-level section classification (the M3 segment
classifier), not a regex.  Do not re-attempt at the regex
layer.

**Depends on**: M3 segment classifier.  `taxon_9ecad903`
also needs a fixture entry.

#### The plate-reference heuristic: right reasoning, measured, does not generalise

Raised by the operator 2026-08-24 on `taxon_a686d7ab`: *"the '(Pl. II)'
below seems to indicate that there are only 2 descriptions here that
have been mangled."*

**The reasoning is correct and it settled the case.** In older
monographs each species treatment carries its own plate reference, so
counting distinct plate numbers bounds the number of species in a
merged block. Here exactly two occur — `(Pl. I)` on *Collybia
olympiana* sp. nov. and `(Pl. II)` on *C. badiialba* Murr. — so the
several description-like passages between them are one species'
scattered text plus comparative discussion, not further species.

**As a corpus detector it is not worth building.** Measured over
39 887 descriptions ≥ 200 chars:

| | count | share |
|---|---:|---:|
| containing ≥ 1 plate reference | 28 | **0.1 %** |
| containing ≥ 2 distinct plates | **3** | 0.0 % |

and all three multi-plate treatments **already carry** a `§6` or
`§12:desc_span_gap` flag, so it would add nothing to detection either.

Recorded so nobody builds it. It stays valuable as a **review
heuristic** — a fast way for a human to bound a merge — which is
exactly how it was used.

**One trap if it is ever revisited: OCR renders `Pl.` as `PI.`**
(capital i, not lowercase L). A first pass with `\(Pl\.` returned zero
matches on a treatment that plainly contains two. Any plate or figure
regex needs both forms.

**The wider point about this treatment.** Its two species are not
merged by a missed heading — they are **shredded**. The English
descriptions of *both* run through `Table`, `Key`, `Figure-caption` and
`Misc-exposition` blocks, and only 8 fragments out of roughly 40 blocks
reached the `Description` field. The cause is upstream of the layout
classifier: the OCR is bad enough that the text does not read as prose
— `Pilous` for *Pileus*, `i m< ta l< trongly farin ( om lamellae
adnate`. Same shredding, same old-scanned-book origin, as
`taxon_a5efbd0b` (§5.2).

That is worth separating from ordinary §12: a swallowed block is a
classifier error on legible text, whereas this is the classifier
behaving reasonably on input no classifier could parse. The fix is
better OCR or source replacement, not a better label model.

### D5 — Label-aware assembly (§12)

Not a detector.  Pass segment-level `(section_label, text)`
tuples through to assembly instead of flat field dicts.
Subsumes several detectors by construction: Diagnosis blocks
stop leaking into `description`, header multiplicities
become species-count signal without a separate detector, and
`Key` segments route out of `description`.

**Depends on**: a schema change through `treatments_prose`
(currently flat).  The v4 layout CRF already emits per-line
section labels; the plumbing exists and is discarded at
assembly.  Phase 3+.

**Sequencing consequence**: D4 and D5 reduce to the same
underlying capability.  Attempting D4 standalone is what
produced the reverted regex.  Sequence them as one piece of
work.

### D6 — Element-join artifact (§15)

**Catches**: words run together at JATS element boundaries —
`Descriptionof`, `Typeof`, `Notes.The`,
`Asteromellapistaciarum`.  Marker: `[a-z]\.[A-Z]`, plus a
gnparser round-trip on candidate binomials.

**Gating fixtures**: must fire on `taxon_30d8d8d4`
(`§15-jats-element-join-no-space`).  Must stay silent on all
13 poster children and on the `pdf`-sourced pathologies,
whose run-together text is §9 OCR corruption with a different
cause and a different fix.

**Two jobs, and the detector is the smaller one.**  The
detector is nearly trivial and worth having as a corpus
health gauge.  The actual repair is upstream, in
`extract_text()` — scope a separator to block-level
boundaries (`<sec>`/`<title>`/`<p>`), then handle inline
`content-type="taxon-name"` children with their own rule.
Never blanket-insert: `H<sub>2</sub>O` must stay `H2O`.

**Why this outranks its apparent severity.**  It fires
nothing today and it blinds gnfinder — `Asteromellapistaciarum`
returns `[]` where `Asteromella pistaciarum` returns the name
at oddsLog10 13.2.  On the 2 629 affected treatments (96.1 % of
the plazi-only path), `§6:authored_binomial` and every
name-based §1/§2 signal is not wrong but *absent*.  Any
measurement of name-detection recall taken over the whole
corpus is currently reading those treatments as
"no names present."

**Ordering note**: this is a data-quality precondition for
§1/§2 name-recall work, not a peer of it.  Fixing the CRF's
Nomenclature recall while 6 % of the corpus has unparseable
names in it will produce a misleading evaluation.

**Depends on**: a re-extraction.  Unlike every other item
here, the fix changes `article.txt` and therefore every
downstream field and every stored `*_spans` offset.  Sequence
with a planned re-extraction, not as a hot patch.

### D7 — Repeated feature labels in the annotation set (§12/§6)

**Catches**: multi-species merge, using Claude's own
annotations instead of text heuristics.  A single-species
treatment names each top-level feature about once; a merged
one repeats them.

**Gating fixtures**: must fire on `taxon_3d0a3c69`
(`§12-multi-genus-fragment-scatter`, 5 × `Pileus` +
5 × `Lamellae`), on `taxon_3d9f50f8`
(`§6-two-species-two-culture-blocks`, `Conidia`,
`Conidiogenous_cells`, `Conidiophores`, `Conidiomata` and
`Culture_characteristics` all ×2 — the low-multiplicity end,
and the harder test), on `taxon_5581a442`
(`§6-genus-description-merged-heading-as-Table`, `Mycelium`,
`Conidiophores`, `Conidia`, `Chlamydospores` ×2), on
`taxon_60758ef3`
(`§6-numbered-species-heading-as-ToC-entry`, `Pileus`,
`Lamellae`, `Spores`, `Stipe` ×2) and on `taxon_2b793602`
(`§8-flora-chapter-slice-unnumbered-key`, 46 × `Pileus`,
40 × `Lamellae`, 16 × `Stipe`, 12 × `Spores`).

**Why it is worth having**: on taxon_3d0a3c69 the text-based
§6 detectors read merge_metric = 1 and
`n_repeated_structural_anatomy` = 0 while Claude's
annotations show the merge outright.  The signal is already
in data we store; nothing new needs extracting.

**The confounders are real and all four are in the
fixture** — which is why this is a backlog item and not a
two-line patch:

* **Latin/English pairs.**  `taxon_d2a4c584`
  (`agaric-latin-english-pair`) has **15 labels at exactly
  2 ×**, by construction — Latin then English.  A naive
  threshold of 2 fires on a poster child.
* **Macro/micro splits.**  `taxon_0029f141`
  (`bolete-review-confirmed`) has `Basidiomata`,
  `Hymenophore`, `Pileipellis` and `Pileus` each at 2 ×.
  Also a poster child.
* **OCR span-splitting.**  `taxon_95dbdfb9`
  (`§9-interstitial-OCR-noise`) shows 3 × for six different
  labels purely because noise fragments each span.  Real
  repetition, wrong cause.
* **Existing §6 false positives.**  `taxon_9e048013` shows
  `Stipe` 8 ×, `Spores` 6 ×.  A signal that fires there
  makes a known false positive worse.
* **Compilatory entries — the confounder that would sink
  it.**  `taxon_6c2dcb7c`
  (`§6-compilatory-double-description`) repeats **seven**
  labels exactly ×2 — `Vegetative_hyphae`, `Sexual_morph`,
  `Asexual_morph`, `Conidiophores`, `Conidiogenous_cells`,
  `Conidia`, `Chlamydospores` — the same signature as the
  genuine merges above.  But it is **one genus described
  twice from two literature sources**, `(Carmichael 1966)`
  and `(de Hoog et al. 2000a, Najafzadeh et al. 2010a, b)`,
  not two taxa.  D7 as drafted fires and is wrong.  The
  outline-paper genre this comes from is large and growing,
  so this is not a rare edge case.

  **The discriminator is in the text, not the counts**: a
  compilatory block ends with a parenthetical source
  citation.  Requiring that *no* repeated-label group is
  terminated by a `(Author year)` citation would separate
  these cleanly — and is worth building into D7 from the
  start rather than bolting on after the first false-positive
  report.

  Inline citations are cheaply and reliably detectable —
  13.6 % of descriptions, zero false positives on
  adversarial anatomical probes.  **D13** carries the
  measurement, the detector and the reasoning for not
  making it an annotation label; this discriminator
  depends on it.

* **Neither annotation count nor distinct-label count is a
  merge signal — only repetition is.**  Recorded 2026-08-25
  because I got this wrong in review: seeing 21 annotations
  on `taxon_fa7f4de6` I predicted a multi-taxon merge, and
  it is a clean single lichen (§0.5).  Ranking all 106
  reviewed treatments by distinct-label count puts it
  **level with a confirmed merge**:

  | treatment | distinct | max repeat | n |
  |---|---:|---:|---:|
  | `taxon_572d470e` — merged | 21 | 10 | 105 |
  | `taxon_fa7f4de6` — clean | **21** | **1** | **21** |
  | `taxon_592128a8` — merged | 17 | 22 | 169 |
  | `taxon_2b793602` — merged | 12 | 50 | 155 |

  The two 21s are indistinguishable on distinct count, and
  the two worst merges in the corpus score **below** the
  clean lichen on it.  Anatomically deep clades — lichens
  especially, with their epicortex/cortex/algal-layer/
  medulla/hymenium stack — simply name more distinct
  features than an agaric does.  `max repeat` separates all
  four cleanly; `n` separates three of four.

  So the candidate metrics below must be built on
  **repetition**, and any form with distinct-label count in
  the numerator is disqualified before implementation.

So the usable form is not a raw count.  Candidates: repeats
per 1000 characters; ratio of max label count to distinct
label count; or a count that collapses the Latin/English
pair the way `count_repeated_structural_anatomy` already
does via `_latin_ratio`.  Settle it against those four
before implementing.

Note that the second candidate — **max ÷ distinct** — is the
one the table above endorses: it reads 10/21 = 0.48 on the
merge and 1/21 = 0.05 on the clean lichen, while
`taxon_2b793602` reads 50/12 = 4.2.  Three orders of
separation with no threshold tuning.

**Depends on**: nothing — annotation counts are already
stored.  But it is evaluated on the annotation set, so it is
a post-annotation triage signal rather than part of
`treatment_signals` as currently shaped.

**Related, and cheaper**: `treatment_signals` reads only
`description` and `diagnosis`.  taxon_3d9f50f8 shows
`count_repeated_section_headers` returning 1 on that doc's
`etymology` field while the treatment scores clean — two
`Etymology.` blocks naming two different honorees.  Running
the existing header counter over `etymology`, `notes` and
`materials_examined` costs nothing and would have caught it.
Worth doing before D7.

### D8 — OCR space-displacement rejoin rate (§9)

**Catches**: scanned-typescript OCR where spaces land inside
words.  Invisible to the existing §9 detector, which keys on
U+FFFD; taxon_43a7b19e contains **zero** U+FFFD.

**Metric**: fraction of tokens inside a run of 2–4 tokens
that is out-of-vocabulary piecewise but forms a dictionary
word when concatenated.  `f ung i` → *fungi*, `wi t h` →
*with*, `hyal ine` → *hyaline*.

**Gating fixtures**: must fire on `taxon_43a7b19e`
(`§9-ocr-space-displacement`).  Must stay silent on all 15
poster children.

**Already measured across the fixture set** — this is the
one backlog item that arrives with its threshold evidence:

* taxon_43a7b19e: **19.13 %**
* next-highest of any kind: **1.50 %**
  (`§1-head-of-Description-citation`)
* all 15 poster children: **0.00 %**
* non-zero at all: 5 of 47 entries

Any threshold in 2–19 % separates.  Pick nearer the bottom
of that range only after checking more OCR-era treatments;
the fixture has few scanned typescripts.

**Rejected alternatives, also measured** — record these so
they are not retried:

* *Orphan single-letter rate* — 12.8 % here, but 15.7 % on
  `§10-head-clip-only` and 13.1 % on the `discomycete`
  poster child.  Measurement notation (`n = 20`, `L/W`,
  `3–5 µm`) produces orphan letters in perfectly clean text.
* *Out-of-vocabulary rate* — 25.0 % here, against a 16–47 %
  range across poster children.  `agaric-latin-english-pair`
  reaches 46.7 % because Latin is not in an English
  dictionary.  Unusable without a Latin-aware vocabulary.

#### Counter-example: `taxon_8d815304` — D8 does not fire, and should not

Added 2026-08-23.  The operator flagged it for "serious OCR
problems", and the OCR *is* severe — `Coremiellu`,
`Oidiode11dron`, `RaiUo`, `KJocker`, `flal'IIS`,
`a•errocosa`, `srriato`, `comb. 11011.`, `fJfiiiiiOrum`,
`Pe11icillium stria·111m`, `Molbronclrea`.  **The rejoin
metric reads essentially baseline anyway:**

| | `taxon_8d815304` | `taxon_43a7b19e` | corpus median | corpus p90 |
|---|---:|---:|---:|---:|
| pairwise rejoin | **2.4 %** | 8.7 % | 0.0 % | 2.1 % |
| windowed rejoin | **2.2 %** | 23.5 % | 0.0 % | 5.3 % |

The damage here is **character substitution**, not space
displacement — a different mode of §9, and D8 is scoped to
the latter.  **Do not add this treatment to D8's gating
fixtures**; it belongs to a character-substitution detector
that does not exist yet.  Its value to D8 is as the proof
that firing on "severe OCR" in general is out of scope: a
treatment can be badly corrupted and correctly invisible to
this metric.

**Pairwise windows are too narrow regardless.**  This
treatment breaks one word five different ways — `Anamorph`,
`Ana mo rph`, `A n am o r ph`, `Ana morph`, `Anamorphs` —
and pairwise rejoin cannot see any of the splits, because no
adjacent *pair* of those fragments is a word.  `Ana mo rph`
needs a 3-token window and `A n am o r ph` a 6-token window.
Widening to a 5-token window recovers `Ana mo rph` →
*anamorph* and lifts `taxon_43a7b19e` from 8.7 % to 23.5 %
against a corpus p99 of 16.7 %, so **the windowed form
separates better and should be the one implemented.**

**Two more alternatives measured and rejected** — recorded
alongside the two above so they are not retried:

* *Out-of-vocabulary rate against the production vocabulary*
  (corpus df ≥ 50 + botanical Latin + `wamerican` +
  `wbritish`, 111 833 forms) — 31.6 % here against a corpus
  median of 11.0 % and p90 of 27.0 %, which looks usable
  until you read the list.  It is dominated by **legitimate
  proper nouns**: genus names (*Cephalotheca*,
  *Pseudogymnoascus*, *Monascus*, *Monascella*,
  *Eremascus*, *Renispora*, *Hamigera*, *Sagenoma*), author
  names (Stolk, Eidam, Cohn, Arx, Chesters, Rajendran) and
  journal abbreviations (Bakt, Naturk, Soc, Beitr,
  ParasitKde).  Nomenclature-dense text scores high
  *correctly*.  Unusable without proper-noun suppression —
  the same contamination trap as the `Genus Epithetii`
  regex in §2.5.
* *Intra-word corruption rate* — a digit inside a letter
  run, an interior capital after a lowercase, an interior
  middle-dot or tilde.  Reads 5.5 % here and 3.6 % on
  `taxon_43a7b19e`, against corpus median 0.6 % and p90
  4.8 %.  Directionally right and the closest thing to a
  character-substitution signal so far, but it false-fires
  on **figure references** (`Fig. 2C`, `6A`) and on
  **micrometre measurements** written `3um` / `4-7um`.  Both
  are cheap to exclude; do that before trusting the number.

**Depends on**: a word list — and it must be **both**
`wamerican` *and* `wbritish` (operator, 2026-08-21: "not all
authors in English use American English").  Neither is a
declared dependency today; `wamerican` happens to be
installed on the dev box and `wbritish` is not.  Per CLAUDE.md
a missing package on production is a packaging error, so add
both to the deb dependencies, or vendor a merged list, before
this ships.

**Why both, measured.**  The reason is *recall*, not
precision.  `colour`, `odour`, `centre`, `localised`,
`characterised`, `utilisation` and `colourless` are all
absent from `american-english`, and the fixture descriptions
already contain `centre`, `colour`, `localised` and `odour`.
An OCR split of a British spelling therefore fails to rejoin
— `col our` → *colour*, `odo ur` → *odour*, `charac terised`
→ *characterised* are all unrecoverable against the American
list alone — so American-only would silently under-report
corruption in British-spelled treatments, which is the
opposite of what the detector is for.

Precision does **not** degrade.  Re-running the measurement
with the American list plus ~7 100 generated British-form
variants: taxon_43a7b19e stays at **19.13 %** and **every
poster child stays at 0.00 %**.  The proxy list deliberately
included junk (it generated non-words like `vere` and `pre`),
and the separation still held — so a real curated `wbritish`
is safe *a fortiori*.

One caution the proxy surfaced: `vere` is not a British word,
it is the OCR corruption of *were* sitting in taxon_43a7b19e
itself.  A word list containing junk tokens can mask exactly
the corruption being hunted.  Use the packaged lists, never a
generated or rule-derived one.

**Latin is a third requirement, and it is not optional.**
Operator, 2026-08-21: the `hunspell-la` package carries a
Latin word list, "but we might need to transform it to make
it work with wamerican and wbritish."  Both halves of that
are right, and the need is now measured.

Latin is effectively invisible to an English dictionary: the
Latin half of taxon_d2a4c584's description is **79.5 %**
out-of-vocabulary against `american-english`, versus 23.0 %
for its English half.  So D8 cannot see corruption in Latin
at all.  Injecting identical simulated space-displacement
into both halves of that one treatment — splitting 40 % of
words of 6+ characters — gives:

| half | rejoin rate after identical corruption |
|---|---:|
| English | **18.18 %** — detected |
| Latin | **1.33 %** — **missed**, below the 2 % floor |

A Latin diagnosis could be as badly damaged as
taxon_43a7b19e and score clean.  Given how many protologues
in this corpus carry Latin diagnoses, D8 without Latin
vocabulary is a detector with a systematic hole in exactly
the oldest, most OCR-damaged material.

**The transform.**  A hunspell dictionary is not a word
list.  `/usr/share/hunspell/en_US.dic` shows the shape:
line 1 is an entry count (`79013`), and every entry is
`word/FLAGS` (`A/SM`, `0th/pt`) where the flags index affix
rules in the companion `.aff`.  For set-membership use it
needs the count line dropped, `/FLAGS` stripped, and —
the part that matters — **the affixes expanded**.  Latin is
heavily inflected; bare stems will not match `ascis`,
`ascorum`, `basidiis`.  `unmunch` from `hunspell-tools` is
the standard expander, though it is known to struggle with
complex affix files, so the expansion needs spot-checking
against real diagnoses rather than trusting the output size.

Two further normalisations this corpus specifically needs,
because its Latin is 19th-century rather than classical:
`æ`/`ae` ligature folding, and `j`/`i` plus `v`/`u` variants
(`ejus`/`eius`).  Fold in the word list, not in the treatment
text — altering the text would corrupt the very offsets the
spans depend on.

**Vocabulary source — measured on puchpuchobs
2026-08-21, after installing `dict-freedict-lat-eng`.**

**FreeDict Latin does not work.**  The package is a
**2 305-headword general *classical* Latin dictionary**, and
this corpus needs descriptive botanical Latin.  Measured
against the 82 Latin word forms in taxon_d2a4c584's Latin
half, it covers **5** — `altus`, `caro`, `cavus`, `latus`,
`raro` — i.e. **6.1 %**, against the ~20 % the coverage sweep
says is needed.  The corrupted-Latin rejoin rate stays at
**1.33 %**, exactly where it was with no Latin vocabulary at
all.  `basidiomata`, `acanthocystides`, `adscendentes`,
`brunneovinescens`, `amyloideus`, even `albus` are all
absent.  Do not plan on it.  (`hunspell-la` is not in this
box's index either, and `hunspell-tools` is not installed.)

Two notes on the extraction one-liner, now that both files
can be compared directly: scraping the `.dict` body yields
**3 670** tokens against the `.index` file's **2 296**
headwords, and the extras are English gloss words — `abbess`,
`abbey`, `abbot`, `abdication`, several with trailing commas
(`abbreviate,`, `abhor,`).  In practice this is **inert** for
D8: the English words are already in `wamerican`, and the
comma-suffixed tokens can never be looked up because the
tokenizer is `[A-Za-z]+`.  The `.index` file is still the
cleaner source.  Headwords also carry macrons (`abscīdō`),
so fold to ASCII — in the *word list*, never in the treatment
text, or the span offsets break.

**A corpus-derived vocabulary does work, and is better on
every axis.**  Collect out-of-vocabulary forms of 4+
characters from `description` + `diagnosis` across the whole
`treatments_prose` DB, keep those with a document frequency
above a threshold, and union with `wamerican`.  Scanning
46 046 treatments (holding out taxon_d2a4c584) gives 308 110
distinct OOV forms.  Sweeping the threshold, measuring Latin
coverage, the corrupted-Latin case, taxon_43a7b19e, and the
worst false positive across all 15 poster children:

| df ≥ | forms | Latin cov. | corrupt-Latin | taxon_43a7b19e | worst poster child |
|---:|---:|---:|---:|---:|---:|
| 2 | 99 806 | 97.6 % | 16.00 % | 13.62 % | 6.59 % |
| 3 | 62 209 | 96.3 % | 18.67 % | 15.61 % | 6.59 % |
| 5 | 37 332 | 96.3 % | 21.33 % | 16.45 % | 1.23 % |
| 10 | 19 478 | 90.2 % | 24.00 % | 18.82 % | 1.23 % |
| 25 | 8 280 | 76.8 % | 26.67 % | 20.43 % | 1.23 % |
| **50** | **4 269** | **62.2 %** | **22.67 %** | **22.19 %** | **0.00 %** |

**Use df ≥ 50.**  Both corruption cases clear 22 % while
every poster child sits at 0.00 % — a wider margin than the
English-only measurement had.  Low thresholds are actively
worse: at df ≥ 2 the vocabulary is large enough to create
spurious rejoins and the worst poster child reaches 6.59 %,
uncomfortably close to the signal.

Three consequences:

* **The Latin packaging problem disappears.**  No
  `hunspell-la`, no `dict-freedict-lat-eng`, no `unmunch`, no
  `libhunspell` binding.  At df ≥ 50 the artifact is **4 269
  forms** — small enough to vendor in the repo, which also
  makes it reproducible and reviewable.
* `wamerican` + `wbritish` are still the base list and still
  need declaring; the corpus vocabulary is the technical and
  Latin layer on top.
* **Circularity caution.**  The vocabulary is derived from a
  corpus that contains OCR-corrupted treatments, so a
  systematic corruption could enter it.  df ≥ 50 is the
  guard — a corrupt form must recur across 50+ distinct
  treatments to survive — and it is the same `vere` hazard
  recorded above, so the generated list needs spot-checking
  before it is trusted.  Regenerate it from a
  §9-mode-B-filtered corpus once D8 itself exists.

**Note on scope**: firing this should mean *reject the
treatment*, not *flag for review*.  taxon_43a7b19e already
fires six flags and still reached a reviewer's queue.  A
seventh flag changes nothing unless it gates admission.

**PRODUCTION VOCABULARY, settled 2026-08-21** —
`data/corpus_vocabulary.txt` + `data/botanical_latin_wordlist.txt`,
**9 472 forms, about 100 KB**, both checked in with citations.

| vocabulary | size | Latin cov. | corrupt-Latin | taxon_43a7b19e | worst poster child |
|---|---:|---:|---:|---:|---:|
| English only | — | 0.0 % | 1.33 % ✗ | 19.13 % | 0.00 % |
| corpus df ≥ 50 | 4 270 | 63.4 % | 22.67 % | 22.19 % | 0.00 % |
| botanical Latin | 5 679 | 26.8 % | 13.33 % | 19.13 % | 0.61 % |
| **both — production** | **9 472** | **67.1 %** | **24.00 %** | **22.19 %** | **0.00 %** |
| + systematic-names | 10 171 | 68.3 % | 24.00 % | 22.34 % | 0.00 % |

**Whitaker's *WORDS* is dropped from the analysis.**  899 973 forms
and 11 MB to reach 14.67 % on corrupted Latin, while *lowering*
`taxon_43a7b19e` from 19.13 % to 16.22 % by absorbing real
corruption.  `bin/whitakers_wordlist.py` and its 20 tests are
**kept** — the generator still reproduces the list, and its AGE-filter
result (that `G`/`H` is Latin for telephones and fax machines, not
botanical Latin) is worth keeping — but the output is gitignored.

The two production sources have **independent failure modes**, which
is the point of pairing them.  They overlap by only 477 forms, 11.2 %
of the corpus list.  The corpus list can absorb OCR corruption
recurring across 50+ documents and carries English, French and
truncation debris; the botanical list is verified against the Ray
Society facsimile of Linnaeus but is only 5 679 forms and, used
alone, is the one vocabulary that puts a poster child above zero
(0.61 %).  Together: 0.00 %.

The systematic-names list is **optional** — 1.2 points of Latin
coverage for a CC BY-SA obligation, so it is left off the critical
path.

**Licence note**: the botanical list is free-use with a standing
**no-charge** condition (see its citation).  Free distribution — PyPI,
a public git remote — is unaffected, but the repo-split and packaging
plan must carry the condition forward rather than assume a permissive
project licence covers it.

### D9 — Head-clip on an opening parenthesis (§10)

**Catches**: a description or diagnosis whose first character is
`(` or `[` — the tail of a sentence that began earlier.

**Gating fixture**: must fire on `taxon_4a5306ac`
(`§10-diag-head-clip-open-paren`), whose diagnosis opens
`(more than 100 basidiomata are present in the type
collection), allowing a detailed study …`.

**The fix is one character.**  `desc_starts_mid_sentence`
fires on a leading character in `;,.:` or a lowercase
letter.  `(` is neither, so `§10:diag_head_clip` stays
silent on a plainly clipped field.  Add `(` and `[` to the
set.

**Safe against the whole regression bar, checked.**
Surveying the opening character of every `description` and
`diagnosis` in the fixture: none opens with a parenthesis,
so nothing currently passing starts firing.  The smallest
item on this list, and the only one that is a one-line
change with its gating case already captured.

**Depends on**: nothing.

### D10 — Genus mismatch between nomenclature and description (§2)

**Catches**: a treatment whose `description` names a
different genus from the one in its nomenclature — the
treatment-grouper missed a genus heading and attached the
description to the previous taxon.

**Gating fixture**: must fire on `taxon_4b89d160`
(`§2-wrong-genus-nomenclature`), whose nomenclature reads
*Pseudonectria* while the description opens `Type species:
Stylonectria applanata Höhn. 1915.` and describes
*Stylonectria*.

#### The OCR case: the rank marker survives, the genus name does not

`taxon_9499dcb0` (added 2026-08-24) is the hardest gating case and the
one that should shape the implementation. Its nomenclature is
*Peniophorella* P. Karst.; its description is of **Dentipellis** Donk —
unrelated genera in different families — and the description is a
three-way pile-up:

| chars | content |
|---|---|
| 0–1180 | **herbarium specimen commentary about a different fungus** — a curator's note on a contaminated packet of *Hyphoderma pubera*, describing the contaminant's spores |
| ~1182–1560 | the **Latin diagnosis, head-clipped** — opens mid-adjective-list at `indetcrminatum, raritcr ciTusoreflexum…`, its subject gone |
| ~1560–end | the **English description, complete** — Fruit-body → Spores, `On rotten wood.`, then the type species and examples |

**The evidence is buried under OCR damage.** The genus reads
`Denttpellis`, the type species reads `H)'drrum fra~ile`, the Latin
reads `Sporac globosac… parictibus lcvilms, amyloidcis`. A D10 built on
exact genus matching, or on gnfinder alone, misses it —
and worse, `§6:authored_binomial` **does** fire here, but on the clean
`Hyphoderma pubera (Fr.) Wallr.` sitting in the *alien* block. Right
flag, wrong reason.

**So key the detector on the rank marker, not the name.** A
`TYPE SPECIES` / `Typus:` declaration inside a description is a
**genus-rank marker**, and that phrase survives OCR far better than the
name it introduces: `TYPE SJ'ECLES` and `Typus:` are both still
recognisable here while `Dentipellis` is not. A Type-species
declaration in the description, plus a nomenclature naming some *other*
genus, is the detectable shape — and it needs no name resolution at
all.

That generalises to **D14** (rank mismatch): a Type-species declaration
means the description is of a genus, whatever rank the nomenclature
claims. The two detectors want the same cheap primitive.

**`merge_metric` reads 0** on this treatment despite three unrelated
blocks, because they are not *repetitive*. A reminder that the metric
measures repetition, not heterogeneity — see D15, which fires on
repeated morph terms for the same reason.

**It is also #404 population.** Persoonia volume 2, no title, no DOI,
19 treatments from the one ingest document. The fixture entry will
dangle after re-ingest, so re-capture this case from a source that is
not scheduled for replacement.

**It has a positive case.**  `taxon_6c2dcb7c`
(`§6-compilatory-double-description`) is the first fixture
entry where this comparison actually fires: gnfinder finds
only *Exophiala* in the description and only
*Cladophialophora* in the nomenclature.  The three earlier
wrong-attribution cases — taxon_4b89d160, taxon_5581a442,
taxon_60758ef3 — all name no resolvable binomial in the
description, so D10 is silent on them.  Its yield is
therefore narrower than the §2/§6 problem it addresses.

**The signal already exists, unused.**  `§6:authored_binomial`
fires on that treatment precisely because the description
contains an authored binomial — but it is read as a *merge*
signal.  Compare the genus of the binomials found in the
description against the genus in the nomenclature field, and
a disagreement is a mis-attribution.  gnfinder already
returns the parsed name, so the genus is in hand.

**Must stay silent on** the poster children, and in
particular on `taxon_09b97d5f` and `taxon_4a5306ac`, whose
*diagnoses* are full of comparative binomials from other
genera — the comparison must read the **description** only,
and must tolerate a description that legitimately names
congeners.

**Blocked on §15 for part of the corpus.**  Name detection
is unavailable on the 2 629 plazi-only treatments where the
element-join artifact makes binomials unparseable, so this
detector is blind exactly there.  Sequence after D6.

**Depends on**: nothing new — gnfinder is already wired in.

#### D9.1 — `§10:diag_tail_clip` is missing, and it is a one-liner

Found 2026-08-24 on `taxon_a8e45990`, which fires **zero flags** while
its `diagnosis` ends mid-sentence at `…compared to`.

The asymmetry is visible in `triage_signals.build_signals`:

```python
'tail_clipped': tail_clipped(desc),
# Reuse the same head-clip predicate on the diagnosis field.
'diag_starts_mid_sentence': desc_starts_mid_sentence(diag),
```

The **head**-clip predicate was deliberately extended to the diagnosis —
that is what `§10:diag_head_clip` is — but the **tail**-clip predicate
was not. Nothing about `tail_clipped` is description-specific;
confirmed by calling it directly on this treatment's diagnosis, where it
returns `True`.

So one line and a flag name closes it, symmetric with the existing pair.
Worth doing before D9/D11, both of which are larger.

**Why it matters more than a missing flag usually would.**
`taxon_a8e45990` is otherwise an excellent treatment — real
nomenclature, six correctly-separated fields, zero OCR damage on every
mode, `merge_metric` 4. It reads as clean to every current signal, so
the truncation would pass into the golden set unremarked. **False
negatives on otherwise-clean treatments are the expensive kind**: a
flagged mess gets reviewed anyway, whereas this one looks finished.

### D16 — Morphology section routed to the wrong field (§12)

**Catches**: a `notes` or `diagnosis` span whose paragraph number falls
*inside* the description's paragraph range **and** whose text opens with
a morphological section heading. That combination means a chunk of
anatomy was routed out of the description it belongs to.

Found 2026-08-24 on `taxon_c421e8b6` (*Lyomyces albofarinaceus*), whose
description holds Basidiomata → Hyphal system → Basidiospores at
paragraphs 49, 51, 55 while paragraph **53** — `Hymenium. Cystidia of
two types… Basidia clavate…` — sits in `notes`. For a corticioid
fungus cystidia and basidia are diagnostic, so the description is
missing core morphology.

**It is systematic within a paper.** All **five** treatments from that
MycoKeys article show the identical shape — description at *n*, *n*+2,
*n*+6 and notes at *n*+4, always opening `Hymenium.` So it is a
repeatable classifier error keyed on one section heading, not a one-off.

#### The measurement path, including two failures

Worth recording in full, because the obvious versions do not work:

| rule | hits | share of eligible | verdict |
|---|---:|---:|---|
| any field interleaved in the description range | 13 362 | **55.4 %** | useless — interleaving is *normal* |
| restricted to `notes`/`diagnosis` | 9 735 | 40.4 % | still the norm |
| …opening with a morphology **word** | 204 | 0.8 % | noisy — mid-sentence fragments like `spores of C. ulkhagarhiensis are…` |
| …opening with a morphology **heading** (`Term` + `.`/`:`) | **29** | **0.12 %** | clean — every hit genuine |

Eligible = the 24 111 treatments with ≥ 2 description spans.

**The heading form is what does the work.** Requiring the term to be
capitalised and terminated by `.` or `:` separates a section heading
from prose that merely begins with an anatomical noun, and takes the
rate from 0.8 % to 0.12 % while every surviving hit is real. Nine of
the first ten are `Hymenium.` in `notes`.

**Do not implement the interleaving test on its own.** At 55 % it would
flag the majority of multi-span treatments, and `materials_examined`
interleaved 23 232 times is simply how papers are written — specimen
data between description paragraphs is normal.

**Gating fixtures**: must fire on `taxon_c421e8b6` and its four
siblings. Must stay silent on all 19 poster children, and in particular
on `taxon_b970d2c2` (`genus-description-no-measurements`), whose short
description is complete.

**Depends on**: nothing — paragraph numbers and a heading regex. The
morphology term list can start from the `feature_label` vocabulary
already in `features_candidate`.

### D17 — Duplicated description block (§6/#405)

**Catches**: a treatment whose `description` contains the same block
twice — the source document holds the article's text more than once,
and the extractor harvested both copies.

Found 2026-08-24 on `taxon_c4aa1185` (*Exophiala clavispora*, Journal
of Fungi 6), where the operator asked whether the two identical
`Description:` sections were the same spans repeated. **They are not**:
paragraphs 637 and 651, chars 91 707–92 541 and 93 601–94 425 —
distinct spans about 1 900 characters apart.

**They differ only in character encoding**, at 94.9 % similarity:

| block 0 | block 1 |
|---|---|
| `μ` U+03BC greek small mu | `µ` U+00B5 micro sign |
| `°C` U+00B0 degree sign | `◦C` U+25E6 white bullet |
| `⎯x` U+23AF | `x` |

Not one word or measurement differs. **The source document contains the
treatment twice, in two different renderings** — and the whole treatment
is doubled, not just the description: `materials_examined_spans` come in
two mirrored pairs (633, 639) and (647, 653) around each copy.

**Measured across the corpus.** Of the 22 955 treatments with ≥ 2
substantial description blocks, **381 (1.66 %)** contain a near-duplicate
pair above 0.90 similarity. **22** of those differ *only* by character
encoding after folding the mu/degree variants — this case's exact
signature.

So the detector is worth building at two sensitivities: exact-or-near
duplication catches 381, and the encoding-fold catches the 22 where the
duplication is provably a rendering artefact rather than an author
repeating themselves.

#### It is a publisher production error, not an extraction artefact

Traced 2026-08-24 to the source PDF. Ingest doc
**`731b55a2-3645-591c-9f7f-1f4ee9923291`** in `skol_dev`,
doi `10.3390/jof6040187` (MDPI). Both copies sit on **PDF page 28** —
the page-28 marker is at char 90 444 and page 29 at 94 979, with the
copies at 91 707 and 93 601.

`pdftotext -layout` on that page shows the figure caption **drawn
twice, overlapping**:

```
 Figure
Figure  20.20.Exophiala  clavispora (CGMCC
               Exophiala clavispora  (CGMCC3.17517).   (A,B)
                                               3.17517).     Forward
                                                          (A,B)      and and
                                                                Forward   reverse  of colony
```

`3.17517` occurs **4 times** on the page and `clavispora` **7 times**.
And the two overlapping caption copies end `10 μm` and `10 µm` — **the
same encoding split as the two description blocks**, so it is the same
pair of text layers throughout.

So the PDF carries **duplicated text objects**: the same content drawn
twice in two font/encoding runs at nearly the same position. On screen
the page looks normal — two identical texts overlaid are visually
indistinguishable from one — but every text extractor reads both.

**Our extractor behaves better than `pdftotext`, and that is the
problem.** `pdftotext` interleaves the layers into obvious garbage
(`20.20.Exophiala clavispora (CGMCC (CGMCC3.17517)`), whereas
`article.txt` separates them into two clean, readable blocks. Good for
legibility, bad for detection: the output looks like a legitimate
repeated section rather than an artefact, which is precisely why this
needed a detector rather than being caught by eye.

**That raises D17's value above what the corpus rate suggests.** At
1.66 % it looks marginal, but it is catching **publisher production
errors** — defects that no dedup keyed on DOIs, documents or ingest
records can see, because there is only one article, one document and
one DOI. The duplication is *inside the page*.

**This is #405 one level down.** Trello #405 concerns the *same article
ingested as two documents* — 36.7 % of ingest docs share a DOI, usually
a `crossref` PDF beside a `pmc` JATS copy. This is the *same article
appearing twice inside one document*, which no DOI-keyed dedup can
see. Both need fixing; only one is currently ticketed.

**Note this revisits a signal already rejected once.** The µ-encoding
profile was tested on 2026-08-23 as a *merge-seam* detector and failed —
8.9 % of descriptions mix the two encodings with no association to
merges, and segregation was mildly *anti*-correlated. It works here
because the question is different: not "did the encoding change at a
boundary" but "are two near-identical blocks distinguished only by
encoding". Same observable, different inference.

**Gating fixtures**: must fire on `taxon_c4aa1185`. Must stay silent on
all 19 poster children — in particular on
`agaric-latin-english-pair`, where a Latin diagnosis and its English
translation legitimately restate the same content and would score high
on a naive similarity test.

**Depends on**: nothing — block splitting and a similarity ratio.

### D18 — Nomenclature absorbed into a Description block (§6/§12)

**Catches**: a `Description` block whose *first sentence* is a
nomenclatural citation. The heading was not lost — it was labelled as
**content**, so no treatment boundary was created and everything after
it merged into the previous taxon.

**This is the mirror of every other D12 case.** Those lose a heading to
a *non-content* label — `Table`, `Misc-exposition`, `Figure-caption`.
Here the classifier is arguably right that the text is content; the
failure is that **the grouper's boundary signal is the `Nomenclature`
label**, so a heading absorbed into `Description` creates no split.

Found on `taxon_d9ffc366` (Hesler & Smith, *North American Species of
Crepidotus*), which merges **four** complete agaric descriptions. The
paper numbers its taxa `Ex 1`…`Ex 4`, and the extractor captured only
the first:

| taxon | citation | outcome |
|---|---|---|
| Ex 1 *Pyrrhoglossum hepatizon* | char 366 087, `Nomenclature` | ✅ boundary made |
| Ex 2 *Naucoria tiliophila* | char 367 869, **first sentence of a `Description` block** | ❌ merged |
| Ex 3 *Melanotus eccentricus* | mid-text | ❌ merged |
| Ex 4 | no heading found at all | ❌ merged |

Four full cycles of the agaric template — Pileus → Lamellae → Stipe →
Spores → Basidia → cystidia → tramas → Pileipellis → Clamps — with
**four different spore measurements**: 4.6–5.5, 5.5–7, 5.3–6 and
8–10 µm. The operator read it as two descriptions; it is four.

#### Measured

Over 250 sampled documents holding 2 218 `Description` blocks:

| | count | rate |
|---|---:|---:|
| blocks opening with a citation shape | **5** | **0.23 %** |
| documents with at least one | 4 | 1.6 % |

**Precision on the sample was 100 %** — every hit is a genuine heading
run together with the description that follows:

* `Gastroboleus dinoffii Nouhra & Castellano sp. nov. Basidiomata usque ad 6 × 4.5…`
* `Coriolopsis brunneoleuca (Berk.) Ryvarden, Norw. Jl Bot. 19: 230 (1972). Basidio…`
* `Claudopus vinaceocontusus Baroni, sp. nov. (Figs. 4-6 and 14) Pileus sordidus…`

A low rate with a high cost: each instance is a **missed treatment
boundary**, so it does not lose a sentence, it merges two taxa.

**Why it is easier than the general heading problem.** The citation sits
at **offset 0 of the block**, not somewhere in running text — so the
match is anchored and needs no scan. That anchoring is what buys the
precision; the same regex applied anywhere in a block would drown in
comparative citations (see D13).

**Gating fixtures**: must fire on `taxon_d9ffc366`. Must stay silent on
the poster children, and in particular on `taxon_01a01c54`
(`§11-gen-nov-plus-type-species`) and `taxon_b970d2c2`
(`genus-description-no-measurements`), where a genus and its species
legitimately share a document — the check is *within* a Description
block, not across blocks, so both should pass.

**Depends on**: nothing — the `.ann` labels and an anchored regex.

### D11 — Mid-description truncation (§10)

**Catches**: a field that is cut off *inside* the
description rather than at its end.  `§10:tail_clip`
inspects only the final characters of the field, so a
description that ends on a clean sentence looks healthy
however mangled its middle is.

**Gating fixtures**: must fire on `taxon_5581a442`
(`§6-genus-description-merged-heading-as-Table`), whose
Culture-characteristics block stops at `…margin entire, `
mid-way through, and on `taxon_4b89d160`
(`§2-wrong-genus-nomenclature`), whose Culture paragraph
lacks its closing period.  Both were spotted by the operator
by eye and neither fires anything.

**Must stay silent on** the 15 poster children, several of
which contain legitimate mid-field commas and semicolons at
paragraph joins.

**The signal is not "no terminal period".**  Descriptions
are assembled from several source paragraphs, so an interior
join legitimately ends mid-clause.  Two better forms, one
structural and one textual:

*Span boundary.*  The interior text ends exactly where a
span boundary falls, and the missing clause sits in the
following paragraph under a different label —
`taxon_5581a442`'s `reverse concolourous.`,
`taxon_6f788487`'s `reverse light-brown.`

*Parallel structure* — contributed by the operator on
`taxon_6f788487`, and the sharper of the two.  That
treatment has three sibling `Culture_on_*` clauses:

```
Culture_on_OA  … flat, slimy growth; reverse olive brown.
Culture_on_PDA … flat, slimy growth; reverse olive brown.
Culture_on_CMA … poor sporulation, flat;
```

The operator identified the truncation from the semicolon
*because its siblings end differently*, and the dropped run
turned out to be exactly `reverse light-brown.`  **A clause
that breaks the template its siblings share is truncated; a
clause that merely lacks a period may not be.**  Sibling
clauses are cheap to find — same label family, same
treatment — and the comparison needs no source lookup.

**Depends on**: nothing, but it needs the span offsets,
which makes it a natural companion to `bin/verify_spans`
rather than a pure text heuristic.

### D12 — Content swallowed by non-content layout labels (§2/§6/§12)

**Catches**: body text that the layout pass assigned to a
"not body text" label — `Table`, `ToC-entry`,
`Bibliography`, `Misc-exposition` — and that the extractor
therefore dropped.

**Two shapes, one root cause.**  Which label eats the text
decides which symptom you see:

*Swallowed **nomenclature headings** → boundary loss → merges:*

| taxon | heading became | result |
|---|---|---|
| `taxon_4b89d160` | missed outright | *Pseudonectria* nomenclature on a *Stylonectria* description |
| `taxon_5581a442` | **`Table`** (synonyms `Bibliography`) | *Acremonium* genus description merged into a *Proliferophialis* species |
| `taxon_60758ef3` | **`ToC-entry`** | Murrill's species 73 and 74 merged |
| `taxon_8d815304` | **`Misc-exposition`**, `Diagnosis`, `Notes` — 21 of 27 headings in the document | **nine genera in one treatment** |
| `taxon_8ebf437c` | **`Table`** (holotype line also `Table`) | *V. dactylidis* description appended to *V. chlamydospora* |
| `taxon_ecb0124d` | **`Figure-caption`** | `Clade A Phialophora verrucosa Medlar, Mycologia 7: 203. 1915 — MycoBank…` → treatment is `Nomen ignotum` |
| `taxon_fdbd1b53` | **`Table`** | `33. cocculi Stigmina Crous & U. Braun sp. nov.` → treatment inherits species **#32**'s nomenclature *and* its holotype |

*Swallowed **description continuations** → content loss → truncations:*

| taxon | what was dropped | symptom |
|---|---|---|
| `taxon_5581a442` | `reverse concolourous.` as **`Misc-exposition`** | Culture-characteristics block ends `…margin entire, ` |
| `taxon_66c1e6e3` | four separate runs as **`Misc-exposition`** | `crumpled, firmly` → `the base whitish`; `frondose spe-` → `erumpent` |
| `taxon_6f788487` | `reverse light-brown.` **and** the whole Chemistry lead-in, both **`Misc-exposition`** | `poor sporulation, flat;` breaks its siblings' template; `Di-n-octyl` → `phthalate` |
| `taxon_8ebf437c` | the description **head** as **`Misc-exposition`**, and a Notes continuation as **`Figure-caption`** | block opens mid-measurement at `diam.`; `and broad cellular` → `pseudoparaphyses` |
| `taxon_a3308621` | **two consecutive** runs as **`Misc-exposition`** | `…under near-UV at ` → `24oC did not yield any ascomat.a.`; then a second break mid-word at `rotia:` |
| `taxon_ecb0124d` | `Cardinal temperatures: minimum below 21 °C, optimum 30 °C, maximum 37 °C.` as **`Misc-exposition`** | culture data dropped from a culture-only description |
| `taxon_b0d687da` | the **nomenclature** as `Misc-exposition`; a `Note:` section split across a page break into `Diagnosis` + `Notes` | `Nomen ignotum` despite `Helicodochium amazonicum J.S. Monteiro…` sitting at 3 941; `…lacking` → `pseudoparenchymatous stromata` |
| `taxon_fdbd1b53` | the **head of the Latin diagnosis** as **`Table`** — `Maculae amphigenae… Mycelium immersum… Stromata… Conidiomata (= sporodochia) hypophylla…` | the Latin block opens at `Conidiophora numerosa`, three features short of its English counterpart |

**`Misc-exposition` is the repeat offender** — three of the
five cases, and the only label to swallow content in more
than one treatment.  It reads as the layout pass's
catch-all, which makes it the first place to look.

#### One block, both symptoms — the case that makes the rule writable

`taxon_fdbd1b53` (*Mycotaxon* 57, Crous & U. Braun) appears in **both
tables above, for the same `Table` block**, and that is what makes it
the gating fixture rather than a seventh anecdote.

The operator's reading was that *"the Latin description lacks the
leading Leaf_spots, Mycelium and Conidiomata, so the Latin is truncated
above."*  Correct, and **the missing text is recoverable** — it is not
an OCR loss.  The block immediately preceding the description holds:

> `[@33. cocculi Stigmina Crous & U. Braun sp. nov.  Fig. 1.  Maculae
> amphigenae, atro-brunneae, angulares, per venas limitatae, 1-4 mm
> latae.  Mycelium immersum: hyphae laeviae, brunneae.  Stromata
> 20-180 × 10-40 µm.  Conidiomata (= sporodochia) hypophylla, densa,
> effusa, brunnea, 50-200 × 40-80 µm.` **`#Table*]`**

One mislabelled block, three losses:

1. **The species heading is gone**, so the treatment picked up the
   *previous* species' nomenclature — `32. clutiicola Pseudocercospora
   Crous & U. Braun, Sydowia 46:` — together with the tail of the
   preceding Notes. **The treatment is filed under the wrong name**:
   it is *Stigmina cocculi*, described under *P. clutiicola*.
2. **Species #32's holotype came with it** (`Clutia cf. affinis…
   PREM 32896`), so `materials_examined` now holds **two holotypes** —
   #32's and, correctly, #33's `Cocculus hirsutus… PREM 42682`.
3. **The Latin diagnosis is beheaded** by exactly the three features
   the operator named, plus `Stromata`.

**What proves Latin and English are one taxon** rather than two merged
species is the measurements: `25-80 × 4-8`, `10-30 × 5-6`,
`17-70 × 5-7` µm appear identically in both. A Latin diagnosis
followed by an English description is the pre-2012 protologue
convention, already in the fixture as `agaric-latin-english-pair`, and
is not itself a defect.

**Note what did *not* fire.** `merge_metric` reads 1 and no triage
flag is raised. D10 (genus mismatch) cannot help either: it compares
nomenclature against the description, and this description — pure
morphology in two languages — never names its genus. The mismatch is
visible only between the **nomenclature** (`Pseudocercospora`) and the
**notes** (`it is a species of Stigmina`, `P. cocculi`), which nothing
currently compares. Cheap addition to D10.

#### A detection rule for D12, with the two refinements that failed

D12 has been a catalogue of cases with no rule. This one is writable,
and the measurement below is over a **200-document random sample** of
the 20 928 in `ann_combined`.

| rule | blocks | extrapolated corpus-wide | verdict |
|---|---:|---:|---|
| nomenclatural act anywhere in a non-content block | 458 | ~48 000 | useless |
| …restricted to the block's **first line** | 119 | ~12 500 | still useless |
| …**and** ≥ 3 lines, ≥ 200 chars of following prose, no trailing page number | **13** | **~1 360** | usable |

`Bibliography` is excluded throughout — it carries an act in **9.8 %**
of blocks by design, since bibliographies cite protologues. The
remaining non-content labels sit at 1–2.6 %, against **20.4 %** for a
correctly-labelled `Nomenclature` block.

**The first-line refinement fails for a nameable reason**, worth
recording because it is the obvious move: the survivors are dominated
by **table-of-contents entries and running heads**, which are
*correctly* labelled non-content —
`Pseudocercospora styracigena sp. nov. (China) ... 231`. Requiring
continuing prose and rejecting a trailing page number removes them.

Spot-reading the 13 survivors, roughly three-quarters are genuine
mislabelled headings, and one shape recurs:

> `[Figure-caption] Saccardoella psidiicola W.Y. Zhuang, W.Y. Li &
> K.D. Hyde, sp. nov.` → `FIGS 4-5  MycoBank MB 512350  Pseudothecia
> subglobosa vel piriformia, 200-365 µm…`

**A protologue that opens with its figure reference gets read as a
figure caption**, taking the heading and the Latin with it — three of
the eight sampled, and the same label that produced `taxon_ecb0124d`
above. The two false positives were a phylogenetic-group listing in a
real `Table` and a `RESULTS` heading in a `Key`.

**Depends on**: reading `ann_combined` attachments, which
`span_resolver` already does.

**Gating fixtures**: must fire on `taxon_fdbd1b53`
(`§12-Table-swallows-heading-and-description-head`), `taxon_5581a442`,
`taxon_60758ef3` and `taxon_ecb0124d`.

#### D12 has a mirror image, and nothing looks for it

Everything above is content **swallowed by** a non-content label.
`taxon_9446b102` (added 2026-08-24) runs the other way: **non-content
promoted *into* `Description`.**

Its second description span, at char 13 452, is a block of **table
footnotes and a table caption** —

> `1 Ex-type cultures are in bold. 2 At 14 d after inoculation where
> 0 = no discolouration… Table 1  Diaporthe cultures isolated from
> sunflower investigated in this study.`

— labelled `Description` by the layout pass, sitting between a `Table`
block at 13 363 and a `Page-header` at 14 153.

**This direction is arguably the worse of the two.** Swallowed content
goes *missing*, which eventually shows up as a truncation or a gap.
Promoted non-content *arrives looking like data*: it reaches the
annotator, gets feature labels attached, and enters the training set as
though it were morphology. A detector written only in D12's direction —
"a `Table`/`ToC-entry`/`Bibliography` span whose text matches a
nomenclature shape" — cannot see it at all.

**Something looks for it now, and it is measurable.** A span covering
more than one layout block has, by construction, absorbed material the
layout pass had separated. Measured 2026-08-25: **0.7 % of treatments**
contain such a span — rare, but every one is this defect.
`bin/treatment_dossier` labels each covered block on hover, which
surfaced two cases within seconds of the feature landing:

* **`taxon_a2e93e8d`** — a **`description`** span covering **14
  blocks**, alternating `Table` / `Misc-exposition`, holding
  `Kotiranta 25567 / KP994354 / KP994387 / Russia / Volobuev et al.
  (2015)`. A **phylogenetic accession table** read as a description.
* **`taxon_bc384990`** — a **`materials_examined`** span covering
  `Notes` + `Page-header` + `Misc-exposition`, opening `† ATCC:
  American Type Culture Collection, Manassas, USA; BCC: BIOTEC Culture
  Collection…`. A **table-footnote legend** read as specimen data.

Both are `taxon_9446b102`'s shape, and the multi-block-span test is a
cheap detector for the whole class: it needs only the `.ann` and the
stored spans, both of which are already read.

The cheap tell here is the inverse of D12's: not a nomenclature shape
inside a non-content label, but **a `Description` block that looks like
apparatus** — footnote markers (`1 `, `2 ` at line starts), a `Table N`
/ `Fig. N` caption opener, or a run of accession numbers. Worth pairing
with D12 rather than building separately, since both are queries over
the same `.ann` labels.

**Two severances in a row** (added 2026-08-24, `taxon_a3308621`,
operator: *"the mating studies are truncated"*).  The description ends
`Mating in all possible combinations and incubation under near-UV at `
at char 891 956; the block at **891 958 — two characters later** —
carries `Misc-exposition` and reads `24oC did not yield any
ascomat.a. CBS 604.75 agrees in every respect…`.  One sentence, cut at
the boundary.

A second `Misc-exposition` block 380 characters further on opens
**mid-word**: `rotia: it has almost completely lost this capacity…` —
the tail of *microsclerotia*, matching this treatment's own
`Microsclerotia` content.  A mid-word opening is the strongest form of
the severed-term tell, and it is worth noting the two breaks are
independent: fixing one would still leave the other.

The content is squarely descriptive — mating trials under near-UV at
24 °C and whether ascomata formed is culture work, and this treatment
is *entirely* cultural characteristics.  So this is not a boundary
judgement call; it is a straightforward loss.

**`Figure-caption` swallows content too** (added 2026-08-23,
`taxon_8ebf437c`).  It was not previously on the list.  The
swallowed run is the second half of a comparative *Notes*
paragraph, and it ends `…chlamydospore-like asexual morph in
culture (Fig. 3).` — a trailing figure reference inside
otherwise ordinary prose, which is the plausible cause of
the misclassification.  So the rule to be careful with is
"contains a `(Fig. n)` reference", which is true of a great
deal of running descriptive text.

#### Label instability across a page boundary

`taxon_b0d687da` (added 2026-08-24) shows a variant where **nothing is
mislabelled** and content is lost anyway.

Its single `Note:` section is split across a page break, and the two
halves receive **different labels**:

| offset | label | text |
|---|---|---|
| 6 193 | `Diagnosis` | `Note: Seifert et al. (2011) described seven helicosporous genera… conidiophores, lacking ` |
| 6 663 | `Page-header` | `--- PDF Page 4 Label 4 ---` |
| 6 707 | `Misc-exposition` | `8 … Monteiro, Gusmão, & Castañeda-Ruiz` (running head) |
| 6 769 | `Notes` | `pseudoparenchymatous stromata and conidia composed of…` |

One sentence — `…lacking` / `pseudoparenchymatous stromata…` — landing
in two *different fields*, `diagnosis` and `notes`.

**Every block here is labelled correctly.** The page marker really is a
page marker; the running head really is furniture. What failed is that
the classifier's label *changed across the page boundary*, and the
assembler then routed the two halves to different fields without
noticing the severed sentence.

**The same document shows the benign case**, which is what makes the
diagnosis clean: description blocks at 4 630 and 5 516 are separated by
an identical `Page-header` + running-head pair, and both carry
`Description`, so both reach the `description` field and the text
survives. The page break is not the problem; **the label change across
it is.**

That points at a different fix from the rest of D12. The other cases
want a better classifier. This one wants a **rejoin step in assembly**:
where consecutive prose blocks are separated only by page furniture and
the earlier block does not end a sentence, they belong together — and
the label of the *later* block should not be trusted over the
continuity of the text.

Cheap to detect, since both conditions are already in hand: the
intervening blocks are `Page-header`/running-head, and the first block
fails `tail_clipped`.

#### One document, three swallowing labels, one false seam

`taxon_8ebf437c` is worth reading whole, because it shows
what the symptom looks like *after* the drops compose.  In
source order:

| char | label | content |
|---|---|---|
| 21512 | `Description` | Culture characters, then `Notes – V. chlamydospora resembles…and broad cellular` |
| 22029 | **`Figure-caption`** | `pseudoparaphyses but differs in…(Fig. 3).` |
| 22354 | **`Table`** | ***Vagicola dactylidis*** … **sp. nov.** — IF551684 |
| 22513 | **`Table`** | `Holotype – MFLU 15-2720` |
| 22549 | `Etymology` | `With reference to the host occurrence` |
| 22615 | **`Misc-exposition`** | `Saprobic on dead stem of Dactylis sp. Sexual morph: Ascomata 120–180 m high, 110–160 m` |
| 22726 | `Description` | `diam. (x = 153.9 × 141.3 m, …` |

Only the two `Description` blocks reach the description
field.  Everything between them is dropped or routed
elsewhere — including a **`sp. nov.` nomenclature heading**
and the **head of the very description that follows**.

The result is a **false seam that reads as continuous
prose**.  Assembled, the field runs `…and broad cellular` →
`diam. (x = 153.9 × 141.3 m, n = 10 , solitary, scattered,
superficial, globose to subglobose, dark brown to black,
coriaceous, ostiolate. Ostiole 50–60 m high…` — ascomata
described as ostiolate, then the ostiole described.  It
scans.  The operator read it as a natural transition, and it
is not one: it is a species boundary with four blocks
excised.  The genuine continuation of `and broad cellular` is
`pseudoparaphyses`, two blocks away.

**This is the case against trusting readability as
evidence of integrity.**  A merge is easiest to spot when
the seam is ugly; this one is invisible precisely because
the dropped material included the heading that would have
announced the boundary.

**And it is why the `nomenclature` field cannot be trusted
to name what the description describes.**  This treatment's
nomenclature is *Vagicola vagans* — harvested from
paragraph 81, about 6 000 characters upstream, where it is
the genus's type species named in passing.  The species
actually described in the appended block is *V.
dactylidis*, whose own heading was the `Table` at 22354.
Three checks agree on *dactylidis*: the holotype
`MFLU 15-2720` matches the appended block's Material
examined (`IT 799 (MFLU 15-2720, holotype)`), the etymology
`with reference to the host occurrence` fits *dactylidis*
from the host *Dactylis*, and the closing Notes discuss *V.
dactylidis*.  D10 compares genus between nomenclature and
description and would pass this — all three are *Vagicola*.
**The mismatch is at species rank, and nothing checks it.**

#### Why the headings are unrecognisable: they are fused, not merely mislabelled

`taxon_8d815304` (added 2026-08-23) is the extreme case and
the one that shows the mechanism.  Its nomenclature is
*Cephalotheca* Fuckel 1871; its prose fields carry **seven
further genus headings**, each with author, journal, page and
year and most with a `Type species` line — *Hamigera*
(`notes`), *Talaromyces* (`diagnosis[0]`),
*Pseudogymnoascus* (`description`), *Monascus*,
*Monascella*, *Eremascus*, *Renispora* (all
`diagnosis[1]`) — plus a species list belonging to an eighth
genus, *Byssochlamys*, whose own heading fell just outside
the slice.  **Nine genera in one treatment.**  The operator
read it as "at least two partial descriptions … conflated"
from the doubled `Anamorph` label; the label count
understates it by a factor of four.

In that one document **21 of 27 genus headings sit inside a
prose label and only 3 in `Nomenclature`.**  The reason is
**positional**: each buried heading is fused to the tail of
the paragraph before it, so it never starts a block and the
layout pass never sees a heading at all.

| preceding tail | fused heading |
|---|---|
| `…composed of filaments.` | *Chaetosartorya* |
| `…name Myxotrichum uncinatum (Eidam) J. Schröt.` | *Ctenomyces* |
| `…synonymized Nannizzia with Arthroderma.` | *Narasimhella* |
| `…arthroconidia with disjunctors in culture.` | *Ascocalvatia* |
| `…(see also Chesters, 1934, Booth, 1961).` | *Hamigera* |
| `…(Talaromyces striatus. Penicillium striatum).` | *Talaromyces* |

Every one of those tails ends in a full stop, so the
boundary was there in the source and was lost in extraction.
This is the **same block-separator loss family as §15** —
different producer, same failure — which matters for
sequencing: fixing separator preservation upstream may
retire a large part of D12 rather than needing a detector at
all.

**Running headers are not eliminated, and fusing one to a
heading is one way a heading gets lost.**  This answers the
operator's 2026-08-22 question directly.  **Second instance,
2026-08-25**: `taxon_ecb0124d`'s block at 44 733 reads
`11  Y. Li et al.: Phialophora verrucosa and relatives in
Chaetothyriales  De` — the running head fused to the opening
`De…` of the description's own section header, the whole thing
labelled `Misc-exposition`.  The result is a culture description
whose **first medium is never named**: it opens `Colonies
growing slowly, olivaceous brown…` and only names a medium at
the *second* one, `On MEA, 30 °C:`.  Grammatically the opening
is a clean sentence, so no head-clip signal fires — this is a
**semantic** head-clip, invisible to `desc_starts_mid_sentence`.  At offset 119 686
the block reads:

> `VO ARX : Re-evoluorfon of Eurorfoles 283 MyxotriciJUm Kunze in Myk…`

— the journal's own running head welded to the front of the
*Myxotrichum* heading, the whole thing inside
`Misc-exposition`.  The **injected** marker
`--- PDF Page 54 Label 54 ---` 46 characters earlier is
handled correctly, with its own `Page-header` label.  So the
distinction to hold onto is: *our* page markers are stripped;
the *publisher's* running heads are not, and they land in
running text where they corrupt the next heading.

**The source is an entire bound volume.**  The ingest
document has `journal: 'Persoonia'`, `volume: '13'`, **no
title and no DOI**, and is carved into 18 treatments
spanning `pdf_label` 7 → 153 and 1 549 paragraphs — Coprinus,
Sordaria, the Eurotiales, Psathyrella, Rhodocybe,
Camarophyllopsis, Hygrocybe.  `taxon_8d815304` is the
pdf-page-53 slice of von Arx's Eurotiales re-evaluation.
Whole-volume ingests give a heading detector far more
chances to fail in one document, and a missing
title/DOI pair is a cheap way to spot them up front.

**Scope of the whole-volume problem, measured 2026-08-23.**
The missing-title-and-DOI signature isolates it exactly:

| | count | share of corpus |
|---|---:|---:|
| treatments, all sources | 81 527 | — |
| from *Persoonia* | 8 228 | 10.1 % |
| from **whole-volume** *Persoonia* ingests | **771** | **0.9 %** |

Those 771 come from **80 ingest documents covering
volumes 1–19 only**, four per volume — the scanned back-run.
Volume 20 onward is already per-article (8–34 documents per
volume) and carries titles and DOIs.  The cut is that clean.

The under-segmentation is the thing to notice: a whole-volume
document yields a median of **9** treatments where the
per-article volumes yield dozens.  Nine genera landing in
one treatment is what that ratio looks like from the inside.

*(Minor data note: `volume` is a string, and volume 19 is
stored both as `'19'` and zero-padded as `'019'`, which sort
and group as different volumes.)*

**Medium-term remediation: replace the source, do not detect
around it.**  Operator, 2026-08-23 — Naturalis publishes
*Persoonia* as separate per-article files with much better
OCR, at <https://repository.naturalis.nl/col/1>.  An
ingestor for it is **Trello #404**.  That retires this
document class at the root: per-article files restore the
title/DOI pair, remove the 1 549-paragraph haystack, and cut
the character-substitution OCR that D8 cannot see anyway.

Two consequences worth holding onto:

* **It does not retire D12.**  Fused headings are a
  block-separator problem, not a *Persoonia* problem, and
  the other scanned sources will keep producing them.  #404
  removes 0.9 % of the corpus from the population, not the
  mechanism.
* **It will orphan this fixture entry.**  Treatment ids are
  content hashes, so re-ingesting *Persoonia* from Naturalis
  gives `taxon_8d815304` a different id and the
  `pathologies.json` entry will dangle.  Re-capture the
  fused-heading case from a source that is *not* scheduled
  for replacement before #404 lands, or the evidence for
  this mechanism goes with it.

**Corpus frequency is unmeasured, not zero.**  A sweep of 60
annotated documents with the heading regex used here
returned no matches outside this volume, because the regex
keys on the pre-1990 monograph citation style
(`Genus Author in Journal vol: page. year.`).  Modern papers
punctuate differently.  Treat the 21-of-27 figure as
document-level evidence only until a style-agnostic heading
matcher exists.

#### When the join is *plausible* — the inverted tell

Every severed-term case above is detectable because the join reads as
**wrong**: `rotia:` cannot begin a word, `pseudoparaphyses` cannot begin
a sentence. `taxon_d08bca1f` (Sydowia 10, 1956, Petrak) shows the
dangerous variant — **a join that produces grammatical text.**

Its description ends one block at `…filiformi-bacillaria, sero` and
opens the next at `adnata, globosa vel late ovoidea…`. Concatenated,
`sero adnata` reads as plausible Latin — *"lately adnate"* — and the
operator's instinct was that the annotation had split a phrase.

**It had not. The fragments are 1 865 characters apart**, on opposite
sides of a page break:

| offset | label | text |
|---|---|---|
| 880 808 | `Description` | `…filiformi-bacillaria, sero` |
| 881 110 | `Misc-exposition` | German translation |
| 882 110 | `Bibliography` | more German |
| 882 535 | `Page-header` | `--- PDF Page 335 ---` |
| 882 583 | **`Misc-exposition`** | `…in uno latere plerumque fasciculo sclerenchymatico` |
| 882 673 | `Description` | `adnata, globosa vel late ovoidea…` |

The true reading joins `adnata` to the **dropped** block at 882 583:

> `…subepidermalia, in uno latere plerumque fasciculo sclerenchymatico
> **adnata**, globosa vel late ovoidea…`
> — *subepidermal, on one side usually adnate to a sclerenchymatous
> bundle, globose or broadly ovoid…*

`sero` is separately the truncated tail of the conidiophore clause;
compare the intact one earlier in the same treatment, which ends with
dimensions: `bacillari-conica, simplicia, 5—7 µ longa, ad basin
2.5—3.5 µ crassa.`

**So a severed-term detector keyed on implausibility would pass this.**
The join is syntactically fine; only the *semantics* betray it —
conidiophores are not globose and are not 150–250 µm in diameter. That
requires domain knowledge the detector does not have, and it was the
operator's mycology, not their Latin, that caught it.

**And the risk compounds outside English.** The reviewer's own
assessment was *"my Latin isn't very good"* — a phantom phrase in a
second language is far harder to reject than one in English. Expanding
to non-English and non-Latin material (an acknowledged lower-priority
todo, 2026-08-24) will widen this gap, so the plausible-join case is
worth treating as a *structural* problem — measurable from span
adjacency and dropped blocks — rather than a linguistic one.

The structural signal is available and needs no language at all:
**two `Description` spans separated by ~1 900 characters, a page break,
and three non-content blocks are not adjacent prose**, whatever their
concatenation reads like.

**The tell to search for is a severed term.**
`taxon_66c1e6e3` split the hyphenated word *spe-cies*,
leaving `spe-` abutting `erumpent`; `taxon_6f788487` split
the compound name *Di-n-octyl phthalate*, leaving the
description to open at `phthalate representing 53.98 %`.
Both are terms cut at the boundary of a dropped run, and
both survived into the annotation set.  A hyphen-final
fragment is the easiest instance to grep for, but the
general form is wider: a span opening on a word that cannot
begin a sentence.

**Gating fixtures**: must fire on all five.  Must stay
silent on the poster children, and in particular on
`taxon_01a01c54` (`§11-gen-nov-plus-type-species`), where a
genus and its type species legitimately share one treatment.

**This is a layout-label problem, not a text problem.**  The
signal is available where the detectors do not currently
look: the `article.txt.ann` labels themselves.  A
`Table`/`ToC-entry`/`Bibliography` span whose text matches a
nomenclature shape — `Genus species Author, Journal vol:
page. year.` — is a misclassification.  gnfinder recognises
that shape for *modern* citations, **but not for the older
orthography** — see the note on abbreviated genera and
capitalized epithets below, which is why the match must not
lean on gnfinder alone.  Reading the labels means resolving
the annotated attachment, which `span_resolver` now makes a
one-liner.

**Note it outranks the merge metric on these cases.**
merge_metric reads 7, 1 and 1 on the three merges above —
the lowest in the fixture — because the merged halves are
*similar* prose rather than obviously seamed.
`taxon_66c1e6e3` scores 1 while being the most fragmented
treatment in the collection.  Whatever is done about §6, it
should not be a threshold on that metric.

**Depends on**: nothing new; the labels are already stored.

### D13 — Inline bibliographic citations (§1/§6)

**Catches**: inline literature citations — `Ju and Rogers
(1996)`, `(Kornerup & Wanscher 1978)`, `(de Hoog et al.
2000a, Najafzadeh et al. 2010a, b)`.  **Nothing detects
these today.**  `gn_client.authored_binomial_in_text()`
handles *nomenclatural* citations (`Genus species Author`),
and the layout pass's `Bibliography` label covers full
reference-list entries, but inline `Author (year)` has no
detector at all.

**Measured 2026-08-23** over 15 540 descriptions, with a
regex over two forms — `(Author et al. year…)` and
`Author & Author (year)`:

* present in **13.6 %** of descriptions;
* clean matches: `(Rayner, 1970)`, `Ju and Rogers (1996)`,
  `Quaedvlieg et al. (2013)`, `(Kornerup & Wanscher 1978)`;
* **zero false positives** on ten adversarial anatomical
  probes: `(3–)3.5–5.5(–6) × (2–)2.5–4 µm`,
  `(av. 4.4 × 4.9 µm)`, `(n = 60/2)`, `(Fig. 117)`,
  `(holotype CBS H-8155)`, `(sub-)globose`, `(up to 6)`,
  `(Lat.)`, `(OA)`, `(25 °C)`.

**Presence is not a defect signal — position is.**  A
citation inside a description is frequently legitimate:
`(Kornerup & Wanscher 1978)` is the colour chart, and
`taxon_343eec40`'s `(in collection De Kesel 1979)` is an
odour comparison in a **poster child**.  So this is a
*signal*, not a flag.  It feeds:

* **D7**, where a citation terminating a repeated-label
  group separates a compilatory entry (`taxon_6c2dcb7c`)
  from a genuine merge;
* **§1**, where position within the description
  distinguishes a leak from a legitimate reference.

**Gating fixtures**: must find the citations in
`taxon_6c2dcb7c` (`(Carmichael 1966)`, `(de Hoog et al.
2000a …)`) and in `taxon_343eec40` (`(in collection De
Kesel 1979)`) — and must flag **neither** treatment on
presence alone.

**Not an annotation label.**  Considered for the round-4
label set and declined; the reasoning is recorded in
[`docs/feature_label_non_synonyms.md`](feature_label_non_synonyms.md)
under "What does NOT get a label at all", which is where
someone asking "should I label this?" will look.

**Depends on**: nothing.

### D14 — Rank mismatch between description and nomenclature (§2)

**Catches**: a description written at one taxonomic rank
attached to a nomenclature at another — a family emendation
under a species combination, a genus entry under a species
name.

**Gating fixtures**: must fire on `taxon_7cb84fba`
(`§2-family-description-on-species-nomenclature`), whose
notes declare `Type genus: Diversispora` — family rank —
while the nomenclature is `Albahypha drummondii … comb.
nov.`; and on `taxon_6c2dcb7c`
(`§6-compilatory-double-description`), an *Exophiala* genus
entry under a *Cladophialophora* species name.

#### Sized: 211 genus descriptions with no name attached

Measured 2026-08-25. **1 491** descriptions contain the phrase
`Type species`; **211 of them (14 %)** have a synthetic or absent
nomenclature. Each is a description that *declares itself* to be of a
genus, attached to no name at all — the sharpest form of the rank
mismatch, and detectable without resolving a single taxon name.

`taxon_e4eb2c9f` is the exemplar (Seaver, *North American Cup-fungi
Inoperculates*): its second block ends
`…paraphyses filiform and surmounted with a fusiform conidium-like
body. **Type species, Diplocarpa Curreyana Massee.**` while its
`treatment` field reads `Nomen ignotum`.

**It also confirms the rank-marker approach on clean text.** The
proposal was made from `taxon_9499dcb0`, where the genus name was
OCR-mangled to `Denttpellis` and only `TYPE SPECIES` survived legibly.
This treatment has **no OCR damage on any mode** and shows the same
marker doing the same work — so the signal is not merely an
OCR-robustness trick, it is the primary evidence in clean text too.

**The rank markers are explicit and cheap.**  A treatment
declares its own rank in prose that is already extracted:

| marker in `notes` / description | rank asserted |
|---|---|
| `Type genus:` | family or above |
| `Type species:` | genus |
| `Emended description:` | a revision, so the rank is whatever the marker above says |

and the nomenclature's own shape gives the other half —
`Genus species` is a species, a bare capitalised word is a
genus, an `-aceae` / `-ales` ending is supra-generic.

**Scale, with a caveat.**  Of 30 000 treatments sampled, 115
declare a `Type genus` and **50 of those carry a binomial
nomenclature**.  That 43 % is an **upper bound on the defect
rate**, not the defect rate: a species treatment may
legitimately mention its family's type genus in discussion.
The signal needs the *description's* rank, not merely the
presence of the phrase — which is why `Emended description:`
plus an absence of species-level characters matters in
taxon_7cb84fba.

**Zero poster children mention `Type genus`** in their
descriptions, so the marker is clean against the reference
set.

**Why it matters more than its size suggests**:
taxon_7cb84fba fires **nothing at all** — merge_metric 1,
single clean span, no §15 markers, no repeated labels.  It is
the tidiest wrong treatment in the fixture, and only a rank
comparison would catch it.

**Correctly-attached emendations exist, and show what the
comparison should accept.**  Searching 40 000 treatments for
a supra-generic nomenclature carrying an emendation returns
5, of which two are well formed:

* `taxon_233e247f` — `Delonicicolaceae R.H. Perera et al.,
  emend. Voglmayr & Jaklitsch`, whose description opens
  `Type genus. Delonicicola R.H. Perera et al., …`.  Family
  name, family rank marker: **matched**, and merge_metric 0.
* `taxon_46ff7dde` — `Endogonales Jacz. & P.A.Jacz., emend.
  Tedersoo`, notes `Type family. Endogonaceae Paol.`  Order
  name, order rank marker: **matched**, zero flags.

D14 must accept both.  Note the rank marker climbs with the
rank — `Type species` for a genus, `Type genus` for a
family, `Type family` for an order — so the comparison is a
ladder, not a pair.

**A §6 false positive falls out of this.**
`taxon_233e247f` fires `§6:authored_binomial` **because a
family description legitimately names its type genus with
its author** — `Delonicicola R.H. Perera et al.` is exactly
what a familial emendation is required to state.  Any §6
tightening must exempt an authored name that follows a
`Type genus.` / `Type species.` / `Type family.` marker.

**Depends on**: nothing.

### D15 — Repeated morph term (§6)

**Catches**: a description containing the *same* morph term
twice — two `Asexual morph:` / `Anamorph`, or two
`Sexual morph:` / `Teleomorph`.  A treatment may legitimately
carry **one of each** (that is what a teleomorph/anamorph pair
*is*); what it cannot legitimately carry is two of the same.

**This is the operator's own diagnostic**, from
`taxon_8ebf437c` on 2026-08-23: *"I believe this because we
have a second 'Asexual morph' clause."*  It was not
implemented.  None of the `multi_*` signals fire on that
treatment despite two complete sexual morphs, because they
key on repeated **section headers** and this document
delimits its parts with `Sexual morph:` / `Asexual morph:`
instead.

**Measured 2026-08-23** across descriptions ≥ 200 chars.
8 656 mention a morph term; **1 821 repeat one** — 21.0 % of
those, 4.6 % of the corpus.  Against a random control:

| | repeated morph | control |
|---|---:|---:|
| more than one `description_span` | **92.6 %** | 60.7 % |
| `§6:` flag | **49.6 %** | 21.4 % |
| `§12:desc_span_gap` | **87.0 %** | 43.2 % |
| mean `merge_metric` | **16.62** | 7.12 |

Every indicator roughly doubles, and 92.6 % multi-span is
the strongest single association any proposed signal has
shown.

**Gating fixtures**: **0 of 17 poster children fire** — it is
silent on every clean treatment in the reference set.  It
independently catches four existing pathologies without
having been designed for any of them:

| taxon | class | asex / sex |
|---|---|---|
| `taxon_173204126fbc7e27` | `§6-compact-congenerics` | 1 / 3 |
| `taxon_5581a442fd1a7fc8` | `§6-genus-description-merged-heading-as-Table` | 0 / 2 |
| `taxon_6c2dcb7cce39089d` | `§6-compilatory-double-description` | 2 / 2 |
| `taxon_8d815304f6fb9dd0` | `§12-nomenclature-headings-fused-into-Misc-exposition` | 2 / 0 |

Plus `taxon_8ebf437c` itself
(`§6-morph-pair-plus-appended-congener`, 2 / 2).

**Watch for the legitimate repeat.**  Comparative prose can
say *"…similar to the asexual morph of X"* inside a
single-species treatment.  Count occurrences of the term as a
**block opener** — start of line, or followed by a colon —
rather than anywhere in running text, and re-measure before
setting a threshold.

**Depends on**: nothing.  Pure text, no vocabulary, no
external service — the cheapest item in the backlog.

**Rejected alternative, measured — do not retry.**
`taxon_8ebf437c` switches micrometre encoding exactly at its
merge seam: **12 of 12** U+03BC GREEK SMALL LETTER MU in the
first description block, then **zero** in the second, which
uses U+00B5 MICRO SIGN or drops the glyph entirely (` m`).
It looks like a perfect mechanical boundary marker.  **It does
not generalise.**  Across 39 887 descriptions, 8.9 % mix the
two encodings, and mixing carries essentially no signal —
66.8 % multi-span against 62.2 % for single-encoding
descriptions, `§6` 23.2 % against 22.2 %.  Narrowing the
hypothesis to *segregated* encoding (exactly one switch, as
here) makes it **worse, not better**: 64.6 % multi-span and
`§6` 23.1 %, against 71.0 % and 31.9 % for single-encoding
controls — mildly *anti*-correlated.  The clean boundary in
this one treatment is a coincidence of its source document.

### D19 — Next taxon's header absorbed into the trailing `diagnosis` (§6/§12)

**Catches**: a treatment whose last block is the *opening* of the next
taxon's section — genus header, protologue citation, genus
introduction — routed into `diagnosis` because it is genus-level
descriptive prose and the classifier has no better label for it.

**What makes this different from D4, D12 and D18**, all of which also
concern boundaries: those infer the error from the swallowed text
alone. Here **the recipient is identifiable and can be checked**, which
turns an inference into a verification.

Found 2026-08-25 on `taxon_fd50457a` (*Neogaeumannomyces
kevinifiliformis*, *Mycosphere*):

| paragraph | content | field |
|---:|---|---|
| 1219 | `Neogaeumannomyces kevinifiliformis … sp. nov.` | nomenclature |
| 1223 / 1237 | etymology / holotype | — |
| 1239, 1245 | the description | description |
| 1247 | material examined | materials_examined |
| 1249 | `Notes – N. kevinifiliformis resembles N. bambusicola…` | notes |
| **1253** | **`Rhodoveronaea Arzanlou, W. Gams & Crous, Studies in Mycology 58: 89 (2007)`** + genus intro | **diagnosis** |
| 1255 | `Rhodoveronaea hyalina X.M. Chen & Karun.` | *next treatment* |

The treatment ends at its `Notes` (1249). Paragraph 1253 is the
*Rhodoveronaea* genus entry, and **`taxon_1cfac841` — the treatment
that owns paragraph 1255 — has an empty `diagnosis`.** Donor has it,
recipient is missing it, and they are adjacent. That is the whole
detector.

**The consequence is not cosmetic.** Claude annotated
`Conidiophores`, `Conidiogenous_cells` and `Conidia` off that
paragraph, so *Rhodoveronaea*'s asexual morph is now recorded as
features of a taxon whose own description says `Asexual morph:
Undetermined.` Three fabricated features on one treatment, and the
underlying text is not even the same genus.

**Shape of the rule**, all from data already stored:

1. `min(diagnosis_spans.paragraph) > max(notes/materials_examined
   spans.paragraph)` — the diagnosis trails the treatment.
2. The next treatment from the same `ingest._id`, by nomenclature
   paragraph, has an empty `diagnosis`.
3. The trailing text opens with a name-plus-authority-plus-citation
   header rather than a `Notes`/`Comments` header.

Test 1 alone is far too broad — see the measurement below. **Test 2 is
the load-bearing one** and nothing else in the backlog uses it.

#### Measurement: what the trailing-diagnosis position actually contains

Over the 14 804 treatments carrying both a `diagnosis` span and a
`notes`/`materials_examined` span, **4 358 (29.4 %) have the diagnosis
trailing**. Classifying the opening of the diagnosis text, against the
10 446 normally-positioned ones as a control:

| opening form | trailing | normal | enrichment |
|---|---:|---:|---:|
| `Notes`/`Comments`/`Remarks`/`Discussion` header | 1 406 (**32.3 %**) | 1 311 (12.6 %) | **2.6×** |
| head-clipped (opens lowercase or on punctuation) | 1 384 (31.8 %) | 3 028 (29.0 %) | 1.1× |
| name + authority + citation header | 57 (1.3 %) | 88 (0.8 %) | 1.6× |
| other | 1 511 (34.7 %) | 6 019 (57.6 %) | 0.6× |

Three things fall out, and only one of them is D19:

* **The dominant tenant of the trailing position is a misrouted
  `Notes` block**, not a next-taxon header — 2.6× enriched, and
  unambiguous, since the source names its own section and it is not
  "Diagnosis". §0.5 already records the field-mapping at a different
  denominator (1 884 Notes-only + 932 both, of 18 787); this adds that
  **position predicts it**.
* **Head-clipping is not positional** — ~30 % either way. So roughly
  4 400 `diagnosis` fields open mid-sentence regardless of where they
  sit, which is §10/D11 applied to a field D11 does not currently read.
  Worth folding into D11 rather than a new item.
* **D19's own signature is rare** — 57 treatments at most, before
  applying the recipient test that would raise its precision. Small,
  but each hit fabricates features on the wrong taxon, so the cost per
  miss is high.

#### A candidate detector that this case suggested and the data killed

*"`Asexual morph: Undetermined` in the description while conidial
features are described anyway"* looks like an airtight self-
contradiction, and `taxon_fd50457a` matches it exactly. It does not
survive contact with the corpus.

| | n | share |
|---|---:|---:|
| `Asexual morph` explicitly undetermined | 1 689 | — |
| …conidial term later in the description | 265 | 15.7 % |
| …conidial term in the diagnosis | 86 | 5.1 % |

**A 7-treatment sample of the in-description form is 7/7 false
positives.** The cause is uniform and obvious in hindsight: the
`Culture characteristics` block that follows almost every such
description mentions conidia belonging to the **host** or to the
culture medium — *"obverse fawn due to conidia of the host"*
(`taxon_2294eee2`), *"Colonies … due to coloured exudates"*
(`taxon_25bbd681`). The term is present; the claim is not.

Recorded so nobody rebuilds it. The diagnosis-side variant (86) is
untested and inherits the same doubt — **do not cite the 5.1 % as a
rate of anything** until it is sampled.

#### An unrelated true negative worth pinning: the trophic-mode opener

The operator's other observation on this treatment was surprise that
the description **opens with ecology** — `Saprobic on dead branches of
bamboo.` before any anatomy.

**That is the house style, not a defect.** The Hyde-school genre
(*Mycosphere*, *Fungal Diversity*) opens every description with the
trophic mode and substrate, then `Sexual morph:` / `Asexual morph:`.
Corpus-wide **1 434 of 42 096 descriptions (3.41 %)** open with a
trophic-mode word — `Saprobic`, `Parasitic`, `Endophytic`,
`Pathogenic`, `Associated with`, and the rest.

It matters because it is a **false-positive magnet for §10**: the
sentence has no anatomical subject and no capitalised feature term, so
a "description starts mid-sentence" rule keyed on an expected opening
noun fires on all 1 434. `taxon_fd50457a` is the gating fixture for
that being legitimate.

**Depends on**: nothing for tests 1 and 3. Test 2 needs sibling
treatments ordered by nomenclature paragraph within an `ingest._id`,
which `bin/treatment_dossier` (plan T3e) already proposes to compute.

**Gating fixtures**: must fire on `taxon_fd50457a`
(`§6-next-genus-header-absorbed-into-diagnosis`) and must **not** fire
on `taxon_17320412` (`§6-compact-congenerics`), whose `diagnosis` is a
genuine `Notes –` block correctly attached to its own taxon.

### Correction to the fix-sequencing list above

Item 2 of that list ranks §5 gating as "likely highest
corpus-wide payoff per unit work" and proposes gating stub
creation on the source paper having a real Nomenclature
heading, or on `skol_dev.taxonomy`.  **`taxon_0a8c1077`
passes both gates.**  Its source paper formally describes
three new species and a new combination.  The gate is
paper-level; the defect is paragraph-level.

The related claim in §5 that these false positives
self-quarantine — Claude returns 0 annotations, so they
never reach the review directory — is also conditional.
taxon_0a8c1077 drew 5 annotations, because a summary of
seven species' colony morphology is written in genuine
morphological language.  `annotation_count = 0` is a marker
for *non-morphological* orphan paragraphs only.

Neither correction changes the §5 diagnosis; both change its
proposed fix, which is why D1 exists.

### Fixture coverage gap

The convention in `tests/fixtures/README.md` is that every
taxon mentioned in this memo has a fixture entry.  As of
2026-08-21, **22 of the 61 referenced taxa do not**:

`taxon_09507677`, `taxon_2114314b`, `taxon_22346900`,
`taxon_3c218a38`, `taxon_418bf6b7`, `taxon_592128a8`,
`taxon_7af2e7c8`, `taxon_841d5cbe`, `taxon_876c18ec`,
`taxon_9e68c26b`, `taxon_9ecad903`, `taxon_a21a83f4`,
`taxon_ba964a8b`, `taxon_d2d26d25`, `taxon_d41b87e4`,
`taxon_d5525987`, `taxon_e0d2e4bb`, `taxon_e44e35bc`,
`taxon_e6402cd3`, `taxon_e74d89b1`, `taxon_ed2a6f1c`,
`taxon_f00f8353`.

Three of those directly gate backlog items — `taxon_d2d26d25`
(D3), `taxon_9ecad903` (D4), and `taxon_418bf6b7` (the
anatomical-noun-clip pattern, whose controls taxon_38b5b1c6
and taxon_09b97d5f are both present).  Capture at least
those three before starting the corresponding item.


## Sample-size caveat

A 5-treatment sample is too small to extrapolate corpus-wide rates,
but a few suggestive observations:

- 3/5 (60%) have `synthetic_nomenclature: true`, with at least one
  (T4) being a clear *real* treatment that lost its name.  This
  rate would warrant urgent investigation if it held corpus-wide.
- 1/5 (20%) is a non-taxonomic-paper false positive (T1).  Even at
  half this rate corpus-wide, that's substantial noise.
- 2/5 (40%) are multi-species merges (T3, T5).  Likely concentrated
  in flora/monograph-style sources rather than uniform across the
  corpus.

What this captures is "issues that exist", not "issues that
dominate."  The next round of sample-then-review (Phase 1
deliverable 6+) will expand the visible surface; revisit the
severity ordering and corpus-rate guesses then.
### 9. Corrupt OCR text (two modes)

**Mode A — U+FFFD runs** (the original observation).

**Symptom**: a Treatment field contains long runs of
`�` (U+FFFD REPLACEMENT CHARACTER), the Python decoder's
substitute for bytes it can't interpret as UTF-8.  Visible in
Fauxton as long strings of replacement-glyph boxes.

**Evidence**:

* **`taxon_43a7b19e...`** — noted 2026-08-21 from round-4.
  **Mode B — space displacement, with zero U+FFFD.**  The
  Mode-A detector cannot see this one at all.  Operator: "a
  mess … OCR errors that seriously altered the placement of
  spaces … I think we should have rejected this treatment due
  to the large number of invalid words."  Agreed on both
  counts.  A scanned typescript, corrupted two ways at once:

  * **spaces inserted inside words** — `demati aceous f ung i
    wi t h fissi on a rthroconidia`, `Howeve r`, `hyal ine`,
    `s t rai ght`;
  * **character substitution** — `artbroconiuia` for
    *arthroconidia*, `vide`/`vith`/`saooth`/`sore`/`celts`
    for *wide*/*with*/*smooth*/*more*/*cells*, `Pigs.` for
    *Figs.*, `Colon~es`, `septu~`, `piqgented`, plus wreckage
    like `Arthi2S£~ph`, `Aq;;uaden9_£Q!l`, `Geotri£hY!`.

  **The content isn't a treatment either**: introductory
  remarks on form-genera, then a materials-and-methods block
  (PYE agar from Baltimore Biological Laboratories, Pablum
  baby food, cellophane-plate preparation), then several
  strain groups (`a) Group 1 (Strains 1148, 507, 511, …)`),
  then a literature discussion of Cole & Kendrick 1969.

  **Worst scatter in the fixture**: 20 `description_spans`
  running char 57 101 → 174 783 — **117 682 characters of
  source across paragraphs 293 → 781**, 488 paragraphs apart
  — harvested into a 7384-char description.  taxon_3d0a3c69,
  the previous worst, spans 15 833.

  `synthetic_nomenclature` is True and the treatment is
  `Nomen ignotum`, so it is a §5 orphan stub too — and like
  taxon_0a8c1077 it did **not** self-quarantine.  Claude
  produced 19 annotations and it reached the reviewer.
  Claude's robustness is itself worth recording: it labelled
  `Arthroconidia` on a block reading `The artbroconiuia
  (rigs. SH, SI) a re di fficult to distinguish`, recovering
  the term through both corruption modes at once.

  **Six flags fire** — `§2:synth_nomen`,
  `§6:authored_binomial`, `§10:tail_clip`,
  `§10:diag_head_clip`, `§12:desc_span_gap`,
  `§13:no_source_anchor`, more than any other fixture entry —
  **and none of them names the OCR damage**, which is the
  actual reason to reject it.

  **Second observed instance, 2026-08-22**:
  `taxon_5fe9223f...` (*Vararia lincangensis*), six
  `plazi`-only source anchors, carrying the same joins —
  `Description.Basidiomata`, `wide.Hyphal`,
  `smaller.Basidiospores`, and in other fields
  `Etymology.Lincangensis` and `examined(paratype)`.  Not
  given its own fixture entry: it is the same root cause and
  the same join shape as taxon_30d8d8d4, and this section
  keeps one canonical exemplar per class.

  It is recorded here because it is otherwise
  **poster-child grade** — 10 annotations covering the
  description with zero interior or tail gaps, only the
  `Description.` header left uncovered, merge_metric = 2,
  every detector silent — and because its corticioid
  vocabulary (`Dichohyphae`, `Gloeocystidia`,
  `Generative_hyphae`, `Skeletal_hyphae`, `Basidioles`,
  `Sterile_margin`, `Hymenial_surface`) appears nowhere else
  in the fixture.  **It cannot be filed in
  `poster_children` while §15 stands**: poster children must
  fire nothing, and **D6 must fire on this treatment**.
  Revisit it as a corticioid poster child after the §15
  re-extraction, when the artifact is gone.

  **Detector evidence, tested against the whole fixture
  set.**  The naive word-validity metrics do *not* separate
  it:

  | metric | this | conflicting entry |
  |---|---:|---|
  | orphan 1-char rate | 12.8 % | 15.7 % (§10-head-clip-only), 13.1 % (discomycete **poster child**) |
  | out-of-vocabulary rate | 25.0 % | poster children span 16–47 %; the Latin/English pair hits 46.7 % |

  What *does* separate it is a **rejoin rate**: the fraction
  of tokens inside a run of 2–4 tokens that is
  out-of-vocabulary piecewise but forms a dictionary word
  when concatenated — `f ung i` → *fungi*, `wi t h` → *with*,
  `hyal ine` → *hyaline*.  This treatment scores **19.13 %**.
  **All 15 poster children score 0.00 %**, the next-highest
  entry of any kind is 1.50 %, and only 5 of 47 fixture
  entries are non-zero at all.  Any threshold between 2 % and
  19 % separates cleanly.  Tracked as **D8**.
* **`taxon_cda95f9f...`** (Colletotrichum, discovered during the
  2026-06-29 50-treatment intermediate run).  The `diagnosis`
  field has hundreds of U+FFFD characters interleaved with
  legible English fragments — looks like an OCR pass where
  every other line failed and got byte-stuffed.  Claude refused
  to annotate it (returned `content=[]`, 1 output token); the
  Phase 1 bootstrap reports as `status: error` with the
  diagnostic message added in commit (to-be).
* **`taxon_95dbdfb9...`** (interstitial-noise variant,
  noted 2026-07-02).  Description contains several long
  U+FFFD runs but text before and after each run is legible
  and mostly coherent — sometimes the surrounding text flows
  together (looks like the OCR dropped a word or two);
  sometimes it reads like a bigger chunk got lost.  **Not
  fatal to the annotator**: Claude produced 19 annotations
  (contrast with cda95f9f where Claude gave up entirely).
  Sub-symptom shape distinguished from cda95f9f: interstitial
  noise inside otherwise-recoverable prose vs. whole-block
  corruption.  Detection would use the same U+FFFD-density
  metric but with a lower threshold — e.g., any single run
  of ≥ 20 consecutive U+FFFD chars is a partial-corruption
  signal even when the overall density stays under 5 %.

**Affected treatments**: `taxon_cda95f9f...` (fatal),
`taxon_95dbdfb9...` (interstitial); likely others —
worth a corpus-wide scan with `count_replacement_chars(text) /
len(text) > 0.05` as the heuristic.

**Likely stage**: upstream of the v4 layout CRF — the original
ingest (`bin/ingest.py`) extracted text from PDFs where some
pages OCR'd cleanly and others didn't, and the failures got
encoded as U+FFFD instead of being skipped.  Possibly a
pdftotext / pdf2txt edge case on damaged or stylized fonts.

**Severity**: low — affects a small fraction of treatments
(rough guess from one hit in 50 sampled: ~2%); the bootstrap
annotator handles them cleanly as `status: error` (since the
fix in commit (to-be)) and they're excluded from features_hand
by §0 rule 2 if a reviewer encounters one.

**Operator action**: skip the whole treatment per §0 rule 2 —
no anatomical features can be extracted from gibberish.
Eventually the upstream OCR step should detect and quarantine
these rather than producing corrupt extracts.

#### §9.1 A literal backslash costs the whole treatment

Found 2026-08-25 during round 5's annotation run.  Two of the
first 250 treatments failed outright with

> `ClaudeResponseError: response is not valid JSON:
> Invalid \escape`

— `annotation_count: 0`, the entire treatment lost.  Not a
truncation, not a rate limit: the model emitted a backslash
sequence that is not a legal JSON escape, and the envelope
failed to parse.

**The predisposing cause is OCR damage, and it is a new
substitution pattern.**  `taxon_b5af6259`'s source reads

> `asci … saccati vel ovato-oblongi, 60—85 \7 25—35 u`
> `sporae … hyalinae, 18—20 \7 9—12 n`

**`\7` is a misread `×`.**  The multiplication sign in a
measurement became backslash-seven — so this belongs with mode
C (character substitution), and it is invisible to both the
rejoin metric and the U+FFFD scan.

**Measured on the round-5 draw**, which is a uniform random
sample of p1 and therefore a corpus estimate: **26 of 1 000
treatments (2.6 %) contain a literal backslash** in their
rendered synthetic document.  The character following it is
almost never a legal escape:

| next char | `'` | `1` | `\` | `7` | `"` | `:` | `v` | `-` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| count | 11 | 5 | 4 | 4 | 4 | 4 | 4 | 3 |

Only `\\` and `\"` are legal; the rest are hazards.

**But the source is not the whole story.**  The other failure,
`taxon_30b728af`, has **zero** backslashes anywhere in its
rendered document and still produced an invalid escape at line
6.  So there are two causes:

1. **Source-borne** — a literal backslash the model must
   double-escape and sometimes does not.  ~2.6 % of treatments
   are exposed.
2. **Model-borne** — an invalid escape emitted spontaneously
   against clean input.

**Both are retryable**, since the annotator sets no temperature
and a re-run samples differently; default mode re-runs
`status='error'`, so re-feeding the round file resumes exactly
these.  Observed rate in chunk 1: **2 of 250 (0.8 %)**.

**Worth fixing upstream rather than only retrying.**  A
pre-render pass that escapes or strips lone backslashes would
remove cause 1 entirely, at no risk to the text — a lone
backslash in a mycological description is always OCR damage,
never content.  Cheaper than losing 0.8 % of every future run,
and it also repairs the *measurement*: `\7` should read `×`.

### 10. Description starts mid-sentence

**Symptom**: a Treatment's `description` field begins with
punctuation (`;`, `,`, `.`) or a lowercase letter, indicating
the layout CRF's `Description` label started at a paragraph
boundary that wasn't actually where the source description
began.  The treatment is real; it's just clipped at the top.
Different from §5 (whole treatment invalid) — this is a real
treatment with valid downstream content, just missing its
opening.

**Evidence**:

* **`taxon_acd88732...`** — discovered during 2026-07-01,
  extended 2026-07-03.  `description` field verbatim:
  ```
  ; perithecia epiphyllous, slightly prominent, black, shining;
  spores subcylindrical, straight or somewhat curved, or
  subflexuous
  ```
  The leading `; ` is a dead giveaway that the source sentence
  started before this cut point.  Claude correctly identified
  the 2 features that ARE described (Perithecia + Spores) and
  labelled them cleanly.  Also co-occurs with §2 (taxonomic
  citation not extracted) — presumably the citation lived in
  the clipped-off head of the sentence.
  **2026-07-03 follow-up**: operator suspects
  **tail-clipping too** — the description ends
  mid-clause (`subflexuous` without a period or a
  connecting-clause resolution).  Two-ended clipping,
  same shape as taxon_23d479f4 and taxon_9ecad903's
  sub-symptom.  Also: **the description is
  clade-incomplete** — an ascomycete treatment showing
  only Perithecia and Spores omits the usual Ascomata,
  Asci, Ascospores, Paraphyses, Peridium, Ostiole
  content.  Operator epistemological note:
    > "I'm not sure how we could decide that we have
    > enough of the features documented, but this pair
    > is clearly not enough."
  **Completeness detection — plan of record (decided
  2026-07-03)**: SBERT-neighborhood comparison.
  Rationale: inherits clade-appropriateness from the
  neighborhood without needing explicit clade
  classification; zero-config; uses the same
  embedding space the search product already
  computes.  taxon_acd88732 (2 features) sits near
  other ascomycetes (typically 8-10 features per
  treatment), so it flags reliably against the
  neighborhood's distribution.  Legitimate
  clade-appropriate short descriptions like
  taxon_d65547ed (asexual mould, cultural-only,
  ~6 features) sit near other asexual moulds with
  similar counts and do NOT flag.

  **Concrete rule**: for each treatment, pull top-K
  SBERT neighbors, compute their `annotation_count`
  distribution, flag treatments whose count is below
  the 10th percentile of their neighborhood.  K to be
  calibrated during implementation (initial guess:
  K = 20).  Threshold percentile also calibrated
  against the false-positive corpus (taxon_b9a6232,
  taxon_9e048013 — must not flag as incomplete
  since they ARE complete for their taxa).

  **Alternatives considered and rejected** (recorded
  so future work doesn't need to re-derive the
  reasoning):
    (a) **Feature-count threshold** — rejected
        because clade-sensitive.  Asexual moulds
        legitimately have 2-3 features
        (taxon_d65547ed poster-child); boletes need
        6+.  A global threshold would either flag
        legitimate short descriptions or miss
        incomplete ascomycete descriptions.
    (b) **Clade-expected feature set** — rejected
        because it requires upstream clade
        inference from the Nomenclature.  Would work
        in principle but the plumbing (nomenclature
        → clade → expected-features table) is
        substantial infrastructure vs. option (c)'s
        zero-config approach.  Might revisit if
        SBERT-neighborhood proves too noisy.
    (d) **Description-length percentile** — rejected
        for the same clade-sensitivity reason as (a),
        AND because character length is a weaker
        proxy for completeness than the annotation
        count (a wordy but repetitive description
        would fool it).

  **Implementation gate**: the SBERT space needs to
  be reasonably populated first — we need enough
  neighbors per treatment for the percentile to be
  meaningful.  The current 45 k-treatment
  production_v4 corpus likely qualifies.  Not
  blocking Phase 1 review work; a candidate for
  post-Phase-1 corpus-cleanup tooling.

  **Advisory-only, not gating (operator note
  2026-07-03, taxon_ae45a05e)**: the flag surfaces
  low-content treatments for triage/review priority,
  but MUST NOT quarantine them from the search
  product.  Truncated descriptions of legitimate
  species (e.g., taxon_ae45a05e's 378-char
  tail-clipped basidiomycete) retain search value
  even with missing content.  The completeness
  signal informs the operator's re-extraction queue,
  not the search index.
* **`taxon_592128a8...`** (Nomenclature-tail variant) —
  reported 2026-07-02.  The description opens with the trailing
  fragment of a taxonomic citation ("... should have been
  Nomenclature").  Same class of extraction failure as
  taxon_acd88732 (Description head clipped), but the clipped
  material is identifiable content: a description starting
  with citation punctuation (`, Author, Year`), a partial
  species epithet, or `sp. nov.` fragment is almost certainly
  a Nomenclature-tail leak — a stronger signal for automated
  detection than the generic "starts with lowercase" rule.
  Combines with §6 findings (this treatment is a multi-species
  merge) — the Nomenclature-tail leak here is the *first* of
  several fragmentary citations that ended up in Description.
* **`taxon_23d479f4...`** — noted 2026-07-02, revised.
  Description opens with a list of carbon-source
  substrates (`glycerol, methanol, hexadecane,
  erythritol, levulinic acid, …`).  Nomenclature is
  real; single species.  Missing lines at both top AND
  bottom of the description.  Two-ended clipping
  variant — same class as taxon_d2d26d25 but here the
  clipping falls on legitimate description content
  rather than a mis-routed diagnosis.
  **Correction 2026-07-02 (operator)**: Cultural
  characteristics DO generally belong in Description;
  this is NOT a §12 leak.  The v4 layout CRF does not
  emit "Methods" / "Results" section labels either.
  What we see is a Cultural characteristics section of
  a legitimate treatment that was clipped at both ends
  during extraction.  §10 detector fired correctly
  (leading `glycerol,` lowercase); the double-ended
  clipping needs a separate detector — text ends with a
  clause fragment or trailing comma rather than a
  sentence-final period could serve as the tail-clip
  signal.
* **`taxon_7fbc71a8...`** — noted 2026-07-02.  Clean
  low-noise §10 case: description opens with
  `subcuticular, black, elliptical to elongated-elliptical,
  raising the substrate` (a lowercase start).  Operator
  confirmed the treatment is otherwise a good single-
  species description; just missing a line or two at the
  top.  No co-occurring §2 / §6 signals (merge_metric = 5,
  synth_nomen = False).  Contrast with taxon_acd88732
  (co-occurring §2 missing citation) and taxon_592128a8
  variant (compound with a multi-species merge) —
  taxon_7fbc71a8 is the cleanest observed instance of
  §10-only, showing the detector fires reliably on
  otherwise-unremarkable treatments.
* **`taxon_3c218a38...`** — round-2 reviewed by
  piggy@puchpuchobs, noted 2026-07-03.  Genus
  treatment of Cordyceps-like fungi (Ophiocordyceps
  or a related segregate genus).  **Compound §1 +
  §10 head/tail + §12 case**:
    - **Head-clipped emendation-and-synonym leak**:
      description opens `emend. G.H. Sung, J.M. Sung,
      Hywel-Jones & Spatafora= Cordycepioideus
      Stifler, …` — the tail of a genus emendation
      (`emend. Authors`) followed by a heterotypic
      synonym (`= Cordycepioideus Stifler`).  The
      genus name and original citation preceded
      "emend." and got clipped.  New §1 sub-variant:
      genus-emendation tails, distinct from the
      binomial-tail variants of taxon_592128a8 and
      taxon_2f276bfa.  Detection: the token `emend.`
      at description head is a strong signal
      independent of binomial parsing.
    - **Type designation clause in Description** —
      §12 leak, same class as
      taxon_95dbdfb9/taxon_f00f8353's Holotype-in-
      Description.
    - **Tail-clipped**: per operator, description
      "finishes with an adjective and no punctuation"
      — pure tail-clip like taxon_ae45a05e.
    - **§6 stays quiet correctly** (single-genus
      treatment, not multi-species).
    - **§10 fires correctly** on the head clip.
  **Reviewer data notable — 1:1 add ratio**: kept 9,
  added 9, deleted 0.  Highest ratio observed in the
  reviewed set (compare to round-1 clean species
  which typically add 0-6 to Claude's baseline).
  Suggests Claude was under-annotating on this
  Cordyceps-like genus description — possibly because
  its vocabulary is under-represented in the seed
  examples, or because the genus-level anatomical
  granularity is different from species-level.
  Worth watching for other genus-treatment reviews to
  see if the pattern holds.

* **`taxon_ae45a05e...`** — noted 2026-07-03.  **First
  pure TAIL-CLIP case** in §10.  Description starts
  cleanly (`Basidiomata small to medium-sized, dry to
  glutinous, yellowish to greenish. Pil…`) and just
  runs out — 378 chars, no proper sentence-final
  punctuation at the tail.  Head-only §10 detector
  (`desc_starts_mid_sentence`) correctly stays silent;
  no current detector catches tail-only clipping.
  Would be caught by the mid-word tail-clip detector
  idea recorded for taxon_9ecad903 (`[a-z]-\s*$`) or
  by a "description doesn't end with a sentence-final
  period" heuristic.  Distinct from taxon_23d479f4
  (both-ends clipped) and from taxon_acd88732 (head +
  tail clipped) — first case observed of tail-only.
  **Search-UX operator note (2026-07-03)**: "This
  would be an OK result to return in a search."  Even
  truncated, the treatment carries useful anatomy
  content for surface-level identification.  Refines
  the completeness-detection plan of record
  (§10 SBERT-neighborhood): flagging as
  "under-populated" should be **advisory only** —
  driving triage/review priority — NOT gating search
  inclusion.  Truncated descriptions of legitimate
  species retain search value.
* **`taxon_418bf6b7...`** — noted 2026-07-03.  Second
  clean §10-only case; single-species ascomycete
  treatment, description opens `immersed, scattered or
  in groups. Venter 390–500 μm diam, …`.  Operator:
  "the gross description of the ascomata is missing."
  Same specific clip pattern as taxon_7fbc71a8:
  **anatomical-noun clip** — an adjective (`immersed`,
  `subcuticular`) describing an anatomical noun
  (`Ascomata`) survives extraction, but the head noun
  itself gets clipped off.  Two cases in the sample now
  makes this a repeatable extraction failure mode —
  worth a distinct label if it recurs.  Layout CRF likely
  splits the `<Noun> <adj-run>` phrase at a line boundary
  where the noun sits alone on the line above.
  Detector fires correctly; content is annotate-able (9
  annotations from a 1367-char description).
* **`taxon_67cc93d2...`** — round-2 reviewed by
  piggy@puchpuchobs, noted 2026-07-03.  At least 2
  species of slime mold (Myxomycota) — a new clade
  for the memo, distinct from the basidio/asco/lichen
  cases catalogued so far.  Multiple detection
  signals present but each partially caught:

  * **§1 head-of-Description citation**: opens with a
    cleanly-OCR'd full taxonomic citation
    `Comatotricha  afroalpina  Rammeloo,  Bull.  Jard.
    Bot.  Belg.  53:  297,  1983 …` (with double-
    spaced characters — plausible OCR artifact from
    the source typography).  Same head-of-Description
    class as taxon_2f276bfa but with a clean binomial
    that gnfinder would parse successfully.
  * **§6 idea #1(b) E→L→E target**: the operator
    identified the English → Latin → English
    ordering pattern (a Latin diagnosis sandwiched
    between English content).  Would be caught by the
    order-aware Latin-block detector that fires on
    single Latin blocks in non-terminal position.
    Current `latin_block_count = 0` (probably 1
    non-firing block) and the ordering detector
    isn't implemented yet.
  * **§12 leak**: the Latin diagnosis carries habitat
    and type designation lines that should have been
    in the treatment's own `habitat` /
    `type_designation` fields.

  **Detectors that fired**: `§2:synth_nomen` correctly
  (nomenclature not attached despite the citation
  being visible in Description).
  **Detectors that missed**: `§6:merge_metric = 7`
  (below threshold — 2 slime mold species with
  clade-specific anatomy don't accumulate above
  k=5), no header repetition, no `sp. nov.` count.
  If §6 idea #1(b) or §6 idea #2 (gnfinder) were
  implemented, this treatment would be caught.

  **Reviewer data**: kept 26, added 7, deleted 0.
  Zero deletions on a multi-species merge is unusual
  (contrast taxon_09507677 with 3 deletions on its
  3-species merge).  Suggests Claude did a
  particularly clean job on this treatment despite
  the compound problems, or the reviewer applied §0
  rule 3 (annotate first species only) strictly and
  the 26 Claude annotations happened to line up with
  species 1's coverage.

* **`taxon_09507677...`** — round-2 reviewed by
  piggy@puchpuchobs.  At least 3 descriptions
  concatenated (per operator's Basidiocarp-clause
  count).  **First observed 3-species case CAUGHT by
  merge_metric** (value = 15 vs threshold 10) —
  contrast with taxon_d41b87e4 (3 species, metric =
  0, MISSED).  The difference: this treatment's 4017-
  char description gives ~1339 chars per species, so
  shared basidiomycete anatomy terms (Basidiocarps,
  Pileus, Stipe, etc.) accumulate to 5+ mentions
  each.  taxon_d41b87e4 was denser (1675 chars / 3
  species ≈ 558 chars each) and fell below the
  count threshold.

  **"Basidiocarp counting" as the operator's manual
  signal** is exactly the "structural anatomy
  doubling" (or here trebling) idea suggested for
  taxon_ed2a6f1c.  Automated form: for a curated
  list of specific structural words (Basidiocarps,
  Basidiomata, Ascomata, Asci, Paraphyses,
  Conidiomata, Conidia, Pileus, Stipe, …), fire on
  count >= 2 (compact congenerics) or >= 3 (3+
  species).

  **Reviewer data**: kept 31 Claude annotations,
  added 6, DELETED 3.  First non-zero deletion count
  in the recently-reviewed set (round-1 2b793602 and
  841d5cbe both had 0 deletions).  The multi-species
  content probably caused some cross-species
  annotation confusion Claude couldn't resolve.
  Interesting operational data: even with §0 rule 3
  ("annotate first species only"), multi-species
  treatments generate more reviewer-rejected
  annotations than clean single-species ones.

* **`taxon_2b793602...`** — round-1 reviewed by
  piggy@puchpuchobs (kept 19 Claude annotations, added
  1, deleted 0 — Claude did well despite the compound
  problems).  Revisited 2026-07-03 with the accumulated
  round-2 lens.  Large (7875-char) treatment stacking
  **three distinct failure modes**:
    - **§6 3+ species merge** — three basidiomycete
      descriptions (Pileus + Lamellae + Spores +
      Stipe), each followed by taxonomic citations.
      merge_metric = 39, correctly flagged.
    - **§11 genus + type-species addition** — one of
      the "descriptions" is actually a GENUS
      description that includes the type species,
      same shape as taxon_01a01c54.
    - **§8 key content in Description tail** — a run
      of key couplets appended at the end of the
      Description.
  Two detector-gap observations:
    (1) **Numberless couplets** — the trailing key
        content has the couplet numbers STRIPPED
        (`_COUPLET_LINE_RE = ^\s*\d+[a-z]?[.)]\s+[A-Z]`
        requires the leading digit; numberless couplets
        slip past entirely).  Cause: probably a source
        typography choice or an extraction step that
        removed line-leading numbers.  Detector
        refinement: fall back to a shape-based signal
        when no leading number is present — see (2).
    (2) **Terse 1-to-2-features-per-line format** —
        the couplet run has very short lines (a
        feature or two per line) while normal
        description prose has multi-sentence
        paragraphs.  Refinement: score lines by
        approximate word count; a run of consecutive
        short lines (<20 words) inside an otherwise
        prose-heavy description is a §8 signal even
        without couplet numbers.  Would compose with
        the existing numbered-couplet regex — either
        signal fires flag §8.

* **`taxon_d41b87e4...`** — noted 2026-07-03.  Three
  complete basidiomycete descriptions merged in one
  treatment.  All three species have the same
  anatomical structure (Pileus + Lamellae + Spores +
  Stipe), so ~3 mentions of each anatomy term across
  a 1675-char description — each stays below the k=5
  count threshold, so `merge_metric = 0`, MISSED.
  Compact-multi-species pattern generalized to 3+
  species (earlier congeneric-2 cases: taxon_173204,
  taxon_ed2a6f1c).

  **Structure per operator**:
    - Species 1: description → `Illustration:` reference
      → taxonomic citation with `sp. nov.` (a new
      species).
    - Species 2: description → type designation →
      DISTRIBUTION line → taxonomic citation → detailed
      taxonomic citation with publication details.
      Species 2 is a redescription of an existing
      species (no `sp. nov.`, hence n_sp_nov = 1 total
      not 3).
    - Species 3: same shape.

  **Detectors that fired correctly**:
    - `§8:key_couplets` fires (`n_key_couplets = 2`) —
      the numbered species headings (e.g., `2.
      Species-name`) trip the couplet-line regex.
      Semantically not key couplets, but they ARE
      species boundary markers, so the fire is
      operationally correct even if the sub-label is
      wrong.  Argues the `n_key_couplets` detector is
      doing double duty: real key couplets (§8) AND
      numbered species headings (§6).  Worth
      distinguishing later — e.g., is the numbered
      line followed by a species name (Genus species)
      or by anatomical prose?  For now, both fire the
      same flag.
    - The `Illustration:` header appears in species 1's
      tail; confirms the taxon_95dbdfb9 watchlist
      addition was worth it.

  **Detectors that missed**:
    - `merge_metric = 0` — as noted above, 3-fold
      repetition of shared anatomy terms doesn't reach
      k=5.
    - `n_sp_nov = 1` — correctly counted the single new
      species, not a miss but a reminder that sp.-nov.
      counts fire only when the merge involves multiple
      NEW species.  Redescriptions of existing species
      don't add to the count.
    - `latin_block_count = 0` — description is English
      throughout.
    - Header-count detectors: no `Diagnosis:` or
      `Description:` headers repeated.

  **§12 sub-symptoms in the same treatment**:
    - **Type designation in Description** (species 2's
      tail) — should have been in the
      `type_designation` field.  Same class as
      taxon_95dbdfb9's Holotype-in-Description leak.
    - **DISTRIBUTION line in Description** — should
      have been in its own field.  **New §12 leak
      target**: `DISTRIBUTION` sub-content in prose.
      The treatment_prose schema DOES have a
      `distribution` field (see taxon_876c18ec doc
      dump earlier); the layout CRF is likely
      mis-labelling these lines as `Description`.
    - **Taxonomic citations in Description** (both
      species 2's brief and the followup with
      publication details) — §1 pattern, gnfinder
      would catch.

  Reviewer treatment: 3-species merge; apply §0 rule
  3 (first species only).

* **`taxon_e0d2e4bb...`** — noted 2026-07-03.  Compound
  §6 + §9 case with an important epistemological question
  from the operator.  Multiple U+FFFD noise runs
  interspersed with legible prose (§9 interstitial-noise
  variant — text before and after the noise aligns in at
  least one instance).  Two species descriptions clearly
  concatenated: 2 sets of Stromata + 2 sets of
  Conidiophores + 2 sets of Conidia.  Mid-body boundary
  marker present:
    ```
    Illustration: Nakashina et al.[FFFD run] Description[FFFD run]
    ```
  — the same `Illustration:` + `Description:` header pair
  documented in taxon_95dbdfb9 (illustrated-monograph
  format), except here U+FFFD noise sits BETWEEN the label
  word and the content, so the current
  `\bDescription\s*[-–—:]` regex doesn't count the
  mid-body occurrence.  `n_description_headers = 1` (only
  the offset-0 header is counted), below the ≥ 2
  threshold, MISSED.  merge_metric = 1, also MISSED.

  **U+FFFD-tolerant header refinement**: extend
  `_DESC_HEADER_RE` (and its Diagnosis sibling) to accept
  U+FFFD noise as a header terminator alongside the
  existing `[-–—:]` set.  Concrete regex to avoid
  false-positive prose matches: `\bDescription
  [\s�]*(?=[A-Z])` — Description followed by any
  mix of whitespace / U+FFFD then a capital letter (the
  start of new content).  Similarly for Diagnosis and
  Illustration.  Would have caught this treatment.

  **Operator epistemological note (2026-07-03)**:
    > "I'm not sure how we'd readily distinguish this from
    > a description with different subfeatures for a given
    > feature, e.g. Conidia shapes, and later Conidia
    > color."
  The answer emerging from the accumulated §6 evidence:
  **use structural boundary markers, not anatomy-mention
  counts, to make the merge call**.  A single species can
  legitimately mention Conidia twice (shapes early, colors
  later — taxon_b9a6232 / taxon_9e048013 false-positive
  pattern).  What distinguishes a real merge is a
  BOUNDARY BETWEEN the two mentions: a Description: /
  Illustration: / Nomenclature header, a formally-cited
  binomial (§6 idea #2), or a Latin/English switch (§6
  idea #1).  In taxon_e0d2e4bb the boundary marker IS
  present — the `Illustration: … Description …` pair —
  but obscured by OCR noise, hence the detector miss.
  When NO boundary is present, treat as single-species:
  accept the risk of missing compact congeneric merges
  (taxon_173204, taxon_ed2a6f1c) so we can catch them via
  §0 rule 3 reviewer inspection rather than automatic
  quarantine.  This is the operating principle
  reflected in the current detector suite.

* **`taxon_9ecad903...`** — noted 2026-07-03.
  2-or-3-species merge stacking half a dozen sub-symptoms
  in one treatment.  Structure per operator:
    1. **Species 1 Description** — starts fine (`Pileus
       5-10 mm, semicircular, …`) but ends abruptly with
       `cinnamon or red-` (trailing hyphen mid-word).
       **New §10 sub-shape: mid-word tail-clip** —
       trailing hyphen at the end of description text is a
       strong signal that a page or paragraph break wasn't
       handled by the extractor.  Cheap regex detector:
       description ends with `[a-z]-\s*$`.
    2. **Latin block sandwiched between two English
       blocks (E → L → E)** — the Latin diagnosis appears
       between species 1's clipped English description and
       species 2's clipped English description.  **This
       ordering IS a pathology** (operator correction
       2026-07-03): normal taxonomic-paper structure puts
       Latin BEFORE its matching English translation (or
       the two live in separate labelled sections).  Latin
       appearing MID-body, surrounded by English on both
       sides, means the assembler collapsed adjacent
       species' content across a Latin diagnosis that was
       supposed to anchor one of them.  See §6 idea #1(b)
       below for the order-aware detector.
    3. **Species 2 Description** — starts lowercase after
       the previous sentence's period, so head-clipped
       (§10 classic).  My `desc_starts_mid_sentence`
       detector does NOT fire because the FIELD starts
       with the capital `P` of species 1; the mid-body
       lowercase-after-period boundary is invisible to
       the current head-only §10 rule.
       **Refinement TRIED and REJECTED (M2 Group C,
       2026-07-05)**: implemented and tested a
       `mid_body_species_boundary` detector matching
       `[a-z]{4,}\.\s*\n\s*[a-z]` (period after a real
       word, newline, lowercase start).  Fixture
       regression exposed a fatal FP class: some
       legitimate single-species treatments write new
       paragraphs starting with lowercase adjectives
       as stylistic continuation (e.g., taxon_b9a6232
       fires 3× on `phragrnosporous`, `a diameter of…`,
       `submembranaccous` — all legit prose within one
       species).  taxon_b9a6232 is a fixture-tracked
       false-positive regression target; any detector
       firing on it is disqualified.  Additionally, this
       detector wouldn't have caught taxon_9ecad903
       itself since species 1's boundary is a
       trailing hyphen, not a period.  Reverted.  A
       robust mid-body boundary detector needs
       paragraph-level section classification (M3
       segment classifier), not a regex.  Tracked as
       **D4** in the Detector backlog; sequence it with
       **D5**, which supplies the same capability.
    4. **Species 2 body** — appears complete, but has a
       **figure caption appended** at the end (§12 leak,
       same shape as taxon_ea7b0ed7).
    5. **Diagnosis field head-clipped** — starts abruptly,
       ends reasonably.  Second observed instance of a
       clipped Diagnosis field (after taxon_e44e35bc's
       double-clipped diagnosis).  Confirms clipping is
       not Description-only.  Detector idea: apply the
       same `desc_starts_mid_sentence` rule to the
       `diagnosis` field.
  Detector coverage: merge_metric = 1 (missed), all §6
  header counts = 0, `latin_block_count = 1` (single
  Latin block is normal — the Latin diagnosis of species
  1 — doesn't trip the ≥ 2 flag).  Zero §6 flags fired
  despite a clear 2-3 species merge.  Argues for the
  refinements listed above: mid-word-hyphen tail-clip,
  mid-body `.\s+[a-z]` transitions, diagnosis-field
  clipping detection — all cheap, orthogonal to the
  existing signals.

* **`taxon_ed2a6f1c...`** — noted 2026-07-02.  Two-species
  merge with the same structural pattern as
  taxon_173204 (compact-congenerics): **2 sets of Asci
  clauses + 2 sets of Paraphyses clauses**.  Anatomy
  vocabulary doubling is exactly 2-fold, so no term hits
  the k=5 count threshold — merge_metric = 2, below the
  10-threshold, MISSED.  No header repetition, no
  binomial, no `sp. nov.`.
  The §10 detector DID fire (desc_starts_mid_sentence
  = True) — first line "s ·ocarps intraepid rmal, …"
  shows both an OCR-corrupted head (middot substitutions
  and dropped characters: `·` for `c`, missing `e` in
  `epidermal`) AND clipped content — the leading `s` is
  a leftover from a previous sentence.
  Diagnosis is a legitimate Differential Diagnosis (§13
  sense), 981 chars, but the operator notes it's "a
  little chopped up" — same class as taxon_d2d26d25's
  clipped-diagnosis pattern (§12) but milder: the block
  is in the right field, just with damaged boundaries.
  Refinement idea (same class as the compact-congenerics
  discussion around taxon_173204): a "structural
  anatomy doubling" secondary metric that fires on
  count == 2 for a small list of specific structural
  words (Asci, Paraphyses, Ascospores, Basidia, Pileus,
  Stipe, Ascomata, Basidiomata, Conidiomata, …) would
  catch this class without inflating false positives
  from ordinary anatomy repetition in single-species
  descriptions.  Requires calibration against the
  taxon_b9a6232 / taxon_9e048013 false positives.

* **`taxon_83e36037...`** — noted 2026-07-02.  **All
  current §6 detectors missed** this two-species merge:
  merge_metric = 4 (below threshold), no header repetition,
  no `sp. nov.` count, no Latin blocks, no
  mid-body `Description:` header, no §10 clipping.  The
  single strong marker is a **formally-cited authored
  binomial in the middle of the description field**:
  `Trichaptum perrottetii (Lév.) Ryvarden`.  No
  legitimate description field should contain a formal
  authored citation — this is the §6-idea-#2 (gnfinder /
  gnparser) target signal in its cleanest form.  Adds
  urgency to implementing gnfinder detection ahead of
  the next bootstrap run.

  **Diagnosis-scoping caveat reinforced**: the diagnosis
  field contains **3 legitimate taxonomic citations** —
  comparisons with related species, which is what a
  **Differential Diagnosis** (§13 polysemy note) DOES
  (same insight as taxon_9e048013's false positive).
  gnfinder detection must scope to `description` only;
  folding the diagnosis in would guarantee false
  positives on every Differential Diagnosis.  This
  treatment is a precise validation of the
  Description-only scoping rule: 1 citation in
  Description (merge) vs 3 citations in Diagnosis
  (normal).

* **`taxon_a21a83f4...`** — noted 2026-07-02.  Extreme
  §10 + missed-§6 case.  Description opens with a **single
  clipped word `inconspicuous.`** — the tail of a sentence
  whose head was omitted (e.g., "Ascomata …
  inconspicuous.").  Even shorter than taxon_acd88732's
  `; perithecia …` opening; establishes that §10 clipping
  can be arbitrarily short.  The `desc_starts_mid_sentence`
  detector fired correctly.
  **Co-occurring uncaught §6 signal**: a literal
  `Description:` header appears mid-body, marking the
  start of a second species' description.  merge_metric
  = 5 (below threshold); `count_description_headers` = 1
  (below the ≥ 2 firing threshold).  Both current §6
  detectors missed the merge.  See §6 for the mid-body
  header refinement this case argues for.

**Affected treatments**: `taxon_acd88732...`,
`taxon_592128a8...` (variant), `taxon_7fbc71a8...`;
unknown corpus-wide rate — worth a scan.

**Detection**: reviewer-detectable by eye (leading punctuation
or lowercase first char).  Automated detection is easy: regex
match against `^[;,.\-\s]*[a-z;,.]` on the raw description
field.  Worth adding to a data-quality audit script.

**Likely stage**: layout CRF's Description label boundary
detection.  When the actual description-opening sentence spans
a page break, an inline formatting boundary, or a partial-OCR
skip, the CRF may pick up mid-sentence rather than at the
paragraph head.  Correlates with §2 (missing citations) —
citations often precede the description on the same or previous
line, so clipping the description head also loses the citation.

**Severity**: low-to-medium.  Claude's annotator handles these
correctly (produces valid annotations on the surviving
content); downstream training won't be poisoned since the
features that ARE labelled are legitimate.  Impact is on
completeness — those treatments contribute fewer features to
the training corpus than they should.  If this pattern is
common, a corpus-wide detect + re-extract pass would restore
significant training signal.

**Operator action** (Phase 1 hand-review): annotate the
features that ARE present (per usual conventions); flag the
treatment for the re-extract queue via a brat AnnotatorNote so
the eventual re-extraction pass can pick it up.

### 11. Genus + species-in-genus concatenation (`gen. nov.` + `sp. nov.` pattern)

**Symptom**: a treatment doc contains BOTH a new genus
diagnosis and one or more species descriptions within that
genus in the same fields.  Distinct from §6 (arbitrary
multi-species merge) because the two taxa are hierarchically
related — the species IS a member of the new genus, and both
sets of anatomical characters are semantically meaningful in
their own right.

**Evidence**:

* **`taxon_01a01c54...`** — discovered during 2026-07-01
  hand-review of the 50-treatment intermediate run.
  `description` field opens with:
  ```
  Pseudotrichia Kirschst. gen. nov.
  Perithecia dispersa vel gregaria, superficialia, solida,
      carbonacea, coactis vestita; ostiolo papillato ...
  Asci euparaphysati, clavati vel cylindracei, 8-spori.
  Sporidia fusiformia, hyalina, pluriseptata.
  102. Pseudotrichia stromatophila Kirschst.
  ```
  The genus diagnosis (`Pseudotrichia … gen. nov.`) followed
  by its type species (`102. Pseudotrichia stromatophila`)
  land in the same description field.  Claude correctly
  identified the dual-level anatomical content — 8 annotations
  across 5 labels, with 2 each for Asci / Perithecia / Spores
  (one from the genus diagnosis, one from the species
  description).

* **`taxon_3e98d44d...`** — noted 2026-07-07 from batch-2.
  **New genus (`Gaillardinia` — yeast) with a clean
  descriptive paragraph but 5 empty fields.**  Silent-
  failure pattern: the DESCRIPTION field is a clean 272-
  char anatomical paragraph (`Sexual reproduction not
  known. Colonies white, butyrous, smooth. Multilateral
  budding cells and blastoconidia are present… coenzyme
  Q-8.`) starting and ending cleanly.  All triage
  detectors correctly silent.  But comparing to the
  operator-supplied complete treatment reveals five
  empty fields with content that should have been
  captured:
    - Nomenclature: `Gaillardinia Q.M. Wang, Yurkov,
      Boekhout & F.Y. Bai, gen. nov. — MycoBank MB
      852166` (missing; 4th nomenclature/synth
      inconsistency in batch-2)
    - Etymology paragraph (missing)
    - Type species declaration `Gaillardinia entomophila
      (D.B. Scott et al.)…` (missing)
    - Long phylogenomic discussion paragraph about the
      C. entomophila clade (missing)
    - Notes paragraph about differences from Danielozyma
      and Metahyphopichia (missing)
  Silent-failure — no current detector catches "clean
  description, everything else empty" because signals
  operate on the extracted content, not on
  extracted-vs-source coverage.  Detection idea:
  `n_populated_fields` threshold, or field-length ratios
  against expected-per-taxon-class distribution.
  Requires modeling what fields SHOULD be populated for
  a given treatment class.
* **`taxon_9b787247...`** — noted 2026-07-07 from
  batch-2.  **New genus (`Rhizogene Syd. nov. gen.`) in a
  German-language paper.**  Operator supplied the complete
  intended extraction; comparison exposes multiple
  compound failures:
    - **Nomenclature missing**: `Rhizogene Syd. nov. gen.`
      is absent from the extracted `nomenclature` field
      (empty) despite `synthetic_nomenclature = False`.
      Data inconsistency — synth flag should be True
      when nomenclature is empty, or the extractor
      should have captured the nomenclature line.
    - **Mid-Latin truncation** at `omnino immersi;` —
      operator-corrected 2026-07-07: this is NOT at a
      language boundary.  The following `asci sporaeque
      adhuc tantum immaturi visi.` is still Latin and
      should have been captured.  The extractor stopped
      at a semicolon mid-clause.  Cause unclear — possible
      PDF pagination, section CRF confidence dropoff,
      semicolon-as-terminator heuristic, or another
      mid-block issue.  Worth investigation as a
      distinct §10 sub-shape (mid-Latin truncation
      distinct from head-clip or tail-clip patterns).
    - **German type-species note dropped**: the trailing
      `— Einzige Art: R. Symphoricarpi Syd. (= Zaszobotrys
      Symphoricarpi Syd.)` note that identifies the type
      species isn't captured.  Separate from the Latin
      truncation — the German content is missing
      regardless.  LOTL/LOTE detector territory (Trello
      #395).
    - **Operator observation**: description ends with
      `;` — confirms the existing `tail_clipped`
      detector correctly fires on non-sentence-final
      punctuation.  `§10:tail_clip` is the current
      captured flag.
    - **Line-classifier granularity limitation (operator
      note 2026-07-07)**: the trailing line
      `tantum immaturi visi. — Einzige Art:
      R. Symphoricarpi Syd. (= Zaszobotrys Symphoricarpi
      Syd.).` is a physical line containing THREE
      sections (Latin description tail, em-dash
      separator, German type designation).  The v4
      layout CRF classifies at line granularity — it
      picks one label per line, losing the others.
      This pattern is more common in **older texts
      (pre-1950 typography)** where authors compressed
      sections onto single lines to save space.
      Concrete evidence for §12's segment-classifier-
      as-assembly-aid plan: sub-line granularity is
      required to handle dense typography without
      losing content.  Segment classifier + em-dash-
      aware post-processing at M3 would recover this
      content class.
  Fixture-tracked as
  §11-gen-nov-latin-german-truncated.

**Affected treatments**: `taxon_01a01c54...`,
`taxon_9b787247...`; likely representative of a broader
pattern in taxonomic papers that propose a new genus
alongside its type species.

**Likely stage**: layout CRF + treatment-grouper.  The CRF
labels the paragraph containing both `gen. nov.` and the
numbered species heading as one Description block; the grouper
treats the pair as a single treatment.  Not clearly a
"missed nomenclature boundary" the way §1/§3 are, since both
nomenclatures ARE recognizable — the question is whether they
should be split into a parent+child pair or preserved as one
hierarchical treatment.

**Severity**: medium.  Requires a different sub-fix from §6:

  * **Splitting at the `sp. nov.` boundary** gives us one
    genus treatment (correct anatomical scope) and one species
    treatment (also correct anatomical scope), but loses the
    parent-child taxonomic relationship — the species record
    no longer "knows" it belongs to the genus.
  * **Keeping as one treatment** makes downstream tools that
    assume one-specimen-per-treatment misinterpret the doc.
    Also inflates annotation counts (double-counting shared
    features).
  * A **hierarchical schema** (species treatment carries a
    ``parent_genus_treatment_id``, or the two are stored as
    linked docs) captures both correctly, but is a larger
    change to the pipeline.

**Search-UX revision (operator, 2026-07-02)**: on
re-inspection of taxon_01a01c54, the operator questions
whether "one treatment per taxon" is even the right target
for the search product.  A `gen. nov.` and its type
species genuinely belong together in the reader's model —
a search hit that surfaces the genus AND its type species
in the same result panel is arguably more useful than two
separate hits users have to correlate manually.  This
shifts the design question:

  * If the search product treats the gen. nov. + type sp.
    pair as ONE natural search unit, then the "single
    treatment with both" outcome (option 2 above) becomes
    the intended behaviour rather than an assembly failure.
    Downstream tools that assume one-specimen-per-treatment
    are the thing that's wrong.
  * The hierarchical schema (option 3) is still cleaner
    for structured queries ("all species in genus X") but
    the flat "genus + type species as one search unit"
    representation may be what's needed for the
    reader-facing search product.

Not a decision yet — recorded because it changes the
framing of the §11 "fix" question.  Might make sense to
prototype a search UI over the current corpus (which
already contains these blended treatments) before
committing to a schema change.

**Latin-only observation (operator, 2026-07-02)**: for
taxon_01a01c54 specifically, only the LATIN description
was captured; the matching English descriptions were not
picked up.  The Latin is the ICBN-required diagnosis; the
English is the modern-reader description.  Missing the
English is a separate assembly-drops-content issue —
argues the extractor treated the English description
paragraphs as different section labels (or dropped them
entirely at a language boundary).  Another §12
label-aware-assembly instance: preserving segment labels
would make the "capture BOTH the Latin diagnosis AND the
English description under this treatment" rule
tractable.

**Design-riff continuation (operator + Claude, 2026-07-02)**
on how the search product should represent the gen.-nov.
+ type-sp. pair.  Three complementary levers to consider:

  1. **SBERT-neighborhood aggregation** — post-process
     search results to cluster genus + its type species
     when they co-appear in the top-K.  No schema change.
     Testable cheaply on taxon_01a01c54: pull the nearest
     neighbors of its vector and see whether other
     Pseudotrichia species-level content clusters tightly
     nearby.
  2. **Blended-treatments-as-search-units** — if we do
     NOT fix cases like taxon_01a01c54 (per the search-UX
     revision above), option 1 works essentially for free
     because the blended vector implicitly carries signal
     from both the genus and the type species.  "Feature,
     not bug" reframing.
  3. **Taxonomic-rank tagging** — tag each treatment with
     the rank it primarily describes (species / genus /
     higher).  Keep ONE SBERT index, facet the search UI
     by rank.  Cheaper than separate rank-indexed spaces
     and works around the data-sparsity concern (higher
     ranks are thinly represented — mostly just genus
     circumscriptions in gen. nov. treatments, and even
     less family-level content).  Users get the "your
     collection could be one of these genera and one of
     these species" affordance without cold-start
     problems.

Three levers compose cleanly.  Blended treatments get
tagged as both "genus-scope" AND "species-scope"; SBERT
does within-rank ranking; the UI clusters or facets as
appropriate.

**Use-case caveat**: "your collection could be one of
these genera / species" implies an IDENTIFICATION
workflow (user has a specimen, wants to know what it
is).  That's distinct from literature-discovery ("show
me everything on genus X").  The two might want
different result-panel layouts; worth distinguishing
which use case each design idea optimizes for before
committing.

Nothing decided; recorded to keep the design thread
alive.  A cheap prototype: run the current search UI
against a query that should match taxon_01a01c54 (e.g.,
`Pseudotrichia` or a phrase from its Latin diagnosis)
and inspect the top-K neighborhood by hand.

**Reviewer action** (Phase 1 hand-review): both sets of
anatomical descriptions are real and valid.  Two acceptable
approaches, at reviewer's judgment:

  * Annotate the FIRST taxon only (the genus in this case) —
    matches §6's "first-species" rule and gives a clean single-
    taxon golden record.
  * Annotate BOTH sets and flag the treatment via a brat
    AnnotatorNote — captures more training signal per API
    dollar, at the cost of one blended golden record until the
    grouper fix arrives.

Either way, add a brat AnnotatorNote flagging the treatment
for the section-classifier re-review queue.  The Trello / fix
work should treat this as a distinct sub-case from §6.

### 12. Segment classification as an aid to Treatment assembly (design note)

**Motivation**: the accumulating §6, §10, §11 evidence points
at a shared root cause.  The treatment-grouper collapses per-
segment section labels (Nomenclature, Description, Diagnosis,
Etymology, Observations, Discussion) into flat treatment
fields WITHOUT preserving those labels through assembly.  Once
the labels are dropped, downstream tooling (memo detectors,
bootstrap annotator, reviewer) has to rediscover section
boundaries from prose — the signal was already computed one
stage upstream and thrown away.

**Line-classifier granularity limitation (2026-07-07,
taxon_9b787247)**: the v4 layout CRF classifies at LINE
granularity — one label per physical line.  In older
taxonomic texts (pre-1950 typography), authors commonly
compressed multiple sections onto single lines to save
space, using em-dashes or explicit markers to separate
them.  Example from taxon_9b787247 (1920 German paper):
`tantum immaturi visi. — Einzige Art: R. Symphoricarpi
Syd. (= Zaszobotrys Symphoricarpi Syd.).` — one line
containing description-tail + em-dash + type-designation
in German + Latin taxonomic reference.  The line-level
CRF must pick one label, dropping the others.
Sub-line granularity (segment classifier) is required
to handle this content class without loss.  Reinforces
the M3 flagship: segment-level classification isn't just
for cross-species boundaries; it also unlocks intra-line
section detection for dense typographic conventions.

**Observed failures that would benefit from label-aware
assembly**:

  * **`taxon_592128a8`** — multiple constituent species had
    their Nomenclature blocks broken across the Description
    field.  Preserving segment labels would have kept per-
    species Nomenclature/Description/Diagnosis grouped
    correctly instead of interleaved.
  * **`taxon_acd88732` and `taxon_592128a8` (§10 variant)** —
    Description opens with a Nomenclature tail.  If the
    segment classifier had labelled that leading fragment
    `Nomenclature`, assembly wouldn't have appended it to
    Description.
  * **`taxon_2a9d07e6`** — two `Diagnosis:` headers survived
    into the Description field because they are the literal
    Diagnosis-label boundary marker.  Label-aware assembly
    could treat a second `Diagnosis:` label as a species-
    boundary signal directly, without needing the
    Diagnosis-count post-hoc detector.
  * **`taxon_01a01c54` (`gen. nov.` + `sp. nov.`)** — both the
    genus and species had proper Nomenclature labels but
    assembly still merged them.  Labels alone aren't
    sufficient here; a hierarchical assembly rule ("second
    Nomenclature under a `gen. nov.` genus creates a linked
    species treatment") is what's needed.
  * **`taxon_5b0a8ce7`** — key-body couplets landed in
    Description because the classifier lacked a `Key` label.
    Adding one gives assembly a way to route them elsewhere
    without reaching for regex.
  * **`taxon_572d470e`** — most of a `Diagnosis` block leaked
    into the Description field.  Same shape as
    taxon_2a9d07e6's Diagnosis-header duplication, but this
    time it's not just the header — it's the entire block
    that got mislabelled/relocated.  A label-aware assembler
    would route Diagnosis-tagged segments to `diagnosis` and
    leave Description clean.
  * **`taxon_95dbdfb9`** — Holotype designation landed at
    the end of Description instead of under a `Type` /
    `Holotype` block.  Same class as
    taxon_f00f8353's Materials_examined leak and
    taxon_572d470e's Diagnosis leak — Type-family metadata
    ending up in prose fields.  Assembly-aware routing
    (`Type` label → `type_designation` field, not
    `description`) would fix it.
  * **`taxon_d41b87e4`** — **DISTRIBUTION line in
    Description**.  New leak target: the
    `treatments_prose` schema has a `distribution`
    field, but the layout CRF is labeling
    DISTRIBUTION lines as `Description` content in
    at least this case.  Same class as
    Materials-examined and Type-designation leaks,
    on a different destination field.  Species 2 of
    the 3-species merge in this treatment had a
    DISTRIBUTION line embedded in Description.
    Assembly-aware routing (`DISTRIBUTION` label →
    `distribution` field) would fix.
  * **`taxon_8d70e41a`** — **whole-treatment content
    loss** with only a truncated Diagnosis tail
    surviving.  desc_length = 0 (empty Description),
    diag_length = 336 (short Diagnosis, head-clipped
    but ending correctly), synthetic_nomenclature =
    True (no Nomenclature), 2 annotations total.  Third
    observed instance of Diagnosis-field head-clipping
    (after taxon_e44e35bc's both-ends clip and
    taxon_9ecad903's head-clip) — enough recurrences
    to promote this from "worth watching" to an
    implementable detector: apply the
    `desc_starts_mid_sentence` rule to the `diagnosis`
    field as well, gated on `diag_length > 0`.  What
    makes taxon_8d70e41a a distinct sub-shape is the
    accompanying **complete loss of Description
    content** — the Description field is empty, not
    just mis-filled.  Distinct from §14's
    shared-diagnosis orphan (which has a COMPLETE
    diagnosis serving multiple sibling species-treatments
    intentionally); here the missing content is a
    failure, not a source-intended fan-out.  Detection
    for this specific sub-shape: `desc_length == 0 AND
    diag_length < 500 AND diag_head_clipped` — the
    combined signals distinguish "diagnosis-only
    orphan" (§14, likely complete) from "everything
    lost except a diagnosis fragment" (this case).
  * **`taxon_c9181340`** — Materials_examined citation
    appended to Description (`Specimen examined: USA,
    Florida, on seed of Podocarpus maki … ex-type
    culture ` with trailing-space truncation).
    `materials_examined` field itself is EMPTY.  Same
    class as taxon_f00f8353's Materials_examined leak,
    from batch-2.  §10:tail_clip fires on the trailing
    space.  Cultural characteristics upstream correctly
    stays in Description (per 2026-07-02 clarification).
  * **`taxon_fd4323fb`** — Diagnosis field ends with a
    leaked Nomenclature for the NEXT species:
    `Sorosporium chamaeraphis Syd. apud Syd. & Petr.,
    Ann. ` (trailing space, truncated).  The Diagnosis
    itself is a legitimate Differential Diagnosis (this
    species vs Farysia olivacea).  Cross-field failure:
    description and diagnosis are 81 source lines apart
    (lines 8004-8006 vs 8087-8091) with the intervening
    content dropped or classified elsewhere.  Different
    from the description-only Nomenclature leaks — this
    is a Nomenclature leaking into the DIAGNOSIS
    field's tail.  Suggests the section CRF loses the
    species boundary somewhere in the 80-line gap.
  * **`taxon_adcb2fcc`** — description assembled from TWO
    NON-CONTIGUOUS source spans (description_spans: lines
    11262-11266 + lines 11282-11283, 15-line gap between
    them).  Per operator PDF cross-check: fragments come
    from species 1's Notes section (should have been
    Diagnosis-classified) + species 2's diagnostic content
    (should have been a separate treatment).  Three
    logical fragments assembled into one Description
    field.  §10:mid_sentence and §10:tail_clip fire
    correctly; the deeper §12 assembly failure isn't
    directly detectable from the flat prose.
    **Detection idea recorded**: `n_description_span_gaps`
    — count line-number gaps between consecutive
    description_spans that exceed a threshold (e.g., >5
    lines).  Would fire on this treatment (1 gap of 15
    lines).  Requires plumbing span metadata through to
    treatment_signals — the span info exists on the
    treatment_prose doc but isn't currently consumed by
    detectors.  Cheap once wired.
  * **`taxon_ea7b0ed7`** — a figure caption landed
    embedded mid-Description instead of the doc's
    `figure_captions` field.
  * **`taxon_d552598708...`** — multi-caption
    variant of the taxon_ea7b0ed7 pattern.
    Complete single-species description of a smut
    fungus (`Sori (Fig. 2) in ovaries …`; intra-
    description figure references are normal),
    then **TWO figure captions appended** at the
    tail, with the **second caption head-clipped**.
    Two new sub-observations:
      (1) The leak isn't always a single caption —
          the treatment-grouper can pick up a RUN of
          consecutive captions.  Detection should
          count captions, not just detect one.
      (2) The captions themselves can carry their
          own clipping — the second caption is
          head-truncated where the run began.
          Combined with the boundary between species
          content and the caption run, this is
          another instance of "extractor lost the
          boundary and clipped what it did capture."
    All triage detectors correctly silent — Claude
    produced 5 annotations, presumably skipping the
    caption run per its §13-analogous behaviour.  The treatment is otherwise
    clean (both Description and Diagnosis look good).
    **Claude implicitly skipped the embedded caption** —
    10 annotations came out, none of them touching the
    caption text.  Same pattern as §13's observation
    that Claude implicitly skips Diagnosis-labelled
    content: the model recognizes section content that
    doesn't fit the anatomical-feature schema and
    silently drops it, even when the block is
    mis-routed into Description.  Argues Claude's
    label-recognition capability is stronger than the
    assembler's — label-aware assembly (§12) could use
    Claude's implicit skip behaviour as a training
    signal for what content-vs-anatomy segmentation
    should look like.
  * **`taxon_d2d26d25`** — a partial Diagnosis block sits
    at the end of Description, missing leading AND
    trailing content (fragmentary at both ends), while the
    Diagnosis field itself is EMPTY.  The diagnosis
    fragment is **English comparative** style
    (differentiating from other species by feature) — NOT
    Latin.  Compound failure:
    (1) routing — Diagnosis label mis-routed to
    Description; (2) clipping — the diagnosis fragment has
    been truncated at both ends, so text that WAS in the
    original diagnosis is nowhere in the doc.  Distinct
    from taxon_572d470e's whole-Diagnosis leak (which
    preserved the block, just put it in the wrong field);
    here content is also lost.  Label-aware assembly would
    fix (1).  For (2), a detector along the lines of
    "diag field empty AND description tail mentions
    multiple binomials or uses Differential-Diagnosis
    language (`differs from …`, `similar to …`)" could
    surface these — the gnfinder / comparative-language
    signals from §6 idea #2 applied only to the last N
    chars of Description.  The fragment is a
    **Differential Diagnosis** (§13 term), not Latin —
    Latin-morphology heuristics do NOT apply.  Currently
    no detector fires (merge_metric = 3; diagnosis field
    empty; description doesn't start mid-sentence).
    Tracked as **D3** in the Detector backlog; this taxon
    is its gating case and still needs a fixture entry.

**Proposal (not a plan yet)**: pass segment-level
`(section_label, text)` tuples through to the assembly stage
instead of flat field dicts.  Assembly rules become label-
aware:

  * One `Nomenclature` per treatment; a second `Nomenclature`
    (except under the §11 `gen. nov.` hierarchical pattern)
    marks a species boundary.
  * The `description` field contains only segments labelled
    `Description`; Nomenclature-tail fragments and Diagnosis
    blocks don't leak.
  * `Diagnosis:` / `Observations:` / `Discussion:` header
    multiplicities become species-count signal by
    construction — no separate detector needed.
  * A `Key` label routes numbered-couplet segments out of
    Description.

**Cost**: a schema change through `treatments_prose`
(currently flat).  The v4 layout CRF already emits per-line
section labels; the plumbing to preserve them exists but is
discarded at assembly.

**Testability**: incrementally verifiable by running a
label-aware assembler alongside the existing extractor and
diffing.  Every §6/§10/§11 case above is a golden regression
target with a known expected split.

**Timing**: tracked as **D5** in the Detector backlog.
Overlaps with the pipeline restructure in
`~/.claude/plans/cozy-forging-locket.md` (per-family Python
modules).  Likely a Phase 3+ candidate after v4 lands —
useful to record now so §6 fix work explicitly weighs
"tighten the merge detector" vs "fix assembly to not need one."

### 12.1 The vocabulary is absorbing *slots* as if they were features

Recorded 2026-08-25, from the one new label round 4 produced.

Reviewing `taxon_fa7f4de6` (§0.5) the operator added
**`Squamules`** for the sentence

> *The majority of squamules are sterile.*

`Squamules` was absent from **both** `features_candidate`
(322 labels) and `features_hand` (314) — genuinely new
vocabulary at treatment 106 of the review, which is itself
worth knowing for the Heaps' curve.

**But it is not a feature.**  The squamules were already
described, four sentences earlier, under `Thallus`:
*"Thallus of convex dispersed squamules, 0.3–1 mm wide…"*.
The new sentence adds no anatomy; it states a **property**
— fertility — of a structure already on the record.  Naming
the label after the sentence's subject noun creates a second
feature for one organ.

Under `docs/structured-form-schema.md` the sentence wants to
land as a slot:

```json
{"feature": "Thallus",
 "structure": ["convex dispersed squamules"],
 "fertility": ["the majority of squamules are sterile"]}
```

**Why this is a class and not a one-off.**  The annotator is
asked for a `feature_label` and nothing else, so *any*
observation that is not itself an organ has to be
expressed by inventing an organ-shaped label.  That is the
same pressure that produced `Asci in culture MEA` and
`Chemical Reaction` — a qualifier with nowhere structural
to go, welded onto the label string.  §12's label-aware
assembly and the schema doc's open "slot vocabulary"
question are the same problem seen from two ends.

**Scale, measured.**  The exact fertility construction is
rare — 13 of 42 096 descriptions (0.03 %) — so `Squamules`
will stay a near-singleton and *will* be counted among the
54 % singleton labels the Heaps analysis reports.  A
meaningful fraction of that singleton tail is likely slots
in feature clothing rather than genuine long-tail anatomy.
**That is testable**: partition the 322 labels into those
that name an organ and those that name a property, before
concluding anything about vocabulary saturation.  Do it as
part of the schema induction in T6, not after.

**Do not rename `Squamules` now.**  It is evidence about the
current prompt and the baseline depends on it, exactly as
with the six deferred `Spores`→`Ascospores` cases.

### 12.2 The layout labels, measured (T3b)

Until 2026-08-25 every claim in this memo about layout labels rested on
individual treatments. This is the corpus view: **a random sample of
500 of the 20 928 documents in `ann_combined`, 57 142 blocks.** Rates
are ±~1 pp on the common labels; the extrapolations are point estimates
on a 2.4 % sample and should be read as magnitudes, not counts.

#### The inventory nobody had

| label | % of blocks | % of documents |
|---|---:|---:|
| **`Misc-exposition`** | **35.4 %** | **85.2 %** |
| `Page-header` | 14.2 % | 61.4 % |
| `Table` | 11.9 % | 41.0 % |
| `Key` | 5.7 % | 50.8 % |
| `Description` | 5.5 % | 50.6 % |
| `Figure-caption` | 5.1 % | 57.8 % |
| `Bibliography` | 4.5 % | 56.6 % |
| `Nomenclature` | 3.3 % | **38.8 %** |
| `Notes` | 3.1 % | 46.2 % |
| `Materials-examined` | 2.6 % | 37.4 % |
| `Biology` | 2.0 % | 33.8 % |
| `Diagnosis` | 1.6 % | 34.2 % |
| `Phylogeny` | 1.5 % | 24.2 % |
| `Materials-and-methods` | 1.3 % | 40.0 % |
| `Type-designation` | 1.0 % | 23.8 % |
| `Etymology` | 0.9 % | 25.0 % |
| `ToC-entry` | 0.2 % | 6.2 % |
| `Index` | 0.2 % | 8.2 % |

18 labels, and **one of them is a third of the corpus.**
`Misc-exposition` was described above as "the layout pass's catch-all,
which makes it the first place to look." It is more than that: at
**35.4 % of all blocks and present in 85.2 % of documents** it is the
*default*, not a residue. Every D12 case where it swallowed content is
drawn from a pool that large.

Second observation, easy to miss: **`Nomenclature` appears in only
38.8 % of documents while `Description` appears in 50.6 %.** More
documents have descriptions than have names.

#### Descriptions that no name precedes — two different faults

Splitting every `Description` block by whether a `Nomenclature` block
came before it in the same document:

| situation | share | ≈ corpus |
|---|---:|---:|
| after the first `Nomenclature` — normal | 78.2 % | — |
| **before** it, in a document that *has* one | 13.5 % | ~18 200 |
| **orphaned** — document has *no* `Nomenclature` block at all | 8.3 % | ~11 100 |

**14.4 % of documents contain `Description` blocks and zero
`Nomenclature` blocks.** That is a different failure from
mis-ordering: the classifier never labelled a name anywhere in them.
Both routes feed `synthetic_nomenclature`, which runs at 39.6 %
corpus-wide — and this is where much of it comes from.

**Two confounds were tested rather than assumed.** Front matter would
put these descriptions at the head of the document; the median sits at
**0.36** of the way through, so that is not the explanation. And the
orphaned/mis-ordered split above exists precisely because the pooled
figure conflated "the name comes later" with "there is no name."

Still uncontrolled: a genus description legitimately precedes the first
*species* heading, so some of the 13.5 % is correct structure. Treat
~18 200 as an upper bound.

#### D12 and D18, as rates

* **D12** — a non-content block opening on a nomenclatural act and
  continuing into prose: **38 in sample → ~1 590 corpus-wide.** An
  independent 200-document sample gave ~1 360, so the estimate is
  stable. `Figure-caption` keeps recurring, as with `taxon_ecb0124d`.
* **D18** — a `Description` block opening on a nomenclatural heading:
  **13 → ~544.** The examples are a distinct genre — numbered monograph
  entries like `2. Tulasnella pruinosa Bourd. & Galz. Bull. Soc. Myc.
  Fr. 39: 264. 1924.` — which suggests D18 is concentrated in floras
  rather than spread evenly.

#### A far better D12 rule: the block names its own section

Found 2026-08-26 during T5 review. The operator, reading round 5 in
brat with `bin/treatment_dossier` open beside it, noticed that
`taxon_0b9a9bfe`'s etymology had vanished. The dossier showed why in
one line:

```
gap nomenclature@625 -> materials_examined@629
    [Misc-exposition] Etymology – ramiconidiophorus (Lat.) refers to
                      multiple-branched primary conidiophores.
```

A complete etymology clause **carrying its own `Etymology –`
header**, labelled `Misc-exposition`, so the treatment has no
`etymology` field at all.

**That header generalises, and it is an order of magnitude better than
the nomenclatural-act rule above.** Measured over 400 documents:
**1 779 blocks open on a named section header** (`Etymology`, `Notes`,
`Description`, `Diagnosis`, `Materials examined`, `Type`, `Habitat`,
`Distribution`, `Ecology`, `Remarks`, `Comments`, terminated by dash,
colon or period). **294 of them — 16.5 % — carry a non-content label**,
extrapolating to roughly **15 400 corpus-wide**.

| header | landed in | n |
|---|---|---:|
| `Notes` | `Misc-exposition` | 131 |
| `Etymology` | `Misc-exposition` | 44 |
| `Note` | `Misc-exposition` | 20 |
| `Remarks` | `Misc-exposition` | 16 |
| `Comments` | `Misc-exposition` | 12 |
| `Type` / `Specimens examined` | `Misc-exposition` | 18 |
| **`Diagnosis`** | `Misc-exposition` | 7 |
| **`Description`** | `Misc-exposition` | 4 |

The last two rows are whole descriptions and diagnoses lost —
`Description: Basidiomata annual, pileate, sessile, gregarious…` sitting
in `Misc-exposition`.

**Why this beats the act-based rule.** That one needed three
conditions to reach ~1 590 blocks at roughly three-quarters precision.
This needs one, and the block *states what it is*: a block whose first
line reads `Etymology –` is an etymology, whatever the layout pass
called it. There is no inference step to get wrong.

It also reinforces §12.2's ordering: `Misc-exposition` is 35.4 % of all
blocks, and this is 15 400 pieces of labelled content sitting inside
it. Splitting that label is the highest-leverage fix measured.

#### Registry identifiers are mislabelled far more often than prose

Found the same day, from `taxon_0ccf38da`. The operator reported three
defects and the dossier showed a fourth:

```
gap nomenclature@1113 -> description@1117
    [Misc-exposition] Index Fungorum number: IF557396;
                      Facesoffungi number: FoF14622
gap materials_examined@1119 -> notes@1123
    [Misc-exposition] GenBank numbers – ITS = PQ800240,
                      SSU = PV072619, tef1-α = PX739628.
```

The nomenclature is therefore truncated at `Fig. 44`, and the GenBank
accessions are lost out of `materials_examined`.

**Identifier lines are the worst-labelled content in the corpus.**
Over the same 400 documents, blocks opening on a registry or accession
name — `MycoBank`, `Index Fungorum`, `Facesoffungi`, `GenBank`,
`Fungal Names` — number 66, and **49 of them (74.2 %) carry a
non-content label**, roughly **2 563 corpus-wide**. `MycoBank` alone is
33 of the 49.

Compare the rates:

| block opens on | mislabelled | ≈ corpus |
|---|---:|---:|
| a registry identifier | **74.2 %** | ~2 600 |
| a named section header | 16.5 % | ~15 400 |

**The mechanism is legible.** `MycoBank MB 847723.` is four tokens with
no sentence structure, no anatomical vocabulary and no verb — nothing
for a layout classifier trained on prose to grip. It falls to the
catch-all almost every time. Section headers survive better precisely
because prose follows them.

That makes it the highest-precision rule available: **a block naming a
registry is nomenclature material** (MycoBank, Index Fungorum,
Facesoffungi, Fungal Names) **or specimen material** (GenBank). There
is no interpretation step, and at 74.2 % base rate a detector barely
needs to be careful.

#### `Key` is a second catch-all, and only half of it is keys

`taxon_134c7e0e` (*Thielavia* Zopf) lost its genus description tail to
a `Key` block. The sentence is cut in half:

> diagnosis ends `…The asexual morph has conidia`
> `key` field holds `496 / are hyaline or brightly coloured and
> produced as simple phialoconidia, aleurioconidia, or arthroconidia.`

**That block contains zero numbered couplets**, which is the whole
signature: a key has `1.` / `2.` markers, and a block labelled `Key`
without any is not a key.

Measuring it needed two corrections, both worth recording because the
first number looked publishable:

* **11 % of `Key` blocks are OCR-destroyed** — runs of U+FFFD where
  couplet detection fails because the text is gone, not because it is
  prose. Judging those is meaningless, so they are excluded.
* The remainder is **not one class**. Decomposed over 400 documents:

| legible `Key` block, no couplets | share | ≈ corpus |
|---|---:|---:|
| other (section headers, fragments) | 69.0 % | ~21 200 |
| **prose continuation, opens mid-sentence** | **22.0 %** | **~6 700** |
| registry identifier + name | 6.5 % | ~2 000 |
| nomenclatural act | 2.6 % | ~800 |

**Only about half of legible `Key` blocks are actually keys.** So
`Key` is a *second* catch-all behind `Misc-exposition` — smaller at
5.7 % of blocks against 35.4 %, but with a worse hit rate for
containing something else.

`taxon_134c7e0e`'s defect is the 22 % row, ~6 700 treatments' worth of
description tail sitting in `Key`. The signature is precise — `Key`
label, zero couplets, opens lower-case — and needs only the `.ann`.

**A correction to the identifier rule above.** Those `registry
identifier + name` rows read `MB 859229 Paracylindrosporium Scrace &
Crous, gen. nov.` — a bare `MB` rather than the word `MycoBank`, which
the 74.2 % measurement did not match. That figure is therefore a
**lower bound**; the true identifier-mislabelling rate is higher, and
the pattern reaches `Key` as well as `Misc-exposition`.

#### Proposed: a deterministic repair pass, not a better classifier

Operator, 2026-08-26 on `taxon_47c3b37d`: *"The index numbers are
pulled off the end of the taxonomic citation by a Misc-exposition —
this seems like something we should be doing with plain regex
patterns — maybe as an extension to gnfinder."*

**Right, and it splits into two mechanisms with different tools.**

**1. Identifier reattachment — pure regex, no name service needed.**
`Index Fungorum number: IF556440; Facesoffungi number: FoF 05781` is
rigidly structured. At **74.2 % mislabelled** (~2 600 blocks) this is
the highest-yield deterministic repair available, and it needs nothing
but a pattern and the preceding block's identity.

**2. Authorship reattachment — this one genuinely wants gnparser.**
The same treatment severed a lone `Blume.` into its own
`Misc-exposition`, taking the plant authority off the etymology's host
name. A regex cannot safely tell an author abbreviation from a
sentence-ending word; **gnparser can**, and the local service already
runs at `localhost:9081`. Tested 2026-08-26:

| input | parsed | canonical | authorship |
|---|---|---|---|
| `Rhizophora apiculate` (etymology tail) | ✓ q1 | *Rhizophora apiculate* | **none** |
| `Rhizophora apiculate Blume.` (rejoined) | ✓ q1 | *Rhizophora apiculate* | **`Blume.`** |

**That difference is the detector.** A block ending in a bare binomial
followed by a block that parses as an authorship is a severed name.
Neither parse fails — both are quality 1 — so the signal is the
*appearance of an authorship on rejoin*, not a validity change.

**Why this is the right shape of fix.** It is the same argument
`docs/structured-form-schema.md` §4 makes about measurements: *do not
spend model capability on something a deterministic tool does better.*
The layout classifier will keep mislabelling four-token identifier
lines because they carry no prose signal (§12.2); a repair pass
downstream does not care.

**Scope honestly.** gnfinder proper finds *names in free text* and is
the wrong tool for identifiers — that half is regex. The frequency of
the severed-authorship case is **not yet measured**; only the
mechanism is validated. And a repair pass mutates extraction output,
so it needs the same fixture-gated treatment as any detector here.

#### What actually decides whether an identifier survives: the `nov.` act

`taxon_4d851975` (*Botryotrichum geniculatum*, *Studies in Mycology*
101) is the counterexample that explains the rule. The operator:
*"nearly perfect… the nomenclature picked up the MycoBank reference and
the defining figure number."* Its nomenclature field reads

> `Botryotrichum geniculatum X.Wei Wang, P.J. Han & F.Y. Bai, sp.`
> `nov. MycoBank MB 840127. Fig. 20.`

— identifier and figure both retained, against the ~74 % that are lost.

**Two hypotheses, both refuted by measurement.**

*Segmentation.* The obvious reading is that an identifier survives when
it shares a block with the name and is lost when it stands alone.
**Backwards**: over 300 documents, a block that is *essentially just*
the identifier is labelled `Nomenclature` **53.0 %** of the time, while
one where the identifier is *embedded in longer text* manages only
**20.5 %**.

*The name.* Next guess: the classifier keys on the binomial. **Barely
matters** — see the table.

**What does decide it is the nomenclatural act:**

| block contains | n | → `Nomenclature` |
|---|---:|---:|
| binomial + `sp./gen. nov.` | 430 | **47.2 %** |
| binomial, no act | 218 | **15.1 %** |
| no binomial, act present | 24 | 50.0 % |
| neither | 105 | 19.0 % |

**The act roughly triples the odds; the binomial moves them by three
points.** And it explains both observed cases exactly:

* `taxon_4d851975` — `… sp. nov. MycoBank MB 840127.` **has the act**,
  and survives.
* `taxon_47c3b37d` — `Index Fungorum number: IF556440; Facesoffungi
  number: FoF 05781` **has no act**, and goes to `Misc-exposition`.

**This sharpens the repair pass proposed above.** The rule is not
"reattach identifier blocks" in general but the narrower, better-founded
**"reattach an act-less identifier block to the preceding act-bearing
block"** — which is where it came from on the page.

**Even with the act, half are still lost**, and `Key` takes 101 of the
430 — the second catch-all again.

#### Self-declared labels: the cue is honored 65 % of the time

`taxon_53dd1485` (*Xylodon daweishanensis*), operator: *"is perfect.  No
defects, Figure-caption does its job."*  The second flawless treatment of
round 5, after `taxon_3b7a80bc` — and **the two share one structural
feature**: every block opens with its own field name.

```
[Nomenclature      ] Xylodon daweishanensis C.L. Zhao sp. nov. Figs 8, 9
[Type-designation  ] Type material. Holotype. China. Yunnan Province…
[Etymology         ] Etymology. Daweishanensis (Lat.): referring to…
[Description       ] Description. Basidiomata annual, resupinate…
[Figure-caption    ] Figure 8. Basidiomata of Xylodon daweishanensis…
```

The document labels itself and the model agreed.  **That is the same cue
carried by the self-labelling blocks recorded above as mislabelled** — so
the interesting question is not "why did this one work" but "how often is
the cue honored at all".

**Measured over 300 documents:** 2 474 blocks open with an explicit
field-name cue; **1 619 (65 %) get the matching label**, leaving
**~59 600 cued-but-ignored blocks corpus-wide** — four times the
self-labelling estimate recorded earlier, which counted only absorption
into non-content labels.

| cue | n | honored | |
|---|---:|---:|---|
| `Etymology` | 401 | **93 %** | unique word, one meaning |
| `Description` | 66 | 89 % | |
| `Materials-examined` | 414 | 79 % | |
| `Figure-caption` | 680 | 66 % | regex also catches cross-references |
| `Type-designation` | 378 | **48 %** | |
| `Diagnosis` | 58 | **45 %** | |
| `Notes` | 477 | **43 %** | shared/structural cue word |

**Correction (2026-08-27): asymmetry alone does not identify a defect.**
The first reading of this table called every asymmetric pair a "finer
distinction collapsing into a coarser one".  Operator: *"Notes ->
Diagnosis and Notes -> Phylogeny are both semantically valid
restrictions of a section labeled Notes.  There's some subtlety here."*

Correct.  `Notes` is a **superordinate** — a commentary section that
legitimately *contains* diagnostic comparison, phylogenetic discussion
and ecological remark.  Labelling a `Notes.`-cued block `Diagnosis` is a
**refinement**, not an error.  `Type-designation` ->
`Materials-examined` is the opposite: a type citation losing its type
status.  **Both are asymmetric with near-zero reverse.**  Asymmetry says
only that the direction is systematic; **what makes it a defect is the
direction on the subsumption order**, which a flat confusion matrix
cannot see.

Assumed order (parent ⊐ child, child being the more specific):

```
Misc-exposition ⊐ everything            (the universal catch-all)
Notes           ⊐ Diagnosis, Phylogeny, Biology
Materials-examined ⊐ Type-designation
Description     ⊐ Diagnosis             (restored — see §12.3.5)
```

**`Description` ⊐ `Diagnosis` was removed and then restored** — see
§12.3.5.  Removing it was a workaround for a rule not yet found; §12.3.3's
referent test does the job properly, so the edge is back and the lattice
has no special cases.  The original reasoning is kept here because the
separability measurement it rests on is still what makes coarsening along
this edge inexcusable:  Operator: *"I agree that Diagnosis can be considered a
special kind of Description, but it seems unlikely to be in a block
labeled 'Description', and it has comparative language, so should be
semantically differentiable."*  Measured, and correct: comparative
language appears in **55 % of `Diagnosis` blocks against 8 % of
`Description` blocks**, a 7:1 separation.  They are lexically
distinguishable siblings, so a `Diagnosis` -> `Description` miss is a
**genuine swap**, not benign coarsening.  This moves 15 blocks from
coarsening to swap and leaves the acceptable rate at 70 %, but it
changes the fix: the signal exists and is unused.

**Re-scored against that order:**

| cue | n | honored | refine | coarsen | absorbed | swap | **acceptable** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Figure-caption` | 680 | 446 | 0 | 0 | 103 | 131 | 66 % |
| `Notes` | 477 | 204 | **109** | 0 | 134 | 30 | **43 → 66 %** |
| `Materials-examined` | 414 | 327 | 12 | 0 | 68 | 7 | 82 % |
| `Etymology` | 401 | 374 | 0 | 0 | 25 | 2 | 93 % |
| `Type-designation` | 378 | 183 | 0 | **124** | 39 | 32 | **48 %** |
| `Description` | 66 | 59 | 0 | 0 | 5 | 2 | 89 % |
| `Diagnosis` | 58 | 26 | 0 | 15 | 13 | 4 | 45 % |
| **ALL** | **2 474** | 1 619 | 121 | 139 | 387 | 208 | **65 → 70 %** |

**~51 200 true defects corpus-wide**, revised down from ~59 600.

**What this moves.** `Notes` is *not* a problem label — 40 % of its
misses are valid refinements.  **`Type-designation` is**: every one of
its 195 misses is a genuine defect, and 124 are coarsening into
`Materials-examined`.  The first reading blamed the wrong label.

**`Figure-caption` is a third mechanism again.**  It has **no lattice
relatives** — 0 refinements, 0 coarsenings — so every miss is a defect,
and its 131 swaps scatter across seven unrelated labels (`Description`
29, `Nomenclature` 17, `Key` 17, `Materials-and-methods` 16, `Notes` 12,
`Phylogeny` 9, `Materials-examined` 8).  That scatter is the signature
of a **boundary/typographic** failure, not a semantic one — a caption is
defined by where it sits on the page, not by what it says.  Part of it
is also detector noise: the `Fig\.?\s*\d` cue matches in-text
cross-references, which land in whatever section cites them.

**The misses split into three mechanisms, and only one is confusion.**

*Absorption* — 387 of 855 misses (45 %) go to `Misc-exposition`.  A block
that literally announces its own field is swept into the catch-all.  This
is §12.2's absorption class, now with a denominator.

*Coarsening* — 139 misses move **up** the order, losing a distinction
the cue asserted.  `Type-designation` -> `Materials-examined` (124) is
almost all of it.

*Sibling swap* — 208 misses land on an unrelated label.  This is the
only genuine confusion, and `Figure-caption` supplies 131 of it.

The original directional table, retained because the counts are still
the evidence — but read now against the order above, where the `Notes`
rows are refinements and only the first is a defect:

| cue says | model said | n | reverse |
|---|---|---:|---:|
| `Type-designation` | `Materials-examined` | 124 | **12** |
| `Notes` | `Diagnosis` | 65 | 0 |
| `Notes` | `Phylogeny` | 39 | 0 |
| `Diagnosis` | `Description` | 15 | 0 |

10:1 is a **finer distinction collapsing into a coarser one**.
`Type-designation` → `Materials-examined` is a type losing its type
status — the two are both specimen citations and only one is
distinguished.  `Notes` → `Diagnosis` → `Description` is one gradient
resolved inconsistently, and it is exactly the call the operator made by
hand on `taxon_47c3b37d`: *"I would have called the notes section a
diagnosis and not just notes."*  **When the human and the model disagree
on that pair, the human is also choosing, not correcting.**

**This answers the deferred pairwise-differentiation question** (asked
2026-08-26, *"is this something we could get out of the model?"*): yes,
and it does not need the model's posteriors or any hand annotation —
the corpus supplies the labels.  Recorded as
`docs/rl-framework-components.md` §1.1.1.

**Caveat on the sample.** The cue is dense in MycoKeys-style journals and
absent from older or OCR-damaged material, so these rates are an **upper
bound**.  Both perfect treatments are from that well-formatted stratum.

### 12.3.25 Sizing the LOTE problem — severe, but 2 % of documents

`taxon_a58b3756`.  Operator: *"starts with the introduction to a French
article introducing a new family and genus, and then **leaps into the
middle of a German article**.  The Latin description of* Asterina
orthosticha *Syd. nov. spec. is buried in a Key and a Table block, with
only part of it in a description… **LOTE and LOTL are clearly a
problem.**"*

§12.3.16 established the severity — `Description` fires at 52 % on
English, 53 % on Latin, 21 % on French and **1 %** on German.  What was
missing is the **volume**, without which the finding cannot be
prioritised.

#### Corpus share

| dominant language | docs | share | treatments/doc | `Description` share of blocks |
|---|---:|---:|---:|---:|
| English | 655 | **98 %** | 3.5 | 5.7 % |
| German | 8 | 1 % | **14.6** | 3.9 % |
| French | 5 | 1 % | 1.2 | 3.7 % |
| Spanish | 1 | 0 % | 1.0 | 8.0 % |

**Non-English-dominant documents are 2 % of the corpus.**  They punch
above that in treatments — the German documents average **14.6
treatments each against 3.5** for English, being large old volume scans
— so they are roughly **5 % of treatments**.

**The 2 % is a curation decision, not a property of the literature —
see §12.3.26.  The ranking implication below is withdrawn.**

~~So the LOTE failure is severe but bounded.  A near-total extraction
loss over 2 % of documents is a real defect and a poor use of ingest
effort, but it is not what is limiting corpus-wide quality.  It should
be ranked accordingly against §12.3.11's boundary theft or §12.3.23's
front-matter harvesting.~~

**Caveat, and it is a serious one: n = 8 German and n = 5 French
documents.**  The share is reliable to about a percentage point; the
per-language yields are not.

#### "LOTL" is a category error, and a useful one

**No document classified as Latin-dominant.**  Latin in this corpus is
not a document language — it is a **within-document register**, the
diagnosis embedded in an English or German paper.  That is why
§12.3.16 measured Latin per *block* and found 53 %, and why it cannot
be measured per document at all.

**The practical consequence is favourable**: Latin needs no separate
handling, because it already succeeds at the English rate wherever it
appears.  The problem is **LOTE alone**.

#### A language switch is an article-boundary cue

This treatment *"starts with the introduction to a French article… and
then leaps into the middle of a German article."*  §12.3.9 recorded
article-boundary detection as a missing structural level with no
proposed mechanism.  **A change of dominant language across adjacent
blocks is one** — cheap, computable from the text alone, and robust to
OCR damage since it rests on function-word frequencies rather than exact
strings.

It is obviously partial: it fires only where a volume mixes languages,
which is a minority even of the 2 %.  Recorded as a candidate signal for
that detector, not as a solution to it.

#### The operator's specific case

*"The Latin description… buried in a Key and a Table block, with only
part of it in a description."*  §12.3.6 and §12.3.15 exactly: in a
document where the content signal fails, `Key` and `Table` absorb the
prose by page geometry.  The Latin **would** have been detected at 53 %
had it been in a document where the surrounding labelling was working —
the failure here is the German context, not the Latin.

### 12.3.39 The `Diagnosis` homograph is how non-taxonomic articles reach p1

`taxon_f94b9c84`.  Operator: *"is not a taxonomic article."*  It is
**"Immunologic Diagnosis of Endemic Mycoses"** (Almeida-Paes et al.), a
clinical immunology review.  `Nomen ignotum`, `synthetic_nomenclature`
true, no DOI, **zero description characters** — and **ten `Diagnosis`
blocks**:

```
"The diagnosis of cryptococcosis, regardless of the causative species (C. n…"
"Antibody detection has been used as the principal coccidioidomycosis diagn…"
"immunologic tests as a tool for sporotrichosis diagnosis.  For this mycosis…"
"178. Imwidthaya, P.; Sekhon, A.S.; Mastro, T.D.…"        <- a bibliography entry
```

**Taxonomic *Diagnosis* — a differential description — and clinical
*diagnosis* — identifying a disease — are the same word.**

#### Measured

| | n | |
|---|---:|---:|
| morphology only — genuinely taxonomic | 357 | **78.1 %** |
| **clinical vocabulary, no morphology** | 12 | **2.6 %** |
| both | 5 | 1.1 % |
| neither | 83 | 18.2 % |

~837 clinical-diagnosis blocks corpus-wide.  **A modest rate with an
outsized consequence.**

#### Why 2.6 % matters more than it looks

`select_for_annotation` computes complexity from **description *or*
diagnosis** presence.  This treatment has **no description at all** —
its complexity is entirely supplied by clinical `Diagnosis` blocks.

**So a 2.6 % labelling error is not merely a mislabel: it decides
population membership.**  It moved a clinical review into **p1**, the
annotatable pool, where it consumed an annotation slot in round 5 and an
operator's reading time.  §12.3.8 framed the taxonomic-article gate as a
data-quality question; this is the same problem seen from the **cost**
side.

**§12.3.16 explains why `Diagnosis` is uniquely exposed.**
`Description` is protected by requiring Latinate *morphological*
register — which is why it failed on German and fired on a drug tablet.
**`Diagnosis` carries no such requirement**: the cue word alone suffices,
and the word is shared with clinical medicine, a field that overlaps
mycology heavily.

#### A first contamination estimate for p1

Round 5 is a **random draw from p1**.  Across roughly 48 treatments
reviewed so far, the operator has flagged **three** as not taxonomic
articles — `taxon_65cf0058` (FDA drug leaflet), `taxon_88431ff4`
(19th-century botanist's report) and this one.

**≈ 6 % of p1, on n = 3** — revised to **≈ 8 % on n = 4** by §12.3.40.
The interval is wide, but it is the **first estimate from a random
sample** rather than from targeted queries.

#### The shared signature is not a gate

All three carry `synthetic_nomenclature: true`, `Nomen ignotum` and no
DOI.  **Tempting, and insufficient**: §12.3.17 measured
`synthetic_nomenclature` at **22 %** of description-bearing treatments.
A gate on that signature would discard roughly one treatment in five to
remove one in sixteen.  **The signature is close to necessary and
nowhere near sufficient**, which is consistent with §12.3.17's finding
that the fields a real gate needs are stripped by `_slim_ingest`.

### 12.3.37 Figure captions should be catalogued separately and linked by name

`taxon_f6fa698e`.  Operator: *"The figure-caption (Figure 53) is for a
third taxon, Diaporthe middletonii"* — while the treatment is *Diaporthe
sojae* — and: *"**adding illustrations to treatments is going to be
challenging, as the figures are not necessarily contiguous with the
treatment.  Maybe catalog the illustrations separately and try to put
binomials on them?**"*

**The proposal is well-founded, and the current behaviour is worse than
the operator suspected.**

#### Measured

Over 846 `Figure-caption` spans attached to 320 treatments:

| | n | |
|---|---:|---:|
| caption **names a taxon** | 514 | **61 %** |
| no binomial in the caption | 332 | 39 % |
| …**matches** the treatment it is attached to | 28 | **3 %** |
| …same genus, different species | 60 | 7 % |
| …different taxon entirely | 426 | 50 % |

**Of the captions that name a taxon, only 5 % name the treatment they
are attached to.**  The operator's *Diaporthe middletonii* case is the
norm, not the exception — **proximity-based attachment is wrong roughly
19 times in 20.**

#### Both halves of the proposal are supported

* **Catalogue separately** — because position does not carry the
  linkage.  Figures are laid out to fit pages, and §12.3.9's whole-volume
  material makes this worse, but even in single modern articles the
  figure block for one species routinely sits inside a neighbour's
  treatment.
* **Link by binomial** — because captions name their taxon.  The
  headline rate is 61 %, **but §12.3.38 shows that is dragged down by
  non-caption content: restricted to blocks that actually open with
  `Figure N`, the naming rate is 95 %.**  The abbreviated form
  (`D. sojae`) is common enough to require handling alongside the full
  one.  The residual needs another route — figure-number
  cross-references from the treatment text (`Figs 8, 9` appears in
  `Nomenclature` blocks throughout this review) being the obvious
  candidate.

#### Two caveats, one of which limits the number

**A selection effect.**  Correct behaviour in the flawless treatments
(§12.3.18, §12.3.28) *excluded* their captions from the treatment
entirely — `Figure 8. Basidiomata of Xylodon daweishanensis` sits beside
its treatment without being attached to it.  So **treatments that have
attached captions are disproportionately those where attachment went
wrong**, and the 5 % is not an estimate of "how often the right caption
reaches the right treatment" across the corpus.  What it does establish
is that **the attachments which exist are almost never right**, which is
the claim that matters for the design.

**Binomial extraction is noisy** — §12.3.32 showed `Mycelium
amphigenum` parses as a name.  That inflates the "different taxon
entirely" bucket but **cannot produce false mismatches**: a caption
naming its own treatment's taxon is still detected, so the 5 % match
rate is robust to it.

#### Where this leaves `figure_caption_spans`

The field is currently populated by proximity and is wrong 95 % of the
time it can be checked.  **Until a name-based linker exists, a
downstream consumer treating `figure_caption_spans` as "this treatment's
illustrations" is being misled**, and that is worth stating plainly
because the field looks authoritative.

### 12.3.44 The taxpub route did not move: 86.2 % was a coverage artefact

§12.3.43 left one question open — why the `taxpub_treatment_extractor`
route showed 86.2 % id stability against the classifier route's 97.3 %,
when the `Phylogeny` fix barely reaches it.  The recorded guess was that
the taxpub path had changed since `production_v4` was extracted on
2026-08-11, with Trello #401's source-anchor work the likely mover.
**Tested 2026-09-01.  The guess is wrong, and so was the metric.**

#### The direction of the measurement hid the answer

The 86.2 % counts **v4 ids that survive into v4_1**.  Measured the other
way — v4_1 ids that already existed in v4 — the taxpub route scores
**948 / 948 = 100 %**.  The route invented nothing; it only failed to
reproduce some of what v4 held.  A content change moves ids in *both*
directions, so zero new ids already falsifies "the taxpub path changed".

| | v4 | v4_1 | v4 ids surviving | v4_1 ids that are new |
|---|---:|---:|---:|---:|
| `classifier_logistic_v3` | 1 013 | 980 | 97.4 % | 15 |
| `taxpub_treatment_extractor` | 1 117 | 948 | 86.7 % | **0** |

#### All 148 losses sit on 18 documents, and none of them are taxpub's

Of 196 taxpub-only documents, **18 account for every lost treatment**.
On those 18, the split is absolute — classified by `source_anchors`
kind, which independently labels the route in v4 data that predates the
`extractor` field:

* **every treatment kept: taxpub-anchored** (`arpha` / `jats_section` /
  `mycobank` / `plazi`), 98 of them;
* **every treatment lost: `pdf`-anchored or unanchored** — the `.ann`
  route's shape — 148 of them.

Not one taxpub-anchored treatment was lost.

#### Re-extracting those 18 documents settles it

`skol_scratch_route_test`, today's code, `--doc-id` scoped to exactly
those 18 documents, both passes allowed to run:

| | treatments |
|---|---:|
| v4 holds for these 18 documents | 246 |
| v4_1 produced | 98 |
| **today's re-run produced** | **246 — 100 % of v4's ids, 0 new** |

148 `classifier_logistic_v3` + 98 `taxpub_treatment_extractor` = 246,
reproduced **id for id**.  The taxpub extractor is bit-stable against a
three-week-old corpus; today's code reproduces v4 exactly on the very
documents that looked least stable.

**The cause is pass scoping, not code.**  The `.ann` pass and the G.1
taxpub sweep select documents independently.  These 18 documents entered
the sweep's 200 but not the Spark pass's `--limit 200`, so v4_1 holds
their taxpub half against v4's whole.  The largest is
`028539777b275cdeaa17e0416aa4b54f` — the same 41 + 9 = 50 document
`85c0188` used to verify the scoping fix.  Its 41 were never *lost*;
they were never *attempted*.

The git history agrees: the last behavioural change anywhere in the
taxpub call graph (`taxpub_treatment_extractor.py`,
`ingestors/jats_to_yedda.py`, `treatment_assembler.py`, `state.py`) is
**`70706d4`, 2026-07-22** — #401 Commit B, already in the code that
produced v4 on 2026-08-11.  Everything after it is scoping (`835171a`)
or provenance (`deea058`), neither of which touches `_CANONICAL_FIELDS`.

#### The corrected comparison

Dropping the 18 half-covered documents leaves **341 documents where
both runs attempted the same passes**:

| route | docs | v4 | v4_1 | v4 ids surviving | new |
|---|---:|---:|---:|---:|---:|
| `classifier_logistic_v3` | 159 | 970 | 959 | **97.3 %** | 15 |
| `taxpub_treatment_extractor` | 178 | 828 | 828 | **100.0 %** | 0 |
| overall | 341 | 1 841 | 1 830 | 98.6 % | 15 |

**The whole of the v4 -> v4_1 movement is the classifier route, and the
whole of that is `8c0148d`.**  Twelve of the fourteen changed documents
show the same signature — **two treatments lost, one gained, and the
gained one carries a populated `phylogeny` field**.  On
`0c40d56b485351eba5eddc2c1535a7a6` the arithmetic is visible: two
`Nomen ignotum` stubs with `figure_captions` of 88 and 1 113 characters
become one with 1 202 (the join adds a newline), every other canonical
field identical, plus 12 524 characters of retained `phylogeny`.  That
is precisely what moving `Phylogeny` out of MISC and into the
treatment-continuing set should do: a block that used to run the gap up
and split a treatment now holds it open.

#### What to carry forward

* **A stability figure computed across passes with different document
  scopes measures the scoping, not the code.**  Restrict to documents
  both passes attempted, or report the two directions separately — one
  direction alone cannot distinguish a rename from an absence.
* **`source_anchors` kind is a usable route discriminator for
  pre-`deea058` data** — 99.7 % agreement with `extractor` on the 1 928
  treatments where both are available (the residue is 6 unanchored
  taxpub treatments).  The conservation audit in §12.3.14 can use it to
  route-label v4 retroactively without a re-extraction.
* §12.3.43's closing claim stands, and now stands twice over: without
  the `extractor` field this would have read as a single 91.66 % verdict
  on `8c0148d`.  With it, the fix's true footprint is 15 treatments on
  14 documents, and the other 148 differences are not the fix at all.

### 12.3.43 The re-run, conditioned on route — and what that separates

`production_v4_1`, re-run 2026-09-01 with the scoping fix (`835171a`)
and the provenance field (`deea058`) in place.

**The scoping holds:** `Scanned skol_dev: 200 is_taxpub docs` against
1 779 before, 948 taxpub treatments against 7 642.  1 928 treatments,
**0 failed saves**, and three `file_exists` lines — the benign worker
race on `db.create`.

#### The provenance field earned itself on its first use

| | treatments |
|---|---:|
| `classifier_logistic_v3` (the `.ann` route) | 980 |
| `taxpub_treatment_extractor` (the `article.xml` route) | 948 |

**359 comparable documents** (200 Spark + 200 taxpub − 41 overlap),
v4 2 087 -> v4_1 1 928, **91.66 % of ids stable**.  Pooled, that number
is uninterpretable — exactly the problem §12.3.42 recorded.  **Split by
route it separates cleanly:**

| route | docs | v4 | v4_1 | ids stable |
|---|---:|---:|---:|---:|
| `classifier_logistic_v3` | 159 | 970 | 959 | **97.3 %** |
| `taxpub_treatment_extractor` | 196 | 1 074 | 926 | **86.2 %** |

**Only the first row speaks to commit `8c0148d`.**  It is the route the
`Phylogeny` fix touches, it moves 970 -> 959 (−1.1 %), and 97.3 %
stability is close to what a grouper change of that size should
produce.

**The second row is not attributable to the fix and needs its own
explanation.**  The taxpub route carries **1 of 948** phylogeny-bearing
treatments, so `Phylogeny` barely reaches it — yet it is the *less*
stable route by 11 points.  The likeliest cause is simply that
`production_v4` was extracted **2026-08-11**, and the taxpub path has
changed since (Trello #401's source-anchor work lands squarely on it).
**Untested**, and recorded as the next question rather than a
conclusion.

**Tested and wrong — see §12.3.44 (2026-09-01).**  The taxpub path has
not changed since v4; the 86.2 % is a coverage artefact of 18 documents
that the sweep reached and the Spark pass did not.  Measured in the
other direction the route scores 100 %, and a re-extraction of those 18
documents reproduces v4 id for id.

#### The fix does what it was meant to

**120 treatments now carry a populated `phylogeny` field, 268 834
characters — 119 of them on the classifier route.**  Content that
commit `8c0148d` was written to stop discarding is being retained, on
the route it targets.

#### What this vindicates

§12.3.42 recommended a separate database on the grounds that an
in-place run would be unverifiable.  **The re-run demonstrates the
sharper version of that argument**: without the `extractor` field the
only available number was 91.66 %, which conflates a 97.3 % route with
an 86.2 % one and would have been read as a single verdict on
`8c0148d`.  **The provenance field, added because a comparison demanded
it, changed the conclusion on its first use.**

### 12.3.42 Re-extraction: 98.8 % of ids survive, but the writer never deletes

Operator, 2026-09-01: *"What consequences would we see to reextracting?
Could we orphan treatments?  I'm trying to decide if I need to create a
new experiment, 5.1, with a new database."*

#### Simulated, not estimated

`group_paragraphs` is pure, so the old and new groupers can be run over
the same `.ann` text and their `taxon_id`s diffed directly.  Over 120
documents:

| | |
|---|---:|
| treatments, old grouper | 568 |
| treatments, new grouper | 564 |
| **ids stable** | **561 — 98.77 %** |
| ids only in old — **orphaned** | 7 |
| ids only in new — created | 3 |
| documents affected | **2 of 120 (1.7 %)** |

**`phylogeny` is not in `_CANONICAL_FIELDS`**, so the new field alone
cannot move an id.  The churn comes from the second half of commit
`8c0148d`: `Phylogeny` joining `_TREATMENT_SECTION_LABELS` **resets
`misc_gap`**, so a treatment that used to close may stay open and absorb
more text — which does change `description`, which is hashed.

Extrapolated to 81 527 treatments: **~1 000 ids would change.**

**Note the direction.**  Treatment count *falls* (568 → 564): the fix
makes treatments longer.  That is the intended behaviour, but it
marginally worsens §12.3.22's finding that nothing bounds treatment
extent.

#### The decisive fact is the write semantics

`bin/extract_treatments_to_couchdb.py:961-971` **upserts and never
deletes**:

```python
existing_doc = db[doc_id]
taxon_doc['_rev'] = existing_doc['_rev']
db.save(taxon_doc)
```

So an in-place re-extraction would update the 98.8 %, create the new
ones, and **leave ~1 000 orphans alive in the database** — carrying
valid-looking content, indistinguishable from current treatments, and
detectable only by re-deriving every id.

**This is the same defect T6 already warns about one layer up**: the
plan notes that `annotation_doc_id` has no deletion pass, so re-running
the annotator into the same DB yields *"the union of two prompts'
vocabularies — worse than useless for a before/after comparison"*.  The
treatment writer has the identical shape.

**And an in-place run is not reversible.**  The upsert overwrites
`_rev`; the previous treatment content is gone.

#### What references a `taxon_id`

| artefact | distinct ids |
|---|---:|
| `data/annotation_rounds/production_v4_round5.txt` | 1 000 |
| this memo | 176 |
| `tests/fixtures/pathologies.json` | 136 |
| `data/merge_review_p2a_20260825.md` | 31 |
| `data/p2a_dossiers/index.html` | 30 |
| `features_candidate` / `_hand` / `_status` | 9 068 / 2 244 / 6 637 docs, keyed `<tid>:<label>:<start>` |

At 1.23 % churn the *expected* damage is small — roughly 12 of round
5's 1 000, 2 of the memo's 176, 2 of the fixtures' 136 — **but a
dangling id fails silently**, and the annotation DBs would keep rows
pointing at treatments that no longer exist.

#### Recommendation: a new database, and the reason is the upsert

**Not because the churn is large — it is not — but because an in-place
run is unverifiable and unreversible.**  A separate target gives:

* **a diff**, which is the only way to confirm the fix did what
  `8c0148d` intends: that `Phylogeny` content is now captured, and that
  boundaries moved only where predicted;
* **orphan detection for free** — ids present in v4 and absent in the
  new DB *are* the orphan list, rather than something to reconstruct;
* **the whole evidence base stays valid.**  The memo, the fixtures, the
  round files and the dossiers all address v4 ids; leaving v4 untouched
  keeps 100 % of that citable while the new extraction is evaluated.

The alternative — re-extracting in place — requires **first** adding a
deletion pass to the writer, which is a change to production extraction
made in order to run an experiment.  **Wrong order.**

#### What was created (2026-09-01)

**`production_v4_1`**, status `draft`.  Deliberately **not**
`production_v5`: v5 is triggered by M3 of
`docs/plans/production-v5-execution.md`, when a change needs a fresh
pipeline container.  This needs no such thing — **it is a
re-*grouping*, not a re-classification.**

| database | |
|---|---|
| `ingest` | `skol_dev` — **shared** |
| `training` | `skol_training_v3_combined_no_golden` — **shared** |
| `annotations` / `spans` | `skol_exp_production_v4_01_00_ann_combined` — **shared** |
| `treatments_prose` | `skol_exp_production_v4_1_02_00_treatments_prose` — **new** |
| `features_candidate` | `skol_exp_production_v4_1_02_50_features_candidate` — **new, empty** |

**Sharing the annotations DB is the point.**  Commit `8c0148d` changed
`treatment.py`, not the classifier: Pass 1 and Pass 2 output is
byte-identical, so the expensive half of the pipeline does not re-run.
Only `extract_treatments_to_couchdb` does.

#### Two configuration defects at creation — the second invalidated a run

**`manage_experiment create` does not copy the CRF model keys.**
`production_v4` carries `classifier_model_pass1`,
`classifier_model_pass2` and `classifier_model_single` pointing at
`v4_layout`, `v4_pass2_combined` and `v4_single_combined`.  A new
experiment gets **none of them** unless `--redis-key-pass1/pass2/single`
are passed explicitly, and they were not.

**Superseded — the CRF keys were not the cause.**  See "The actual
cause" below.  `classifier_logistic_v3`'s docstring states plainly that
it *"does not re-run the classifier; it consumes the existing
attachment"*, contributing the YEDDA verbatim.  The keys were missing
and adding them is right for consistency, but they cannot have changed
this run.  Recorded rather than deleted because the reasoning was wrong
in an instructive way: **the component was never read before the theory
was formed.**

~~That is not cosmetic, because extraction re-labels.~~
`extract_treatments_to_couchdb` imports `group_paragraphs` but never
calls it.  The real path is
`Dispatcher.extract` -> `treatment_assembler`, whose input is
**`state.merged_ann_text()`** — the merged output of whichever labeler
the dispatcher selected — and only then
`parse_annotated -> remove_interstitials -> group_paragraphs`.

**So "re-grouping, not re-classification" was wrong.**  The stored
`.ann` attachment seeds the labeler; it is not used directly.  With the
CRF keys absent the run labelled with a fallback, and its output is not
comparable to `production_v4`.

**Evidence that caught it.**  The worst-affected document
(`028539777b275cdeaa17e0416aa4b54f`, 557 `Nomenclature` blocks) has
**zero `Phylogeny` blocks**, so commit `8c0148d` provably cannot change
it — yet it went 50 treatments to 9.  Running today's grouper directly
on its attachment gives **41 either way**, with or without
`remove_interstitials`.  Three different numbers for one document is
the signature of a different labeling, not a different grouping.

**Everything measured from that run is withdrawn** — the 95.57 % id
stability, the 9 006 -> 8 622 treatment delta, the 121 phylogeny-bearing
treatments, and the reading that the worst document's consolidation was
an improvement.  The keys were added afterwards; the run must be redone.

**And the simulation in §12.3.42 was measured on the wrong path too.**
It called `group_paragraphs` on the attachment, skipping the labeler,
`CouchDBFile.read_line` and `remove_interstitials`.  **Its 98.77 %
id-stability figure does not model the pipeline** and should not be
quoted; a corrected number requires a run with the CRF keys in place.

#### The actual cause: a second extraction path that ignores `--doc-id`

Running the extractor for **one** document produced this:

```
[G.1] Scanned skol_dev: 1779 is_taxpub docs with article.xml
[G.1] Added 7642 taxpub treatments from skol_dev
[DRY RUN] Would save 7683 treatments
```

**`--limit` was never at fault.**  It does exactly what it says —
`annotated_df.select("doc_id").distinct().limit(limit)` bounds distinct
input documents from the annotations DB, and the log confirms
`Limited to 200 documents: 200 attachment(s)`.  The 1 936 documents the
run covered is **200 (Spark) + 1 779 (unscoped sweep) − 43 overlap**.
**The whole anomaly was the sweep**, and `--limit` was only ever
suspicious by association.

**7 683 treatments from a single-document request.**
`iter_taxpub_treatments` — the "Phase G.1 non-Spark sweep" — scans the
whole of `skol_dev` for `is_taxpub` documents carrying `article.xml` and
extracts from the **XML**, bypassing the `.ann` entirely.  **It honours
neither `--doc-id` nor `--limit`.**

That is the T0f class again: a flag accepted and silently not applied.
`--doc-id` reads as "process this document"; it scopes only the Spark
half.

**And the anomalous document is `is_taxpub: true`.**  Its `skol_dev`
attachments are `article.xml`, `article.txt`,
`article.page-headers.json` and `article.spans.v4.json` — **no `.ann` at
all**; the `.ann` lives in the annotations DB.  So it is reachable by
*both* routes, which disagree:

| route | treatments |
|---|---:|
| `.ann` via the Spark path (today's code, verified locally) | **41** |
| `article.xml` via the taxpub sweep | **9** — what `v4_1` stored |
| whatever `production_v4` used on 2026-08-11 | **50** |

**Nothing records which route produced a treatment.**
`treatment_assembler` hard-codes
`attachment_name=_DEFAULT_ATTACHMENT_NAME`, so **all 8 622 `v4_1`
treatments claim `article.txt.ann`** including the XML-derived ones.
The two paths are indistinguishable in the stored data.

#### What this means for the comparison

**The v4 / v4_1 diff was never interpretable.**  It compares documents
that may have been extracted by different routes in each run, with no
field to condition on.  Every number from it is withdrawn — as recorded
above, but now for the right reason.

**Three defects to fix before re-running:**

1. ~~`--doc-id` and `--limit` must scope the taxpub sweep~~ — **fixed
   2026-09-01** (`835171a`).  `taxpub_doc_admitted` applies
   `skip_doc_ids` first, then `only_doc_ids`, then the `is_taxpub` and
   `article.xml` requirements, with `None` meaning unfiltered and an
   empty set meaning nothing.  When `--doc-id` is given the sweep
   iterates that set instead of the whole database, so it costs one
   lookup rather than a full scan.

   **Verified end to end**, and the verification settled the routing
   question too:

   ```
   before:  Scanned 1779 is_taxpub docs   Added 7642   Would save 7683
   after:   Scanned    1 is_taxpub doc    Added    9   Would save   50
   ```

   **50 = 41 (`.ann` path) + 9 (taxpub path)** — **exactly what
   `production_v4` holds for this document.**  So v4's 50 was correct,
   both paths contribute, and today's code reproduces it once the sweep
   is scoped.  The `v4_1` run's 9 was an artefact of the unscoped sweep,
   not of any code change.
2. ~~Treatments need a provenance field~~ — **fixed 2026-09-01.**
   `Treatment.set_extractor` / `as_row()['extractor']`, fed by
   `PipelineState.winning_label_source()`, which selects **identically
   to `merged_ann_text`** (same `max` over the same list) so the
   recorded provenance always names the contribution whose text was
   actually used, ties included.  Added to `EXTRACT_SCHEMA` and
   deliberately **not** to `_CANONICAL_FIELDS` — provenance, not
   identity.

   Verified on the document that exposed the problem:

   | path | treatments | `extractor` |
   |---|---:|---|
   | `.ann` (Spark flow) | 41 | `classifier_logistic_v3` |
   | `article.xml` (G.1 sweep) | 9 | `taxpub_treatment_extractor` |

   **41 + 9 = 50 = what `production_v4` holds.**  The two routes are now
   distinguishable in stored data, so a v4/v4_1 diff can be conditioned
   on route and §12.3.14's conservation audit can tell a missing
   treatment from a differently-routed one.
3. **The routing rule needs stating.**  For a document reachable both
   ways, which wins, and why?  9 versus 41 is not a rounding
   difference.

**The first defect:** `manage_experiment create` derived
`spans` as `skol_exp_production_v4_1_01_00_ann`, a database that does
not exist.  In `production_v4` that key mirrors `annotations`, and it
was set to match — otherwise span resolution would have pointed at
nothing.  Worth knowing before the next experiment is created this way.

**The comparison this enables**, and the reason it was worth a separate
database:

1. **the orphan list is free** — ids in `production_v4` and absent in
   `production_v4_1` *are* the orphans, no re-derivation needed;
2. **the fix is verifiable** — does `phylogeny` actually populate, and
   at the ~39 300 blocks §12.3.12 measured?
3. **the side effect is measurable** — treatment count fell 568 → 564
   in simulation, so §12.3.22's unbounded-extent concern can be checked
   directly rather than assumed;
4. **nothing citable breaks** — the memo's 176 ids, the fixtures' 136,
   the round files' 1 000 and the annotation DBs all still address v4.



### 12.3.41 Boundary theft is a document property — and so is almost everything else

`taxon_fcaca7fe`.  Operator: *"The Nomenclature… lost 2 lines to a
Misc-exposition.  The type_designation… loses its last line to a
Misc-exposition.  Etymology loses its last line to a Misc-exposition…
The biology block loses its last line to a Misc-exposition."*

**Four tail thefts in one treatment**, against §12.3.27's corpus rate of
15.5 %.  That is not plausible as four independent events.

#### Measured

Over 126 documents with at least five theft opportunities (a content
block immediately followed by `Misc-exposition`):

| per-document theft **rate** | |
|---|---:|
| p10 | **0 %** |
| p25 | 19 % |
| median | **38 %** |
| p75 | 48 % |
| p90 | **60 %** |

**10 % of documents have no theft at all; the top 10 % hold 57 % of
every theft in the sample.**

**A caveat on my own statistic.**  The raw count dispersion
(variance/mean) is **59.7**, which reads as extreme clustering — but it
is inflated by document length, since long documents have more
opportunities *and* more thefts.  **The rate distribution above is the
fair measure**, and it shows real but less dramatic concentration: a
band from 0 % to 60 % rather than a two-population split.

#### The cross-cutting pattern this review keeps producing

**This is the third independent finding of the same shape**, and it is
probably the most actionable structural result in §12.3:

| finding | concentration |
|---|---|
| §12.3.1 — #407's manual residual | **80 % of the work sits in 4 % of documents** |
| §12.3.6 — rogue `Key` | a **document**-level property; the median document has none |
| §12.3.41 — boundary theft | top **10 %** of documents hold **57 %** of thefts |

Add §12.3.40's non-taxonomic documents — 4 documents producing 81
treatments against a median of 1 — and the pattern is consistent:
**defects concentrate in documents, not across blocks.**

**The strategic consequence.**  Every remedy discussed in this section
has been framed as a model or schema change — better cues, a subsumption
lattice, sub-line labelling, reflow.  **The measurements keep saying the
cheaper lever is document triage**: identify the bad documents, and
either repair them upstream (re-OCR, reflow, re-ingest) or exclude them,
rather than teaching the model to cope with material that is broken
before it arrives.

That does not retire the model work — the median document still has a
38 % theft rate, so this is concentration rather than a clean
two-population split, and §12.3.27's structural asymmetry (cues protect
heads, nothing protects tails) applies everywhere. **But it does say
where the first dollar goes.**

#### The operator's other two observations

* *"The type_designation eats a short diagnosis"* — over-extension
  (§12.3.36), in the same treatment as four thefts.  **Extent errors run
  in both directions within one document**, which argues the cause is
  segmentation quality rather than a directional bias.
* *"The final two notes sections belong together and the intervening
  page header was correctly identified"* — span reconstruction working
  (§12.2's 37 % that survive), and the fifth such success recorded.

### 12.3.40 Non-taxonomic documents are prolific — which is good news for the gate

`taxon_fc47df1e`.  Operator: *"is not a taxonomic article."*  It is the
**IAS 2021 Abstract Book** — the International AIDS Society conference
proceedings, **2 443 blocks**.

**A second entry route.**  §12.3.39 found the clinical review entered p1
through the `Diagnosis` homograph.  This one entered through
`description` (862 characters), as did the FDA leaflet and the
botanist's report.  **Three of four came in via `Description`, one via
`Diagnosis`** — consistent with §12.3.16: register-based labelling has
no notion of document context, so any text in the right register
qualifies.

#### The blast radius

| document | treatments | synthetic | p1-eligible |
|---|---:|---:|---:|
| FDA leaflet | 2 | 2 | 2 |
| Botanist report | 7 | 6 | 6 |
| Clinical review | 5 | 5 | 4 |
| **IAS abstract book** | **67** | **67** | 23 |
| **total** | **81** | 80 | **35** |

**Median treatments per source document corpus-wide: 1.**  These four
average twenty, and the abstract book sits at the **99th percentile**.

**Non-taxonomic documents are far more prolific than typical ones**, and
the reason is structural: they are long, contain no real nomenclature,
and so the grouper synthesises a stub every time a section-labelled
block appears with no name ahead of it.  **All 67 abstract-book
treatments are synthetic.**

#### This substantially improves the case for a document-level gate

The contamination estimate is **treatment-level** — the operator reviewed
treatments, and four of roughly 49 came from non-taxonomic documents, so
**≈ 8 %**.  But because these documents are prolific, the corresponding
**document-level** rate is far lower: four documents produced 81
treatments against a corpus average near 4.6.

**So a gate operating on documents removes ~8 % of treatments by
rejecting ~2 % of documents.**  That leverage was not visible from
§12.3.8, which argued the gate on data quality, or §12.3.39, which
argued it on annotation cost.  **It is a better intervention than either
framing suggested**, and it acts before extraction rather than after.

#### A candidate gate feature that is not the one already rejected

§12.3.39 rejected `synthetic_nomenclature` as a gate: 22 % of
description-bearing treatments carry it, so gating on the flag would
discard one in five to remove one in sixteen.

**But that is the per-*treatment* flag.**  Every one of the abstract
book's 67 treatments is synthetic — a **per-document synthetic fraction
of 100 %**, against a corpus base rate of 22 % per treatment.  **The
fraction may separate where the flag does not**, and it is computable
from data already in CouchDB without the title or journal that
`_slim_ingest` discards (§12.3.17).

**Untested** — recorded as the next measurement rather than a
recommendation.  The obvious risk is that legitimate whole-volume scans
(§12.3.9) also lose their nomenclature and would score high; the test
must therefore separate "no names because it is not taxonomic" from "no
names because the layout defeated us".

### 12.3.38 Only 41 % of `Figure-caption` blocks open like a caption

`taxon_f6fa698e`, operator follow-up: *"The figure_caption block is
really a **continuation of a Figure-caption on the previous page**."*

**Confirmed as a mechanism, and it is a small one.**  Of 1 253
`Figure-caption` blocks:

| | n | |
|---|---:|---:|
| opens with `Figure N` | 512 | **41 %** |
| no opener | 741 | **59 %** |
| …preceded by another `Figure-caption` across furniture | 50 | 4 % |
| …**with page furniture between — a split caption** | 41 | **3 %** |

The operator's case is real and measurable at ~3 %, and it is §12.2's
page-break class applied to a label that had not been checked for it.

#### But the no-opener population is mostly not captions at all

The examples are decisive:

```
'DiscussionBacteria and fungi are among the m…'
'ConclusionIn conclusion, to the best of our …'
'Table 7 GenBank accession numbers of species…'
```

Section prose and a table header, labelled `Figure-caption`.  **59 % of
blocks carrying this label do not begin like a caption**, and only a
twelfth of that is explained by page splits.

**Read together with §12.3, this pins the label from both sides:**

| direction | measure |
|---|---|
| blocks **cued** `Fig N` that get the label | **66 %** (§12.3) |
| blocks **labelled** `Figure-caption` that are cued | **41 %** |

**Neither precise nor complete.**  That is consistent with §12.3.2's
finding that `Figure-caption` has no semantic handle and its 131 errors
scatter across seven unrelated labels — a caption is defined by page
position, and without layout the label drifts in both directions.

#### The practical consequence improves §12.3.37's recommendation

**Naming rate splits sharply**: blocks that open with `Figure N` name a
taxon **95 %** of the time; those that do not, **67 %**.

So §12.3.37's *"61 % of captions name their taxon"* understates what a
name-based linker can reach, because that 61 % was computed over a
population **59 % of which is not caption text**.

**Restricting the linker to caption openers gives 95 % naming coverage
rather than 61 %** — and it simultaneously filters out the Discussion,
Conclusion and Table prose that would otherwise be catalogued as
illustrations.  **The opener test is doing double duty: a caption
detector and a name-availability filter in one.**

The 3 % split captions must be rejoined *before* that filter is applied,
or their names — which sit in the opener on the previous page — are lost
with them.

### 12.3.36 Block *extent* is a separate axis from block *label*

`taxon_ecfa2e69` (*Taeniolella vermicularis*).  Operator: *"The
type-designation starts OK, but then **runs into a Habitat (biology)
block, and a Notes block**.  Admittedly, the Notes block starts with a
discussion of the type specimen."*

That is **over-extension** — the mirror of §12.3.11's theft.  There a
block loses material to a neighbour; here it keeps going and swallows
whole following sections.  Confirmed in the treatment: its 578-character
`Type-designation` block contains an embedded `Notes` cue.

#### Generalising §12.3.13's test to every label

§12.3.13 counted `Fig. 3 –` openers appearing mid-block as missed
splits.  **The same test applies to every self-declaring cue**: a
`Habitat.`, `Notes.` or `Holotype.` appearing past the opening of a
block marks a boundary the segmenter did not draw.

| host label | blocks | over-extended | rate |
|---|---:|---:|---:|
| **`Biology`** | 580 | 89 | **15 %** |
| **`Notes`** | 911 | 104 | **11 %** |
| `Materials-examined` | 613 | 44 | 7 % |
| `Key` | 980 | 62 | 6 % |
| `Phylogeny` | 302 | 17 | 6 % |
| `Description` | 1 548 | 81 | 5 % |
| `Type-designation` | 282 | 13 | 5 % |
| `Misc-exposition` | 4 814 | 224 | 5 % |
| `Figure-caption` | 1 053 | 9 | **1 %** |
| **all** | **14 878** | **762** | **5 %** |

#### This is a different axis from cue-honoring, and they are independent

§12.3 measures whether a block's **label** is right — 65 % corpus-wide.
This measures whether its **extent** is right.  **A block can honor its
own opening cue perfectly and still swallow the next three sections**,
which is exactly what happened here: the `Type-designation` block opens
`Lectotype (designated here, MycoBank MBT…` — correctly cued, correctly
labelled — and runs on through a habitat statement into the notes.

**5 % of blocks are labelled right and bounded wrong.**  Any accuracy
metric computed per block will score them as successes.

#### The `Notes` ↔ `Biology` boundary is genuinely fuzzy

| pair | n |
|---|---:|
| `Notes` swallowed `Biology` | 85 |
| `Biology` swallowed `Notes` | **78** |

**Near-perfect symmetry.**  A directional collapse (§12.3's
`Type-designation` → `Materials-examined` at 124 against 12) indicates a
distinction being lost; **symmetry indicates a boundary nobody draws
consistently** — including the operator, whose *"admittedly, the Notes
block starts with a discussion of the type specimen"* is the same
hesitation measured from the inside.

That places `Notes`/`Biology` with the pairs recorded in §12.3.30 where
**the reviewer is choosing rather than correcting**, and it is the
strongest case yet: the two directions are within 8 % of each other.

#### An independent confirmation, and a success

`Misc-exposition` swallowed `Figure-caption` **76** times — §12.3.13
measured the same class at 195 with a stricter pattern over all hosts.
Two routes, consistent magnitudes.

And the operator's closing observation is a success worth recording:
*"The Notes section continues in a notes block.  We correctly classify
intermediate material and conclude the Notes section with another notes
block."*  The two `Notes` blocks are separated by **seven** intervening
blocks — two page headers, two copyright lines, two editor blocks and a
figure caption — all correctly labelled, with the notes reassembled
across them.  **Span reconstruction at its best**, in the same treatment
as a five-section over-extension.

### 12.3.35 Reflow versus `Table` detection — the tension is smaller than it looks

Operator: *"I'm wondering about the interaction between Table detection
and reflow.  Both of the short-line fragments here could have been
avoided if we reflowed the document before detecting Nomenclature and
Description blocks.  **OTOH, Table detection does require line length as
an input.  Reflow and Table should each emit a quality metric** to help
us decide which case we are looking at.  I'm also thinking about
investigating an **OCR solution that extracts non-text cues**."*

#### A discriminator was tried and failed

The natural quality metric is **column alignment**: a real table has
regular columns and therefore low variance in line length, while
shattered prose has arbitrary fragment lengths and high variance.
Measured as the coefficient of variation over line lengths, restricted
to short-lined blocks:

| group | n | median CV | CV < 0.35 |
|---|---:|---:|---:|
| `Table` (short-lined) | 2 384 | 0.59 | 19 % |
| debris between two `Description` blocks | 216 | 0.67 | 12 % |

**Barely separable, and both are high.**  The hypothesis is refuted.

**The reason matters more than the result: the test was contaminated by
the phenomenon it was testing.**  §12.3.15 established that `Table` is a
short-line detector, not a table detector — 67 % of `Table` blocks
contain a binomial and ~3 100 carry an outright nomenclatural act.  So
the comparison set on the left is *also* mostly debris.  **There is no
ground-truth set of real tables in this corpus to calibrate a metric
against.**

#### Which reframes the tension

The concern was that reflow destroys an input `Table` detection needs.
**But that input is not currently producing table information.** The
cost of destroying it is therefore much lower than it appears:

| | |
|---|---|
| what reflow would break | a signal that already mislabels citations and prose as `Table` (§12.3.15) and lets treatments over-extend (§12.3.22, `Table` is transparent to `MISC_GAP_LIMIT`) |
| what reflow would fix | micro-fragment shattering (§12.3.34, 49 % of description interruptions), mid-line boundaries where newlines were lost (§12.3.13, §12.3.31), and the short-line citations `Table` swallows |

**So reflow-before-detection is the stronger ordering on current
evidence** — and the quality metric the operator asks for cannot be
built from the existing `Table` label in any case, because that label
does not mark real tables.  **Building it requires table ground truth
first**, which is a small annotation task and a prerequisite rather than
a side effect.

*(Existing machinery: `pdf_section_extractor.py` already contains
reflow-adjacent code.  Not audited here.)*

#### The OCR direction addresses the root cause of five recorded classes

*"an OCR solution that extracts non-text cues"* is the instrument this
section has been circling without naming.  Every failure above is an
attempt to **infer layout from text after layout was discarded**:

| class | what it is really inferring |
|---|---|
| §12.3.15 `Table` as short-line detector | column boundaries, from line length |
| §12.3.6 rogue `Key` | block role, from block length |
| §12.3.13 lost newlines | line structure, after it was thrown away |
| §12.3.34 micro-fragments | paragraph continuity, across column breaks |
| §12.3.31 mid-line boundaries | where a name ends and prose begins |

**Font size, weight, indentation, ruling lines and column boxes would
supply four of these five directly**, without inference — a real table
is identified by its ruling and cell geometry, not by its line lengths.
**And §12.3.13's lost newlines are a pure artefact of text-only
extraction**, so they would not arise at all.

This does not resolve §12.3.31's mid-line boundary, which is a
*labelling granularity* question rather than a layout one, and it does
nothing for §12.3.16's LOTE failure.  **But it is the single change on
the table that would retire the most recorded classes at once**, and it
targets causes rather than symptoms.

### 12.3.34 Micro-fragments shatter descriptions — and a mid-line citation *causes* a merge

`taxon_ec570d25` (*Teratosphaeria viscida*).  Operator: *"The first two
description blocks are separated by **three blocks of very short lines
which are clearly part of the description**… The type-designation… should
include the material in the following Misc-exposition.  **The trailing
description is a completely different organism.**"*

#### The micro-fragments

```
IN  209c  mean-line 52  [Description]     Diagnosis: Leaf spots circular to irregular,
      2c  mean-line  2  [Misc-exposition] to
     11c  mean-line  5  [Table]           pale | brown,
     21c  mean-line 10  [Misc-exposition] slightly | verruculose,
IN  210c  mean-line 69  [Description]     cylindrical, straight to variously curved,
```

The interrupting "blocks" are the words **`to`**, **`pale brown,`** and
**`slightly verruculose,`** — one sentence of a description shattered by
column layout into fragments of 2, 11 and 21 characters.  This is
§12.3.15's short-line mechanism at its limit: a **two-character** mean
line length.

**Measured over 844 description-to-description gaps in 300 documents:**

| interrupting label | n | share | short-lined (≤30 c) |
|---|---:|---:|---:|
| `Misc-exposition` | 926 | 44 % | 553 — **60 %** |
| `Page-header` | 368 | 17 % | 340 — 92 % |
| `Biology` | 157 | 7 % | 7 — 4 % |
| `Figure-caption` | 94 | 4 % | 20 — 21 % |
| `Table` | 76 | 4 % | 72 — **95 %** |

**49 % of everything interrupting a description is short-lined.**  So
the dominant cause of a fractured description is not a genuine
intervening section but **layout fragments**, and §12.3.11's boundary
theft and §12.3.15's `Table` capture are two views of the same
underlying failure: **the segmenter emits column debris as blocks, and
the labeller must then assign it something.**

#### The trailing organism: §12.3.31 is a *merge cause*, not only a loss

```
IN  724c [Description] Teratosphaericola Quaedvl. & Crous, Persoonia…
```

**A `Description` block opening with the nomenclatural citation of a
different genus** — `Teratosphaericola`, against the treatment's
`Teratosphaeria`.  That is §12.3.31's run-on citation, and here its
consequence is not merely a buried name.

**`group_paragraphs` closes a treatment when it sees a `Nomenclature`
block.**  Because this citation is *inside* a `Description` block, **no
such block exists**, so the treatment never closed and absorbed a second
organism.

**This upgrades §12.3.31 from a content-loss class to a merge
mechanism.**  A mid-line label boundary does not just destroy one of two
labels — when the destroyed label is `Nomenclature`, it removes the only
signal that ends a treatment.  It is a concrete causal path from the
line-based architecture to the merges the operator keeps finding, and it
ties §12.3.31 to §12.3.20's under-production and §12.3.22's unbounded
extent.

#### A fourth consecutive undetected merge

`n_terms_above_5 = 1` against a threshold of 15, on a treatment holding
two different genera.  With `taxon_5bdbc707` (0), `taxon_6e02ee31` (4)
and `taxon_7d321149` (0), **the merge metric has now missed four in a
row.**  §12.3.4's length-blindness is compounded here: the second
organism contributes only 724 characters, far below any repetition
threshold.

#### And the type-designation

*"should include the material in the following Misc-exposition"* —
`BRIP 49804, culture ex-type CBS 12…`, mean line length 9.  **Tail theft
on `Type-designation`**, which §12.3.27 measured at 12.3 %, and the
thief is itself a short-line fragment.  The same mechanism, twice in one
treatment.

### 12.3.33 A dry-run of two pending changes against a clean treatment

`taxon_e9ece99e` (*Pseudochaetosphaeronema lincangensis*).  Operator:
*"looks like a poster child.  I could quibble with the last two notes.
I think we are contemplating changes which would reclassify the first as
Nomenclature, and the second as Phylogeny."*

**Fifth flawless treatment, and 5 for 5 on self-labelling** —
`Etymology.`, `Holotype.`, `Description.`, `Culture characteristics.`,
`Material examined.`, `Figure 5.`, `Notes.` all cued and all honored, no
boundary theft either side.

Because the only outstanding items are two *pending changes*, this is a
usable dry-run of both.

#### Prediction 1 — first note → `Nomenclature`: **not supported**

```
[Notes] GenBank numbers. ZHKUCC 23–0800 = ITS: OR853095, LSU: OR922336,
        SSU: OR922342, tef1-α: OR966290;
```

| test | result |
|---|---|
| gnfinder names found | **0** |
| authority string | **none** |

**§12.3.32's splitter would not touch this block.**  It fires on
taxonomic citations, and there is no name here at all.

**The distinction worth recording**: these are **sequence** accessions,
not **nomenclatural registry** identifiers.  MycoBank and Facesoffungi
numbers register the *name* and belong with `Nomenclature` — that is the
class §12.3 measured at 74.2 % mislabelled, and which the operator
flagged on `taxon_8f4ac1f5`.  GenBank/ITS/LSU accessions register
*sequences* and are voucher-linked.  **Grouping them under
`Nomenclature` would conflate two different identifier classes**, and
the change under contemplation does not do it.

#### Prediction 2 — second note → `Phylogeny`: supported by content, but not by the landed fix

```
[Notes] Notes. In the phylogenetic analyses, Pseudochaetosphaeronema
        lincangensis clusters distinctly, sister to…
```

Nine distinct phylogeny markers (`phylogenetic`, `analyses`, `sister`,
`support`, `ITS`, `LSU`, `SSU`, `phylogeny`).  **The content reading is
clearly right.**

**But commit `8c0148d` does not cause this reclassification.**  It
removes the *penalty* — before it, a `Phylogeny` label here would have
discarded all 1 053 characters (§12.3.12).  **It does not change what
the model labels.**  Relabelling requires the model to prefer the
content signal over the document's declared `Notes.` cue.

**And that is in direct tension with the strongest positive finding in
this section.**  Cue-honoring is what distinguishes the five flawless
treatments — 5 for 5, against 65 % corpus-wide.  A change that teaches
the model to override declared cues risks the mechanism that makes these
documents work.

**The lattice already resolves it, and the resolution is "neither is an
error."**  `Notes` ⊐ `Phylogeny` makes the refinement *permitted*, not
*required*; honoring the cue is correct by §12.3's measure and refining
is correct by content.  **The choice is a product decision** — a
`phylogeny` field is more useful downstream than a `notes` field for
this content — **not a correctness fix**, and it should be argued on
that basis rather than as a defect repair.

#### §12.3.32's trap, confirmed in the wild

gnfinder found **15 names** in that second `Notes` block — it is
comparative discussion naming congeners (`P. magnoliae` at 6.87,
`P. siamensis` at 6.31).  **A splitter run naively over `Notes` blocks
would shred it into fragments.**  This is the same failure mode as
§12.3.32's `Mycelium amphigenum`, in a different label, and it confirms
that the splitter must be **scoped to block types and positions where a
citation is expected** rather than applied wherever a name is found.

### 12.3.32 Two design directions for the next model, with feasibility evidence

Operator: *"two things to account for in the next model: **(1) We can
use gnfinder to locate taxonomic citations inside other blocks**… look
for nomenclature inside Description blocks and split off a Nomenclature
block if we find it.  **(2) We need an issue front-matter detector that
can extract publication information.**"*

#### 1. gnfinder as a sub-line splitter — tested, feasible, with one trap

gnfinder is installed locally (`~/bin/gnfinder`, service on
`localhost:9080`) and returns **character offsets** plus an
`AnnotNomenType` for nomenclatural acts — precisely what §12.3.31's
unrepresentable mid-line boundary needs.

Tested against the five degraded shapes this review produced:

| case | source | result |
|---|---|---|
| run-on citation + Latin description | §12.3.31 | **found**, 0-20 |
| citation broken across newlines | §12.3.15 `Boletus\nananaeceps` | **found**, spans the break |
| OCR-damaged French | §12.3.6 `Botryohyooc.hnus h~belocutosoorus` | **found**, and `SP_NOV` detected |
| German article, Latin diagnosis | §12.3.16 | **found** |
| run-together, lost newline | §12.3.27 `…Maire.Description based on…` | **found** |

**It works on exactly the material where citations are currently being
swallowed** — line-broken, OCR-damaged and newline-stripped text all
parse.

**The trap: Latin descriptive prose parses as binomials.**  In the
run-on case gnfinder returned *two* names — the real
`Asterina orthosticha` at **10.64** log-odds, and
**`Mycelium amphigenum` at 8.13**, which is Latin description opening
*"Mycelium amphigenous, thin, of brown septate hyphae…"*.

**And the false positive outscores a genuine name**: the OCR-damaged
`Botryohyooc.hnus h~belocutosoorus` scored **3.54**.  **A confidence
threshold cannot separate them** — any cut admitting the real damaged
name admits the Latin noun-phrase, and any cut excluding the noun-phrase
discards the damaged name.

**Consequence for the design.**  A naive "split wherever gnfinder finds
a name" would **fragment Latin descriptions**, which are currently among
the *working* cases (§12.3.16: Latin 53 %, at the English rate).  The
splitter needs a discriminator that is not confidence:

* `AnnotNomenType` — but three of the five real citations above are
  `NO_ANNOT`, so acts are necessary-if-present, not sufficient;
* **position** — a citation opens a treatment; a Latin noun-phrase sits
  mid-sentence;
* **a following authority string** — `Syd.`, `(J.C. Schmidt ex Fr.)
  Maire` — which `Mycelium amphigenum` lacks.

The authority test looks strongest and is independently motivated:
§12.3.15 found `binomial + authority` the discriminating signal for
`Table`-swallowed citations, at 13 % against a 67 % bare-binomial rate.
**The same feature separates both cases.**

#### 2. An issue front-matter detector unblocks four recorded items

Recorded here because the dependency is larger than it looks.  A
detector extracting publication metadata — journal, volume, year, and
per-article title and boundaries — would unblock:

| item | what it needs |
|---|---|
| §12.3.8 taxonomic-article gate | journal + title, **and** §12.3.17 found these are stripped by `_slim_ingest`, which keeps only `_id, url, pdf_url, xml_url, db_name, doi` |
| §12.3.9 / §12.3.25 article boundaries | per-article extents in whole-volume scans |
| §12.3.31 "older literature" hypothesis | a **publication date** — the era test failed for want of one |
| §12.3.17 false-negative risk | a "Report of the Botanist" is only distinguishable from an FDA leaflet **with** publication context |

**The operator's second sentence is the cheaper half**: *"Article
extraction may also have publication information that we can use."*
Running heads carry journal, volume and page on nearly every page —
`Persoonia – Volume 39, 2017`, `MYCOTAXON Vol. XIV, No. 1… January-March
1982`, both quoted verbatim in §12.3.9 and §12.3.6 — and the corpus
already labels them `Page-header` at 4 409 blocks per 300 documents.
**The publication metadata is already segmented; it has simply never
been parsed.**

That is a materially smaller task than a front-matter detector, and it
would supply the date §12.3.31 needed and the journal §12.3.8 needs,
without touching ingest.  **The volume-level front matter remains
necessary for per-article titles and boundaries**, which running heads
do not carry.

### 12.3.31 A label boundary inside a line cannot be represented at all

Operator, on `taxon_e59c0add`: *"a formatting issue that I think is
likely to be a problem for other older literature: **The taxonomic
citation is on the same line as the start of the description.  A
line-based solution can not cope with this case.**"*

**Correct, and it is categorical rather than statistical.**  Both v4
passes label **per line** (`docs/extraction_pipeline.md`: *"per-line
treatment labels"*).  A line holding a nomenclatural citation followed
by the opening of a description can receive exactly one label, so
**one of the two is destroyed by construction** — whichever loses, the
loss is not a model error and no amount of training will fix it.

The two outcomes are both observed in this review:

* labelled `Description` → the citation is buried, and
  `Nomenclature` capture drops (§12.3.21 measured 5 % of names lost);
* labelled `Nomenclature` → the description's first line is lost, which
  is §12.3.27's head-theft signature arriving from a different cause.

#### It corrects a filing in §12.3.16

`taxon_871bb4ea`'s *"Latin descriptions start with the taxonomic
citations"* was recorded there as **the same Pass-1 missed split as
§12.3.13's embedded figure captions.**  That was wrong.  An embedded
caption sits **between** lines and *is* splittable — the segmenter
simply failed to split it.  A run-on citation sits **within** a line and
is **not** splittable at the current granularity.  **Same symptom,
incompatible fixes**: one needs a better segmenter, the other needs
sub-line labelling.

#### And it unifies with the lost-newline defect

§12.3.13 found `taxon_7a36746e` had **lost its line breaks** — blocks
reading `3. Results and Discussion3.1. Identification of…` — which put
its figure-caption boundary mid-line and made the line-anchored detector
blind to it.

**Those are the same failure reached by two routes:**

| route | cause | result |
|---|---|---|
| run-on citation (here) | source typography — older literature sets the name inline | boundary inside a line |
| lost newlines (§12.3.13) | text extraction discards line structure | boundary inside a line |

**The second is the more troubling**, because it *manufactures*
unrepresentable boundaries out of documents that had representable ones.
Whatever fraction of the corpus has lost its newlines has been moved
into this class by our own pipeline.

#### Prevalence: a floor of 0.3 %, and a failed era test

Lines of 70+ characters inside `Description`, `Diagnosis` or
`Nomenclature` blocks, containing a binomial with authority followed by
40+ characters of morphological prose:

| | |
|---|---:|
| lines examined | 12 395 |
| name + description on one line | 32 — **0.3 %** |

**A floor, not an estimate.**  The pattern demands a full authority
string in a recognisable shape; run-on lines with abbreviated or
OCR-damaged authorities are missed.

**The "older literature" association could not be tested.**  Using
`ingest.doi` as an age proxy, the sample yielded **144 lines** from
DOI-bearing documents against 12 251 without — far too lopsided to
compare, and the ratio it produced is noise.  **It is not recorded.**
Testing the operator's hypothesis needs a real publication-date field,
which is not among the six keys `_slim_ingest` retains (§12.3.17).

**The architectural point does not depend on the frequency.**  A 0.3 %
floor still describes a class the current design cannot represent, and
one whose true size is unknown in exactly the material — older, scanned,
newline-damaged — where the review has found every other structural
defect.

### 12.3.30 How much weight German observations carry — and one that is not German at all

`taxon_e59c0add`, a German article.  Operator: *"**I don't know how much
weight to put on my comments given what we've already learned about
German language articles.**"*

#### The answer: do not pool them

§12.3.16 established German as a **separate regime** — `Description`
fires at 1 % against 52 % on English.  Pooling German observations with
English ones would repeat precisely the error §12.3 was written to
prevent, and would make the English pipeline look worse than it is.

**The right disposition is that German observations are high-information
for one purpose and uninformative for another.**  They are direct
evidence for the LOTE work that §12.3.26 moved *ahead* of LOTE ingest;
they say nothing about the 98 % of the corpus that is English-dominant.
Both of the operator's first two observations here — nomenclature
over-extending into German prose, and a single `Description` block
holding the taxonomic citation, the whole Latin description and the
first line of the German one — are consistent with §12.3.16's account
and are recorded as **LOTE evidence, not corpus evidence**.

#### The third observation is language-independent, and it is a judgement boundary

*"the materials-examined block really does discuss the materials
examined, though **I probably would have classified it as "biology"**
because it describes the location and host and lacks fungarium
references."*

That is a schema question, not a German one: **what separates
`Materials-examined` from `Biology` when there is no voucher?**  Measured
over 668 `Materials-examined` blocks:

| | n | |
|---|---:|---:|
| has a voucher, herbarium code or collector | 451 | **67.5 %** |
| author frontmatter — not materials at all (§12.3.2) | 21 | 3.1 % |
| **no voucher** | 196 | 29.3 % |
| …host/habitat, **no date** → reads as `Biology` | 11 | **1.6 %** |
| …host/habitat **+ date** → collection record without a voucher | 52 | 7.8 % |
| …neither | 133 | 19.9 % |

**The operator's exact test — host and location, no fungarium reference
— matches 1.6 %, about 767 blocks corpus-wide.**  A real class, and a
small one.

**It belongs with `Notes`/`Diagnosis` and
`Type-designation`/`Materials-examined` as a pair where the operator is
*choosing* rather than correcting** (§12.3).  A collection record
stripped of its voucher genuinely reads as habitat information; both
labels are defensible, and the lattice has no edge between them because
neither subsumes the other.

#### Two defects in my own measurement, recorded

* **The voucher pattern was case-sensitive**, so it missed `Coll. C. H.
  Peck` in its own first example.  Fixing it moved "has voucher" from
  58.7 % to **67.5 %** and the Biology-reading bucket from 2.4 % to
  1.6 %.  **The uncorrected figures were never recorded**; this note
  exists so the correction is not re-derived.
* **It still under-detects.**  Single-letter and spaced herbarium codes
  — `BRY C21898`, `K(M)`, `NY` — do not match `[A-Z]{2,6}[- ]?\d{3,}`.
  **So 67.5 % is a floor and 29.3 % "no voucher" is a ceiling**, and the
  1.6 % is an upper bound on the operator's class.

#### An incidental confirmation

**3.1 % of `Materials-examined` blocks are author frontmatter** —
`Department of`, `University`, an email address.  §12.3.2 measured that
absorption from the other direction (7 % of frontmatter-cued blocks land
in `Materials-examined`).  Two independent samples, consistent
magnitudes.

### 12.3.29 Reference-only descriptions — a real class, and a rare one

`taxon_e4150d1a`.  Operator: *"darn close to a poster child, but the
type designation got labeled "materials_examined"… **It's a special kind
of description block that I don't know we've cataloged yet — it's all
references to other descriptions.**"*

#### The type designation: a fourth `Type.`

```
[Materials-examined] Type. SWEDEN, Hällnäs, Västerbotten, from the galleries
                     of Acanthocinus aedilis in pine wood, A. Mathiesen-Käärik,
                     lectotype designated here…
```

The bare **`Type.`** cue again — §12.3.3, §12.3.7, §12.3.24, and now
here.  **Four instances of the same missing cue form**, each producing a
`Materials-examined` coarsening on the worst-performing cued label.  The
`Type <rank>.` fix proposed in §12.3.24 would have caught all four.

#### The new class

```
Descriptions. Mathiesen-Käärik (1950, p. 298); Mathiesen-Käärik (1951,
pp 212–215, fig. 2); Hunt (1956, pp 29–30); Griffin (1968, pp 707–708,
figs 49–52, 82); Olchowecki and Reid (1974, pp 1699–1700, Pl. XIII fig.
262); Upadhyay (1981, pp 52–54, figs 116–121); Mouton et al. (1993…)
```

**A `Description` whose entire content is bibliographic pointers to
descriptions published elsewhere.**  The operator is right that it was
not catalogued — though it has appeared once before, in `taxon_62ffeff0`
(*"Descriptions. Davidson (1971, pp 7–10, figs 2, 12…)"*), where it was
noted in passing as *"a pointer to descriptions published elsewhere"*
and not pursued.  **Two instances is a class.**

#### Measured — and it is rare

A first detector (≥2 citations, no measurement, no morphology) returned
9 blocks, **but its hits were wrong**: extrolite lists and ordinary prose
that happens to cite literature.  Recorded as discarded.

The refined test asks for **citation *coverage*** rather than count —
three or more `Author (year, pp N–M, figs …)` pointers occupying ≥45 %
of the block:

| | |
|---|---:|
| `Description` blocks examined | 2 523 |
| reference-only | **0** |
| citation-dominated (25–45 %) | 0 |
| fires on the operator's block | **yes** — 3 pointers, 52 % coverage |

**Zero in 2 523 blocks**, against a detector verified to fire on a known
positive.  That bounds the class at roughly **0.1 % of `Description`
blocks** — under ~100 corpus-wide.

**Both known instances come from the same specialist literature** —
ophiostomatoid fungi associated with bark beetles (Mathiesen-Käärik
here, Davidson 1971 in `taxon_62ffeff0`).  This looks like a **genre
convention of monographic revisions that cite prior descriptions instead
of redescribing**, not a general phenomenon.

#### Addendum: vector information has no home either

Operator, on the same treatment: *"The first notes section should have
been biology (vector)."*

**Second instance.**  The same call was made on `taxon_62ffeff0`, where
`Insect vectors. Dendroctonus sp.` carried `Notes` between two correct
`Biology` blocks (§12.3.7).  **Both come from ophiostomatoid
bark-beetle literature** — the same genre as the reference-only
descriptions above, and one in which vectoring is a standard treatment
component with its own conventional heading.

Measured over 300 documents, blocks mentioning vectoring or naming a
vector beetle genus:

| label | n | |
|---|---:|---:|
| `Misc-exposition` | 16 | 33 % |
| `Key` | 11 | 23 % |
| `Materials-and-methods` | 8 | 17 % |
| `Notes` | 6 | 12 % |
| **`Biology`** | **2** | **4 %** |
| other | 5 | 11 % |

**Vector content reaches `Biology` 4 % of the time.**

**Two caveats, both real.**  The pattern catches any *mention* of
vectoring — an isolation protocol citing beetle galleries lands in
`Materials-and-methods` legitimately — so most of these 48 are
incidental references rather than vector statements.  And **zero blocks
in the sample open with an explicit vector cue**, so the honoring rate
cannot be measured here at all; both known instances fall outside a
300-document draw.  The genre is thinly represented.

**The diagnosis is a schema gap, not a labelling error.**  `Biology` is
reached through a narrow cue set — `Habitat`, `Host`, `Distribution`
(§12.3.4) — and **`Insect vectors.` is not in it.**  A conventional
heading with no field behind it lands in the catch-all, which is exactly
the shape of §12.3.7's higher-taxon chains and §12.3.12's `Phylogeny`.

**This is a third named sub-component for Trello #407.**  §12.3.7
already recorded `Host trees.` / `Insect vectors.` / `Distribution.`
appearing as three separate headings in one document, and the operator's
*"eventually ecology"*.  **Vector is now named twice.**  It does not
widen the ticket — the `Distribution` split remains the measurable,
high-volume win — but it strengthens the recorded position that the
post-split `Biology` must stay explicitly provisional.

#### Why it matters despite being rare

**The label is correct.**  The document says `Descriptions.` and the
model honored it, so this block is a *success* by every measure in
§12.3 — cue honored, boundaries clean, no theft.

**But it contains no morphological characters at all.**  Downstream, a
treatment carrying this description looks fully populated while offering
the flagship character extraction nothing.  **It is invisible to
labelling-accuracy metrics by construction**, which is precisely why it
had to be found by reading rather than by measuring, and why the
operator found it and the memo had not.

That makes it a small member of an important family: **defects where the
label is right and the content is not what the consumer expects.**  The
front-matter descriptions of §12.3.23 are the large member of the same
family.

### 12.3.28 A poster child that survives every detector

`taxon_c7fdca00` (*Macrophomina vaccinii*).  Operator: *"is a poster
child.  Fight me!"*

**No fight available.**  Every detector developed across §12.2 and
§12.3 was run against it:

| check | result |
|---|---|
| merge metric `n_terms_above_5` | 0 |
| `synthetic_nomenclature` | false |
| cue-honoring (§12.3) | **all cues honored** |
| head boundary theft (§12.3.11) | none |
| tail boundary theft (§12.3.27) | none |
| embedded caption openers (§12.3.13) | 0 |
| lost newlines (§12.3.13) | 0 |
| front-matter position (§12.3.23) | after the taxonomy section |

```
IN    62c [Nomenclature]       Macrophomina vaccinii Y. Zhang ter & L. Zh…
IN   238c [Type-designation]   Holotype. CHINA, Fujian province, Nanping…
IN    60c [Etymology]          Etymology. from "Vaccinium", in reference…
IN  1070c [Description]        Description. Sexual stage not observed. As…
     260c [Figure-caption]     Figure 2. Macrophomina vaccinii (from ex-t…
IN   263c [Description]        Culture characteristics. Colonies on MEA a…
IN   311c [Materials-examined] Additional specimens examined. CHINA, Fuji…
IN   605c [Notes]              Note. Based on phylogenetic analysis, M. v…
    2325c [Misc-exposition]    Pathogenicity test All the three isolates…
```

Both figure captions and the trailing pathogenicity-test section are
correctly **excluded**, and `Culture characteristics.` is correctly a
second `Description` rather than a new treatment.

**4 for 4 on self-labelling.**  Every flawless treatment of round 5 is a
document that labels itself throughout.  §12.3 measured cue-honoring at
65 % corpus-wide; these are the documents where it reaches 100 %.

#### The one honest caveat, and it is not a defect

The `Note.` block is phylogenetic — *"Based on phylogenetic analysis,
M. vaccinii…"*.  By §12.3's measure the model is **right**: the document
declares `Note.` and the cue is honored.  By the lattice it is a valid
**refinement** direction (`Notes` ⊐ `Phylogeny`), so it is benign in the
scoring either way.

**It is now also an opportunity rather than a loss.**  Before commit
`8c0148d` a `Phylogeny` label here would have discarded the 605
characters (§12.3.12).  With the grouper fixed, this is exactly the
block that a `Notes` -> `Phylogeny` refinement would now place in its own
field — the same observation the operator made on `taxon_8dc658f0`,
arriving for the second time in a flawless treatment.

**A caveat on the merge metric line above**: `n_terms_above_5 = 0` is
uninformative here, since §12.3.4 established the metric is blind to
merges in short treatments.  It is recorded for completeness, not as
evidence.

### 12.3.27 Cues protect the boundary they mark — and nothing marks a block's end

`taxon_bb02abc4` (*Nectriopsis violacea*).  Operator: *"nearly a poster
child… The description is missing its first line, eaten by the
Misc-exposition.  **The three intervening blocks are correctly
identified and the description continues in the last block.**"*

**The contrast inside one treatment is the diagnostic.**

```
IN   725c [Nomenclature]     Nectriopsis violacea (J.C. Schmidt ex Fr.) Mai…
      58c [Misc-exposition]  Description based on ex-epitype culture CBS 91…   <- STOLEN HEAD
IN  2059c [Description]      morph: perithecia immersed in mycelium, becomi…
      32c [Page-header]      --- PDF Page 88 Label 88 --- 110
      10c [Misc-exposition]  Hou et al.
     296c [Figure-caption]   Fig. 41. Nectriopsis violacea (ex-epitype cult…
      32c [Page-header]      --- PDF Page 89 Label 89 --- 111
      79c [Misc-exposition]  www.studiesinmycology.org Redisposition of acr…
IN   290c [Description]      luteous at periphery, with white radial lines;…   <- RESUMES
```

The description **reassembles correctly across five intervening
blocks** — two page headers, a running head, a figure caption and a URL
line — and **loses its first line to one**.  So the failure is not an
inability to handle interruptions; §12.3's span-reconstruction success
is working here at full strength.  It is specifically a **head-boundary**
failure.

**And the stolen fragment is the cue itself.**  `Description based on
ex-epitype culture CBS 91…` carries the word that marks where the
description begins.  **The theft removes the very signal that would have
protected the boundary.**

#### Position bias, measured

The operator has now reported lost *first* lines four times
(`taxon_5180d088`, `taxon_5c661438`, `taxon_8f4ac1f5`, here) and lost
*last* lines twice.  Testing that over 14 940 content blocks:

| label | blocks | HEAD stolen | TAIL stolen | head/tail |
|---|---:|---:|---:|---:|
| `Biology` | 559 | **31.7 %** | 27.0 % | 1.17 |
| `Notes` | 2 190 | 24.4 % | 22.5 % | 1.09 |
| `Diagnosis` | 892 | 24.2 % | 21.7 % | 1.11 |
| `Description` | 2 541 | **23.7 %** | 15.9 % | **1.48** |
| `Figure-caption` | 1 937 | 13.8 % | 13.0 % | 1.06 |
| `Materials-examined` | 1 440 | 5.8 % | **17.4 %** | **0.33** |
| `Type-designation` | 488 | 4.1 % | 12.3 % | **0.33** |
| `Etymology` | 529 | **2.3 %** | **48.2 %** | **0.05** |
| **all** | **14 940** | 14.5 % | 15.5 % | 0.94 |

**There is no global position bias** — head and tail theft are within a
percentage point of each other overall, and the "descriptions lose their
first line" intuition does not generalise across labels.

**But the per-label spread is enormous and it tracks §12.3's cues
exactly:**

| | head/tail | cue honored (§12.3) |
|---|---:|---:|
| `Etymology` | 0.05 | **93 %** |
| `Materials-examined` | 0.33 | 79 % |
| `Type-designation` | 0.33 | 48 %\* |
| `Description` | 1.48 | 89 %\* |

*\*both measured on cue lists later shown incomplete — §12.3.3, §12.3.24.*

**The generalisation: a self-declaring cue protects the boundary it
marks, and only that boundary.**  `Etymology.` states where an etymology
*begins*; **nothing in the house style states where it ends**, so the
tail bleeds into whatever follows — 48.2 % of the time, the highest
single figure in the table.  Labels whose openings are not conventionally
announced (`Biology`, `Notes`, `Diagnosis`, and `Description` when its
cue is absent or stolen) lose their heads instead.

**This is a structural asymmetry in the source material, not a model
defect**, and it predicts where repair effort pays: head boundaries are
recoverable from cues that already exist, while **tail boundaries have
no signal at all** and need the continuity test of §12.3.11 — which is
precisely the detector §12.3.19 measured at 100 % precision.

#### One caution on the `Description` row

`Description`'s 1.48 sits between the two groups because it is
*sometimes* cued (`Description.`) and sometimes not.  **This treatment
shows the mechanism that moves it**: when the cue-bearing fragment is
itself stolen, a cued description becomes an uncued one, and the ratio
reflects a mixture of the two regimes rather than a property of the
label.

### 12.3.26 The 2 % is curation, and the Latin detector is the suffix channel

Operator, correcting §12.3.25: *"I have **deliberately left out major
journals in LOTE** (other than Sydowia).  These numbers will be much
higher in the future.  Also, **all the early literature in mycology is
entirely in Latin** and I have not included any of that.  I suspect that
parts of the current system implements an **implicit Latin detector**."*

#### The prioritisation in §12.3.25 is withdrawn

That section measured LOTE at 2 % of documents and concluded it should
rank below boundary theft and front-matter harvesting.  **That reasoning
is invalid**: the 2 % is the result of a deliberate exclusion, and the
exclusion is going to be reversed.

**The correct inference runs the other way.**  Ingesting LOTE material
into the pipeline as it stands would produce near-zero extraction —
German descriptions are detected at **1 %** (§12.3.16) — so the ingest
cost would be spent to produce unextractable documents, and the corpus
would gain a stratum that has to be reprocessed later.  **LOTE handling
should therefore precede LOTE ingest, not follow it.**  It is a
prerequisite for a planned corpus expansion rather than a repair of the
current one.

#### The implicit Latin detector, located

**Confirmed, and it is the suffix channel.**
`skol_classifier/feature_extraction.py:36` declares
`suffix_vocab_size: int = 200` — a TF-IDF over word **suffixes**,
learned from training data, 200 features wide, alongside 800 word
features.

That is precisely a language-morphology detector:

* Latin descriptive vocabulary carries a distinctive suffix inventory —
  *-atus*, *-ata*, *-oideus*, *-formis*, *-inus*, *-escens*, *-icus*;
* English scientific Latinate terms **share it** — *-oid*, *-ate*,
  *-ose*, *-escent*, *-iform*;
* German morphological adjectives do **not** — *-ig*, *-lich*,
  *-förmig*, *-artig*.

**This explains §12.3.16's numbers precisely.**  Latin 53 % and English
52 % are one detector firing on a shared suffix inventory; German 1 % is
that inventory being absent; French 21 % is partial overlap.  §12.3.16
attributed the effect to "Latinate morphological vocabulary" from the
outside; this identifies the actual channel.

#### And there is dormant *explicit* Latin machinery

Built, tested, and never wired into the v4 pipeline:

| asset | state |
|---|---|
| `paragraph.py:159-189` — `latinate` reinterpretation rewriting Latin-suffixed words to a ` PLATINATE ` token | **`--reinterpret latinate` appears only in `paragraph_test.py`** — no production path enables it |
| `data/botanical_latin_wordlist.txt` — 5 679 entries | read only by `bin/corpus_vocabulary.py` and `treatments_to_structured/ocr_damage.py` — **OCR damage detection, not classification** |
| `data/systematic_names_wordlist.txt` — 849 entries | read only by its own test |
| `data/dcc/greek-core-wordlist.txt` — 873 entries | **nothing reads it** |

So the system contains a deliberate Latin feature that is switched off,
and a learned one that is doing the work.

#### What this predicts for the early Latin literature

**Favourably.**  The suffix channel is language-appropriate for Latin
and already performs at the English rate, so wholly-Latin early
literature should not fail the way German does.

**Two caveats worth stating before relying on that.**  The 53 % was
measured on Latin **diagnoses embedded in modern papers** — short,
formulaic, and stereotyped.  Early literature is **narrative** Latin,
with different sentence structure and a wider vocabulary, and it has not
been tested.  And that material carries the OCR and layout problems of
§12.3.6, §12.3.15 and §12.3.22, which are independent of language and
were the dominant failure in every historical document reviewed in round
5.

#### The concrete lever for LOTE

The mechanism is already in place and is **per-language by
construction**: a suffix vocabulary learned from German training data
would give German what Latin already has.  That reframes LOTE support
from "a new capability" to **"training data in the target language, plus
possibly a larger `suffix_vocab_size`"** — and makes the dormant
`PLATINATE` machinery worth revisiting, since an explicit
language-morphology feature would not need to be re-learned per corpus.

### 12.3.24 Type-designation vocabulary is rank-dependent, and only the species-level forms are recognised

`taxon_a21ae068` (*Borikeniomycota* — an article erecting a new phylum
and class).  Operator: *"The phylum is defined phylogenetically.  **The
type designation for the Phylum was misclassed as notes despite very
clear labeling.**  The class description is correctly not in this
treatment."*

**Confirmed on every point:**

```
description       -            <- correct; the phylum is defined phylogenetically
diagnosis      384c  "Diagnosis.Distinguishable from other fungi based on a di…"
notes          518c  "Type class.Borikeniomycetes Tedersoo.\n\nNotes.Recognized…"
type_designation  -                    ^^^^^^^^^^^ the type designation, inside notes
```

**The absent `description` is correct behaviour, not a defect.**  A
phylum erected on phylogenetic grounds has no morphology to describe,
and §12.3.16 explains why nothing fired: `Description` responds to
Latinate *morphological* register, which a phylogenetic circumscription
does not contain.  Likewise the operator's *"class description is
correctly not in this treatment"* — the grouper kept two ranks apart,
which is worth recording as a **success** given §5.5's multi-rank
problems elsewhere.

#### A third missing cue form, on the worst-performing label

§12.3's cue for `Type-designation` is `Type material|Holotype|Typus`.
It has now been shown to miss:

| form | first seen |
|---|---|
| bare `Type.` | §12.3.3 (`taxon_57e92419`) |
| `Type.` again | §12.3.7 (`taxon_62ffeff0`) |
| **`Type class.`** | here |

And by obvious extension `Type genus.`, `Type family.`, `Type species.`
— **the higher-rank forms as a family.**  `Type-designation` is already
the worst-performing cued label at **48 %** honored, and that figure is
measured on a cue list that excludes at least three real forms.  **Its
n = 378 is an undercount and its 48 % rests on a biased subset**, in the
direction of flattering the result.

#### The rank gradient

Field presence by rank over 2 600 named treatments (phylum, subphylum
and class had n < 25 and are not reported):

| rank | n | `description` | `diagnosis` | `notes` | `type_designation` | `biology` |
|---|---:|---:|---:|---:|---:|---:|
| order | 41 | 44 % | 34 % | 54 % | **10 %** | 15 % |
| family | 56 | 50 % | 25 % | 57 % | 32 % | 12 % |
| genus/species | 2 474 | **69 %** | 29 % | 53 % | 31 % | 29 % |

Two gradients, and they are different phenomena:

* **`description` falls with rank** — 69 % → 50 % → 44 %.  Partly
  correct (higher taxa are diagnosed, not described) and partly
  §12.3.16 (no morphological register to detect).  Directionally clear,
  but n = 41 and n = 56 are thin.
* **`type_designation` collapses at order level** — 31 % and 32 % at
  genus/species and family, but **10 %** at order.  That is not
  taxonomic practice: orders do have type families.  **It is the missing
  cue vocabulary**, and `Type class.` landing in `notes` here is the
  mechanism caught in the act.

**The fix is small and well-defined**: recognise `Type <rank>.` as a
`Type-designation` opener alongside `Holotype`/`Typus`.  It is a T6
input, and it should be made before `Type-designation`'s honor rate is
quoted again — the current 48 % measures a cue list, not the model.

### 12.3.23 Treatments built from a taxonomic paper's front matter

`taxon_99094ada`.  Operator: *"appears to be the introductory material
on a taxonomic article.  The description certainly is taxonomic
description vocabulary, but two lines earlier in the preceding
Misc-exposition we see that **we are in the Introduction to the paper**…
In that last Misc-exposition we see that **we are only at the Materials
and methods section**."*

**This is a new class, distinct from §12.3.8.**  There the *article* was
not taxonomic (an FDA leaflet).  Here the article is a perfectly good
taxonomic paper and the **wrong part of it** was harvested.  This
treatment begins at **char 0** of a document whose `Results` heading is
at char 21 076.

#### Measured

Locating `Introduction`, `Materials and methods` and
`Results`/`Taxonomy` headings, then asking where each treatment's first
span begins:

| | n | |
|---|---:|---:|
| **before `Materials and methods`** | 40 | **18 %** |
| between M&M and `Results`/`Taxonomy` | 70 | 32 % |
| after `Results`/`Taxonomy` (expected) | 110 | 50 % |

**The 18 % is the hard number** — a treatment starting before the
methods section sits in the title, abstract or introduction, and cannot
be a real treatment.  **The 50 % should be treated as softer**: some
papers place taxonomy without a `Results` heading, and `Discussion`
appears in the pattern, so the middle band mixes genuine and spurious
cases.

**Caveats that run in the conservative direction.**  Only the *first*
occurrence of each heading is used, so in a whole-volume scan (§12.3.9)
every treatment after article one is scored "after Results" regardless
of where it truly sits.  **The measurement therefore understates front
matter harvesting on exactly the material where it is likeliest.**

#### Why the introduction reads as a description — §12.3.16 already answered this

§12.3.16 established that `Description` fires on **Latinate
morphological register**, not on being a description.  **A taxonomic
paper's introduction is written in precisely that register** — it
discusses conidiophores, ascospores, septation and dimensions while
reviewing prior work.  The operator's own phrasing captures it exactly:
*"the description certainly is taxonomic description vocabulary."*

**It is the same root cause as the FDA tablet.**  Register, not context.
The model has no representation of *where in a paper it is*, so text
that reads like a description is labelled one, whether it appears in a
protologue, an introduction, or a drug leaflet.

**And it is a second argument for §12.3.9's structural level.**  A
document-level gate cannot help — the document passes.  What is missing
is **intra-article position**: knowing that char 0-21 076 is front
matter would remove this class outright, and the headings needed to know
it are already detectable, since this measurement used them.

#### The operator's other observations

* *"The biology clauses… should be one block — the intervening
  Misc-exposition ate the linking text."*  This is §12.3.11's
  **bridges-both** case, the 2.5 % high-confidence core where a sentence
  runs into the interloper and out of it.
* *"The second biology block lost its last line to the next
  Misc-exposition."*  §12.3.11, head-theft variant.
* *"The materials-examined block was mislabeled."*  §12.3.2.

### 12.3.22 `MISC_GAP_LIMIT` does not bound treatment extent

`taxon_910039ca`.  Operator: *"describes an order, and maybe a family,
and a genus.  I notice that **there are 50 lines between the Order
nomenclature and the first description** — but that description does
appear to be correct… There is another large gap to the final
description which appears to be a genus."*

`Treatment.MISC_GAP_LIMIT = 4` reads as a bound on how far a treatment
may stretch.  **It is not one.**  The counter increments on
`Misc-exposition` only, and the label set divides three ways:

| behaviour | labels |
|---|---|
| **resets** the counter (and is stored as a section) | `Description`, `Diagnosis`, `Etymology`, `Biology`, `Notes`, `Type-designation`, `Materials-examined`, `Figure-caption`, `Key`, `Phylogeny`, `Distribution` |
| **increments** it | `Misc-exposition` |
| **transparent** — neither | `Table`, `Page-header`, `Bibliography`, `Index`, `ToC-entry`, `Materials-and-methods`, `New-combinations` |

*(`Nomenclature` is handled explicitly by the state machine and is in
none of these three.)*

#### Measured gap between a treatment's `Nomenclature` and its first `Description`

| | blocks |
|---|---:|
| median | 4 |
| p75 | 11 |
| p90 | **39** |
| p99 | **2 345** |
| max | **7 449** |

**27 % of treatments have a gap of 10 blocks or more.**  A limit of 4
coexisting with a p99 of 2 345 is not a limit.

What fills those large gaps — 28 466 blocks:

| label | share | effect on the counter |
|---|---:|---|
| `Misc-exposition` | 40 % | increments |
| `Table` | 16 % | **transparent** |
| `Page-header` | 15 % | **transparent** |
| `Description` | 5 % | **resets** |
| `Figure-caption` | 4 % | **resets** |
| `Key` | 4 % | **resets** |
| `Bibliography` | 3 % | **transparent** |
| `Materials-examined` | 2 % | **resets** |

**37 % of the blocking material is transparent to the counter**, and a
further ~15 % actively *resets* it.  So the extreme tail is not one long
run of `Misc-exposition` — it is `Misc-exposition` **punctuated** by
material that either exerts no pressure or cancels what has accumulated.

#### The compounding link to rogue `Key` and `Table`

`Key` is a **section** label, so it resets the counter *and* is stored
as treatment content.  §12.3.6 showed rogue `Key` absorbs descriptive
prose in whole documents.  **Those two facts multiply**: in a
rogue-`Key` document, the mislabelled prose does not merely go to the
wrong field — it **holds the treatment open indefinitely**, resetting the
gap every time it appears.

`Table` is the mirror case.  §12.3.15 showed it swallows short-line
citations and prose in OCR'd multi-column scans; being transparent, it
lets a treatment stretch across arbitrarily much of that material
without ever registering as a gap.

**So the same two labels that §12.3.6 and §12.3.15 identified as
content-blind are also the ones defeating the only mechanism that limits
treatment extent.**

#### A note on units, and on this treatment

The operator counted **lines**; this measurement counts **blocks**.  For
`taxon_910039ca` the nomenclature-to-first-description gap is only two
blocks, one of which is a large `Misc-exposition` — consistent with ~50
lines.  **Block counts understate what a reader experiences**, and the
p99 above should be read with that in mind.

The operator's remaining observations are recorded classes: an order,
family and genus in one treatment (§5.5's multi-rank problem, and the
fourth route to it); `Notes` that *"look suspiciously like a diagnosis"*
(`Notes` ⊐ `Diagnosis`); and a second description that *"should have
continued into the Table and Misc-exposition that follows"* — §12.3.15
and §12.3.11 compounded.

### 12.3.21 Nomenclature blocks are captured 95 % of the time — so the yield problem is about *sections*

`taxon_8f4ac1f5` (Tian, Hyde & Maharachchikumbura — a large multi-family
paper, 74 treatments).  Operator, among eight observations: *"**Weirdly,
the next block is correctly labeled Nomenclature but did not get pulled
into the nomenclature.**"*

**Traced, and it is an outright loss rather than a boundary choice:**

```
IN    59c [Nomenclature]     Dictyosporium Corda, Weitenweber's Beitr. Nat. 1: 87…
     584c [Misc-exposition]  MycoBank number: MB8001; Facesoffungi number: FoF0…
IN    69c [Nomenclature]     Dictyosporium cycadicola W.H. Tian, K.D. Hyde & Ma…
      64c [Nomenclature]     Fig. 5  MycoBank number: MB854679; Facesoffungi…   <- LOST
IN    22c [Type-designation] Holotype – HKAS 134909
IN   646c [Description]      …
```

The block is absent from this treatment's `nomenclature_spans` **and
did not start a sibling** — the next treatment begins at char 34 597,
well past it.  The blocks *after* it belong to this treatment, so the
grouper did not close and reopen.  **The paragraph was simply skipped.**
The mechanism is not established; it warrants a targeted trace rather
than a guess.

#### How often does this happen

Applying §12.3.14's conservation audit to `Nomenclature`, over 97
documents holding at least three treatments:

| | |
|---|---:|
| `Nomenclature` blocks in the `.ann` | 3 083 |
| captured into some treatment | 2 938 — **95 %** |
| **lost** | 145 — **5 %** |

| per-document capture | |
|---|---:|
| p10 | 67 % |
| p25 | 97 % |
| median | **100 %** |
| p90 | 100 % |

**The median document loses no names at all.**  Loss is real but modest
and concentrated in a tail.  This document sits at **90 %** — ten names
lost of 101.

#### The contrast with §12.3.20 is the finding

| | |
|---|---:|
| `Nomenclature` blocks captured | **95 %** |
| documents yielding ~1 treatment per name (§12.3.20) | **38 %** |

**Names are found.  Treatments are not built around them.**  That
sharpens §12.3.20's hypothesis considerably: since the grouper needs
`has_nomenclature() and has_section()` and the nomenclature half is
succeeding 95 % of the time, **the failure is overwhelmingly on the
section side** — prose that should have become `Description` or
`Diagnosis` carrying `Table`, `Key`, `Misc-exposition` or `Bibliography`
instead (§12.3.16).

This does not prove the two populations coincide, but it removes the
competing explanation: **name detection is not the bottleneck.**

#### The operator's other observations

Six of the eight are instances of recorded classes — identifiers and a
Notes section absorbed into `Misc-exposition` (§12.3.11); a clearly
cued `Etymology` labelled `Misc-exposition` (§12.3, the 7 % that miss);
a well-formed `Material examined` block likewise; `Notes` that should be
`Phylogeny` (§12.3's lattice, and §12.3.12's field); two `Notes` blocks
that are one diagnosis split by a correctly-identified page number
(§12.2); and a `Figure-caption` losing its tail across a page break
(§12.2 again).

One is worth quoting: on the description losing its first line, *"it
looks like **proper sentence identification** would fix this particular
instance."*  **That is exactly the mechanism §12.3.11's detector uses**,
arrived at independently — and §12.3.19 measured that detector at 100 %
precision against operator judgement.  Operator intuition and the
measurement have converged on the same repair.

### 12.3.20 Treatment yield per `Nomenclature` block — bimodal, and 43 % under-produce

`taxon_8eab0272`.  Operator: *"an older article with descriptions of
many species… **The boundaries are all messed up.**  The description
blocks seem to be partial blocks of descriptions, and the biology blocks
actually contain biology information.  Everything else seems pretty
random."*

*"Boundaries are all messed up"* is measurable as a ratio: **how many
treatments did a document yield per `Nomenclature` block it contains?**
One apiece is the intended behaviour.

#### A measurement error, recorded

The first attempt built the treatment counts with
`tp.view('_all_docs', include_docs=True, limit=20000)` against a
**81 527-document** database.  It reported a median ratio of **0.24** —
which is `20000 / 81527`.  **The statistic measured the sampling
fraction, not the corpus.**  Caught only because that coincidence is
recognisable; recorded because a plausible-looking ratio with an
off-by-a-limit cause would otherwise have entered the memo as fact.

#### The real distribution

Full scan: 81 527 treatments over 17 645 source documents.  Over 60
sampled documents holding at least three `Nomenclature` blocks:

| | ratio |
|---|---:|
| p10 | **0.00** |
| p25 | **0.03** |
| median | **1.00** |
| p75 | 1.05 |
| p90 | 1.33 |

| | docs | |
|---|---:|---:|
| under-producing (< 0.75) | 26 | **43 %** |
| about right (0.75-1.25) | 23 | 38 % |
| over-producing (> 1.25) | 11 | 18 % |

**The median document behaves exactly as designed — one treatment per
name.**  The distribution around it does not: it is **bimodal**, with a
large tail producing almost nothing.  **p25 = 0.03 means a quarter of
these documents yield roughly three treatments per hundred nomenclatural
headings**, and p10 = 0.00 means some yield none at all *despite having
names detected*.

This document sits at **0.55** (37th percentile), 18 treatments from 33
names, six of them synthetic.

#### A mechanism, offered as hypothesis not measurement

`group_paragraphs` yields a treatment only when
`has_nomenclature() and has_section()` both hold.  A document can
therefore have its names detected and still produce nothing, if the
prose that should have become sections was labelled something that is
not a section.

**§12.3.16 showed exactly that happening**: when the content signal
fails, `Table`, `Key`, `Misc-exposition` and `Bibliography` absorb the
descriptive text.  **This document is a case in point** — 77 % of its
675 blocks carry those four labels (`Misc-exposition` 267, `Table` 116,
`Page-header` 87, `Bibliography` 47), against **31** `Description`
blocks for an article describing many species.

So the under-producing tail and the layout-label-dominated documents may
be **the same population**.  That is a testable claim and it has not been
tested; recorded as the next measurement rather than as a finding.

#### The operator's partial-success reading

*"the biology blocks actually contain biology information… everything
else seems pretty random"* matches §12.3.16's account.  `Biology` has
strong, short lexical cues — `Habitat`, `Host`, `Distribution` — which
survive where longer-range descriptive structure does not.  **The labels
that keep working under degradation are the ones anchored to a small
fixed vocabulary**, and the ones that fail are those needing the text to
read as a particular register.

### 12.3.19 Calibrating boundary theft against the operator's own verdicts

`taxon_8e97ffe3`.  Operator: *"Other than the first Misc-exposition
eating the start of the description, this nearly worked as designed."*

Round 5's review has by now produced **eight treatments the operator
identified as `Misc-exposition` boundary theft and three called
flawless** — enough to calibrate §12.3.11's detector against human
judgement, which that section explicitly recorded as not yet done.

| treatment | operator | detector | |
|---|---|---|---|
| `taxon_8e97ffe3` | theft | **THEFT** | this one |
| `taxon_0ccf38da` | theft | **THEFT** | identifiers, GenBank numbers |
| `taxon_47c3b37d` | theft | **THEFT** | *"Misc-exposition stealing"* |
| `taxon_5c661438` | theft | **THEFT** | two blocks |
| `taxon_6e02ee31` | theft | **THEFT** | two lines of nomenclature |
| `taxon_0b9a9bfe` | theft | clean | **miss** |
| `taxon_5180d088` | theft | clean | **miss** |
| `taxon_57698832` | theft | clean | **miss** |
| `taxon_3b7a80bc` | clean | clean | flawless |
| `taxon_53dd1485` | clean | clean | flawless |
| `taxon_8dc658f0` | clean | clean | flawless |

**Precision 100 %, recall 62 %.**

#### The caution in §12.3.11 ran the wrong way

That section warned the 25.3 % might over-report, since one-sided
continuation has innocent causes.  **It did not fire once on the three
treatments the operator called perfect** — the hardest available
negatives — and every firing was confirmed.  The detector's problem is
the opposite one.

**All three misses have identifiable causes, and two are already covered
elsewhere:**

* `taxon_5180d088` — the thieving `Misc-exposition` is separated from
  the `Description` by a `Page-header` **and** a `Key`-labelled page
  number.  The detector inspects **immediate** neighbours, so
  intervening furniture defeats it.  **This is §12.2's page-break class,
  which measures exactly that configuration** — the two detectors are
  complementary, and neither alone covers the field.
* `taxon_57698832` — author frontmatter absorbed into
  `Materials-examined`, which is §12.3.2's class, not continuation theft
  at all.
* `taxon_0b9a9bfe` — *"the Etymology clause disappeared into the first
  Misc-exposition"*.  A complete clause ending in a full stop leaves **no
  continuation signal**.  This is the genuine residual: theft of
  well-formed sentences is invisible to a continuity test, and no
  refinement of that test will reach it.

#### The methodological point

**Treatment-by-treatment review is a validation set, not only a source
of findings.**  Eleven operator verdicts converted a number this memo had
flagged as unusable into a calibrated one, at zero additional cost —
the judgements had already been made for other reasons.

The obvious caveat: **n = 11, of which 5 positive firings.**  A 100 %
precision on five cases is consistent with a true rate well below that,
and these treatments were selected by the operator's attention rather
than at random, so they are not a sample of the detector's corpus-wide
firings. **What is established is that the detector does not fire on
clean treatments, not that every one of its 4 366 firings is a defect.**

### 12.3.18 The third flawless treatment — and a "wrong" label that saved the data

`taxon_8dc658f0` (*Aquapteridospora linzhiensis*).  Operator: *"such a
relief… this beautiful example of a perfect treatment extraction.  OK,
if I were really picky I could say that the notes block should be
phylogeny."*

**Every block is self-labelled** — `Etymology.`, `Holotype.`,
`Description.`, `Figure 2.`, `Culture characteristics.`, `Material
examined.`, `Notes.` — and every cue is honored.

**That is 5 for 5** (with `taxon_c7fdca00` §12.3.28 and `taxon_e9ece99e`
§12.3.33).  Every flawless treatment of round 5 is a document that
labels itself throughout.  §12.3 measured cue-honoring at 65 % overall;
these are the documents where it reaches 100 %, and they are the
documents that come out clean.  **On the evidence so far, complete
self-labelling is not merely correlated with flawless extraction — it is
the only condition under which it has been observed.**

#### The picky observation is a live instance of §12.3.12

The block is **cued `Notes.`** and its content is phylogenetic —
`Notes.Phylogenetic analyses show that Aquapteridospora…`, carrying six
distinct phylogeny markers.  So:

* by §12.3's measure the model was **right**: it honored the document's
  own declared label;
* by content the operator is **right**: it is `Phylogeny`;
* and under the grouper as it stood before commit `8c0148d`, labelling
  it `Phylogeny` **would have deleted all 667 characters.**

**The model's "error" is what preserved the data.**  That is §12.3.12's
perverse incentive caught in the act, in a treatment the operator rates
as perfect.  The fix removes the incentive; the case is kept as its
clearest illustration.

#### Both phylogeny scopes appear in this one treatment

§12.3.12 left open whether `Phylogeny` is article-scoped or
treatment-scoped.  Here it is visibly **both**:

| block | label | scope |
|---|---|---|
| 1 528 c `Phylogenetic analysesThe concatenated sequence…` | `Misc-exposition` | **article** — the paper's analysis section |
| 667 c `Notes.Phylogenetic analyses show that Aquapteridospora…` | `Notes` | **treatment** — this taxon's placement |

The article-level section is a third again larger than the
treatment-level one and carries a different label.  **Any `phylogeny`
field must distinguish them**, or the paper's shared analysis will be
copied into every sibling treatment.  *(Note also `analysesThe` — the
lost-newline defect of §12.3.13, in a document that is otherwise
perfect.)*

### 12.3.17 A hard case for the §12.3.8 gate — and the gate cannot see the field it needs

`taxon_88431ff4`.  Operator: *"is not a taxonomic article."*

The document opens:

```
HAyoO^ ^iy V / ie^ REPORT OF THE BOTANIST-
Hon. David Murray, LL. D., Secretary of the Board of Regents of the
Universitij : Sir — Since the date of my last report, specimens…
```

A **19th-century State Botanist's annual report** to a Board of
Regents — administratively framed, OCR-damaged, and correctly flagged
as not an article in the modern sense.  `Nomen ignotum`,
`synthetic_nomenclature: true`, no title.

**But it carries 27 `Description` blocks out of 392**, and the genre is
not incidental: New York State Botanist reports are a **canonical source
of North American fungal taxonomy**, describing hundreds of new species
inside exactly this administrative wrapper.

#### This is a false-negative risk for the document-level gate

§12.3.8 recommended a document-level *"is this a taxonomic article"*
gate, on the strength of an FDA drug leaflet.  §12.3.9 already
qualified it for whole-volume scans.  **This adds a second
qualification, and a sharper one:** the memo's own first-cut signal was
*journal + title keyword*, and a title of "Report of the Botanist"
matches nothing taxonomic while the document is genuinely full of new
taxa.

**A keyword gate would discard high-value historical material.**  The
FDA leaflet and this report look alike from the outside — no
taxonomic title, synthetic nomenclature, non-article framing — and are
opposite cases.  Whatever the gate keys on, it cannot be the title
alone.

#### And the gate cannot currently be evaluated from the treatments DB

`treatment.py:95` projects the source `ingest` down to
`_ESSENTIAL_INGEST_KEYS = {_id, url, pdf_url, xml_url, db_name, doi}` —
**no `title`, no `journal`.**  Measured over 1 500 description-bearing
treatments: **1 500 of 1 500 have no title**, not because the sources
lack one but because it is slimmed away at assembly.

So any title- or journal-based gate needs a **join back to `skol_dev`
via `ingest._id`**, which is a per-document lookup rather than a field
read.  That is a real implementation cost for the §12.3.8 proposal and
should be priced before the gate is scoped.  *(The `_slim_ingest`
docstring says "four keys" while the constant lists six — harmless, but
noted since it was read for this.)*

#### One number worth keeping

Of 1 500 treatments **that have a description** — i.e. p1-like, the
annotatable population — **22 % carry `synthetic_nomenclature`.**  That
is the population being sampled for annotation rounds, so roughly one
treatment in five drawn has an invented boundary rather than a real
nomenclatural heading.

### 12.3.16 `Description` does not survive translation — German scores 1 %

`taxon_871bb4ea` (Sydow, *Phyllachora*, a German-language paper with
Latin diagnoses).  Operator: *"The German description blocks were
thoroughly misclassified — **description identification does not
generalize well to German.**"*

**Confirmed, and the size of the effect is not a degradation but a
collapse.**  Holding the descriptive register constant — every block
carries a dimensioned measurement, is over 200 characters, and is
language-classified by function words:

| language | blocks | -> `Description` |
|---|---:|---:|
| English | 1 150 | **52 %** |
| **Latin** | 55 | **53 %** |
| French | 28 | 21 % |
| **German** | **396** | **1 %** — three blocks |

Where the 396 German blocks actually go:

| label | n | |
|---|---:|---:|
| `Table` | 163 | **41 %** |
| `Misc-exposition` | 99 | 25 % |
| `Key` | 55 | 14 % |
| `Bibliography` | 50 | 13 % |
| `Description` | **3** | **1 %** |

#### This corrects §12.3.8, and explains why it read the way it did

§12.3.8 concluded that `Description` is a **register** detector rather
than a vocabulary detector, on the evidence that it labelled a drug
tablet correctly.  **That conclusion was measured with an English
morphological-adjective lexicon**, so what it actually established was
"not keyed to *fungal* vocabulary" — not "not keyed to vocabulary".

**Latin at 53 % is the tell.**  Latin diagnoses share their descriptive
roots with English scientific prose — *hyalinus/hyaline*,
*ellipsoideus/ellipsoid*, *cylindricus/cylindric*, *fuscus/fuscous*.
German uses native Germanic terms instead — *glatt*, *braun*,
*eiförmig*, *walzenförmig* — and scores 1 %.

**So the register is real but it is lexical: it is Latinate
morphological vocabulary.**  That reconciles every observation:

* it transfers to a **purple, oval, biconvex tablet** and to an
  **amoeba** (§12.3.13), because both are described in Latinate English;
* it transfers to **Latin**, by cognacy;
* it does **not** transfer to German, and only partly to French.

§12.3.8's answer to *"will description generalize to other taxa?"*
stands — **across taxa, yes; across languages, no.**

#### Where the German content goes is the same story again

`Table` 41 %, `Key` 14 %, `Misc-exposition` 25 % — **the layout labels
absorb it.**  §12.3.15 showed `Table` tracks short mean line length and
§12.3.6 showed rogue `Key` tracks long couplet-free blocks; neither
reads content.  **When the content signal fails, the typographic labels
are all that remain**, so the text is partitioned by page geometry
alone.

This is the unifying account for three separate cases now: French
rogue-`Key` (§12.3.6), the francophone whole-volume scan (§12.3.10), and
German here.  **Non-Latinate or damaged text is not merely labelled
worse — it is labelled by a different mechanism entirely.**

#### Scale, and a caveat on it

German measured blocks appeared in **13 of 680 documents** scanned.  So
this is a small stratum — but within it, extraction is close to total
loss, and the 1 % is measured on 396 blocks rather than a handful.
**The stratum size is the uncertain number here, not the failure rate.**

#### The operator's other four observations

* **"Both of the Latin descriptions start with the taxonomic citations,
  so we appear to have missed them with the Nomenclature detector."** —
  filed here as a Pass-1 missed split, **which §12.3.31 corrects**: the
  boundary falls *inside a line*, so it is not a missed split but an
  **unrepresentable** one.
* **"Phyllachora leptasca Syd. nov. spec. starts in the middle of a
  Table block."** — §12.3.15 exactly: a short-line nomenclatural citation
  swallowed by `Table`, here with a language switch at the same point.
* **"The Latin description had its middle pulled out as a
  Misc-exposition followed by a Table."** — §12.3.11 boundary theft
  compounded by the `Table` mechanism.
* **"I have no idea how that block of German description got classified
  as a type_designation."** — `Type-designation` takes 2 % of German
  measured blocks.  With the content signal absent, the assignment is
  close to arbitrary; there is no semantic explanation to find.

### 12.3.15 `Table` is a short-line detector — confirming the operator's mechanism

`taxon_7d321149` (Cooke & Massee material, *Grevillea* 18 — an old
two-column Australian bolete paper).  Operator: *"appears to be
fragments of 3 different species glued together… The nomenclature blocks
for the other two species were consumed by Table blocks.  I observe that
both species taxonomic citations are fairly short lines, **which may be
what the Table recognizer is latching on to.**"*

**Confirmed, and the proof is inside the treatment:**

```
121c  mean-line 14c  [Table]  Boletus | ananaeceps | Berk. Linn., Soc. Lond. | …
189c  mean-line 26c  [Table]  With | this | scanty documentation, it seems to be…
```

The first is a **nomenclatural citation** line-wrapped into fragments;
the second is **ordinary prose** line-wrapped the same way.  Neither is
a table.  Both carry `Table`.

#### Mean line length orders the whole label set

| label | blocks | median mean-line |
|---|---:|---:|
| **`Table`** | 6 284 | **14 c** |
| `Index` | 81 | 18 c |
| `Page-header` | 4 409 | 28 c |
| `Misc-exposition` | 9 566 | 38 c |
| `Key` | 1 176 | 42 c |
| **`Nomenclature`** | 800 | **46 c** |
| `Bibliography` | 1 501 | 61 c |
| `Materials-examined` | 821 | 71 c |
| `Description` | 1 420 | 72 c |
| `Diagnosis` | 458 | 75 c |
| `Notes` | 990 | 81 c |

**`Table` sits 24 characters below the next label and 32 below
`Nomenclature`.**  Short-line-ness is not merely *correlated* with
`Table` — it is nearly the whole signal, and `Nomenclature` is the
content label closest to it.

#### What is actually inside `Table` blocks

| | n | |
|---|---:|---:|
| all `Table` blocks | 6 284 | 100 % |
| contains a binomial | 4 182 | **67 %** |
| binomial **+ authority string** | 828 | 13 % |
| contains a nomenclatural act (`sp. nov.` etc.) | 45 | 1 % |

The 67 % is suggestive but not decisive — genuine taxonomic tables list
binomials.  **The 13 % carrying a binomial *and* an authority is the
harder number**: an authority is a mark of *citation*, not of tabulation.
~3 100 `Table` blocks corpus-wide carry an outright nomenclatural act.

#### The general shape: layout labels selected by surface form

This is **§12.3.6's rogue `Key` with a different feature.**  Both `Table`
and `Key` are layout labels with no semantic handle, and both latch onto
a typographic property:

| label | feature it tracks | what that swallows |
|---|---|---|
| `Table` | **short** mean line length | column-wrapped citations and prose |
| `Key` | long, couplet-free blocks | descriptive prose (§12.3.6) |

**Neither is reading content.**  So the failure is systematic in exactly
the material where line geometry is unreliable — OCR of multi-column
scans, which is where both of these cases come from.  A repair that
gives these labels *any* content signal would move both at once.

#### And a third consecutive undetected merge

`n_terms_above_5 = 0` against a threshold of 15, on a treatment the
operator reads as **three** species fused.  With `taxon_5bdbc707` (0) and
`taxon_6e02ee31` (4), the merge metric has now missed three merges in a
row, all in short treatments.  §12.3.4's length-blindness is not an edge
case; **on this evidence the metric does not detect merges in the size
range where the operator is finding them.**

### 12.3.14 "Did our extraction remove it?" — a conservation audit says no

`taxon_7b742390` (Visagie et al., *Fungal Systematics and Evolution*, a
multi-species paper).  Operator: *"an article that should have a LOT of
descriptions as indicated in the Abstract… The intermediate blocks end
abruptly with "species concept for most fungal groups has led", a clear
fragment… I can't tell if it was missing in the original source.  It
seems more likely our extraction process removed it."*

**It did not.  The accounting closes exactly:**

| | |
|---|---:|
| `Description` blocks in the `.ann` | 29 blocks, **25 323 chars** |
| captured across all 34 sibling treatments | **25 807 chars** |

Over 100 %, because `*_spans` offsets bound the whole block **including
the `[@` … `#Label*]` markup** (§ coordinate spaces).  **Every
description in the article was extracted.**  They went to the other 33
treatments — *Bisifusarium solicola*, *Talaromyces podocarpi*,
*Penicillium dabashanicum*, *Ophiocordyceps kuchinaraiensis*, and so on.

#### What actually went wrong, twice

**The fragment is a page-break split, not missing text.**  *"species
concept for most fungal groups has led"* sits at char 5 347 of 210 376 —
**2.5 % through the document**, in the *introduction*, not a
description.  What follows it in the `.ann` is a page header, a
copyright line and the journal's editor block.  §12.2's class exactly:
the sentence resumes after the furniture.

**The treatment itself is a synthetic stub, and it is a *rank* split.**
`Nomen ignotum`, `synthetic_nomenclature: true`, 579 characters.  The
operator first read it as *Cadophorella* and then corrected that to
**Neosatchmopsis** — which is right, and makes the case far more
diagnostic than a stray orphan:

| | span | content |
|---|---|---|
| **stub treatment** | 110 148 – 110 727 | *"…resembling Satchmopsis, but conidiomata lack lateral walls and are sporodochial…"* — the **genus** description |
| `Neosatchmopsis ogrovei` | 111 042 – 112 745 | the **species** description |

**315 characters apart.**  The genus description and the species
description of the same taxon sat adjacent in the source; the species
kept its `Nomenclature` heading and became a proper treatment, the genus
did not and was wrapped in a synthetic stub.

**This is §5.5's finding arriving from a third direction: a `Treatment`
is implicitly one taxon at one rank**, and taxonomic papers routinely
describe a genus and its type species back to back.  It joins §12.3.7's
higher-taxon chains and §5.5's own genre cases.  The structural defect
is not that a description was orphaned but that **the schema has no way
to say "this description belongs to the genus of the treatment that
follows."**

**11 of the 34 siblings have empty descriptions.**

#### The generalisable part: a conservation audit

The check used here is cheap and reusable, and answers a question that
has now been raised more than once:

> For a source document, does `sum(len(block))` over blocks of label *L*
> in the `.ann` equal `sum(span length)` over field *L* across **all**
> treatments derived from that document?

**Text conservation and text placement are separable, and only the
second is broken here.**  A corpus-wide run would settle "is extraction
losing material" as a class — replacing per-treatment suspicion with a
number — and would isolate any document where the sums genuinely
disagree.  Not built; recorded as the method, with the markup-overcount
caveat that makes the ratio slightly exceed 1.

**Why the distinction matters operationally.**  Loss and misplacement
have different fixes and different costs.  Misplacement is recoverable
from data already in CouchDB — the blocks are there, correctly
labelled, merely attached to the wrong treatment.  Loss would require
re-ingesting from the PDF.  Establishing which one is in play should
precede any repair work.

### 12.3.13 Embedded figure captions: a Pass-1 missed split, distinct from a Pass-2 mislabel

`taxon_7a36746e`, a **Protosteloid amoeba** (*Schizoplasmodiopsis
micropunctata*).  Operator: *"This document has several paragraphs that
run up against their preceding paragraphs, e.g. "Fig. 1 – Light
micrographs of Schizoplasmodiopsis micropunctata YIP-40." in the second
description block should be the start of a figure-caption.  Fig. 2
caption is later in the same block."*

**This is a different failure from every `Figure-caption` finding so
far.**  §12.3.2 measured captions that were *mislabelled*; here the
caption was never made into a block at all.  Two captions live inside
one 2 655-character `Description` block.  **Pass 1 missed the boundary,
so Pass 2 never had a chance to label it.**

#### The two modes separate cleanly

Counting line-initial caption openers (`Fig. 3 –`, `Figure 2.`) by
position and host label, 300 documents:

| | in `Figure-caption` | elsewhere |
|---|---:|---:|
| **block-initial** | 618 | **217** — Pass-2 **mislabel** |
| **embedded (mid-block)** | 209 | **195** — Pass-1 **missed split** |

**~13 600 missed splits corpus-wide.**  The block-initial/elsewhere cell
(217) is the class §12.3.2 already measures; the embedded/elsewhere cell
(195) is new and needs a different fix — segmentation, not
classification.

Which blocks swallow them:

| host label | n | |
|---|---:|---:|
| `Misc-exposition` | 83 | 43 % |
| `Key` | 26 | 13 % |
| `Materials-and-methods` | 23 | 12 % |
| `Phylogeny` | 14 | 7 % |
| `Description` | 10 | 5 % |
| `Materials-examined` | 9 | 5 % |

#### The detector undercounts, and this document is why

**The regex is line-anchored, and this treatment has lost its line
breaks.**  Its blocks read `3. Results and Discussion3.1.
Identification of…` and `…µm.These morphological features…` — headings
and sentences run together with no separator.  So the operator's own
example is **not** counted by the measurement above: `Fig. 1 –` sits
mid-line, not line-initial.

That is the operator's first observation — *"paragraphs that run up
against their preceding paragraphs"* — and it is a **third**, upstream
defect: the newline structure was destroyed before Pass 1 ran, which
denies the layout CRF the strongest boundary cue it has.  **195 is
therefore a floor**, and the true count is higher by however much of the
corpus has suffered the same loss.  Relaxing the anchor was not done
here: without it the pattern collides with in-text cross-references,
which §12.3.2 already recorded as a contamination source.

#### Two incidental notes

* **A non-fungal taxon, processed normally.**  Amoebozoa, not Fungi —
  and the pipeline produced ordinary `Description`, `Phylogeny` and
  `Materials-and-methods` blocks.  That is a live confirmation of
  §12.3.8's claim that `Description` is a **register** detector rather
  than a fungal-vocabulary one: it generalises to other taxa because it
  was never keyed to this one.
* **This document is Phylogeny-heavy** — five `Phylogeny` blocks
  totalling ~5 100 characters against three `Description` blocks.  Under
  the grouper as it stood before commit `8c0148d`, **all of it was
  discarded**.  It is a useful regression fixture for that fix.

### 12.3.12 The grouper silently drops `Phylogeny` — and improving the classifier makes it worse

Operator: *"Does our grouper not pull out phylogeny as a treatment
component?  That seems like a significant oversight."*

**Correct.**  `treatment.py`'s `_LABEL_TO_FIELD` and
`_LABEL_TO_SPANS_FIELD` have **no `Phylogeny` entry in either map**, so
every block the Pass-2 classifier labels `Phylogeny` is discarded during
assembly.

#### The full audit of the two maps

| | label | status |
|---|---|---|
| **dropped** | `Phylogeny` | Pass-2 label, no field, no spans field |
| **dropped** | `Materials-and-methods` | ditto — arguably correct, not treatment content |
| **dropped** | `New-combinations` | ditto — but **never emitted**: 0 blocks in 300 documents |
| **orphan** | `Distribution` | field *and* spans field exist for a **deprecated** tag no active model emits — 0 blocks |
| inconsistent | `Key` | text field, no spans field |
| inconsistent | `Nomenclature` | spans field, no text field (by design — the name lives in `treatment`) |

**`Distribution` and `Phylogeny` are exact mirror images**, and both
trace to the same 2026-05 schema churn (§12.3.1): a field kept for a
label that died, and no field for a label that lived.

#### The volume

Over 300 documents:

| label | blocks | characters | corpus-wide blocks |
|---|---:|---:|---:|
| `Phylogeny` | 564 | **665 995** | **~39 300** |
| `Materials-and-methods` | 504 | 666 092 | ~35 200 |
| *(captured, for scale)* `Diagnosis` | 790 | 529 330 | — |
| *(captured, for scale)* `Notes` | 2 016 | 1 628 363 | — |

**`Phylogeny` carries 26 % more text than `Diagnosis`**, a field that is
captured and that the whole extraction pipeline treats as first-class.

#### The perverse incentive — the part that matters most

`Notes` **is** captured; `Phylogeny` is not.  Phylogenetic discussion
that the classifier files under `Notes` therefore survives into the
treatment, while the same content correctly labelled `Phylogeny` is
thrown away.

**So improving Pass-2's `Phylogeny` recall actively reduces the content
captured per treatment.**  The pipeline currently rewards the classifier
for getting this label *wrong*.  That is a structural defect, not a
tuning problem, and it silently taxes exactly the work §12.3 is
measuring — every point of `Notes` -> `Phylogeny` refinement recorded
there as *correct* (39 blocks in the cued sample) is content the grouper
then discards.

#### Whether `Phylogeny` should be a treatment field at all

There is a real design question underneath, and it is **scope**:

* **Article-scoped** — one ML/BI tree covers every new taxon in the
  paper.  Attaching it per-treatment duplicates it across siblings, and
  is plausibly why it was dropped.
* **Treatment-scoped** — the placement discussion for *this* taxon.
  `taxon_6e02ee31` is exactly that case: the operator found a phylogeny
  section that *"should have been identified for the 2nd notes block"*,
  i.e. attached to a specific treatment.

**Both exist, and the current design serves neither.**  Resolving it
needs the article/treatment scope distinction from §12.3.9 — the same
missing structural level, arriving for a fourth time.

**Recommendation, and its limit.**  Adding `phylogeny` /
`phylogeny_spans` to both maps is a small change that stops the silent
loss and removes the perverse incentive.  It does **not** resolve the
scope question, and it would duplicate article-level trees across
sibling treatments until that is resolved.  Retiring the orphan
`Distribution` field should wait for Trello #407, which will want that
name.

### 12.3.11 `Misc-exposition` boundary theft — the most-reported defect, quantified

`taxon_6e02ee31`.  Operator: *"a Misc-exposition that consumed 2 lines
of the nomenclature block… the materials_examined block had its last
line consumed by a Misc-exposition."*

**This is the single most frequently reported defect of round 5.**  The
operator has now named it in at least seven treatments — `taxon_0b9a9bfe`
(Etymology), `taxon_0ccf38da` (identifiers, then GenBank numbers),
`taxon_47c3b37d` (*"Misc-exposition stealing"*), `taxon_5180d088` (first
line of a description), `taxon_5c661438` (two blocks), `taxon_57698832`,
and here twice.  It had never been measured.

#### Measured

A `Misc-exposition` block **steals** when the text runs continuously
across its boundary: the neighbouring content block ends without
terminal punctuation and the other side resumes lower-case or mid-token.

| | n | of all `Misc-exposition` |
|---|---:|---:|
| stole the **tail** of the previous block | 2 424 | 14.1 % |
| stole the **head** of the next block | 2 370 | 13.7 % |
| **bridges both — pure interpolation** | **428** | **2.5 %** |
| either | 4 366 | **25.3 %** |

over 17 249 `Misc-exposition` blocks in 300 documents.  **Median length
of a thieving block: 102 characters** — matching the operator's "two
lines" exactly.

#### Which number to trust

**The 2.5 % "bridges both" figure is the defensible core — ~29 800
corpus-wide.**  There the sentence runs *into* the block and *out of*
it, so the block is demonstrably an interpolation inside a single
sentence.  That is very hard to satisfy by accident.

**The 25.3 % figure was recorded as an upper bound, unverified.  It has
since been calibrated (§12.3.19) and the caution ran the wrong way:**
precision against operator judgement is 100 % and recall 62 %, so the
detector under-reports rather than over-reports.  The corpus-wide
extrapolation of ~305 000 is still not quoted as a defect count, because
per-firing precision over *random* blocks remains untested — but the
concern that one-sided continuation would fire spuriously on clean
treatments did not materialise.

#### Its relation to the page-break class

§12.2 measured continuity across **page breaks** and found a 63 % label
change.  This is the same test **without requiring furniture to
intervene**, which makes the page-break class a special case of this
one: there the interpolated material is a running head or page number,
here it is any block the model could not place.  **The general
mechanism is that `Misc-exposition` absorbs whatever interrupts a
sentence**, and page furniture is merely its commonest cause.

#### A second merge invisible to the detector

`n_terms_above_5 = 4` against a threshold of 15, on a treatment the
operator describes as having *"a second (related) species glued onto the
end"*.  **That is the second consecutive undetected merge**, after
`taxon_5bdbc707` scored 0.  Both are short.  §12.3.4's length-blindness
finding now has two independent instances and should be treated as
established rather than provisional.

#### The operator's remaining observations

* **First `Notes` compares two genera rather than the treatment
  species**, and **starts a phylogeny section that belongs to the second
  `Notes`** — `Notes` ⊐ `Phylogeny` (§12.3), with the boundary drawn in
  the wrong place.
* **Third `Notes` is a diagnosis, "which we figure out after the page
  number"** — `Notes` ⊐ `Diagnosis` **plus** a page break, i.e. §12.2 and
  §12.3 co-occurring in one block.
* **The second treatment starts with the erroneous `Type-designation`
  block** — the merge boundary lands one block early.
* **The final `Figure-caption` is the start of a bibliography** — within
  §12.3.2's `Figure-caption` swap scatter, which has no semantic handle.

### 12.3.10 Did English-model OCR destroy the French text? Plausible, and my test cannot settle it

Operator hypothesis: *"French documents are coming out with poor OCR
because the OCR thought it was processing English text."*

**The direct evidence in the affected documents is strong.**  Every
corruption in the Mycotaxon XIV(1) blocks is an accented character
replaced by an ASCII lookalike:

| observed | intended | substitution |
|---|---|---|
| `d!gle de prioritl!` | *règle de priorité* | è→`!`, é→`!` |
| `caractCdsCe` | *caractérisée* | é→`C` |
| `dEcembre`, `Cpaissie` | *décembre*, *épaissie* | é→`E`, é→`C` |
| `£amille` | *famille* | f→`£` |
| `arachn!en`, `l!tat` | *arachnéen*, *l'état* | é→`!` |

That is exactly the failure mode of an engine whose character model has
no French: it picks the nearest ASCII glyph.

#### The population test, and why it inverts

Predicting that French documents should show **few accents** and **many
mid-word capitals**, measured per 1 000 characters over 207 documents:

| | docs | accents | mid-word CAP | punct-in-word |
|---|---:|---:|---:|---:|
| French (>50 % FR function words) | 5 | **18.02** | **0.00** | 0.00 |
| mixed (20-50 %) | 5 | 7.82 | 0.18 | 0.00 |
| English | 197 | 0.17 | 0.31 | 0.03 |

*(clean French prose runs ~25-30 accents per 1 000 characters)*

**Both predictions come out backwards.**  Documents classified as French
*retain* their accents at 18 per 1 000 and have **zero** mid-word
capitals — cleaner than the English documents.

#### But the test is confounded by the very phenomenon it tests

**A destroyed French document stops looking French.**  The classifier
counts French function words, and the two damage mechanisms both attack
them:

* **accent substitution** — `été` survives, but `règle` becomes `d!gle`;
* **word splitting** — the same documents show `l es`, `l arges`,
  `spor es`, and `les` split into `l` + `es` matches nothing.

So the worse the damage, the less French the document scores, and the
**"French" band selects for documents the hypothesis predicts are
undamaged.**  The per-document table shows the survivors of that
selection: the two lowest-accent documents in the French+mixed set score
only 21 % and 31 % French-ness with **0.00 and 2.06** accents and the
highest mid-word capital rate in the sample — precisely the profile
predicted, sitting just outside the band that was measured as French.

**Verdict: not refuted, not confirmed, and not testable this way.**
Recorded because the confound is instructive — **the measurement used
the signal that the hypothesised mechanism destroys**, which is a
failure mode worth recognising before it is repeated.

#### The test that would work

Identify French-language documents by a **damage-independent** channel —
source metadata: journal title and publisher (*Bulletin de la Société
Mycologique de France*, *Revue de Mycologie*, *Cryptogamie*, and the
francophone papers in *Mycotaxon*) — then measure accent rate within
that set.  Language is then established from the container rather than
from the text being assessed, and the circularity disappears.

**If it confirms, the payoff is concrete and unusual for this memo: it
is repairable.**  Accent destruction is a *scanning* defect, not a model
defect, so the affected documents can be re-OCR'd with a French model
and recovered — feeding `data/ocr_rescan_targets.md` rather than any
classifier work.  That would make it the first pathology recorded here
with a fix that does not involve retraining anything.

### 12.3.9 Article boundaries are a missing structural level — and §12.3.8's gate needs them

`taxon_686f39e0`, a French whole-volume scan with heavy OCR damage.
Operator: *"We start with front matter of an issue — mostly a table of
contents which is correctly labeled Misc-exposition.  **I'm thinking
that detecting the start and end of an article would be helpful.**"*
Plus: the first two `Materials-examined` blocks are not materials
examined; the `Notes` block looks like an abstract; the bulk of the
descriptive text lands in `Key`; **and the promised key at the end of
the article was dropped entirely.**

#### The inversion is worth stating on its own

**Real descriptive prose was labelled `Key`, and the real key was
lost.**  Whatever `Key` is selecting on in these documents, it is
anti-correlated with actually being a key — consistent with §12.3.6's
finding that the discriminator is couplet absence × block length, and a
reminder that the label's failures are not merely noisy but inverted.

#### Two more hypotheses tested; both fail to predict rogue `Key`

**Language, retested with a better detector** (24 discriminative French
function words, normalised against English):

| | docs | mean `Key` | median | >10 % |
|---|---:|---:|---:|---:|
| mostly French (>50 %) | 5 | 1.7 % | **0.0 %** | **0 %** |
| some French (20-50 %) | 5 | 5.1 % | 3.8 % | 20 % |
| mostly English | 200 | 2.7 % | 0.0 % | 7 % |

**The refutation holds** — French documents are not *uniformly* rogue;
their median `Key` share is zero and none exceeds 10 %.  **But state the
limit precisely: 5 mostly-French documents in 210 cannot resolve a
moderate effect**, and both rogue cases the operator has surfaced happen
to be French.  What is ruled out is French *causing* it; what is not
ruled out is French material being over-represented among whatever does.

**Whole-volume scale — a weak gradient, not a predictor:**

| | docs | mean `Key` | median |
|---|---:|---:|---:|
| ≥200 page markers | 10 | 5.4 % | 3.8 % |
| 60-199 | 10 | 3.4 % | 2.4 % |
| 20-59 | 41 | 3.1 % | 0.7 % |
| <20 | 149 | 2.4 % | 0.0 % |

Monotone but shallow, on 10 documents at the top.  `ToC-entry` presence
gives 3.2 % against 2.7 % — nothing.

**So there is still no predictor of which documents go rogue.**  §12.3.6's
*description* of rogue `Key` stands; a *cause* remains unfound, and three
candidate covariates (language, OCR damage, document scale) are now
recorded as tested and insufficient.

#### The proposal: article start/end detection

The operator's suggestion is a **missing structural level**, and this
memo has now hit it from four directions:

* front matter and tables of contents treated as document content (here);
* treatments straddling a **species** boundary because no boundary
  existed to cut on (§12.3.4, 1924 *North American Flora*);
* whole-volume Persoonia (Trello #404) — 771 treatments from volumes
  with no per-article title or DOI;
* and most sharply, **§12.3.8's document-level taxonomic-article gate**.

**That last one is a correction to what §12.3.8 endorsed.**  A
document-level gate is sound for an FDA leaflet, which *is* one document
and one non-article.  **It is unsound for a whole-volume scan**, where a
single PDF holds twenty articles of which some are taxonomic and some —
like Korf's memorial notice on Marcelle Le Gal in this very volume — are
not.  A document-level decision there either admits the volume whole or
discards it whole, and both are wrong.

**So article segmentation is a prerequisite for the gate on exactly the
material where the gate matters most.**  Recorded as a design
dependency, not a validated design: no article-boundary detector has
been built or measured, and its feasibility on OCR-damaged multi-column
scans is unknown.

#### The operator's remaining observations, filed to existing classes

* **First two `Materials-examined` blocks are not materials examined** —
  a further instance of that label absorbing frontmatter-like content
  (§12.3.2, ~3 800 corpus-wide).
* **The `Notes` block looks like an abstract** — §5.7: the schema has no
  `Abstract` tag, so abstracts distribute across whatever is nearest.
* **Later `Materials-examined` and `Figure-caption` blocks are valid**,
  and the `Description` blocks are genuine excerpts — so the damage is
  *localised*, not a whole-document failure, which is consistent with
  rogue `Key` being a competing catch-all rather than a collapse.

### 12.3.8 `Description` is a register detector — which is both the generalisation and the bug

`taxon_65cf0058`.  Operator: *"I feel like I've been pranked.  This is
not a taxonomic article at all.  I'm weirdly proud that the two
description blocks really do describe the drug.  Does this suggest that
description will generalize well to other taxa?"*

The document is an **FDA prescribing-information leaflet for BREXAFEMME
(ibrexafungerp)**, an antifungal drug.  `treatment` is `Nomen ignotum`,
`synthetic_nomenclature` is `true`, the title is empty — **all three of
§T3d's pathology signatures firing at once**.  And the `Description`
blocks are genuinely descriptions:

```
[Description] BREXAFEMME tablet for oral administration is a purple, oval…
[Description] BREXAFEMME (ibrexafungerp tablets) are purple, oval, biconvex…
[Description] What are the ingredients in BREXAFEMME?  Active ingredient…
```

#### The measurement: register beats vocabulary 2.75 to 1

Classifying every block over 150 characters by whether it carries the
**descriptive register** (a dimensioned measurement *and* a
morphological adjective — colour, shape, texture, wall) and separately
whether it carries **fungal vocabulary** (`hypha`, `conidi-`,
`ascospor-`, `pileus`, `asci`, …):

| register | fungal vocabulary | n | -> `Description` |
|---|---|---:|---:|
| yes | yes | 1 972 | **67 %** |
| **yes** | **no** | 160 | **44 %** |
| no | yes | 6 078 | **16 %** |
| no | no | 14 946 | 2 % |

**Register without any fungal vocabulary reaches 44 %; fungal
vocabulary without the register reaches 16 %.**  The register is
necessary and nearly sufficient; taxonomic vocabulary is a modest
booster on top of it.

#### The answer: yes, and that is precisely why it misfires

> **Qualified 2026-08-28 by §12.3.16: it generalises across *taxa* but
> not across *languages*.  German scores 1 %.  The "register" is
> Latinate morphological vocabulary, which English and Latin share and
> German does not.**

**`Description` will transfer to other taxa**, because it was never
keyed on fungal vocabulary in the first place.  It recognises
dimensioned, morphologically-adjectival prose.  Plants, insects,
minerals — anything described in that register should carry over
without retraining.

**But the generalisation and the false-positive generation are the same
behaviour.**  The reason it labelled a tablet correctly is the reason it
builds treatments out of pharmacology leaflets.  There is no version of
this model that describes *Amanita* and other genera well while
declining to describe a purple biconvex tablet — the two cases are
indistinguishable at block level, because at block level they genuinely
are the same kind of text.

**So this cannot be fixed inside the labeller, and should not be
attempted there.**  It is the strongest single argument yet for the
**document-level "is this a taxonomic article" gate** already proposed
in this memo's empty-description section, which estimated ~14 000
spurious treatments removable ahead of extraction.  This document is
that gate's ideal specimen: it needs no block-level judgement at all —
an FDA drug label is identifiable from its first 200 characters.

**A corollary worth keeping.**  The 16 % row says the converse risk is
real: **fungal text lacking the register is under-detected.**  A
qualitative description with no dimensions scores like ordinary prose no
matter how taxonomic its vocabulary — which is a plausible mechanism
behind §12.3.6's older and non-anglophone material, and a reason not to
"fix" the register dependence by leaning on it harder.

### 12.3.7 Every lattice edge has now been named independently by the operator

*Leptographium olivaceapini* (a modern mycological revision).  Operator:
*"extracted all components more or less correctly… I could argue that
the first notes block should be biology (eventually ecology).  The second
notes block concludes with what should be a second materials_examined
block.  The first materials_examined block is a type specification."*

**All four claims confirmed:**

```
  537c [Materials-examined] Type. USA, New Mexico, Santa Fe, from Pinus…
  137c [Description       ] Descriptions. Davidson (1971, pp 7–10, figs…
   28c [Biology           ] Host trees. Pinus ponderosa.
   32c [Notes             ] Insect vectors. Dendroctonus sp.
   18c [Biology           ] Distribution. USA.
 1046c [Notes             ] Notes. No living culture associated with the
                            holotype… <- ends in a specimen citation
```

* The `Description` block is a **pointer to descriptions published
  elsewhere**, not a description — so carrying no API labels is correct,
  not a miss.
* `Type. USA, New Mexico…` under `Materials-examined` is **coarsening**
  along `Materials-examined` ⊐ `Type-designation` (§12.3.3), the third
  instance this session.  **And it uses the bare `Type.` cue** that
  §12.3.3 recorded as missing from §12.3's cue list — confirmed in the
  wild, and further reason that label's n = 378 is an undercount.
* The 1 046-character `Notes` block ends in a specimen citation: a
  **line-level mis-cut** (§12.3.5), Pass 1 rather than Pass 2.
* `Insect vectors. Dendroctonus sp.` is a **textbook island** — 32
  characters of `Notes` sandwiched between two correct `Biology` blocks
  of 28 and 18 characters.  §12.2's island work applies directly.

#### The meta-observation: hand review and the confusion matrix agree

Across round 5 the operator has volunteered five label corrections
without reference to the lattice.  **Every one of them is a lattice
edge, and together they name all four:**

| treatment | operator's words | edge |
|---|---|---|
| `taxon_47c3b37d` | *"I would have called the notes section a diagnosis"* | `Notes` ⊐ `Diagnosis` |
| `taxon_57e92419` | *"the type designation fell under materials-examined"* | `Materials-examined` ⊐ `Type-designation` |
| `taxon_5c661438` | *"the diagnosis did indeed get classified with its superclass"* | `Description` ⊐ `Diagnosis` |
| `taxon_62ffeff0` | *"the first notes block should be biology"* | `Notes` ⊐ `Biology` |
| `taxon_62ffeff0` | *"the first materials_examined block is a type specification"* | `Materials-examined` ⊐ `Type-designation` |

The lattice was derived from the **cued-block confusion matrix**
(§12.3), an entirely mechanical source.  That a human reviewer,
proceeding case by case with no sight of those counts, lands on exactly
the same four pairs is **independent confirmation that the lattice
captures the real confusion structure** — and it means the same
structure was recoverable from either source alone.

**This raises confidence in the §1.1.1 programme specifically**, since
its whole premise is that the corpus can supply what would otherwise
need hand annotation.  Here the two methods were run separately and
agreed.

#### A note for #407: the operator's "eventually ecology"

This document declares **three** distinct headings that all currently
land in or near `Biology` — `Host trees.`, `Insect vectors.`,
`Distribution.`  §12.3.1 scoped #407 as a two-way split
(`Distribution` / `Biology`); this argues the residual `Biology` will
itself want subdividing into ecology, host and vector.  **Not a reason
to widen #407** — the two-way split is the measurable, high-volume
win — but a reason to keep the new `Biology` definition explicitly
provisional rather than treating it as the terminal category.

### 12.3.6 Rogue `Key` is a document-level phenomenon — not language, not OCR

Mycotaxon XIV(1), 1982: a French revision of Gabonese
*Botryohypochnoideae*.  Operator: *"has poor OCR and is mostly in
French.  I think we also have an unfamiliar genre.  The outer keys
appear to be almost randomly assigned."*

The document alternates two labels for its entire length:

```
[Key               ] 2282c  I) rcdCfinir l e genre Sotryohypochnus…
[Materials-examined]  611c  paroi jaune, un peu Cpaissie, cyanophile…
[Key               ]  957c  Botryohyfochnus isabelli.nus ; toutefois…
[Materials-examined]  587c  li:t depuis rikoltli: au Gabon : LY 8581…
[Key               ] 1249c  le rattacher au genre Hypochnicium…
   …repeating for ~80 blocks…
```

**`Materials-examined` is largely correct** — `Récoltes: LY 8975, sur
bois mort dans le "bush"` really are collection records.  **`Key` has
swallowed every block of French descriptive prose**, 600–3 500
characters each, and not one of them is a key.  This is §12.2's
"`Key` is a second catch-all" at maximum intensity.

#### Two hypotheses tested, both refuted

**Language: no effect.**  Classifying 210 documents by French/English
function-word ratio — French documents average **2.1 %** `Key` blocks
against English **2.7 %**.  If anything lower.  (Only 8 French documents
in the sample, so this rules out a large effect, not a small one.)

**OCR damage: no effect, and in the wrong direction.**  Measuring word
breakage as the rate of stray single-letter tokens, the top decile by
`Key` share averages **5.4 %** breakage against **6.2 %** for everything
else.  Rogue-`Key` documents are *cleaner* than average.

*(A first attempt used the share of tokens ≤2 characters as an OCR
proxy.  It put 207 of 210 documents in one band — ordinary short words
dominate it.  Discarded, and recorded so it is not tried again.)*

#### What does hold: concentration, length, and couplet absence

| | median | top decile |
|---|---:|---:|
| per-document `Key` share | **0.0 %** | **19 %** |
| median `Key` block length | 430 c | **868 c** |
| `Key` blocks containing a numbered couplet | 12 % | 39 % |

**The median document has no `Key` blocks at all** — the label is
extremely concentrated, with individual documents reaching 81 %, 51 %
and 25 %.

**And the top decile is bimodal**, which is why its couplet rate looks
*better* than average.  It mixes two unrelated populations:

* **genuinely key-rich documents** — revisions and floras with many
  keys: couplet rates of 88 %, 80 %, 100 %, blocks of normal length;
* **rogue-`Key` documents** — couplet rates of 0 %, 3 %, 5 %, with
  median `Key` block lengths of 3 244 and **8 956** characters.

**So the discriminator is neither language nor scan quality but
`couplet absence × block length`.**  A `Key` block over ~1 000
characters with no numbered couplet is not a key.  That sharpens
§12.2's block-level finding with a document-level one: **rogue `Key` is
a property of the document, not of the block**, so it can be detected
once per document and applied to all of its blocks — far cheaper, and
far more reliable, than judging blocks individually.

**The operator's third observation — "an unfamiliar genre" — is the
one that survives all of this.**  A 1982 francophone regional revision
is a genre the model has essentially never seen, and §5.5's genre axis
predicts exactly this: not degraded performance, but a *different*
catch-all being selected.

### 12.3.5 `Description` ⊐ `Diagnosis` restored — one rule, no special cases

*Longistipes albus* (Fungal Diversity).  Operator: *"I think that last
block should be diagnosis.  This looks like a case where the diagnosis
did indeed get classified with its superclass."*

**Confirmed, and the signal was sitting in the block:**

```
 9  254c [Materials-examined] Rhododendron sp., 9℃-13℃, 23 July 2022…
10   94c [Misc-exposition   ] study, with a high ML bootstrap support of 96%…
11  563c [Description       ] a Bayesian probability of 1.0 … Longistipes…
                              ^ contains "distinguished" AND "however"
```

Block 11 carries two comparative markers and was labelled `Description`
anyway — a single-referent block, the signal present, unused.  This is
the concrete instance of §12.3's finding that comparative language runs
55 % in `Diagnosis` against 8 % in `Description`.

#### The edge comes back

§12.3 removed `Description` ⊐ `Diagnosis` on the operator's objection
that the two are lexically separable.  **That was the right verdict
reached by the wrong mechanism.**  The concern was that scoring the miss
as *benign coarsening* would excuse it — but §12.3.3, derived
afterwards, established that **coarsening is a defect unless the block
has more than one referent**.  With that rule in place the edge is safe:
block 11 has one referent, so the coarsening is a defect, exactly as the
operator says.

**Restoring it removes a special case rather than adding one.**  Both
subsumption edges now behave identically under one rule:

| edge | coarsening on a **single**-referent block | on a **multi**-referent block |
|---|---|---|
| `Materials-examined` ⊐ `Type-designation` | defect | **correct** — no specific label covers both (§12.3.3) |
| `Description` ⊐ `Diagnosis` | **defect** — separable at 55 % vs 8 % | correct, same reasoning |

**Net effect on the counts:** the 15 `Diagnosis` -> `Description` misses
move from *swap* back to *coarsening*, and are then defects or not
according to the referent test.  The acceptable rate is unchanged at
70 % either way; what changes is that the framework now states one rule
instead of two, and the ontology is no longer distorted to make the
scoring come out right.

**The general lesson, recorded because it nearly went the other way.**
An edge should be removed only when the subsumption *does not hold*.  If
it holds but coarsening along it is unacceptable, that is a statement
about **block composition**, not about the ontology — and encoding it by
deleting the edge loses real structure and creates a special case that
has to be remembered.

#### Two other defects in this treatment, both already-recorded classes

* **A page-break split that worked.**  Blocks 6-9 are one
  `Material examined` citation interrupted by a `Page-header` and the
  page number `4770` labelled `Key` — the third instance of §12.2's
  furniture pattern and the second of a page number wearing `Key`.
  **This one kept `Materials-examined` on both sides**, so it belongs
  with the 37 % that survive rather than the 63 % that split.
* **Line-level mis-cuts.**  The operator reports the first
  `Misc-exposition` belonging to the preceding `Nomenclature`, the second
  to the following `Description`, and one line of the second
  `Materials-examined` belonging to the two blocks after it.  These are
  block *boundaries* drawn mid-record, not label errors — Pass 1, not
  Pass 2, and the same mechanism as §12.3.4's re-segmentation problem.

### 12.3.4 The merge metric is length-blind — and `Biology` carries its own #407 ground truth

*North American Flora*, Agaricaceae, 1924 — a whole-volume scan.
Operator: *"looks like two unrelated descriptions.  Interestingly, the
biology block correctly groups the HABITAT and DISTRIBUTION blocks."*

```
[Description   ] becoming smoky-olivaceous, the edges white-fimbriate; stipe…
[Misc-exposition] TYPE LOCALITY: England.
[Biology       ] HABITAT: On the ground in low woods.  DISTRIBUTION: Michigan;…
[Bibliography  ] ILLUSTRATIONS: Cooke, Brit. Fungi pl. 398…
[Description   ] Pileus thin, subconic to conic-campanulate, then expanded-umbo…
```

#### The merge scores **zero**

`n_terms_above_5 = 0` against a threshold of 15.  **Two unrelated
descriptions, and the merge detector is completely blind to them** — the
treatment sits in p1 as a clean annotatable case.

The mechanism is length.  The metric counts terms occurring **five or
more times**; these two descriptions total about 1 500 characters, and
almost nothing repeats five times in that span.  **The merge metric can
only see merges in long treatments.**  Two short descriptions fused are
structurally undetectable by it, at any threshold.

That is a **false-negative mechanism for p1 contamination** independent
of the 10 -> 15 threshold work: raising the threshold shrank p2a, but
short merges were never in p2a to begin with.  The p2a precision review
measured the metric where it fires; it says nothing about where it
cannot.  **A length-normalised or type-token variant is the obvious
candidate**, and this treatment is its regression fixture.

#### The structural reading, and a detector that failed twice

Operator: *"this treatment has only distribution and biology.  I would
generally expect more treatment components.  It's also kind of weird for
two description blocks to be split by a biology."*

**Both observations are one thing.**  *North American Flora* house style
runs `<name heading> / Description / TYPE LOCALITY / HABITAT /
DISTRIBUTION / ILLUSTRATIONS` per species.  Read against that, the
treatment is the **tail of species A, all of A's metadata, and the head
of species B**:

```
[Description ] becoming smoky-olivaceous…   <- TAIL of species A
[Misc-exp    ] TYPE LOCALITY: England.       <- A's metadata
[Biology     ] HABITAT… DISTRIBUTION…        <- A's metadata
[Bibliography] ILLUSTRATIONS: Cooke…         <- A's metadata
[Description ] Pileus thin, subconic…        <- HEAD of species B
```

**There is no `Nomenclature` block anywhere in it.**  The species-name
headings were never detected, so the grouper had no boundary to cut on —
which is why the component set is thin *and* why the descriptions
interleave.  The thin component set is a **symptom of the missing
`Nomenclature`**, not an independent defect.

**Hypothesis: a `Description` returning after a record-closing component
marks a straddled boundary.  Tested and refuted, twice.**

| detector | fires | with `Nomenclature` | without |
|---|---:|---:|---:|
| terminal = {Biology, Bibliography, Type-designation, Materials-examined, Etymology} | **49 %** | 52 % | 33 % |
| tightened to {Biology, Bibliography}, no intervening `Nomenclature` | **26 %** | 27 % | 21 % |

Both fire at a base rate far too high to be a pathology, and both are
**more** common when `Nomenclature` *is* present — the opposite of what a
boundary-straddling signature predicts.

**Why it fails, which is the part worth keeping: component order is
genre-dependent, and the two genres invert.**

* In *North American Flora*, `Bibliography` (`ILLUSTRATIONS: …`) **closes**
  a species record.
* In a modern journal, `Bibliography` is the **protologue citation** and
  **opens** one — `Nomenclature > Misc-exposition > Bibliography > … >
  Description` was the commonest firing pattern.

Likewise `Description > Figure-caption > Description` ("Culture
characteristics") is normal and appears in **both** of round 5's flawless
treatments.

**So no fixed component-order model can work corpus-wide.**  Ordering is
a real signal, but it has to be conditioned on genre — which puts it
behind §5.5's genre axis rather than beside it.  Recorded as a refuted
hypothesis so it is not re-proposed.

**What does survive is descriptive.**  Distinct content components per
treatment, 347 sampled: 9 % have one, 23 % have two, **32 % have two or
fewer**.  The operator's expectation of "more components" is right as an
intuition, but a component count is not on its own a usable detector —
untested against ground truth, and a legitimate nomenclature-only entry
has exactly one.

#### `Biology` blocks announce their own #407 split

The operator's second observation generalises.  Self-declared headings,
300 documents:

| cue | n | -> `Biology` | -> `Misc-exposition` | other |
|---|---:|---:|---:|---:|
| `DISTRIBUTION` | 87 | **74 (85 %)** | 10 | 3 |
| `HABITAT` | 55 | 39 (71 %) | 9 | 7 |
| `HOST` | 42 | 34 (81 %) | 3 | 5 |
| `ECOLOGY` / `SUBSTRATE` | 7 | 6 | 1 | 0 |

~13 300 cued blocks corpus-wide, and **`DISTRIBUTION` is 46 % of them —
independently reproducing §12.3.1's 45 % geographic-only figure by a
completely different method.**  Two unrelated measurements agreeing at a
point is the strongest evidence yet that the split is real and sized
correctly.

**But this revises the #407 scoping, and in the direction the operator's
7-unit estimate assumed.**  Classifying `Biology` blocks by which
headings they carry:

| | n | share | what #407 must do |
|---|---:|---:|---|
| distribution cue only | 407 | 33 % | **relabel** — trivial |
| biology cue only | 80 | 6 % | **relabel** — trivial |
| **both — one block, two referents** | **97** | **8 %** | **split the block** |
| no explicit heading | 653 | 53 % | gazetteer + lexicon, then the ~12 % manual |

**17 % of *cued* `Biology` blocks carry both referents**, and this
treatment is one of them: `HABITAT: On the ground in low woods.
DISTRIBUTION: Michigan;…` is a single 77-character block.  The operator
is right that the grouping is *correct at block level* — and that is
exactly why the split is not pure relabelling.  **Those 97 blocks need
re-segmentation, which is a different and harder operation.**

This is §12.3.3's referent-count problem arriving in a second place, and
it is the same 17 % share.  **§12.3.1's "roughly 52 % / 37 %,
mechanical" understated the work**: the *labels* are mechanical, the
*boundaries* are not.

### 12.3.3 Coarsening is a defect only when the block has one referent

*Phyllosticta pterospermi*.  Operator: *"near perfect.  The type
designation fell under materials-examined.  If we don't already mention
it, we should observe that type designation is a specialization of
materials-examined."*

**We do** — `Materials-examined` ⊐ `Type-designation` is one of the three
lattice edges recorded in §12.3, and it is the **largest single
coarsening class**: 124 blocks, and the reason `Type-designation` is the
worst-performing cued label at 48 % honored.  This treatment is a fresh
instance of it:

```
[Materials-examined] Type. China, Hainan Province: Bawangling National…
[Materials-examined] Additional specimen examined. China, Hainan…
```

**But testing the mechanism turned up a limit on that scoring.**  The
obvious alternative explanation is segmentation: if the type *and* the
additional material land in one block, only one label is possible, and
the more general one is then **correct**.  Measured over 300 documents,
splitting type-cued blocks by whether they also carry an
additional-material cue:

| | merged block | clean block |
|---|---:|---:|
| n | 282 (17 %) | 1 419 (83 %) |
| → `Type-designation` | **6 %** | 25 % |
| → `Materials-examined` | **66 %** | 15 % |

**On merged blocks the model picks the covering label two-thirds of the
time, and that is right, not wrong.**  A block containing both referents
has no correct specific label.

**So the lattice needs one more condition: coarsening counts as a defect
only when the block has a single referent.**  Scoring every upward move
as an error over-counts `Type-designation`'s defects by whatever share
of its 124 coarsenings sit on merged blocks.  This is the mirror of the
correction the operator forced earlier — there, downward moves were
wrongly counted as errors; here, some upward moves are too.  **The
direction on the lattice is necessary but not sufficient; block
composition is the other half.**

#### Two defects in my own cue, recorded so the numbers are not re-used naively

* **An unanchored type cue matches figure captions.**  This run allowed
  `Holotype` anywhere in the block, so it caught
  `Figure 3. Phyllosticta pterospermi (holotype SAUCC210104)…` — 179
  blocks, 13 % of the "clean" set, are captions legitimately *citing* a
  type rather than designating one.  **The clean-block column above is
  contaminated and should not be quoted.**  §12.3's cue is anchored at
  block start with following punctuation and does not have this problem;
  **that is the sound number.**
* **The §12.3 cue misses bare `Type.`** — which is exactly the form in
  this treatment.  The cue list is `Type material|Holotype|Typus`, so
  `Type. China, Hainan…` was never counted at all.  §12.3's
  `Type-designation` n of 378 is therefore an **undercount**, and its
  48 % honor rate is measured on a subset that excludes one common house
  style.

### 12.3.2 Three findings from `taxon_57698832` — a lexical handle, a schema gap, and an inconsistency

*Cladoriella kinglakensis*, a **Fungal Planet description sheet**
(Persoonia 39, 2017) — a highly standardised one-page genre.  Operator:
*"pretty good.  The only serious problems is that author frontmatter was
identified as materials_examined, and the first description block looks
to me like a misplaced figure caption.  The tell is 'scale bars'."*

```
[Materials-examined] Pedro W. Crous & Johannes Z. Groenewald, Westerdijk…
[Misc-exposition   ] Michael J. Wingfield, Forestry and Agricultural…
[Description       ] Colour illustrations. Symptomatic Eucalyptus leaves;
                     conidiophores … scale bars …
[Etymology         ] Etymology. Named after Kinglake National Park.
[Misc-exposition   ] Classification — Cladoriellaceae, Cladoriellales,
                     Dothideomycetes.
```

#### "scale bar" is a lexical handle for `Figure-caption` — the label that has no semantic one

§12.3 found `Figure-caption` has **no lattice relatives**: every miss is
a defect, and its 131 swaps scatter across seven unrelated labels
because a caption is defined by page position, not content.  **A lexical
cue is therefore worth more for this label than for any other.**

Blocks containing `scale bar` or `bar = <n>` (300 documents):

| label | n | |
|---|---:|---:|
| `Figure-caption` | **673** | **79 %** |
| `Description` | 68 | 8 % — the operator's case |
| `Key` | 36 | 4 % |
| `Misc-exposition` | 25 | 3 % |
| other | 55 | 6 % |

**79 % precision from one phrase**, against a label with a 66 % honor
rate and no semantic signal at all.  The 68 `Description` hits are the
defect class named here, and they matter more than their count: a
caption absorbed into `Description` contaminates the flagship
morphological extraction with text about *illustrations*.

#### Higher-taxon chains are homeless — a schema gap, not a defect

Operator: *"The Misc-exposition block after the Etymology block perhaps
points to a gap in our definition of Taxonomic treatment.  If the
article gives a chain of higher taxa, it seems like a legitimate
treatment component."*

**Agreed, and the schema has no label for it.**  Blocks under 200
characters naming two or more higher ranks (`-mycota`, `-mycetes`,
`-ales`, `-aceae`, `-ineae`):

| label | n | |
|---|---:|---:|
| `Table` | 268 | 34 % |
| `Key` | 201 | 25 % |
| **`Misc-exposition`** | **187** | **24 %** |
| `Nomenclature` | 83 | 10 % |
| other | 52 | 7 % |

**`Table` and `Key` are mostly legitimate** — taxon lists and key entries
naming families genuinely contain rank chains.  **The 187
`Misc-exposition` blocks are the candidate class**: a
`Classification — Cladoriellaceae, Cladoriellales, Dothideomycetes.`
line is treatment content with nowhere to go, so it falls to the
catch-all.  The 83 in `Nomenclature` are the same content given the
nearest available label.

This is a **T6 schema input**, and it is the second time this session
that the schema — not the model — has been the limit: it joins §5.5's
finding that a `Treatment` is implicitly *one taxon at one rank*, which
is the same gap seen from the other side.  A chain of higher taxa is
precisely the multi-rank information the flat schema cannot hold.

#### Author frontmatter into `Materials-examined`, inconsistently

Author affiliations look like specimen citations — personal names,
institutions, cities, countries.  Blocks under 600 characters carrying
an email, `Department of`, `University` or `Institute`:

| label | n | |
|---|---:|---:|
| `Misc-exposition` | 605 | 80 % — the right answer |
| **`Materials-examined`** | **54** | **7 %** |
| other | 101 | 13 % |

~3 800 corpus-wide — modest, and the model gets it right 80 % of the
time.  **The interesting part is that it is inconsistent within a single
document**: in this treatment two adjacent blocks of identical content
type — one naming Crous & Groenewald, the next naming Wingfield — were
split between `Materials-examined` and `Misc-exposition`.  Adjacent,
same genre, same house style, different labels.  That is not a missing
signal; it is an unstable decision boundary, and it will not respond to
better features.

### 12.3.1 The `Biology` collapse has no recorded rationale — and it buried the majority class

Operator, 2026-08-27: *"Biology is already a collapse of Host,
Distribution, and trophic relationship.  I can certainly imagine
sections labeled 'Notes' that contain one or more of those.  Is there a
record of why I collapsed these into a single label?"*

**Searched: no.** Every artefact records the *fact*; none records the
reason.

| date | commit | what it says |
|---|---|---|
| 2026-03-30 | `c7b2c21` | `Distribution` **added as a peer** of `Biology`. `sec_type_to_tag` maps `distribution/habitat` -> `Distribution`, `biology/ecology/host` -> `Biology` |
| 2026-05-16 | `b2e6c55` | docs sync **reports** the fold as already-existing fact |
| 2026-05-20 | `95b8d91` | `DEPRECATED_TAGS` formalises it; reason given as *"folds into BIOLOGY"* — a restatement |

They were deliberately separate at birth, seven weeks earlier.  **The
likely actual cause is empirical, not ontological**:
`docs/v3_buildout.md:133` records *"Distribution field never populated —
v3_hand model omits the label."*  The label was dead in the model, and
the fold ratified that rather than deciding it.

**And `b2e6c55` explicitly left the question open** — *"whether to keep
a separate Distribution embedding is a design decision worth its own
pass, not a search-and-replace."*  **That pass never happened.**

**Measured composition of `Biology`** (300 documents, 1 237 blocks over
40 characters):

| content | n | |
|---|---:|---:|
| geographic cue only — no host, no phenology | **554** | **45 %** |
| no cue matched | 377 | 30 % |
| geographic + host | 159 | 13 % |
| host only | 77 | 6 % |
| everything else (phenology combinations) | 70 | 6 % |

**`Biology` is not a balanced mixture of host, distribution and trophic
relationship — it is predominantly `Distribution` wearing a `Biology`
label.**  45 % is geographic-only and only 20 % carries any host cue at
all.  The collapse folded the *majority* class into the minority's name.

#### Scoping the split — Trello #407

Filed 2026-08-27 as *"Split Biology label in training data"*, estimated
at **7 units on the assumption of a significant manual step**.  The
residual was measured afterwards, and it argues the manual step is
smaller than that assumption.

Re-probing the 377 no-cue blocks with a second pass (place names,
bare `on <Genus>`, trophic vocabulary, voucher, morphology, citation):

| | n | of residual |
|---|---:|---:|
| trophic / ecology vocabulary | 156 | 41 % — genuine `Biology` |
| bare country or place name | 91 | 24 % — more `Distribution` |
| bare `on <Genus>` host | 24 | 6 % — genuine `Biology` |
| voucher / morphology / citation | 13 | 3 % |
| **nothing matched — needs eyes** | **144** | **38 %** |

**Rolled up over all 1 237 `Biology` blocks:**

| destination | share | how |
|---|---:|---|
| `Distribution` | **~52 %** | mechanical — gazetteer + the geographic phrase list |
| `Biology` (host, substrate, trophic) | **~37 %** | mechanical — trophic lexicon |
| **needs eyes** | **~12 %** | not separable by vocabulary |

**And the manual 12 % is concentrated, not spread.**  Those 144 blocks
sit in **39 of 300 documents**, and **80 % of them in just 13 documents
(4.3 %)** — with a single document holding 62 blocks, 43 % of the whole
residual on its own.

That document is the tell: its blocks are *"Chaga conk grown on birch in
sparse wood edge"*, *"Dead chaga conk collected from dead birch"*,
*"Bark with phloem in cross-section of affected birch"* — 41–51
characters, near-identical, repeated.  **These are figure captions
mislabelled `Biology`**, which is not a `Biology`-split question at all
but §12.3's `Figure-caption` boundary failure showing up in a second
place.

**Implication for the estimate.** The manual pass is **per-document and
per-pattern, not per-block**: roughly 4 % of documents, most
contributing one repeated artefact each.  The 7-unit estimate looks
conservative — though deliberately so, and the two mechanical buckets
still need their gazetteer and lexicon built and validated, which is
where the real work sits.

**Why this matters beyond tidiness.**
`docs/segment-detector-scope.md:30,67` already draws the distinction the
schema erased: `Distribution` is *"pure geographic NER — off-the-shelf
tools apply"*, while `Biology` is *"mixed prose… semantic topic tagging,
not segment extraction."*  **Those are different tools**, and merging
the labels guarantees that whichever is built will be trained on 55 %
material it cannot handle.  Un-collapsing is a **T6 input**, and 45 % is
the size of the recoverable class.

**The `Notes` ⊐ `Biology` edge stands, with a sharper reading.**  Given
the composition above, "a `Notes` section containing biological
material" is in practice "a `Notes` section containing *distribution*",
which is exactly the operator's intuition.

**Corroboration for `Notes` ⊐ `Diagnosis`.** Comparative language
appears in 45 % of `Notes` blocks — between `Diagnosis` (55 %) and
`Description` (8 %).  `Notes` genuinely carries diagnostic content
roughly half the time, which is why `Notes` -> `Diagnosis` runs 65 with
zero reverse and is a refinement rather than an error.

#### Page breaks: the highest-precision signal found so far

`taxon_5180d088` (*Quercicola fusiformis*), operator: *"nearly
perfect. The first line of the description was eaten by a
Misc-exposition that comes before the page number (which is encoded as
a Key block)."*

```
[Etymology       ] Etymology – Referring to the prominent guttule…
[Misc-exposition ] Saprobic on fruit of Fagaceae plant. Sexual morph:
                   Ascomata 275–300 μm            <<< lost
[Page-header     ] --- PDF Page 33 Label 33 ---
[Key             ] 33                             <<< 2-char page number
[Description     ] μm diam. (x = 290 × 340 μm; n = 10), gregarious…
```

Worse than "first line lost": `Ascomata 275–300 μm` and `μm diam.` are
**one measurement torn in half**. The `description` field begins
mid-unit.

**Measured over 300 documents:**

| | n | |
|---|---:|---|
| page breaks examined | 5 937 | |
| …where the text **clearly continues** across the break | 327 | 6 % |
| …of those, **same label either side** | 122 | 37 % — correctly kept together |
| …of those, **different label** | **205** | **63 % — the split** |

**~14 300 corpus-wide**, comparable in size to the self-labelling
blocks (~15 400), and **higher precision than any of them**: if the
text is demonstrably one sentence, the two halves belong to the same
field. There is no judgement call, no domain knowledge, and no
threshold to tune.

Examples from the sample, all mid-sentence:

* `Materials-and-methods` → `Misc-exposition`:
  *"…and here the rocks are generally" | "not covered with weathered
  sedimen…"*
* `Notes` → `Type-designation`: *"…these genera have affinity with the"
  | "extant Hypoxylon of the family Xyl…"*
* `Materials-examined` → `Notes`: *"…Notes: Pluricellaesporites" |
  "mexicanus Kalgutkar & Janson. 2000…"*

**Why it stayed hidden.** Only **6 %** of page breaks have continuing
text — most fall at paragraph boundaries, where a label change is
legitimate. The signal is confined to a small, precisely identifiable
subset, which is exactly why a per-block metric never surfaces it.

**The repair is unambiguous too.** Both halves take one label; mass
decides which, and §12.2's mass-weighted island work already supplies
that rule.

**Also: the page number `33` is a two-character `Key` block.** More
evidence for `Key` as a second catch-all, and a note for the furniture
sets used throughout this section — they exclude `Page-header` but not
tiny `Key` blocks, so page furniture can still interrupt a gap
calculation.

#### Two mechanisms that demonstrably work

Worth recording against the weight of failures above, because both are
things D12 fixes must not break.

**Span reconstruction across interruptions works.** The same treatment
has its diagnosis split over paragraphs 685 and 697, with `Wang et
al.`, a full `Fig. 20` caption and the page number `165` in between —
and the `diagnosis` field is correctly reassembled from both. **The
joining machinery is not the problem.** D12's cases fail because
mislabelled blocks are never *candidates* for joining, not because
joining cannot span a gap.

**`Figure-caption` continues to do its job when the boundary is right**
— as in `taxon_3b7a80bc` (§0.5). The label is sound; its boundary
decisions are what fail.

The one defect here: the etymology is split between a
`Misc-exposition` (`Etymology: The name refers to its ascomatal hairs,
which are`) and the first line of the description — the §12.2
self-labelling-block class, mid-sentence.

#### And a fourth defect, invisible in brat

`taxon_0ccf38da` also carries a **second `description` span at
paragraph 1137**, after its notes, holding

> *"Sexual morph of this family features perithecial ascomata,
> unitunicate, cylindrical-clavate, stipitate asci…"*

— a **family description of Conioscyphaceae**, preceded by a
`Misc-exposition` reading *"Réblová et al. (2016) established
Conioscyphaceae with Conioscypha as the type genus."* The treatment is
*Claviformispora phyllostachydis*, a different family.

**`merge_metric` reads 0.** The two descriptions share almost no
repeated terms — one is a species description, the other a family
diagnosis — so the repetition metric is blind to exactly the kind of
absorption §6.1 already showed it mis-measures. `§12:desc_span_gap`
fires, which is the signal that actually catches it.

#### The same treatment carried two defects the reviewer could not see

`taxon_0b9a9bfe` looked otherwise fine in brat, and the fixture test
disagreed — it fires `§10:tail_clip` and `§10:diag_head_clip`, and both
are true positives. The description ends `…Zygospores formed` and the
diagnosis begins `in axial alignment with conjugating segments after
14 d`: **the field boundary cuts one sentence in half.** The two flags
are one defect seen from either side, and the consequence is that the
`diagnosis` field holds description-tail plus a `Notes –` section
rather than a diagnosis — D19's measured class.

Worth recording as a review-process point, not just a data one: reading
labels in brat answers "is this feature named correctly" and cannot
answer "does this field begin where it should". The two reviews need
different instruments and T5 supplies only the first.

#### Gap *density* is not the signal — a failed detector

`taxon_341b4bc0` (*Pucciniastrum*, *Mycosphere* 15) is the densest
cascade reviewed: the operator enumerated **ten distinct boundary
defects in one treatment** and every one checks out. `Fig. 95` and the
index numbers lost to a `Misc-exposition` that also ate the etymology;
the description truncated at `and minutely echinulate.`; the host
absorbed by that same block; a phylogeny line labelled `Key`; a figure
caption labelled `Key`; and **two `Key` blocks holding the nomenclature
of a second species**, *Pucciniastrum boehmeriae*, plus its `Fig. 96`
— whose description is this treatment's second description span. The
nomenclature field itself stacks three ranks: family, genus, then the
`sp. nov.`

The obvious detector is **gap-block density**: a treatment whose spans
are separated by many unclaimed blocks is a treatment in an incoherent
region. Measured over 201 round-5 treatments with ≥ 2 spans, this one
scores **1.25 gap-blocks per span against a median of 1.00** — rank 86
of 201, entirely unremarkable.

**It fails because the defects are small.** One block here, four there.
The damage is in labels being *wrong*, not in the volume of unclaimed
material, and a block count cannot see wrongness. What density does
find is treatments whose spans bracket most of a document — the worst
scores 3 182 blocks per span — which is the §5.6 / p2b population, a
different problem.

**What does catch it**: `§6:authored_binomial`, `§6:multi_description`
and `§12:desc_span_gap` all fire. `merge_metric` reads **0**. That is
§6.1's lesson again — the repetition metric is the weakest of the
available signals, and the span-based ones do the work.

#### Label islands: a detector that took two failures to find

`taxon_4cb3fcb6` is one of §5.6's ten correct refusals, and the
operator added the detail that matters: *"the sole surviving
description block is from the middle of a (mostly correctly
identified) materials and methods section."*

```
[Materials-and-methods ] 2. Materials and Methods  2.1. Fungal Strains…
[Misc-exposition       ] 2.4. Phenotype Assay
[Description           ] Growth rate and conidiation were detected …   <<<
[Page-header           ] --- PDF Page 3 Label 3 ---
[Materials-and-methods ] after 3, 7, and 10 d for quantity determination…
```

A single `Description` stranded in a methods run. **The obvious
detector — a block whose label differs from both neighbours — took two
refinements to become usable, and both failures are instructive.**

| version | rate | why it failed |
|---|---:|---|
| any label differing from both neighbours | **37.8 %** | `Misc-exposition` is 35.4 % of blocks, so "surrounded by Misc-exposition" *is* the base rate |
| restricted to content labels | **13.4 %** | `Materials-examined` between two `Description`s is the **normal alternation of a monograph**, not an anomaly |
| **restricted to implausible pairs** | **1.10 %** | usable — **~4 464 corpus-wide** |

"Implausible" means taxon content (`Description`, `Diagnosis`,
`Nomenclature`, `Etymology`, `Type-designation`, `Materials-examined`)
stranded inside an apparatus run (`Materials-and-methods`,
`Phylogeny`), or the reverse:

| island | inside a run of | n |
|---|---|---:|
| `Phylogeny` | `Etymology` | 12 |
| **`Description`** | **`Materials-and-methods`** | **10** |
| `Phylogeny` | `Description` | 7 |
| `Description` | `Phylogeny` | 6 |
| `Diagnosis` | `Phylogeny` | 5 |

**Note what the third version needed: a table of which transitions are
plausible.** That is exactly what a linear-chain CRF's transition
matrix already encodes. Hand-coding the table works, but the better
form is to **read the model's own learned transition weights and flag
low-probability transitions** — self-supervised, no hand-tuning, and it
adapts when the label set changes. Recorded as a component in
`docs/rl-framework-components.md` §1.5.

#### Mass, not block count — and one hypothesis that measured zero

The operator refined the shape: *"lots of A, small Misc-exposition,
small B, small Misc-exposition, lots of A. It seems likely that B
should be A. In this case, the first Misc-exposition should also be A,
but I don't see how to detect this."*

**The refinement is right and my version missed it: I counted blocks,
not mass.** `taxon_4cb3fcb6` shows why the asymmetry matters —

| chars | label | |
|---:|---|---|
| 2 227 | `Materials-and-methods` | |
| **20** | `Misc-exposition` | `2.4. Phenotype Assay` |
| **245** | `Description` | the island |
| 26 | `Page-header` | |
| 29 | `Misc-exposition` | running head |
| 3 866 | `Materials-and-methods` | |

**6 093 chars of A flanking 245 chars of B**, with 75 chars of small
matter between. Measured as "a block under an eighth the size of the
smaller flanking block, between two same-labelled blocks":
**717 of 35 339 interior blocks, 2.0 %, ~50 000 corpus-wide**, of which
**31 % have `Misc-exposition` as the island**.

**A hypothesis that failed, recorded because it was mine and it was
confident.** The operator's *"I don't see how to detect this"* referred
to the first `Misc-exposition`, `2.4. Phenotype Assay`. I proposed
**hierarchical section numbering** — `2.4.` nests under the `2.
Materials and Methods` block right above it, so numbering should
identify it. Measured over the same 300 documents:

* only **9 of 222** enclosed `Misc-exposition` blocks open with a
  section number at all (4 %);
* **zero** of those nest under the preceding block's number.

**~0 corpus-wide.** Mycological papers largely do not number their
sections, and `taxon_4cb3fcb6` is a *Journal of Fungi* paper following
a house style that does. The operator's instinct was closer to right
than my proposal.

**What is left for that case.** Sentence continuity — the mechanism
that caught `taxon_fdbd1b53`'s `Zygospores formed`, `taxon_134c7e0e`'s
`The asexual morph has conidia` and the severed `Blume.` — **does not
help either**, because a heading is not a severed sentence. The
remaining signal is bare size: a 20-character block is a heading, and
headings belong to the section they head. That is a rule, but a weak
one, and it is not yet measured.

**A caution the first two versions earn.** Furniture had to be skipped
(page headers and bare page numbers sit between the block and its real
neighbours), and OCR-destroyed blocks had to be excluded — the first
run's `Description` examples were all U+FFFD runs. Neither exclusion is
optional.

#### One number not to trust yet

**Adjacent `Nomenclature` runs** — two or more in a row with no body
between — extrapolate to **~6 100**, and the obvious reading is "a
heading whose body was lost." Testing it, only **32 %** are
synonym-shaped by an explicit-marker regex (`=`, `≡`, `syn. nov.`).
But a basionym listed *without* a leading `=` is legitimate synonymy
and reads as "not synonym" here, so **68 % is an upper bound and
probably a bad one.** Recorded so the ~6 100 is not quoted as a defect
count.

#### What this says about fix order

By magnitude, and independent of how hard each is:

1. **`Misc-exposition` at 35 %** — not a bug in itself, but the pool
   every swallowed-content case is drawn from. Splitting it into real
   categories would shrink D12's search space more than any detector.
2. **~29 000 descriptions with no name ahead of them** (~18 200
   mis-ordered + ~11 100 orphaned) — the largest single structural
   defect measured, and the direct source of much of the 39.6 %
   synthetic-nomenclature rate.
3. **~1 590 D12** and **~544 D18** — real, bounded, and small by
   comparison. Worth detectors; not worth doing first.

### 13. Diagnosis segments may not be worth Claude annotation (operational note)

**Observation** (operator, 2026-07-02): during hand-review of
the round-1 + round-2 treatments, labelling anatomical
features INSIDE `diagnosis` blocks is harder than labelling
the same anatomy inside `description`.  Diagnosis text is
Latin-morphology-heavy, dense, and typically one long
paragraph rather than the `Feature: value; Feature: value;`
structure that Description has.  Anatomical terms in a
diagnosis (`asci clavati`, `basidia ovoidea`) share
vocabulary with the labelled feature classes but the label
attribution is often arbitrary without context the Diagnosis's
telegraphic style withholds — `clavati` alone could apply to
Asci, Basidia, or Paraphyses.

**Consequence**: Claude's annotator produces Diagnosis
annotations at the same rate it produces Description
annotations, but a larger fraction of them get rejected in
review.  API spend on Diagnosis annotation buys less golden-
data signal per dollar than the equivalent Description spend.

**Proposal (not a plan yet)**: skip Diagnosis blocks in
`bin/llm_annotate_features`.  Two implementation shapes:

  * **`--skip-diagnosis` CLI flag** (default False initially;
    consider flipping the default once measured).  Simplest
    change; keeps the option open.
  * **Downweight, not skip** — annotate as today, but tag
    Diagnosis-derived candidates so review filtering can
    prioritize Description-derived ones.  Doesn't save
    API spend but improves reviewer-throughput.

**Still a proposal, 2026-08-23.**  The operator asked
whether an intent to skip Diagnosis had been recorded.  It
has not: this section is an observation plus a proposal, and
the data below has never been gathered.  What *has* happened
is that reviewers have been skipping Diagnosis blocks in
practice since round 1 — a de facto convention with no
written decision behind it, which is worth naming so it
either becomes one or gets revisited deliberately.

**A counter-example, and why it does not overturn the
proposal.**  `taxon_7e3011a6`
(`§12-caption-prefixed-description-block-dropped`) has a
diagnosis that appears to add two features its description
lacks — `lyocystidia with a narrow capillary lumen that
extends up to the apex`, and `larger basidiospores`.  Both
are in a `Description`-labelled block that the grouper
dropped: the missing text reads `… narrow capillary lumen
extending up to the apex … Basidiospores 7.8–11 × 2.5–3.5
µm …`.  **The diagnosis is not supplying unique information;
it is the only surviving witness to information the
extractor lost.**

That distinction is the one this decision turns on.
Diagnosis content divides in two:

* **Recoverable** — anatomy that is also in the description,
  or would be if the description were intact.  Annotating it
  duplicates work that §12 fixes should make unnecessary.
* **Inherently diagnostic** — the *comparative* claim
  (`larger than T. hirtellus and T. effugiens`), which no
  complete description contains.

`taxon_7e3011a6` cannot settle which predominates, because
its description is truncated.  **Decide §13 on treatments
whose descriptions are complete** — otherwise content loss
will keep making Diagnosis look indispensable.

**Data needed before deciding**: the kept-vs-rejected rate
by source field (`description` vs `diagnosis`) across the
round-1 + round-2 hand review.  `bin/triage_treatments`
already surfaces per-treatment kept/added/deleted counts —
extending it (or a companion script) to split by candidate
source-field would give the ratio directly.

**Cost estimate**: Diagnosis blocks are typically 20-40%
of a treatment's synth doc by character count.  Skipping
them would cut per-treatment API spend by roughly that
fraction on multi-field treatments — meaningful at
production-corpus scale (30 k+ treatments).

**Interaction with §12**: label-aware assembly makes this
trivial to implement — Diagnosis-labelled segments simply
don't get routed to the Claude annotator.  Reinforces the
§12 case: passing labels through gives us cheap operational
levers (skip Diagnosis, route Key elsewhere, etc.) that
require regex-scraping today.

**Concrete evidence (2026-07-02, `taxon_8f93bded...`)**:

* Description opens with a literal `Diagnosis —` block
  followed by a detailed description.  Total 17
  annotations came out of this treatment; **Claude
  skipped the entire Diagnosis paragraph** — every
  annotation traces to the description body downstream.
  Empirical support for the "skip Diagnosis" hypothesis:
  Claude is ALREADY skipping Diagnosis-flavoured blocks
  implicitly.  Making the skip explicit via
  `--skip-diagnosis` would save the tokens spent parsing
  the Diagnosis text (currently sent to the API but
  produces no annotations) — pure cost reduction with no
  observed signal loss.

**"Diagnosis" is polysemous in taxonomic literature**:

The polysemy has been debated within systematics.  Cifelli
& Kielan-Jaworowska (2005), *Acta Palaeontologica Polonica*
50(3): 650-652 ("Diagnosis: Differing interpretations of
the ICZN") argue that per Linnaeus's *differentia specifica*
and ICZN Art. 13.1.1 + Recommendation 13A, **a diagnosis
must be differential** — it must include, at least
implicitly, comparison to similar taxa.  They quote Mayr
& Ashlock (1991: 391): "Respectable taxonomists go well
beyond this minimal requirement… by comparing the newly
proposed taxon with its closest relative(s) AND describing
the diagnostic characters carefully."  So the ICZN-
preferred sense IS Differential Diagnosis; non-differential
character lists labeled "Diagnosis" are informal/loose
usage the discipline is arguing against.  This matches the
skol training corpus: `Diagnosis` mostly means Differential
Diagnosis.

The single word `Diagnosis` still covers three
semantically distinct block types in practice.  Using the
technical names consistently:

  1. **Latin Diagnosis** — formal Latin morphology block,
     once required by the ICBN.  Dense, telegraphic.
     Hardest to annotate; the Diagnosis→skip rule's
     original motivation.
  2. **Differential Diagnosis** — differentiates the
     target species from close relatives.  Contains
     multiple species names / binomials (legitimate — the
     comparison IS the content).  Evidence:
     taxon_9e048013 (false-positive discussion in §6),
     taxon_d2d26d25 (leaked-into-description fragment,
     §12), taxon_83e36037's 3 legitimate diagnosis
     citations.
  3. **Diagnostic Characters** — defining features of
     THIS species.  Reads like a Description
     (taxon_8f93bded is this variant).  Semantically
     description-like; the source may still call it
     `Diagnosis` in the header, but the content is
     distinct from a Differential Diagnosis.
     **Note (Cifelli & Kielan-Jaworowska 2005; Mayr &
     Ashlock 1991)**: a proper Differential Diagnosis
     WILL describe the target's diagnostic characters as
     part of the comparison ("differs from X by having
     larger P4, longer m1, …").  So a "Diagnostic
     Characters" block existing on its own — without any
     comparative framing — is the informal/loose usage
     the ICZN discussion argues against, not a legitimate
     distinct category.  For the skol schema this
     matters because our current training data would
     flatten both onto `Diagnosis`; only if we ever
     needed to LEARN the distinction would the split
     become important.

The section classifier isn't necessarily wrong to label a
Diagnostic Characters paragraph as `Diagnosis` — the
source header says so, and the reviewer can defend either
call.  For the operational question (should we skip?) the
answer is the same across all three: yes.  The
`--skip-diagnosis` rule doesn't need to distinguish
between the sub-types.

**Training-data audit item (2026-07-02)**: the training
set mostly uses `Diagnosis` to mean **Differential
Diagnosis** (the comparative form) — consistent with the
ICZN-preferred sense per Cifelli & Kielan-Jaworowska
(2005).  Worth a corpus-wide audit to check whether any
**Diagnostic Characters**-only paragraphs (no comparative
framing) have been labelled as `Diagnosis` in the golden
data — per the C&KJ argument these are borderline
mis-labelled; they SHOULD have been either promoted to
`Description` or paired with a Comparisons/Remarks block
to complete the differential form.  Concrete audit
signal: `Diagnosis`-labelled spans containing zero
authored binomials and zero comparative-language
markers (`differs from`, `similar to`, `distinguished
from`, etc.) are candidates.  Cheap manual scan on the
~30-treatment golden set; deferred until the label-aware
assembly work (§12) forces the schema question.

**Detector-miss (related)**: `count_diagnosis_headers` in
`triage_signals.py` uses the regex `\bDiagnosis:` — literal
colon required.  taxon_8f93bded's block starts with
`Diagnosis —` (em-dash) and slipped past.  Simple
extension: match `\bDiagnosis\s*[-—–:]` (colon, hyphen,
em-dash, en-dash).  Filed alongside the taxon_f00f8353
detector-miss note in §6.

### 14. Shared diagnosis serving multiple species (structural note)

**Symptom**: a treatment doc has an **empty** `description`
field and a substantial `diagnosis` field.  It represents a
diagnosis section from an article covering N species where
ONE diagnosis serves multiple per-species descriptions —
the extraction split them apart and orphaned the diagnosis
as its own treatment_prose doc.

**Evidence**:

* **`taxon_715c2164...`** — noted 2026-07-02.  Perfectly
  clean 2139-char diagnosis; description length 0.
  Sibling treatment_prose docs (not yet enumerated) hold
  each of the 6 species' descriptions with empty
  diagnosis fields.  Article covers 6 species (5 new);
  the diagnosis differentiates them collectively.
  Triage detector flags this as
  `§2:synth_nomen` — correctly, because no proper
  Nomenclature attaches to a shared-diagnosis block.
* **`taxon_7af2e7c8...`** (variant, noted 2026-07-03) —
  **Distinct sub-shape**: a treatment with a REAL
  Nomenclature (`synthetic_nomenclature = False`),
  desc_length = 0, and only a short Differential
  Diagnosis (188 chars).  1 Claude annotation total.
  Not a shared-diagnosis orphan (that class has
  synth_nomen) — this is a legitimate single-species
  treatment whose source paper carried nothing but a
  brief diagnostic snippet, likely a redescription
  associated with a key or an existing-species mention
  in a monograph.  Operator judgment: "we probably
  don't care about this treatment" — accurate; too
  little content to bootstrap meaningfully, and the
  little that's there is Differential Diagnosis
  (§13-skippable per the operator's earlier note).
  **Detection**: `desc_length == 0 AND diag_length <
  500 AND synthetic_nomenclature == False` — the
  synth_nomen distinction separates this "legitimate
  low-value" case from taxon_715c2164's shared-diagnosis
  orphan.  **Reviewer treatment**: skip per §0 rule 2
  (not enough content to annotate).  Not a
  data-quality bug; catalog for triage completeness.

**Likely stage**: treatment-grouper's assumption that each
treatment has exactly one (Nomenclature, Description,
Diagnosis) tuple.  When a paper's structure fans-out (1
shared Diagnosis → N Descriptions), the grouper splits
along Nomenclature boundaries and orphans the shared
Diagnosis as a standalone treatment.

**Severity**: low — probably rare (multi-new-species papers
with a shared diagnosis format aren't the norm).
Consequence is one orphaned diagnosis doc + N descriptions
each missing their diagnosis.  Downstream annotation still
works on the descriptions; only the diagnosis-derived
signal is lost.

**Assembly-aware fix (§12 reinforcement)**: label-aware
assembly could recognize the shared-diagnosis pattern and
either (a) duplicate the diagnosis onto each per-species
treatment (simple, materializes the join, but adds
N-fold weight to shared clauses in training), or (b)
maintain a reference relationship (species treatments
carry a `shared_diagnosis_treatment_id` pointing at the
orphan diagnosis).  Option (a) is simpler and probably
sufficient given the rarity; option (b) is cleaner but
requires a schema change.

**Reviewer treatment**: skip (per §0 rule 2 — no
Nomenclature, no target species to annotate).  If a
future assembler fixes the fan-out, this treatment
disappears from the corpus.  Not worth Phase 1 review
effort at current-rarity estimates.

**Detection**: `desc_length == 0 and diag_length > 0`.
Trivial; can be added to `triage_signals` as an
"orphan_diagnosis" flag.

### 15. JATS element boundaries joined without a separator

**Symptom**: words run together in `description` and every other
extracted field — `Descriptionof the asteromella-like spermatial
morph.Infection localised`, `Typeof Asteromellapistaciarum`,
`Notes.The classification`, `Liberomycespistaciae`.

**Root cause, reproduced in isolation**: `extract_text()` in
[`ingestors/jats_to_yedda.py`](../ingestors/jats_to_yedda.py)
concatenates with `"".join(parts)`.  That is correct XML
semantics — `H<sub>2</sub>O` must render as `H2O` — but it drops
the separator at boundaries where the rendered article has a
space.  Two distinct variants:

* **Block-level** (unambiguous bug).  A section title and its
  paragraph are sibling elements:

  ```xml
  <sec sec-type="treatment-description">
    <title>Description</title>
    <p><bold>of the asteromella-like spermatial morph.</bold>
       Infection localised …</p>
  </sec>
  ```

  → `Descriptionof the asteromella-like spermatial
  morph.Infection localised …`.  Same shape produces `Typeof`
  from `<title>Type</title><p><bold>of …` and `Notes.The` from
  `<title>Notes.</title><p>The …`.  `<title>`, `<p>` and `<sec>`
  are block-level; there is no counterexample where they should
  be joined tight.

* **Inline taxon names** (needs a scoped rule).  ARPHA/Pensoft
  marks names as split `named-content` elements:

  ```xml
  <named-content content-type="taxon-name">
    <named-content content-type="genus">Asteromella</named-content>
    <named-content content-type="species">pistaciarum</named-content>
  </named-content>
  ```

  → `Asteromellapistaciarum`.  **The source XML is inconsistent
  within a single document** — in PMC6160797, 93 genus|species
  boundaries carry no whitespace and 149 do.  So this is sloppy
  source markup rather than a convention that can simply be
  read off.

**Scale** (whole `treatments_prose`, 42 096 treatments with a
description, marker = `[a-z]\.[A-Z]`):

| `source_anchors` kinds | treatments | with marker | rate |
|---|---:|---:|---:|
| `plazi` only | 2 735 | 2 629 | **96.1 %** |
| `arpha`+`jats_section`(+`mycobank`)+`plazi` | 2 593 | 10 | 0.4 % |
| `pdf` | 26 496 | 798 | 3.0 % |
| none | 9 988 | 976 | 9.8 % |
| **overall** | **42 096** | **4 414** | **10.5 %** |

The `pdf` and no-anchor rates are a different population
(genuine OCR and typography noise, §9).  The signal is the
96.1 % vs 0.4 % split: treatments reaching us through the
plazi-only path are almost universally affected, and those
carrying `jats_section` anchors are almost universally clean.

**Severity: high, and quiet.**  It fires no flag.  Worse, it
**blinds gnfinder**.  Against the live service:

```
"Asteromellapistaciarum is here added"   -> []
"Asteromella pistaciarum is here added"  -> Asteromella pistaciarum, oddsLog10 13.2
"Liberomycespistaciae sp. nov."          -> []
"Liberomyces pistaciae sp. nov."         -> Liberomyces pistaciae, SP_NOV
```

So `§6:authored_binomial`, the `authored_binomial_in_desc`
fixture label, and any future name-based work in §1/§2 are
silently unavailable on ~2 629 treatments.  Not "wrong" —
*absent*, which is harder to notice.  This is the same failure
shape as taxon_2f276bfa's mid-word OCR defeating gnfinder
(Trello #395) but a completely different cause: markup joining,
not character corruption, and fixable upstream rather than
needing fuzzy matching.

**Exemplar**: `taxon_30d8d8d4...` — noted 2026-08-21 from
round-4.  Operator: "a fairly complete description of an
unnamed morph, possibly missing the rest of the description of
the base species."

**Nothing is missing.**  The article gives no base-species
description.  Its *Septoria pistaciarum* treatment is exactly
nomenclature + Type + "Description of the asteromella-like
spermatial morph" + Notes + Figure 7, because the treatment's
purpose is to designate a lectotype for *Asteromella
pistaciarum* and synonymise it under *Septoria pistaciarum* as
its spermatial morph.  Both of the article's treatments are in
the corpus — this one and `taxon_1cf6a119` (*Liberomyces
pistaciae*) — and every field maps correctly: nomenclature =
the accepted name, `materials_examined` = the synonym's
lectotype, `description` = the morph, `notes` = the synonymy
rationale.  6 Claude annotations, contiguous, merge_metric = 0,
zero flags.

**This is therefore a legitimate treatment shape**: a
morph-scoped description under an accepted name, where the type
material belongs to the synonym.  A detector that expects a
description to cover the whole organism would false-fire on it.
It is filed as a pathology only for the text-layer defect; the
structure is a poster-child-grade extraction.

**Fix angle**: scope the separator to block-level boundaries
first — insert `\n` (or a space) between a `<sec>`'s `<title>`
and its following children, and between sibling block elements.
That alone removes `Descriptionof`, `Typeof` and `Notes.The`.
Handle the inline taxon-name case separately with a rule keyed
on `content-type="taxon-name"` children (and the TaxPub
`<tp:taxon-name-part>` equivalent), inserting a space between
adjacent parts.  **Do not** insert separators at every element
boundary: `H<sub>2</sub>O` and similar must stay joined.

Tracked as **D6** in the Detector backlog.

**Re-extraction required.**  Unlike a detector change, fixing
this changes `article.txt` and therefore every downstream field
and every stored `*_spans` offset.  Sequence it with a planned
re-extraction, not as a hot patch.  Note that the
re-extraction must also rewrite every `*_spans` offset, since
those are indexed to `article.txt.ann` (§16) and changing
`article.txt` changes what the annotator emits.


#### 15.1 §15 relocates whole fields, and that blinds the detectors

`taxon_b673586a` (added 2026-08-24), *Cyanosporus miscanthi* from
MycoKeys 107 — Pensoft, so JATS, and the ingest doc carries
`article.xml`.

The operator read it as *"a Diagnosis only… only characters that
differentiate this species"*, and asked whether that was correct.
**It is not.** Only the first ~260 characters are a diagnosis:

> `Diagnosis. Cyanosporus miscanthi is characterized by
> effused-reflexed to pileate tiny basidiomata … basidiospores,
> 4–5 × 1.5–2 µm.`

The remaining **~1 440 characters are a complete description** —
Basidiomata → Pileal surface → Hymenophore → Context → Tubes → Hyphal
system → Cystidia → Basidia → Basidiospores, with full measurements and
`L = 4.2 µm, W = 1.9 µm, Q = 2.2–2.4 (n = 120/4)`.

**`description` is empty. All of it is in `diagnosis`.**

The cause is §15, and the signature is unmistakable — **nine** section
boundaries with the separator gone:

`µm.Basidiomata` · `KOH.Hyphal` · `diam.Cystidia` · `µm.Basidia` ·
`connection.Basidiospores` · `rot.Brown` · `Holotype.China`

Even the binomial lost its internal space (`Cyanosporusmiscanthi`), as
did ten accession numbers (`ITSPP479786` for `ITS PP479786`).

**So §15 is not only a text-corruption problem — it relocates entire
fields.** With the boundaries gone the classifier sees one block and
labels it once, and everything downstream of that heading lands in the
wrong field.

##### The second-order harm: an empty `description` disables the suite

This treatment fires **zero flags**. Not because it is clean — it has a
complete description in the wrong field, and its spore span is
mislabelled — but because **almost every signal reads `description`**.
`tail_clipped`, `desc_starts_mid_sentence`, `mid_body_description_header`,
the merge metric and `authored_binomial` all take `desc`, and all
short-circuit on empty text.

So §15's field-collapse is doubly harmful: it loses the field *and*
blinds the detectors that would have noticed. **An empty `description`
beside a long `diagnosis` is itself a cheap flag**, and nothing
currently emits it.

##### Correction, 2026-08-25: the flag is real but it is not a §15 flag

The paragraph above originally claimed this pairs with D6 — D6 finding
the element-join corruption, the imbalance finding its consequence.
**Measured, that attribution is wrong.** Of the 3 329 treatments with an
empty `description` and a diagnosis over 300 characters:

| | count | share |
|---|---:|---:|
| **genus-rank** (carries a `Type species` marker) | 606 | 18 % |
| **§15 element-join signature** | **12** | **0 %** |
| neither | 2 711 | 81 % |

So `taxon_b673586a` is a real §15 case but a **rare** one — twelve in
the corpus. The imbalance flag catches something much broader, and 81 %
of it is currently unexplained.

**And 18 % of it is legitimate.** For a **genus** entry the diagnosis
*is* the description: `taxon_ec30a049` (*Mycorrhaphium*, Ryvarden's
*Hydnoid Genera*) puts a complete genus account — basidiocarps,
hymenophore, hyphal system, gloeocystidia, basidiospores, distribution —
in `diagnosis`, with `description` empty, and there is nothing wrong
with the text. Its prose is clean, with no join damage.

So any flag on this imbalance **must exempt genus-rank entries**, which
the `Type species` marker identifies for free (D14). Without that
exemption it starts at 18 % false positives on a known, nameable class.

**The consumer problem survives the exemption**, and is the reason to
keep the flag at all: morphology sitting in `diagnosis` is invisible to
anything reading `description`, whatever the rank convention says. That
is a routing question for downstream code, not a defect in the
treatment.

### 16. `*_spans` are indexed to `article.txt.ann` (not a defect)

**This section previously reported a high-severity data
defect — that stored `*_spans` offsets did not locate their
own text in 86 % of treatments.  That was wrong, and it was
a lookup error on my part.  Withdrawn 2026-08-21.**

**What is actually true.**  `*_spans` character and line
offsets are indexed to **`article.txt.ann`**, the
YEDDA-annotated file, as every treatment's
`attachment_name` says.  They are correct there.  Verified on
taxon_4b89d160: its stored span `[175572:175765]` lands
exactly on

```
[@Type species: Stylonectria applanata Höhn. 1915.
Stroma thin, whitish or yellow, hyphal or subiculum-like. …
```

and stored `start_line` 4381 is exactly
`Stroma thin, whitish or yellow …`.  `article.txt.ann` runs
**3.65 %** longer than `article.txt` for that document, which
matches the 3.81 % median inflation measured across the
corpus — the `[@…#Tag*]` markup, accumulating.

**Where the file lives, which is the trap.**  The treatment's
`ingest.db_name` says **`skol_dev`**, and `skol_dev` holds
`article.txt`, `article.pdf`, `article.page-headers.json` and
`article.spans.v4.json` — but **not** `article.txt.ann`.  The
annotated file is written by the v4 predictor to the
experiment's *annotations* database, which for production_v4
is **`skol_exp_production_v4_01_00_ann_combined`** (20 928
docs).  Note also that the unversioned sibling
`skol_exp_production_v4_ann_combined` holds only 1 826 docs
and does *not* have most documents — checking that one and
concluding the attachment is missing is the same trap twice.

Resolving a span means: annotations DB + `attachment_name`,
never `ingest.db_name` + `article.txt`.
`django/search/views.py` already does this correctly —
`_collect_ann_db_candidates()` tries the doc's explicit
`annotations_db`, then the ingest DB, then the experiment's
`databases.annotations`, and prefers the stored
`attachment_name` before falling back to `article.pdf.ann` /
`article.txt.ann`.

**Consequences of the withdrawal.**

* **`§12:desc_span_gap` is fine.**  It measures line deltas
  *between* spans within one coordinate space, so the
  inflation cancels.
* **The span-derived magnitudes elsewhere in this memo are
  valid** — taxon_3d0a3c69's 15 833 characters,
  taxon_43a7b19e's 117 682 across 488 paragraphs,
  taxon_3d9f50f8's 4 647 across 28.  The caveats added
  against them are withdrawn.  They are `article.txt.ann`
  distances, about 3.7 % larger than the corresponding
  `article.txt` distances, which does not affect any
  conclusion drawn from them.
* **Trello #401 deep-linking is not broken by this.**
* Nothing here outranks the detector backlog.

**What is worth keeping.**  Only the lookup rule above.  If a
future check wants to confirm span integrity, resolve against
`article.txt.ann` in the annotations DB — sampling against
`article.txt` will report ~86 % failure and mean nothing.

### 16.1 `Span.head` backfill — what actually ran

Recorded 2026-08-24 from `/var/log/skol/`, where the logs
live under `logrotate` `daily`/`rotate 7` and had **already
been truncated once**; the numbers below were recovered from
the `.1.gz` copies with roughly six days to spare.  That is
the whole reason this section exists.

`Span.head` is the fingerprint that makes a wrong-attachment
read loud instead of silent, so a span written before the
field existed verifies *vacuously*.
`fixes/backfill_span_heads.py` resolves each such span once
and records what it found.

| experiment | examined | heads set | treatments | attachment reads | skipped |
|---|---:|---:|---:|---:|---:|
| production_v4 | 81 527 | 567 516 | **81 527** | 17 645 | **0** |
| production_v3_hand *(pre-fix — bugged)* | 73 139 | 33 339 | 7 392 | 5 402 | **65 747** |
| production_v3_hand *(post-fix re-run)* | 73 139 | **527 610** | **58 099** | 9 558 | 7 648 |

**The pre-fix row is the bug, kept deliberately.**
`AttachmentCache` carried a private copy of the
attachment-name probe instead of calling
`span_resolver.candidate_attachments()`, so it never tried
the fallback name and skipped **65 747 of 73 139**
treatments — while exiting 0 and printing a confident
summary. Its skip reasons all read `attachment
skol_exp_production_v3_hand_ann/<id>`. Fixed in `d349b98` by
making `candidate_attachments()` public and shared; the
re-run's skips read `no annotated attachment on …`, which is
a genuinely missing annotation rather than a probe that gave
up.

**This does not license raising `--min-pass-rate 90`.**
`debian/skol.cron` floors production_v3_hand at 90 because
some source documents carry **no annotated attachment at
all** — a different failure from a missing *head*, and one
the backfill cannot touch. The 7 648 remaining skips are
exactly that population. The floor moves when a
`bin/verify_spans` run says so, not by inference from these
counts.

