# Data quality observations — production_v4 model

Notes from a Phase 1 bootstrap-annotation sample of 5 treatments selected
via `bin/select_for_annotation --experiment production_v4 --n 5
--bands low:1,mid:2,high:2 --seed 1` on 2026-06-28.  Four of the five
exhibited issues serious enough to flag for later attention; this file
captures the categories with concrete evidence so future fix work
doesn't start from scratch.

Tracking: see the corresponding Trello item.

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

* **T3** — `description` begins:
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
  single Description block.
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

**Affected treatments**: T3, `taxon_2a9d07e6...`.

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

**Affected treatments**: T1 (vacuously), T2, T3, T4,
`taxon_acd88732...`.

**Likely stage**: layout CRF likely labels formal-citation paragraphs
as `Figure-caption` (T2) or misses them entirely (T4).  Where the
species heading IS labelled, the treatment-grouper's
Nomenclature-recognition rule may be too narrow (T3, T4).

**Severity**: high — without correct Nomenclature, downstream taxon
identification, name-resolution lookups, and per-species aggregation
all fail or fall back to `synthetic_nomenclature`-flagged stubs.

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

**Affected treatments**: T1.

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

### 6. Multiple species merged into one treatment

**Symptom**: one Treatment doc contains descriptive content for
two or more distinct species.

**Evidence**:

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
* **`taxon_173204...`** — discovered 2026-07-01 in the 50-
  treatment run.  Real nomenclature (`Setiferotheca nipponica
  Matsush.`), then a description field containing TWO similar-
  species descriptions concatenated.  Label distribution shows
  the tell-tale doubling: Asci × 2, Ascomata × 2, Ascospores
  × 2, Peridium × 2, plus singletons (Chlamydospores, Mycelium,
  Necks, Subiculum).  Only 12 annotations total — well within
  the single-species range for a rich ascomycete.  **Slipped
  past the merge-metric filter** (metric value 2, threshold 10)
  because the two species share anatomical vocabulary and each
  term appears only ~2 times, below the k=5 count threshold.
  Compact 2-species merges where species are similar
  (congenerics, same family) are a documented blind spot of
  the current metric — see 'Merge-metric limitations' below.
* **`taxon_2a9d07e6...`** — discovered 2026-07-01.  Nomenclature
  `Teratosphaeria dunnii Crous & Carnegie` correctly parsed;
  description contains a SECOND full species description
  (*Teratosphaeria obscuris* with its own formal citation).
  **Two structural markers** that would have caught the merge:
  (1) the `description` field contains the literal string
  `Diagnosis:` twice (once at the top for T. dunnii, once
  mid-body for T. obscuris) — a properly-single-species
  description has one such header at most; (2) the second
  citation would parse cleanly via gnparser as an authored
  binomial, and no legitimate `description` field should
  contain a formal citation (see §1's Description-vs-Diagnosis
  distinction).  **Slipped past the merge-metric filter with
  metric = 0** — 7 annotations total across 6 labels; both
  species are compact enough that no term reaches k=5.  Worst
  of the observed blind spots.
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
      previously-noted `Brumm., spec. llOU.` is the worst
      case, but not universal.  Some embedded citations are
      only lightly corrupted — e.g., `Mycostigtna Jiilich,
      gm . nov.` is one character (`m`→`n`) plus one stray
      space away from `Mycostigtna Jülich, gn. nov.` (§11's
      `gen. nov.` pattern under OCR).  gnfinder + fuzzy
      matching would have partial coverage even here: the
      lightly-corrupted citations parse; heavily-corrupted
      ones don't.  Argues gnfinder detection (§6 idea #2)
      is worth trying even on high-OCR-noise treatments,
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

**Affected treatments**: T3, T5, `taxon_592128a8...`.

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
     per species.  A description containing MORE THAN ONE
     Latin block is a very strong merge signal.  Detection
     is robust to OCR corruption (Latin morphology — endings
     `-us`, `-a`, `-um`, `-orum`, `-arum`, `-ibus`; vocabulary
     `apothecia`, `sessilia`, `ascosporae` — survives typos
     that break binomial parsing).  Cheap to compute
     paragraph-by-paragraph via langdetect / pycld3 / a
     Latin-suffix heuristic.  Would have caught the
     `taxon_572d470e` case cleanly.  Pre-bootstrap; no API
     spend needed.
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
     gnfinder/gnparser install.  Caveat: fails on
     heavily-OCR-corrupted binomials (e.g., `taxon_572d470e`'s
     `spec. llOU.` for `spec. nov.`) — the Latin/English
     signal is more OCR-robust.
  3. **Count section-header keyword repetitions in the raw
     description**.  A single-species treatment has each
     section-header keyword appearing at most once; two or
     more of the same header is a strong merge signal.
     Concrete headers to watch — `Diagnosis:`, `Description:`,
     `Observations:`, `Illustration:`, `Cultural
     characteristics`, `Culture characteristics`,
     `Colonies on`, `Etymology:`, `Habitat:`, `Type:`,
     `Holotype:`.  `taxon_2a9d07e6` had two `Diagnosis:`
     headers; `taxon_592128a8` had three `Observations:`
     headers; `taxon_e74d89b1` had many `Cultural
     characteristics` sections; `taxon_95dbdfb9` had 3
     `Illustration:` + 3 `Description:` pairs (illustrated
     monograph format).
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

**Affected treatments**: `taxon_5b0a8ce7...`; almost certainly
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

## Notes for fix sequencing

These issues are deferred — not blocking Phase 1 bootstrap-annotation
work in `treatments_to_structured/`.  Suggested triage order when
the work is picked up:

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
### 9. Corrupt OCR text (U+FFFD runs in `diagnosis` / `description`)

**Symptom**: a Treatment field contains long runs of
`�` (U+FFFD REPLACEMENT CHARACTER), the Python decoder's
substitute for bytes it can't interpret as UTF-8.  Visible in
Fauxton as long strings of replacement-glyph boxes.

**Evidence**:

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

* **`taxon_acd88732...`** — discovered during 2026-07-01
  hand-inspection.  `description` field verbatim:
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

**Affected treatments**: `taxon_01a01c54...`; likely
representative of a broader pattern in taxonomic papers that
propose a new genus alongside its type species.

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
  * **`taxon_23d479f4`** — Description field contains
    what looks like carbon-source utilization data
    (opens with `glycerol, methanol, hexadecane,
    erythritol, levulinic acid, …`) rather than a
    normal anatomical description.  Missing lines at
    both top AND bottom.  Nomenclature is real
    (`synthetic_nomenclature = False`) and the
    treatment appears to be a single species; only the
    Description content is off.  Two interpretations
    the operator floated:
      (a) content is legitimate Cultural characteristics
          (some asexual moulds and yeasts include
          detailed physiological / carbon-source
          assays in their treatments) that landed in
          Description via a lost label.  A label-aware
          assembler would route it to a `Cultural
          characteristics` field.
      (b) content is from experimental Methods /
          Results sections not intended as taxonomic
          content, extracted by mistake.
    Distinguishing between (a) and (b) requires source
    inspection; the extracted content alone reads
    consistent with either.  Detection: `§10:mid_sentence`
    fired correctly (leading `glycerol,` is lowercase).
    Adds a distinct sub-shape to the §12 leak list —
    experimental/physiological content in Description,
    distinct from the anatomical-block leaks
    (Diagnosis-into-Description, Materials-examined-into-
    Description) and from the figure-caption leak in
    taxon_ea7b0ed7.
  * **`taxon_ea7b0ed7`** — a figure caption landed
    embedded mid-Description instead of the doc's
    `figure_captions` field.  The treatment is otherwise
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

**Timing**: overlaps with the pipeline restructure in
`~/.claude/plans/cozy-forging-locket.md` (per-family Python
modules).  Likely a Phase 3+ candidate after v4 lands —
useful to record now so §6 fix work explicitly weighs
"tighten the merge detector" vs "fix assembly to not need one."

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
