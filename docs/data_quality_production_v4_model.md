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

**Affected treatments**: T1, `taxon_fb7bd18d...`.

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

## §0.5. Poster-child treatments (reference)

The rest of this memo catalogs what goes wrong.  For
balance and calibration, this section records
extraction-poster-child treatments the operator has
called out — what a clean, correctly-extracted single-
species treatment looks like.  Useful when explaining
"what right looks like" to future reviewers, and as
regression targets: any future detector or assembler
tightening MUST NOT surface these as false positives.

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
       segment classifier), not a regex.
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
