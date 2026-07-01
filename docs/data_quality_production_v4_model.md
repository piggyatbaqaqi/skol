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

**Affected treatments**: T3.

**Likely stage** (best guess, not investigated): the layout CRF
labelled these short numbered heading lines as `Description`
continuations, OR the treatment-grouper failed to split on them.
Either way the symptom is downstream — the heading text never made
it to a Nomenclature slot.

**Cascade effect**: T3 was also flagged with a synthetic Nomenclature
stub, presumably because the first paragraph the grouper saw
already had `Description` label rather than `Nomenclature`.

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
  a single Nomenclature.  Critical structural detail noted by
  the operator: **each constituent species treatment has its
  normal sections internally** — description, diagnosis, etc.
  appear correctly per-species — so the grouper broke purely at
  the inter-species boundary, NOT inside individual treatments.
  Argues the layout CRF labelled the per-species sections
  correctly but the treatment-grouper failed to split when one
  Nomenclature was immediately followed (after section labels)
  by another Nomenclature.  Easier failure mode to fix than the
  T3/T5 cases because the section structure is intact.

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
spending API budget on them.

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

**Affected treatments**: (fill in)

**Likely stage** (best guess): treatment-grouper boundary
detection fails to recognize the transition from in-treatment
description prose to in-document key prose.  Possibly related
to §6 (multi-species merge) — when a treatment is sliced from
a flora chapter that contains both a species description AND
the genus-level key, the slice may include both without a
boundary signal.

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

**Affected treatments**: `taxon_cda95f9f...`; likely others —
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

**Affected treatments**: `taxon_acd88732...`; unknown
corpus-wide rate — worth a scan.

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
