# GoldenGATE Imagine: pipeline structure and lessons for SKOL

Source read: `/data/piggy/src/github.com/plazi/GoldenGATE-Imagine`
(HEAD `343dc68`), comprising the application shell + plugin sources
(~37 kLOC Java), the shipped configuration, and the 73-page
`GoldenGATE_Imagine_V1_end_user_manual.pdf` (DRAFT 20161104, Miller,
Agosti, Sautter, Catapano & Klingenberg).

Note on scope: this repository is the *editor shell and its basic plugins*.
The taxon-specific analysers ("Mark Treatments", "Treatment Structure",
"Mark Materials Citations", RefParse, …) are not in this tree — they ship as
downloadable **configurations**. What *is* in the tree is the framework those
analysers plug into, and the framework is where the interesting design is.

The config host baked into `files/imagine/ConfigHosts.cnfg` —
`plazi.cs.umb.edu` — is dead (NXDOMAIN as of 2026-08-27). The live server is
**`tb.plazi.org`**; see [§6](#6-obtaining-the-configuration-the-analyser-plugins)
for the current URLs, the wire protocol, and a working install path. Note
also that the source in this repo (HEAD 2016) is *older* than the shipped
configuration (2024) — see the `MATCHER`/`FILTER` discrepancy noted in §3.1.

---

## 1. The core architectural idea

GG-Imagine is not a pipeline that produces an output. It is a **single
mutable document object that a sequence of tools progressively enriches**,
with a human in the loop at every step. There is no "run" and no "result
file" until the user chooses to export.

The document model (`de.uka.ipd.idaho.im.*`, `lib/ImageMarkup.jar`):

| Layer | Objects | What it is |
|---|---|---|
| Physical | `ImPage`, `ImWord` (bounding box, font, size, bold/italic) | Where ink sits on the page |
| Layout | `ImRegion` typed `column`, `block`, `line`, `paragraph`, `table`, `tableRow/Col/Cell`, `image` | Geometric grouping |
| Logical | **text streams** — each `ImWord` has a `predecessor`/`successor` and a `textStreamType` | Reading order, independent of geometry |
| Semantic | `ImAnnotation` — typed span from first word to last word | `treatment`, `taxonomicName`, `bibRef`, `materialsCitation`, `caption`, … |

The **text stream** is the load-bearing abstraction. `ImWord.textStreamType`
is one of `mainText`, `caption`, `footnote`, `table`, `pageTitle`,
`artifact`, `deleted`. Page headers, captions, footnotes and OCR junk are
*cut out of the main stream into their own streams*, rather than deleted or
flagged in place. Once that surgery is done, "the body text of the article"
is a single linear token sequence that crosses column and page breaks
correctly, and every downstream analyser reads that sequence and never has
to think about layout again.

Semantic annotations are anchored to **word IDs**, not character offsets, so
re-flowing paragraphs or fixing a hyphenation does not invalidate existing
markup.

### Normalization levels

`ImageMarkupToolManager.java:696-706` — every tool declares which *view* of
the document it wants, and the framework wraps the image-markup document as
a GAMTA XML document at that level:

- `Raw` — words strictly in layout order
- `Words` — layout order, but de-hyphenated
- `Paragraphs` — logical paragraphs kept together
- `Text Streams` — logical streams one after another

So the same document is simultaneously a page image, a layout tree, and a
clean linear text — and each analyser picks the projection it needs.

---

## 2. The workflow, step by step

From the manual's "The markup process" chapter. The manual is explicit that
after structure detection, **"generally, all the tools can be used in either
sequence"** — the order below is the recommended one, not an enforced one.

Legend: 🤖 automatic · 👤 manual · 🤝 automatic-then-confirmed (the tool
proposes, the user steps through with **OK & Next**)

### Stage 0 — Ingestion 🤖

Open a PDF as *born-digital*, *scanned*, or *unknown* (the last lets
GG-Imagine decide). Born-digital: decode embedded fonts, extract words with
positions. Scanned: render page images, run OCR (`ImageMarkupOCR.jar`).
Either way the output is words-with-boxes-and-fonts, then automatic
segmentation into columns → blocks → lines → paragraphs.

Alternatively, reopen an existing `.imf` (Image Markup File) from disk or
**check one out of the Plazi server**, which locks it for other editors
until you close it. Saving uploads a *delta*, not the whole file.

### Stage 0b — Style template lookup 🤖

`DocumentStyleProvider.java` — before any detection runs, GG-Imagine tries
to identify *which journal/series layout* this document is, by matching
**anchors**: a `.docStyle` resource declares N anchors, each a
`(bounding box on page 1, min/max font size, bold?, italics?, all-caps?,
regex)` tuple. A style is accepted when **more than two-thirds of its
anchors match** (`bestDocStyleAnchorMatchScore = 0.66f`, with a tie-break on
absolute match count). Failing that, it falls back to matching the style
*name* against the document's bibliographic metadata (journal name, then
year — sorted descending so the closest style *at or before* the document's
year wins — then publication type).

A matched style supplies layout parameters that the detectors consult
instead of guessing:

```
layout.page.{first|odd|even}.headerAreas     layout.contentArea
layout.page.{first|odd|even}.number.area     layout.minBlockMargin
layout.page.*.number.{fontSize,isBold,pattern}
layout.caption.{minFontSize,maxFontSize,startIsBold}
layout.caption.{above,below,beside}{Figure,Table}
layout.footnote.{area,minFontSize,maxFontSize}
layout.minColumnMargin  layout.minRowMargin
style.heading.levels + per-level font/emphasis specs
```

### Stage 1 — Detect Document Structure 🤖 (then 👤 repair)

`DocumentStructureDetectorProvider.java:278` — one automatic pass, whose
internal stage list (from its own progress messages) is:

1. gather per-page data (header areas from the style, or page-edge blocks)
2. score and select page numbers; **check the page-number sequence across
   the document**; fill in missing page numbers from the sequence
3. detect page headers (frequent words at page top/bottom, split odd/even)
4. correct empty tables; detect tables (column/row margins)
5. detect OCR artifacts in images
6. compute main-text font size
7. detect captions; detect footnotes
8. index paragraph-end words; **de-hyphenate line breaks**
9. **merge interrupted paragraphs** (across column and page breaks)
10. identify caption *target* areas (which figure/table each caption is for)
11. merge tables within pages, then across pages
12. mark caption citations ("Fig. 3" in running text) and link them
13. mark headings and emphases; assess heading hierarchy

Two things are worth stealing wholesale. First, **cross-page global
reasoning**: page numbers are not decided per page, they are decided by
fitting a sequence over the whole document and back-filling. Second, the
**dual-path pattern**, visible at `:1309`:

```java
//  extract headings using heuristics
if (headingStyles == null) { markHeadings(...); assessHeadingHierarchy(...); }
//  extract headings using style templates
else for (...) markHeadings(page, dpi, headingStyles, pm);
```

If a style template is available, use its declared parameters; otherwise
fall back to document-internal heuristics. Same code path, same output,
different confidence.

**Manual repair loop 👤.** The manual's worked example: the detector put a
footnote and a page number into the body text. The user turns on the "Word"
display layer, *sees the grey line tracing word order*, draws a box round
the offending text, and picks **Mark Page Header** — which cuts those words
out of the main text stream and the reading order visibly re-routes.

This is the single most important UI idea in the whole system: **the
intermediate representation is directly visible and directly editable, and
the effect of a correction is immediately visible in the same view.**

### Stage 2 — Mark Taxon Names 🤖 + 👤

Automatic detection of taxonomic names throughout the document, then
**atomization** (genus / species / subspecies / authority / year into
attributes) and **reconciliation against external services** — Catalogue of
Life, GBIF, IPNI — to fill in higher ranks. The manual warns this requires
network access. `Parse Taxonomic Names` is offered as an optional
verification pass ("in most cases, it is not necessary").

Linking to nomenclatural acts is **entirely manual**: the user pastes a
ZooBank LSID as an `LSID-ZBK` attribute. The manual is candid that there is
no agreed identifier scheme to automate against.

### Stage 3 — Tables 🤝

Detected automatically; the manual's example immediately shows three failure
modes (bad row names from stacked digits, unrecognised table note, two rows
merged into one) and the corresponding repairs: remove table region, redraw,
**Mark Table**, **Mark Table Note**, **Split Table Row**, **Connect Table
Rows** across a table continued on a later page. When connecting rows whose
labels differ, the tool *asks for confirmation rather than refusing*.

### Stage 4 — Parse Bibliography 🤝

Precondition the user is told to check **first**: each reference must be its
own paragraph. Paragraph errors are fixed by hand (remove / split / mark
region) *before* running the parser. Then RefParse assigns tokens to fields
(author, year, title, journal, volume, pagination, DOI) and the user
confirms or reassigns per reference. Reassignment is
click-the-field-label-above-the-highlighted-text; a token can belong to only
one field, so conflicting highlights must be removed first.

Merged references (the parser ran two into one) are split *after the fact*
in the main view: highlight the start of the second reference →
**Split bibRef Before** → re-run **Parse Reference** on the new fragment.

### Stage 5 — Document Metadata 🤝 / 👤

Three routes, in `DocumentMetaDataEditorProvider.java`:
- **Extract** — show the first lines of the document, user highlights text
  and clicks the field button; the field's outline goes red → green as it
  is filled.
- **Search** — query the Plazi repository for similar documents (using the
  already-parsed references) and offer their metadata to adopt wholesale.
- **Manual** — type it.

Plus a **Validate** button that checks required fields are populated. Field
outlines coloured by filled/unfilled is a cheap, extremely legible
completeness display.

### Stage 6 — Mark Treatments 🤝 or 👤

Two alternatives, side by side:
- **Mark Treatments tool** — proposes treatment boundaries, user steps
  through with **OK & Next**. The same dialog also assigns other document
  subsections, with `multiple` as the explicit escape hatch for "doesn't fit
  any category".
- **Manual** — click the first word → *Start Annotation* (a red banner
  reminds you an annotation is open) → scroll to the last token → pick
  `treatment` from the type dialog.

Afterwards the user is told to re-check paragraph boundaries, because
treatment starts are exactly where the automatic paragraph grouping tends to
fail; **Revise Block Paragraphs** offers canned fixes ("make each line a
separate paragraph").

### Stage 7 — Treatment Structure 🤝

Per-paragraph classification within each treatment, stepped through with
**OK & Next**. The label set (from `files/imagine/GoldenGATE.cnfg`,
`AEP.ASS.subSubSection.type.v_a_l_u_e_*`, restricted-value list):

```
nomenclature  description  diagnosis  discussion  distribution
etymology     biology_ecology  materials_examined  multiple
```

plus `reference_group` for the literature-citation block at the head of a
treatment. Compare SKOL's 12 tags — this is essentially the same label set,
minus a few, at paragraph rather than line granularity, and *always
human-confirmed*.

### Stage 8 — Mark Materials Citations 🤝 (three sub-stages)

Explicitly "a complex, multi-part tool", and the clearest example of
decomposing one hard decision into a chain of cheap human confirmations:

1. Find text matching **collection-code** patterns → user ticks the true
   positives.
2. Decide which **paragraphs** contain materials citations, and tag country
   / major region within them → user unticks paragraphs ("Exclude
   paragraph") and removes bad country/region spans.
3. Establish **record boundaries** — one line per specimen record → user
   applies *Split materialsCitation*, *Remove annotation*, *Merge
   materialsCitation annotations* until each highlighted line is exactly one
   record.

Note the manual's aside: this tool **does not use** the
`materials_examined` sections marked in Stage 7. Two independent paths to
the same conclusion, deliberately not coupled.

### Stage 9 — Parse Materials Citations 🤝

Field assignment within each record (collector, date, locality, elevation,
coordinates, specimen count, type status, …), same UI as Parse Bibliography.
`backReference` is the designated marker for records that inherit values
from the previous record — an explicit representation of "same as above"
rather than silent duplication.

### Stage 10 — Export 👤

XML at any stage; and once the minimum bar is met — **(1) document metadata
present, (2) treatments marked, (3) nominate taxon name marked** — TaxonX,
Darwin Core Archive, figures + captions, table grids as delimited text, or
upload to the Plazi treatment repository.

---

## 3. The framework mechanisms behind the workflow

These are the parts that generalise, independent of taxonomy.

### 3.1 Tools as data, not code

`ImageMarkupToolManager.java` — an *Image Markup Tool* is a `.imTool`
settings file:

```
LABEL, TOOLTIP, <name>.help.html   — user-facing text
PROCESSOR_NAME / PROCESSOR_PROVIDER_CLASS  — the analyser to run
XML_WRAPPER_FLAGS                  — which normalization level it sees
LOCATION                           — Tools menu / selection action / both
PRECLUSION<n>FILTER + MESSAGE      — preconditions
FILTER<n>                          — applicability predicates
```

Adding a step to the pipeline is authoring a settings file, not writing
Java. The analyser itself is a GoldenGATE `DocumentProcessor` that can
equally be a rule set, a dictionary, a regex cascade, a *pipeline* of other
processors, or code.

The real `TreatmentTagger.imTool` from the live `Default.imagine`
configuration, in its entirety:

```
LABEL              = "Mark Treatments";
LOCATION           = "Tools Menu";
PRECLUSION0FILTER  = "not(./taxonomicName)";
PRECLUSION0MESSAGE = "W:Taxonomic names are not marked in the document, so
                      treatment detection might produce many errors";
PROCESSOR_NAME     = "TDS.TreatmentTaggerOnline.analyzer@...AnalyzerManager";
TOOLTIP            = "Mark taxonomic treatments";
XML_WRAPPER_FLAGS  = "1F";
```

That is the entire definition of the pipeline's central step. Note the
`W:` prefix — running treatment detection before taxon names are marked is a
*warning*, not an error: it degrades quality but is allowed.

`MarkMaterialsCitations.imTool` is the same shape but points at a
`MaterialsCitations.pipeline`, which is itself an ordered list of four
sub-processors — the three-stage tool described in §2 Stage 8:

```
INTERACTIVITY_LEVEL = "Feedback only";
PART_0 = MaterialsCitationsPreprocessor.filteredDp
PART_1 = MaterialsCitationsAbbreviationTagger.filteredDp
PART_2 = AbbreviatedMaterialsCitationTagger.filteredDp
PART_3 = MCT.MaterialsCitationTaggerOnline.analyzer
```

`INTERACTIVITY_LEVEL = "Feedback only"` is the knob that decides how much of
this chain stops to ask the user.

**Version caveat.** The shipped 2024 configuration uses `MATCHER<n>` where
this repo's 2016 `ImageMarkupToolManager` reads `FILTER<n>`
(`FILTER_ATTRIBUTE = "FILTER"`). The attribute was renamed at some point
after this source snapshot. Read the source in this repo as *approximately*
current, not exactly.

### 3.2 Preconditions with human-readable messages, and override

`DpImageMarkupTool.getPrecludingError()` — each tool carries a list of GPath
(XPath-like) expressions over the wrapped document, each paired with a
message. Before running, they are evaluated. A hit prefixed `W:` is a
*warning*; anything else is an *error*. Either way the user is shown:

> The document does not seem to be fit for **Parse Bibliography**:
> *<message>*. Executing Parse Bibliography anyway might produce undesired
> results. Proceed?

...and can proceed regardless. Errors short-circuit; warnings are collected
and only reported if no error fires. This is how "you must fix paragraph
structure before parsing the bibliography" is *encoded* rather than merely
documented in the manual — while still never blocking an expert who knows
better.

The mirror-image mechanism, `FILTER<n>`, decides whether a tool even
*appears* in the right-click menu for the current selection.

### 3.3 Reactive consistency maintenance

`ReactionProvider` (`plugins/ReactionProvider.java`) — plugins subscribe to
`typeChanged` / `attributeChanged` / `regionAdded` / `regionRemoved` /
`annotationAdded` / `annotationRemoved`, each with an `allowPrompt` flag
saying whether this context permits asking the user something.

`CaptionCitationHandler.java` is the worked example: when a caption's span,
target box, target page or URI changes, it finds every in-text citation of
that caption and rewrites their attributes to match. The user edits one
thing; the derived facts repair themselves.

Compare the pull model (re-run the whole analysis after every edit) — the
push model means a manual correction *costs* only its own consequences.

### 3.4 Everything is undoable and atomic

`beginAtomicAction` / `endAtomicAction` wrap each compound edit;
`GoldenGATE.cnfg` configures undo depth (`DDP.UNDO_MAX_ITEM_COUNT = 50`,
`DDP.UNDO_MAX_ITEM_AGE = 1800000`). A user who is being asked to make
thousands of judgement calls must be able to be wrong cheaply.

### 3.5 QA views over the markup

`ImageMarkupObjectListProvider.java` — a filterable table of annotations,
described in its own javadoc as *"helpful for sorting out annotations, and
for finding ones with specific error conditions, e.g. a lacking
attribute"*, with **pre-configurable saved listings** exposed as menu items
in administrative mode. Alongside it, `XmlViewerProvider` shows the nested
annotation structure as editable XML, and the display-control panel toggles
every layer (word / line / block / column / paragraph / region, and each
annotation type, each with its own colour from `GgImagine.cnfg`).

The **OCR checker** (`OcrCheckerProvider.java`) deserves special mention: it
validates OCR by *re-rendering the recognised text in the recognised font
and measuring the overlay against the original word image*. A self-check
that needs no ground truth. The UI counterpart is a transparency slider that
fades between the OCR text and the underlying page image.

---

## 4. What SKOL could take from this

Ordered by expected value.

### 4.1 Journal style templates keyed by anchor match — highest value

SKOL currently treats every document with the same extraction logic and
absorbs per-publisher layout variation as classifier noise. GG-Imagine's
`.docStyle` mechanism is directly portable and does not require a UI:

- A style is a small parameter file. Matching is (a) anchor regexes at fixed
  page-1 boxes with font constraints, ≥⅔ must match, or (b) fall back to
  journal-name + year + publication-type from metadata. SKOL already *has*
  the metadata for most documents, so route (b) is nearly free.
- The pattern that matters is `if (style != null) use_style(); else
  heuristics();` — one code path, parameterised. SKOL could start with
  per-source parameters for the things that already misbehave: heading
  detection, running-head stripping, the `MISC_GAP_LIMIT` in `taxon.py`,
  section-label priors per journal.
- This connects to the deferred **Persoonia → Naturalis source swap**
  (Trello #404): the vols 1–19 whole-volume scans are exactly a case where a
  per-source style ("no title, no DOI, treatments start at pattern X") would
  let the extractor handle the class rather than the instance.

### 4.2 Preconditions as data, with messages and override

SKOL's stages fail late and diffusely — a bad paragraph segmentation
surfaces as a bad treatment three stages later. GG-Imagine's preclusion
filters are a cheap fix: each pipeline stage declares predicates over the
document's current state, each with a human-readable message, each
classified error-vs-warning. In an automated pipeline the "user" is the
run: an error precludes the stage and records *why* on the document; a
warning proceeds and records a flag. That gives a per-document,
per-stage provenance trail of exactly which preconditions were violated —
which is the review queue, for free.

This is a better shape than a boolean `is_valid`, because the message and
the severity travel with the record.

### 4.3 Text streams as the intermediate representation

SKOL's `plaintext_from_jats()` / PDF extraction flattens to lines and then
classifies lines. GG-Imagine instead **routes** each word into one of a
small set of named streams (`mainText`, `caption`, `footnote`, `table`,
`pageTitle`, `artifact`, `deleted`) and only then runs semantics over the
main stream.

The equivalent for SKOL is to make "which stream does this line belong to"
an explicit, inspectable, correctable first-class field on every line —
rather than an implicit consequence of the 12-tag classifier having to
model captions, running heads and table debris as just more label classes.
Two benefits: the treatment classifier sees a cleaner sequence, and stream
misrouting becomes a separately measurable error class instead of being
folded into label accuracy.

This is the same instinct as the **§12 segment-labels-to-assembly** idea
already floated — pass structural information through rather than
flattening it — one level lower down.

### 4.4 Decompose one hard decision into a chain of cheap confirmations

Mark Materials Citations is the template: *find candidates by pattern →
confirm/reject* → *narrow the scope to paragraphs → confirm/reject* →
*establish boundaries → split/merge*. Each step is a decision a human can
make in under a second and a model can make with calibrated confidence.

For SKOL's annotation rounds this reframes the ask. The current brat rounds
present a full labelled document and ask "is this right?" A chained design
would present three separate, much cheaper queues: *are these lines
treatment starts?*, *do these paragraphs belong to this treatment?*, *where
does section X end?* — each of which produces training signal for a
narrower, more learnable sub-decision. Round 3 (the random sample) already
tells us label quality is precision 100 % / recall 99 %; the residual errors
are structural, and structure is what chained confirmation targets.

### 4.5 Reactive repair after a correction

When a hand annotation lands in SKOL (via brat ingest), the derived
artefacts downstream of it — treatment boundaries, assembled taxa, embedded
sections — are not recomputed. GG-Imagine's `ReactionProvider` is the
pattern: declare, per derived artefact, what upstream change invalidates it,
and repair *only that*. `CaptionCitationHandler` is 840 lines to keep one
kind of cross-reference in sync — a measure of how much this actually costs,
but also of how much it is worth to them.

For SKOL the minimum viable version is a dependency note on each derived
record ("this treatment assembly derives from lines L1..Ln of doc D at
classifier version V"), so that a corrected line can invalidate exactly the
records that consumed it. This overlaps with the **source_anchors** work
(Trello #401) — the anchors already record *where* a record came from;
adding *what it was derived from* makes reactive repair possible.

### 4.6 A no-ground-truth self-check

`OcrCheckerProvider` validates OCR by re-rendering and measuring pixel
overlay — an internal consistency check that costs nothing per document and
needs no annotation. SKOL's analogue: reconstruct the source text from the
extracted treatment structure and diff it against the original plaintext.
Any dropped or duplicated span shows up immediately, on every document,
without a golden set. Given the finding that 36.7 % of ingest docs share a
DOI and dedup is missing (Trello #405), a cheap round-trip check that flags
"this treatment's text does not account for lines 40–95 of its source" is
the kind of thing that would have surfaced the problem earlier.

### 4.7 Saved QA listings as review queues

`ImageMarkupObjectListProvider`'s pre-configured filtered listings —
"annotations of type T lacking attribute A" — are the direct ancestor of a
SKOL review queue. SKOL has the Django app and CouchDB views to do this
already; what is missing is the *curated set of saved error-condition
queries*, treated as a first-class, versioned artefact rather than as
ad-hoc queries typed at investigation time.

### 4.8 What is deliberately *not* worth copying

- **Per-document human curation as the throughput model.** GG-Imagine
  targets ~1 M treatments at human speed with many curators; SKOL targets
  automation. The lesson is the *shape* of the decisions, not the staffing.
- **Order-independence of tools.** The manual's "all the tools can be used
  in either sequence" is a UI affordance that costs them correctness (hence
  the preclusion machinery bolted on to recover ordering constraints). An
  automated pipeline should just declare the DAG.
- **Anchoring to word IDs.** Right for a mutable editing session; SKOL's
  line-oriented, re-runnable pipeline does not need it and would pay for it.

---

## 5. One-page summary of the automated/manual split

| Step | Automated | Human |
|---|---|---|
| PDF → words | font decoding or OCR, page segmentation | choose born-digital vs scanned; fix OCR words via image-overlay slider; fix font/symbol mappings globally |
| Style match | anchor match ≥⅔, else name+year+type | author the `.docStyle`; override the match |
| Document structure | page numbers (sequence-fitted), headers, captions, footnotes, tables, de-hyphenation, paragraph merging, caption↔figure linking, heading hierarchy | re-mark page headers/footnotes/artifacts; merge/split blocks and paragraphs; redraw tables; split/connect table rows |
| Taxon names | detect, atomize, reconcile vs CoL/GBIF/IPNI | verify combinations; add ZooBank/HNS LSIDs by hand |
| Bibliography | RefParse field assignment | ensure one reference per paragraph *first*; reassign fields; split merged refs |
| Metadata | extract-from-first-lines; search repository for a match | select/correct fields; Validate |
| Treatments | propose boundaries | confirm each (OK & Next), or mark fully by hand |
| Treatment structure | propose per-paragraph labels from a 9-value restricted set | confirm each |
| Materials citations | collection codes → paragraphs+country/region → record boundaries | tick/untick at each of the three stages; split/merge records |
| Materials parsing | field assignment | reassign; mark `backReference` |
| Export | TaxonX / DwC-A / XML / SRS upload | must satisfy: metadata + treatments + nominate name |

The consistent pattern: **automation proposes at every stage, a human
confirms at every stage, and the intermediate representation is visible and
editable at every stage.** No step is allowed to be a black box whose output
can only be accepted or rejected wholesale.

---

## 6. Obtaining the configuration (the analyser plugins)

Verified 2026-08-27 by reading the loader source and probing the live server.

### 6.1 The old hosts are dead; the live one is `tb.plazi.org`

`files/imagine/ConfigHosts.cnfg` and `UpdateHosts.cnfg` ship these:

```
http://plazi.cs.umb.edu/GgServer/Configurations/     -> NXDOMAIN
http://plazi2.cs.umb.edu/GgServer/Configurations/    -> 301 to https, then dead
http://plazi.cs.umb.edu/GgServer/Updates             -> NXDOMAIN
```

The GgServer instance now runs at **`tb.plazi.org`** (TreatmentBank, Tomcat
9). Same servlet, same paths:

```
https://tb.plazi.org/GgServer/Configurations/    (200)
https://tb.plazi.org/GgServer/Downloads/         (200, browsable listing)
https://tb.plazi.org/GgServer/Updates            (200)
```

### 6.2 The wire protocol is plain static HTTP

From `ConfigurationUtils.getRemoteConfigurations()` and
`UrlConfiguration` (both in `lib/GoldenGATE.jar`, sources included in the
jar), with the constants from `GoldenGateConfiguration`:

```
FILE_INDEX_NAME  = "files.txt"
TIMESTAMP_NAME   = "timestamp.txt"
DESCRIPTOR_FILE_NAME = "configuration.xml"
```

- `GET <host>/files.txt` → newline-separated list of configuration names
- `GET <host>/<ConfigName>/timestamp.txt` → a millisecond epoch (used only
  for update comparison)
- `GET <host>/<ConfigName>/files.txt` → newline-separated list of every file
  in the configuration
- `GET <host>/<ConfigName>/<relative/path>` → that file, verbatim

No authentication, no API, no manifest format beyond newline-separated
paths. **Any static web server can host a GoldenGATE configuration.**

`GET https://tb.plazi.org/GgServer/Configurations/files.txt` currently
returns 15 configurations:

```
Default.imagine        Online.imagine         PensoftImporter
SearchPortalEditor     ServerBatch.editor     ServerBatch.imagine
ServerBatch.xpedite    Slim.mat.imagine       TaxonX-PDF.editor
TaxonX.editor          TaxonX.markupWizard    WebEditor
WebServices.Taggers    ZooTaxa.markupWizard   pro-iBiosphere.editor
```

`Default.imagine` is the one the manual tells users to select.
`ServerBatch.imagine` is the headless/batch variant.

### 6.3 …but the in-app "download from host" path is broken on this server

Individual files serve fine:

```
GET /GgServer/Configurations/Default.imagine/files.txt      -> 200, 48 KB, 833 entries
GET /GgServer/Configurations/Default.imagine/GoldenGATE.cnfg -> 200, 24 KB
```

but **`timestamp.txt` 404s for every configuration** (checked
`Default.imagine`, `Online.imagine`, and the host root). Since
`getRemoteConfigurations()` builds each `ConfigurationDescriptor` by calling
`getTimestamp(configHost + fileName)` inside a `try`, an `IOException` there
makes it log and *skip* that configuration. So even after repointing
`ConfigHosts.cnfg` at `tb.plazi.org`, the configuration selector would list
**nothing** from the host. Do not spend time debugging that; use 6.4.

### 6.4 What actually works: the Downloads directory

`https://tb.plazi.org/GgServer/Downloads/` is a plain browsable listing and
is **actively maintained** — the current builds are dated 2026-04-20.
Relevant assets:

| File | Size | Last modified | What it is |
|---|---|---|---|
| `GgImagine+Default.imagine.zip` | 110 MB | **2026-04-20** | **app + Default.imagine config — start here** |
| `GgImagine.zip` | 92 MB | 2026-04-20 | app only, no configuration |
| `Default.imagine.zip` | 19 MB | 2024-05-29 | **the configuration alone** (see 6.5) |
| `GgImagine-Default.imagine.zip` | 92 MB | 2024-06-09 | older `VersionPacker`-style bundle; the URL the manual cites |
| `GgImagineBatch.zip` | 356 kB | 2019-04-03 | headless batch runner |
| `Slim.{bast,mat,tax,treat}.imagine.zip` | ~8–11 MB | 2023-02-16 | trimmed single-purpose configurations |
| `GgXpedite+Default.xpedite.zip` | 20 MB | 2026-04-20 | the Xpedite variant |
| `PdfChunkerTool.zip` | 7.6 MB | 2024-09-11 | standalone PDF chunker |

The naming convention comes straight from
`VersionPacker.getVersionZipName()`:
`"GgImagine" + ("-" + configName)`, with `-Full` for the local master
configuration. The `+` variants are a newer build's convention for
app-plus-config.

### 6.5 Installing a configuration from a ZIP (no network needed)

`ConfigurationUtils.getZipConfigurations()` scans `<install>/Configurations/`
for any `*.zip` containing a `timestamp.txt` at its **root**, and lists it in
the selector under the host label **"Local ZIP"**. Downloaded and verified:

```
Default.imagine.zip  ->  19,664,645 bytes, 750 entries
root:  timestamp.txt (1706839301935 = 2024-02-02)
       configuration.xml  files.txt  GoldenGATE.cnfg
       GgImagine.menus.cnfg  GgImagine.contextMenu.cnfg  README.txt
```

So the working install is:

1. `GgImagine.zip` (or the repo's GitHub release) → unpack anywhere but `/`
2. drop `Default.imagine.zip` into `<install>/Configurations/`
3. start; pick **Default.imagine** (shown as host "Local ZIP")

Equivalently, take `GgImagine+Default.imagine.zip` and skip steps 1–2. The
Ant `imaginezip` target in this repo bundles `lib/` + `files/imagine/` + the
jars and **no configuration**, which is why the GitHub release alone is not
enough.

### 6.6 What is inside `Default.imagine`

750 files. By extension: 184 `.txt` (word lists, gazetteers, regex lists),
118 `.jar` (analyser code), 83 `.xml`, 45 `.cnfg`, 30 `.analyzer`, 29 `.csv`,
**25 `.imTool`**, 20 `.annotator`, 19 `.markupConverter`, **18 `.pipeline`**,
13 `.regEx`, 11 `.gScript`, 8 `.errorCheckList`, 5 `.gPath`,
5 `.annotationPattern`.

The 25 tool definitions — this is the actual pipeline inventory, and it maps
onto the workflow in §2:

```
MarkTaxonNames          ParseTaxonName            AugmentTaxonNameAuthorities
AddHigherTaxonomies     LinkTaxonNames
MarkBibRefCitations     ParseBibRef               ParseBibliography
LinkBibRefCitation      RemoveBibRefs
TreatmentTagger         TreatmentTaggerStyled     TreatmentStructurer
StructureTreatment      MarkTreatmentCitations    RemoveTreatments
KeysToTreatments
MarkMaterialsCitations  ExtractMaterialsCitations ExtractMaterialsCitationsTreatment
ParseMaterialsCitations ParseMaterialsCitationDetails
RemoveMaterialsCitations
MarkAccessionNumbers    TraitTagger
```

Two observations worth carrying into §4. First, note the `Remove*` tools —
every marking step has an explicit inverse, so a curator can undo a bad
automated pass wholesale and redo it. Second, `TreatmentTagger` vs
`TreatmentTaggerStyled` is the dual-path pattern from §1 surfacing again as
*two separately invocable tools*: heuristic and style-driven.

The `.errorCheckList` resources are the QA queues of §3.5/§4.7 as shipped
data. `ImpDocStyleBatchSelectorData/batchDescriptors.cached.json` shows that
document styles and bulk fix-up batches have since moved server-side —
entries like `LegacyDocumentCatchup` ("rename, update, and clean up
annotations, retro-apply normalizations, add missing external links, and
enforce current quality control standards") are Plazi's answer to
re-processing a corpus after the markup standard changes. The three
`.docStyle` files still in the config are all suffixed `.gone`.

### 6.7 If you want an archival copy

The config protocol makes mirroring trivial and independent of whether
Plazi's Downloads page survives:

```bash
BASE=https://tb.plazi.org/GgServer/Configurations/Default.imagine
curl -sS "$BASE/files.txt" -o files.txt
while read -r f; do
  mkdir -p "$(dirname "$f")"
  curl -sS --create-dirs "$BASE/$f" -o "$f"
done < files.txt
echo "$(date +%s)000" > timestamp.txt     # the server's copy 404s
zip -qr Default.imagine.zip .             # -> drop into Configurations/
```

Adding the `timestamp.txt` the server is missing is also exactly what would
be needed to re-host the tree yourself and have 6.3's in-app download path
work again.
