# Page Header Detection Design Notes

## Problem statement

Journal articles extracted from PDFs contain running page headers (and sometimes
footers) that interrupt the body text.  These lines carry information like the
journal name, volume/issue, author name, article title, and the printed page
number.  Detecting and removing — or annotating — them is a prerequisite for
clean treatment extraction, because the treatment assembly state machine can
misread a page number as a gap-filler and terminate a treatment prematurely.

---

## Core algorithm sketch

### Step 1 — Collect candidates

Scan every line for a token that is a plausible page number:

- String of 1–4 decimal digits
- Appears at the **start** or **end** of the line (after stripping whitespace)
- Does **not** match a year pattern: `(19|20)\d{2}` at line-end is a strong
  negative signal because years cluster rather than increment by 1

Five or more consecutive digits almost certainly are not page numbers (accession
numbers, specimen IDs, etc.) and can be excluded immediately.

### Step 2 — Build and score sequences

From the candidates, find a monotonically increasing subsequence with consistent
first differences:

- Expected gap 1 (every page numbered), occasionally 2 (every other page)
- Larger gaps are allowed to accommodate journals that omit page numbers from
  some pages (e.g. Mycologia, which starts each article with a header that
  lists only the journal name)
- **RANSAC-style fitting**: model `page_number ≈ a × doc_position + b`;
  inliers are candidates with small residuals; OCR digit transpositions produce
  large residuals and are rejected naturally
- **Gap histogram**: before committing, compute the histogram of first
  differences across all candidates.  A sharp peak at 1 or 2 confirms the right
  candidates; a flat histogram suggests noise.  This doubles as a per-document
  quality score.

### Step 3 — Recto/verso alternation

Many journals alternate placement:

- Even page numbers at line-start (left/verso page)
- Odd page numbers at line-end (right/recto page), or vice versa

Fit odd and even candidate sets independently, then verify the two subsequences
interleave cleanly.  Clean interleaving sharply raises confidence.

### Step 4 — Journal name matching

The non-numeric portion of a confirmed header line often contains the journal
name or a standard abbreviation.  Once a few sequence members are confirmed,
cluster the non-numeric tokens from all candidates.  Journals with running
headers produce two clusters: journal name (or abbreviation) and author/title.

Approximate matching (edit distance or token overlap) is needed because:

- OCR introduces substitution errors
- Abbreviations vary between volumes
- Some journals use ALL CAPS, others title case

### Step 5 — Two-pass header block recovery

- **Pass 1**: identify the page-number sequence (Steps 1–4)
- **Pass 2**: use confirmed sequence members as anchors to recover the full
  header block — including adjacent lines that lack a number (volume/issue
  lines, blank separator lines)

This allows marking and stripping the complete header region rather than only
the numbered line.

---

## Additional line-level signals

| Signal | Notes |
|---|---|
| **Blank-line adjacency** | Headers almost always appear next to blank or very short lines; a candidate number embedded in a paragraph is almost certainly body text |
| **Line length** | Header lines are short; body text clusters at the column width |
| **Repeated non-numeric content** | Non-numeric tokens from header candidates should cluster tightly; scattered tokens indicate noise |
| **Capitalization style** | Journal names in headers are often ALL CAPS or title case, anomalous relative to body text — detectable without a dictionary |
| **PDF page proximity** | The proximity of a candidate line to a PDF page boundary is a strong hint, though PDF page markers are programmatically reconstructed in some training documents and may be slightly wrong |

---

## OCR-specific issues

**Digit substitution errors** (`0`/`O`, `1`/`l`/`I`, `5`/`S`, `6`/`G`, `8`/`B`).
Normalize common substitutions before sequence analysis, or try both
interpretations when a candidate almost fits the sequence.

**Line merging and fragmentation**.  Some extractors merge the header into the
first body line; others split one header line into two.  A sliding window over
two-line concatenations catches the merge case.  PDF page proximity is
especially valuable here as a position anchor independent of extraction quality.

---

## Roman numerals

Front matter in some journals uses roman numerals (i, ii, iii, …).  The
consensus is to **skip this for now**:

- Front matter rarely contains taxonomic treatments
- The detection payoff is therefore low
- False-positive rate against single-letter abbreviations is high without the
  sequence consistency check
- Can be added later as an optional post-processing step if needed

---

## Per-document calibration

Header patterns vary between journals and between volumes of the same journal.
Calibrate the following per document before applying document-wide:

- Left / right / alternating placement
- Line length threshold
- Journal name template (from confirmed header clusters)

Per-document calibration improves recall substantially over a universal model.

---

## Architectural decision: remove vs. annotate

Options:

1. **Remove** — irreversible; simplest downstream
2. **Annotate** as a YEDDA tag (e.g. `Page-header`) — preserves auditability;
   lets the treatment assembler skip headers while keeping the pipeline
   composable
3. **Flag only** — mark candidates without committing to removal

Recommendation: **annotate** rather than delete, consistent with the existing
YEDDA pipeline.  The treatment state machine already skips non-treatment labels,
so annotated headers will be ignored without any further changes.

---

## Empirical questions to answer on the training corpus first

Before finalizing the algorithm, two measurements would sharpen design choices:

1. **What fraction of training documents have reliable page headers?**  If low,
   the feature is less valuable to optimize.  If high, even a rough detector
   improves treatment boundary detection.

2. **What is the PDF-page-to-printed-page offset distribution?**  If the offset
   is consistent within a document but varies between documents, the PDF
   proximity hint is most useful as a per-document calibration anchor rather
   than a universal prior.

---

## Open questions

- Should detected headers feed back into the treatment assembler as a
  `Page-header` YEDDA tag, or be stripped before YEDDA annotation?
- Is a heuristic rule system sufficient, or is a small trained classifier
  (logistic regression over the signals above) warranted?
- How should the detector handle two-column layout, where PDF extraction often
  interleaves left and right column lines, potentially disrupting line-end
  number detection?
- Where does this fit in the pipeline?  Candidate location: after
  `extract_plaintext` and before `predict_classifier`, as a new
  `annotate_headers` step.
