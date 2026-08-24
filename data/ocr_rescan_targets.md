# Works to re-source for better OCR

Major works whose OCR is bad enough to damage extraction, ranked by
evidence.  Started 2026-08-24 from round-4 review plus a corpus-wide
measurement.

**Why this list is worth keeping.** Three separate round-4 findings
pointed the same way: the biggest available wins in this corpus are
*upstream of the models*.  A better layout classifier cannot recover
text that reads `i m< ta l< trongly farin ( om lamellae adnate`, and a
better grouper cannot find a heading OCR has dissolved.

## How the ranking was produced

`treatments_to_structured.ocr_damage` computes three independent damage
modes; the **space-displacement (rejoin) rate** is the one that
discriminates scanned text cleanly, so it drives the measured table.

```python
from treatments_to_structured.ocr_damage import OcrDamage, load_vocabulary
profile = OcrDamage(text, vocabulary=load_vocabulary()).profile()
```

**Three cautions, all measured, before trusting any number here:**

1. **The intra-word corruption rate alone is misleading.** Ranked on it,
   the corpus's "worst" sources are modern molecular papers — `Fig. 2C`,
   `4-7um` and `KY784257` read as corruption.  `ocr_damage` excludes
   those, but the raw metric ranked *A compendium of macrofungi of
   Pakistan* above every scanned monograph.
2. **The rates describe what survived extraction, not the source.**  On
   a shredded document the mangled passages are routed to `Table` /
   `Key` / `Misc-exposition` and never reach the treatment fields, so
   the remnant reads clean.  `taxon_a686d7ab` scores **0.0 %** rejoin
   over 384 tokens while its source region holds 2 222 tokens at 2.4 %.
   **This systematically under-ranks the worst-shredded works** — which
   is why the second table below exists.
3. **Character-substitution damage is nearly invisible to rejoin.**
   Persoonia 13 (`taxon_8d815304`) is among the worst-damaged
   treatments seen and scores 4.3 %, below the corpus p90 of 5.7 %.

Corpus baseline, 48 738 treatments: windowed rejoin median **0.0 %**,
p90 **5.7 %**, p99 **20.0 %**.

## Tier 1 — measured, ranked by rejoin rate

Works whose median rejoin ≥ 8 %.  **None has a DOI**; all are scans.

| work | volumes | treatments | median rejoin |
|---|---|---:|---:|
| **Mycotaxon** | 3–31 | **1 144** | **13.3 %** |
| *Phaeocollybia* of Pacific NW | — | 34 | 12.5 % |
| Annales Mycologici 1903–11 | 1 | 15 | 17.4 % |
| Memoir of the NY State Museum | 4 | 15 | 8.3 % |
| Annales Mycologici 1911–12 | 9 | 14 | 10.0 % |
| Annales Mycologici 1905–10 | 3 | 11 | 8.3 % |
| Ann. Rep. NY State Mus. | 26 | 10 | 12.5 % |
| Sydowia | 28 | 20 | 8.0 % |

**1 263 treatments**, of which **Mycotaxon vols 3–31 is 91 %.**  Early
Mycotaxon is the single largest OCR liability in the corpus — worse,
and larger, than the Persoonia back-run that prompted Trello #404.

Worst individual volumes: Mycotaxon **12** (24.3 %), **6** (23.5 %),
**20** (20.7 %), **16** (20.0 %).

## Tier 2 — identified by review, under-ranked by the metrics

These are shredded or character-substituted, so caution 2 or 3 applies
and the measured rate understates them.  Each was found by reading a
treatment, not by ranking.

| work | treatments | evidence |
|---|---:|---|
| **Persoonia vols 1–19** (whole-volume scans) | **871** | `taxon_8d815304` — 9 genera fused; `taxon_9499dcb0` — genus unreadable (`Denttpellis`).  Trello **#404** |
| The Agaricaceae of Michigan | 259 | 96.9 % synthetic nomenclature |
| Lloyd, *Mycological writings* II / III / IV | 386 | 84–90 % synthetic; `taxon_a5efbd0b` shredded to 35 chars |
| Researches on Fungi, vols 1–2 | 147 | 96.7 % synthetic (vol 1) |
| North American Polypores, vols 1–2 | 115 | vol 1 rejoin 13.8 % |
| One Thousand American Fungi | 63 | 90.5 % synthetic |
| Our Edible Toadstools and Mushrooms | 44 | 97.7 % synthetic — worst in corpus |
| A.H. Smith, *Studies of North American Agarics-I* | 10 | `taxon_a686d7ab` — two species shredded across `Table`/`Key` |

**~1 895 treatments.**

## Combined scale

Roughly **3 150 treatments**, about **3.9 %** of the 81 527-treatment
corpus, concentrated in a few dozen works.  Mycotaxon 3–31 and
Persoonia 1–19 alone are **2 015** — 64 % of the total.

## Suggested order

1. **Mycotaxon vols 3–31** — largest and worst-measured, and the
   cheapest to act on.  Its page images are already **300 ppi RGB**, so
   this looks like a re-OCR of PDFs we hold rather than a re-sourcing
   job — and **BHL publishes hOCR with per-word `x_wconf` confidence
   for it**, so the diagnosis costs a download rather than an engine
   run.  Start here.
2. **Persoonia vols 1–19** — already scoped as Trello #404, with
   per-article files at <https://repository.naturalis.nl/col/1>.  Note
   #405 (deduplication) gates it: without dedup the new files add a
   *third* copy rather than replacing anything.  Genuinely needs
   re-sourcing rather than re-OCR — its page images are **150 ppi
   greyscale**, below the usual threshold.
3. **Lloyd + Kauffman + Buller + McIlvaine** — old books, likely on
   BHL/archive.org, plausibly with better scans than we hold.
4. Everything else, opportunistically.

## Proposed: OCR-engine confidence as a direct signal (not yet done)

Operator suggestion, 2026-08-24: run an OCR engine such as Tesseract
over the page images and use its **per-word confidence** as the quality
measure, instead of inferring damage from the text after the fact.

**It answers both gaps in `ocr_damage` directly.**  Confidence is
computed on the *image*, so it measures the source rather than what
survived extraction — and it needs no vocabulary, so it neither misses
garbling-beyond-recognition nor inherits the proper-noun contamination
that keeps `oov_rate` from being a mode.

### What we already hold

Every ingest document keeps `article.pdf`, and those PDFs contain both
page images and a text layer.  **The text layer is itself OCR output**:
its fonts are non-embedded standard Type 1 faces (`Helvetica`,
`Times-Roman`, `emb no`) across every work checked — the signature of a
layer painted over a scan, not of a born-digital document.

So the stored `article.txt` is somebody *else's* OCR pass, and its
quality is not a property of the images we hold.

### The measurement that matters, and it reorders the list

Sampling page images from the two largest targets:

| work | page images | text layer |
|---|---|---|
| **Mycotaxon vols 6, 12** | **987 × 1470 RGB, 300 ppi** | non-embedded → OCR |
| **Persoonia vol 13** | **781 × 1200 grey, 150 ppi** | non-embedded → OCR |

**Mycotaxon — the worst-measured work in the corpus — has good source
images.**  300 ppi RGB is a perfectly adequate scan; its 13.3 % rejoin
rate is a *bad OCR pass over good images*, not a bad scan.  That is a
hypothesis rather than a result, but if it holds, **re-OCRing PDFs we
already have would fix 1 144 treatments** with no external sourcing, no
new ingest, and no dependency on #405.

**Persoonia 13 at 150 ppi greyscale is below the usual 300 ppi
threshold for reliable OCR.**  Re-OCRing the same images has a low
ceiling; that one needs a genuinely better source, which is what #404
provides.

### The triage this enables

Comparing engine confidence against our stored text is more informative
than either alone:

| engine confidence | stored text | reading | action |
|---|---|---|---|
| high | bad | good images, bad OCR pass | **re-OCR locally** — cheap |
| low | bad | the image itself is poor | **re-source** — expensive |
| high | good | fine | leave alone |
| low | good | OCR beat expectations | spot-check |

### BHL already publishes the confidence data

Operator, 2026-08-24: **hOCR HTML is available from the Biodiversity
Heritage Library for at least Mycotaxon**, and hOCR carries
`x_wconf` — a 0–100 OCR confidence **per word**:

```html
<span class='ocrx_word' title='bbox 412 1583 508 1610; x_wconf 92'>Pileus</span>
<span class='ocrx_word' title='bbox 515 1583 604 1616; x_wconf 34'>rugu1ose</span>
```

**This removes the engine run entirely for those works.** No Tesseract,
no page rasterisation, no CPU budget — fetch the hOCR and parse an
attribute. A regex over `x_wconf (\d+)` is enough to get the
distribution; a proper parse gets you the word and its bounding box
alongside, so low-confidence regions can be located on the page.

Three things this makes cheap that were not:

* **Per-page and per-region confidence**, not just per-work. The
  bounding boxes mean a low-confidence *block* can be identified —
  which is what matters for the shredded works, where damage is
  concentrated rather than uniform.
* **Word-level ground truth for the `ocr_damage` thresholds.** Its
  cut-offs are corpus quantiles chosen for lack of anything better.
  `x_wconf` gives an independent measure to calibrate against, and in
  particular to test whether the substitution threshold of 4 % is
  anywhere near right.
* **A second OCR pass to diff against.** BHL's text may or may not be
  the same pass that produced our `article.txt`. If it differs and
  scores better, BHL is not just a measuring instrument but a
  **replacement source** — and a far cheaper one than re-scanning.

**Check first whether BHL's OCR is the same pass we already hold.** If
the text matches ours, `x_wconf` is still a perfectly good quality
signal but tells us nothing new about alternatives.

### Practical notes

* **Tesseract is not installed** on puchpuchobs — `pytesseract` 0.3.13
  is present but the binary is not.  A missing package on production is
  a packaging error (CLAUDE.md), so `tesseract-ocr` plus the language
  data belongs in the deb dependencies **if** an engine run is needed
  at all — which, given BHL's hOCR, it may not be for the works that
  matter most.
* `pdftoppm`, `pdfimages` and `pdffonts` **are** installed, so page
  extraction and the font/resolution triage above need nothing new.
* **Sample, don't sweep.**  Ten pages per work over ~50 works is a few
  hundred pages — minutes, and enough to rank.  Full re-OCR is a
  separate, larger decision that this measurement should inform.
* Report the **median** and **10th-percentile** word confidence, not
  the mean: a page of clean running text with one ruined block is the
  case that matters, and a mean hides it.

## Caveats worth carrying

* **Re-sourcing changes treatment ids.**  They are content hashes
  (`sha256` over the prose fields), so every fixture entry anchored to a
  re-sourced work will dangle.  Four entries in
  `tests/fixtures/pathologies.json` carry a
  `source_scheduled_for_replacement` block naming the work and the
  consequence.
* **Better OCR does not fix the grouper.**  Fused headings, front-matter
  treatments and swallowed continuations occur in cleanly-OCR'd modern
  papers too.  This list removes a cause, not the class.
* **`synthetic_nomenclature` rate is a good proxy for these works but
  not in general.**  Corpus-wide it is 39.6 % on per-article sources and
  31.6 % on book-like ones — the signal is per-document, not per-class.

## Cross-references

* [docs/data_quality_production_v4_model.md](../docs/data_quality_production_v4_model.md)
  — §9 (OCR modes), §5.2 (monographic books), D8 (rejoin detector)
* [treatments_to_structured/ocr_damage.py](../treatments_to_structured/ocr_damage.py)
  — the measurement code
* [docs/plans/annotation-activity-split.md](../docs/plans/annotation-activity-split.md)
  — Trello #404 / #405 sequencing
