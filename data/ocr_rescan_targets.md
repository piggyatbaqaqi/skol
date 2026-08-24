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

1. **Mycotaxon vols 3–31** — largest, worst-measured, and Mycotaxon is
   a current journal, so a better scan may already exist.
2. **Persoonia vols 1–19** — already scoped as Trello #404, with
   per-article files at <https://repository.naturalis.nl/col/1>.  Note
   #405 (deduplication) gates it: without dedup the new files add a
   *third* copy rather than replacing anything.
3. **Lloyd + Kauffman + Buller + McIlvaine** — old books, likely on
   BHL/archive.org, plausibly with better scans than we hold.
4. Everything else, opportunistically.

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
