# Sources page — alternative grouping axes

Status: **parked**, captured from session brainstorm.  Pursue after the
data-quality work that gates a beta release.

## Context

The Ingestion Sources page currently groups treatments by `journal`.
After fixing the PDF-fallback bug (commit cd4cabe) and the Crossref
journal backfill (commit 9acdac9 + the live data run), we have decent
per-journal counts plus two domain-specific badges (`new_taxa_acts`,
`sanctioned_markers` — commit ae79ba0).

The user's follow-up question was about *additional grouping axes* —
other categorical buckets a treatment could be sorted into for an
alternative tabular view, not extra columns on the per-journal table.
Examples raised: "Monographs", "Field guides".

This document records the brainstormed axes so they aren't lost when
the priority returns to data quality / beta release.

## Candidate axes

### A. Publication form

Categorical buckets the *source publication* falls into.  Detected
from the source doc's title + page count + treatment-density.

| Bucket | Detection signal |
|---|---|
| Monograph | Title contains "Monograph" / "Revision" / "Studies in"; or ≥30 treatments from a single source doc |
| Atlas / Flora / Mycota | Title matches `Flora\|Mycota\|Atlas of` + regional or group qualifier |
| Catalog / Checklist | Title matches `Catalog(ue)\|Checklist\|Census\|Inventory` |
| Field guide | Title matches `Field guide\|Mushrooms of\|Guide to` (rare in current corpus — no DOIs) |
| Original article | Default — 1–5 treatments, no monographic title hint |

Single pass over skol_dev + a treatment-count rollup.  Closest in
flavour to the user's original prompt.

### B. Taxonomic scope

How wide an evolutionary slice the source paper covers.

| Bucket | Detection |
|---|---|
| Single genus | All treatments share the first word of `treatment` |
| Family revision | Treatments span 2–10 genera within one family (needs taxonomy lookup) |
| Order / phylum survey | Treatments span ≥10 genera or multiple families |

Needs an external taxonomy map (Index Fungorum / MycoBank) to know
which genera belong to which family.

### C. Methodology

Read off the Description / Diagnosis text.

| Bucket | Signal |
|---|---|
| Molecular | Mentions of ITS / LSU / RPB2 / GenBank accession numbers |
| Phylogenetic | Mentions of clade / Bayesian / ML tree / RAxML |
| Morphological only | Absence of the above |
| Cultural | Mentions of PDA / MEA / culture / colony / mycelium growth |

Splits modern (~1990 onward) from classical work.

### D. Geographic scope

| Bucket | Detection |
|---|---|
| Cosmopolitan | Default — no detected place |
| Country-specific | NER or country-name regex against `materials_examined` text |
| Continent-specific | Aggregate of country |
| Locality-specific | "Found at … Park" — free-form, hard without NER |

Lower priority unless a globe-map view is desired.  NER work is real.

### E. Era / authority lineage

| Bucket | Detection |
|---|---|
| Pre-1900 classical | Authority year < 1900 (parseable from `treatment` text) |
| Modern (1900–1990) | year 1900–1990 |
| Molecular era | year ≥ 1990 or has molecular markers |

Parses cleanly from the authority citation when present
(e.g., `(L.) Fr. 1821` → 1821).

### F. Author lineage (extending the existing Fries/Persoon badges)

| Bucket | Detection |
|---|---|
| Fries-sanctioned | `: Fr.` / `(Fr.)` / `ex Fries` — **already counted** in `sanctioned_markers` |
| Persoon-sanctioned | `: Pers.` / `(Pers.)` / `ex Persoon` — **already counted** |
| Linnaean | `(L.)` / `L. 1753` |
| Crous school (Westerdijk) | author = Crous |
| No authority | the synthetic `Nomen ignotum` stubs (`synthetic_nomenclature: true`) |

Fries and Persoon are already counted; Linnaean and "no authority"
are nearly free additions; named-school filters are interesting for
tracking modern revisionary work.

## Recommended starting point if/when this is revisited

**Axis A (Publication form)** is the natural complement to Journal —
same shape (categorical bucket), useful to mycologists thinking about
"where do I look for treatments of this group?"  Signal is entirely
in data already in skol_dev (title + treatment-density per source
doc).  Implement as a dropdown on the Sources page that switches the
grouping axis (Journal vs Publication Form), with the same
Total / Taxonomy / Treatments / New-taxa / Sanctioned counts.

**Axis F extended (author lineage)** is the deepest mycology-specific
story — Fries / Persoon / Linnaeus / modern-Crous-school columns would
surface real intellectual genealogy.  Could be a small extension to
`build_sources_stats` rather than a UI redesign.

Either can be picked up post-beta without breaking anything; both
write new per-experiment Redis blobs alongside the existing one.
