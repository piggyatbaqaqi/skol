# Data quality observations — production_v4 model

Notes from a Phase 1 bootstrap-annotation sample of 5 treatments selected
via `bin/select_for_annotation --experiment production_v4 --n 5
--bands low:1,mid:2,high:2 --seed 1` on 2026-06-28.  Four of the five
exhibited issues serious enough to flag for later attention; this file
captures the categories so future fix work doesn't start from scratch.

Tracking: see the corresponding Trello item (the list of affected
treatment IDs is duplicated below for in-repo readability).

## Affected treatments (sample)

Source database:
`skol_exp_production_v4_02_00_treatments_prose` on puchpuchobs as of
2026-06-28.

- `taxon_ba964a8b803eaf40672ba3561a79866d14054fe9ef993b4032161a8e05d3d55e`
- `taxon_2114314b6d1bf58aa91b2b99bb30442e7dc30c5fb9bc4a17b107586482e983fd`
- `taxon_22346900a8a1da8533cf8eed86a4ec07619320aa690696132a6c8514094320c2`
- `taxon_841d5cbed697b1882ba6b0f044556d801ae2df2f698fcc72c7a52bcb2349ce44`
- `taxon_2b793602153da2c98370528e7950159efd9fec7a49d8a4fb79b35f678c3cf6a9`

(Per-issue affected-ID lists below are left blank for the future fixer
to fill in during detailed triage — the sample is small enough to
re-inspect by hand.)

## Observed issue categories

### 1. Taxonomic citation in the `description` field

**Symptom**: the bibliographic citation (author, year, journal, page)
lands in `description` rather than in `nomenclature` or its own slot.

**Likely stage** (best guess, not investigated): layout CRF mis-labels
the citation paragraph as Description, OR the treatment CRF includes
it in the section boundary.

**Affected treatments**: (fill in)

### 2. Taxonomic citation not extracted at all

**Symptom**: the citation text exists in the source plaintext but
appears in no field of the Treatment doc.

**Likely stage** (best guess): layout CRF missed the citation
paragraph entirely; nothing for the treatment CRF to consume.

**Affected treatments**: (fill in)

### 3. Biology and Materials-examined confusion

**Symptom**: content that should be `materials_examined` (specimen
collection records — date, locality, collector, herbarium accession)
lands in `biology` (habitat / distribution context), or vice versa.

**Likely stage** (best guess): layout CRF confusion between the two
adjacent section labels.  Section headings are sometimes ambiguous in
the source (e.g., "Habitat" vs "Materials examined" may share visual
formatting).

**Affected treatments**: (fill in)

### 4. `pdf_url` null in Treatment, set correctly in skol_dev

**Symptom**: `treatment.ingest.pdf_url` is `None` even though the
corresponding `skol_dev` ingest document carries a non-null
`pdf_url`.

**Likely stage** (best guess): the ingest-doc projection inside
`bin/extract_treatments_to_couchdb.py` drops the field when copying
through to the Treatment's `ingest` sub-doc.  Probably mechanical and
fixable in one place.

**Affected treatments**: (fill in)

## Other observations

(fill in as you encounter more)

## Notes for fix sequencing

These issues are deferred — not blocking Phase 1 bootstrap-annotation
work in `treatments_to_structured/`.  Reasonable triage order when the
work is picked up:

1. **`pdf_url`** (§4) is likely a one-line fix in the ingest projection —
   quick win, no model retraining.
2. **Biology / Materials-examined confusion** (§3) needs layout-CRF
   training-data review and probably partial retraining.  Medium lift.
3. **Citation issues** (§1, §2) likely need layout-CRF retraining with
   citation-specific labelled training data.  Biggest lift; plan
   alongside the next v4 model refresh.

## Sample-size caveat

A 5-treatment sample is too small to extrapolate corpus-wide rates.
What this captures is "issues that exist", not "issues that dominate."
The next round of sample-then-review (Phase 1 deliverable 6+) will
expand the visible surface; revisit the severity ordering then.
