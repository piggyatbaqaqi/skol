# Treatment source anchors — polymorphic deep-linking schema

Design write-up from the 2026-07-10 batch-2 discussion.  Reframes
Trello #399 / #401 from "populate `pdf_page`" to "each Treatment
carries a rich, polymorphic list of source anchors" — enabling
deep-linking into whichever infrastructure (PDF, TreatmentBank,
JATS-HTML, external registries) actually serves the user best for
that treatment.

## The problem the old schema hit

The current Treatment record carries `pdf_page: int` and
`pdf_label: str` — a pair of PDF-native fields.  This was fine while
the corpus was PDF-derived Mycotaxon papers.  For the
Pensoft-family subset (Trello #401 finding) the `.ann` attachment
contains no `--- PDF Page N ---` markers at all — page numbers were
never there to extract — and the schema has no field for the
JATS-native anchors that ARE present:

- Per-treatment ARPHA UUID inside `<tp:taxon-name>`.
- Per-section `id="SECID0…"` attribute on every `<tp:treatment-sec>`,
  usable as an HTML fragment against the article's Pensoft URL.
- MycoBank registration ID inside `<object-id content-type="mycobank">`.

Meanwhile `doc.plazi.uuids[]` was already being stored at the doc
level from an independent Plazi-lookup pipeline — separate deep-link
target, separate resolver infrastructure.

Same shape, five different anchor kinds, all worth preserving:
polymorphic.

## Anchor-kind inventory

| Kind | Format | Source | Coverage | Resolver | Tier |
|---|---|---|---|---|---|
| `pdf` | `{page: int, label: str}` | `pdf_section_extractor` over `article.pdf` | Any doc with a PDF whose plaintext has `--- PDF Page N ---` markers | Local viewer, `pdf_url#page=<page>` | 1 (render) |
| `plazi` | 32-hex string, no dashes | `doc.plazi.uuids[]` (already stored) | Any doc Plazi ingested (~Pensoft subset) | `https://treatment.plazi.org/id/<uuid>` — verified 2026-07-10 | 1 (render) |
| `jats_section` | `{doi: str, section_id: str}` | JATS `<tp:treatment-sec id="…">` | Pensoft with `is_taxpub` | `<publisher article url>#<section_id>` | 1 (render, pending browser verification) |
| `mycobank` | Integer | JATS `<object-id content-type="mycobank">` | Any JATS with a MycoBank registration | `https://www.mycobank.org/page/Name%20details%20page/<id>` | 2 (store, content-thin) |
| `arpha` | Dashed UUID (`BA154B2C-A975-…`) | JATS `<object-id content-type="arpha">` on `<tp:taxon-name>` | Pensoft with `is_taxpub` | `https://openbiodiv.net/<uuid>` (**requires ARPHA app key — 404s without**) | 2 (store, credential-gated) |

Tier 1 kinds render as user-facing links today.  Tier 2 kinds are
persisted so we can flip on rendering as either credentials arrive
(ARPHA) or content quality improves (MycoBank) — cheap to store
now, expensive to backfill later.

Not all anchors are always present — a Mycotaxon PDF gets just
`pdf`, a Pensoft `is_taxpub` doc gets up to five.  The `plazi` kind
is per-article (Plazi stores a UUID array on `doc.plazi.uuids`),
not per-treatment; every treatment from the same article inherits
the same UUID list — Plazi's `treatment.plazi.org` resolver takes
users to Plazi's own treatment-picking UI.

## Polymorphic schema

Add a single field to the Treatment record:

```python
source_anchors: List[Dict[str, Any]]
```

Each dict is a JSON-serialisable shape whose `kind` key discriminates
the payload:

```python
[
    {"kind": "pdf",          "page": 3, "label": "3"},
    {"kind": "plazi",        "uuid": "0A4F6E6CD877BD32697F3B6BB9EF2AB5"},
    {"kind": "jats_section", "doi": "10.3897/mycokeys.108.130565",
                             "section_id": "SECID0EGZGK"},
    {"kind": "mycobank",     "id": 853632},
    {"kind": "arpha",        "uuid": "BA154B2C-A975-5BFF-BEEA-42184807F9D3"},
]
```

Conventions:

- Kind strings are lowercase snake_case; the set is closed
  (add-only in a follow-up commit — no free-form kinds).
- **Storage order is NOT part of the contract.**  The extractor
  emits whatever the code path produces (roughly insertion order,
  which is implementation detail).  Priority policy — which anchor
  should be shown first, used as default, or rendered at all —
  lives in the Django serializer, not the extractor.  Session
  decision 2026-07-22: single-authority priority table in Django
  avoids clients (React, cron scripts, third-party API users) each
  reimplementing the ordering.
- Missing/unavailable anchors are omitted, not stored as
  `{"kind": "…", "value": null}`.  An empty list is legal but
  triggers a triage flag (see below).
- URL construction is NOT baked into the extracted record.  The
  view/renderer knows per-kind URL patterns; the record stores raw
  identifiers.  When `treatment.plazi.org` migrates, we change one
  function, not re-extract 81 k treatments.

### Django serializer contract

The API layer sorts and resolves anchors before serving:

```python
# django/search/serializers.py (Phase 1c)
_KIND_PRIORITY = {
    'pdf': 10, 'plazi': 20, 'jats_section': 30,
    'mycobank': 90, 'arpha': 100,
}
_LINKABLE_KINDS = {'pdf', 'plazi', 'jats_section'}

def resolve_anchors(source_anchors, ingest):
    ordered = sorted(
        source_anchors,
        key=lambda a: _KIND_PRIORITY.get(a['kind'], 999),
    )
    return [
        {'kind': a['kind'],
         'href':  _href(a, ingest),
         'label': _label(a)}
        for a in ordered
        if a['kind'] in _LINKABLE_KINDS
    ]
```

The Treatment JSON delivered to React carries:

```json
"deep_links": [
    {"kind": "pdf",   "href": "https://…/article.pdf#page=3",         "label": "Page 3 of PDF"},
    {"kind": "plazi", "href": "https://treatment.plazi.org/id/0A4F…", "label": "Open at Plazi"}
]
```

React iterates and renders `<a href={href}>{label}</a>` per entry.
It never inspects `kind` for URL construction and never
reimplements the priority table — flipping the ordering (or
promoting a Tier-2 kind to linkable) is a single Django commit.

UX shape (ordered list vs primary + secondaries) is a serializer
concern the API can vary later without extractor changes; Phase 1c
lands the ordered-list form.

Keep `pdf_page`, `pdf_label`, `empirical_page_number` on the
Treatment record unchanged for at least one release — the Django
search UI + BackReference view read them directly.  Migration path
below.

## Data-quality contract

`Trello #401` graduates from "populate `pdf_page`" to a coverage
invariant:

> Every extracted Treatment should carry ≥1 `source_anchor`.  A
> treatment with an empty `source_anchors` list is unlinkable and
> is flagged for review.

Concrete signal in `triage_signals.treatment_signals()`:

- `n_source_anchors: int` — length of the anchors list.
- New flag `§13:no_source_anchor` fires when `n_source_anchors == 0`.

Poster-children and healthy treatments always fire ≥1.  Fixture
entries in `tests/fixtures/pathologies.json` get an optional
`source_anchors: [...]` field alongside the existing `description`
and `diagnosis` — omission defaults to `[]` and would fire the flag
(matching the pre-fix taxon_9f0c4134 shape).

## Phased rollout

### Phase 1 — schema + Tier-1 rendering (LANDED 2026-07-22)

- Commit A (`922b236`): ``source_anchors`` schema field in
  ``EXTRACT_SCHEMA``, PDF + Plazi emitters in
  ``Treatment.as_row()``, ``n_source_anchors`` triage signal (no
  flag yet), CSV column in ``bin/triage_treatments.py``.
- Commit B (`70706d4`): ``extract_taxpub_anchor_bundles`` in
  ``ingestors/jats_to_yedda.py``, parallel channel via
  ``PipelineState.taxpub_treatment_anchors``,
  ``Treatment.set_taxpub_anchors`` called by
  ``treatment_assembler`` on the assembled treatments, ARPHA /
  jats_section / MycoBank emission wired into
  ``_build_source_anchors``.  End-to-end integration test in
  ``components_test.py``.
- Commit C: ``§13:no_source_anchor`` triage flag activation +
  ``django/search/deep_links.py`` resolver (single
  ``_KIND_PRIORITY`` policy table, per-kind URL construction,
  Tier-2 kinds filtered out).  Wired into ``TreatmentsInfoView``
  as the new ``DeepLinks`` response field alongside legacy
  ``PDFPage``/``PDFLabel``.
- Backfill: operator runs
  ``bin/manage_experiment runstep production_v4 extract_treatments --force``
  to repopulate ``source_anchors`` on the existing 81 k treatments
  (~5 min, same shape as the Trello #399 backfill).

### Phase 2 — Tier-2 activation (when credentials or content warrant)

- Once we have an ARPHA app key (or an anonymous mirror surfaces),
  add `arpha` to the renderer's linkable set.  No schema change; no
  re-extract.
- If MycoBank pages become useful for taxonomy context (or we build
  our own overlay), add `mycobank` to the renderer's linkable set.
  Same story.

### Phase 3 — deprecate the flat `pdf_page` field (later, contingent)

Only after every downstream consumer (Django search view, BackReference
view, CSV export, brat XREF, treatment_signals) has been switched to
read from `source_anchors`.  Two release cycles of dual-writing is the
safe minimum.

## Deferred: Plazi treatments as a golden set

An adjacent idea surfaced during this discussion: **ingest Plazi
treatments as first-class data and use them as an eval baseline**
against our extractor's output for the Pensoft-covered subset of the
corpus (~10-20% by article count — see the [Plazi overlap
memo](../MEMORY.md)).

Two layers:

1. **Coverage/count eval (cheap)** — did we find the same N
   treatments Plazi did?  Same taxon binomials?  No content alignment
   needed; high signal for missing-treatments and phantom-treatments
   defects.

2. **Field-level eval (expensive)** — field-by-field content
   comparison after span alignment (Plazi normalises whitespace and
   citations differently than we do).  Requires fuzzy-comparison
   design (token overlap? sentence embedding cosine?).

**Status: shelved pending direct conversation with the Plazi team.**
The design questions worth asking them include: (a) preferred
programmatic access mode (REST API vs RDF dump vs bulk XML),
(b) treatment/paragraph granularity of their annotations, (c)
availability of an alignment-friendly export.  Until that
conversation happens the ingest surface stays unbuilt.

## Verification the design owes before landing Phase 1

- Browser test: `https://mycokeys.pensoft.net/article/130565/#SECID0EGZGK`
  actually scrolls to the description section on Pensoft's page.
- Confirm no other JATS-family source in `skol_dev` — spot MDPI /
  Mycotaxon indices already showed PDF-only, but a search for
  `xml_available: true` docs whose meta.source is not `pensoft` would
  close the loop.
- Data-quality re-audit after Phase 1 backfill: what fraction of
  treatments end up with `n_source_anchors == 0`?  #401 is fixed
  when that fraction hits zero on the Pensoft `is_taxpub` subset;
  a non-zero residual is the next investigation.
