# Normalizing publication metadata into JOURNALS + SOURCES

Status: **plan under review**.  Refactor of `ingestors/publications.py`
to separate journal-level facts from ingestion-source mechanics.

## Goal

Split the current single `SOURCES` table into two related tables:

- **`JOURNALS`** — keyed by canonical journal name; one row per
  real-world journal.  Owns the facts that are properties of the
  *journal*: official website, ISSN(s), publisher, historical-name
  aliases.
- **`SOURCES`** — keyed by ingestion-source name; one row per
  *way to fetch* articles for some journal (RSS, PMC, Crossref,
  local mirror, etc.).  Each row references its journal by
  canonical name (foreign key into `JOURNALS`).

The parallel `JOURNAL_NAME_ALIASES` dict goes away — aliases live
on their journal record in `JOURNALS`.

## The smell we're paying off

Today's `SOURCES` is denormalized: 9 of 20 journals have multiple
entries (Persoonia has 3, Mycology has 3, Journal of Fungi has 4),
and each entry duplicates the journal-level fields (ISSN,
address-of-the-actual-journal, etc.).

To keep those duplicate entries' `journal` field unique, some
entries embed publisher information into the journal name:
`"Mycology (PMC)"`, `"Journal of Fungi (PMC)"`, `"Mycology: An
International Journal on Fungal Biology (Taylor & Francis)"`.
Those compound strings leak into the rendered Sources page where
they shouldn't be — `"(PMC)"` is a mechanism, not the name of a
journal.

Aliases (`JOURNAL_NAME_ALIASES`) sit in their own dict — fine in
isolation, but they ARE journal-level facts and should be co-located
with the rest.

The `role='metadata'` patch I sketched in an earlier draft of this
doc would have added a parallel kind of SOURCES row to host
journal-level fields.  That made the denormalization worse, not
better.  Discard.

## New shape

```python
# JOURNALS — one row per real-world journal.
JOURNALS: Dict[str, Dict[str, Any]] = {
    'Sydowia': {
        'address':   'https://www.sydowia.at/',
        'issn':      '0082-0598',
        'aliases':   ['Sydowia Beih.'],
        'publisher': 'Verlag Ferdinand Berger & Söhne',
    },
    'Persoonia': {
        'address':   'https://persoonia.org/',
        'aliases':   ['Persoonia - Molecular Phylogeny and Evolution of Fungi'],
        'publisher': 'Naturalis Biodiversity Center',
    },
    'Journal of Fungi': {
        'address':   'https://www.mdpi.com/journal/jof',
        'publisher': 'MDPI',
    },
    # ...
}

# SOURCES — one row per scrape mechanism; each row's `journal`
# field is a foreign key into JOURNALS.
SOURCES: Dict[str, Dict[str, Any]] = {
    'persoonia-pmc': {
        'journal':         'Persoonia',
        'address':         '...PMC URL...',
        'source':          'pmc',
        'ingestor_class':  'PMCIngestor',
    },
    'persoonia-rss': {
        'journal':         'Persoonia',
        ...
    },
    'jof-pmc': {
        'journal':         'Journal of Fungi',   # NOT 'Journal of Fungi (PMC)'
        ...
    },
    # ...
}
```

Helper methods on `PublicationRegistry`:
- `get_journal(name) -> Optional[dict]` — JOURNALS lookup with alias
  resolution.
- `find_journal_by_issn(issn) -> Optional[str]` — scans JOURNALS
  for an entry whose `issn` or `eissn` matches; returns the canonical
  journal name.
- `normalize_journal_name(name)` — consults `JOURNALS[*].aliases`
  lists; returns the canonical name on the LHS of the matching
  entry.

## Phased migration

Five phases, sized so each one is a self-contained landing.

### Phase 1 — build `JOURNALS`; rename compound `journal` fields in `SOURCES`

**Goal**: structurally split the table.  No behavior change on the
rendered Sources page yet.

1. Write a one-shot scaffolding script (in `bin/` or a throwaway)
   that walks current `SOURCES` and emits a draft `JOURNALS`
   dict — one entry per unique `journal` value, fields populated
   from whichever SOURCES entry had them.
2. Hand-edit the draft `JOURNALS`: deduplicate, normalize journal
   names (strip publisher tags like `"(PMC)"`, `"(Taylor &
   Francis)"`), fill in missing official-website URLs and ISSNs
   from public sources.
3. Update `SOURCES` entries that carried compound names to point
   at the canonical journal: `"Journal of Fungi (PMC)"` →
   `"Journal of Fungi"`, etc.
4. All existing classmethods on `PublicationRegistry` keep working
   — `get_by_journal()`, `list_publications()`, etc.  Tests pass.

**Visible state after phase 1**: nothing changes on the Sources
page yet — the rendered rows are driven by skol_dev's `journal`
field, which still carries the old compound names from past
ingestions.  That's fixed in phase 2.

### Phase 2 — rename existing skol_dev `journal` fields

**Goal**: rewrite the docs in skol_dev that carry old compound
journal names so they match the new canonical ones.

1. New `bin/rename_journals.py`: given a mapping from old → new,
   walk skol_dev and rewrite any doc whose `journal` field matches
   an old name.  Idempotent — re-running is a no-op once docs are
   at the target state.  Supports `--dry-run`, `--limit`,
   `--verbosity`.
2. Mapping for this round comes from the compound names enumerated
   in phase 1 (we'll enumerate them while we're in there).
3. Apply, rebuild `production_v3_hand` Sources stats.

**Visible state after phase 2**: rows on the Sources page no
longer carry `"(PMC)"` / publisher suffixes; previously-separate
rows for the same journal-via-multiple-ingestors merge into one.

### Phase 3 — move alias handling into `JOURNALS`

**Goal**: delete `JOURNAL_NAME_ALIASES`.

1. Move each entry of `JOURNAL_NAME_ALIASES` into the
   `aliases` list on its canonical `JOURNALS` entry.
2. Update `normalize_journal_name()` to consult
   `JOURNALS[name].aliases`.  Keep a short back-compat branch
   that also checks the legacy dict, then delete that branch and
   the dict.
3. Run a one-shot pass over skol_dev to rewrite any doc whose
   `journal` field is currently a known alias to the canonical
   form (e.g., docs whose journal field literally says
   `"Sydowia Beih."` get rewritten to `"Sydowia"`).  Same script
   from phase 2 with an alias-source mapping.
4. Rebuild Sources stats.

### Phase 4 — add `find_journal_by_issn`, ISSN-fallback in resolve_source_name

**Goal**: surface the 249 archive.org Sydowia docs under
`"Sydowia"` without writing to skol_dev.

1. Implement `PublicationRegistry.find_journal_by_issn(issn)`.
2. Extend `resolve_source_name(doc)` in
   `bin/build_sources_stats.py`: if `journal` is empty, consult
   `find_journal_by_issn(doc.get('issn'))` (and `eissn`) before
   falling through to the mykoweb fallback / Unknown.
3. Rebuild Sources stats.  Sydowia bucket grows by 249.

### Phase 5 (optional) — persistent ISSN-driven journal backfill

**Goal**: write `journal=<canonical>` onto skol_dev docs whose
ISSN matches a JOURNALS entry, so downstream consumers don't have
to re-resolve.  Mirrors the existing `bin/backfill_journal.py`
ISSN pass but the source of truth is the in-tree registry rather
than Crossref — useful exactly for cases (like Sydowia) where
Crossref's ISSN endpoint has gaps.

Decide whether to do this after phase 4 — phase 4 alone covers the
Sources page; phase 5 only matters for other consumers of the
`journal` field.

## Things to confirm before phase 1

- No automated tooling writes back to `SOURCES` or
  `JOURNAL_NAME_ALIASES`.  Both are hand-curated source files.
- The compound-name renames in phase 1 (e.g.,
  `"Journal of Fungi (PMC)"` → `"Journal of Fungi"`) will merge
  what currently appear as separate rows on the Sources page —
  confirmed wanted.
- Phase 2 writes to skol_dev.  Idempotent + dry-run-able.  No
  destructive deletes.

## Open questions

- **Where to put the scaffolding script for phase 1?** Throwaway
  in `/tmp/`, committed `bin/scaffold_journals.py`, or just
  inline in this doc?  Throwaway keeps `bin/` from accumulating
  one-shot tools; committed gives a paper trail.
- **Should `JOURNALS` track per-journal stats** (e.g., total docs,
  first-seen date) or stay purely descriptive?  Recommend purely
  descriptive — stats live in Redis (`skol:sources:stats:*`),
  not in the source registry.
- **`address` semantics**: today some `SOURCES` entries' `address`
  is a scrape endpoint (`https://api.crossref.org/...`); on the
  Sources-page row it's intended as the journal's homepage.  In
  the new shape, `JOURNALS[name].address` is the journal homepage
  (what gets displayed); `SOURCES[key].address` is the scrape
  endpoint (only used by the ingestor).  Worth being explicit
  about this in the schema docstrings.
