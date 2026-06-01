# Consolidating publication metadata into a single registry

Status: **phase 1 in progress**.  Tracking the migration of
journal-level facts (display URL, ISSN, name aliases) into
`ingestors/publications.SOURCES`.

## Background

`ingestors/publications.py` carries publication state in two places:

- **`SOURCES`** — per-publication scrape configuration: `name`,
  `journal`, `address`, `source`, `ingestor_class`, `local_path`,
  etc.  Each entry feeds `bin/ingest.py` and one `Ingestor`
  subclass.
- **`JOURNAL_NAME_ALIASES`** — flat dict mapping variant journal
  names (`'Sydowia Beih.'`, `'Cryptogamie Mycologie'`, etc.) to
  canonical names.  Consumed by `normalize_journal_name()` to
  consolidate per-journal stats on the Sources page.

The Ingestion Sources page also currently routes journal-less docs
through `resolve_source_name()` (in `bin/build_sources_stats.py`),
which falls back to a hard-coded `'mykoweb'` label.  Other "this
doc came from journal X but we don't know how to display X yet"
cases (Sydowia via archive.org) have no home — they end up in
``Unknown``.

We need a place to record journal-level facts (display URL, ISSN,
historical-name aliases) without inventing another top-level dict
and without coupling that data to scraper invocation.

## Schema decision

Extend `SOURCES` with a new optional field:

```python
'role': 'ingest' | 'metadata'   # default: 'ingest'
```

- **`role='ingest'`** (default; absent field implied): the entry
  is a scrape source as today; `bin/ingest.py` may invoke its
  `ingestor_class`.
- **`role='metadata'`**: the entry is a display-only record.
  `bin/ingest.py` skips it.  Used to host the journal's official
  website, ISSN(s), historical-name aliases, etc.

Each canonical journal is allowed **at most one `role='metadata'`
entry** — the home for `aliases` and journal-level fields.  Scraper
entries (`role='ingest'`) stay narrowly focused on "how to scrape
from here".

Future role values (`'deprecated'`, `'preview'`, `'archived'`) can
be added later without revisiting the schema.

## Phased migration

### Phase 1 (this thread)

1. Add `role` filtering to `bin/ingest.py` — skip entries where
   `role != 'ingest'`.
2. Add a single pilot metadata entry: `'sydowia-journal'`
   (`role='metadata'`, owns `aliases=['Sydowia Beih.']`, carries
   the official Sydowia website + ISSN `0082-0598`).
3. Extend `normalize_journal_name()` to consult per-entry
   `aliases` lists *in addition to* the legacy
   `JOURNAL_NAME_ALIASES` dict.  Keeps the migration incremental;
   no flag-day.
4. Extend `resolve_source_name()` (in `bin/build_sources_stats.py`)
   to consult metadata entries' `issn`/`eissn` fields — so a doc
   whose ISSN matches a metadata entry's ISSN gets bucketed under
   that entry's `journal` even when its own `journal` field is
   empty.  This is the path that surfaces the 249 archive.org
   Sydowia docs.
5. Rebuild `production_v3_hand` Sources stats and verify Sydowia
   absorbs the 249 docs.

### Phase 2

Migrate the remaining nine `JOURNAL_NAME_ALIASES` entries into
per-journal metadata entries, one at a time.  Each migration:

- Add `'<journal>-journal'` metadata entry to `SOURCES` (if a
  metadata home doesn't already exist for that journal).
- Move the alias from `JOURNAL_NAME_ALIASES` to the entry's
  `aliases` list.
- Run the registry tests; visually inspect the Sources page for
  any consolidation regression.

Aliases that don't correspond to a real journal (or that exist
only to clean up parser garbage like
`'and Leucophlebs in North America. Ann. Missouri Bot. Gard.'`)
either go under the parent journal's metadata entry or are
deleted in favour of fixing the extractor that produced them.

### Phase 3

Delete `JOURNAL_NAME_ALIASES` (now empty) and the back-compat
branch in `normalize_journal_name()`.  Single source of truth:
SOURCES alone.

### Phase 4 (optional, later)

Backfill `journal=<canonical>` onto skol_dev docs whose ISSN
matches a metadata-entry ISSN — so downstream consumers don't have
to re-resolve.  Mirrors the existing `bin/backfill_journal.py`
ISSN pass but driven by the in-tree registry rather than Crossref.

## Things to confirm before starting

- ✓ No automated tooling writes back to `JOURNAL_NAME_ALIASES`.
- ✓ No external consumers import `JOURNAL_NAME_ALIASES` directly
  (only via `normalize_journal_name()`).
- ⚠ Two-tier resolution order during phase 1–2: per-entry
  `aliases` are consulted first; legacy `JOURNAL_NAME_ALIASES`
  second.  If a name is in both, the metadata entry wins — but
  the migration removes entries from the legacy dict so this
  conflict shouldn't occur in practice.

## Notes / non-goals

- Out of scope for phase 1: an admin UI for editing journal
  metadata.  All entries hand-edited in `publications.py`.
- Out of scope for phase 1: ISSN normalization on the lookup side
  (lookup is exact match on the stored ISSN string).  The
  `normalize_issn` helper in `backfill_journal.py` exists for the
  Crossref-call path; we don't need it here yet.
- Out of scope: changing the `JOURNAL_NAME_ALIASES` value side
  (canonical names) — the canonical name on the LHS of the
  current dict matches the `journal` field on the existing
  SOURCES entries.  Phase 2 just relocates the alias rows.
