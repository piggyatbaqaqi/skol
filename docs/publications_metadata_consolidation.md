# Normalizing publication metadata into JOURNALS + SOURCES

Status: **plan under review**.  Refactor of `ingestors/publications.py`
to separate journal-level facts from ingestion-source mechanics.

## Goal

Split the current single `SOURCES` table into two related tables:

- **`JOURNALS`** — keyed by a stable opaque slug; one row per
  real-world journal (or journal-like reference work).  Owns the
  facts that are properties of the *journal*: display name,
  official website, ISSN(s), eISSN, journal-DOI, ISBN, publisher,
  historical-name aliases.
- **`SOURCES`** — keyed by ingestion-source name; one row per
  *way to fetch* articles for some journal (RSS, PMC, Crossref,
  local mirror, etc.).  Each row references its journal by **slug**
  (foreign key into `JOURNALS`).

The parallel `JOURNAL_NAME_ALIASES` dict goes away — aliases live
on their journal record in `JOURNALS`.

## Identity / primary key

`JOURNALS` is keyed by a **stable human-readable slug**, not by
ISSN.  Examples: `sydowia`, `persoonia`, `journal-of-fungi`,
`california-fungi`, `funga-nordica`.

ISSN-as-key was rejected because it doesn't cover what we
actually have:

- Pre-ISSN journals (pre-~1971, plus tiny journals that never
  registered).
- Books / reference works the literature ingestor treats as
  journals (`funga-nordica` has an ISBN, not an ISSN;
  `california-fungi` has neither).
- Synthetic roll-ups (no formal identifier at all).

The slug is stable across display-name renames (changing
``'name': 'Sydowia'`` to ``'name': 'Sydowia (Austria)'`` doesn't
break the foreign keys from `SOURCES`), works as a future URL
component (`/sources/sydowia/`), and is greppable.

`ISSN`, `eISSN`, journal-`DOI`, and `ISBN` live on the record as
**searchable attributes** — `find_journal_by_issn(issn)` and
similar return the slug.  None is required; the scaffolding
script fills them in where Crossref / DOAJ / NLM have them and
leaves them absent otherwise.

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
# JOURNALS — one row per real-world journal, keyed by stable slug.
JOURNALS: Dict[str, JournalEntry] = {
    'sydowia': {
        'name':       'Sydowia',
        'address':    'https://www.sydowia.at/',
        'issn':       '0082-0598',
        'aliases':    ['Sydowia Beih.'],
        'publisher':  'Verlag Ferdinand Berger & Söhne',
    },
    'persoonia': {
        'name':       'Persoonia',
        'address':    'https://persoonia.org/',
        'aliases':    ['Persoonia - Molecular Phylogeny and Evolution of Fungi'],
        'publisher':  'Naturalis Biodiversity Center',
    },
    'journal-of-fungi': {
        'name':       'Journal of Fungi',
        'address':    'https://www.mdpi.com/journal/jof',
        'publisher':  'MDPI',
    },
    'california-fungi': {  # no ISSN — book-like reference
        'name':       'California Fungi',
        'aliases':    [],
    },
    'funga-nordica': {     # no ISSN — book series
        'name':       'Funga Nordica',
        'isbn':       '978-87-983961-3-2',
    },
    # ...
}

# SOURCES — one row per scrape mechanism; each row's `journal`
# field is a foreign key (slug) into JOURNALS.
SOURCES: Dict[str, SourceEntry] = {
    'persoonia-pmc': {
        'journal':         'persoonia',         # FK slug
        'address':         '...PMC URL...',
        'source':          'pmc',
        'ingestor_class':  'PMCIngestor',
    },
    'persoonia-rss': {
        'journal':         'persoonia',         # same slug, different fetch
        ...
    },
    'jof-pmc': {
        'journal':         'journal-of-fungi',  # slug, NOT 'Journal of Fungi (PMC)'
        ...
    },
    # ...
}
```

## Static typing — TypedDict

The dict literals stay (they're the readable source-of-truth
format), but mypy gets real type narrowing at every call site
via `TypedDict` declarations co-located with the registry:

```python
from typing import List, Optional
# Python 3.11+: from typing import NotRequired
# Earlier:      from typing_extensions import NotRequired

class JournalEntry(TypedDict):
    name:      str                            # required
    aliases:   NotRequired[List[str]]
    address:   NotRequired[str]               # journal homepage
    issn:      NotRequired[str]
    eissn:     NotRequired[str]
    isbn:      NotRequired[str]               # for book-like refs
    doi:       NotRequired[str]               # journal-DOI, not article
    publisher: NotRequired[str]
    abbrev:    NotRequired[str]               # ISO 4 abbreviation


class SourceEntry(TypedDict):
    journal:         str                       # required: FK slug → JOURNALS
    source:          str                       # required: ingestor family
    ingestor_class:  str                       # required
    address:         NotRequired[str]          # scrape endpoint
    local_path:      NotRequired[str]
    local_path_prefix: NotRequired[str]
    url_prefix:      NotRequired[str]
    rate_limit_min_ms: NotRequired[int]
    rate_limit_max_ms: NotRequired[int]
```

The benefit lands at every consumer:

```python
def render_journal_row(slug: str) -> str:
    journal = JOURNALS[slug]                    # JournalEntry
    return f'<a href="{journal.get("address", "#")}">{journal["name"]}</a>'
    #             mypy knows 'name' is required (str)
    #             and 'address' is Optional[str] via .get()
```

`name` is the only field flagged as required on `JournalEntry`
(every journal must have a display name); everything else is
optional because the data really is heterogeneous (some journals
have ISSN, some have ISBN, some have neither).  `SourceEntry`
requires `journal` (the FK), `source`, and `ingestor_class`
because without those the row isn't actionable as an ingestion
source.

Type-checking validation (`mypy ingestors/publications.py`) catches
typos in field names, missing required fields, and wrong types at
edit time — replaces the "single `_validate_journal_entry` function
called at import" idea proposed earlier.  Per CLAUDE.md the new
code passes mypy.

Helper methods on `PublicationRegistry`:
- `get_journal(slug_or_name: str) -> Optional[JournalEntry]` —
  JOURNALS lookup with alias resolution.  Accepts either the slug
  directly or a display name (or alias).
- `find_journal_by_issn(issn: str) -> Optional[str]` — scans
  JOURNALS for an entry whose `issn` or `eissn` matches; returns
  the slug.
- `find_journal_by_doi(doi: str) -> Optional[str]` — same shape,
  scans the journal-DOI attribute.  (Per-article DOIs are out of
  scope — those need a Crossref lookup, not a JOURNALS scan.)
- `find_journal_by_isbn(isbn: str) -> Optional[str]` — same shape
  for book-like reference works.
- `normalize_journal_name(name: str) -> str` — consults
  `JOURNALS[*].aliases` lists; returns the canonical display name
  (the `name` field) of the matching journal, or the input
  unchanged if no alias matches.

## Enrichment workflow (one-shot, no runtime Crossref dependency)

`JOURNALS` is built once via a scaffolding script and committed
to Git.  Runtime code never talks to Crossref.

Rationale:
- `JOURNALS` is itself the cache — a hand-curated, Git-tracked
  Python literal.  No "is the cache stale?" decision; `git log`
  tells you when the source of truth last changed.
- Hand-edits survive.  Crossref's title for Sydowia is `"Sydowia"`
  but we may want `"Sydowia Beih."` recorded as an alias — that's
  a hand-edit, not a Crossref override layer.
- No new CouchDB database to maintain.  The skol stack already
  has ~10; adding one for ~20 hand-curated rows doesn't earn its
  keep.

Crossref doesn't cover everything we want.  It gives canonical
title, ISSN, eissn, publisher reliably; ISO 4 abbreviations are
sometimes in `short-container-title` but inconsistent; official
journal websites are rarely returned.  The scaffolding script
treats Crossref as one input among several — abbreviation and
website fields stay hand-edited.

### Scaffolding script — shape

```
bin/scaffold_journals.py
    --crossref-mailto piggy.yarroll+skol@gmail.com   # polite pool
    --output ingestors/JOURNALS_draft.py
    [--journal sydowia]                              # one at a time, OR
    [--all]                                          # bulk scaffold from SOURCES
```

Behaviour:
1. Walk current `SOURCES` and collect unique `journal` strings
   (deduplicating canonical-name-mixed-with-publisher-tag forms).
2. For each, infer the slug (`'Sydowia' → 'sydowia'`,
   `'Journal of Fungi (PMC)' → 'journal-of-fungi'`).
3. If any current `SOURCES` row has an ISSN, look up the journal
   on Crossref's `/journals/{issn}` endpoint (polite pool).
4. Emit a draft `JOURNALS_draft.py` whose entries combine
   Crossref response fields + ISSNs harvested from `SOURCES` +
   placeholder hand-edit fields (`# TODO: website`,
   `# TODO: aliases`).
5. Operator hand-edits the draft, then moves entries into
   `ingestors/publications.py` as part of the phase-1 commit.

Re-runnable for a single journal when adding a new one
(`--journal new-slug`).  The script doesn't overwrite existing
`JOURNALS` entries — it appends a draft for inspection.

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

### Phase 2 — deliver `bin/rename_journals.py` (script only)

**Goal**: deliver a generic rewrite-script that walks the ingest
database, runs a mapping function over each doc's ``journal``
field, and writes back when the mapping changes the value.

1. Generic `bin/rename_journals.py` — takes a mapping function;
   default for phase 2 is ``strip_publisher_suffix``.  Idempotent;
   supports ``--dry-run`` / ``--limit`` / ``--verbosity``.

**Live finding (recorded for the record)**: a phase-2 dry-run
against skol_dev returned **zero eligible docs**.  The
``" (PMC)"`` / ``" (Taylor & Francis)"`` / ``" (Internet Archive)"``
suffixes only ever lived on ``SOURCES[*].name`` (the ingestor's
display label), never on the per-doc ``journal`` value.  Every
PMC ingest already wrote the clean journal name to skol_dev.

So phase 2 commits the script as an artifact but does not write
to skol_dev.  The actual rewrite work — the legacy aliases —
moves up into phase 3 as the script's first real use.

### Phase 3 — migrate aliases into `JOURNALS`; rewrite docs

**Goal**: delete ``JOURNAL_NAME_ALIASES``; rewrite skol_dev docs
whose ``journal`` field is a known alias to its canonical form.

Aliases that need migrating into ``JOURNALS[slug].aliases``:

| Alias | Canonical | skol_dev docs |
|---|---|---|
| `'Persoonia - Molecular Phylogeny and Evolution of Fungi'` | `Persoonia` | 879 |
| `'Cryptogamie. Mycologie'` | `Cryptogamie, Mycologie` | 576 |
| `'Cryptogamie Mycologie'` | `Cryptogamie, Mycologie` | 89 |
| `'Open Access Journal of Mycology &amp; Mycological Sciences'` | `Open Access Journal of Mycology & Mycological Sciences` | 22 |
| `'Sydowia Beih.'` | `Sydowia` | (a few, mostly in scrape artefacts) |
| `'mycosphere'` (lowercase) | `Mycosphere` | 873 (NEW — not in current legacy dict) |

Plus four aliases in the current legacy dict pointing at journals
that don't have a JOURNALS entry yet (`Annals of the Missouri
Botanical Garden`, `Ann. Missouri Bot. Gard.`, the two
"in North America" fragments, the misdirected `Mycology:` →
`Mycology: ... (Taylor & Francis)` row).  Sub-decisions for each
of those are noted in the phase-3 commit when it lands.

Steps:
1. Add an ``aliases`` list to each affected ``JOURNALS`` entry
   carrying the alias variants from the table above.
2. Add the lowercase ``mycosphere`` alias to the JOURNALS entry
   (one new value).
3. Resolve the four orphan aliases — either add the missing
   JOURNALS entries (Annals of the Missouri Botanical Garden,
   etc.) or delete the orphan aliases.
4. Fix the misdirected ``'Mycology: An International Journal on
   Fungal Biology'`` → ``'Mycology: An International Journal on
   Fungal Biology (Taylor & Francis)'`` row in the legacy dict
   (it points the wrong way given phase-1B's canonical short form).
5. Update ``normalize_journal_name()`` to consult
   ``JOURNALS[*].aliases``.  Keep a short back-compat branch that
   also checks ``JOURNAL_NAME_ALIASES``; remove it (and the dict)
   in a follow-up once verified.
6. Run ``bin/rename_journals.py`` with ``normalize_journal_name``
   as the mapping function.  The dry-run should show ~2400+
   updates from the table above.
7. Apply; rebuild ``production_v3_hand`` Sources stats.

**Visible state after phase 3**: the Sources page collapses the
duplicate Persoonia / Cryptogamie / OAJMMS / mycosphere rows into
one row each.

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

- **Should `JOURNALS` track per-journal stats** (e.g., total docs,
  first-seen date) or stay purely descriptive?  Recommend purely
  descriptive — stats live in Redis (`skol:sources:stats:*`),
  not in the source registry.
- **`address` semantics**: today some `SOURCES` entries' `address`
  is a scrape endpoint (`https://api.crossref.org/...`); on the
  Sources-page row it's intended as the journal's homepage.  In
  the new shape, `JOURNALS[slug].address` is the journal homepage
  (what gets displayed); `SOURCES[key].address` is the scrape
  endpoint (only used by the ingestor).  Worth being explicit
  about this in the schema docstrings.
