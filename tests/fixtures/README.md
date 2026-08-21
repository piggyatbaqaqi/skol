# tests/fixtures/pathologies.json

Machine-readable catalog of pathology exemplars and poster-child
treatments used to regression-test the detector suite in
[`treatments_to_structured/triage_signals.py`](../../treatments_to_structured/triage_signals.py)
and [`treatments_to_structured/gn_client.py`](../../treatments_to_structured/gn_client.py).

The prose narrative for each entry lives in
[`docs/data_quality_production_v4_model.md`](../../docs/data_quality_production_v4_model.md).

That memo's **Detector backlog** section runs the other way:
each proposed-but-unimplemented detector names the fixture
entries that must fire it and the ones that must stay silent.
Adding an entry that gates a backlog item — or noticing that
an existing entry disqualifies a proposed formulation — is
worth a line there too.
This file is the machine-readable pair: the actual treatment
content plus the labelled expected detector outputs.

Consumed by [`tests/pathologies_test.py`](../pathologies_test.py) —
the CI regression bar for M1's detector suite.

## Convention

**Every taxon mentioned in the data-quality memo should have a
fixture entry** (or a note that its content isn't captured
because the source treatment isn't accessible).  This pairing
turns memo prose into automated regression coverage.

**Two sections**:

* `poster_children` — §0.5 exemplars.  Must fire ZERO detector
  flags.  A regression against a poster-child means the detector
  suite has become too aggressive against clean single-species
  content.  Do not add or remove entries without corresponding
  §0.5 memo edits.
* `pathologies` — one canonical exemplar per pathology class
  (§1 through §14).  Each entry's `expected_flags` is the
  ground-truth baseline of what the current detector suite
  reports.  Regression tests assert exact match.

## Schema

```jsonc
{
  "_schema_version": 1,
  "_docs": "See tests/fixtures/README.md",
  "poster_children": [ /* array of entries */ ],
  "pathologies":     [ /* array of entries */ ]
}
```

### Entry shape

```jsonc
{
  "id": "taxon_<sha256>",             // full 64-char treatment ID
  "source_experiment": "production_v4",// which experiment context
  "source_db":  "skol_exp_production_v4_02_00_treatments_prose",
  "captured_at": "YYYY-MM-DD",        // ISO date fixture was captured
  "captured_rev": "3-abcd...",        // CouchDB _rev at capture time
  "class": "§N-short-label",          // human-readable class label
  "description": "...",               // full description text
  "diagnosis":   "...",               // full diagnosis text (may be "")
  "synthetic_nomenclature": false,    // synth-nomen fallback state
  "authored_binomial_in_desc": true,  // labelled from live gn services
  "expected_flags": ["§6:merge_metric", "§10:tail_clip", ...],
  "known_missed_flags": [],           // detectors that SHOULD fire but don't
  "notes": "..."                      // narrative prose from memo
}
```

### Field semantics

* **`id`** — the treatment_prose doc `_id` in CouchDB.  Full
  64-char hash, not a prefix, to guarantee uniqueness.
* **`source_experiment`** + **`source_db`** — provenance.  The
  same taxon-hash may exist in v4 and v5 with different content
  after re-extraction; the fixture is anchored to one specific
  extraction pass.  Adding v5 entries later should keep the v4
  ones (both preserved for cross-generation regression).
* **`captured_rev`** — CouchDB `_rev` at capture time.  A
  maintenance script can compare this to the current live `_rev`
  and flag drift.  Drift doesn't force an update (drift may be
  intentional; the fixture is frozen ground truth) but should be
  visible.
* **`class`** — human-readable label following the §N-shortname
  convention.  §1 through §14 mirror the memo section
  numbering; refine with sub-labels
  (`§6-compact-congenerics`, `§10-tail-clip-only`, etc.).
* **`description` / `diagnosis`** — verbatim treatment content
  at capture time.  DO NOT normalize whitespace, strip OCR
  noise, or fix encoding artifacts.  The whole point is to
  capture the actual input the detectors see.
* **`synthetic_nomenclature`** — the `synthetic_nomenclature`
  field of the treatment_prose doc.  Consumed by the detector
  suite.
* **`authored_binomial_in_desc`** — pre-computed via
  `gn_client.authored_binomial_in_text` at capture time.
  Stored to keep tests hermetic (no HTTP in the test suite).
  Drift from current gn-services output should be caught by a
  separate periodic re-labelling script; the CI test uses the
  stored value.
* **`expected_flags`** — array of flag PREFIXES that should
  fire.  Use prefixes like `§6:merge_metric`, not
  `§6:merge_metric=39`, because the metric value fluctuates
  with content changes.  The test strips `=<value>` before
  comparison.
* **`known_missed_flags`** — informational.  Records
  detectors that SHOULD conceptually catch this class but
  currently don't (e.g., taxon_2f276bfa's mid-word OCR that
  defeats gnfinder; deferred to Trello #395).  NOT asserted by
  tests — this field documents gaps, not regressions.
* **`notes`** — brief prose (1-3 sentences) explaining what
  this entry demonstrates.  Should reference the memo §
  number.  If the entry has a known false-positive flag that's
  currently in `expected_flags`, note it here so future
  detector work can find it.

## Adding a new entry

**When you add a taxon reference to the memo, add a fixture
entry.**  Steps:

1. **Pick the class label.**  Which memo § does it exemplify?
   If it's a new sub-shape of an existing §, use a compound
   label (e.g., `§10-mid-word-hyphen-tail`).  If it's a new §
   entirely, coordinate with the memo edit.
2. **Extract the treatment content.**  From the production_v4
   corpus:
   ```python
   import sys
   sys.path.insert(0, 'bin')
   from env_config import get_env_config
   c = get_env_config()
   import couchdb
   s = couchdb.Server(c['couchdb_url'])
   s.resource.credentials = (
       c['couchdb_username'], c['couchdb_password'],
   )
   db = s['skol_exp_production_v4_02_00_treatments_prose']
   doc = db['taxon_<full-id>']
   ```
3. **Capture the labelled state.**  Run every detector against
   the extracted content and record the flags that actually fire
   (as prefixes, without the `=<value>` suffix).  For
   `authored_binomial_in_desc`, call
   `gn_client.authored_binomial_in_text(desc)` with live
   services.
4. **Assemble the entry** matching the schema above.  Insert
   into the appropriate section (`poster_children` or
   `pathologies`).
5. **Run the tests**: `pytest tests/pathologies_test.py`.  If the
   entry is a poster-child, `expected_flags` must be `[]`; if
   any flag fires, either the entry doesn't belong in
   `poster_children` or a detector needs adjustment.
6. **Update the memo** with the taxon reference and the class
   label.

For bulk additions (e.g., after M2 introduces new detectors),
adapt the one-time extraction script referenced in the M1.5
commit message.

## Maintenance

**When treatment content changes** (re-extraction, correction,
schema migration):

* If `captured_rev` differs from the current CouchDB `_rev`,
  the fixture is stale.  Options:
  * **Bump the fixture**: re-extract content, re-label
    `expected_flags`, update `captured_at` and `captured_rev`.
    Prefer this when the content drift is legitimate.
  * **Freeze the fixture**: keep the old snapshot as
    historical ground truth, add a fresh entry with the new
    content.  Prefer this when reproducibility of past
    regression tests matters.
* If a detector's behavior changes (M2/M3 detector additions,
  Group C precision improvements), update `expected_flags` for
  every affected entry.  The test failure will list which
  entries need attention.

**When adding a new detector**:

* Rerun the extraction script (or manually update flags) to
  record which pathology entries fire the new flag.
* If a poster-child fires the new flag, either the detector is
  too aggressive or the poster-child's class label needs
  revisiting.

## Not in this fixture

* **Live gn services state**.  `authored_binomial_in_desc` is
  labelled at capture time and stored; the CI test doesn't call
  gnfinder/gnparser.  If gn services precision changes (e.g.,
  we swap gnfinder for a newer version), a separate script
  should relabel affected entries.
* **Reviewer_action counts**.  Round-1/round-2 kept/added/
  deleted data lives in the CouchDB `features_status` DB and
  in memo prose.  Not part of the detector regression bar.
* **CouchDB access**.  Tests are hermetic.  Extraction and
  relabelling scripts touch CouchDB; tests don't.

## Cross-references

* [`docs/data_quality_production_v4_model.md`](../../docs/data_quality_production_v4_model.md) — prose narrative + §0.5 poster-children
* [`tests/pathologies_test.py`](../pathologies_test.py) — CI test harness
* [`treatments_to_structured/triage_signals.py`](../../treatments_to_structured/triage_signals.py) — detector suite
* [`docs/plans/production-v5-execution.md`](../../docs/plans/production-v5-execution.md) — M1.5 rationale
