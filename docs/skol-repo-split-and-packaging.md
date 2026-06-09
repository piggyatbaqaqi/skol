# Plan: Spin skol's extractable packages into siblings

**Status:** PLANNING ONLY. Execution is tracked in Trello (not here). Do after the v4 model ships (all three are library/packaging refactors, independent of the model).
**Created:** 2026-06-04

Three packages, **three different destinations and three different owners.** They are not all "skol siblings" — that's the key realization.

| Package | Source today | Destination | Owner | Provenance |
|---------|--------------|-------------|-------|------------|
| **catalog** | `skol_classifier/extraction/catalog/` | **AutonLab GitLab repo → official AutonLab PyPI** | CMU / AutonLab | their code, returning home |
| **extraction engine** | `skol_classifier/extraction/` (minus catalog) | new piggyatbaqaqi repo → PyPI | piggyatbaqaqi | yours |
| **gnservices packaging** | `packaging/skol-gnservices/` | peer repo under piggyatbaqaqi; packaging upstreamed later | globalnames (not ours) | not yours |

---

## 1. catalog → AutonLab (the twist)

**The twist, stated plainly.** Earlier framing was: "you'd republish CMU's Apache-2.0 code under your *own* piggyatbaqaqi name, carrying the attribution/NOTICE burden." Returning it to AutonLab **inverts that entirely.** It stops being derivative-redistribution-under-your-name and becomes an **upstream contribution**: factor ngautonml's `catalog/` into a standalone installable package, in **AutonLab's own repo on CMU GitLab**, published under **AutonLab's PyPI namespace** as an official AutonLab publication. skol becomes a plain downstream consumer.

Consequences of the twist:
- **It is NOT a skol sibling.** Different org (AutonLab/CMU), different host (CMU GitLab, *not* github.com/piggyatbaqaqi), CMU governance, their release/legal/PyPI process, their naming, their CI and maintainership. The "make siblings for skol" framing does not apply to this one.
- **Requires AutonLab buy-in — cannot be done unilaterally.** Needs a sponsor/contact in the lab who can accept the refactor and own the published package. This has *external lead time*, so the conversation should start early even though execution is gated behind v4.
- **Canonical source is ngautonml's `catalog/`, not skol's copy.** skol's copy is an "import-paths-only" derivative (per the header in `catalog.py`). Carve the standalone package from *upstream*, reconcile any drift, and **delete** skol's local copy rather than promoting it.
- **Net win:** skol sheds vendored code and the attribution burden; the code lives where it belongs; ngautonml ideally consumes the same standalone package (avoid a permanent fork).

**AutonLab-side work (their repo):**
- Standalone package: `pyproject.toml`, Apache-2.0 LICENSE + NOTICE retained, package name under the AutonLab namespace.
- Move `catalog.py`, `memory_catalog.py`, `catalog_element_mixin.py`, the `widgets/` test-fixture plugins, and the tests.
- Publish to PyPI via AutonLab's account/CI.

**skol-side work (this repo — small):**
- Add a dependency on the published package (`setup.py`).
- Rewrite the ~11 import sites: `from skol_classifier.extraction.catalog import …` → `from <autonlab-pkg> import …` (plus the relative `.catalog` / `..catalog` forms in `dispatcher.py`, `dispatcher_test.py`, `components_test.py`, `inspectors_test.py`).
- Delete `skol_classifier/extraction/catalog/`.
- Optional one-release compat shim re-exporting from the new package so nothing breaks mid-migration.
- Verify extraction tests pass (the `widgets/` fixtures travel upstream with the package).

**Coordination questions for AutonLab:** who sponsors/owns it; PyPI namespace + package name; standalone repo vs. a subpackage of ngautonml published separately; who maintains the standalone vs. ngautonml's in-tree copy (avoid a fork — ideally ngautonml depends on the standalone too).

---

## 2. extraction engine → PyPI under piggyatbaqaqi

This one **is** a real piggyatbaqaqi sibling — your own code, normal PyPI publish, no provenance wrinkle.

**Boundary is surprisingly clean** (verified 2026-06-04): `skol_classifier/extraction/` has **no imports from skol's core** (`skol_classifier.*` outside `extraction.`). The single outward tendril is:

- `from couchdb_file import CouchDBFile` in `extraction/components/treatment_assembler.py:32` (a top-level skol module, not deep core).

**The one seam to resolve:** abstract that `CouchDBFile` payload behind a small `Protocol`/interface and inject it, **or** also extract `couchdb_file`. That decision is the bulk of the scoping work; everything else is mechanical.

**Layering after catalog leaves:** the extraction engine depends on the AutonLab catalog package — `extraction-engine (piggyatbaqaqi) → catalog (AutonLab)`. So do this *after/with* the catalog move, pointing the dependency at the published catalog rather than re-migrating imports twice.

**Scope to decide:** what's in (components, inspectors, `dispatcher.py`) vs. out; package name; whether the toy `widgets/` fixtures stay with catalog (they're catalog's, not extraction's).

---

## 3. gnservices → peer under piggyatbaqaqi, upstream eventually

`packaging/skol-gnservices/` (`build-deb.sh`, `debian/`, `fetch_binaries.sh`) only **packages upstream binaries** — gnfinder + gnparser (globalnames.org / Mozzherin). **Not your code.**

- **Short term:** move to a peer repo under piggyatbaqaqi to decouple it from skol's tree (it has no reason to live inside skol).
- **Long term:** contribute the Debian packaging upstream to the gnfinder / gnparser projects, where it belongs.

Fully independent of the other two — do anytime.

---

## 4. Sequencing

- All three are library/packaging refactors, **independent of the v4 model** — gated only by the general "v4 first" preference.
- **catalog**: start the AutonLab conversation *early/in parallel* (external lead time), even though code work waits for v4.
- **extraction engine**: after/with catalog, so its dependency points at the published catalog package.
- **gnservices**: independent, anytime.
- **Tracking lives in Trello** (user owns it). This file is reference only — do not add task tracking here.
