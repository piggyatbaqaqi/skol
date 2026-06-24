#!/usr/bin/env python3
"""Rename experiment-doc database keys to post-2026-06-10 canonical names.

Targets the ``skol_experiments`` database, NOT individual treatments
databases.  Renames:

    databases.treatments       → databases.treatments_prose
    databases.treatments_full  → databases.treatments_structured

The actual treatments DBs (e.g.
``skol_exp_production_v4_02_00_treatments_prose``) already have the
post-cleanup names; this script just brings the experiment registry
docs in line.

Conflict handling: a doc that carries BOTH ``treatments`` AND
``treatments_prose`` is OK iff the values match (the legacy key is
dropped).  If they differ, raise ``ValueError`` — silently picking
one would lose data.

The migration is idempotent: re-running on an already-migrated doc
is a no-op.

Usage::

    # Print what would change without writing.
    python bin/migrate_treatments_to_prose.py --dry-run

    # Apply.
    python bin/migrate_treatments_to_prose.py --execute

Environment variables (or ~/.skol_env):
    COUCHDB_URL         CouchDB server URL
    COUCHDB_USER        CouchDB username
    COUCHDB_PASSWORD    CouchDB password
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

# env_config is lazy-imported inside _connect_server() so the pure-helper
# unit tests don't need a couchdb / env_config install to run.

_EXPERIMENTS_DB = "skol_experiments"
_BULK_CHUNK_SIZE = 200

# Mid-migration → post-2026-06-10 canonical.  Order matters for logging
# only; the rewrite is keyed on the dict.
_RENAMES: Tuple[Tuple[str, str], ...] = (
    ("treatments", "treatments_prose"),
    ("treatments_full", "treatments_structured"),
)


# ---------------------------------------------------------------------------
# Pure helper (no CouchDB)
# ---------------------------------------------------------------------------

def rename_experiment_databases(
    doc: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Rewrite ``databases.treatments`` → ``databases.treatments_prose``
    (and the ``_full`` → ``_structured`` pair) on an experiment doc.

    Returns ``(maybe_new_doc, changed)``.  When no change is needed,
    returns the original doc unchanged.

    Raises ``ValueError`` if a doc carries both the legacy key and
    the canonical key with *different* values — that's data divergence
    that needs human attention rather than a silent decision.
    """
    databases = doc.get("databases")
    if not isinstance(databases, dict):
        return doc, False

    new_dbs = dict(databases)
    changed = False
    for legacy_key, canonical_key in _RENAMES:
        if legacy_key not in new_dbs:
            continue
        legacy_value = new_dbs[legacy_key]
        canonical_value = new_dbs.get(canonical_key)
        if canonical_value is not None and canonical_value != legacy_value:
            raise ValueError(
                f"Doc {doc.get('_id')!r}: databases.{legacy_key}="
                f"{legacy_value!r} but databases.{canonical_key}="
                f"{canonical_value!r}; values differ — resolve by hand "
                "before migrating.",
            )
        # Equal values OR canonical absent: drop legacy, set canonical.
        del new_dbs[legacy_key]
        new_dbs[canonical_key] = legacy_value
        changed = True

    if not changed:
        return doc, False
    new_doc = dict(doc)
    new_doc["databases"] = new_dbs
    return new_doc, True


# ---------------------------------------------------------------------------
# CouchDB operations
# ---------------------------------------------------------------------------

def _connect_server() -> Any:
    """Connect to CouchDB using the project's env config."""
    import couchdb as couchdb_lib
    from env_config import get_env_config

    config = get_env_config()
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server


def rewrite_documents(
    server: Any,
    db_name: str,
    rewrite_fn: Any,
    dry_run: bool,
    chunk_size: int = _BULK_CHUNK_SIZE,
) -> Tuple[int, int]:
    """Apply ``rewrite_fn(doc) -> (new_doc, changed)`` to every doc in
    a DB.  Skips design documents.  Writes back via ``_bulk_docs`` in
    batches.  Returns ``(scanned, written)``.
    """
    db = server[db_name]
    scanned = 0
    pending: List[Dict[str, Any]] = []
    written = 0

    def _flush() -> int:
        nonlocal pending
        if not pending:
            return 0
        if dry_run:
            count = len(pending)
            pending = []
            return count
        results = db.update(pending)
        count = sum(1 for ok, _id, _rev in results if ok)
        failures = [
            (_id, str(_rev))
            for ok, _id, _rev in results if not ok
        ]
        if failures:
            logging.error(
                "  %s: %d/%d bulk failures; first: %s",
                db_name, len(failures), len(results), failures[0],
            )
        pending = []
        return count

    for doc_id in db:
        if doc_id.startswith("_design/"):
            continue
        scanned += 1
        doc = db[doc_id]
        new_doc, changed = rewrite_fn(dict(doc))
        if not changed:
            continue
        pending.append(new_doc)
        if dry_run:
            logging.info(
                "  [DRY-RUN] would rewrite %s: %s",
                doc_id,
                {k: new_doc["databases"][k] for k in new_doc["databases"]
                 if k in {"treatments_prose", "treatments_structured"}},
            )
        if len(pending) >= chunk_size:
            written += _flush()

    written += _flush()
    return scanned, written


# ---------------------------------------------------------------------------
# Top-level migration
# ---------------------------------------------------------------------------

def migrate(server: Any, dry_run: bool) -> Dict[str, Any]:
    """Run the migration.  Returns a summary dict."""
    if _EXPERIMENTS_DB not in server:
        logging.warning("%s not present; nothing to do", _EXPERIMENTS_DB)
        return {"scanned": 0, "written": 0, "dry_run": dry_run}

    logging.info(
        "%s mode: %s",
        "DRY-RUN" if dry_run else "EXECUTE", _EXPERIMENTS_DB,
    )
    scanned, written = rewrite_documents(
        server, _EXPERIMENTS_DB, rename_experiment_databases, dry_run=dry_run,
    )
    logging.info("  scanned=%d rewritten=%d", scanned, written)
    return {"scanned": scanned, "written": written, "dry_run": dry_run}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Print every change without writing to CouchDB.",
    )
    mode.add_argument(
        "--execute", action="store_true",
        help="Actually apply the migration.  Run --dry-run first.",
    )
    parser.add_argument(
        "-v", "--verbosity", action="count", default=1,
        help="Increase output verbosity (repeatable).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbosity >= 1 else logging.WARNING,
        format="%(message)s",
    )

    server = _connect_server()
    summary = migrate(server, dry_run=args.dry_run)

    print("\n=== Migration summary ===")
    print(f"mode:        {'DRY-RUN' if summary['dry_run'] else 'EXECUTE'}")
    print(f"scanned:     {summary['scanned']} docs")
    print(f"rewritten:   {summary['written']} docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
