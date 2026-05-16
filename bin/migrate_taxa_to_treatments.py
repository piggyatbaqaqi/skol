#!/usr/bin/env python3
"""Migrate CouchDB taxa databases to the treatments vocabulary.

This is Step 3 from docs/taxon_to_treatment_plan.md.  Run once against
dev as a rehearsal, then again against prod via SSH.

The migration is **additive**: original ``*_taxa*`` databases are not
modified or deleted.  New ``*_treatments*`` databases are created by
CouchDB replication and then have their per-document field names
rewritten (``taxon`` → ``treatment``).  Rollback = drop the new
databases.

Experiment documents in ``skol_experiments`` are rewritten in place:
``databases.taxa`` → ``databases.treatments`` (and same for the
``taxa_full`` variant), and the *values* are transformed via
``rename_db_name()`` so non-local references (e.g. prod-only DBs seen
in a dev experiment doc) are also moved to the new vocabulary.

Usage::

    # Print what would change without writing anything.
    python bin/migrate_taxa_to_treatments.py --dry-run

    # Actually apply the migration.
    python bin/migrate_taxa_to_treatments.py --execute

    # Limit to specific source DBs (repeatable).
    python bin/migrate_taxa_to_treatments.py --execute \\
        --source-db skol_taxa_dev

Environment variables (or ~/.skol_env):
    COUCHDB_URL         CouchDB server URL
    COUCHDB_USER        CouchDB username
    COUCHDB_PASSWORD    CouchDB password
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Databases that match is_taxa_db_name() but are not actually taxa data
# stores — exclude from auto-discovery.  Add new entries here as the
# project evolves; the prod run can supply its own denylist via CLI.
_DEFAULT_DB_DENYLIST = frozenset({
    # Dedup history table from a Feb 2026 cleanup; maps old taxon_* IDs
    # to dedup winners.  Not a current taxa store.
    "skol_taxa_migration_dev",
})

_EXPERIMENTS_DB = "skol_experiments"
_BULK_CHUNK_SIZE = 200


# ---------------------------------------------------------------------------
# Pure helpers (no CouchDB)
# ---------------------------------------------------------------------------

def rename_db_name(old_name: str) -> str:
    """Underscore-component-wise replacement of ``taxa`` with ``treatments``.

    ``skol_taxa_dev`` → ``skol_treatments_dev``
    ``skol_taxa_full_dev`` → ``skol_treatments_full_dev``
    ``skol_exp_hand_annotated_taxa`` → ``skol_exp_hand_annotated_treatments``
    ``skol_taxon_clusters`` → unchanged (the 'taxon' component is not 'taxa')
    """
    return "_".join(
        "treatments" if part == "taxa" else part
        for part in old_name.split("_")
    )


def is_taxa_db_name(name: str) -> bool:
    """True when ``taxa`` appears as an underscore-delimited word.

    Distinct from substring match so e.g. ``skol_taxon_clusters`` and
    ``skol_taxonomy`` are not misidentified as taxa stores.
    """
    return "taxa" in name.split("_")


def discover_source_dbs(
    all_dbs: Iterable[str],
    denylist: Iterable[str] = _DEFAULT_DB_DENYLIST,
) -> List[str]:
    """Filter the server's database list down to taxa stores to migrate."""
    denyset = set(denylist)
    return sorted(
        db for db in all_dbs
        if is_taxa_db_name(db)
        and not db.startswith("_")
        and db not in denyset
    )


def rename_taxon_field(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Rename the document's ``taxon`` field to ``treatment``.

    Returns ``(maybe_new_doc, changed)``.  When no change is needed,
    returns the *original* doc unchanged (callers can ``is`` test).
    Raises ``ValueError`` if a doc carries both ``taxon`` and
    ``treatment`` — that's data divergence we should not silently
    paper over.
    """
    if "taxon" not in doc:
        return doc, False
    if "treatment" in doc:
        raise ValueError(
            f"Doc {doc.get('_id')!r} has both 'taxon' and 'treatment' fields"
        )
    new_doc = dict(doc)
    new_doc["treatment"] = new_doc.pop("taxon")
    return new_doc, True


def rename_experiment_databases(
    doc: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Rewrite ``databases.taxa`` / ``databases.taxa_full`` keys in an
    experiment document, and transform their values via ``rename_db_name``.
    Idempotent.
    """
    databases = doc.get("databases")
    if not isinstance(databases, dict):
        return doc, False

    changes = [
        ("taxa", "treatments"),
        ("taxa_full", "treatments_full"),
    ]
    changed = False
    new_dbs = dict(databases)
    for old_key, new_key in changes:
        if old_key not in new_dbs:
            continue
        value = new_dbs.pop(old_key)
        if isinstance(value, str) and is_taxa_db_name(value):
            value = rename_db_name(value)
        new_dbs[new_key] = value
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

    config = get_env_config()
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server


def replicate(
    server: Any,
    source: str,
    target: str,
    dry_run: bool,
) -> None:
    """Run a one-shot CouchDB replication source → target.

    Uses ``_replicate`` with ``create_target=true`` so the new DB is
    created if needed.  Skipped when ``target`` already exists with a
    matching ``doc_count`` (a previous run completed).
    """
    if target in server:
        target_db = server[target]
        target_count = target_db.info()["doc_count"]
        source_count = server[source].info()["doc_count"]
        if target_count == source_count:
            logging.info(
                "  %s → %s: target already exists with matching "
                "doc_count=%d (skipping)",
                source, target, target_count,
            )
            return
        logging.warning(
            "  %s → %s: target exists but doc_count differs "
            "(source=%d target=%d); replication will sync",
            source, target, source_count, target_count,
        )

    if dry_run:
        logging.info("  [DRY-RUN] would replicate %s → %s", source, target)
        return

    started = time.time()
    server.replicate(source, target, create_target=True)
    elapsed = time.time() - started
    logging.info("  %s → %s: replicated in %.1fs", source, target, elapsed)


def rewrite_documents(
    server: Any,
    db_name: str,
    rewrite_fn: Any,
    dry_run: bool,
    chunk_size: int = _BULK_CHUNK_SIZE,
) -> Tuple[int, int]:
    """Apply ``rewrite_fn(doc) -> (new_doc, changed)`` to every doc in a DB.

    Skips design documents.  Writes back via ``_bulk_docs`` in batches.
    Returns ``(scanned, written)``.
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
        if len(pending) >= chunk_size:
            written += _flush()

    written += _flush()
    return scanned, written


# ---------------------------------------------------------------------------
# Top-level migration
# ---------------------------------------------------------------------------

def migrate(
    server: Any,
    dry_run: bool,
    source_dbs: Optional[List[str]] = None,
    denylist: Iterable[str] = _DEFAULT_DB_DENYLIST,
) -> Dict[str, Any]:
    """Run all three migration phases.  Returns a summary dict."""
    if source_dbs is None:
        source_dbs = discover_source_dbs(list(server), denylist=denylist)

    logging.info(
        "Migration plan (%s mode):", "DRY-RUN" if dry_run else "EXECUTE"
    )
    for db in source_dbs:
        logging.info("  - %s → %s", db, rename_db_name(db))

    # Phase 1: replicate each source DB to its new name.
    logging.info("\nPhase 1: replication")
    for source in source_dbs:
        target = rename_db_name(source)
        replicate(server, source, target, dry_run=dry_run)

    # Phase 2: rewrite the 'taxon' field in each new DB.
    logging.info("\nPhase 2: rewrite 'taxon' → 'treatment' field in new DBs")
    per_db_counts: Dict[str, Tuple[int, int]] = {}
    for source in source_dbs:
        target = rename_db_name(source)
        if dry_run and target not in server:
            # Target wasn't actually created in dry-run; skip rewriting.
            logging.info("  %s: [DRY-RUN] target not present; estimating "
                         "from source", target)
            scanned, written = rewrite_documents(
                server, source, rename_taxon_field, dry_run=True,
            )
        else:
            scanned, written = rewrite_documents(
                server, target, rename_taxon_field, dry_run=dry_run,
            )
        per_db_counts[target] = (scanned, written)
        logging.info(
            "  %s: scanned=%d rewritten=%d", target, scanned, written
        )

    # Phase 3: rewrite experiment documents.
    logging.info("\nPhase 3: rewrite experiment documents")
    if _EXPERIMENTS_DB in server:
        scanned, written = rewrite_documents(
            server, _EXPERIMENTS_DB, rename_experiment_databases,
            dry_run=dry_run,
        )
        logging.info(
            "  %s: scanned=%d rewritten=%d", _EXPERIMENTS_DB, scanned, written
        )
    else:
        logging.warning("  %s not present; skipping", _EXPERIMENTS_DB)
        scanned = written = 0

    return {
        "source_dbs": source_dbs,
        "per_db_doc_counts": per_db_counts,
        "experiment_doc_scanned": scanned,
        "experiment_doc_written": written,
        "dry_run": dry_run,
    }


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
        "--dry-run",
        action="store_true",
        help="Print every change without writing to CouchDB.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually apply the migration.  Run --dry-run first.",
    )
    parser.add_argument(
        "--source-db",
        action="append",
        dest="source_dbs",
        metavar="DB",
        help="Restrict to specific source DBs (repeatable). "
             "Default: auto-discover all *_taxa* DBs minus the denylist.",
    )
    parser.add_argument(
        "-v", "--verbosity",
        action="count",
        default=1,
        help="Increase output verbosity (repeatable).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbosity >= 1 else logging.WARNING,
        format="%(message)s",
    )

    server = _connect_server()
    summary = migrate(
        server,
        dry_run=args.dry_run,
        source_dbs=args.source_dbs,
    )

    print("\n=== Migration summary ===")
    print(f"mode:               {'DRY-RUN' if summary['dry_run'] else 'EXECUTE'}")
    print(f"source databases:   {len(summary['source_dbs'])}")
    for source in summary["source_dbs"]:
        target = rename_db_name(source)
        scanned, written = summary["per_db_doc_counts"].get(target, (0, 0))
        print(f"  {source} → {target}: scanned={scanned} rewritten={written}")
    print(f"experiment docs:    scanned={summary['experiment_doc_scanned']} "
          f"rewritten={summary['experiment_doc_written']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
