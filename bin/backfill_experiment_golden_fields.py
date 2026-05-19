#!/usr/bin/env python3
"""Backfill ``databases.golden`` and ``databases.golden_ann`` on every
v1 experiment in ``skol_experiments``.

Per Step 1.A of docs/golden_v2_plan.md.  The current code hardcodes
``skol_golden`` and ``skol_golden_ann_hand`` for every experiment's
evaluate step in ``bin/manage_experiment.py``.  After Step 1.C lands,
the evaluate-step builder will read these names from each experiment's
own doc instead.  This script writes the right v1 values onto the six
known experiments so the rewire is a no-op behaviourally for v1, while
unblocking v2 to point at the new ``*_v2`` golden DBs.

Two important details:

1. **Idempotent.**  Existing values are never overwritten; the script
   only adds keys that are missing.  Running it twice produces the same
   final state.

2. **JATS experiments score against the JATS silver standard.**  The
   current hardcoded literal mis-pairs ``jats_v1`` and the
   ``taxpub_v1*`` family against ``skol_golden_ann_hand``.  The plan
   doc calls this out explicitly: the backfill restores the intended
   train/test pairing by setting their ``databases.golden_ann`` to
   ``skol_golden_ann_jats``.

Usage::

    # Print what would change, write nothing.
    python bin/backfill_experiment_golden_fields.py --dry-run

    # Actually apply.
    python bin/backfill_experiment_golden_fields.py --execute

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# Per-experiment value map (the heart of the script)
# ---------------------------------------------------------------------------

# (golden, golden_ann) pairs per experiment _id.
#
# Hand-trained experiments evaluate against the hand standard.
# JATS-trained experiments evaluate against the JATS silver, matching
# their training distribution — this corrects the latent mis-pairing
# the existing hardcoded literal in bin/manage_experiment.py had.
_EXPERIMENT_GOLDEN_MAP: Dict[str, Tuple[str, str]] = {
    "production":         ("skol_golden", "skol_golden_ann_hand"),
    "hand_annotated":     ("skol_golden", "skol_golden_ann_hand"),
    "jats_v1":            ("skol_golden", "skol_golden_ann_jats"),
    "taxpub_v1":          ("skol_golden", "skol_golden_ann_jats"),
    "taxpub_v1_int8":     ("skol_golden", "skol_golden_ann_jats"),
    "taxpub_v1_onnx_int8": ("skol_golden", "skol_golden_ann_jats"),
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def backfill_one_doc(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Apply the backfill to a single experiment doc.

    Returns ``(maybe_new_doc, changed)``.  When no change is needed,
    returns the original ``doc`` unmodified so callers can ``is``-test
    to detect no-ops.

    Design docs (``_id`` starting with ``_``) and experiments not in
    ``_EXPERIMENT_GOLDEN_MAP`` are returned unchanged.
    """
    doc_id = doc.get("_id", "")
    if doc_id.startswith("_"):
        return doc, False

    expected = _EXPERIMENT_GOLDEN_MAP.get(doc_id)
    if expected is None:
        return doc, False
    expected_golden, expected_ann = expected

    databases = doc.get("databases")
    have_golden = isinstance(databases, dict) and "golden" in databases
    have_ann = isinstance(databases, dict) and "golden_ann" in databases
    if have_golden and have_ann:
        return doc, False

    new_doc = dict(doc)
    new_databases: Dict[str, Any] = (
        dict(databases) if isinstance(databases, dict) else {}
    )
    if "golden" not in new_databases:
        new_databases["golden"] = expected_golden
    if "golden_ann" not in new_databases:
        new_databases["golden_ann"] = expected_ann
    new_doc["databases"] = new_databases
    return new_doc, True


# ---------------------------------------------------------------------------
# DB walker
# ---------------------------------------------------------------------------

def backfill(db: Any, dry_run: bool) -> Dict[str, int]:
    """Walk every doc in the experiments DB and apply ``backfill_one_doc``.

    Returns ``{'updated': N, 'unchanged': M, 'skipped': K}``:
      - ``updated`` — wrote (or in dry-run, would write) at least one
        new key.
      - ``unchanged`` — already has both keys; nothing to do.
      - ``skipped`` — design doc, or experiment _id not in the map.
    """
    counts = {"updated": 0, "unchanged": 0, "skipped": 0}
    for doc_id in db:
        if doc_id.startswith("_"):
            counts["skipped"] += 1
            continue
        if doc_id not in _EXPERIMENT_GOLDEN_MAP:
            logging.warning(
                "experiment %r is not in the backfill map — skipped",
                doc_id,
            )
            counts["skipped"] += 1
            continue

        doc = db[doc_id]
        new_doc, changed = backfill_one_doc(doc)
        if not changed:
            counts["unchanged"] += 1
            continue
        counts["updated"] += 1
        if not dry_run:
            db.save(new_doc)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _connect_experiments_db() -> Any:
    """Open the skol_experiments database via the project's env config."""
    import couchdb as couchdb_lib  # local import — only needed at runtime

    config = get_env_config()
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server["skol_experiments"]


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
        help="Print what would change without writing to CouchDB.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Apply the backfill.  Run --dry-run first.",
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

    db = _connect_experiments_db()
    counts = backfill(db, dry_run=args.dry_run)

    print()
    print(f"Mode:      {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    print(f"updated:   {counts['updated']}")
    print(f"unchanged: {counts['unchanged']}")
    print(f"skipped:   {counts['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
