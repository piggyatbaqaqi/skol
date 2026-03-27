#!/usr/bin/env python3
"""Manage named experiments in the SKOL experiment registry.

Experiments tie together databases, Redis keys, and classifier models
for systematic comparison. Stored as documents in the skol_experiments
CouchDB database.

Subcommands:
    create   - Create a new experiment
    list     - List all experiments
    show     - Show experiment details
    update   - Update experiment fields
    archive  - Archive an experiment
    deploy   - Promote an experiment to production

Examples:
    # Create a new experiment
    python manage_experiment.py create --name jats_v1 \\
        --notes "JATS-derived training" \\
        --training-db skol_golden_ann_jats

    # List experiments
    python manage_experiment.py list

    # Show details
    python manage_experiment.py show production

    # Deploy an experiment to production
    python manage_experiment.py deploy jats_v1
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Allow running as a script or as a module.
if __name__ == "__main__" and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    bin_dir = str(Path(__file__).resolve().parent)
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)

from env_config import get_env_config


# ---------------------------------------------------------------------------
# Experiment schema defaults
# ---------------------------------------------------------------------------

_STATUS_VALUES = ("draft", "testing", "evaluated", "deployed", "archived")


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _default_experiment(name: str) -> Dict[str, Any]:
    """Build a default experiment document."""
    return {
        "_id": name,
        "notes": "",
        "status": "draft",
        "databases": {
            "ingest": "skol_dev",
            "training": "skol_training",
            "annotations": f"skol_exp_{name}_ann",
            "taxa": f"skol_exp_{name}_taxa",
            "taxa_full": f"skol_exp_{name}_taxa_full",
        },
        "redis_keys": {
            "classifier_model": f"skol:classifier:model:{name}",
            "embedding": f"skol:embedding:{name}",
            "menus": f"skol:ui:menus_{name}",
        },
        "evaluation": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _production_experiment() -> Dict[str, Any]:
    """Build the default production experiment document."""
    doc = _default_experiment("production")
    doc["notes"] = (
        "Current production pipeline: logistic regression on "
        "hand-annotated training data"
    )
    doc["status"] = "deployed"
    doc["databases"] = {
        "ingest": "skol_dev",
        "training": "skol_training",
        "taxa": "skol_taxa_dev",
        "taxa_full": "skol_taxa_full_dev",
    }
    doc["redis_keys"] = {
        "classifier_model": "skol:classifier:model:logistic_sections_v2.0",
        "embedding": "skol:embedding:v1.1",
        "menus": "skol:ui:menus_latest",
    }
    return doc


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _connect_experiments_db(config: Dict[str, Any]):
    """Connect to the experiments database, creating it if needed."""
    import couchdb as couchdb_lib

    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )

    db_name = config.get("experiments_database", "skol_experiments")
    if db_name not in server:
        db = server.create(db_name)
        # Seed with production experiment
        prod = _production_experiment()
        db.save(prod)
    else:
        db = server[db_name]

    return db


def _get_experiment(db, name: str) -> Optional[Dict[str, Any]]:
    """Get an experiment document by name, or None."""
    try:
        return db[name]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_create(db, args) -> None:
    """Create a new experiment."""
    name = args.name

    if _get_experiment(db, name):
        print(f"Error: experiment '{name}' already exists.", file=sys.stderr)
        sys.exit(1)

    doc = _default_experiment(name)
    if args.notes:
        doc["notes"] = args.notes
    if args.training_db:
        doc["databases"]["training"] = args.training_db
    if args.ingest_db:
        doc["databases"]["ingest"] = args.ingest_db
    if args.annotations_db:
        doc["databases"]["annotations"] = args.annotations_db

    db.save(doc)
    print(f"Created experiment '{name}'")


def cmd_list(db, args) -> None:
    """List all experiments."""
    rows = []
    for row in db.view("_all_docs", include_docs=True):
        if row.id.startswith("_design/"):
            continue
        doc = row.doc
        rows.append({
            "name": doc["_id"],
            "status": doc.get("status", "?"),
            "training": doc.get("databases", {}).get("training", "?"),
            "notes": (doc.get("notes", "") or "")[:60],
        })

    if not rows:
        print("No experiments found.")
        return

    # Print table
    fmt = "{:<25} {:<12} {:<25} {}"
    print(fmt.format("NAME", "STATUS", "TRAINING DB", "NOTES"))
    print("-" * 90)
    for r in rows:
        print(fmt.format(r["name"], r["status"], r["training"], r["notes"]))


def cmd_show(db, args) -> None:
    """Show experiment details."""
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    # Remove CouchDB internals for display
    display = {k: v for k, v in doc.items() if not k.startswith("_rev")}
    display["_id"] = doc["_id"]
    print(json.dumps(display, indent=2, default=str))


def cmd_update(db, args) -> None:
    """Update experiment fields."""
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    changed = False
    if args.notes is not None:
        doc["notes"] = args.notes
        changed = True
    if args.status:
        if args.status not in _STATUS_VALUES:
            print(
                f"Error: invalid status '{args.status}'. "
                f"Must be one of: {', '.join(_STATUS_VALUES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        doc["status"] = args.status
        changed = True
    if args.training_db:
        doc.setdefault("databases", {})["training"] = args.training_db
        changed = True
    if args.ingest_db:
        doc.setdefault("databases", {})["ingest"] = args.ingest_db
        changed = True
    if args.annotations_db:
        doc.setdefault("databases", {})["annotations"] = args.annotations_db
        changed = True

    if changed:
        doc["updated_at"] = _now_iso()
        db.save(doc)
        print(f"Updated experiment '{args.name}'")
    else:
        print("Nothing to update.")


def cmd_archive(db, args) -> None:
    """Archive an experiment."""
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    if doc["_id"] == "production":
        print("Error: cannot archive the production experiment.",
              file=sys.stderr)
        sys.exit(1)

    doc["status"] = "archived"
    doc["updated_at"] = _now_iso()
    db.save(doc)
    print(f"Archived experiment '{args.name}'")


def cmd_deploy(db, args) -> None:
    """Promote an experiment to production.

    Copies the experiment's databases and redis_keys to the 'production'
    experiment record.
    """
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    if doc["_id"] == "production":
        print("Error: 'production' is already the production experiment.",
              file=sys.stderr)
        sys.exit(1)

    # Get or create production record
    prod = _get_experiment(db, "production")
    if prod is None:
        prod = _production_experiment()

    # Copy databases and redis_keys from the deployed experiment
    prod["databases"] = dict(doc.get("databases", {}))
    prod["redis_keys"] = dict(doc.get("redis_keys", {}))
    prod["notes"] = (
        f"Deployed from '{args.name}': {doc.get('notes', '')}"
    )
    prod["status"] = "deployed"
    prod["updated_at"] = _now_iso()
    prod["deployed_from"] = args.name
    db.save(prod)

    # Mark the source experiment as deployed
    doc["status"] = "deployed"
    doc["updated_at"] = _now_iso()
    db.save(doc)

    print(f"Deployed '{args.name}' as production")
    print(f"  Databases: {json.dumps(prod['databases'])}")
    print(f"  Redis keys: {json.dumps(prod['redis_keys'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage SKOL experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = subparsers.add_parser("create", help="Create a new experiment")
    p_create.add_argument("--name", required=True, help="Experiment name")
    p_create.add_argument("--notes", type=str, help="Description/notes")
    p_create.add_argument(
        "--training-db", type=str,
        help="Training database name",
    )
    p_create.add_argument(
        "--ingest-db", type=str,
        help="Ingest database name",
    )
    p_create.add_argument(
        "--annotations-db", type=str,
        help="Annotations output database name",
    )

    # list
    subparsers.add_parser("list", help="List all experiments")

    # show
    p_show = subparsers.add_parser("show", help="Show experiment details")
    p_show.add_argument("name", help="Experiment name")

    # update
    p_update = subparsers.add_parser("update", help="Update experiment")
    p_update.add_argument("name", help="Experiment name")
    p_update.add_argument("--notes", type=str, help="Update notes")
    p_update.add_argument(
        "--status", type=str,
        help=f"Set status ({', '.join(_STATUS_VALUES)})",
    )
    p_update.add_argument("--training-db", type=str, help="Training DB")
    p_update.add_argument("--ingest-db", type=str, help="Ingest DB")
    p_update.add_argument(
        "--annotations-db", type=str, help="Annotations output DB",
    )

    # archive
    p_archive = subparsers.add_parser("archive", help="Archive experiment")
    p_archive.add_argument("name", help="Experiment name")

    # deploy
    p_deploy = subparsers.add_parser(
        "deploy", help="Promote experiment to production",
    )
    p_deploy.add_argument("name", help="Experiment name to deploy")

    args = parser.parse_args()
    config = get_env_config()
    db = _connect_experiments_db(config)

    if args.command == "create":
        cmd_create(db, args)
    elif args.command == "list":
        cmd_list(db, args)
    elif args.command == "show":
        cmd_show(db, args)
    elif args.command == "update":
        cmd_update(db, args)
    elif args.command == "archive":
        cmd_archive(db, args)
    elif args.command == "deploy":
        cmd_deploy(db, args)


if __name__ == "__main__":
    main()
