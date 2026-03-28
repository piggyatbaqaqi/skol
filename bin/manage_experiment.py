#!/usr/bin/env python3
"""Manage named experiments in the SKOL experiment registry.

Experiments tie together databases, Redis keys, and classifier models
for systematic comparison. Stored as documents in the skol_experiments
CouchDB database.

Subcommands:
    create    - Create a new experiment
    list      - List all experiments
    show      - Show experiment details
    update    - Update experiment fields
    archive   - Archive an experiment
    deploy    - Promote an experiment to production
    pipeline  - Show pipeline step status
    runnext   - Run the next pending pipeline step
    runstep   - Run one or more named pipeline steps
    resetstep - Reset one or more steps back to pending
    skipstep  - Mark one or more steps as skipped

Examples:
    # Create a new experiment
    python manage_experiment.py create --name jats_v1 \\
        --notes "JATS-derived training" \\
        --training-db skol_golden_ann_jats

    # List experiments
    python manage_experiment.py list

    # Show pipeline status
    python manage_experiment.py pipeline jats_v1

    # Run the next pending step
    python manage_experiment.py runnext jats_v1

    # Run specific step(s)
    python manage_experiment.py runstep jats_v1 embed
    python manage_experiment.py runstep jats_v1 evaluate,build_vocab

    # Skip steps that don't apply
    python manage_experiment.py skipstep jats_v1 annotate_jats

    # Reset a failed step to retry
    python manage_experiment.py resetstep jats_v1 extract_taxa

    # Deploy an experiment to production
    python manage_experiment.py deploy jats_v1
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _default_step(name: str) -> Dict[str, Any]:
    """Build a default pipeline step record."""
    return {
        "name": name,
        "status": "pending",
        "started_at": None,
        "completed_at": None,
    }


# Ordered list of pipeline step names.
_PIPELINE_STEPS = [
    "train",
    "predict",
    "annotate_jats",
    "extract_taxa",
    "embed",
    "evaluate",
    "build_vocab",
]

# Statuses that count as "done" for dependency purposes.
_DONE_STATUSES = ("completed", "skipped")

# Steps that must all be done before steps 5+ (evaluate, build_vocab) can run.
# Steps 0-4 are sequential among themselves (each requires the prior one done).
_SEQUENTIAL_COUNT = 5  # train, predict, annotate_jats, extract_taxa, embed


def _default_pipeline() -> Dict[str, Any]:
    """Build the default pipeline tracking structure."""
    return {
        "current_step": 0,
        "steps": [_default_step(name) for name in _PIPELINE_STEPS],
    }


def _default_experiment(name: str) -> Dict[str, Any]:
    """Build a default experiment document."""
    return {
        "_id": name,
        "model_name": "",
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
        "pipeline": _default_pipeline(),
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
    if args.model_name:
        doc["model_name"] = args.model_name
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
    if args.model_name is not None:
        doc["model_name"] = args.model_name
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
# Pipeline helpers
# ---------------------------------------------------------------------------

_BIN_DIR = Path(__file__).resolve().parent


def _ensure_pipeline(doc: Dict[str, Any]) -> None:
    """Lazily add/repair the pipeline field on pre-existing experiment docs."""
    if "pipeline" not in doc:
        doc["pipeline"] = _default_pipeline()
        return
    # Ensure all current step names are present (handles newly-added steps).
    existing: Dict[str, Dict[str, Any]] = {
        s["name"]: s for s in doc["pipeline"].get("steps", [])
    }
    doc["pipeline"]["steps"] = [
        existing.get(name, _default_step(name)) for name in _PIPELINE_STEPS
    ]


def _find_step(steps: List[Dict[str, Any]], name: str) -> Optional[int]:
    """Return the index of a step by name, or None."""
    for i, step in enumerate(steps):
        if step["name"] == name:
            return i
    return None


def _check_dependencies(
    steps: List[Dict[str, Any]], step_idx: int,
) -> Optional[str]:
    """Return an error message if dependencies for step_idx are not met."""
    if step_idx < _SEQUENTIAL_COUNT:
        # Sequential block: each step requires all prior steps to be done.
        for i in range(step_idx):
            prev = steps[i]
            if prev["status"] not in _DONE_STATUSES:
                return (
                    f"Step {i + 1} ({prev['name']}) must be completed or "
                    f"skipped first (current status: {prev['status']})"
                )
    else:
        # Steps after the sequential block require all sequential steps done.
        for i in range(_SEQUENTIAL_COUNT):
            prev = steps[i]
            if prev["status"] not in _DONE_STATUSES:
                return (
                    f"Step {i + 1} ({prev['name']}) must be completed or "
                    f"skipped first (current status: {prev['status']})"
                )
    return None


def _build_step_command(step_name: str, experiment_name: str) -> List[str]:
    """Build the subprocess command list for a pipeline step."""
    templates: Dict[str, List[str]] = {
        "train": [
            sys.executable, str(_BIN_DIR / "train_classifier.py"),
            "--experiment", "{name}",
            "--force",
        ],
        "predict": [
            sys.executable, str(_BIN_DIR / "predict_classifier.py"),
            "--experiment", "{name}",
            "--incremental", "--skip-existing",
        ],
        "annotate_jats": [
            sys.executable, str(_BIN_DIR / "jats_to_yedda.py"),
            "--experiment", "{name}",
            "--all", "--taxpub-only",
            "--output-to", "couchdb",
            "--skip-existing",
        ],
        "extract_taxa": [
            sys.executable, str(_BIN_DIR / "extract_taxa_to_couchdb.py"),
            "--experiment", "{name}",
            "--skip-existing",
        ],
        "embed": [
            sys.executable, str(_BIN_DIR / "embed_taxa.py"),
            "--experiment", "{name}",
            "--force",
        ],
        "evaluate": [
            sys.executable, str(_BIN_DIR / "evaluate_golden.py"),
            "--experiment", "{name}",
            "--golden-db", "skol_golden_ann_hand",
            "--plaintext-db", "skol_golden",
            "--save-to-experiment",
        ],
        "build_vocab": [
            sys.executable, str(_BIN_DIR / "build_vocab_tree.py"),
            "--experiment", "{name}",
        ],
    }
    template = templates[step_name]
    return [arg.format(name=experiment_name) for arg in template]


def _run_step(
    db: Any,
    doc: Dict[str, Any],
    step_idx: int,
    experiment_name: str,
    verbosity: int = 1,
) -> bool:
    """Run a pipeline step, updating CouchDB status before and after.

    Returns True on success, False on failure.
    """
    step = doc["pipeline"]["steps"][step_idx]
    step_name = step["name"]
    cmd = _build_step_command(step_name, experiment_name)

    if verbosity >= 1:
        print(f"\nRunning step {step_idx + 1}/{len(_PIPELINE_STEPS)}: {step_name}")
        if verbosity >= 2:
            print(f"  Command: {' '.join(cmd)}")

    # Mark step as running and persist.
    step["status"] = "running"
    step["started_at"] = _now_iso()
    step["completed_at"] = None
    doc["updated_at"] = _now_iso()
    db.save(doc)

    # Execute the step.
    try:
        result = subprocess.run(cmd, check=False)
        exit_code = result.returncode
    except Exception as exc:
        print(f"Error launching step: {exc}", file=sys.stderr)
        exit_code = 1

    # Reload doc to get latest _rev (subprocess may have updated it).
    doc_fresh = db[doc["_id"]]
    step_fresh = doc_fresh["pipeline"]["steps"][step_idx]

    if exit_code == 0:
        step_fresh["status"] = "completed"
        step_fresh["completed_at"] = _now_iso()
        if verbosity >= 1:
            print(f"✓ Step {step_idx + 1} ({step_name}) completed")
    else:
        step_fresh["status"] = "failed"
        step_fresh["completed_at"] = _now_iso()
        print(
            f"✗ Step {step_idx + 1} ({step_name}) failed "
            f"(exit code {exit_code})",
            file=sys.stderr,
        )

    doc_fresh["updated_at"] = _now_iso()
    db.save(doc_fresh)

    return exit_code == 0


# ---------------------------------------------------------------------------
# Pipeline subcommands
# ---------------------------------------------------------------------------

def cmd_pipeline(db: Any, args: Any) -> None:
    """Show pipeline step status with timestamps."""
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    _ensure_pipeline(doc)
    steps = doc["pipeline"]["steps"]

    fmt_hdr = "{:<4} {:<15} {:<10} {:<25} {}"
    fmt_row = "{:<4} {:<15} {:<10} {:<25} {}"

    print(f"\nPipeline: {args.name}")
    print("-" * 80)
    print(fmt_hdr.format("Step", "Name", "Status", "Started", "Completed"))
    print("-" * 80)

    now = datetime.now(timezone.utc)
    warnings: List[str] = []

    for i, step in enumerate(steps):
        started = step.get("started_at") or "-"
        completed = step.get("completed_at") or "-"

        # Truncate timestamps for display
        started_disp = started[:19] if started != "-" else "-"
        completed_disp = completed[:19] if completed != "-" else "-"

        status = step["status"]
        print(fmt_row.format(
            i + 1, step["name"], status, started_disp, completed_disp,
        ))

        if status == "running" and step.get("started_at"):
            try:
                started_dt = datetime.fromisoformat(step["started_at"])
                elapsed = (now - started_dt).total_seconds()
                if elapsed > 3600:
                    warnings.append(
                        f"  ⚠  Step {i + 1} ({step['name']}): running for "
                        f"{int(elapsed // 3600)}h — may be stalled"
                    )
            except Exception:
                pass
        elif status == "failed":
            warnings.append(
                f"  ⚠  Step {i + 1} ({step['name']}): failed — "
                f"run 'resetstep {args.name} {step['name']}' to retry"
            )

    if warnings:
        print()
        for w in warnings:
            print(w)
    print()


def cmd_runnext(db: Any, args: Any) -> None:
    """Run the next pending pipeline step."""
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    _ensure_pipeline(doc)
    steps = doc["pipeline"]["steps"]

    # Find the first pending step.
    step_idx = None
    for i, step in enumerate(steps):
        if step["status"] == "pending":
            step_idx = i
            break

    if step_idx is None:
        all_done = all(s["status"] in _DONE_STATUSES for s in steps)
        if all_done:
            print(f"All pipeline steps are done for '{args.name}'.")
        else:
            statuses = ", ".join(
                f"{s['name']}={s['status']}" for s in steps
                if s["status"] not in _DONE_STATUSES and s["status"] != "pending"
            )
            print(
                f"No pending steps found for '{args.name}'. "
                f"Non-done steps: {statuses or 'none'}",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    # Check dependencies.
    err = _check_dependencies(steps, step_idx)
    if err:
        print(f"Cannot run step {step_idx + 1} ({steps[step_idx]['name']}): {err}",
              file=sys.stderr)
        sys.exit(1)

    success = _run_step(db, doc, step_idx, args.name)
    if not success:
        sys.exit(1)


def cmd_runstep(db: Any, args: Any) -> None:
    """Run one or more named pipeline steps (no dependency checking)."""
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    _ensure_pipeline(doc)
    steps = doc["pipeline"]["steps"]

    step_names = [s.strip() for s in args.steps.split(",") if s.strip()]
    for step_name in step_names:
        step_idx = _find_step(steps, step_name)
        if step_idx is None:
            print(
                f"Error: unknown step '{step_name}'. "
                f"Valid steps: {', '.join(_PIPELINE_STEPS)}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Reload doc before each step to get latest _rev.
        doc = db[args.name]
        _ensure_pipeline(doc)
        steps = doc["pipeline"]["steps"]

        success = _run_step(db, doc, step_idx, args.name)
        if not success:
            sys.exit(1)


def cmd_resetstep(db: Any, args: Any) -> None:
    """Reset one or more steps back to pending."""
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    _ensure_pipeline(doc)
    steps = doc["pipeline"]["steps"]

    step_names = [s.strip() for s in args.steps.split(",") if s.strip()]
    for step_name in step_names:
        step_idx = _find_step(steps, step_name)
        if step_idx is None:
            print(
                f"Error: unknown step '{step_name}'. "
                f"Valid steps: {', '.join(_PIPELINE_STEPS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        steps[step_idx]["status"] = "pending"
        steps[step_idx]["started_at"] = None
        steps[step_idx]["completed_at"] = None
        print(f"Reset step '{step_name}' to pending")

    doc["updated_at"] = _now_iso()
    db.save(doc)


def cmd_skipstep(db: Any, args: Any) -> None:
    """Mark one or more steps as skipped."""
    doc = _get_experiment(db, args.name)
    if doc is None:
        print(f"Error: experiment '{args.name}' not found.", file=sys.stderr)
        sys.exit(1)

    _ensure_pipeline(doc)
    steps = doc["pipeline"]["steps"]

    step_names = [s.strip() for s in args.steps.split(",") if s.strip()]
    for step_name in step_names:
        step_idx = _find_step(steps, step_name)
        if step_idx is None:
            print(
                f"Error: unknown step '{step_name}'. "
                f"Valid steps: {', '.join(_PIPELINE_STEPS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        steps[step_idx]["status"] = "skipped"
        steps[step_idx]["completed_at"] = _now_iso()
        print(f"Marked step '{step_name}' as skipped")

    doc["updated_at"] = _now_iso()
    db.save(doc)


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
        "--model-name", type=str,
        help="Model config name (e.g. logistic_sections_taxpub_v1)",
    )
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
    p_update.add_argument(
        "--model-name", type=str,
        help="Model config name (e.g. logistic_sections_taxpub_v1)",
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

    # pipeline
    p_pipeline = subparsers.add_parser(
        "pipeline", help="Show pipeline step status",
    )
    p_pipeline.add_argument("name", help="Experiment name")

    # runnext
    p_runnext = subparsers.add_parser(
        "runnext", help="Run the next pending pipeline step",
    )
    p_runnext.add_argument("name", help="Experiment name")

    # runstep
    p_runstep = subparsers.add_parser(
        "runstep", help="Run one or more named pipeline steps",
    )
    p_runstep.add_argument("name", help="Experiment name")
    p_runstep.add_argument(
        "steps",
        help="Comma-separated step name(s) to run (e.g. embed or evaluate,build_vocab)",
    )

    # resetstep
    p_resetstep = subparsers.add_parser(
        "resetstep", help="Reset step(s) back to pending",
    )
    p_resetstep.add_argument("name", help="Experiment name")
    p_resetstep.add_argument(
        "steps",
        help="Comma-separated step name(s) to reset",
    )

    # skipstep
    p_skipstep = subparsers.add_parser(
        "skipstep", help="Mark step(s) as skipped",
    )
    p_skipstep.add_argument("name", help="Experiment name")
    p_skipstep.add_argument(
        "steps",
        help="Comma-separated step name(s) to skip",
    )

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
    elif args.command == "pipeline":
        cmd_pipeline(db, args)
    elif args.command == "runnext":
        cmd_runnext(db, args)
    elif args.command == "runstep":
        cmd_runstep(db, args)
    elif args.command == "resetstep":
        cmd_resetstep(db, args)
    elif args.command == "skipstep":
        cmd_skipstep(db, args)


if __name__ == "__main__":
    main()
