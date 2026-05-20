#!/usr/bin/env python3
"""Copy a training DB minus any docs that appear in a golden DB.

Step 2.A of docs/production_v3_plan.md. Produces a contamination-free
training corpus by skipping any doc whose ID also appears in the
supplied ``--golden-ann`` DB. Idempotent: re-running against an
already-populated target counts existing docs as ``skipped_exists``
and writes nothing.

Example:
    python bin/build_no_golden_training_db.py \\
        --source skol_training_v2 \\
        --golden-ann skol_golden_ann_hand_v2 \\
        --output skol_training_v2_no_golden
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Allow running as a script or as a module.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config  # noqa: E402


def _copy_doc_with_attachments(
    source_db: Any, target_db: Any, doc_id: str,
) -> None:
    """Copy one doc plus every attachment from source to target.

    Reads each attachment's ``content_type`` from the source doc's
    ``_attachments`` metadata so we never MIME-guess by filename.
    """
    source_doc = source_db[doc_id]
    # Strip CouchDB-managed fields before saving into the target —
    # we want a fresh doc with the same doc_id but a new _rev chain
    # in the target DB.
    target_doc: Dict[str, Any] = {
        k: v for k, v in source_doc.items()
        if not k.startswith("_") or k == "_id"
    }
    target_db.save(target_doc)
    # Re-fetch to pick up the _rev that save() assigned.
    target_doc = target_db[doc_id]
    attachments_meta = source_doc.get("_attachments") or {}
    for att_name, att_meta in attachments_meta.items():
        att = source_db.get_attachment(doc_id, att_name)
        if att is None:
            continue
        data = att.read()
        content_type = (
            att_meta.get("content_type")
            or "application/octet-stream"
        )
        target_db.put_attachment(
            target_doc, data,
            filename=att_name, content_type=content_type,
        )
        # Re-fetch so subsequent put_attachment calls see the latest
        # _rev (each attachment write bumps the rev).
        target_doc = target_db[doc_id]


def build_no_golden_db(
    source_db: Any,
    golden_ann_db: Any,
    target_db: Any,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Copy every doc from ``source_db`` to ``target_db`` whose ID is
    not present in ``golden_ann_db`` and not already in ``target_db``.

    Returns counts: ``{copied, skipped_golden, skipped_exists}``.
    """
    counts = {"copied": 0, "skipped_golden": 0, "skipped_exists": 0}
    for doc_id in source_db:
        if isinstance(doc_id, str) and doc_id.startswith("_"):
            continue
        if doc_id in golden_ann_db:
            counts["skipped_golden"] += 1
            continue
        if doc_id in target_db:
            counts["skipped_exists"] += 1
            continue
        if dry_run:
            counts["copied"] += 1
            continue
        _copy_doc_with_attachments(source_db, target_db, doc_id)
        counts["copied"] += 1
    return counts


def _connect(config: Dict[str, Any], name: str) -> Any:
    import couchdb
    server = couchdb.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"], config["couchdb_password"],
    )
    if name not in server:
        server.create(name)
    return server[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="Source training DB name.")
    parser.add_argument("--golden-ann", required=True,
                        help="Golden answer-key DB; its IDs are excluded.")
    parser.add_argument("--output", required=True,
                        help="Target DB name (created if absent).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print counts without writing.")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()
    config = get_env_config()
    source_db = _connect(config, args.source)
    golden_db = _connect(config, args.golden_ann)
    target_db = _connect(config, args.output)
    print(
        f"Source: {args.source} ({len(source_db)} docs)\n"
        f"Golden: {args.golden_ann} ({len(golden_db)} docs)\n"
        f"Target: {args.output} ({len(target_db)} docs before)",
        file=sys.stderr,
    )
    counts = build_no_golden_db(
        source_db, golden_db, target_db, dry_run=args.dry_run,
    )
    print(
        f"copied={counts['copied']} "
        f"skipped_golden={counts['skipped_golden']} "
        f"skipped_exists={counts['skipped_exists']}",
        file=sys.stderr,
    )
    if not args.dry_run:
        print(
            f"Target {args.output}: {len(target_db)} docs after",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
