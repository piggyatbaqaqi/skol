#!/usr/bin/env python3
"""Union docs from N source training DBs into a single target DB.

Step 2.E of docs/production_v3_plan.md. Used to materialise the v3
combined corpus by merging the hand and JATS no-golden DBs. Raises
``ValueError`` on doc-ID collision across sources rather than silently
overwriting one corpus's annotations with the other's.

Example:
    python bin/build_combined_training_db.py \\
        --sources skol_training_v2_no_golden \\
                  skol_training_taxpub_v2_no_golden \\
        --output  skol_training_v3_combined_no_golden
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_no_golden_training_db import (  # noqa: E402
    _copy_doc_with_attachments,
)
from env_config import get_env_config  # noqa: E402


def build_combined_db(
    source_dbs: List[Any],
    target_db: Any,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Copy every doc from each ``source_dbs`` entry into ``target_db``.

    Raises ``ValueError`` if the same doc ID appears in more than one
    source DB — the v3 use case (hand ∪ JATS) is doc-ID disjoint by
    construction, and a collision means the operator passed the wrong
    corpora.

    Returns counts: ``{copied, skipped_exists}``.
    """
    # First pass: pre-check for collisions across sources, so we fail
    # before touching the target. Build a map id -> first-source-name.
    seen: Dict[str, str] = {}
    for db in source_dbs:
        for doc_id in db:
            if isinstance(doc_id, str) and doc_id.startswith("_"):
                continue
            if doc_id in seen:
                raise ValueError(
                    f"Doc-ID collision across source DBs: {doc_id!r} "
                    f"is present in both {seen[doc_id]!r} "
                    f"and {db.name!r}."
                )
            seen[doc_id] = db.name

    counts = {"copied": 0, "skipped_exists": 0}
    for db in source_dbs:
        for doc_id in db:
            if isinstance(doc_id, str) and doc_id.startswith("_"):
                continue
            if doc_id in target_db:
                counts["skipped_exists"] += 1
                continue
            if dry_run:
                counts["copied"] += 1
                continue
            _copy_doc_with_attachments(db, target_db, doc_id)
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
    parser.add_argument(
        "--sources", required=True, nargs="+",
        help="Two or more source DB names to union.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Target DB name (created if absent).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()
    if len(args.sources) < 2:
        parser.error("--sources needs at least two DB names")
    config = get_env_config()
    source_dbs = [_connect(config, n) for n in args.sources]
    target_db = _connect(config, args.output)
    sizes = ", ".join(
        f"{db.name}={len(db)}" for db in source_dbs
    )
    print(
        f"Sources: {sizes}\n"
        f"Target:  {args.output} ({len(target_db)} docs before)",
        file=sys.stderr,
    )
    counts = build_combined_db(
        source_dbs, target_db, dry_run=args.dry_run,
    )
    print(
        f"copied={counts['copied']} "
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
