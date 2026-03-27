#!/usr/bin/env python3
"""Export document IDs from the golden dataset to stdout.

Prints one document ID per line from skol_golden, suitable for
use with ``jats_to_yedda.py --exclude-ids``.

Usage:
    python export_golden_ids.py > golden_ids.txt
    python export_golden_ids.py --database skol_golden
"""

import argparse
import sys
from pathlib import Path

# Add parent and bin directories to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import couchdb as couchdb_lib

from env_config import get_env_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export document IDs from the golden dataset."
    )
    parser.add_argument(
        "--database",
        type=str,
        default="skol_golden",
        help="Golden database name (default: skol_golden)",
    )
    args = parser.parse_args()

    config = get_env_config()
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )

    if args.database not in server:
        print(
            f"Error: database '{args.database}' not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    db = server[args.database]
    count = 0
    for row in db.view("_all_docs"):
        if not row.id.startswith("_design/"):
            print(row.id)
            count += 1

    print(f"Exported {count} IDs from {args.database}", file=sys.stderr)


if __name__ == "__main__":
    main()
