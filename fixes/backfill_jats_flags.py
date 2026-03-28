#!/usr/bin/env python3
"""
Backfill is_jats and is_taxpub boolean flags on existing ingest documents.

New ingestions set these flags automatically.  Run this script once against
the ingest database(s) to set the flags on documents ingested before the
flag was introduced.

Logic:
  - is_jats   = True when xml_format in ('jats', 'taxpub')
  - is_taxpub = True when xml_format == 'taxpub'

predict_classifier skips is_taxpub=True documents by default (use
--include-taxpub to override).  Plain JATS (PMC, is_jats=True but
is_taxpub=False) is never skipped: jats_to_yedda only produces
Misc-exposition for non-TaxPub JATS, so the ML classifier is still needed.

Existing documents that have neither xml_format nor xml_available are left
unchanged (is_jats and is_taxpub will be absent / treated as False).

Usage:
    python fixes/backfill_jats_flags.py --dry-run
    python fixes/backfill_jats_flags.py
    python fixes/backfill_jats_flags.py --database skol_dev
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config


def backfill(database: str, dry_run: bool = True, verbosity: int = 1) -> None:
    import couchdb

    config = get_env_config()
    couchdb_url = f"http://{config['couchdb_host']}"
    server = couchdb.Server(couchdb_url)
    if config['couchdb_username'] and config['couchdb_password']:
        server.resource.credentials = (
            config['couchdb_username'],
            config['couchdb_password'],
        )

    if database not in server:
        print(f"Database not found: {database}", file=sys.stderr)
        sys.exit(1)

    db = server[database]
    updated = 0
    skipped = 0

    for doc_id in db:
        try:
            doc = db[doc_id]
        except Exception as exc:
            if verbosity >= 2:
                print(f"  Warning: could not fetch {doc_id}: {exc}")
            continue

        xml_fmt = doc.get('xml_format')

        # Only process docs that already have xml_format set
        if not xml_fmt:
            skipped += 1
            continue

        is_jats = xml_fmt in ('jats', 'taxpub')
        is_taxpub = xml_fmt == 'taxpub'

        # Skip if flags already set correctly
        if doc.get('is_jats') == is_jats and doc.get('is_taxpub') == is_taxpub:
            skipped += 1
            continue

        updated += 1
        if verbosity >= 2:
            action = "Would set" if dry_run else "Setting"
            print(f"  {action} {doc_id}: is_jats={is_jats} is_taxpub={is_taxpub}")

        if not dry_run:
            try:
                doc['is_jats'] = is_jats
                doc['is_taxpub'] = is_taxpub
                db.save(doc)
            except Exception as exc:
                print(f"  ERROR saving {doc_id}: {exc}", file=sys.stderr)
                updated -= 1

    if verbosity >= 1:
        action = "Would update" if dry_run else "Updated"
        print(
            f"{'DRY RUN: ' if dry_run else ''}"
            f"{action} {updated} documents in '{database}' "
            f"({skipped} already correct or no xml_format)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill is_jats/is_taxpub flags on existing ingest documents."
    )
    parser.add_argument(
        '--database',
        default=None,
        help='CouchDB database (default: ingest_db_name from env config)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without writing',
    )
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    config = get_env_config()
    database = args.database or config.get('ingest_db_name', 'skol_dev')

    backfill(database=database, dry_run=args.dry_run, verbosity=args.verbosity)


if __name__ == '__main__':
    main()
