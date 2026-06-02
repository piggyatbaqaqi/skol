#!/usr/bin/env python3
"""Backfill ``doi`` and ``xml_url`` onto treatment docs' nested
``ingest`` map.

After ``_ESSENTIAL_INGEST_KEYS`` in ``treatment.py`` was extended
to propagate ``doi`` and ``xml_url`` from the parent ingest doc,
newly-extracted treatments carry both fields.  This one-shot
script populates them on the pre-change treatments by walking the
treatment DB(s), reading each treatment's ``ingest._id``, looking
up that ingest doc in skol_dev, and copying the missing fields.

Idempotent — re-running is a no-op once fields are populated.
``--treatments-db`` (repeatable) selects which DB(s) to walk;
defaults to the env_config ``treatments_db_name``.  ``--dry-run``
/ ``--limit`` / ``--verbosity`` supported.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# Fields this backfill propagates from parent ingest → treatment.
# Mirrors a subset of treatment._ESSENTIAL_INGEST_KEYS — only the
# fields that may be missing on existing treatments because they
# weren't in the keyset at extract time.
_BACKFILL_FIELDS = ('doi', 'xml_url')


# ---------------------------------------------------------------------------
# Pure helpers (covered by backfill_treatment_ingest_fields_test.py)
# ---------------------------------------------------------------------------


def parent_doc_id(treatment: Dict[str, Any]) -> Optional[str]:
    """Pull the parent ingest doc's ``_id`` out of a treatment doc's
    nested ``ingest`` map.  Returns ``None`` when the field is
    missing or malformed (defensive against bad data shapes)."""
    ingest = treatment.get('ingest')
    if not isinstance(ingest, dict):
        return None
    doc_id = ingest.get('_id')
    if not isinstance(doc_id, str) or not doc_id:
        return None
    return doc_id


def compute_ingest_update(
    treatment: Dict[str, Any],
    parent: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the fields to write into the treatment's ``ingest``
    map.  For each backfill field: include when the treatment's
    current value is missing / empty AND the parent has a non-empty
    value.  Empty dict means no-op.
    """
    ingest = treatment.get('ingest')
    if not isinstance(ingest, dict):
        return {}
    update: Dict[str, Any] = {}
    for field in _BACKFILL_FIELDS:
        existing = ingest.get(field)
        if isinstance(existing, str) and existing.strip():
            continue  # already set
        parent_value = parent.get(field)
        if not isinstance(parent_value, str) or not parent_value.strip():
            continue  # parent doesn't have it either
        update[field] = parent_value
    return update


# ---------------------------------------------------------------------------
# CouchDB iteration / write (network-touching)
# ---------------------------------------------------------------------------


def _iter_treatments(
    db: Any, verbosity: int = 1,
) -> Iterable[Dict[str, Any]]:
    """Yield every non-design doc from ``db``."""
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        try:
            yield db[doc_id]
        except Exception as exc:
            if verbosity >= 1:
                print(f'  warning: could not load {doc_id}: {exc}',
                      file=sys.stderr)


def backfill_db(
    treatments_db: Any,
    ingest_db: Any,
    parent_cache: Dict[str, Optional[Dict[str, Any]]],
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk one treatment DB; for each doc, look up the parent via
    ``parent_doc_id``, compute the update, persist when not dry-run.

    Uses ``parent_cache`` to avoid re-reading the same ingest doc
    when multiple treatments share a parent — important since a
    typical ingest doc fans out to dozens of treatments.

    Returns ``{scanned, eligible, written, parent_missing}``.
    """
    counts = {
        'scanned': 0, 'eligible': 0, 'written': 0, 'parent_missing': 0,
    }
    for treatment in _iter_treatments(treatments_db, verbosity=verbosity):
        counts['scanned'] += 1
        if verbosity >= 2 and counts['scanned'] % 1000 == 0:
            print(f'  scanned {counts["scanned"]} treatments...')
        pid = parent_doc_id(treatment)
        if pid is None:
            continue
        if pid not in parent_cache:
            try:
                parent_cache[pid] = (
                    ingest_db[pid] if pid in ingest_db else None
                )
            except Exception:
                parent_cache[pid] = None
        parent = parent_cache[pid]
        if parent is None:
            counts['parent_missing'] += 1
            continue
        update = compute_ingest_update(treatment, parent)
        if not update:
            continue
        counts['eligible'] += 1
        if limit is not None and counts['written'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        if verbosity >= 1:
            tag = '(DRY RUN) ' if dry_run else ''
            preview = ', '.join(f'{k}={v[:40]!r}'
                                for k, v in update.items())
            print(f'  {tag}{treatment["_id"]}: + {preview}')
        if not dry_run:
            treatment['ingest'].update(update)
            treatments_db.save(treatment)
        counts['written'] += 1
    return counts


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Backfill doi / xml_url onto treatment docs\' '
                    'nested ingest map from the parent ingest doc.',
    )
    parser.add_argument(
        '--treatments-db', type=str, action='append', default=None,
        help='Treatment database to walk (repeat for multiple).  '
             'Defaults to env_config treatments_db_name.',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview updates without writing.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after N successful updates per DB.')
    parser.add_argument('--verbosity', type=int, default=None,
                        help='0=quiet, 1=normal, 2=verbose.')
    args, _unknown = parser.parse_known_args()

    from env_config import get_env_config
    config = get_env_config()
    verbosity = (args.verbosity if args.verbosity is not None
                 else config.get('verbosity', 1))

    import couchdb
    server = couchdb.Server(config['couchdb_url'])
    src_user = config.get('couchdb_username') or ''
    src_pass = config.get('couchdb_password') or ''
    if src_user and src_pass:
        server.resource.credentials = (src_user, src_pass)

    ingest_db_name = (config.get('ingest_db_name')
                      or config.get('couchdb_database')
                      or 'skol_dev')
    if ingest_db_name not in server:
        print(f'error: ingest DB {ingest_db_name!r} not found',
              file=sys.stderr)
        return 2
    ingest_db = server[ingest_db_name]

    treatments_db_names: List[str] = (
        args.treatments_db
        or [config.get('treatments_db_name') or 'skol_treatments_dev']
    )

    # One shared parent_cache across all DBs — many treatments share
    # parent ingest docs.
    parent_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    for tname in treatments_db_names:
        if tname not in server:
            print(f'  skip {tname}: not present on server',
                  file=sys.stderr)
            continue
        if verbosity >= 1:
            print(f'\n=== {tname} {"(DRY RUN)" if args.dry_run else ""} ===')
        counts = backfill_db(
            treatments_db=server[tname],
            ingest_db=ingest_db,
            parent_cache=parent_cache,
            dry_run=args.dry_run,
            limit=args.limit,
            verbosity=verbosity,
        )
        print(f'  Scanned:        {counts["scanned"]}')
        print(f'  Eligible:       {counts["eligible"]}')
        print(f'  {"Would write" if args.dry_run else "Written"}:'
              f'    {counts["written"]}')
        print(f'  Parent missing: {counts["parent_missing"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
