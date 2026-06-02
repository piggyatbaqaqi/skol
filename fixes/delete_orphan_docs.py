#!/usr/bin/env python3
"""Delete skol_dev docs that have no identifying fields.

A doc qualifies as an "orphan" when ALL of these are missing /
empty: ``journal``, ``doi``, ``issn``, ``pdf_url``, ``pmcid``.
These are stubs left by partial or failed ingests — no path to
recovery, and the user has explicitly OK'd re-scraping with
current code.

Supports ``--dry-run`` / ``--limit`` / ``--verbosity``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))


# ---------------------------------------------------------------------------
# Pure helpers (covered by delete_orphan_docs_test.py)
# ---------------------------------------------------------------------------


_IDENTIFYING_FIELDS = ('journal', 'doi', 'issn', 'pdf_url', 'pmcid')


def is_orphan(doc: Dict[str, Any]) -> bool:
    """True when ``doc`` has no usable identifying field.

    Treats empty strings / whitespace-only strings / ``None`` /
    ``0`` as missing.  Any non-empty / truthy value on one of the
    identifying fields protects the doc.
    """
    for field in _IDENTIFYING_FIELDS:
        value = doc.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return False
        elif value:  # truthy non-string (int, etc.)
            return False
    return True


# ---------------------------------------------------------------------------
# CouchDB iteration / delete (network-touching)
# ---------------------------------------------------------------------------


def _iter_db(db: Any, verbosity: int = 1) -> Iterable[Dict[str, Any]]:
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


def delete_orphans(
    db: Any,
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk ``db``; delete every doc for which ``is_orphan`` is True.

    Returns ``{scanned, orphaned, deleted}``.
    """
    counts = {'scanned': 0, 'orphaned': 0, 'deleted': 0}
    for doc in _iter_db(db, verbosity=verbosity):
        counts['scanned'] += 1
        if verbosity >= 2 and counts['scanned'] % 1000 == 0:
            print(f'  scanned {counts["scanned"]} docs...')
        if not is_orphan(doc):
            continue
        counts['orphaned'] += 1
        if limit is not None and counts['deleted'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        if verbosity >= 1:
            tag = '(DRY RUN) ' if dry_run else ''
            has_attach = '_attachments' in doc
            attach_flag = ' [has attachments]' if has_attach else ''
            print(f'  {tag}delete {doc["_id"]}{attach_flag}')
        if not dry_run:
            db.delete(doc)
        counts['deleted'] += 1
    return counts


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Delete skol_dev docs that have no identifying '
                    'fields (journal / doi / issn / pdf_url / pmcid '
                    'all empty).',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview deletions without writing.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after N successful deletes.')
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
    db_name = (config.get('ingest_db_name')
               or config.get('couchdb_database')
               or 'skol_dev')
    if db_name not in server:
        print(f'error: database {db_name!r} not found at '
              f'{config["couchdb_url"]}', file=sys.stderr)
        return 2
    if verbosity >= 1:
        print(f'Target DB: {db_name} '
              f'{"(DRY RUN)" if args.dry_run else ""}')

    counts = delete_orphans(
        db=server[db_name],
        dry_run=args.dry_run,
        limit=args.limit,
        verbosity=verbosity,
    )

    print()
    print(f'Scanned:   {counts["scanned"]}')
    print(f'Orphaned:  {counts["orphaned"]}')
    print(f'{"Would delete" if args.dry_run else "Deleted"}: '
          f'{counts["deleted"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
