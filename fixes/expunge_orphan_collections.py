#!/usr/bin/env python3
"""One-time fix: delete orphaned test collections from the local CouchDB.

Until 2026-11-17 the Django tests synced every Collection they saved
into the developer's local CouchDB (docs/django-test-isolation.md;
Trello 9ZkM5Wuq).  Test databases are destroyed after each run, so
those documents name collections that exist nowhere.

A document in the collections database, or a record in its history
database, is an **orphan** when its collection_id is not in the live
Django database (default ``django/db.sqlite3``, the dev server's).
Owner names cannot decide it: 'niemand' and 'jsmith' are both fixture
users and real dev accounts.  ``_design/`` documents are never touched.

Dry run by default: prints what would go, by database and by owner.
``--apply`` deletes in batches.

Usage::

    fixes/expunge_orphan_collections.py               # dry run
    fixes/expunge_orphan_collections.py --apply
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

DEFAULT_DJANGO_DB = str(
    Path(__file__).resolve().parent.parent / 'django' / 'db.sqlite3')
DEFAULT_COLLECTIONS_DB = 'skol_collections_dev'
DEFAULT_HISTORY_DB = 'skol_collections_history_dev'

_COLLECTION_ID = re.compile(r'^collection_(\d+)(?:_|$)')


class RefusedError(Exception):
    """The live-collection list cannot be trusted; nothing is deleted."""


@dataclass
class Plan:
    """What a run would delete: per database, (doc id, rev) pairs."""
    orphans: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    kept: Dict[str, int] = field(default_factory=dict)
    by_owner: Counter[str] = field(default_factory=Counter)


def load_live_ids(django_db: str) -> Set[int]:
    """The collection_ids in a Django database, opened read-only.

    Refuses (``RefusedError``) when the database is missing, has no
    ``search_collection`` table, or has no collections: an empty list
    would make every document look orphaned.
    """
    if not Path(django_db).is_file():
        raise RefusedError(f'no Django database at {django_db}')
    try:
        con = sqlite3.connect(f'file:{django_db}?mode=ro', uri=True)
        try:
            ids = {int(row[0]) for row in con.execute(
                'select collection_id from search_collection')}
        finally:
            con.close()
    except sqlite3.Error as e:
        raise RefusedError(f'cannot read collections from {django_db}: {e}')
    if not ids:
        raise RefusedError(
            f'{django_db} has no collections; refusing to treat every '
            f'CouchDB document as an orphan')
    return ids


def collection_id_of(doc_id: str) -> Optional[int]:
    """The collection_id in a collection or history document id, else None.

    ``collection_<id>`` and ``collection_<id>_<timestamp>``.
    """
    match = _COLLECTION_ID.match(doc_id)
    return int(match.group(1)) if match else None


def _owner(doc: Dict[str, Any]) -> str:
    owner = doc.get('owner')
    if isinstance(owner, dict):
        owner = owner.get('username')
    return str(owner) if owner else '(no owner)'


def make_plan(dbs: Dict[str, Any], live_ids: Set[int]) -> Plan:
    """Orphans in each database: documents whose collection_id is not live.

    Owners are counted from the collection documents (history records
    carry no owner).
    """
    plan = Plan()
    for name, db in dbs.items():
        orphans: List[Tuple[str, str]] = []
        kept = 0
        for row in db.view('_all_docs', include_docs=True):
            cid = collection_id_of(row.id)
            if cid is None:
                continue
            if cid in live_ids:
                kept += 1
                continue
            orphans.append((row.id, row.value['rev']))
            if row.id == f'collection_{cid}':
                plan.by_owner[_owner(row.doc or {})] += 1
        plan.orphans[name] = orphans
        plan.kept[name] = kept
    return plan


def apply_plan(plan: Plan, dbs: Dict[str, Any], batch: int = 500) -> int:
    """Delete the planned orphans with bulk updates; returns how many."""
    deleted = 0
    for name, orphans in plan.orphans.items():
        db = dbs[name]
        for start in range(0, len(orphans), batch):
            chunk = orphans[start:start + batch]
            results = db.update([
                {'_id': doc_id, '_rev': rev, '_deleted': True}
                for doc_id, rev in chunk])
            deleted += sum(1 for ok, _id, _rev in results if ok)
    return deleted


def report(plan: Plan) -> str:
    """Counts per database and per owner of the orphaned collections."""
    lines = [f'{name}: {len(plan.orphans[name])} orphaned, '
             f'{plan.kept[name]} kept'
             for name in plan.orphans]
    lines.append('Orphaned collections by owner:')
    lines.extend(f'  {owner}: {count}'
                 for owner, count in plan.by_owner.most_common())
    return '\n'.join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Strict CLI: script flags plus env_config's ``common_parser()``."""
    from env_config import common_parser

    parser = argparse.ArgumentParser(
        description='Delete orphaned test collections (and their history) '
                    'from the local CouchDB.  Dry run unless --apply.',
        parents=[common_parser()],
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Delete the orphans (default: report only)')
    parser.add_argument(
        '--django-db', default=DEFAULT_DJANGO_DB,
        help='Django SQLite database holding the live collections '
             f'(default: {DEFAULT_DJANGO_DB})')
    parser.add_argument(
        '--collections-db', default=DEFAULT_COLLECTIONS_DB,
        help=f'CouchDB collections database (default: {DEFAULT_COLLECTIONS_DB})')
    parser.add_argument(
        '--history-db', default=DEFAULT_HISTORY_DB,
        help=f'CouchDB history database (default: {DEFAULT_HISTORY_DB})')
    return parser


def main() -> int:
    """Report the orphans; delete them with --apply."""
    args = build_parser().parse_args()
    import couchdb
    from env_config import get_env_config

    config = get_env_config(cli_args=args)
    try:
        live_ids = load_live_ids(args.django_db)
    except RefusedError as e:
        print(f'Refusing: {e}', file=sys.stderr)
        return 2

    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (config['couchdb_username'],
                                   config['couchdb_password'])
    dbs = {name: server[name]
           for name in (args.collections_db, args.history_db)
           if name in server}
    print(f'Live collections in {args.django_db}: {len(live_ids)}')
    plan = make_plan(dbs, live_ids)
    print(report(plan))
    if not args.apply:
        print('Dry run: nothing deleted.  Re-run with --apply to delete.')
        return 0
    deleted = apply_plan(plan, dbs)
    print(f'Deleted {deleted} documents.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
