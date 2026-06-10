#!/usr/bin/env python3
"""Rename the ``production_v4`` experiment to ``production_v4_combined``.

Q5 of the 2026-06-09 DB-naming cleanup decision: fold the model-
corpus variant (``_combined``) into the experiment name so the
search-UI experiment selector shows ``production_v4_combined``
as the identity (instead of ``production_v4`` with a hidden
``_combined`` DB-suffix).

This is the highest-blast-radius single change in the rename
plan because the doc ``_id`` is the experiment name; CouchDB
has no in-place ``_id`` rename, so it's a copy + delete + DB
re-replicate + Redis-key rename combo.

Operations (in order, each idempotent):

1. Copy ``skol_experiments`` doc ``production_v4`` to
   ``production_v4_combined``.  Strip ``_id``/``_rev``, update
   ``databases.*`` fields so every per-experiment DB name has
   the new experiment slug.
2. For each per-experiment DB that still has ``_combined``
   anywhere in the name, replicate to the new name where
   ``_combined`` moves from the suffix to the experiment slot.
   E.g.::

     skol_exp_production_v4_01_00_ann_combined
       → skol_exp_production_v4_combined_01_00_ann

3. Rename Redis keys that derive from the experiment name:
   ``skol:embedding:production_v4`` →
   ``skol:embedding:production_v4_combined``, same for
   ``skol:classifier:model:*`` if pinned per-experiment and
   ``skol:ui:menus_*``.
4. Delete the old ``production_v4`` doc.

The OLD per-experiment DBs are NOT deleted by default —
``--delete-legacy`` opts in.  Defer that until you've verified
the search UI works against the new names.

Run with --dry-run first, ALWAYS.

Usage::

    python fixes/rename_experiment_q5.py --dry-run
    python fixes/rename_experiment_q5.py --execute

NOT EXECUTED IN THIS SESSION (2026-06-10) — written for a
follow-up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402


OLD_NAME = 'production_v4'
NEW_NAME = 'production_v4_combined'


def _rewrite_db_name(old_db: str) -> str:
    """Rewrite a per-experiment DB name from the ``production_v4``
    namespace to the ``production_v4_combined`` namespace.

    Pattern:
      skol_exp_production_v4_<stage>_<role>_combined
        → skol_exp_production_v4_combined_<stage>_<role>

    For DBs without `_combined` suffix (the eval siblings of
    non-combined variants), just rewrite the prefix:
      skol_exp_production_v4_<X>
        → skol_exp_production_v4_combined_<X>
    """
    if not old_db.startswith(f'skol_exp_{OLD_NAME}'):
        return old_db
    rest = old_db[len(f'skol_exp_{OLD_NAME}'):]
    # Strip trailing _combined (and _combined_eval — both forms).
    if rest.endswith('_combined'):
        rest = rest[:-len('_combined')]
    elif rest.endswith('_combined_eval'):
        rest = rest[:-len('_combined_eval')] + '_eval'
    return f'skol_exp_{NEW_NAME}{rest}'


def _rewrite_doc_databases(databases: Dict[str, str]) -> Dict[str, str]:
    """Apply _rewrite_db_name to every per-experiment value in the
    ``databases`` block.  Shared DBs (ingest, training, golden*)
    pass through unchanged."""
    new_db = {}
    for k, v in databases.items():
        if isinstance(v, str) and v.startswith(f'skol_exp_{OLD_NAME}'):
            new_db[k] = _rewrite_db_name(v)
        else:
            new_db[k] = v
    return new_db


def _redis_key_renames(redis_keys: Dict[str, str]) -> List[Tuple[str, str]]:
    """Identify Redis keys whose values reference the old
    experiment name and need to be renamed to the new one."""
    renames: List[Tuple[str, str]] = []
    for k, v in redis_keys.items():
        if isinstance(v, str) and OLD_NAME in v:
            new_v = v.replace(OLD_NAME, NEW_NAME)
            renames.append((v, new_v))
    return renames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--execute', action='store_false', dest='dry_run')
    parser.add_argument(
        '--delete-legacy', action='store_true',
        help='Delete the old production_v4 doc + DBs after migration.',
    )
    args = parser.parse_args()

    config = get_env_config()
    import couchdb  # type: ignore[import]
    import requests  # type: ignore[import]
    creds = (config['couchdb_username'], config['couchdb_password'])
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = creds

    # 1. Read the old experiment doc.
    exp_db = server['skol_experiments']
    try:
        old_doc = exp_db[OLD_NAME]
    except Exception:
        print(f'  ✗ {OLD_NAME} not found in skol_experiments', file=sys.stderr)
        return 1

    if NEW_NAME in exp_db:
        print(
            f'  {NEW_NAME} already exists — skipping doc copy.  '
            f'Re-running for DB-rename / Redis-key steps only.',
        )
        new_doc = exp_db[NEW_NAME]
    else:
        new_doc = dict(old_doc)
        new_doc.pop('_id', None)
        new_doc.pop('_rev', None)
        new_doc['_id'] = NEW_NAME
        new_doc['databases'] = _rewrite_doc_databases(
            new_doc.get('databases', {}),
        )
        # Redis keys: rename inline to point at the new namespace.
        rk = new_doc.get('redis_keys', {})
        new_rk = {
            k: (v.replace(OLD_NAME, NEW_NAME)
                if isinstance(v, str) and OLD_NAME in v else v)
            for k, v in rk.items()
        }
        new_doc['redis_keys'] = new_rk

        print(f'  Would create doc {NEW_NAME}:')
        for k in sorted(new_doc.get('databases', {})):
            print(f'    databases.{k}: {new_doc["databases"][k]}')
        for k in sorted(new_rk):
            print(f'    redis_keys.{k}: {new_rk[k]}')

        if not args.dry_run:
            exp_db.save(new_doc)
            print(f'  ✓ saved {NEW_NAME}')

    # 2. Per-experiment DB renames.
    all_dbs = [str(db) for db in server]
    db_renames = []
    for db in all_dbs:
        if not db.startswith(f'skol_exp_{OLD_NAME}'):
            continue
        new_db_name = _rewrite_db_name(db)
        if new_db_name != db and new_db_name not in all_dbs:
            db_renames.append((db, new_db_name))

    print(f'\n  {len(db_renames)} DB renames:')
    for old, new in db_renames:
        print(f'    {old}  →  {new}')

    if not args.dry_run:
        sess = requests.Session()
        for old, new in db_renames:
            body = {
                'source': {
                    'url': f'{config["couchdb_url"]}/{old}',
                    'auth': {'basic': {
                        'username': creds[0], 'password': creds[1],
                    }},
                },
                'target': {
                    'url': f'{config["couchdb_url"]}/{new}',
                    'auth': {'basic': {
                        'username': creds[0], 'password': creds[1],
                    }},
                },
                'create_target': True,
                'use_bulk_get': False,
            }
            resp = sess.post(
                f'{config["couchdb_url"]}/_replicate',
                json=body, auth=creds, timeout=600,
            )
            if resp.status_code in (200, 202):
                print(f'    ✓ {old} → {new}')
            else:
                print(f'    ✗ {old} → {new} HTTP {resp.status_code}')

    # 3. Redis-key renames — not implemented yet (would need a
    # ``rename`` operation on the Redis client; not all
    # consumers may handle it cleanly).  Operators should rename
    # via redis-cli RENAME or just let the new key materialise
    # on the next runnext that writes to it.
    print('\n  Redis key renames (manual):')
    for old, new in _redis_key_renames(old_doc.get('redis_keys', {})):
        print(f'    redis-cli RENAME "{old}" "{new}"')

    if args.delete_legacy and not args.dry_run:
        print('\n  Deleting legacy:')
        for old, _ in db_renames:
            try:
                del server[old]
                print(f'    deleted {old}')
            except Exception as exc:  # noqa: BLE001
                print(f'    failed: {old}: {exc}')
        try:
            exp_db.delete(old_doc)
            print(f'    deleted experiment doc {OLD_NAME}')
        except Exception as exc:  # noqa: BLE001
            print(f'    failed: experiment doc: {exc}')

    if args.dry_run:
        print('\nDry-run.  Re-run with --execute to apply.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
