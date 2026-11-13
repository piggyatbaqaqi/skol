#!/usr/bin/env python3
"""One-time fix: make every experiment record name its menus tree key.

``redis_keys.menus`` names the Redis key holding the vocabulary tree
(docs/plans/experiment-record-canonical.md, Step 1).  Older records
name a *pointer* key instead (``skol:ui:menus_latest``, or the
2026-11-13 stopgap ``skol:ui:menus_production_v4_1_pointer``) whose
value is the tree key.  This rewrites each such record to the
pointer's target, then deletes the pointer keys no record names.

Usage::

    fixes/fix_menus_pointer_records.py --dry-run
    fixes/fix_menus_pointer_records.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

LATEST = 'skol:ui:menus_latest'


@dataclass(frozen=True)
class MenusFix:
    """One record whose ``redis_keys.menus`` names a pointer."""
    doc_id: str
    old: str
    new: str


def is_pointer(value: Optional[str]) -> bool:
    """True when a stored menus value is a key name, not tree JSON."""
    if not value:
        return False
    try:
        json.loads(value)
    except ValueError:
        return True
    return False


def _menus(record: Dict[str, Any]) -> Optional[str]:
    key = (record.get('redis_keys') or {}).get('menus')
    return key if isinstance(key, str) and key else None


def _holds_tree(r: Any, key: str) -> bool:
    try:
        data = json.loads(r.get(key) or '')
    except ValueError:
        return False
    return isinstance(data, dict) and 'tree' in data


def plan_menus_fix(records: Iterable[Dict[str, Any]], r: Any) -> List[MenusFix]:
    """The records whose menus key holds a pointer, with its target.

    Only a pointer whose target holds a tree is rewritten.  Records
    with no menus key, a key holding a tree, a key holding nothing, or
    a pointer to a missing tree are left alone.
    """
    fixes = []
    for record in records:
        key = _menus(record)
        if key is None:
            continue
        target = r.get(key)
        if is_pointer(target) and _holds_tree(r, target):
            fixes.append(MenusFix(record['_id'], key, target))
    return fixes


def pointer_keys_to_delete(r: Any, fixes: Sequence[MenusFix],
                           records: Iterable[Dict[str, Any]]) -> List[str]:
    """Pointer keys safe to delete once ``fixes`` are applied: each
    fixed record's old key, plus ``skol:ui:menus_latest`` if it is a
    pointer, excluding any key a record still names afterwards."""
    fixed = {f.doc_id: f.new for f in fixes}
    still_named = set()
    for record in records:
        key = fixed.get(record['_id'], _menus(record))
        if key is not None:
            still_named.add(key)
    candidates = {f.old for f in fixes}
    if is_pointer(r.get(LATEST)):
        candidates.add(LATEST)
    return sorted(candidates - still_named)


def apply_menus_fix(db: Any, fixes: Sequence[MenusFix]) -> int:
    """Rewrite each record's ``redis_keys.menus``; return the count."""
    for fix in fixes:
        doc = db[fix.doc_id]
        doc.setdefault('redis_keys', {})['menus'] = fix.new
        db.save(doc)
    return len(fixes)


def delete_keys(r: Any, keys: Sequence[str]) -> int:
    """Delete ``keys`` one at a time; return how many existed.

    tsqali is a Redis Cluster: a multi-key DELETE whose keys hash to
    different slots fails with CROSSSLOT, so each key is its own call.
    """
    return sum(int(r.delete(key)) for key in keys)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    import couchdb

    from env_config import (  # noqa: E402
        common_parser, create_redis_client, get_env_config,
    )

    parser = argparse.ArgumentParser(
        description='Rewrite experiment records whose redis_keys.menus '
                    'names a pointer key; delete pointers no record names.',
        parents=[common_parser()],
    )
    args = parser.parse_args(argv)
    config = get_env_config(cli_args=args)
    dry_run = bool(config.get('dry_run'))

    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (config.get('couchdb_username') or '',
                                   config.get('couchdb_password') or '')
    db = server[config.get('experiments_database') or 'skol_experiments']
    records = [row.doc for row in db.view('_all_docs', include_docs=True)
               if not row.id.startswith('_design/')]

    r = create_redis_client(
        host=config.get('redis_host'), port=config.get('redis_port'),
        username=config.get('redis_username'),
        password=config.get('redis_password'), decode_responses=True)

    fixes = plan_menus_fix(records, r)
    doomed = pointer_keys_to_delete(r, fixes, records)
    tag = '(DRY RUN) ' if dry_run else ''
    for fix in fixes:
        print(f'{tag}{fix.doc_id}: redis_keys.menus {fix.old} -> {fix.new}')
    for key in doomed:
        print(f'{tag}delete pointer key {key} (-> {r.get(key)})')
    if not fixes and not doomed:
        print('Nothing to fix.')
    if dry_run:
        return 0
    apply_menus_fix(db, fixes)
    delete_keys(r, doomed)
    print(f'Rewrote {len(fixes)} record(s); deleted {len(doomed)} key(s).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
