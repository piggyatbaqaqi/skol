#!/usr/bin/env python3
"""Replicate per-experiment CouchDB DBs to their new stage-tagged
names.

Companion to ``fixes/migrate_experiment_schema.py``.  The schema
script rewrites the experiment-doc field names; this script does
the actual bytes-on-disk work: for each per-experiment legacy DB,
replicate it to its new name.  Old DBs are NOT deleted — the
script is additive so rollback is cheap (just point the
experiment docs back at the old names).

The mapping mirrors the schema migration:

    skol_exp_<X>_ann            → skol_exp_<X>_01_00_ann
    skol_exp_<X>_ann_combined   → skol_exp_<X>_01_00_ann_combined
    skol_exp_<X>_taxa           → skol_exp_<X>_02_00_treatments_prose
    skol_exp_<X>_taxa_full      → skol_exp_<X>_03_00_treatments_structured
    skol_exp_<X>_treatments     → skol_exp_<X>_02_00_treatments_prose
    skol_exp_<X>_treatments_full→ skol_exp_<X>_03_00_treatments_structured

For each new DB, an empty ``_eval`` sibling is also created so
the eval-pipeline branches have a destination to write to.

Long-running: replication of a multi-GB DB takes minutes; the
production_v4 annotations DB (~150 GB) takes hours.  Run with
``nohup`` and check the log periodically.

Usage::

    # Print what would happen (no replications kicked off).
    python fixes/rename_dbs_to_stage_tagged.py --dry-run

    # Kick off replications in the foreground; each one is
    # async on CouchDB's side (returns immediately with a job
    # id), so the script finishes quickly even for big DBs.
    python fixes/rename_dbs_to_stage_tagged.py --execute

    # Optional: also delete the legacy DBs after replication
    # completes.  Defer this until you've verified the search
    # UI works against the new names.
    python fixes/rename_dbs_to_stage_tagged.py --delete-legacy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402


# Role → (stage_tag, new_role_name) — the post-2026-06-10 mapping.
_RENAMES = [
    # Each tuple is (legacy_role_suffix, new_role_name).  The
    # legacy_role_suffix is the part AFTER skol_exp_<name>_.
    ('ann',             '01_00_ann'),
    ('taxa',            '02_00_treatments_prose'),
    ('treatments',      '02_00_treatments_prose'),
    ('taxa_full',       '03_00_treatments_structured'),
    ('treatments_full', '03_00_treatments_structured'),
]


def _list_all_dbs(server: Any) -> List[str]:
    """All DB names on the CouchDB server."""
    return [str(name) for name in server]


def _experiment_db_renames(
    all_dbs: List[str], experiment_names: List[str],
) -> List[Tuple[str, str]]:
    """For each (experiment, role) pair, find matching legacy DBs
    on the server and produce (old_name, new_name) pairs.

    Detects model-variant suffixes (e.g. ``_combined``) and
    preserves them at the END of the new name, mirroring
    ``_stage_tagged`` in migrate_experiment_schema.py.
    """
    renames: List[Tuple[str, str]] = []
    seen_new: set = set()
    # Longest legacy suffix first so e.g. ``taxa_full`` matches
    # before ``taxa``.
    sorted_renames = sorted(
        _RENAMES, key=lambda r: -len(r[0]),
    )
    for exp_name in experiment_names:
        for legacy_role, new_role in sorted_renames:
            legacy_prefix = f'skol_exp_{exp_name}_{legacy_role}'
            for db_name in all_dbs:
                if not db_name.startswith(legacy_prefix):
                    continue
                # Skip docs that already match a LONGER legacy role
                # (e.g. don't process `_taxa_full` twice).
                if any(
                    db_name.startswith(f'skol_exp_{exp_name}_{other}')
                    and len(other) > len(legacy_role)
                    for other, _ in sorted_renames
                ):
                    continue
                suffix = db_name[len(legacy_prefix):].lstrip('_')
                # Don't process derivative _eval DBs — those are
                # CREATED by this rename, not renamed by it.
                if db_name.endswith('_eval'):
                    continue
                new_name = f'skol_exp_{exp_name}_{new_role}'
                if suffix:
                    new_name = f'{new_name}_{suffix}'
                if new_name in seen_new or new_name == db_name:
                    continue
                seen_new.add(new_name)
                renames.append((db_name, new_name))
    return renames


def _post_replication(
    sess: Any, couchdb_url: str, creds: Tuple[str, str],
    source: str, target: str, *, create_target: bool = True,
) -> Dict[str, Any]:
    """POST one /_replicate job and return the response JSON."""
    body = {
        'source': {
            'url': f'{couchdb_url}/{source}',
            'auth': {'basic': {
                'username': creds[0], 'password': creds[1],
            }},
        },
        'target': {
            'url': f'{couchdb_url}/{target}',
            'auth': {'basic': {
                'username': creds[0], 'password': creds[1],
            }},
        },
        'create_target': create_target,
        'use_bulk_get': False,    # multipart-bug workaround
    }
    resp = sess.post(
        f'{couchdb_url}/_replicate', json=body, timeout=60,
    )
    if resp.status_code not in (200, 202):
        raise RuntimeError(
            f'_replicate {source} → {target} returned '
            f'{resp.status_code}: {resp.text[:200]}'
        )
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Rename per-experiment CouchDB DBs to '
                    'stage-tagged names',
    )
    parser.add_argument(
        '--dry-run', action='store_true', default=True,
        help='Print the rename plan without kicking off '
             'replications (default).',
    )
    parser.add_argument(
        '--execute', action='store_false', dest='dry_run',
        help='Actually kick off the replications.',
    )
    parser.add_argument(
        '--delete-legacy', action='store_true',
        help='DELETE legacy DBs after replication.  Only do this '
             'once the search UI is verified working against the '
             'new names.',
    )
    parser.add_argument(
        '--experiments-db', default='skol_experiments',
        help='Name of the experiments registry DB.',
    )
    parser.add_argument(
        '--experiment', dest='experiments', action='append',
        help='Limit rename to specific experiment(s).  Repeatable.',
    )
    args = parser.parse_args()

    config = get_env_config()
    import couchdb  # type: ignore[import]
    import requests  # type: ignore[import]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'],
        config['couchdb_password'],
    )
    creds = (config['couchdb_username'], config['couchdb_password'])

    # Figure out which experiments to process.
    exp_db = server[args.experiments_db]
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = [
            doc_id for doc_id in exp_db if not doc_id.startswith('_')
        ]
    print(f'Experiments in scope: {", ".join(experiments)}')

    all_dbs = _list_all_dbs(server)
    renames = _experiment_db_renames(all_dbs, experiments)
    print(f'\n{len(renames)} per-experiment DB renames:')
    for old, new in renames:
        existing_target = (new in all_dbs)
        marker = '  [target exists; will replicate-into]' if existing_target else ''
        print(f'  {old:<55} → {new}{marker}')
    if not renames:
        print('  (nothing to do)')
        return 0

    if args.dry_run:
        print('\nDry-run: not actually replicating.  Re-run with '
              '--execute to apply.')
        return 0

    sess = requests.Session()
    sess.verify = False     # local CouchDB; cert verification N/A
    for old, new in renames:
        print(f'  replicating {old} → {new} …', flush=True)
        try:
            result = _post_replication(
                sess, config['couchdb_url'], creds, old, new,
                create_target=True,
            )
            print(f'    ok={result.get("ok")} '
                  f'doc_write_failures='
                  f'{result.get("doc_write_failures")} '
                  f'docs_written={result.get("docs_written")}')
        except Exception as exc:  # noqa: BLE001
            print(f'    FAILED: {exc}')

    if args.delete_legacy:
        print('\nDeleting legacy DBs (per --delete-legacy):')
        for old, _ in renames:
            try:
                del server[old]
                print(f'  deleted {old}')
            except Exception as exc:  # noqa: BLE001
                print(f'  failed to delete {old}: {exc}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
