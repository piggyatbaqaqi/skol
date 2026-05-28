#!/usr/bin/env python3
"""Replicate an experiment's CouchDB databases to a remote host.

Idempotent: CouchDB's ``_replicate`` endpoint streams only the
changes since the last replication, so re-running is safe and cheap
once the first full copy is in place.

Resolves the database list from the experiment doc (``databases.*``
fields under ``skol_experiments``), and always also replicates
``skol_experiments`` itself so the experiment doc travels with the
data.

Usage:
    bin/replicate_experiment.py \\
        --experiment production_v3_hand \\
        --target-host 10.42.0.99 \\
        [--target-user USER] [--target-pass PASS] \\
        [--continuous] [--include-golden]

When ``--target-user``/``--target-pass`` are omitted, the script
assumes the target is in "admin party" mode (no auth) — useful for a
fresh CouchDB install before security is locked down.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)


def databases_for_experiment(
    experiment_doc: Dict[str, Any],
    include_golden: bool = False,
) -> List[str]:
    """Return the unique CouchDB database names referenced by an
    experiment doc's ``databases.*`` block.

    Always includes ``skol_experiments`` (the registry that holds the
    doc itself) so the experiment metadata replicates alongside the
    data.  Order: ``skol_experiments`` first, then the per-field
    references in stable alphabetical order with duplicates removed.

    ``include_golden=False`` (default) drops the ``golden`` and
    ``golden_ann`` references, which are typically already shared
    across experiments and may already exist on the target.
    """
    excluded_keys = set()
    if not include_golden:
        excluded_keys.update({'golden', 'golden_ann'})

    seen: List[str] = ['skol_experiments']
    db_block = (experiment_doc.get('databases') or {})
    for key in sorted(db_block):
        if key in excluded_keys:
            continue
        value = db_block[key]
        if isinstance(value, str) and value and value not in seen:
            seen.append(value)
    return seen


def build_couchdb_url(
    host: str,
    user: Optional[str] = None,
    password: Optional[str] = None,
    port: int = 5984,
    scheme: str = 'http',
) -> str:
    """Build a CouchDB base URL with embedded credentials.

    Percent-encodes the username and password so values containing
    ``@``, ``:``, ``/``, etc. survive the URL.  Returns the bare host
    URL (no credentials) when both ``user`` and ``password`` are
    falsy — useful for an "admin party" target.
    """
    netloc = f'{host}:{port}'
    if user and password:
        user_enc = urllib.parse.quote(user, safe='')
        pass_enc = urllib.parse.quote(password, safe='')
        netloc = f'{user_enc}:{pass_enc}@{netloc}'
    return f'{scheme}://{netloc}'


def _build_replicate_body(
    source_url: str,
    target_url: str,
    continuous: bool = False,
) -> Dict[str, Any]:
    """Construct the JSON body for a ``POST /_replicate`` call."""
    body: Dict[str, Any] = {
        'source': source_url,
        'target': target_url,
        'create_target': True,
    }
    if continuous:
        body['continuous'] = True
    return body


def replicate(
    source_admin_url: str,
    source_creds: Optional[tuple],
    source_url: str,
    target_url: str,
    db_name: str,
    continuous: bool = False,
    timeout: int = 7200,
    verbosity: int = 1,
) -> Dict[str, Any]:
    """POST a single-database replication request to the local
    CouchDB's ``/_replicate`` endpoint.

    Returns the parsed JSON reply on success, or a dict with
    ``ok=False`` and ``error`` on failure.
    """
    import requests
    body = _build_replicate_body(
        f'{source_url}/{db_name}',
        f'{target_url}/{db_name}',
        continuous=continuous,
    )
    url = f'{source_admin_url}/_replicate'
    if verbosity >= 1:
        action = 'continuous replication' if continuous else 'replication'
        print(f'  starting {action} for {db_name} ...')
    try:
        resp = requests.post(
            url, json=body, auth=source_creds, timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        return {'ok': False, 'error': str(exc), 'db': db_name}
    try:
        data = resp.json()
    except ValueError:
        return {
            'ok': False,
            'error': f'non-JSON reply (HTTP {resp.status_code}): '
                     f'{resp.text[:200]}',
            'db': db_name,
        }
    if resp.status_code >= 400:
        return {
            'ok': False,
            'error': data.get('error', resp.status_code),
            'reason': data.get('reason'),
            'db': db_name,
        }
    return {**data, 'db': db_name, 'ok': True}


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Replicate an experiment\'s CouchDB databases to '
                    'a remote host.',
    )
    parser.add_argument('--experiment', required=True,
                        help='Experiment doc name in skol_experiments '
                             '(e.g. production_v3_hand)')
    parser.add_argument('--target-host', required=True,
                        help='Target CouchDB host (e.g. 10.42.0.99)')
    parser.add_argument('--target-port', type=int, default=5984,
                        help='Target CouchDB port (default: 5984)')
    parser.add_argument('--target-scheme', default='http',
                        choices=('http', 'https'),
                        help='Target CouchDB URL scheme (default: http)')
    parser.add_argument('--target-user', default=None,
                        help='Target CouchDB username (default: no auth)')
    parser.add_argument('--target-pass', default=None,
                        help='Target CouchDB password')
    parser.add_argument('--continuous', action='store_true',
                        help='Set continuous=true on each replication '
                             '(creates a persistent _replicator job '
                             'instead of one-shot)')
    parser.add_argument('--include-golden', action='store_true',
                        help='Also replicate experiment.databases.golden '
                             'and .golden_ann (off by default — these '
                             'tend to be shared across experiments)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List the databases that would be '
                             'replicated; do not POST')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='0=quiet, 1=normal, 2=verbose')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Per-database replicate POST timeout in '
                             'seconds (default: 7200 = 2h)')
    # Use parse_known_args so env_config's auto-generated flags
    # (--couchdb-username, --couchdb-password, --couchdb-url, etc.)
    # pass through instead of erroring out as unknown.
    args, _unknown = parser.parse_known_args()

    # Lazy imports so the unit tests don't need couchdb / env_config.
    from env_config import get_env_config
    config = get_env_config()

    src_admin_url = config['couchdb_url']
    src_user = config.get('couchdb_username') or ''
    src_pass = config.get('couchdb_password') or ''
    src_creds = (src_user, src_pass) if (src_user and src_pass) else None
    # ``source`` URL gets embedded in the replicate body and must be
    # resolvable from the local CouchDB's perspective.
    src_url = build_couchdb_url(
        # The source URL embedded in the replicate body must be
        # reachable from the local CouchDB process.  ``localhost``
        # is the common case; the rest of the URL is reconstructed
        # from the admin URL to honour any non-default port.
        host=urllib.parse.urlparse(src_admin_url).hostname or 'localhost',
        user=src_user, password=src_pass,
        port=urllib.parse.urlparse(src_admin_url).port or 5984,
        scheme=urllib.parse.urlparse(src_admin_url).scheme or 'http',
    )
    tgt_url = build_couchdb_url(
        host=args.target_host,
        user=args.target_user, password=args.target_pass,
        port=args.target_port, scheme=args.target_scheme,
    )

    # Resolve the experiment doc.
    import couchdb
    server = couchdb.Server(src_admin_url)
    if src_creds:
        server.resource.credentials = src_creds
    try:
        experiment_doc = server['skol_experiments'][args.experiment]
    except Exception as exc:
        print(f'error: could not load experiment {args.experiment!r}: '
              f'{exc}', file=sys.stderr)
        return 2

    dbs = databases_for_experiment(
        experiment_doc, include_golden=args.include_golden,
    )

    if args.verbosity >= 1:
        print(f'Replicating experiment {args.experiment} → '
              f'{args.target_host}:{args.target_port}')
        print(f'  Mode: {"CONTINUOUS" if args.continuous else "one-shot"}'
              f'{" (DRY RUN)" if args.dry_run else ""}')
        print(f'  Databases ({len(dbs)}):')
        for db in dbs:
            print(f'    - {db}')

    if args.dry_run:
        return 0

    started = time.monotonic()
    results = []
    for db in dbs:
        result = replicate(
            source_admin_url=src_admin_url,
            source_creds=src_creds,
            source_url=src_url,
            target_url=tgt_url,
            db_name=db,
            continuous=args.continuous,
            timeout=args.timeout,
            verbosity=args.verbosity,
        )
        results.append(result)
        if args.verbosity >= 1:
            if result.get('ok'):
                docs = (result.get('docs_written')
                        or result.get('history', [{}])[0].get(
                            'docs_written', '?'))
                print(f'  ✓ {db}: docs_written={docs}')
            else:
                print(f'  ✗ {db}: {result.get("error")} '
                      f'{result.get("reason", "")}')

    elapsed = time.monotonic() - started
    ok = sum(1 for r in results if r.get('ok'))
    fail = len(results) - ok
    print()
    print(f'Done in {elapsed:.1f}s: {ok} succeeded, {fail} failed.')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
