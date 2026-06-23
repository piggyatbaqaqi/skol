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
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared endpoint-resolution convention with bin/replicate_dbs:
# named endpoints look up `<NAME>_COUCHDB_URL/USER/PASSWORD` env
# vars; ``local`` / ``default`` / ``self`` fall back to the bare
# ``COUCHDB_*`` triple.
from replicate_dbs import Endpoint, resolve_endpoint  # noqa: E402

logger = logging.getLogger(__name__)


# Legacy/canonical field-name pairs from the 2026-06-10 DB-naming
# migration.  The schema migration kept both names on each doc for
# transition-window backward compat; once the canonical field is
# present, the legacy field points at a stale snapshot of the same
# data and shouldn't be replicated.
_LEGACY_TO_CANONICAL: Dict[str, str] = {
    'taxa':            'treatments_prose',
    'treatments':      'treatments_prose',
    'taxa_full':       'treatments_structured',
    'treatments_full': 'treatments_structured',
}


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

    Legacy field names from the 2026-06-10 DB-naming migration
    (``taxa`` / ``taxa_full`` / ``treatments`` / ``treatments_full``)
    are dropped when their canonical counterpart (``treatments_prose``
    / ``treatments_structured``) is present.  Unmigrated docs that
    carry only the legacy fields still get their DBs replicated so
    backward compat is preserved for any operator who hasn't yet
    run the schema migration on their doc.
    """
    excluded_keys = set()
    if not include_golden:
        excluded_keys.update({'golden', 'golden_ann'})

    db_block = (experiment_doc.get('databases') or {})
    # Skip a legacy field iff its canonical counterpart is present
    # AND populated on the same doc.  This is per-pair: one pair
    # can be migrated while another isn't.
    for legacy_key, canonical_key in _LEGACY_TO_CANONICAL.items():
        if db_block.get(canonical_key):
            excluded_keys.add(legacy_key)

    seen: List[str] = ['skol_experiments']
    for key in sorted(db_block):
        if key in excluded_keys:
            continue
        value = db_block[key]
        if isinstance(value, str) and value and value not in seen:
            seen.append(value)
    return seen


def build_couchdb_url(
    host: str,
    port: int = 5984,
    scheme: str = 'http',
) -> str:
    """Build a bare CouchDB base URL — no embedded credentials.

    Credentials travel separately in the structured ``auth`` block of
    the replicate body (see ``_build_replicate_body``).  CouchDB's
    replicator does not percent-decode URL userinfo before forwarding
    it as Basic Auth, so passwords containing ``@`` (e.g. our local
    admin password) silently fail when embedded in the URL.
    """
    return f'{scheme}://{host}:{port}'


def _resolve_target_endpoint(
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> Endpoint:
    """Resolve the target endpoint.

    Prefers the new ``--target NAME`` shortcut — same env-var
    convention as :func:`bin.replicate_dbs.resolve_endpoint`:
    ``<NAME>_COUCHDB_URL`` / ``_USER`` / ``_PASSWORD``, with
    ``local`` / ``default`` / ``self`` falling back to the bare
    ``COUCHDB_*`` triple.

    Falls back to the legacy ``--target-host`` / ``-port`` /
    ``-scheme`` / ``-user`` / ``-pass`` flags when ``--target NAME``
    is not provided — preserves cron entries and operator habits
    from before the shortcut existed.

    When both are present, the ``--target NAME`` shortcut wins; a
    stale legacy flag won't silently override the new one.

    Raises ``ValueError`` when neither path can produce a target.
    """
    if getattr(args, 'target', None):
        return resolve_endpoint(args.target, env)
    if not getattr(args, 'target_host', None):
        raise ValueError(
            'No target specified.  Use `--target NAME` '
            '(preferred — resolves <NAME>_COUCHDB_URL/USER/PASSWORD '
            'env vars) or the legacy `--target-host`.'
        )
    url = build_couchdb_url(
        host=args.target_host,
        port=args.target_port,
        scheme=args.target_scheme,
    )
    return Endpoint(
        name=args.target_host,
        url=url.rstrip('/'),
        username=args.target_user or 'admin',
        password=args.target_pass or '',
    )


def _resolve_source_endpoint(
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> Endpoint:
    """Resolve the source endpoint.

    Default behaviour is ``local`` (the script's host CouchDB) —
    matches the existing operational reality where
    ``replicate_experiment`` pushes from the box it runs on.

    Pass ``--source NAME`` to override; same env-var convention as
    :func:`_resolve_target_endpoint`.
    """
    name = getattr(args, 'source', None) or 'local'
    return resolve_endpoint(name, env)


def _wrap_side(
    url: str, auth: Optional[tuple],
) -> Any:
    """Format one side of a replicate body.

    Returns the bare URL string when ``auth`` is missing or has empty
    user/password (admin-party).  Otherwise returns the structured
    ``{"url": ..., "auth": {"basic": {...}}}`` form CouchDB needs to
    avoid the URL-userinfo bug.
    """
    if not auth:
        return url
    user, password = auth
    if not (user and password):
        return url
    return {
        'url': url,
        'auth': {'basic': {'username': user, 'password': password}},
    }


def _build_replicate_body(
    source_url: str,
    target_url: str,
    source_auth: Optional[tuple] = None,
    target_auth: Optional[tuple] = None,
    continuous: bool = False,
) -> Dict[str, Any]:
    """Construct the JSON body for a ``POST /_replicate`` call.

    When ``source_auth`` / ``target_auth`` are supplied, the
    corresponding side is emitted as the structured
    ``{"url": ..., "auth": {"basic": {"username": ..., "password":
    ...}}}`` form — avoiding the CouchDB replicator bug where
    percent-encoded ``@`` in URL userinfo is forwarded verbatim into
    the Basic Auth header.
    """
    body: Dict[str, Any] = {
        'source': _wrap_side(source_url, source_auth),
        'target': _wrap_side(target_url, target_auth),
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
    source_auth: Optional[tuple] = None,
    target_auth: Optional[tuple] = None,
    continuous: bool = False,
    timeout: int = 7200,
    verbosity: int = 1,
) -> Dict[str, Any]:
    """POST a single-database replication request to the local
    CouchDB's ``/_replicate`` endpoint.

    ``source_creds`` authenticates the POST to ``/_replicate`` itself
    (i.e. the admin call to the local CouchDB).  ``source_auth`` and
    ``target_auth`` go inside the replicate body's structured ``auth``
    block — they're what the replicator uses to talk to the source
    and target databases.  Often ``source_creds == source_auth``, but
    they're separate to allow admin-party setups on either side.

    Returns the parsed JSON reply on success, or a dict with
    ``ok=False`` and ``error`` on failure.
    """
    import requests
    body = _build_replicate_body(
        f'{source_url}/{db_name}',
        f'{target_url}/{db_name}',
        source_auth=source_auth,
        target_auth=target_auth,
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
    parser.add_argument(
        '--source', default=None, metavar='NAME',
        help=(
            'Source endpoint NAME — resolves '
            '<NAME>_COUCHDB_URL / _USER / _PASSWORD env vars '
            '(matches the bin/replicate_dbs convention).  '
            '"local" / "default" fall back to the bare COUCHDB_URL '
            'triple.  Default: local.'
        ),
    )
    parser.add_argument(
        '--target', default=None, metavar='NAME',
        help=(
            'Target endpoint NAME — same env-var convention as '
            '--source.  Preferred over the legacy --target-host / '
            '--target-port / --target-user / --target-pass flags '
            '(which still work when --target is omitted).'
        ),
    )
    parser.add_argument('--target-host', default=None,
                        help='Target CouchDB host (legacy — use '
                             '--target NAME instead).')
    parser.add_argument('--target-port', type=int, default=5984,
                        help='Target CouchDB port (legacy).')
    parser.add_argument('--target-scheme', default='http',
                        choices=('http', 'https'),
                        help='Target CouchDB URL scheme (legacy).')
    parser.add_argument('--target-user', default=None,
                        help='Target CouchDB username (legacy).')
    parser.add_argument('--target-pass', default=None,
                        help='Target CouchDB password (legacy).')
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
    # This script resolves endpoints from os.environ (the bin/replicate_dbs
    # NAME convention), not env_config's CLI flags, so parse strictly:
    # an unknown flag here is a typo, not a passthrough.
    args = parser.parse_args()

    # Resolve source and target via the shared NAME-shortcut
    # convention (see _resolve_source_endpoint / _resolve_target_endpoint).
    try:
        source_ep = _resolve_source_endpoint(args, os.environ)
        target_ep = _resolve_target_endpoint(args, os.environ)
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    src_url = source_ep.url
    src_creds = (
        (source_ep.username, source_ep.password)
        if source_ep.username and source_ep.password else None
    )
    tgt_url = target_ep.url
    tgt_auth = (
        (target_ep.username, target_ep.password)
        if target_ep.username and target_ep.password else None
    )

    # Resolve the experiment doc from the source endpoint.
    import couchdb
    server = couchdb.Server(src_url)
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
        print(f'Replicating experiment {args.experiment}: '
              f'{source_ep.name} → {target_ep.name}')
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
            # With the new endpoint resolution, the POST target
            # (_replicate endpoint) and the replicator's source
            # URL are the same; the creds travel separately via
            # source_creds.  Pre-refactor they came from
            # different places (env_config's admin URL had
            # embedded creds; build_couchdb_url stripped them) —
            # the same-string convention is the post-refactor
            # simplification.
            source_admin_url=src_url,
            source_creds=src_creds,
            source_url=src_url,
            target_url=tgt_url,
            db_name=db,
            source_auth=src_creds,
            target_auth=tgt_auth,
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
