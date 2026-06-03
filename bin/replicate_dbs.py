#!/usr/bin/env python3
"""Replicate a CouchDB database through a chain of endpoints.

Use case that motivated this script: pushing ``skol_dev`` from
puchpuchobs → tsqali → production.  Each hop POSTs to the previous
endpoint's ``/_replicate`` so the previous endpoint pushes to the
next — same direction the manual curl chain used.

Endpoints are named.  Each name resolves to a URL + admin
credentials via env vars::

    <NAME>_COUCHDB_URL          (required)
    <NAME>_COUCHDB_USERNAME     (default: 'admin')
    <NAME>_COUCHDB_PASSWORD     (required if URL needs auth)

``local`` / ``default`` / ``self`` fall back to the standard
``COUCHDB_URL`` / ``COUCHDB_USER`` / ``COUCHDB_PASSWORD`` so the
default puchpuchobs box is the implicit source unless overridden.

Authentication is sent in CouchDB's object form
(``auth.basic.{username, password}``) rather than URL-embedded
``http://user:pw@host`` so passwords containing ``@`` or other
URL-reserved characters don't trip the URL parser — same trade-off
we made in the original puchpuchobs/tsqali curl chain.

Usage::

    bin/replicate_dbs.py --db skol_dev \\
        --source local --target tsqali --target skol

Default behaviour is incremental replication (CouchDB checkpoints).
Pass ``--recreate-target`` to drop and recreate each target before
replicating — the "exactly match" mode.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # type: ignore[import-untyped]  # noqa: E402


# ---------------------------------------------------------------------------
# Endpoint model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str         # without trailing slash
    username: str
    password: str


_LOCAL_ALIASES = frozenset({'local', 'default', 'self', ''})


def resolve_endpoint(
    name: str, env: Mapping[str, str],
) -> Endpoint:
    """Resolve an endpoint name to a populated :class:`Endpoint`.

    Convention: ``<NAME>_COUCHDB_URL`` + ``<NAME>_COUCHDB_USERNAME``
    + ``<NAME>_COUCHDB_PASSWORD``.  The aliases ``local`` /
    ``default`` / ``self`` (and the empty string) fall back to the
    standard ``COUCHDB_URL`` / ``COUCHDB_USER`` / ``COUCHDB_PASSWORD``
    pair the rest of skol uses.

    Raises ``ValueError`` if no URL is found.
    """
    if name.lower() in _LOCAL_ALIASES:
        url = env.get('COUCHDB_URL')
        username = env.get('COUCHDB_USER', 'admin')
        password = env.get('COUCHDB_PASSWORD', '')
        resolved_name = name or 'local'
    else:
        prefix = name.upper()
        url = env.get(f'{prefix}_COUCHDB_URL')
        username = env.get(f'{prefix}_COUCHDB_USERNAME', 'admin')
        password = env.get(f'{prefix}_COUCHDB_PASSWORD', '')
        resolved_name = name
    if not url:
        raise ValueError(
            f"No URL for endpoint {name!r}: expected "
            f"{name.upper()}_COUCHDB_URL (or COUCHDB_URL for "
            "local/default)."
        )
    return Endpoint(
        name=resolved_name,
        url=url.rstrip('/'),
        username=username,
        password=password,
    )


# ---------------------------------------------------------------------------
# Replication payload
# ---------------------------------------------------------------------------


def build_replication_payload(
    source: Endpoint, target: Endpoint, db_name: str,
    *, create_target: bool = True,
) -> Dict[str, Any]:
    """Build the JSON body to POST to a CouchDB ``/_replicate`` endpoint.

    Uses the object form of source / target so the password lives
    inside ``auth.basic.password`` rather than being embedded in the
    URL — the latter trips up on any URL-reserved character.
    """
    return {
        'source': {
            'url': f'{source.url}/{db_name}',
            'auth': {'basic': {
                'username': source.username,
                'password': source.password,
            }},
        },
        'target': {
            'url': f'{target.url}/{db_name}',
            'auth': {'basic': {
                'username': target.username,
                'password': target.password,
            }},
        },
        'create_target': create_target,
    }


# ---------------------------------------------------------------------------
# Recreate target
# ---------------------------------------------------------------------------


def recreate_target_db(
    target: Endpoint, db_name: str, *,
    http: Any, verify: bool = False,
) -> None:
    """DELETE then PUT the target database.

    A 404 on the initial DELETE just means the DB doesn't exist yet —
    that's fine, we'd PUT it anyway.  A non-2xx PUT, by contrast, is
    a real failure (auth, disk, etc.) that the caller must see.
    """
    auth = (target.username, target.password)
    url = f'{target.url}/{db_name}'

    delete_resp = http.delete(url, auth=auth, verify=verify)
    if delete_resp.status_code not in (200, 202, 404):
        raise RuntimeError(
            f'DELETE {url} returned {delete_resp.status_code}: '
            f'{delete_resp.text[:200]}'
        )

    put_resp = http.put(url, auth=auth, verify=verify)
    if put_resp.status_code not in (200, 201, 202, 412):
        # 412 = 'file_exists' — fine (DB came back into existence
        # between our DELETE and PUT, e.g. a replication checkpoint).
        raise RuntimeError(
            f'PUT {url} returned {put_resp.status_code}: '
            f'{put_resp.text[:200]}'
        )


# ---------------------------------------------------------------------------
# Per-hop and chain
# ---------------------------------------------------------------------------


def _replicate_one_hop(
    source: Endpoint, target: Endpoint, db_name: str, *,
    http: Any, verify: bool = False,
    create_target: bool = True,
    recreate: bool = False,
    verbosity: int = 1,
) -> Dict[str, Any]:
    """Do one ``source → target`` replication via source's
    ``/_replicate`` endpoint."""
    if recreate:
        if verbosity >= 1:
            print(f'  → {target.name}/{db_name}: recreate '
                  '(DELETE + PUT before replicate)')
        recreate_target_db(target, db_name, http=http, verify=verify)

    payload = build_replication_payload(
        source, target, db_name, create_target=create_target,
    )
    url = f'{source.url}/_replicate'
    if verbosity >= 1:
        print(f'  → POST {url}  ({source.name} → {target.name}/{db_name})')
    resp = http.post(
        url, json=payload,
        auth=(source.username, source.password),
        verify=verify,
    )
    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(
            f'_replicate POST returned {resp.status_code}: '
            f'{resp.text[:300]}'
        )
    body = resp.json() if hasattr(resp, 'json') else {}
    return body if isinstance(body, dict) else {}


def replicate_chain(
    endpoints: List[Endpoint], db_name: str, *,
    http: Any, verify: bool = False,
    create_target: bool = True,
    recreate: bool = False,
    dry_run: bool = False,
    verbosity: int = 1,
) -> List[Dict[str, Any]]:
    """Walk the chain pairwise: ``endpoints[i]`` pushes to
    ``endpoints[i+1]`` for each consecutive pair.

    With ``dry_run=True`` no HTTP traffic is sent — useful for
    confirming the chain interpretation before pulling the trigger.
    """
    results: List[Dict[str, Any]] = []
    if dry_run:
        if verbosity >= 1:
            print('  *** DRY RUN — no HTTP traffic ***')
            for prev, nxt in zip(endpoints, endpoints[1:]):
                print(f'  would POST {prev.url}/_replicate  '
                      f'({prev.name} → {nxt.name}/{db_name})')
        return results
    for prev, nxt in zip(endpoints, endpoints[1:]):
        result = _replicate_one_hop(
            prev, nxt, db_name, http=http, verify=verify,
            create_target=create_target, recreate=recreate,
            verbosity=verbosity,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Doc-count verification
# ---------------------------------------------------------------------------


def get_doc_count(
    endpoint: Endpoint, db_name: str, *,
    http: Any, verify: bool = False,
) -> Optional[int]:
    """Return ``doc_count`` reported by CouchDB for the given DB, or
    ``None`` if the request fails."""
    try:
        resp = http.get(
            f'{endpoint.url}/{db_name}',
            auth=(endpoint.username, endpoint.password),
            verify=verify,
        )
        if resp.status_code != 200:
            return None
        body = resp.json()
        return int(body.get('doc_count', 0))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Replicate a CouchDB database through a chain of endpoints. '
            'Endpoints are named; each name resolves to '
            '<NAME>_COUCHDB_URL / _USERNAME / _PASSWORD env vars.'
        ),
    )
    parser.add_argument(
        '--db', required=True, metavar='DB_NAME',
        help='Database to replicate.',
    )
    parser.add_argument(
        '--source', required=True, metavar='NAME',
        help=(
            'Source endpoint name.  Use "local" (or "default") to fall '
            'back to the standard COUCHDB_URL/USER/PASSWORD env vars.'
        ),
    )
    parser.add_argument(
        '--target', action='append', required=True,
        metavar='NAME',
        help='Target endpoint name.  Repeat for a chain.',
    )
    parser.add_argument(
        '--recreate-target', action='store_true',
        help=(
            'DELETE + PUT each target DB before replicating.  Use '
            'when target has drifted from source and you want an '
            'exact match.'
        ),
    )
    parser.add_argument(
        '--no-create-target', action='store_true',
        help=(
            'Skip "create_target=true" in the replication payload.  '
            'Default is to ensure the target DB exists before pushing.'
        ),
    )
    parser.add_argument(
        '--insecure', action='store_true',
        help=(
            'Skip TLS certificate verification.  Needed when targeting '
            'tsqali via its self-signed cert without the local TLS '
            'stop-gap applied; not appropriate for production.'
        ),
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print the chain interpretation; make no HTTP requests.',
    )
    parser.add_argument(
        '-v', '--verbosity', type=int, default=1,
        help='0=quiet  1=normal  2=verbose',
    )
    args = parser.parse_args()

    try:
        source = resolve_endpoint(args.source, os.environ)
        targets = [
            resolve_endpoint(name, os.environ) for name in args.target
        ]
    except ValueError as exc:
        print(f'✗ {exc}', file=sys.stderr)
        return 1

    endpoints = [source, *targets]
    verify = not args.insecure
    sess = requests.Session()

    if args.verbosity >= 1:
        print(
            f'Replicating {args.db!r} through chain: ' +
            ' → '.join(f'{ep.name} ({ep.url})' for ep in endpoints)
        )
        if args.recreate_target:
            print('  (each target dropped + recreated first)')

    try:
        replicate_chain(
            endpoints, args.db, http=sess, verify=verify,
            create_target=not args.no_create_target,
            recreate=args.recreate_target,
            dry_run=args.dry_run,
            verbosity=args.verbosity,
        )
    except Exception as exc:  # noqa: BLE001
        print(f'✗ replication failed: {exc}', file=sys.stderr)
        return 2

    if args.verbosity >= 1 and not args.dry_run:
        print()
        print('Doc counts after replication:')
        for ep in endpoints:
            count = get_doc_count(
                ep, args.db, http=sess, verify=verify,
            )
            print(f'  {ep.name:<12} {ep.url:<48} '
                  f'{count if count is not None else "?"} docs')

    return 0


if __name__ == '__main__':
    sys.exit(main())
