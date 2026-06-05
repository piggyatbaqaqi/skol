#!/usr/bin/env python3
"""Sync Plazi treatment UUIDs onto skol_dev docs.

For each doc with a non-empty ``doi`` field, query Plazi's
``searchByDOI`` endpoint and write the resulting Plazi document
UUIDs (and the parallel LnkDoi list) into ``doc['plazi']``.  Empty
hits are stored too (with an empty ``uuids`` list and a
``looked_up_at`` timestamp) so the doc is marked as checked —
idempotent re-runs skip stamped docs unless ``--force`` is given
or the lookup is older than ``--re-check-after-days`` days.

Lives in ``bin/`` because it's expected to be run periodically as
Plazi's coverage grows and as new docs with DOIs are ingested.  The
freshness guard makes recurring sweeps cheap (only re-queries entries
older than the threshold).

Uses the canonical rate-limited HTTP client
(``ingestors/rate_limited_client.py``) so we play nicely with
Plazi's small community service.  Default pacing: 1-2 s between
requests (≈ 1.5 req/s avg).  ``--rate-limit-min-ms`` /
``--rate-limit-max-ms`` tune that.

Storage shape:

    doc['plazi'] = {
        'uuids':        ['DOCUUID32CHAR...', ...],
        'lnk_dois':     ['10.5281/zenodo.X', ...],   # parallel to uuids
        'looked_up_at': '2026-06-03T17:42:31Z',
        'source':       'plazi:GgSrvApi:v1',
    }
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import couchdb  # type: ignore[import]  # noqa: E402

from env_config import get_env_config  # type: ignore[import]  # noqa: E402
from ingestors.rate_limited_client import (  # noqa: E402
    RateLimitedHttpClient,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PLAZI_URL = 'https://api.plazi.org/GgSrvApi/v1'
_USER_AGENT = (
    'skol-plazi-backfill/1.0 '
    '(https://synoptickeyof.life)'
)
_SOURCE_TAG = 'plazi:GgSrvApi:v1'
_DEFAULT_RATE_LIMIT_MIN_MS = 1000
_DEFAULT_RATE_LIMIT_MAX_MS = 2000
_DEFAULT_RE_CHECK_AFTER_DAYS = 365

# Plazi's searchByDOI sometimes returns its full ~700 k-entry index
# for DOIs it has no real match on (observed 2026-06).  Stamping
# that into the doc balloons it past CouchDB's 8 MB document limit.
# 100 is well above any plausible real match count for one DOI but
# far below the misbehavior threshold, so the guard rejects only
# the runaway responses.
_MAX_PLAZI_ENTRIES = 100


# ---------------------------------------------------------------------------
# Pure helpers (covered by backfill_plazi_uuids_test.py)
# ---------------------------------------------------------------------------


def parse_plazi_response(
    body: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Decompose the Plazi searchByDOI JSON array into parallel
    ``(uuids, lnk_dois)`` lists.

    Entries without ``DocUuid`` are dropped (defensive against API
    shape drift).  Entries without ``LnkDoi`` keep an empty string in
    the parallel slot so the two lists stay index-aligned.
    """
    uuids: List[str] = []
    lnk_dois: List[str] = []
    for entry in body:
        if not isinstance(entry, dict):
            continue
        uuid = entry.get('DocUuid')
        if not isinstance(uuid, str) or not uuid:
            continue
        uuids.append(uuid)
        lnk = entry.get('LnkDoi')
        lnk_dois.append(lnk if isinstance(lnk, str) else '')
    return uuids, lnk_dois


def compute_plazi_update(
    body: List[Dict[str, Any]],
    now_iso: str,
) -> Dict[str, Any]:
    """Build the dict that goes at ``doc['plazi']``.

    An empty ``body`` (Plazi has no record for this DOI) still produces
    a fully-formed dict with ``uuids=[]`` and a ``looked_up_at`` stamp
    — that's what makes idempotent re-runs skip the doc next time.
    """
    uuids, lnk_dois = parse_plazi_response(body)
    return {
        'uuids': uuids,
        'lnk_dois': lnk_dois,
        'looked_up_at': now_iso,
        'source': _SOURCE_TAG,
    }


def should_skip(
    doc: Dict[str, Any],
    *,
    force: bool,
    re_check_after_days: int,
    now_iso: str,
) -> bool:
    """True when the doc has been looked up recently enough and
    ``--force`` is not set.  Per CLAUDE.md rule 11, default behaviour
    is idempotent."""
    if force:
        return False
    plazi = doc.get('plazi')
    if not isinstance(plazi, dict):
        return False
    looked_up = plazi.get('looked_up_at')
    if not isinstance(looked_up, str) or not looked_up:
        return False
    try:
        prev = datetime.strptime(looked_up, '%Y-%m-%dT%H:%M:%SZ')
        prev = prev.replace(tzinfo=timezone.utc)
        now = datetime.strptime(now_iso, '%Y-%m-%dT%H:%M:%SZ')
        now = now.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return (now - prev) < timedelta(days=re_check_after_days)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def query_plazi(
    doi: str,
    *,
    plazi_url: str,
    http_client: Any,
) -> Optional[List[Dict[str, Any]]]:
    """Hit ``/Treatments/searchByDOI?DOI=<doi>&format=json`` via the
    shared rate-limited client.

    Returns the parsed JSON array on 200, or ``None`` on any non-200
    status, network timeout, connection error, or malformed JSON.
    ``None`` is the caller's signal to leave the doc unstamped so the
    next run retries it — important because Plazi's ReadTimeout
    exception from a slow server response shouldn't crash a 24k-doc
    backfill.
    """
    base = plazi_url.rstrip('/')
    url = (
        f'{base}/Treatments/searchByDOI'
        f'?DOI={quote(doi, safe="")}'
        f'&format=json'
    )
    try:
        resp = http_client.get(url)
    except Exception:  # noqa: BLE001 — ReadTimeout / ConnectionError / DNS
        return None
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 — malformed JSON
        return None
    if not isinstance(body, list):
        return None
    if len(body) > _MAX_PLAZI_ENTRIES:
        # Server-side misbehavior — treat as if the query failed so
        # the doc is left unstamped and skipped on the next run too
        # (further retries against a still-misbehaving API would
        # keep losing).  See _MAX_PLAZI_ENTRIES comment.
        return None
    return body


# ---------------------------------------------------------------------------
# Per-doc processing
# ---------------------------------------------------------------------------


def process_doc(
    doc: Dict[str, Any],
    *,
    http_client: Any,
    plazi_url: str,
    now_iso: str,
    dry_run: bool = False,
) -> str:
    """Look up ``doc['doi']`` at Plazi and stamp ``doc['plazi']``.

    Returns a status string: ``'updated'`` (doc was changed and, if
    not dry-run, saved to the DB by the caller), ``'dry_run'`` (would
    have updated; in-memory mutation done so verbose output can show
    the diff), or ``'http_failure'`` (Plazi unreachable / non-200;
    the doc is left untouched and will be retried on the next run).
    """
    doi = doc.get('doi')
    if not isinstance(doi, str) or not doi:
        return 'no_doi'
    body = query_plazi(doi, plazi_url=plazi_url, http_client=http_client)
    if body is None:
        return 'http_failure'
    doc['plazi'] = compute_plazi_update(body, now_iso)
    if dry_run:
        return 'dry_run'
    return 'updated'


# ---------------------------------------------------------------------------
# Save with retry
# ---------------------------------------------------------------------------


def save_with_retry(
    db: Any,
    doc: Dict[str, Any],
    *,
    sleep_seconds: float = 0.5,
    max_attempts: int = 2,
    sleep_fn: Any = time.sleep,
    on_error: Any = None,
) -> bool:
    """Call ``db.save(doc)`` with one polite retry on transient
    failure.

    The live Plazi backfill against skol_dev originally tripped a
    handful of *transient* HTTP 413s that cleared on a second
    attempt; the retry catches those.  But we later hit *permanent*
    document_too_large errors caused by a 43 MB plazi response,
    and the silent retry made that very hard to diagnose.  So the
    failure path now surfaces the exception via ``on_error`` (the
    CLI hooks it into stderr) instead of being swallowed.

    Returns ``True`` on success, ``False`` after exhausting
    ``max_attempts`` failures.  Caller increments its
    ``save_failures`` counter on ``False`` and leaves the doc
    unstamped so the next run retries it.

    ``sleep_fn`` is injectable so the tests don't actually sleep.
    ``on_error`` is a ``Callable[[str], None]`` invoked once per
    failed attempt with a one-line ``"attempt N/M: <type>: <msg>"``
    string.  ``None`` (the default) drops the messages.
    """
    for attempt in range(max_attempts):
        try:
            db.save(doc)
            return True
        except Exception as exc:  # noqa: BLE001
            # couchdb.http.ServerError + transport errors + …
            if on_error is not None:
                on_error(
                    f'attempt {attempt + 1}/{max_attempts}: '
                    f'{type(exc).__name__}: {exc}'
                )
            if attempt + 1 >= max_attempts:
                return False
            sleep_fn(sleep_seconds)
    return False


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


def iter_doi_docs(db: Any) -> Iterator[Dict[str, Any]]:
    """Yield each doc in ``db`` that has a non-empty ``doi`` field,
    skipping design docs."""
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        try:
            doc = db[doc_id]
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(doc, dict):
            continue
        doi = doc.get('doi')
        if not isinstance(doi, str) or not doi:
            continue
        yield doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Cross-reference skol_dev DOIs with Plazi UUIDs.',
    )
    parser.add_argument(
        '--source-db', default=None,
        help='CouchDB database (default: env_config couchdb_database).',
    )
    parser.add_argument(
        '--plazi-url', default=_DEFAULT_PLAZI_URL,
        help=f'Plazi API root (default: {_DEFAULT_PLAZI_URL}).',
    )
    parser.add_argument(
        '--rate-limit-min-ms', type=int,
        default=_DEFAULT_RATE_LIMIT_MIN_MS,
        help=(
            f'Min delay between requests in ms '
            f'(default: {_DEFAULT_RATE_LIMIT_MIN_MS}).'
        ),
    )
    parser.add_argument(
        '--rate-limit-max-ms', type=int,
        default=_DEFAULT_RATE_LIMIT_MAX_MS,
        help=(
            f'Max delay between requests in ms '
            f'(default: {_DEFAULT_RATE_LIMIT_MAX_MS}).'
        ),
    )
    parser.add_argument(
        '--re-check-after-days', type=int,
        default=_DEFAULT_RE_CHECK_AFTER_DAYS,
        help=(
            f'Re-query docs whose lookup is older than N days '
            f'(default: {_DEFAULT_RE_CHECK_AFTER_DAYS}).'
        ),
    )
    args, _ = parser.parse_known_args()

    config = get_env_config()
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    force = bool(config.get('force', False))
    limit_raw = config.get('limit')
    limit = (
        int(limit_raw) if limit_raw not in (None, '') else None
    )

    db_name = args.source_db or config.get('couchdb_database', 'skol_dev')
    server = couchdb.Server(config['couchdb_url'])
    username = config.get('couchdb_username')
    password = config.get('couchdb_password')
    if username and password:
        server.resource.credentials = (username, password)
    if db_name not in server:
        print(f"✗ database {db_name!r} not found on "
              f"{config['couchdb_url']}", file=sys.stderr)
        return 1
    db = server[db_name]

    http_client = RateLimitedHttpClient(
        user_agent=_USER_AGENT,
        verbosity=max(0, verbosity - 1),
        rate_limit_min_ms=args.rate_limit_min_ms,
        rate_limit_max_ms=args.rate_limit_max_ms,
    )

    counts: Dict[str, int] = {
        'scanned': 0, 'skipped_fresh': 0, 'queried': 0,
        'hits': 0, 'empty': 0, 'http_failure': 0,
        'save_failure': 0, 'updated': 0,
    }

    if verbosity >= 1:
        print(f'Plazi backfill — db={db_name} plazi_url={args.plazi_url}')
        if dry_run:
            print('  *** DRY RUN — no writes ***')

    for doc in iter_doi_docs(db):
        counts['scanned'] += 1
        if limit is not None and counts['queried'] >= limit:
            break
        now_iso = _utc_now_iso()
        if should_skip(
            doc, force=force,
            re_check_after_days=args.re_check_after_days,
            now_iso=now_iso,
        ):
            counts['skipped_fresh'] += 1
            continue
        counts['queried'] += 1
        result = process_doc(
            doc, http_client=http_client,
            plazi_url=args.plazi_url,
            now_iso=now_iso, dry_run=dry_run,
        )
        if result == 'http_failure':
            counts['http_failure'] += 1
            if verbosity >= 2:
                print(f'  ✗ {doc["_id"]}: HTTP failure on {doc["doi"]!r}')
            continue
        plazi = doc.get('plazi', {})
        n_uuids = len(plazi.get('uuids', []))
        if n_uuids:
            counts['hits'] += 1
        else:
            counts['empty'] += 1
        if not dry_run:
            save_errors: List[str] = []
            if save_with_retry(db, doc, on_error=save_errors.append):
                counts['updated'] += 1
            else:
                counts['save_failure'] += 1
                if verbosity >= 1:
                    print(
                        f'  ✗ {doc["_id"]}: save failed after retry '
                        '(doc left unstamped; next run will retry)',
                        file=sys.stderr,
                    )
                    for line in save_errors:
                        print(f'      {line}', file=sys.stderr)
                continue
        if verbosity >= 2:
            tag = '(DRY RUN) ' if dry_run else ''
            print(f'  {tag}{doc["_id"]}: doi={doc["doi"]!r} '
                  f'-> {n_uuids} uuid(s)')

    if verbosity >= 1:
        hit_rate = (
            counts['hits'] / counts['queried']
            if counts['queried'] else 0.0
        )
        print()
        print(f'  docs scanned        : {counts["scanned"]:>6}')
        print(f'  skipped (fresh)     : {counts["skipped_fresh"]:>6}')
        print(f'  queries issued      : {counts["queried"]:>6}')
        print(f'  queries with hits   : {counts["hits"]:>6}')
        print(f'  queries empty       : {counts["empty"]:>6}')
        print(f'  http failures       : {counts["http_failure"]:>6}')
        print(f'  save failures       : {counts["save_failure"]:>6}')
        print(f'  docs updated        : {counts["updated"]:>6}')
        print(f'  hit rate            : {hit_rate:>6.1%}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
