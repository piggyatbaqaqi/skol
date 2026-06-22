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

Storage shape (success):

    doc['plazi'] = {
        'uuids':        ['DOCUUID32CHAR...', ...],
        'lnk_dois':     ['10.5281/zenodo.X', ...],   # parallel to uuids
        'looked_up_at': '2026-06-03T17:42:31Z',
        'source':       'plazi:GgServer:srsStats:v1',
    }

A *sticky* server-side failure instead stamps an error record (no
``looked_up_at``/``uuids``, so it's never mistaken for a successful
empty lookup).  The freshness guard backs it off for
``--retry-failed-after-days`` days; transient (pre-response) failures
are left unstamped and retried every run:

    doc['plazi'] = {
        'error':     {'reason': 'http_status', 'detail': '500',
                      'url': 'https://tb.plazi.org/.../srsStats/stats?...'},
        'failed_at': '2026-06-05T12:00:00Z',
        'source':    'plazi:GgServer:srsStats:v1',
    }
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
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

_DEFAULT_PLAZI_URL = 'https://tb.plazi.org/GgServer'
_USER_AGENT = (
    'skol-plazi-backfill/1.0 '
    '(https://synoptickeyof.life)'
)
# Source tag bumped 2026-06-10 to mark the srsStats migration.
# Docs whose `source` is the old `plazi:GgSrvApi:v1` were stamped
# via the searchByDOI endpoint (which matched against LnkDoi); a
# fresh re-run under the new tag will overwrite with the more
# accurate `pubLnk.articleDoi`-matched data.
_SOURCE_TAG = 'plazi:GgServer:srsStats:v1'
_DEFAULT_RATE_LIMIT_MIN_MS = 1000
_DEFAULT_RATE_LIMIT_MAX_MS = 2000
_DEFAULT_RE_CHECK_AFTER_DAYS = 365
_DEFAULT_RETRY_FAILED_AFTER_DAYS = 7
_DEFAULT_HEARTBEAT_EVERY = 500

# Plazi's searchByDOI sometimes returns its full ~700 k-entry index
# for DOIs it has no real match on (observed 2026-06).  Stamping
# that into the doc balloons it past CouchDB's 8 MB document limit.
# 100 is well above any plausible real match count for one DOI but
# far below the misbehavior threshold, so the guard rejects only
# the runaway responses.
_MAX_PLAZI_ENTRIES = 100

# Failure reasons.  All but ``request_error`` mean Plazi returned an
# HTTP response we couldn't use ("server-engaged"); those are sticky and
# earn the weak N-day backoff.  ``request_error`` is a pre-response
# network failure (timeout / connection / DNS), retried every run.
_STICKY_REASONS = frozenset(
    {'http_status', 'bad_json', 'not_list', 'runaway'},
)

# Marker + grammar for the reproduction-grade failure log line.  The
# bug-report tool parses these back out, so format/parse must stay in
# lock-step (round-tripped in the tests).
_FAILURE_MARKER = 'PLAZI_FAILURE'
_FAILURE_LINE_RE = re.compile(
    r'✗\s+(?P<doc_id>\S+):\s+' + _FAILURE_MARKER +
    r'\s+reason=(?P<reason>\S+)'
    r'\s+detail=(?P<detail>\S+)'
    r'\s+doi=(?P<doi>\S+)'
    r'\s+url=(?P<url>\S+)\s*$'
)


class PlaziResult(NamedTuple):
    """Outcome of one ``searchByDOI`` call.

    On success ``body`` is the parsed JSON array and ``reason`` is None.
    On failure ``body`` is None and ``reason`` names the mode, with
    ``detail`` carrying a stringified specifier (HTTP status code,
    runaway entry count, or exception class name) where one applies.
    ``url`` is always the exact URL queried, for reproduction.
    """
    body: Optional[List[Dict[str, Any]]]
    reason: Optional[str]
    detail: Optional[str]
    url: str


class FailureRecord(NamedTuple):
    """A failure recovered from a log line, consumed by the bug-report
    tool to build a reproduction."""
    doc_id: str
    reason: str
    detail: str
    doi: str
    url: str


def is_sticky_reason(reason: str) -> bool:
    """True for failures that earn the weak N-day backoff; False for the
    transient ``request_error`` (retried every run)."""
    return reason in _STICKY_REASONS


def is_heartbeat_tick(scanned: int, every: int) -> bool:
    """True when a progress heartbeat is due: every ``every`` scanned
    docs.  ``every <= 0`` disables it, and zero scanned never fires (so a
    fresh run doesn't print a spurious 0-line)."""
    return every > 0 and scanned > 0 and scanned % every == 0


def format_heartbeat(counts: Dict[str, int]) -> str:
    """A one-line progress heartbeat for long, mostly-skipped runs that
    would otherwise look frozen."""
    return (
        f'… scanned {counts["scanned"]} '
        f'(skipped {counts["skipped_fresh"]}, queried {counts["queried"]})'
    )


# ---------------------------------------------------------------------------
# Pure helpers (covered by backfill_plazi_uuids_test.py)
# ---------------------------------------------------------------------------


def parse_plazi_response(
    body: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Decompose the Plazi srsStats data list into parallel
    ``(uuids, lnk_dois)`` lists.

    Post-2026-06-10 migration: entries carry ``PubLnkArticleDoi``
    (the DOI of the article each treatment was *extracted from*),
    not the legacy ``LnkDoi`` (which was the DOI the treatment
    merely *linked to*).  The doc-level field name ``lnk_dois`` is
    preserved for storage backward-compat — every doc previously
    written carries that key, and renaming would require a full
    corpus migration.

    Entries without ``DocUuid`` are dropped (defensive against API
    shape drift).  Entries without ``PubLnkArticleDoi`` keep an
    empty string in the parallel slot so the two lists stay
    index-aligned.

    Legacy ``LnkDoi`` is deliberately NOT accepted as a fallback —
    its semantics differ from ``PubLnkArticleDoi`` and silently
    bringing those back would mask shape drift.
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
        article_doi = entry.get('PubLnkArticleDoi')
        lnk_dois.append(
            article_doi if isinstance(article_doi, str) else '',
        )
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


def compute_plazi_error(result: PlaziResult, now_iso: str) -> Dict[str, Any]:
    """Build the error record written to ``doc['plazi']`` for a sticky
    failure.

    Deliberately carries no ``looked_up_at``/``uuids`` so it can never be
    mistaken for a successful (possibly empty) lookup.  The ``failed_at``
    stamp is what ``should_skip`` reads to back the doc off for
    ``--retry-failed-after-days`` days.
    """
    return {
        'error': {
            'reason': result.reason,
            'detail': result.detail,
            'url': result.url,
        },
        'failed_at': now_iso,
        'source': _SOURCE_TAG,
    }


def format_failure_log_line(
    doc_id: str, result: PlaziResult, doi: str,
) -> str:
    """One greppable, reproduction-grade line per failure: doc id, reason,
    detail, DOI, and the exact URL.  Inverse of
    ``parse_failure_log_line``."""
    return (
        f'✗ {doc_id}: {_FAILURE_MARKER} '
        f'reason={result.reason} detail={result.detail} '
        f'doi={doi} url={result.url}'
    )


def parse_failure_log_line(line: str) -> Optional[FailureRecord]:
    """Recover a :class:`FailureRecord` from a failure log line (tolerant
    of leading indentation/prefixes).  Returns None for non-failure
    lines."""
    m = _FAILURE_LINE_RE.search(line)
    if m is None:
        return None
    return FailureRecord(
        doc_id=m.group('doc_id'),
        reason=m.group('reason'),
        detail=m.group('detail'),
        doi=m.group('doi'),
        url=m.group('url'),
    )


def _within_days(then_iso: str, now_iso: str, days: int) -> bool:
    """True when ``then_iso`` is less than ``days`` before ``now_iso``.
    Unparseable timestamps return False so the caller proceeds rather
    than blocking forever on a malformed stamp."""
    try:
        then = datetime.strptime(then_iso, '%Y-%m-%dT%H:%M:%SZ')
        now = datetime.strptime(now_iso, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        return False
    then = then.replace(tzinfo=timezone.utc)
    now = now.replace(tzinfo=timezone.utc)
    return (now - then) < timedelta(days=days)


def should_skip(
    doc: Dict[str, Any],
    *,
    force: bool,
    re_check_after_days: int,
    now_iso: str,
    retry_failed_after_days: int = _DEFAULT_RETRY_FAILED_AFTER_DAYS,
    current_source: str = _SOURCE_TAG,
) -> bool:
    """True when the doc was checked or failed recently enough to skip and
    ``--force`` is not set.  Per CLAUDE.md rule 11, default behaviour is
    idempotent.

    A stamp from a *different* ``source`` tag than ``current_source`` is
    never skipped: it predates an endpoint/semantics migration (e.g. the
    pre-srsStats ``searchByDOI`` data) and must be re-queried regardless
    of freshness.  Otherwise two freshness windows apply:

    - a successful ``looked_up_at`` within ``re_check_after_days``; and
    - a sticky-failure ``failed_at`` within ``retry_failed_after_days``
      (the weak block so restarts don't re-hit known-bad DOIs).
    """
    if force:
        return False
    plazi = doc.get('plazi')
    if not isinstance(plazi, dict):
        return False
    if plazi.get('source') != current_source:
        return False
    looked_up = plazi.get('looked_up_at')
    if isinstance(looked_up, str) and looked_up:
        return _within_days(looked_up, now_iso, re_check_after_days)
    failed_at = plazi.get('failed_at')
    if isinstance(failed_at, str) and failed_at:
        return _within_days(failed_at, now_iso, retry_failed_after_days)
    return False


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def build_search_url(doi: str, plazi_url: str) -> str:
    """Construct the Plazi srsStats query URL.  Shared by
    ``query_plazi`` and the bug-report tool so the reproduction
    URL matches what we queried.

    Trims a trailing slash on ``plazi_url`` and percent-encodes the
    *quoted* DOI as one token.

    Post-2026-06-10 migration: switched from
    ``/Treatments/searchByDOI?DOI=<doi>`` to
    ``/srsStats/stats?FP-pubLnk.articleDoi="<doi>"&outputFields=…``
    for two reasons:

    1. **Semantics**: searchByDOI matched against ``LnkDoi``
       (treatments that *reference* the DOI), but we want
       treatments *extracted from* an article with that DOI —
       which is ``pubLnk.articleDoi`` on the stats endpoint.
    2. **Dash bug**: Plazi's parser treats a bare ``-`` in the
       DOI value as a range operator.  On searchByDOI the quote-
       strip was broken, so quoting returned 0 even for
       no-dash DOIs that should have matched (because the
       semantics were wrong too — see #1).  On srsStats the
       quote-strip works correctly, so quoted DOIs exact-match
       and dashes no longer trigger the runaway.

    Live verification 2026-06-10:
    - ``FP-pubLnk.articleDoi=%22<no-dash-DOI>%22`` returns the
      treatments for that DOI (Plazi's example: 15 rows).
    - ``FP-pubLnk.articleDoi=%22<dash-DOI>%22`` exact-matches
      correctly (0 rows if Plazi lacks the article, N rows if
      they have it).
    - ``FP-pubLnk.articleDoi=<dash-DOI>`` (unquoted) still
      triggers the range bug — that's why the quotes are
      load-bearing.

    The ``_MAX_PLAZI_ENTRIES`` cap in ``query_plazi`` remains as
    defense-in-depth.
    """
    base = plazi_url.rstrip('/')
    quoted_doi = f'"{doi}"'
    fields = 'doc.uuid+pubLnk.articleDoi+tax.name'
    return (
        f'{base}/srsStats/stats'
        f'?outputFields={fields}'
        f'&groupingFields={fields}'
        f'&FP-pubLnk.articleDoi={quote(quoted_doi, safe="")}'
        f'&format=JSON'
    )


def query_plazi(
    doi: str,
    *,
    plazi_url: str,
    http_client: Any,
) -> PlaziResult:
    """Hit ``/srsStats/stats?…&FP-pubLnk.articleDoi="<doi>"&format=JSON``
    via the shared rate-limited client.

    Returns a :class:`PlaziResult` carrying the unwrapped ``data``
    list (i.e. the rows) — callers don't need to know about the
    outer ``{data: [...]}`` envelope.

    Each failure mode carries a distinct ``reason`` so the caller
    can pick a disposition (sticky vs transient) and log
    reproduction info:

    - ``request_error`` — the GET raised before any response
      (timeout / connection / DNS).  The lone *transient* reason;
      ``detail`` is the exception class name.
    - ``http_status``   — non-200 response; ``detail`` is the code.
    - ``bad_json``      — 200 but the body didn't parse as JSON.
    - ``not_list``      — 200, valid JSON, but not a dict (or its
      ``data`` value isn't a list).  Reason name preserved from
      the pre-migration era for caller-disposition stability.
    - ``runaway``       — 200 data list larger than the sanity cap;
      ``detail`` is the entry count.  See ``_MAX_PLAZI_ENTRIES``.

    A dict response with no ``data`` key (or with ``data`` absent)
    is treated as zero hits, not an error — Plazi sometimes
    returns that shape and the caller's contract is "store an
    empty uuids list and move on" in that case.
    """
    url = build_search_url(doi, plazi_url)
    try:
        resp = http_client.get(url)
    except Exception as exc:  # noqa: BLE001 — ReadTimeout/ConnError/DNS
        return PlaziResult(None, 'request_error', type(exc).__name__, url)
    if resp.status_code != 200:
        return PlaziResult(None, 'http_status', str(resp.status_code), url)
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 — malformed JSON
        return PlaziResult(None, 'bad_json', None, url)
    if not isinstance(body, dict):
        return PlaziResult(None, 'not_list', None, url)
    data = body.get('data', [])
    if not isinstance(data, list):
        return PlaziResult(None, 'not_list', None, url)
    if len(data) > _MAX_PLAZI_ENTRIES:
        return PlaziResult(None, 'runaway', str(len(data)), url)
    return PlaziResult(data, None, None, url)


class CachingHttpClient:
    """Memoize GET responses by URL for the lifetime of one run.

    A DOI is per-article, but many docs can share it; with the URL being
    a deterministic function of the DOI, this caches the response so each
    distinct DOI hits Plazi — and the rate limiter — exactly once.  It
    wraps the rate-limited client (checked *first*), so cache hits return
    immediately without sleeping or making a request.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, url: str, **kwargs: Any) -> Any:
        if url in self._cache:
            self.hits += 1
            return self._cache[url]
        self.misses += 1
        resp = self._inner.get(url, **kwargs)
        self._cache[url] = resp
        return resp


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
) -> Tuple[str, Optional[PlaziResult]]:
    """Look up ``doc['doi']`` at Plazi and stamp ``doc['plazi']``.

    Returns ``(status, result)``:

    - ``('no_doi', None)``              — doc has no usable DOI.
    - ``('updated', result)``           — success; ``doc['plazi']``
      stamped (caller saves unless dry-run).
    - ``('dry_run', result)``           — success; stamped in memory only.
    - ``('sticky_failure', result)``    — server-engaged failure; an error
      record is stamped (caller saves it unless dry-run) so the doc is
      backed off on future runs.
    - ``('transient_failure', result)`` — pre-response failure; the doc is
      left untouched and retried on the very next run.

    ``result`` is the :class:`PlaziResult` (None only for ``no_doi``) so
    the caller can log reproduction info for failures.
    """
    doi = doc.get('doi')
    if not isinstance(doi, str) or not doi:
        return ('no_doi', None)
    result = query_plazi(doi, plazi_url=plazi_url, http_client=http_client)
    if result.reason is not None:
        if is_sticky_reason(result.reason):
            doc['plazi'] = compute_plazi_error(result, now_iso)
            return ('sticky_failure', result)
        return ('transient_failure', result)
    assert result.body is not None  # reason is None ⇒ success ⇒ body set
    doc['plazi'] = compute_plazi_update(result.body, now_iso)
    if dry_run:
        return ('dry_run', result)
    return ('updated', result)


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
    # Stream output even when stdout is redirected to a logfile: Python
    # block-buffers a non-tty, so manual/cron runs otherwise show nothing
    # until the buffer fills or the process exits.  (Guarded for test
    # harnesses that replace stdout with a non-reconfigurable capture.)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

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
    parser.add_argument(
        '--retry-failed-after-days', type=int,
        default=_DEFAULT_RETRY_FAILED_AFTER_DAYS,
        help=(
            f'Weak block: back off docs whose last lookup was a sticky '
            f'server-side failure for N days before retrying '
            f'(default: {_DEFAULT_RETRY_FAILED_AFTER_DAYS}).'
        ),
    )
    parser.add_argument(
        '--heartbeat-every', type=int,
        default=_DEFAULT_HEARTBEAT_EVERY,
        help=(
            f'Print a progress line every N scanned docs so long, '
            f'mostly-skipped runs visibly advance; 0 disables '
            f'(default: {_DEFAULT_HEARTBEAT_EVERY}).'
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

    # Dedup-by-DOI: cache responses so a DOI shared by multiple docs
    # hits Plazi (and the rate limiter) only once per run.
    http_client = CachingHttpClient(RateLimitedHttpClient(
        user_agent=_USER_AGENT,
        verbosity=max(0, verbosity - 1),
        rate_limit_min_ms=args.rate_limit_min_ms,
        rate_limit_max_ms=args.rate_limit_max_ms,
    ))

    counts: Dict[str, int] = {
        'scanned': 0, 'skipped_fresh': 0, 'queried': 0,
        'hits': 0, 'empty': 0,
        'sticky_failure': 0, 'transient_failure': 0,
        'save_failure': 0, 'updated': 0,
    }
    failure_reasons: Dict[str, int] = {}

    if verbosity >= 1:
        print(f'Plazi backfill — db={db_name} plazi_url={args.plazi_url}')
        if dry_run:
            print('  *** DRY RUN — no writes ***')

    for doc in iter_doi_docs(db):
        counts['scanned'] += 1
        if verbosity >= 1 and is_heartbeat_tick(
            counts['scanned'], args.heartbeat_every,
        ):
            print(format_heartbeat(counts))
        if limit is not None and counts['queried'] >= limit:
            break
        now_iso = _utc_now_iso()
        if should_skip(
            doc, force=force,
            re_check_after_days=args.re_check_after_days,
            retry_failed_after_days=args.retry_failed_after_days,
            now_iso=now_iso,
        ):
            counts['skipped_fresh'] += 1
            continue
        counts['queried'] += 1
        status, result = process_doc(
            doc, http_client=http_client,
            plazi_url=args.plazi_url,
            now_iso=now_iso, dry_run=dry_run,
        )
        if status in ('sticky_failure', 'transient_failure'):
            counts[status] += 1
            if result is not None and result.reason is not None:
                failure_reasons[result.reason] = (
                    failure_reasons.get(result.reason, 0) + 1
                )
            # Reproduction-grade line: sticky failures are actionable
            # (candidate Plazi bug reports) so log at -v1; transient
            # network blips only at -v2.
            should_log = (
                verbosity >= 1 if status == 'sticky_failure'
                else verbosity >= 2
            )
            if result is not None and should_log:
                print('  ' + format_failure_log_line(
                    doc['_id'], result, doc['doi']))
            # Persist the sticky error stamp so the weak block survives
            # restarts; transient failures are left unstamped to retry.
            if status == 'sticky_failure' and not dry_run:
                save_errors: List[str] = []
                if not save_with_retry(
                    db, doc, on_error=save_errors.append,
                ):
                    counts['save_failure'] += 1
                    if verbosity >= 1:
                        print(
                            f'  ✗ {doc["_id"]}: error-stamp save failed '
                            'after retry (will retry next run)',
                            file=sys.stderr,
                        )
            continue
        plazi = doc.get('plazi', {})
        n_uuids = len(plazi.get('uuids', []))
        if n_uuids:
            counts['hits'] += 1
        else:
            counts['empty'] += 1
        if not dry_run:
            save_errors = []
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
        print(f'  sticky failures     : {counts["sticky_failure"]:>6}')
        print(f'  transient failures  : {counts["transient_failure"]:>6}')
        for reason in sorted(failure_reasons):
            print(f'    - {reason:<15} : {failure_reasons[reason]:>6}')
        print(f'  save failures       : {counts["save_failure"]:>6}')
        print(f'  docs updated        : {counts["updated"]:>6}')
        print(f'  hit rate            : {hit_rate:>6.1%}')
        print(f'  plazi calls (deduped): {http_client.misses:>5}'
              f'  (saved {http_client.hits} duplicate-DOI calls)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
