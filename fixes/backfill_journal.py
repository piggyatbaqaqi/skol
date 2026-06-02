#!/usr/bin/env python3
"""Backfill the ``journal`` field on skol_dev docs via Crossref lookups.

Many skol_dev docs land without a ``journal`` field — PMC-ingested
docs, in particular, carry just ``pmcid``/``pmid``/``title``/``doi``
and no journal name.  That breaks per-journal reporting on the
Ingestion Sources page (everything bucketed as "Unknown").

This script scans skol_dev for docs that have a ``doi`` but no
``journal``, looks each one up via the Crossref REST API
(``container-title``), and writes ``journal`` back to the doc.
Idempotent — re-runs only touch docs that are still missing the
field.  ``--dry-run`` previews without writing.

Usage:
    fixes/backfill_journal.py [--limit N] [--dry-run] [--verbosity N]

Environment variables:
    COUCHDB_URL, COUCHDB_USER, COUCHDB_PASSWORD  — connection
    INGEST_DATABASE                              — defaults to skol_dev
    CROSSREF_MAILTO                              — for Crossref "polite pool"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# Add parent + bin dirs to path so this script runs both standalone
# and via the with_skol wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import couchdb
from env_config import get_env_config

logger = logging.getLogger(__name__)

# Default Crossref polite-pool contact (CLAUDE.md memory: the user
# email is piggy.yarroll@gmail.com).  Overridable via CROSSREF_MAILTO.
_DEFAULT_MAILTO = 'piggy.yarroll@gmail.com'


def extract_journal_from_crossref_work(
    work: Dict[str, Any],
) -> Optional[str]:
    """Pull the canonical journal name out of a Crossref ``works`` reply.

    Mirrors the convention used by ``ingestors/crossref.py:221``:
    take the first entry of ``container-title``, strip whitespace,
    treat empty/whitespace-only as missing.
    """
    titles = work.get('container-title')
    if not titles:
        return None
    first = titles[0]
    if not isinstance(first, str):
        return None
    stripped = first.strip()
    return stripped or None


def extract_journal_from_crossref_journal(
    msg: Dict[str, Any],
) -> Optional[str]:
    """Pull the canonical journal name out of a Crossref ``journals``
    reply (the ISSN-keyed endpoint).  Unlike the ``works`` endpoint
    which returns an array under ``container-title``, the
    ``journals/{issn}`` endpoint returns a single ``title`` string."""
    title = msg.get('title')
    if not isinstance(title, str):
        return None
    stripped = title.strip()
    return stripped or None


def normalize_issn(issn: Any) -> Optional[str]:
    """Coerce a raw skol_dev ISSN value into Crossref's canonical
    ``NNNN-NNNN`` form (or ``NNNN-NNNX`` with an uppercase check digit).

    skol_dev has at least three malformed shapes in the wild:
      * ``'0166-0616'``  — already canonical
      * ``'10520368'``   — hyphen dropped during ingest
      * ``'1660616'``    — leading zero also dropped, only 7 digits
    Returns ``None`` for anything that can't be reasonably normalized.
    """
    if not isinstance(issn, str):
        return None
    cleaned = issn.strip().upper().replace('-', '')
    if len(cleaned) == 7:
        cleaned = '0' + cleaned
    if len(cleaned) != 8:
        return None
    if not cleaned[:7].isdigit():
        return None
    if not (cleaned[7].isdigit() or cleaned[7] == 'X'):
        return None
    return f'{cleaned[:4]}-{cleaned[4:]}'


def needs_backfill(doc: Dict[str, Any]) -> bool:
    """True iff this skol_dev doc is eligible for the DOI-based
    Crossref ``works`` lookup: must carry a non-empty ``doi`` and
    must not already carry a non-empty ``journal``.
    """
    if not doc.get('doi'):
        return False
    journal = doc.get('journal')
    return not (isinstance(journal, str) and journal.strip())


def needs_issn_backfill(doc: Dict[str, Any]) -> bool:
    """True iff this skol_dev doc is eligible for the ISSN-based
    Crossref ``journals`` lookup: must carry an ``issn`` or
    ``eissn`` and must not already carry a non-empty ``journal``.

    Note this returns True even for docs that *do* have a DOI — the
    ISSN pass runs after the DOI pass and naturally picks up the
    leftovers (Crossref 404 on the DOI, no container-title in the
    reply, etc.) by re-checking the journal field.
    """
    journal = doc.get('journal')
    if isinstance(journal, str) and journal.strip():
        return False
    return bool(doc.get('issn') or doc.get('eissn'))


def _iter_eligible_docs(db, verbosity: int = 1) -> Iterator[Dict[str, Any]]:
    """Yield skol_dev docs that need a Crossref journal backfill.

    Streams via ``for doc_id in db`` so memory stays bounded across
    a 30k-row corpus.  Skips design docs.  Re-reads the doc body
    (rather than trusting an _all_docs include_docs window) so the
    eligibility check sees the latest state — important when this
    script runs more than once.
    """
    scanned = 0
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        try:
            doc = db[doc_id]
        except Exception as exc:
            logger.warning('skip %s: read failed: %s', doc_id, exc)
            continue
        scanned += 1
        if verbosity >= 2 and scanned % 1000 == 0:
            print(f'  scanned {scanned} docs...')
        if needs_backfill(doc):
            yield doc


def backfill_journals(
    db,
    cr,
    limit: Optional[int] = None,
    dry_run: bool = False,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Iterate eligible docs and backfill their ``journal`` field.

    Returns a small dict of counters for the caller to print.
    """
    stats = {
        'scanned': 0,
        'fetched': 0,
        'updated': 0,
        'no_journal_in_crossref': 0,
        'crossref_error': 0,
        'write_error': 0,
    }
    for doc in _iter_eligible_docs(db, verbosity=verbosity):
        stats['scanned'] += 1
        if limit is not None and stats['scanned'] > limit:
            stats['scanned'] -= 1  # don't count the unprocessed sentinel
            break
        doi = doc['doi']
        try:
            resp = cr.works(ids=doi)
        except Exception as exc:
            stats['crossref_error'] += 1
            if verbosity >= 1:
                print(f'  crossref error for doi={doi!r}: {exc}')
            continue
        stats['fetched'] += 1
        work = resp.get('message') if isinstance(resp, dict) else None
        if not work:
            stats['no_journal_in_crossref'] += 1
            continue
        journal = extract_journal_from_crossref_work(work)
        if not journal:
            stats['no_journal_in_crossref'] += 1
            if verbosity >= 2:
                print(f'  no container-title for {doc["_id"]} doi={doi}')
            continue
        if dry_run:
            stats['updated'] += 1
            if verbosity >= 1:
                print(f'  [dry-run] {doc["_id"]} doi={doi}: would set journal={journal!r}')
            continue
        doc['journal'] = journal
        try:
            db.save(doc)
            stats['updated'] += 1
            if verbosity >= 2:
                print(f'  {doc["_id"]} doi={doi}: journal={journal!r}')
        except Exception as exc:
            stats['write_error'] += 1
            if verbosity >= 1:
                print(f'  write error for {doc["_id"]}: {exc}')
    return stats


def _iter_issn_eligible_docs(
    db, verbosity: int = 1,
) -> Iterator[Dict[str, Any]]:
    """Yield skol_dev docs eligible for ISSN-based journal backfill.

    Companion to ``_iter_eligible_docs`` (the DOI pass); same
    streaming + design-doc skipping discipline.  Re-reads each doc
    so a doc that picked up its journal in the DOI pass naturally
    drops out of the ISSN pass on the same run.
    """
    scanned = 0
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        try:
            doc = db[doc_id]
        except Exception as exc:
            logger.warning('skip %s: read failed: %s', doc_id, exc)
            continue
        scanned += 1
        if verbosity >= 2 and scanned % 1000 == 0:
            print(f'  scanned {scanned} docs...')
        if needs_issn_backfill(doc):
            yield doc


def backfill_journals_via_issn(
    db,
    cr,
    limit: Optional[int] = None,
    dry_run: bool = False,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Second-pass backfill via the Crossref ``journals/{issn}``
    endpoint.

    Eligibility (``needs_issn_backfill``): any doc with no journal
    and an ``issn`` or ``eissn`` — including docs that have a DOI
    but had their DOI lookup fail in the first pass.

    Caches each ISSN's resolved journal so a corpus like skol_dev
    (1,934 leftover docs across 3 unique ISSNs as of the audit
    that prompted this code) costs at most one Crossref call per
    unique ISSN.  A ``None`` cache entry records a failed lookup so
    the same dead ISSN isn't re-queried.
    """
    stats = {
        'scanned': 0,
        'fetched': 0,
        'updated': 0,
        'no_journal_in_crossref': 0,
        'crossref_error': 0,
        'write_error': 0,
        'unique_issns': 0,
    }
    issn_cache: Dict[str, Optional[str]] = {}

    def _resolve(canonical_issn: str) -> Optional[str]:
        if canonical_issn in issn_cache:
            return issn_cache[canonical_issn]
        try:
            resp = cr.journals(ids=canonical_issn)
        except Exception as exc:
            stats['crossref_error'] += 1
            if verbosity >= 1:
                print(f'  crossref error for issn={canonical_issn}: {exc}')
            issn_cache[canonical_issn] = None
            return None
        stats['fetched'] += 1
        stats['unique_issns'] += 1
        msg = resp.get('message') if isinstance(resp, dict) else None
        journal_name = (
            extract_journal_from_crossref_journal(msg) if msg else None
        )
        issn_cache[canonical_issn] = journal_name
        if verbosity >= 1:
            label = repr(journal_name) if journal_name else '(no title)'
            print(f'  issn={canonical_issn} → {label}')
        return journal_name

    for doc in _iter_issn_eligible_docs(db, verbosity=verbosity):
        stats['scanned'] += 1
        if limit is not None and stats['scanned'] > limit:
            stats['scanned'] -= 1
            break
        raw_issn = doc.get('issn') or doc.get('eissn')
        canonical = normalize_issn(raw_issn)
        if not canonical:
            stats['no_journal_in_crossref'] += 1
            if verbosity >= 2:
                print(
                    f'  {doc["_id"]}: raw issn={raw_issn!r} '
                    f'failed normalize, skipping'
                )
            continue
        journal_name = _resolve(canonical)
        if not journal_name:
            stats['no_journal_in_crossref'] += 1
            continue
        if dry_run:
            stats['updated'] += 1
            if verbosity >= 1:
                print(
                    f'  [dry-run] {doc["_id"]} issn={canonical}: '
                    f'would set journal={journal_name!r}'
                )
            continue
        doc['journal'] = journal_name
        try:
            db.save(doc)
            stats['updated'] += 1
            if verbosity >= 2:
                print(
                    f'  {doc["_id"]} issn={canonical}: '
                    f'journal={journal_name!r}'
                )
        except Exception as exc:
            stats['write_error'] += 1
            if verbosity >= 1:
                print(f'  write error for {doc["_id"]}: {exc}')
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Backfill the journal field on skol_dev docs with a DOI '
            'but no journal, using the Crossref REST API.'
        ),
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Process at most N docs (default: all eligible)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Look up journals but do not write back')
    parser.add_argument('--verbosity', type=int, default=None,
                        help='0=quiet, 1=normal, 2=per-record')
    parser.add_argument('--mailto', type=str, default=None,
                        help='Crossref polite-pool contact email '
                             '(default: $CROSSREF_MAILTO or hardcoded)')
    args, _ = parser.parse_known_args()

    config = get_env_config()
    verbosity = args.verbosity if args.verbosity is not None else config.get('verbosity', 1)
    mailto = args.mailto or os.environ.get('CROSSREF_MAILTO') or _DEFAULT_MAILTO

    # Lazy import so the unit tests (which don't need habanero) stay
    # fast and don't pull in the dependency.
    from habanero import Crossref
    cr = Crossref(mailto=mailto)

    # Connect to the ingest DB.
    couchdb_url = config['couchdb_url']
    db_name = config.get('ingest_db_name') or 'skol_dev'
    server = couchdb.Server(couchdb_url)
    user, pw = config.get('couchdb_username'), config.get('couchdb_password')
    if user and pw:
        server.resource.credentials = (user, pw)
    db = server[db_name]

    if verbosity >= 1:
        print(f'Backfilling journal from Crossref on {db_name}')
        print(f'  Crossref polite-pool mailto: {mailto}')
        print(f'  Mode: {"DRY RUN" if args.dry_run else "WRITE"}')
        if args.limit:
            print(f'  Limit: {args.limit}')

    started = time.monotonic()

    if verbosity >= 1:
        print()
        print('Pass 1: DOI → Crossref works lookup')
    doi_stats = backfill_journals(
        db, cr,
        limit=args.limit,
        dry_run=args.dry_run,
        verbosity=verbosity,
    )
    pass1_elapsed = time.monotonic() - started

    # Pass 2 catches the docs the work-API pass couldn't resolve plus
    # the no-DOI-but-has-ISSN docs (e.g., older Mycotaxon, Sydowia
    # records).  Cheap: one Crossref call per *unique* ISSN, then
    # cache-hits across the rest.
    if verbosity >= 1:
        print()
        print('Pass 2: ISSN → Crossref journals lookup')
    pass2_started = time.monotonic()
    issn_stats = backfill_journals_via_issn(
        db, cr,
        limit=args.limit,
        dry_run=args.dry_run,
        verbosity=verbosity,
    )
    pass2_elapsed = time.monotonic() - pass2_started
    elapsed = time.monotonic() - started

    print()
    print(f'Done in {elapsed:.1f}s  '
          f'(pass1 {pass1_elapsed:.1f}s, pass2 {pass2_elapsed:.1f}s)')
    print('Pass 1 (DOI → works):')
    print(f'  Eligible scanned:    {doi_stats["scanned"]:,}')
    print(f'  Crossref fetched:    {doi_stats["fetched"]:,}')
    print(f'  Journal updated:     {doi_stats["updated"]:,}  '
          f'{"(dry-run)" if args.dry_run else ""}')
    print(f'  No journal in reply: {doi_stats["no_journal_in_crossref"]:,}')
    print(f'  Crossref errors:     {doi_stats["crossref_error"]:,}')
    print(f'  Write errors:        {doi_stats["write_error"]:,}')
    print('Pass 2 (ISSN → journals):')
    print(f'  Eligible scanned:    {issn_stats["scanned"]:,}')
    print(f'  Unique ISSNs queried: {issn_stats["unique_issns"]:,}')
    print(f'  Journal updated:     {issn_stats["updated"]:,}  '
          f'{"(dry-run)" if args.dry_run else ""}')
    print(f'  No journal resolved: {issn_stats["no_journal_in_crossref"]:,}')
    print(f'  Crossref errors:     {issn_stats["crossref_error"]:,}')
    print(f'  Write errors:        {issn_stats["write_error"]:,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
