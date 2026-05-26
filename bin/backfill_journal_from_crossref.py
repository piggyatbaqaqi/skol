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
    bin/backfill_journal_from_crossref.py [--limit N] [--dry-run] [--verbosity N]

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def needs_backfill(doc: Dict[str, Any]) -> bool:
    """True iff this skol_dev doc is eligible for a journal-backfill
    Crossref lookup: must carry a non-empty ``doi`` and must not
    already carry a non-empty ``journal``.
    """
    if not doc.get('doi'):
        return False
    journal = doc.get('journal')
    return not (isinstance(journal, str) and journal.strip())


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
    stats = backfill_journals(
        db, cr,
        limit=args.limit,
        dry_run=args.dry_run,
        verbosity=verbosity,
    )
    elapsed = time.monotonic() - started

    print()
    print(f'Done in {elapsed:.1f}s')
    print(f'  Eligible scanned:    {stats["scanned"]:,}')
    print(f'  Crossref fetched:    {stats["fetched"]:,}')
    print(f'  Journal updated:     {stats["updated"]:,}  '
          f'{"(dry-run)" if args.dry_run else ""}')
    print(f'  No journal in reply: {stats["no_journal_in_crossref"]:,}')
    print(f'  Crossref errors:     {stats["crossref_error"]:,}')
    print(f'  Write errors:        {stats["write_error"]:,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
