#!/usr/bin/env python3
"""Backfill ``journal`` onto skol_dev docs by URL / DOI fingerprint.

Walks the ingest database; for each doc whose ``journal`` field is
missing or empty, tries fingerprint matching:

1. DOI → ``PublicationRegistry.find_journal_by_doi`` (matches
   journal-level DOIs, e.g. ``10.23880/oajmms``).
2. Ingenta URL → ``find_journal_by_ingenta_url`` (matches
   ``ingentaconnect.com/contentone/<publisher>/<journal>/...``).

When a match is found, writes the canonical journal name to the
doc.  Idempotent; supports ``--dry-run`` / ``--limit`` / ``--verbosity``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))


# ---------------------------------------------------------------------------
# Pure helper (covered by backfill_fingerprint_journals_test.py)
# ---------------------------------------------------------------------------


def compute_fingerprint_update(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Try to recover a ``journal`` value for ``doc`` from its
    DOI or pdf_url.  Returns ``{'journal': <canonical name>}`` on
    a match, ``{}`` otherwise.

    DOI takes priority over URL — it identifies a journal directly
    when the DOI is a journal-level one (e.g. ``10.23880/oajmms``).
    """
    existing = doc.get('journal')
    if isinstance(existing, str) and existing.strip():
        return {}
    from ingestors.publications import PublicationRegistry
    # Try DOI first (most specific).
    doi = doc.get('doi')
    if isinstance(doi, str) and doi.strip():
        slug = PublicationRegistry.find_journal_by_doi(doi.strip())
        if slug is not None:
            return {'journal': PublicationRegistry.JOURNALS[slug]['name']}
    # Then Ingenta URL fingerprint.
    url = doc.get('pdf_url') or doc.get('url')
    if isinstance(url, str) and url.strip():
        slug = PublicationRegistry.find_journal_by_ingenta_url(url)
        if slug is not None:
            return {'journal': PublicationRegistry.JOURNALS[slug]['name']}
    return {}


# ---------------------------------------------------------------------------
# CouchDB iteration / write (network-touching)
# ---------------------------------------------------------------------------


def _iter_db(db: Any, verbosity: int = 1) -> Iterable[Dict[str, Any]]:
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        try:
            yield db[doc_id]
        except Exception as exc:
            if verbosity >= 1:
                print(f'  warning: could not load {doc_id}: {exc}',
                      file=sys.stderr)


def backfill(
    db: Any,
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk ``db``; apply ``compute_fingerprint_update`` to each
    doc; persist matches.  Returns ``{scanned, updated}``."""
    counts = {'scanned': 0, 'updated': 0}
    for doc in _iter_db(db, verbosity=verbosity):
        counts['scanned'] += 1
        if verbosity >= 2 and counts['scanned'] % 1000 == 0:
            print(f'  scanned {counts["scanned"]} docs...')
        update = compute_fingerprint_update(doc)
        if not update:
            continue
        if limit is not None and counts['updated'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        if verbosity >= 1:
            tag = '(DRY RUN) ' if dry_run else ''
            print(f'  {tag}{doc["_id"]}: journal={update["journal"]!r}')
        if not dry_run:
            doc.update(update)
            db.save(doc)
        counts['updated'] += 1
    return counts


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Backfill journal field via URL / DOI fingerprint '
                    'matching against PublicationRegistry.JOURNALS.',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview updates without writing.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after N successful updates.')
    parser.add_argument('--verbosity', type=int, default=None,
                        help='0=quiet, 1=normal, 2=verbose.')
    args, _unknown = parser.parse_known_args()

    from env_config import get_env_config
    config = get_env_config()
    verbosity = (args.verbosity if args.verbosity is not None
                 else config.get('verbosity', 1))

    import couchdb
    server = couchdb.Server(config['couchdb_url'])
    src_user = config.get('couchdb_username') or ''
    src_pass = config.get('couchdb_password') or ''
    if src_user and src_pass:
        server.resource.credentials = (src_user, src_pass)
    db_name = (config.get('ingest_db_name')
               or config.get('couchdb_database')
               or 'skol_dev')
    if db_name not in server:
        print(f'error: database {db_name!r} not found at '
              f'{config["couchdb_url"]}', file=sys.stderr)
        return 2
    if verbosity >= 1:
        print(f'Target DB: {db_name} '
              f'{"(DRY RUN)" if args.dry_run else ""}')

    counts = backfill(
        db=server[db_name],
        dry_run=args.dry_run,
        limit=args.limit,
        verbosity=verbosity,
    )

    print()
    print(f'Scanned: {counts["scanned"]}')
    print(f'{"Would update" if args.dry_run else "Updated"}: '
          f'{counts["updated"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
