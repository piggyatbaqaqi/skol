#!/usr/bin/env python3
"""Backfill ``journal`` (and ``doi`` when missing) onto PMC docs
in skol_dev by re-parsing their attached ``article.xml``.

No network — re-uses XML attachments already in CouchDB.  Useful
for docs whose ``journal`` field is empty because the original
ingest predates JATS journal-title extraction.

Idempotent: re-running is a no-op once fields are populated.
Supports ``--dry-run`` / ``--limit`` / ``--verbosity``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Pure helpers (covered by backfill_pmc_jats_metadata_test.py)
# ---------------------------------------------------------------------------


def compute_jats_field_update(
    doc: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide which JATS-extracted fields to apply.

    For each of ``journal`` and ``doi``: include in the update
    when the doc's current value is missing / empty AND the
    JATS extraction returned something non-empty.

    ``journal`` is run through ``normalize_journal_name`` so the
    JATS long-form (e.g., ``Persoonia - Molecular Phylogeny and
    Evolution of Fungi``) becomes the canonical short form
    (``Persoonia``) via the JOURNALS[*].aliases lists.
    """
    from ingestors.publications import PublicationRegistry
    update: Dict[str, Any] = {}
    for field in ('journal', 'doi'):
        existing = doc.get(field)
        if isinstance(existing, str) and existing.strip():
            continue  # don't overwrite a set value
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            continue  # nothing to write
        if field == 'journal':
            value = PublicationRegistry.normalize_journal_name(value)
        update[field] = value
    return update


# ---------------------------------------------------------------------------
# CouchDB iteration / write (network-touching)
# ---------------------------------------------------------------------------


def _fetch_xml_attachment(db: Any, doc: Dict[str, Any]) -> Optional[str]:
    """Read the ``article.xml`` attachment as a UTF-8 string,
    or None if absent / unreadable."""
    attachments = doc.get('_attachments') or {}
    if 'article.xml' not in attachments:
        return None
    try:
        raw = db.get_attachment(doc, 'article.xml')
        if raw is None:
            return None
        data = raw.read()
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='replace')
        return data
    except Exception:
        return None


def _is_pmc_doc_with_xml(doc: Dict[str, Any]) -> bool:
    """Eligibility filter: a PMC doc carrying an ``article.xml``
    attachment.  Other source types are out of scope for this
    backfill."""
    if doc.get('source') != 'pmc' and (
        (doc.get('meta') or {}).get('source') != 'pmc'
    ):
        return False
    attachments = doc.get('_attachments') or {}
    return 'article.xml' in attachments


def backfill(
    db: Any,
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk ``db``; for each PMC doc with an article.xml
    attachment, re-parse and apply any missing fields.

    Returns ``{scanned, with_xml, updated}``.
    """
    from ingestors.pmc import PmcIngestor
    counts = {'scanned': 0, 'with_xml': 0, 'updated': 0}
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        counts['scanned'] += 1
        if verbosity >= 2 and counts['scanned'] % 1000 == 0:
            print(f'  scanned {counts["scanned"]} docs...')
        try:
            doc = db[doc_id]
        except Exception as exc:
            if verbosity >= 1:
                print(f'  warning: could not load {doc_id}: {exc}',
                      file=sys.stderr)
            continue
        if not _is_pmc_doc_with_xml(doc):
            continue
        counts['with_xml'] += 1
        if limit is not None and counts['updated'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        xml_string = _fetch_xml_attachment(db, doc)
        if not xml_string:
            continue
        metadata = PmcIngestor._extract_metadata_from_xml(xml_string)
        update = compute_jats_field_update(doc, metadata)
        if not update:
            continue
        if verbosity >= 1:
            tag = '(DRY RUN) ' if dry_run else ''
            preview = ', '.join(f'{k}={v!r}' for k, v in update.items())
            print(f'  {tag}{doc["_id"]}: + {preview}')
        if not dry_run:
            doc.update(update)
            db.save(doc)
        counts['updated'] += 1
    return counts


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Backfill journal / doi on PMC docs in skol_dev '
                    'by re-parsing attached article.xml.',
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
    print(f'Scanned:    {counts["scanned"]}')
    print(f'With XML:   {counts["with_xml"]}')
    print(f'{"Would update" if args.dry_run else "Updated"}: '
          f'{counts["updated"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
