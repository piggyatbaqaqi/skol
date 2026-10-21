#!/usr/bin/env python3
"""Delete structured treatments whose description had no usable text.

Before 8d15386, treatments_to_json sent every description to the
model, including empty ones, ones without letters, and OCR noise.
The model answered each with an invented taxon (a frog, for one
Nomen ignotum).  The pipeline now drops such descriptions before
inference and never saves them; this one-shot fix removes the ones
saved before that, using the same rule
(``treatments_json_translator.is_translatable``).

The target database must be named explicitly, with ``--experiment``
(its treatments_structured database) or ``--db``.

Supports ``--dry-run`` / ``--limit`` / ``--verbosity``.

Usage::

    fixes/delete_invented_structured_treatments.py \\
        --experiment production_v4_1 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))


# ---------------------------------------------------------------------------
# Pure helpers (covered by delete_invented_structured_treatments_test.py)
# ---------------------------------------------------------------------------


def is_invented(doc: Dict[str, Any]) -> bool:
    """True when ``doc``'s description gave the model nothing to read."""
    from treatments_json_translator import is_translatable

    return not is_translatable(doc.get('description') or '')


def target_db_name(
    db_arg: Optional[str],
    experiment_name: Optional[str],
    config: Dict[str, Any],
) -> str:
    """The database to clean: ``--db``, else the experiment's
    treatments_structured database.

    Raises:
        ValueError: If neither ``--db`` nor ``--experiment`` was given;
            a delete must never fall back to a default database.
    """
    if db_arg:
        return db_arg
    if experiment_name:
        return str(config['treatments_structured_db_name'])
    raise ValueError(
        'name the database to clean with --experiment or --db')


# ---------------------------------------------------------------------------
# CouchDB iteration / delete (network-touching)
# ---------------------------------------------------------------------------


def delete_invented(
    db: Any,
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """Walk ``db``; delete every doc for which ``is_invented`` is True.

    Returns ``{scanned, invented, deleted}``.  Under ``dry_run``,
    ``deleted`` counts what would have been deleted.
    """
    counts = {'scanned': 0, 'invented': 0, 'deleted': 0}
    for doc in _iter_db(db, verbosity=verbosity):
        counts['scanned'] += 1
        if verbosity >= 2 and counts['scanned'] % 1000 == 0:
            print(f'  scanned {counts["scanned"]} docs...')
        if not is_invented(doc):
            continue
        counts['invented'] += 1
        if limit is not None and counts['deleted'] >= limit:
            if verbosity >= 1:
                print(f'  stop: hit --limit {limit}')
            break
        if verbosity >= 1:
            tag = '(DRY RUN) ' if dry_run else ''
            print(f'  {tag}delete {doc["_id"]}')
        if not dry_run:
            db.delete(doc)
        counts['deleted'] += 1
    return counts


def _iter_db(db: Any, verbosity: int = 1) -> Iterable[Dict[str, Any]]:
    """Yield every non-design doc from ``db``."""
    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue
        try:
            yield db[doc_id]
        except Exception as exc:
            if verbosity >= 1:
                print(f'  warning: could not load {doc_id}: {exc}',
                      file=sys.stderr)


def main() -> int:
    """CLI entry point."""
    from env_config import common_parser, get_env_config

    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        description='Delete structured treatments whose description had '
                    'no usable text (the model invented their JSON).',
    )
    parser.add_argument(
        '--db', default=None, metavar='NAME',
        help='Structured treatments database to clean (default: the '
             '--experiment\'s treatments_structured database)')
    args = parser.parse_args()
    config = get_env_config(cli_args=args)
    verbosity = config.get('verbosity', 1)

    try:
        db_name = target_db_name(
            args.db, config.get('experiment_name'), config)
    except ValueError as exc:
        parser.error(str(exc))

    import couchdb
    server = couchdb.Server(config['couchdb_url'])
    user = config.get('couchdb_username') or ''
    password = config.get('couchdb_password') or ''
    if user and password:
        server.resource.credentials = (user, password)
    if db_name not in server:
        print(f'error: database {db_name!r} not found at '
              f'{config["couchdb_url"]}', file=sys.stderr)
        return 2

    dry_run = bool(config.get('dry_run'))
    if verbosity >= 1:
        print(f'Target DB: {db_name} {"(DRY RUN)" if dry_run else ""}')
    counts = delete_invented(
        server[db_name],
        dry_run=dry_run,
        limit=config.get('limit'),
        verbosity=verbosity,
    )
    if verbosity >= 1:
        verb = 'would delete' if dry_run else 'deleted'
        print(f'Scanned {counts["scanned"]}, invented {counts["invented"]}, '
              f'{verb} {counts["deleted"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
