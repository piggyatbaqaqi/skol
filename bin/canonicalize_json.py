#!/usr/bin/env python3
"""Write ``json_canonical`` beside ``json_annotated`` in an experiment's
structured treatments database.

Proposal 2a of docs/plans/vocab-tree-narrowing.md (Trello AZeJMCR3).
Runs after ``treatments_to_json``, as the pipeline step
``canonicalize_json``.

The experiment record pins the map version it applies::

    "vocab_canonical": {"synonyms": "<blob hash>",
                        "spellings": "<blob hash>"}

``--pin`` writes the installed files' hashes into the record; it is the
only thing that changes the pin.  Otherwise the run refuses when the
installed files do not match the pin, and rewrites only the documents
that are stale: no ``json_canonical``, another map version, or a
``json_annotated`` changed since.  ``--force`` rewrites every document.

Usage::

    canonicalize_json --experiment production_v4_1 --pin
    canonicalize_json --experiment production_v4_1 [--dry-run] [--force]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skol_share import share_file  # noqa: E402
from treatments_to_structured.vocab_canonical import (  # noqa: E402
    SPELLINGS_NAME,
    SYNONYMS_NAME,
    MapError,
    Spellings,
    Synonyms,
    Version,
    canonical_fields,
    installed_version,
    is_fresh,
    load_spellings,
    load_synonyms,
    parse_annotated,
)

RECORD_FIELD = 'vocab_canonical'
EXPERIMENTS_DB = 'skol_experiments'
BATCH = 500


class PinError(Exception):
    """The record's pin is missing or does not match the installed map."""


def build_parser() -> argparse.ArgumentParser:
    """Strict CLI: script flags plus env_config's ``common_parser()``."""
    from env_config import common_parser

    parser = argparse.ArgumentParser(
        description="Write json_canonical into an experiment's structured "
                    'treatments.  Rewrites stale documents only, unless '
                    '--force.',
        parents=[common_parser()],
    )
    parser.add_argument(
        '--pin', action='store_true',
        help="Pin the experiment record's vocab_canonical to the installed "
             'synonym map and spelling table, then stop')
    return parser


def pin_record(record: Mapping[str, Any],
               installed: Version) -> Dict[str, Any]:
    """A copy of the record pinned to ``installed``."""
    pinned = dict(record)
    pinned[RECORD_FIELD] = dict(installed)
    return pinned


def check_pin(record: Mapping[str, Any], installed: Version) -> Version:
    """The record's pin, if it matches ``installed``; else PinError."""
    name = record.get('_id', '<experiment>')
    pin = record.get(RECORD_FIELD)
    if not pin:
        raise PinError(
            f'experiment {name} has no {RECORD_FIELD} pin.  To pin the '
            f'installed map {installed}, run: canonicalize_json '
            f'--experiment {name} --pin')
    if dict(pin) != dict(installed):
        raise PinError(
            f'experiment {name} pins {dict(pin)} but the installed map is '
            f'{dict(installed)}.  Install the pinned release, or re-pin '
            f'with: canonicalize_json --experiment {name} --pin')
    return dict(pin)


def plan_updates(docs: Iterable[Mapping[str, Any]], synonyms: Synonyms,
                 spellings: Spellings, version: Version,
                 force: bool = False
                 ) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    """The documents to write (each a full copy with the canonical
    fields set) and counts: ``checked``, ``updated``, ``fresh`` and
    ``unusable`` (no usable ``json_annotated``)."""
    counts: Counter[str] = Counter(
        {'checked': 0, 'updated': 0, 'fresh': 0, 'unusable': 0})
    updates: List[Dict[str, Any]] = []
    for doc in docs:
        if str(doc.get('_id', '')).startswith('_design/'):
            continue
        counts['checked'] += 1
        if parse_annotated(doc) is None:
            counts['unusable'] += 1
            continue
        if not force and is_fresh(doc, version):
            counts['fresh'] += 1
            continue
        fields = canonical_fields(doc, synonyms, spellings, version)
        assert fields is not None
        updated = dict(doc)
        updated.update(fields)
        updates.append(updated)
        counts['updated'] += 1
    return updates, counts


def apply_updates(db: Any, docs: List[Dict[str, Any]],
                  batch: int = BATCH) -> int:
    """Save with bulk updates; returns how many were saved."""
    saved = 0
    for start in range(0, len(docs), batch):
        results = db.update(docs[start:start + batch])
        saved += sum(1 for ok, _id, _rev in results if ok)
    return saved


def _batches(db: Any, limit: Any) -> Iterator[List[Dict[str, Any]]]:
    """The database's documents, BATCH at a time."""
    batch: List[Dict[str, Any]] = []
    seen = 0
    for row in db.iterview('_all_docs', BATCH, include_docs=True):
        if limit and seen >= int(limit):
            break
        seen += 1
        batch.append(dict(row.doc))
        if len(batch) >= BATCH:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    """Pin, or bring the experiment's documents up to the pin."""
    args = build_parser().parse_args()
    import couchdb
    from env_config import get_env_config

    config = get_env_config(cli_args=args)
    name = config.get('experiment_name')
    db_name = config.get('treatments_structured_db_name')
    if not name or not db_name:
        print('--experiment naming a structured treatments database is '
              'required', file=sys.stderr)
        return 2
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (config['couchdb_username'],
                                   config['couchdb_password'])
    experiments = server[EXPERIMENTS_DB]
    record = experiments[name]
    share = config.get('share_dir')
    installed = installed_version(share)

    if args.pin:
        if args.dry_run:
            print(f'Dry run: would pin {name} to {installed}')
            return 0
        experiments.save(pin_record(record, installed))
        print(f'Pinned {name}: {RECORD_FIELD} = {installed}')
        return 0

    try:
        version = check_pin(record, installed)
        synonyms = load_synonyms(share_file(SYNONYMS_NAME, share))
        spellings = load_spellings(share_file(SPELLINGS_NAME, share))
    except (PinError, MapError) as e:
        print(f'Refusing: {e}', file=sys.stderr)
        return 1

    db = server[db_name]
    totals: Counter[str] = Counter()
    saved = 0
    for batch in _batches(db, config.get('limit')):
        updates, counts = plan_updates(batch, synonyms, spellings, version,
                                       force=bool(args.force))
        totals.update(counts)
        if updates and not args.dry_run:
            saved += apply_updates(db, updates)
    print(f'{db_name}: checked {totals["checked"]:,}, fresh '
          f'{totals["fresh"]:,}, unusable {totals["unusable"]:,}, '
          f'{"would update" if args.dry_run else "updated"} '
          f'{totals["updated"]:,}')
    if not args.dry_run and saved != totals['updated']:
        print(f'Saved only {saved:,} of {totals["updated"]:,} (conflicts); '
              f'run again', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
