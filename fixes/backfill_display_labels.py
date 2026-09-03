#!/usr/bin/env python3
"""Backfill display labels onto annotations that predate the capture.

``brat_safe_type`` strips every character outside ``[A-Za-z0-9_-]``,
because brat's storage regex demands it, and until the parse-time
capture landed the original was discarded.  So the corpus holds
``Sch ffer s reaction`` where a person wrote *Schäffer's reaction*, and
``Conidium length width ratio`` where the treatment said
*length/width* — a relation flattened into three nouns.

**The span is the witness**, and
:func:`treatments_to_structured.display_labels.recover_display_label`
reads it.  This script fans that over a database and decides between
the candidates a label collects across its several spans: the
most-witnessed form wins, because letting iteration order decide is
the mistake ``vocabulary_index`` made.

**It does not repair the source.**  Three personal names are damaged in
the span itself — ``Sabouraud ' s``, ``Leonian •s``, ``Silva-Hutner' s``
— where OCR destroyed the apostrophe before any label existed.  Those
are recorded as known-unrecoverable in
``docs/feature_label_singletons.md`` rather than guessed at here.

Idempotent: an annotation already carrying a display form is skipped,
so a captured original is never overruled by an inferred one.
"""

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

import couchdb  # noqa: E402

from env_config import get_env_config  # noqa: E402
from treatments_to_structured.display_labels import (  # noqa: E402
    recover_display_label,
)


def choose_display_labels(
        annotations: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Map each label to the display form its spans best support.

    A label appears on many annotations, and their spans do not always
    agree — ``Clamp connections`` was witnessed as
    ``Clamp-connections`` ten times and as a hyphen-deleted rejoin
    once.  The corpus votes; ties break lexicographically so a re-run
    gives the same answer.
    """
    votes: Dict[str, 'collections.Counter[str]'] = collections.defaultdict(
        collections.Counter,
    )
    for annotation in annotations:
        if annotation.get('display_label'):
            continue
        label = str(annotation.get('feature_label') or '')
        recovered = recover_display_label(
            label, str(annotation.get('source_text') or ''),
        )
        if recovered:
            votes[label][recovered] += 1

    chosen: Dict[str, str] = {}
    for label, counter in votes.items():
        best = min(counter.items(), key=lambda item: (-item[1], item[0]))
        chosen[label] = best[0]
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--db', action='append', dest='dbs', metavar='NAME',
        help='database to backfill; repeatable.  Defaults to the '
             'production_v4 candidate and hand DBs.',
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='report what would change and write nothing')
    args = parser.parse_args()

    dbs = args.dbs or [
        'skol_exp_production_v4_02_50_features_candidate',
        'skol_exp_production_v4_02_55_features_hand',
    ]
    config = get_env_config()
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (config['couchdb_username'],
                                   config['couchdb_password'])

    status = 0
    for name in dbs:
        if name not in server:
            print(f'{name}: not found', file=sys.stderr)
            status = 1
            continue
        database = server[name]
        docs = [
            dict(row.doc)
            for row in database.view('_all_docs', include_docs=True)
            if row.doc and not row.id.startswith('_design/')
        ]
        chosen = choose_display_labels(docs)
        updates: List[Dict[str, Any]] = []
        for doc in docs:
            if doc.get('display_label'):
                continue
            display = chosen.get(str(doc.get('feature_label') or ''))
            if display:
                doc['display_label'] = display
                updates.append(doc)

        print(f'{name}: {len(docs):,} annotations, '
              f'{len(chosen)} labels recover, '
              f'{len(updates):,} docs to stamp')
        for label in sorted(chosen):
            print(f'    {label:<44} -> {chosen[label]}')
        if args.dry_run:
            print('    --dry-run: nothing written.')
            continue
        if updates:
            database.update(updates)
            print(f'    wrote {len(updates):,} docs.')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
