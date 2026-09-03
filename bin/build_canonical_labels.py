#!/usr/bin/env python3
"""Build the canonical feature-label DB from the candidate DB.

Derives, never mutates.  The candidate DB keeps every label exactly as
the annotator emitted it; this pass writes a parallel DB where each
record carries a top-level ``feature_label``, an ``attribute_path``
into the ragged attribute tree, and the named dimensions taken out of
the label string (``medium``, ``condition``, ``presence``).  Every
record keeps ``raw_label`` and ``source_db``, so the two DBs are
diffable — which is the argument that chose deriving over mutating
(operator decision 2026-09-03, the same reasoning as §12.3.42's
separate database for the v4_1 re-extraction).

The rules and their guards live in
:mod:`treatments_to_structured.canonical_annotation`.  This script
supplies the vocabulary indices, fans the transform out over a
database, and reports which rule fired how often.

**The indices here are corpus-wide on purpose.**  A one-time build
should canonicalize with everything known.  The *curve* is a different
question and uses ``heaps.prequential_band``, whose index is cumulative
so no point consults its own future.

Usage::

    bin/build_canonical_labels --experiment production_v4 --dry-run
    bin/build_canonical_labels --experiment production_v4
"""

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Container, Dict, Iterable, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import couchdb  # noqa: E402

from env_config import common_parser, get_env_config  # noqa: E402
from treatments_to_structured.canonical_annotation import (  # noqa: E402
    canonical_records,
    vocabulary_index,
)
from treatments_to_structured.feature_label_rules import (  # noqa: E402
    canonicalize,
    load_canonicalization,
)

_ESTABLISHED_MIN_DF = 5

# Order matters only for readability of the report.
_RULES = ('case_fold', 'condition', 'compound', 'sub_attribute')


def canonicalize_all(
    annotations: Iterable[Mapping[str, Any]],
    *,
    known: Mapping[str, str],
    established: Mapping[str, str],
    protected: Container[str],
    source_db: str,
) -> Tuple[List[Dict[str, Any]], 'collections.Counter[str]']:
    """Fan the transform out over annotations; return records + tally.

    ``raw`` counts annotations in and ``records`` counts documents out;
    a compound makes them differ, and conflating them would hide the
    fan-out.  Each rule is tallied by name so the report says *which*
    rule fired, and ``protected`` is tallied as evidence that map-wins
    precedence engaged rather than silence.
    """
    tally: 'collections.Counter[str]' = collections.Counter()
    for rule in _RULES + ('raw', 'records', 'unchanged', 'protected',
                          'dropped', 'duplicate'):
        tally[rule] = 0

    by_id: Dict[str, Dict[str, Any]] = {}
    for annotation in annotations:
        tally['raw'] += 1
        produced = canonical_records(
            annotation, known=known, established=established,
            protected=protected, source_db=source_db,
        )
        if not produced:
            tally['dropped'] += 1
            continue
        label = str(annotation.get('feature_label') or '').strip()
        if label.lower() in protected:
            tally['protected'] += 1
        transforms = produced[0].get('transforms') or []
        if not transforms:
            tally['unchanged'] += 1
        for rule in _RULES:
            if rule in transforms:
                tally[rule] += 1
        for record in produced:
            if record['_id'] in by_id:
                tally['duplicate'] += 1
                continue
            by_id[record['_id']] = record

    records = list(by_id.values())
    tally['records'] = len(records)
    return records, tally


def orphaned_ids(existing: Any, produced: Any) -> List[str]:
    """Documents in the DB that this build did not write.

    A derived DB owns its contents: anything the current build did not
    produce is stale and would otherwise sit beside the current record
    with no way for a reader to tell them apart.  Safe because every
    document here is reproducible from the candidate DB.

    Design documents are exempt — views belong to whoever made them.
    """
    return sorted(
        doc_id for doc_id in set(existing) - set(produced)
        if not str(doc_id).startswith('_design/')
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--source-db', default=None, metavar='NAME',
        help='Annotation DB to canonicalize.  Default: the '
             "experiment's features_candidate.",
    )
    parser.add_argument(
        '--canonical-db', default=None, metavar='NAME',
        help='Destination DB.  Default: the experiment record\'s '
             'features_canonical, else the naming-convention fallback '
             'skol_exp_<experiment>_02_57_features_canonical.',
    )
    # --dry-run comes from common_parser(): it is a shared work-control
    # flag, and redeclaring it is an argparse conflict.
    args = parser.parse_args()
    config = get_env_config(cli_args=args)

    experiment = config.get('experiment') or 'production_v4'
    source_db = args.source_db or config.get('features_candidate_db_name') \
        or f'skol_exp_{experiment}_02_50_features_candidate'
    canonical_db = args.canonical_db \
        or config.get('features_canonical_db_name') \
        or f'skol_exp_{experiment}_02_57_features_canonical'

    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (config['couchdb_username'],
                                   config['couchdb_password'])
    if source_db not in server:
        print(f'{source_db}: not found', file=sys.stderr)
        return 1

    annotations = [
        dict(row.doc)
        for row in server[source_db].view('_all_docs', include_docs=True)
        if row.doc and not row.id.startswith('_design/')
    ]
    print(f'{source_db}: {len(annotations):,} annotations')

    mapping = load_canonicalization()

    def canon(label: str) -> str:
        return canonicalize(label, mapping)

    known = vocabulary_index(annotations, canonicalizer=canon, min_df=1)
    established = vocabulary_index(
        annotations, canonicalizer=canon, min_df=_ESTABLISHED_MIN_DF)
    protected = frozenset(value.lower() for value in mapping.values())
    print(f'  vocabulary: {len(known):,} labels, '
          f'{len(established):,} established (df >= '
          f'{_ESTABLISHED_MIN_DF}), {len(protected):,} protected')

    records, tally = canonicalize_all(
        annotations, known=known, established=established,
        protected=protected, source_db=source_db)

    print(f'\n  {"raw annotations":<22}{tally["raw"]:>8,}')
    print(f'  {"canonical records":<22}{tally["records"]:>8,}')
    for key in _RULES + ('protected', 'unchanged', 'duplicate', 'dropped'):
        print(f'  {key:<22}{tally[key]:>8,}')

    labels_before = len({
        canon(str(a.get('feature_label') or '')) for a in annotations
    } - {''})
    labels_after = len({r['feature_label'] for r in records})
    print(f'\n  distinct labels: {labels_before:,} -> {labels_after:,}')

    if config.get('dry_run'):
        print('\n  --dry-run: nothing written.')
        return 0

    if canonical_db not in server:
        server.create(canonical_db)
        print(f'  created {canonical_db}')
    database = server[canonical_db]
    for record in records:
        existing = database.get(record['_id'])
        if existing is not None:
            record['_rev'] = existing['_rev']
    database.update(records)
    print(f'  wrote {len(records):,} docs to {canonical_db}')

    stale = orphaned_ids(
        (row.id for row in database.view('_all_docs')),
        {record['_id'] for record in records},
    )
    if stale:
        database.update([
            {'_id': doc_id, '_rev': database[doc_id]['_rev'],
             '_deleted': True}
            for doc_id in stale
        ])
        print(f'  deleted {len(stale):,} stale docs a previous build left')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
