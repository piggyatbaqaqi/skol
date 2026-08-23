#!/usr/bin/env python3
"""Populate ``Span.head`` on spans written before fingerprints existed.

``span_resolver`` verifies a span's ``head`` against the text it
resolves to, which is what turns a wrong-attachment read from silent
into loud (see §16 in
``docs/data_quality_production_v4_model.md``).  Spans written before
that field existed carry no fingerprint, so the check passes
vacuously.  This resolves each span once and records what it found.

One-shot, hence ``fixes/`` rather than ``bin/``.  The recurring guard
is ``bin/verify_spans``.

Two properties worth knowing:

* **Idempotent.**  An existing ``head`` is never overwritten, so an
  interrupted run can simply be repeated.
* **Refuses to guess.**  A treatment whose coordinate space cannot be
  determined raises rather than getting a fabricated fingerprint — a
  wrong ``head`` is worse than none, because ``verify_head()`` would
  then reject a *correct* resolution for ever after.

Usage::

    fixes/backfill_span_heads --experiment production_v4 --dry-run
    fixes/backfill_span_heads --experiment production_v4
"""

import argparse
import collections
import sys
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple,
)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'bin'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from span_resolver import (  # noqa: E402
    SpanResolutionError,
    coordinate_space,
    span_head,
)

_SPAN_FIELDS = (
    'nomenclature_spans', 'description_spans', 'diagnosis_spans',
    'etymology_spans', 'distribution_spans', 'materials_examined_spans',
    'type_designation_spans', 'biology_spans', 'notes_spans',
    'figure_caption_spans',
)


class AttachmentCache:
    """Least-recently-used cache of attachment text.

    Many treatments share one source document, and the annotated
    attachment runs to hundreds of kilobytes; re-reading it per
    treatment is the whole cost of the backfill.
    """

    def __init__(self, server: Any, max_entries: int = 8) -> None:
        self._server = server
        self._max_entries = max(1, max_entries)
        self._cache: 'collections.OrderedDict[Any, str]' = \
            collections.OrderedDict()
        self.reads = 0

    def text(self, db: str, doc_id: str, attachment: str) -> str:
        """Attachment contents, reading it at most once per key."""
        key = (db, doc_id, attachment)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if db not in self._server:
            raise SpanResolutionError(f"database {db!r} not found")
        blob = self._server[db].get_attachment(doc_id, attachment)
        if blob is None:
            raise SpanResolutionError(
                f"attachment {db}/{doc_id}/{attachment} not found"
            )
        self.reads += 1
        decoded: str = blob.read().decode('utf-8', errors='replace')
        self._cache[key] = decoded
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        return decoded

    def clear(self) -> None:
        """Drop everything cached."""
        self._cache.clear()


def backfill_treatment(
    treatment: Dict[str, Any], cache: AttachmentCache,
) -> int:
    """Set ``head`` on each span that lacks one.

    Returns the number of spans changed, so the caller can skip
    writing a document that needs nothing.

    Raises:
        SpanResolutionError: if the coordinate space is unknown or the
            offsets fall outside the attachment.
    """
    fields = [f for f in _SPAN_FIELDS if treatment.get(f)]
    if not fields:
        return 0
    pending = [
        span for field in fields for span in treatment[field]
        if 'head' not in span
    ]
    if not pending:
        return 0

    space = coordinate_space(treatment)
    text = cache.text(space.db, space.doc_id, space.attachment)
    changed = 0
    for span in pending:
        raw_start, raw_end = span.get('start_char'), span.get('end_char')
        if raw_start is None or raw_end is None:
            raise SpanResolutionError(
                f"span has no character offsets: {span!r}"
            )
        try:
            start, end = int(raw_start), int(raw_end)
        except (TypeError, ValueError) as exc:
            raise SpanResolutionError(
                f"span has non-numeric character offsets: "
                f"start_char={raw_start!r} end_char={raw_end!r}"
            ) from exc
        if start < 0 or end > len(text) or start > end:
            raise SpanResolutionError(
                f"span [{start}:{end}] falls outside {space} "
                f"({len(text)} chars)"
            )
        span['head'] = span_head(text[start:end])
        changed += 1
    return changed


def group_by_source(
    pairs: Sequence[Tuple[str, Optional[str]]],
) -> List[str]:
    """Order treatment ids so those sharing a source document adjoin.

    ``_all_docs`` returns treatments ordered by their taxon hash, so
    neighbours almost never share an ingest document and the
    attachment cache thrashes — a 400-treatment dry run made 398
    reads.  Grouping first makes it one read per source document.

    Treatments with no ingest id sort last: they cannot be resolved
    anyway, so they should not interleave with work that can.
    """
    grouped: 'collections.OrderedDict[str, List[str]]' = \
        collections.OrderedDict()
    orphans: List[str] = []
    for treatment_id, source_id in pairs:
        if source_id is None:
            orphans.append(treatment_id)
        else:
            grouped.setdefault(source_id, []).append(treatment_id)
    out: List[str] = []
    for ids in grouped.values():
        out.extend(ids)
    out.extend(orphans)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Imported here rather than at module scope: env_config lives in
    # bin/, and the pure functions above must stay importable from a
    # test without dragging in the CLI config machinery.
    from env_config import common_parser, get_env_config

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # --dry-run and --limit come from common_parser().
    parser.add_argument(
        '--cache-size', type=int, default=8, metavar='N',
        help='Attachments held in memory (default 8).',
    )
    args = parser.parse_args()
    config = get_env_config(cli_args=args)

    if not config.get('experiment_name'):
        print("error: --experiment is required", file=sys.stderr)
        return 2

    import couchdb  # type: ignore[import-untyped]
    server = couchdb.Server(config['couchdb_url'])
    server.resource.credentials = (
        config['couchdb_username'], config['couchdb_password'],
    )
    db_name = config['treatments_db_name']
    if db_name not in server:
        print(f"error: {db_name} not found", file=sys.stderr)
        return 2
    db = server[db_name]

    cache = AttachmentCache(server, max_entries=args.cache_size)
    # One pass to learn each treatment's source document, so the
    # attachment cache sees them grouped rather than hash-ordered.
    pairs = [
        (r.id, ((r.doc or {}).get('ingest') or {}).get('_id'))
        for r in db.view('_all_docs', include_docs=True)
        if r.id.startswith('taxon_')
    ]
    ids = group_by_source(pairs)
    if args.limit:
        ids = ids[:args.limit]

    seen = written = spans_set = 0
    failures: "collections.Counter[str]" = collections.Counter()
    for treatment_id in ids:
        treatment = db[treatment_id]
        seen += 1
        try:
            changed = backfill_treatment(treatment, cache)
        except SpanResolutionError as exc:
            failures[str(exc).split(':')[0][:70]] += 1
            continue
        if not changed:
            continue
        spans_set += changed
        if not args.dry_run:
            db.save(treatment)
        written += 1
        if written % 500 == 0:
            print(f"  {written} treatments, {spans_set} spans, "
                  f"{cache.reads} attachment reads")

    verb = 'would set' if args.dry_run else 'set'
    print(f"{seen} treatments examined; {verb} {spans_set} heads "
          f"across {written} treatments; {cache.reads} attachment reads")
    if failures:
        print(f"{sum(failures.values())} treatments skipped:")
        for reason, count in failures.most_common(10):
            print(f"  {count:6d}  {reason}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
