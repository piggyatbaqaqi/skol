#!/usr/bin/env python3
"""v4 Step-1 detector orchestrator.

For every doc in a target CouchDB DB, runs all four Step-1 detectors
and attaches two JSON outputs:

* ``article.spans.v4.json`` — conflict-resolved union of
  gnfinder TaxonName + gnparser Author + particle + section-header
  spans.

* ``article.page-headers.json`` — the JSON-serialisable dict
  ``page_header_detector.detect_page_headers`` returns, fed with
  PDF-page-marker anchors pulled from the particle output so the
  detector benefits from the marker hand-off landed in commit
  c5d8b2c.

Idempotent: a doc is skipped only when **both** v4 attachments are
present (and ``--force`` isn't set).  If a previous run wrote one but
crashed before the other, the next run retries.

Lives alongside ``bin/annotate_spans.py`` (the v3 path) rather than
replacing it — the versioned attachment name keeps v3 outputs intact
for side-by-side comparison.

Usage::

    bin/annotate_v4 --database skol_training_v2_no_golden --limit 3 --dry-run
    bin/annotate_v4 --doc-id 0011a29622c4...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import couchdb  # type: ignore[import]  # noqa: E402

from env_config import get_env_config  # type: ignore[import]  # noqa: E402
from ingestors.extract_plaintext import (  # noqa: E402
    plaintext_from_pdf, plaintext_from_yedda,
)
from ingestors.gnfinder_client import (  # noqa: E402
    NameSpan, find_names,
)
from ingestors.gnparser_client import (  # noqa: E402
    parse_authorship_after_name,
)
from ingestors.page_header_detector import detect_page_headers  # noqa: E402
from ingestors.particle_detector import detect_particles  # noqa: E402
from ingestors.section_header_detector import (  # noqa: E402
    detect_section_headers,
)
from ingestors.spans import (  # noqa: E402
    Span, resolve_conflicts, spans_to_json,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPANS_ATTACHMENT = 'article.spans.v4.json'
_PAGE_HEADERS_ATTACHMENT = 'article.page-headers.json'
_PLAINTEXT_ATTACHMENT = 'article.txt'
_ANN_ATTACHMENT = 'article.txt.ann'
_AUTHOR_WINDOW = 80  # chars after a name end to pass to gnparser


# ---------------------------------------------------------------------------
# CouchDB helpers (duplicated from annotate_spans.py; extract to a
# shared module once a third caller appears).
# ---------------------------------------------------------------------------


def _iter_doc_ids(
    db: Any, limit: Optional[int] = None,
) -> Iterator[str]:
    """Yield doc IDs from ``db``, skipping design docs."""
    count = 0
    for row in db.view('_all_docs'):
        if str(row.id).startswith('_'):
            continue
        yield str(row.id)
        count += 1
        if limit is not None and count >= limit:
            break


def _read_attachment_text(
    db: Any, doc_id: str, name: str,
) -> Optional[str]:
    """Fetch a text attachment as UTF-8 string, or ``None`` if absent."""
    try:
        raw = db.get_attachment(doc_id, name)
        if raw is None:
            return None
        if hasattr(raw, 'read'):
            raw = raw.read()
        if isinstance(raw, bytes):
            return raw.decode('utf-8', errors='ignore')
        return str(raw)
    except Exception:  # noqa: BLE001
        return None


def _read_attachment_bytes(
    db: Any, doc_id: str, name: str,
) -> Optional[bytes]:
    """Fetch an attachment as raw bytes, or ``None`` if absent."""
    try:
        raw = db.get_attachment(doc_id, name)
        if raw is None:
            return None
        if hasattr(raw, 'read'):
            raw = raw.read()
        if isinstance(raw, str):
            return raw.encode('utf-8')
        return raw  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return None


def _load_plaintext_v4(
    db: Any, doc_id: str,
) -> Tuple[Optional[str], str]:
    """3-path plaintext fallback (same pattern as bin/embed_lines.py).

    1. ``article.txt`` (canonical plaintext, when present).
    2. ``article.pdf`` -> :func:`plaintext_from_pdf`.  Covers
       hand-annotated PDF docs that lack a bare ``article.txt``.
    3. ``article.txt.ann`` -> :func:`plaintext_from_yedda`.  Covers
       the JATS-derived majority of ``skol_training_v3_combined_no_golden``
       which only carry the YEDDA annotation file.

    Returns ``(text, source_tag)`` where ``source_tag`` is one of
    ``'article.txt'`` / ``'article.pdf'`` / ``'article.txt.ann'`` /
    ``'missing'``.
    """
    text = _read_attachment_text(db, doc_id, _PLAINTEXT_ATTACHMENT)
    if text is not None:
        return text, _PLAINTEXT_ATTACHMENT

    pdf_bytes = _read_attachment_bytes(db, doc_id, 'article.pdf')
    if pdf_bytes is not None:
        return plaintext_from_pdf(pdf_bytes), 'article.pdf'

    ann_text = _read_attachment_text(db, doc_id, _ANN_ATTACHMENT)
    if ann_text is not None:
        return plaintext_from_yedda(ann_text), _ANN_ATTACHMENT

    return None, 'missing'


def _xml_attachments_present(
    doc: Dict[str, Any],
) -> List[str]:
    """Return a list of ``*.xml`` attachment names on ``doc``.

    Used in the "no plaintext source" diagnostic so operators
    can tell at a glance whether a doc that can't be processed
    is truly orphan (no attachments at all) or just carries
    its content in a format (JATS XML, etc.) the v4 pipeline
    doesn't yet read directly.  We match on suffix because the
    JATS attachment name varies across ingestors (``article.jats.xml``
    from PMC, ``article.xml`` from Pensoft, ``jats.xml`` for some
    hand-imported corpora).
    """
    atts = (doc.get('_attachments') or {}) if isinstance(doc, dict) else {}
    return sorted(
        name for name in atts.keys() if name.lower().endswith('.xml')
    )


def _both_attachments_present(db: Any, doc_id: str) -> bool:
    """True iff doc has BOTH v4 attachments already."""
    try:
        doc = db[doc_id]
    except Exception:  # noqa: BLE001
        return False
    atts = doc.get('_attachments') or {}
    return (
        _SPANS_ATTACHMENT in atts
        and _PAGE_HEADERS_ATTACHMENT in atts
    )


def _open_db(config: Dict[str, Any], db_name: str) -> Any:
    """Open a CouchDB database using ``config`` credentials.

    Credentials are set via ``server.resource.credentials`` rather
    than embedded in the URL — passwords containing ``@`` would
    otherwise break the URL parser.
    """
    server = couchdb.Server(config['couchdb_url'])
    username = config.get('couchdb_username')
    password = config.get('couchdb_password')
    if username and password:
        server.resource.credentials = (username, password)
    return server[db_name]


def _save_attachment(
    db: Any, doc_id: str, name: str,
    body_bytes: bytes, content_type: str = 'application/json',
) -> None:
    """Write ``body_bytes`` as the named attachment on ``doc_id``."""
    doc = db[doc_id]
    db.put_attachment(
        doc, body_bytes,
        filename=name,
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# Particle / marker bridge
# ---------------------------------------------------------------------------


def _markers_from_spans(
    spans: List[Span], plaintext: str,
) -> List[Tuple[int, int]]:
    """Pull ``(line_index, page_number)`` tuples from PDF-page-marker
    particle spans.

    line_index follows the same convention used by
    ``tests/test_page_header_golden.py``:
    ``plaintext.count('\\n', 0, span.start)``.
    """
    markers: List[Tuple[int, int]] = []
    for span in spans:
        if span.label != 'PDF-page-marker':
            continue
        page_number = span.metadata.get('page_number')
        if page_number is None:
            continue
        line_index = plaintext.count('\n', 0, span.start)
        markers.append((line_index, int(page_number)))
    return markers


# ---------------------------------------------------------------------------
# Span production helpers (gnfinder)
# ---------------------------------------------------------------------------


def _name_spans_to_spans(name_spans: List[NameSpan]) -> List[Span]:
    """Convert gnfinder NameSpan objects to Layer-2 Span objects."""
    result: List[Span] = []
    for ns in name_spans:
        confidence = min(
            1.0,
            max(
                0.0,
                10 ** ns.odds_log10 / (1 + 10 ** ns.odds_log10)
                if ns.odds_log10 < 5 else 0.99,
            ),
        )
        meta: Dict[str, Any] = {
            'canonical': ns.canonical,
            'cardinality': ns.cardinality,
        }
        if ns.annot_nomen:
            meta['annot_nomen'] = ns.annot_nomen
            meta['annot_nomen_type'] = ns.annot_nomen_type
        result.append(Span(
            start=ns.start, end=ns.end,
            label='TaxonName', text=ns.verbatim,
            source='gnfinder', confidence=confidence,
            metadata=meta,
        ))
    return result


# ---------------------------------------------------------------------------
# Per-document orchestration
# ---------------------------------------------------------------------------


def annotate_document_v4(
    plaintext: str,
    ann_text: Optional[str],
    doc_id: str,
    *,
    gnfinder_url: str,
    gnparser_url: str,
    verbosity: int = 0,
) -> Tuple[List[Span], Dict[str, Any]]:
    """Run all four Step-1 detectors on ``plaintext``.

    Returns ``(spans, page_headers_dict)`` where ``spans`` is the
    conflict-resolved union of TaxonName + Author + particle +
    section-header spans, and ``page_headers_dict`` is what
    ``detect_page_headers`` returned (fed PDF-page-marker anchors
    pulled from the particle output).
    """
    all_spans: List[Span] = []

    # 1. gnfinder + gnparser
    try:
        name_spans = find_names(plaintext, gnfinder_url=gnfinder_url)
        taxon_spans = _name_spans_to_spans(name_spans)
        all_spans.extend(taxon_spans)
        if verbosity >= 2:
            print(f'  gnfinder: {len(taxon_spans)} TaxonName spans')

        for ns in name_spans:
            window = plaintext[ns.end: ns.end + _AUTHOR_WINDOW]
            if not window.strip():
                continue
            try:
                auth = parse_authorship_after_name(
                    window, gnparser_url=gnparser_url,
                )
                if auth and auth.verbatim:
                    abs_start = ns.end + auth.offset_in_window
                    abs_end = abs_start + auth.length
                    if abs_end <= len(plaintext):
                        all_spans.append(Span(
                            start=abs_start, end=abs_end,
                            label='Author', text=auth.verbatim,
                            source='gnparser',
                            metadata={
                                'year': auth.year,
                                'authors': auth.authors,
                            },
                        ))
            except Exception as exc:  # noqa: BLE001
                if verbosity >= 2:
                    print(f'  gnparser warning for {ns.canonical}: {exc}')
    except Exception as exc:  # noqa: BLE001
        if verbosity >= 1:
            print(f'  gnfinder error for {doc_id}: {exc}',
                  file=sys.stderr)

    # 2. particle_detector
    particle_spans = detect_particles(plaintext, redis_client=None)
    all_spans.extend(particle_spans)
    if verbosity >= 2:
        print(f'  particle_detector: {len(particle_spans)} spans')

    # 3. section_header_detector
    section_spans = detect_section_headers(plaintext)
    all_spans.extend(section_spans)
    if verbosity >= 2:
        print(f'  section_header_detector: {len(section_spans)} spans')

    resolved = resolve_conflicts(all_spans)

    # 4. page_header_detector with PDF-page-marker anchors
    lines = plaintext.split('\n')
    markers = _markers_from_spans(particle_spans, plaintext)
    page_headers = detect_page_headers(
        lines, seed=42, pdf_page_markers=markers,
    )
    if verbosity >= 2:
        n_regions = len(page_headers.get('regions', []))
        print(f'  page_header_detector: {n_regions} regions '
              f'({len(markers)} marker anchors)')

    return resolved, page_headers


# ---------------------------------------------------------------------------
# DB-iteration loop
# ---------------------------------------------------------------------------


def process_documents_v4(
    db: Any,
    doc_ids: List[str],
    *,
    skip_existing: bool,
    force: bool,
    dry_run: bool,
    gnfinder_url: str,
    gnparser_url: str,
    verbosity: int = 0,
) -> Dict[str, int]:
    """Per-doc loop.

    Returns counts: ``{processed, skipped, errors}``.  ``skip_existing``
    only skips when BOTH v4 attachments are present; partial outputs
    from a crashed previous run are retried.  ``force`` overrides.
    """
    counts: Dict[str, int] = {
        'processed': 0, 'skipped': 0, 'errors': 0,
    }
    for doc_id in doc_ids:
        if (
            skip_existing and not force
            and _both_attachments_present(db, doc_id)
        ):
            counts['skipped'] += 1
            if verbosity >= 2:
                print(f'  skip {doc_id}: both v4 attachments present')
            continue

        plaintext, plaintext_source = _load_plaintext_v4(db, doc_id)
        if plaintext is None:
            counts['errors'] += 1
            if verbosity >= 1:
                try:
                    doc = db[doc_id]
                except Exception:  # noqa: BLE001
                    doc = {}
                xml_present = _xml_attachments_present(doc)
                msg = (
                    f'  ✗ {doc_id}: no plaintext source '
                    f'({_PLAINTEXT_ATTACHMENT}, article.pdf, '
                    f'{_ANN_ATTACHMENT} all absent)'
                )
                if xml_present:
                    # Operator hint: doc isn't orphan — it has JATS /
                    # other XML content the v4 pipeline doesn't yet
                    # consume directly.
                    msg += (
                        f' — but XML attachment present: '
                        f'{", ".join(xml_present)}'
                    )
                print(msg)
            continue
        ann_text = _read_attachment_text(
            db, doc_id, _ANN_ATTACHMENT,
        )
        if verbosity >= 2:
            print(f'  {doc_id}: plaintext from {plaintext_source}')

        try:
            spans, page_headers = annotate_document_v4(
                plaintext, ann_text, doc_id,
                gnfinder_url=gnfinder_url,
                gnparser_url=gnparser_url,
                verbosity=verbosity,
            )
        except Exception as exc:  # noqa: BLE001
            counts['errors'] += 1
            if verbosity >= 1:
                print(f'  ✗ {doc_id}: annotate failed: {exc}',
                      file=sys.stderr)
            continue

        if dry_run:
            counts['processed'] += 1
            if verbosity >= 1:
                tag = '(DRY RUN) '
                print(
                    f'  {tag}{doc_id}: '
                    f'{len(spans)} spans, '
                    f'{len(page_headers.get("regions", []))} page-header '
                    'regions'
                )
            continue

        spans_json = spans_to_json(
            spans, doc_id=doc_id,
            source_attachment=_PLAINTEXT_ATTACHMENT,
        )
        page_headers_json = json.dumps(page_headers)
        try:
            _save_attachment(
                db, doc_id, _SPANS_ATTACHMENT,
                spans_json.encode('utf-8'),
            )
            _save_attachment(
                db, doc_id, _PAGE_HEADERS_ATTACHMENT,
                page_headers_json.encode('utf-8'),
            )
        except Exception as exc:  # noqa: BLE001
            counts['errors'] += 1
            if verbosity >= 1:
                print(f'  ✗ {doc_id}: save failed: {exc}',
                      file=sys.stderr)
            continue

        counts['processed'] += 1
        if verbosity >= 1:
            print(
                f'  ✓ {doc_id}: '
                f'{len(spans)} spans, '
                f'{len(page_headers.get("regions", []))} page-header '
                'regions'
            )

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description='v4 Step-1 detector orchestrator.',
    )
    parser.add_argument(
        '--database', default=None,
        help='CouchDB database (default: env_config couchdb_database).',
    )
    parser.add_argument(
        '--doc-id', dest='doc_id', default=None,
        help='Process only this doc instead of iterating the DB.',
    )
    parser.add_argument(
        '--gnfinder-url', default=None,
        help='gnfinder API URL (default: env_config gnfinder_url).',
    )
    parser.add_argument(
        '--gnparser-url', default=None,
        help='gnparser API URL (default: env_config gnparser_url).',
    )
    args, _ = parser.parse_known_args()

    config = get_env_config()
    verbosity = int(config.get('verbosity', 1) or 0)
    dry_run = bool(config.get('dry_run', False))
    force = bool(config.get('force', False))
    skip_existing = not force
    limit_raw = config.get('limit')
    limit = (
        int(limit_raw) if limit_raw not in (None, '') else None
    )

    db_name = (
        args.database or config.get('couchdb_database', 'skol_dev')
    )
    gnfinder_url = (
        args.gnfinder_url or config['gnfinder_url']
    )
    gnparser_url = (
        args.gnparser_url or config['gnparser_url']
    )

    try:
        db = _open_db(config, db_name)
    except Exception as exc:  # noqa: BLE001
        print(f'✗ cannot open {db_name!r}: {exc}', file=sys.stderr)
        return 1

    if verbosity >= 1:
        print(f'annotate_v4 — db={db_name}')
        print(f'  gnfinder_url={gnfinder_url}')
        print(f'  gnparser_url={gnparser_url}')
        if dry_run:
            print('  *** DRY RUN — no attachments written ***')

    if args.doc_id:
        doc_ids = [args.doc_id]
    else:
        doc_ids = list(_iter_doc_ids(db, limit=limit))

    counts = process_documents_v4(
        db, doc_ids,
        skip_existing=skip_existing,
        force=force, dry_run=dry_run,
        gnfinder_url=gnfinder_url,
        gnparser_url=gnparser_url,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print()
        print(f'  processed : {counts["processed"]}')
        print(f'  skipped   : {counts["skipped"]}')
        print(f'  errors    : {counts["errors"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
