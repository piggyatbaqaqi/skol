"""Lookup helpers for the mykoweb PDF → journal/book metadata JSON
produced by ``bin/extract_mykoweb_pdf_metadata.py``.

Consumed by ``LocalMykowebLiteratureIngestor`` to enrich each
ingested literature PDF with curated bibliographic fields (journal,
volume, issue, pages, author, year, title).  Missing JSON / missing
entries are tolerated — callers fall back to filename-as-title /
itemtype='book' behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_metadata_index(
    path: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    """Load the metadata JSON into a dict keyed by site-relative PDF
    path.

    Returns an empty dict if ``path`` is None or the file doesn't
    exist — the ingestor degrades gracefully to its pre-integration
    behaviour rather than crashing.
    """
    if path is None:
        return {}
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def lookup_pdf_metadata(
    full_filepath: str,
    metadata_index: Dict[str, Dict[str, Any]],
    site_root: str,
) -> Optional[Dict[str, Any]]:
    """Look up a PDF's metadata record by its filesystem path.

    The JSON keys are site-relative (``systematics/literature/...``);
    this strips ``site_root`` off ``full_filepath`` to construct the
    lookup key.  Returns None if the path is outside ``site_root`` or
    the resulting key isn't in the index.
    """
    root = site_root.rstrip('/')
    if not full_filepath.startswith(root + '/'):
        return None
    relpath = full_filepath[len(root):].lstrip('/')
    return metadata_index.get(relpath)


def metadata_to_doc_fields(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate a metadata record into fields to merge into the
    ingestor's outgoing ``doc_dict``.

    ``container_title`` propagates to ``journal`` regardless of
    ``kind`` so collection-roll-up records (CAF protologue,
    Funga Nordica chapters, etc.) surface on the Sources page
    under their parent collection.  ``kind=journal`` is the
    journals ingestor's domain — return empty so this path
    doesn't accidentally rewrite a journal-volume doc.

    Empty / None values are dropped so the doc doesn't carry
    spurious null fields.
    """
    kind = record.get('kind')
    if kind == 'journal':
        return {}
    itemtype_for_kind = {
        'journal_article': 'article',
        'book':             'book',
        'key':              'key',
        'misc':              'misc',
    }
    itemtype = itemtype_for_kind.get(kind)
    if itemtype is None:
        return {}

    fields: Dict[str, Any] = {'itemtype': itemtype}

    container = record.get('container_title')
    if container:
        fields['journal'] = container

    # Volume / issue / pages are meaningful only for actual journal
    # articles — a CAF book carrying ``container='California Fungi'``
    # gets the journal field but no spurious vol/pages.
    if kind == 'journal_article':
        for k in ('volume', 'issue', 'pages'):
            value = record.get(k)
            if value:
                fields[k] = value

    # Title / author / year apply to every kind.
    for k in ('title', 'author', 'year'):
        value = record.get(k)
        if value:
            fields[k] = value

    return fields
