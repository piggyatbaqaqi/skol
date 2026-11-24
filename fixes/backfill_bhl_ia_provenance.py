#!/usr/bin/env python3
"""Backfill IA scan provenance onto BHL volume documents stored before
BhlIngestor recorded it (commit 07e2b08): the six Lloyd volumes in
skol_dev, 2026-11-23.

Adds ``ia_contributor``, ``ia_scanning_center``, ``ia_scan_date``,
``ia_image_count`` and ``ia_pdf_md5`` from archive.org's metadata API.
For a volume stored whole (no ``withheld_pages``), the stored
article.pdf's CouchDB md5 digest is compared with IA's; mismatches are
reported, not acted on.  Idempotent: documents that already carry any
``ia_*`` field are skipped.  One metadata request per document, rate
limited; no PDF is downloaded.

Usage: fixes/backfill_bhl_ia_provenance.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

IA_FIELDS = ('ia_contributor', 'ia_scanning_center', 'ia_scan_date',
             'ia_image_count', 'ia_pdf_md5')


def needs_provenance(doc: Dict[str, Any]) -> bool:
    """A BHL volume with an IA identifier and no ``ia_*`` field yet."""
    raise NotImplementedError


def attachment_md5_hex(doc: Dict[str, Any],
                       name: str = 'article.pdf') -> Optional[str]:
    """Hex md5 of an attachment from its CouchDB ``md5-<base64>``
    digest, or None if absent or not an md5 digest."""
    raise NotImplementedError


def bhl_docs(db: Any) -> Iterable[Dict[str, Any]]:
    """Every ``source: bhl`` document, by Mango query."""
    raise NotImplementedError


def backfill(
    db: Any,
    fetch: Callable[[str], Dict[str, Any]],
    dry_run: bool = False,
    limit: Optional[int] = None,
    verbosity: int = 1,
) -> Dict[str, int]:
    """
    Fill provenance on every BHL document that needs it.

    ``fetch(identifier)`` returns the ``ia_*`` fields (``{}`` on
    failure).  Returns counts: ``scanned``, ``eligible``, ``written``,
    ``lookup_failed``, ``md5_mismatch``.
    """
    raise NotImplementedError


def main() -> int:
    """CLI entry point."""
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
