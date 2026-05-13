#!/usr/bin/env python3
"""Enrich skol_ann_merged with publication metadata, attachments, and a
golden-membership flag.

For each document in ``skol_ann_merged`` (or only the ones named by
``--doc-id``), this script performs three independent steps; each is
skipped if its result is already present (use ``--force`` to refresh
Crossref metadata):

1.  Copy ``article.txt`` and ``article.pdf`` attachments from
    ``skol_training`` if they're not already on the merged doc.
2.  Set ``is_golden`` on the doc to reflect whether its ``_id`` exists
    in ``skol_golden``.
3.  Extract a DOI from the first page of ``article.txt`` (between the
    first two ``--- PDF Page ... ---`` markers when present), call
    ``https://api.crossref.org/works/{doi}``, and store a shaped subset
    of the response under ``publication_metadata``.

Environment variables (or ~/.skol_env):
    COUCHDB_URL         CouchDB server URL
    COUCHDB_USER        CouchDB username
    COUCHDB_PASSWORD    CouchDB password
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config  # type: ignore[import]  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TXT_ATTACHMENT = "article.txt"
_PDF_ATTACHMENT = "article.pdf"
_ATTACHMENTS_TO_COPY = (_TXT_ATTACHMENT, _PDF_ATTACHMENT)
_PAGE_MARKER_RE = re.compile(
    r"^---\s+PDF\s+Page\s+\d+\s+Label\s+\S+\s+---\s*$",
    re.MULTILINE,
)
# Crossref-recommended DOI pattern, applied case-insensitively.  We deliberately
# avoid greedy matches that swallow trailing sentence punctuation: a trailing
# '.' or ',' (but not all dots — DOIs commonly contain dots) is stripped after
# the match.
_DOI_RE = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:a-z0-9]+)",
    re.IGNORECASE,
)
_FIRST_PAGE_FALLBACK_CHARS = 4000
_CROSSREF_URL = "https://api.crossref.org/works/{}"
_USER_AGENT = "skol-enrich/1.0 (mailto:piggy.yarroll@gmail.com)"
_HTTP_TIMEOUT = 30
_HTTP_RETRIES = 3
_HTTP_BACKOFF = 2.0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _first_page_text(text: str) -> str:
    """Return the text of the first PDF page.

    If PDF page markers are present, take the slice between the first
    marker and the second.  Otherwise return the head of the document
    (capped at ``_FIRST_PAGE_FALLBACK_CHARS``) so we don't search the
    whole document for a DOI — references on later pages would yield
    false positives.
    """
    markers = list(_PAGE_MARKER_RE.finditer(text))
    if not markers:
        return text[:_FIRST_PAGE_FALLBACK_CHARS]
    start = markers[0].end()
    end = markers[1].start() if len(markers) > 1 else len(text)
    return text[start:end]


def _extract_doi(text: str) -> Optional[str]:
    """Return the first DOI found in text, or None."""
    match = _DOI_RE.search(text)
    if not match:
        return None
    doi = match.group(1)
    # Strip terminal sentence punctuation that the regex may have included.
    while doi and doi[-1] in ".,;:)":
        doi = doi[:-1]
    return doi


def _shape_crossref_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Project a Crossref ``message`` object into the fields we keep.

    Crossref's response carries many fields; we extract a compact subset
    that's useful for citing or grouping records and translate snake_case
    where Crossref uses hyphens (which CouchDB views handle awkwardly).
    """
    out: Dict[str, Any] = {}
    if "DOI" in message:
        out["doi"] = message["DOI"]
    titles = message.get("title")
    if isinstance(titles, list) and titles:
        out["title"] = titles[0]
    authors = message.get("author")
    if isinstance(authors, list) and authors:
        out["authors"] = [
            {k: v for k, v in a.items() if k in ("given", "family", "name")}
            for a in authors
        ]
    container = message.get("container-title")
    if isinstance(container, list) and container:
        out["container_title"] = container[0]
    year = _extract_year(message)
    if year is not None:
        out["year"] = year
    for src_key, dst_key in (
        ("volume", "volume"),
        ("issue", "issue"),
        ("page", "page"),
        ("publisher", "publisher"),
        ("type", "type"),
        ("URL", "url"),
    ):
        if src_key in message:
            out[dst_key] = message[src_key]
    if "ISSN" in message:
        out["issn"] = message["ISSN"]
    return out


def _extract_year(message: Dict[str, Any]) -> Optional[int]:
    """Read the publication year from Crossref date-parts fields."""
    for key in ("published-print", "published-online", "issued"):
        block = message.get(key)
        if not isinstance(block, dict):
            continue
        parts = block.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) \
                and parts[0]:
            try:
                return int(parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# Skip-if-present predicates
# ---------------------------------------------------------------------------

def needs_attachment(doc: Dict[str, Any], filename: str) -> bool:
    return filename not in (doc.get("_attachments") or {})


def needs_golden_flag(doc: Dict[str, Any]) -> bool:
    return "is_golden" not in doc


def needs_crossref(doc: Dict[str, Any], force: bool) -> bool:
    if force:
        return True
    return "publication_metadata" not in doc


# ---------------------------------------------------------------------------
# Crossref HTTP
# ---------------------------------------------------------------------------

def fetch_crossref(http: Any, doi: str) -> Optional[Dict[str, Any]]:
    """Fetch Crossref metadata for a DOI, returning the ``message`` body.

    Returns None on a 404 (DOI not registered with Crossref).  Retries
    transient failures up to ``_HTTP_RETRIES`` times.
    """
    url = _CROSSREF_URL.format(doi)
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    last_exc: Optional[Exception] = None
    for attempt in range(1, _HTTP_RETRIES + 1):
        try:
            response = http.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
            if response.status_code == 404:
                logging.warning("Crossref 404 for DOI %s", doi)
                return None
            response.raise_for_status()
            return response.json().get("message")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _HTTP_RETRIES:
                wait = _HTTP_BACKOFF * (2 ** (attempt - 1))
                logging.warning(
                    "Crossref attempt %d for %s failed (%s); retrying in %.0fs",
                    attempt, doi, exc, wait,
                )
                time.sleep(wait)
    raise RuntimeError(
        f"Crossref lookup failed after {_HTTP_RETRIES} attempts for {doi}: "
        f"{last_exc}"
    )


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

def _copy_attachments_if_missing(
    doc: Dict[str, Any],
    doc_id: str,
    merged_db: Any,
    training_db: Any,
) -> List[str]:
    """Copy article.txt and article.pdf from training into merged if missing.

    Returns the list of attachment names actually copied.
    """
    copied: List[str] = []
    for name in _ATTACHMENTS_TO_COPY:
        if not needs_attachment(doc, name):
            continue
        att = training_db.get_attachment(doc_id, name)
        if att is None:
            continue
        data = att.read()
        # Re-fetch doc to get the current rev before each put_attachment.
        try:
            current = merged_db[doc_id]
        except KeyError:
            merged_db.save({"_id": doc_id})
            current = merged_db[doc_id]
        merged_db.put_attachment(
            current, data, filename=name,
            content_type="text/plain" if name.endswith(".txt")
            else "application/pdf",
        )
        copied.append(name)
    return copied


def process_doc(
    doc_id: str,
    merged_db: Any,
    training_db: Any,
    golden_db: Any,
    http: Any,
    force: bool = False,
) -> Dict[str, Any]:
    """Enrich one merged document.

    Returns a small dict describing what changed (used for logging/stats).
    Each step is independent — a failure in one (e.g. Crossref 404) does
    not abort the others.
    """
    try:
        doc = merged_db[doc_id]
    except KeyError:
        merged_db.save({"_id": doc_id})
        doc = merged_db[doc_id]

    summary: Dict[str, Any] = {
        "attachments_copied": [],
        "golden_set": None,
        "doi": None,
        "crossref_updated": False,
    }

    # 1. Copy missing attachments from training.
    summary["attachments_copied"] = _copy_attachments_if_missing(
        doc, doc_id, merged_db, training_db,
    )
    # Re-fetch so subsequent updates use the latest rev.
    doc = merged_db[doc_id]

    # 2. Set is_golden if missing.
    updated = False
    if needs_golden_flag(doc):
        is_golden = doc_id in golden_db
        doc["is_golden"] = is_golden
        summary["golden_set"] = is_golden
        updated = True

    # 3. Crossref lookup.
    if needs_crossref(doc, force):
        txt_att = merged_db.get_attachment(doc_id, _TXT_ATTACHMENT)
        if txt_att is not None:
            try:
                text = txt_att.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                text = ""
            doi = _extract_doi(_first_page_text(text))
            summary["doi"] = doi
            if doi:
                message = fetch_crossref(http, doi)
                if message:
                    doc["publication_metadata"] = _shape_crossref_message(
                        message
                    )
                    summary["crossref_updated"] = True
                    updated = True

    if updated:
        # Drop _attachments stub so save() doesn't try to round-trip it.
        doc.pop("_attachments", None)
        merged_db.save(doc)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _connect(database: str) -> Any:
    import couchdb as couchdb_lib

    config = get_env_config()
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server[database]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--merged-db",
        default="skol_ann_merged",
        help="Target database to enrich (default: skol_ann_merged).",
    )
    parser.add_argument(
        "--training-db",
        default="skol_training",
        help="Source for attachments (default: skol_training).",
    )
    parser.add_argument(
        "--golden-db",
        default="skol_golden",
        help="Database whose membership sets is_golden (default: skol_golden).",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        metavar="ID",
        help="Process only this document ID (repeatable).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch Crossref metadata even when publication_metadata "
             "is already present.",
    )
    parser.add_argument(
        "-v", "--verbosity",
        action="count",
        default=1,
        help="Increase output verbosity (repeatable).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.verbosity < 2 else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    merged_db = _connect(args.merged_db)
    training_db = _connect(args.training_db)
    golden_db = _connect(args.golden_db)

    if args.doc_ids:
        ids = list(args.doc_ids)
    else:
        ids = [
            row.id
            for row in merged_db.view("_all_docs", include_docs=False)
            if not row.id.startswith("_design/")
        ]

    http = requests.Session()
    processed = changed = failed = 0
    for doc_id in ids:
        try:
            summary = process_doc(
                doc_id, merged_db, training_db, golden_db, http,
                force=args.force,
            )
            processed += 1
            if (summary["attachments_copied"]
                    or summary["golden_set"] is not None
                    or summary["crossref_updated"]):
                changed += 1
            if args.verbosity >= 1:
                bits = []
                if summary["attachments_copied"]:
                    bits.append(
                        "att=" + ",".join(summary["attachments_copied"])
                    )
                if summary["golden_set"] is not None:
                    bits.append(f"golden={summary['golden_set']}")
                if summary["doi"]:
                    bits.append(f"doi={summary['doi']}")
                    if summary["crossref_updated"]:
                        bits.append("crossref=ok")
                if bits:
                    print(f"  {doc_id}: {' '.join(bits)}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logging.error("%s: enrichment failed: %s", doc_id, exc)

    print(
        f"Done: {processed} processed, {changed} changed, {failed} failed"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
