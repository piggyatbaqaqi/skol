#!/usr/bin/env python3
"""Annotate article plaintext with entity spans and save as article.spans.json.

Produces Layer 2 entity-span annotations to complement the Layer 1 YEDDA
section annotations.  For each document:

1. Fetch ``article.txt`` from CouchDB.
2. Fetch ``article.txt.ann`` and parse section-label → char-range mapping.
3. Run **gnfinder** on the full text → TaxonName spans.
4. For each TaxonName, run **gnparser** on the 80-char window after the name
   → Author spans.
5. For each YEDDA section passage, run **particle_detector** with the section
   label → DOI, MB-number, Page-ref, GBIF-ID, Fungarium-code spans.
6. Resolve overlapping spans (keep shorter / higher-confidence).
7. Serialise to JSON and write ``article.spans.json`` as a CouchDB attachment.

Usage::

    python bin/annotate_spans.py --experiment NAME [options]
    python bin/annotate_spans.py --doc-id DOC_ID --database DB [options]

Examples::

    python bin/annotate_spans.py --experiment taxpub_v1_onnx_int8 --skip-existing
    python bin/annotate_spans.py --doc-id abc123 --database skol_dev --dry-run -v
    python bin/annotate_spans.py --experiment NAME --source gnfinder --limit 50
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import couchdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config
from ingestors.gnfinder_client import NameSpan, find_names
from ingestors.gnparser_client import parse_authorship_after_name
from ingestors.particle_detector import detect_particles
from ingestors.spans import Span, resolve_conflicts, spans_to_json

_ATTACHMENT_NAME = "article.spans.json"
_PLAINTEXT_ATTACHMENT = "article.txt"
_ANN_ATTACHMENT = "article.txt.ann"
_AUTHOR_WINDOW = 80  # chars after name end to pass to gnparser

# YEDDA block pattern: [@text#Tag*]
_YEDDA_RE = re.compile(r'\[@(.*?)#([\w-]+)\*\]', re.DOTALL)


# ---------------------------------------------------------------------------
# YEDDA parsing
# ---------------------------------------------------------------------------


def parse_yedda_sections(
    ann_text: str,
    plaintext: str,
) -> List[Tuple[str, int, int]]:
    """Extract section passages and their character offsets from YEDDA annotation.

    Each ``[@text#Label*]`` block in *ann_text* is located in *plaintext*
    and recorded as ``(label, start, end)``.  Blocks whose text is not found
    in *plaintext* are skipped.

    Args:
        ann_text: Contents of ``article.txt.ann``.
        plaintext: Contents of ``article.txt``.

    Returns:
        List of ``(label, start, end)`` tuples in document order, where
        *start* and *end* are character offsets into *plaintext*.
    """
    sections: List[Tuple[str, int, int]] = []
    search_from = 0
    for m in _YEDDA_RE.finditer(ann_text):
        block_text = m.group(1)
        label = m.group(2)
        idx = plaintext.find(block_text, search_from)
        if idx == -1:
            # Try from the start (in case ordering differs)
            idx = plaintext.find(block_text)
        if idx == -1:
            continue
        end = idx + len(block_text)
        sections.append((label, idx, end))
        search_from = end
    return sections


# ---------------------------------------------------------------------------
# Span production
# ---------------------------------------------------------------------------


def _name_spans_to_spans(
    name_spans: List[NameSpan],
    source: str = "all",
) -> List[Span]:
    """Convert gnfinder NameSpan objects to Layer-2 Span objects."""
    result: List[Span] = []
    if source not in ("all", "gnfinder"):
        return result
    for ns in name_spans:
        confidence = min(1.0, max(0.0, 10 ** ns.odds_log10 / (1 + 10 ** ns.odds_log10)
                                  if ns.odds_log10 < 5 else 0.99))
        meta: Dict[str, Any] = {
            "canonical": ns.canonical,
            "cardinality": ns.cardinality,
        }
        if ns.annot_nomen:
            meta["annot_nomen"] = ns.annot_nomen
            meta["annot_nomen_type"] = ns.annot_nomen_type
        result.append(
            Span(
                start=ns.start,
                end=ns.end,
                label="TaxonName",
                text=ns.verbatim,
                source="gnfinder",
                confidence=confidence,
                metadata=meta,
            )
        )
    return result


def annotate_document(
    plaintext: str,
    ann_text: Optional[str],
    doc_id: str,
    source: str = "all",
    gnfinder_url: str = "https://finder.globalnames.org/api/v1/find",
    gnparser_url: str = "https://parser.globalnames.org/api/v1",
    verbosity: int = 0,
) -> List[Span]:
    """Produce all entity spans for a single document.

    Args:
        plaintext: Contents of ``article.txt``.
        ann_text: Contents of ``article.txt.ann`` (may be ``None`` if absent).
        doc_id: CouchDB document ``_id`` (used only for logging).
        source: Which detectors to run: ``"all"``, ``"gnfinder"``, or ``"regex"``.
        gnfinder_url: gnfinder API endpoint.
        gnparser_url: gnparser API endpoint.
        verbosity: Logging verbosity level.

    Returns:
        Conflict-resolved list of :class:`Span` objects.
    """
    all_spans: List[Span] = []

    # ── 1. gnfinder: taxon names across the full text ──────────────────────
    if source in ("all", "gnfinder"):
        if verbosity >= 2:
            print(f"  gnfinder: scanning {len(plaintext)} chars …")
        try:
            name_spans = find_names(plaintext, gnfinder_url=gnfinder_url)
            taxon_spans = _name_spans_to_spans(name_spans, source=source)
            all_spans.extend(taxon_spans)
            if verbosity >= 2:
                print(f"  gnfinder: {len(taxon_spans)} TaxonName spans")

            # ── 2. gnparser: authorship after each taxon name ───────────────
            for ns in name_spans:
                window = plaintext[ns.end: ns.end + _AUTHOR_WINDOW]
                if not window.strip():
                    continue
                try:
                    auth = parse_authorship_after_name(
                        window, gnparser_url=gnparser_url
                    )
                    if auth and auth.verbatim:
                        abs_start = ns.end + auth.offset_in_window
                        abs_end = abs_start + auth.length
                        if abs_end <= len(plaintext):
                            all_spans.append(
                                Span(
                                    start=abs_start,
                                    end=abs_end,
                                    label="Author",
                                    text=auth.verbatim,
                                    source="gnparser",
                                    metadata={
                                        "year": auth.year,
                                        "authors": auth.authors,
                                    },
                                )
                            )
                except Exception as exc:
                    if verbosity >= 2:
                        print(f"  gnparser warning for {ns.canonical}: {exc}")
        except Exception as exc:
            if verbosity >= 1:
                print(f"  gnfinder error for {doc_id}: {exc}", file=sys.stderr)

    # ── 3. particle_detector: per-section for context-aware confidence ──────
    if source in ("all", "regex"):
        sections = parse_yedda_sections(ann_text or "", plaintext) if ann_text else []

        if sections:
            for label, sec_start, sec_end in sections:
                passage = plaintext[sec_start:sec_end]
                particle_spans = detect_particles(
                    passage,
                    redis_client=None,
                    section_label=label,
                )
                # Adjust offsets to absolute
                for span in particle_spans:
                    all_spans.append(
                        Span(
                            start=sec_start + span.start,
                            end=sec_start + span.end,
                            label=span.label,
                            text=span.text,
                            source=span.source,
                            confidence=span.confidence,
                            metadata=span.metadata,
                        )
                    )
            if verbosity >= 2:
                particle_count = sum(
                    1 for s in all_spans if s.source == "regex"
                )
                print(f"  particle_detector: {particle_count} spans across "
                      f"{len(sections)} sections")
        else:
            # No annotation — run on full text without section context
            particle_spans = detect_particles(plaintext, redis_client=None)
            all_spans.extend(particle_spans)
            if verbosity >= 2:
                print(f"  particle_detector: {len(particle_spans)} spans "
                      f"(no section context)")

    return resolve_conflicts(all_spans)


# ---------------------------------------------------------------------------
# CouchDB helpers
# ---------------------------------------------------------------------------


def _iter_doc_ids(
    db: Any,
    limit: Optional[int] = None,
) -> Iterator[str]:
    """Yield all document IDs from *db*, optionally capped at *limit*."""
    count = 0
    for row in db.view("_all_docs"):
        if str(row.id).startswith("_"):
            continue
        yield str(row.id)
        count += 1
        if limit is not None and count >= limit:
            break


def _get_text_attachment(db: Any, doc_id: str, name: str) -> Optional[str]:
    """Fetch a text attachment from *db* as a UTF-8 string."""
    try:
        raw = db.get_attachment(doc_id, name)
        if raw is None:
            return None
        content = raw.read()
        return content.decode("utf-8")
    except Exception:
        return None


def _spans_attachment_exists(db: Any, doc_id: str) -> bool:
    """Return True if ``article.spans.json`` is already attached."""
    try:
        doc = db[doc_id]
        return _ATTACHMENT_NAME in (doc.get("_attachments") or {})
    except Exception:
        return False


def _save_spans(
    db: Any,
    doc_id: str,
    json_str: str,
    verbosity: int = 0,
) -> None:
    """Write *json_str* as ``article.spans.json`` on *doc_id*."""
    doc = db[doc_id]
    db.put_attachment(
        doc,
        json_str.encode("utf-8"),
        filename=_ATTACHMENT_NAME,
        content_type="application/json",
    )
    if verbosity >= 1:
        print(f"  Saved {_ATTACHMENT_NAME} → {doc_id}")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_documents(
    db: Any,
    doc_ids: List[str],
    source: str = "all",
    skip_existing: bool = False,
    dry_run: bool = False,
    gnfinder_url: str = "https://finder.globalnames.org/api/v1/find",
    gnparser_url: str = "https://parser.globalnames.org/api/v1",
    verbosity: int = 0,
) -> Dict[str, int]:
    """Run the span annotation pipeline over a list of document IDs.

    Args:
        db: CouchDB database instance.
        doc_ids: Document IDs to process.
        source: Detector selection: ``"all"``, ``"gnfinder"``, or ``"regex"``.
        skip_existing: Skip documents that already have ``article.spans.json``.
        dry_run: If True, produce spans but do not write to CouchDB.
        gnfinder_url: gnfinder API endpoint.
        gnparser_url: gnparser API endpoint.
        verbosity: Logging verbosity level.

    Returns:
        Dict with counts: ``{"processed": N, "skipped": N, "errors": N}``.
    """
    counts: Dict[str, int] = {"processed": 0, "skipped": 0, "errors": 0}

    for doc_id in doc_ids:
        if verbosity >= 2:
            print(f"\n── {doc_id} ──")

        if skip_existing and _spans_attachment_exists(db, doc_id):
            if verbosity >= 2:
                print(f"  skip (spans already present)")
            counts["skipped"] += 1
            continue

        plaintext = _get_text_attachment(db, doc_id, _PLAINTEXT_ATTACHMENT)
        if not plaintext:
            if verbosity >= 2:
                print(f"  skip (no {_PLAINTEXT_ATTACHMENT})")
            counts["skipped"] += 1
            continue

        ann_text = _get_text_attachment(db, doc_id, _ANN_ATTACHMENT)

        try:
            spans = annotate_document(
                plaintext=plaintext,
                ann_text=ann_text,
                doc_id=doc_id,
                source=source,
                gnfinder_url=gnfinder_url,
                gnparser_url=gnparser_url,
                verbosity=verbosity,
            )
        except Exception as exc:
            if verbosity >= 1:
                print(f"  ERROR annotating {doc_id}: {exc}", file=sys.stderr)
            counts["errors"] += 1
            continue

        json_str = spans_to_json(
            spans,
            doc_id=doc_id,
            source_attachment=_PLAINTEXT_ATTACHMENT,
        )

        if dry_run:
            if verbosity >= 1:
                print(f"  [DRY RUN] would write {len(spans)} spans → "
                      f"{doc_id}/{_ATTACHMENT_NAME}")
            counts["processed"] += 1
            continue

        try:
            _save_spans(db, doc_id, json_str, verbosity=verbosity)
            counts["processed"] += 1
        except Exception as exc:
            if verbosity >= 1:
                print(f"  ERROR saving {doc_id}: {exc}", file=sys.stderr)
            counts["errors"] += 1

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _open_db(config: Dict[str, Any], db_name: str) -> Any:
    """Open a CouchDB database using *config*.

    Credentials are set via ``server.resource.credentials`` rather
    than embedded in the URL — passwords containing ``@`` (or any
    other URL-reserved character) break the embedded-credentials form.
    """
    server = couchdb.Server(config["couchdb_url"])
    username = config.get("couchdb_username")
    password = config.get("couchdb_password")
    if username and password:
        server.resource.credentials = (username, password)
    return server[db_name]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Annotate article plaintext with entity spans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment",
        metavar="NAME",
        default=None,
        help="Experiment name (reads database config from skol_experiments).",
    )
    parser.add_argument(
        "--database",
        metavar="DB",
        default=None,
        help="CouchDB database name (overrides experiment config).",
    )
    parser.add_argument(
        "--doc-id",
        metavar="ID",
        default=None,
        help="Process a single document by ID.",
    )
    parser.add_argument(
        "--source",
        choices=["all", "gnfinder", "regex"],
        default="all",
        help="Which detectors to run (default: all).",
    )
    parser.add_argument(
        "--gnfinder-url",
        metavar="URL",
        default=None,
        help="gnfinder API endpoint URL. "
             "Default: env_config's gnfinder_url "
             "(GNFINDER_URL env var or http://localhost:9080/api/v1/find).",
    )
    parser.add_argument(
        "--gnparser-url",
        metavar="URL",
        default=None,
        help="gnparser API endpoint URL. "
             "Default: env_config's gnparser_url "
             "(GNPARSER_URL env var or http://localhost:9081/api/v1).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents that already have article.spans.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate even if article.spans.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Produce spans but do not write to CouchDB.",
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Process at most N documents.",
    )
    parser.add_argument(
        "-v", "--verbosity",
        action="count",
        default=0,
        help="Increase verbosity (-v info, -vv debug).",
    )
    args = parser.parse_args()

    config = get_env_config()

    # CLAUDE.md rule 11: CLI > env var > config > hardcoded default.
    # env_config has already merged the env var with its hardcoded
    # default; we only override here if the CLI flag was given.
    if args.gnfinder_url is None:
        args.gnfinder_url = config['gnfinder_url']
    if args.gnparser_url is None:
        args.gnparser_url = config['gnparser_url']

    # Resolve database name
    db_name: Optional[str] = args.database
    if db_name is None and args.experiment:
        try:
            exp_db = _open_db(
                config,
                config.get("experiments_database", "skol_experiments"),
            )
            exp_doc = exp_db[args.experiment]
            db_name = (
                exp_doc.get("databases", {}).get("annotations")
                or exp_doc.get("databases", {}).get("ingest")
                or "skol_dev"
            )
        except Exception as exc:
            print(f"✗ Could not load experiment '{args.experiment}': {exc}",
                  file=sys.stderr)
            sys.exit(1)
    if db_name is None:
        db_name = config.get("couchdb_database", "skol_dev")

    try:
        db = _open_db(config, db_name)
    except Exception as exc:
        print(f"✗ Cannot open database '{db_name}': {exc}", file=sys.stderr)
        sys.exit(1)

    if verbosity := args.verbosity:
        print(f"Database : {db_name}")
        print(f"Source   : {args.source}")
        if args.doc_id:
            print(f"Doc ID   : {args.doc_id}")
        if args.dry_run:
            print(f"Mode     : DRY RUN")
        print()

    # Resolve document IDs
    if args.doc_id:
        doc_ids: List[str] = [args.doc_id]
    else:
        doc_ids = list(_iter_doc_ids(db, limit=args.limit))

    skip = args.skip_existing and not args.force

    counts = process_documents(
        db=db,
        doc_ids=doc_ids,
        source=args.source,
        skip_existing=skip,
        dry_run=args.dry_run,
        gnfinder_url=args.gnfinder_url,
        gnparser_url=args.gnparser_url,
        verbosity=args.verbosity,
    )

    if args.verbosity >= 1:
        print(
            f"\nDone: {counts['processed']} processed, "
            f"{counts['skipped']} skipped, "
            f"{counts['errors']} errors."
        )


if __name__ == "__main__":
    main()
