#!/usr/bin/env python3
"""Curate a golden evaluation dataset from hand-annotated and JATS sources.

Selects articles from skol_training (hand-annotated) and skol_dev (JATS),
obtains plaintext, and populates golden databases:

- skol_golden:          Union of all selected articles + article.txt
- skol_golden_ann_hand: Hand-annotated article.txt.ann from skol_training
- skol_golden_ann_jats: JATS-derived article.txt.ann
- skol_golden_ann_bioc: BioC-derived article.txt.ann

Examples:
    # Dry run to see what would be selected
    python curate_golden_dataset.py --dry-run

    # Curate with defaults (min-tags=4, jats-limit=75)
    python curate_golden_dataset.py --all

    # Custom selection parameters
    python curate_golden_dataset.py --all --min-tags 3 --jats-limit 50
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow running as a script or as a module.
if __name__ == "__main__" and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    bin_dir = str(Path(__file__).resolve().parent)
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)

from env_config import get_env_config
from ingestors.extract_plaintext import (
    count_yedda_tags,
    plaintext_from_bioc,
    plaintext_from_efetch,
    plaintext_from_jats,
    plaintext_from_pdf,
    plaintext_from_yedda,
)
from ingestors.jats_to_yedda import jats_xml_to_yedda
from ingestors.bioc_to_yedda import bioc_json_to_yedda


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _connect_server(config: Dict[str, Any]):
    """Connect to CouchDB server."""
    import couchdb as couchdb_lib
    server = couchdb_lib.Server(config["couchdb_url"])
    server.resource.credentials = (
        config["couchdb_username"],
        config["couchdb_password"],
    )
    return server


def _get_or_create_db(server, name: str):
    """Get an existing database or create it."""
    if name in server:
        return server[name]
    return server.create(name)


# ---------------------------------------------------------------------------
# Step 1: Select hand-annotated documents from skol_training
# ---------------------------------------------------------------------------

def select_training_docs(
    server,
    training_db_name: str,
    min_tags: int,
    verbosity: int,
) -> List[Dict[str, Any]]:
    """Select hand-annotated training docs with sufficient tag diversity.

    Args:
        server: CouchDB server connection.
        training_db_name: Name of the training database.
        min_tags: Minimum number of distinct tag types required.
        verbosity: Logging verbosity.

    Returns:
        List of selected document dicts (with metadata, not attachments).
    """
    db = server[training_db_name]
    selected: List[Dict[str, Any]] = []
    skipped = 0

    for row in db.view("_all_docs", include_docs=True):
        if row.id.startswith("_design/"):
            continue

        doc = row.doc
        # Read the YEDDA annotation to count tags
        att = db.get_attachment(row.id, "article.txt.ann")
        if att is None:
            if verbosity >= 2:
                print(f"  {row.id}: no article.txt.ann, skipping",
                      file=sys.stderr)
            skipped += 1
            continue

        yedda_text = att.read().decode("utf-8")
        tags, block_count = count_yedda_tags(yedda_text)

        if len(tags) < min_tags:
            if verbosity >= 2:
                print(
                    f"  {row.id}: only {len(tags)} tags "
                    f"({', '.join(sorted(tags))}), need {min_tags}",
                    file=sys.stderr,
                )
            skipped += 1
            continue

        selected.append({
            "doc_id": row.id,
            "doc": doc,
            "yedda_text": yedda_text,
            "tags": tags,
            "block_count": block_count,
        })

    if verbosity >= 1:
        print(
            f"Training: {len(selected)} selected, {skipped} skipped "
            f"(min_tags={min_tags})",
            file=sys.stderr,
        )

    return selected


def subsample_training_docs(
    selections: List[Dict[str, Any]],
    hand_limit: int,
    verbosity: int,
) -> List[Dict[str, Any]]:
    """Subsample training docs for the golden holdout set.

    Uses stratified selection: sorts by tag count (descending) then doc_id,
    and picks evenly-spaced documents so the holdout covers the full range
    of annotation complexity.

    Args:
        selections: All qualifying training doc infos.
        hand_limit: Maximum number to retain for golden dataset.
        verbosity: Logging verbosity.

    Returns:
        Subsampled list of doc infos.
    """
    if len(selections) <= hand_limit:
        return selections

    # Sort by tag diversity (descending), then block count (descending),
    # then doc_id for determinism.
    sorted_sels = sorted(
        selections,
        key=lambda s: (-len(s["tags"]), -s["block_count"], s["doc_id"]),
    )

    # Pick evenly spaced indices through the sorted list.
    n = len(sorted_sels)
    step = n / hand_limit
    sampled = [sorted_sels[int(i * step)] for i in range(hand_limit)]

    if verbosity >= 1:
        tag_counts = [len(s["tags"]) for s in sampled]
        block_counts = [s["block_count"] for s in sampled]
        print(
            f"Hand-annotated holdout: {len(sampled)} of {n} "
            f"(tag types: {min(tag_counts)}-{max(tag_counts)}, "
            f"blocks: {min(block_counts)}-{max(block_counts)})",
            file=sys.stderr,
        )

    return sampled


# ---------------------------------------------------------------------------
# Step 2: Obtain plaintext for training documents
# ---------------------------------------------------------------------------

def obtain_plaintext(
    server,
    training_db_name: str,
    dev_db_name: str,
    doc_info: Dict[str, Any],
    config: Dict[str, Any],
    verbosity: int,
) -> Tuple[str, str]:
    """Obtain plaintext for a training document.

    Priority: article.txt in training > article.pdf > JATS > BioC > efetch > YEDDA

    Args:
        server: CouchDB server connection.
        training_db_name: Training database name.
        dev_db_name: Dev database name (to check for existing article.txt).
        doc_info: Dict with doc_id, doc, yedda_text.
        config: Environment config.
        verbosity: Logging verbosity.

    Returns:
        Tuple of (plaintext, source_name).
    """
    doc_id = doc_info["doc_id"]
    doc = doc_info["doc"]
    training_db = server[training_db_name]
    atts = doc.get("_attachments", {})

    # 1. Check for existing article.txt in training DB
    if "article.txt" in atts:
        att = training_db.get_attachment(doc_id, "article.txt")
        if att:
            text = att.read().decode("utf-8")
            if text.strip():
                if verbosity >= 2:
                    print(f"  {doc_id}: from article.txt", file=sys.stderr)
                return text, "article.txt"

    # 2. PDF extraction
    if "article.pdf" in atts:
        att = training_db.get_attachment(doc_id, "article.pdf")
        if att:
            try:
                text = plaintext_from_pdf(att.read())
                if text.strip():
                    if verbosity >= 2:
                        print(f"  {doc_id}: from PDF", file=sys.stderr)
                    return text, "pdf"
            except (ImportError, Exception) as exc:
                if verbosity >= 2:
                    print(
                        f"  {doc_id}: PDF extraction failed: {exc}",
                        file=sys.stderr,
                    )

    # 3. JATS XML
    if doc.get("xml_available") and doc.get("xml_format") == "jats":
        att = training_db.get_attachment(doc_id, "article.xml")
        if att:
            try:
                text = plaintext_from_jats(att.read().decode("utf-8"))
                if text.strip():
                    if verbosity >= 2:
                        print(f"  {doc_id}: from JATS", file=sys.stderr)
                    return text, "jats"
            except ValueError:
                pass

    # 4. BioC JSON
    bioc_json = doc.get("bioc_json")
    if bioc_json:
        try:
            text = plaintext_from_bioc(bioc_json)
            if text.strip():
                if verbosity >= 2:
                    print(f"  {doc_id}: from BioC", file=sys.stderr)
                return text, "bioc"
        except ValueError:
            pass

    # 5. E-utilities efetch
    pmcid = doc.get("pmcid")
    if pmcid:
        try:
            text = plaintext_from_efetch(
                pmcid, api_key=config.get("ncbi_api_key"),
            )
            if text.strip():
                if verbosity >= 2:
                    print(f"  {doc_id}: from efetch", file=sys.stderr)
                return text, "efetch"
        except ValueError:
            pass

    # 6. Strip plaintext from YEDDA annotation (last resort)
    yedda_text = doc_info["yedda_text"]
    text = plaintext_from_yedda(yedda_text)
    if text.strip():
        if verbosity >= 2:
            print(f"  {doc_id}: from YEDDA (stripped)", file=sys.stderr)
        return text, "yedda"

    # Should never reach here since YEDDA always has text
    return "", "none"


# ---------------------------------------------------------------------------
# Step 3: Select JATS articles from skol_dev
# ---------------------------------------------------------------------------

def select_jats_docs(
    server,
    dev_db_name: str,
    exclude_ids: Set[str],
    jats_limit: int,
    verbosity: int,
) -> List[Dict[str, Any]]:
    """Select JATS articles from skol_dev for the golden dataset.

    Filters: xml_available=True, xml_format="jats", not in exclude_ids.
    Tries to span multiple journals for diversity.

    Args:
        server: CouchDB server connection.
        dev_db_name: Dev database name.
        exclude_ids: Doc IDs to exclude (already in training selection).
        jats_limit: Maximum number of JATS articles to select.
        verbosity: Logging verbosity.

    Returns:
        List of selected document info dicts.
    """
    db = server[dev_db_name]
    candidates: List[Dict[str, Any]] = []

    for row in db.view("_all_docs", include_docs=True):
        if row.id.startswith("_design/"):
            continue
        if row.id in exclude_ids:
            continue

        doc = row.doc
        if not (doc.get("xml_available") and doc.get("xml_format") == "jats"):
            continue

        # Must have article.xml attachment
        if "article.xml" not in doc.get("_attachments", {}):
            continue

        journal = doc.get("journal", "unknown")
        candidates.append({
            "doc_id": row.id,
            "doc": doc,
            "journal": journal,
        })

    if verbosity >= 1:
        # Count by journal
        journal_counts: Dict[str, int] = {}
        for c in candidates:
            j = c["journal"]
            journal_counts[j] = journal_counts.get(j, 0) + 1
        print(
            f"JATS candidates: {len(candidates)} across "
            f"{len(journal_counts)} journals",
            file=sys.stderr,
        )
        if verbosity >= 2:
            for j, n in sorted(journal_counts.items(),
                               key=lambda x: -x[1]):
                print(f"  {j}: {n}", file=sys.stderr)

    # Select up to jats_limit, spreading across journals for diversity.
    # Round-robin selection from each journal.
    if len(candidates) <= jats_limit:
        selected = candidates
    else:
        by_journal: Dict[str, List[Dict[str, Any]]] = {}
        for c in candidates:
            j = c["journal"]
            by_journal.setdefault(j, []).append(c)

        selected: List[Dict[str, Any]] = []
        journal_lists = list(by_journal.values())
        idx = 0
        while len(selected) < jats_limit and journal_lists:
            for jl in list(journal_lists):
                if idx < len(jl):
                    selected.append(jl[idx])
                    if len(selected) >= jats_limit:
                        break
            # Remove exhausted journals
            journal_lists = [jl for jl in journal_lists if idx + 1 < len(jl)]
            idx += 1

    if verbosity >= 1:
        print(f"JATS selected: {len(selected)}", file=sys.stderr)

    return selected


# ---------------------------------------------------------------------------
# Step 4: Populate golden databases
# ---------------------------------------------------------------------------

def _copy_doc_to_golden(
    source_db,
    golden_db,
    doc_id: str,
    doc: Dict[str, Any],
    golden_sources: Dict[str, Any],
    plaintext: Optional[str],
    copy_attachments: List[str],
    verbosity: int,
) -> None:
    """Copy a document to the golden database with provenance metadata.

    Args:
        source_db: Source CouchDB database.
        golden_db: Target golden CouchDB database.
        doc_id: Document ID.
        doc: Source document dict.
        golden_sources: Provenance metadata dict.
        plaintext: Article plaintext (saved as article.txt attachment).
        copy_attachments: List of attachment names to copy from source.
        verbosity: Logging verbosity.
    """
    # Build golden document (strip CouchDB internal fields)
    golden_doc: Dict[str, Any] = {}
    for k, v in doc.items():
        if k in ("_rev", "_attachments"):
            continue
        golden_doc[k] = v

    golden_doc["_id"] = doc_id
    golden_doc["golden_sources"] = golden_sources

    # Check if doc already exists in golden
    if doc_id in golden_db:
        existing = golden_db[doc_id]
        golden_doc["_rev"] = existing["_rev"]

    golden_db.save(golden_doc)

    # Attach plaintext
    if plaintext:
        golden_doc = golden_db[doc_id]
        golden_db.put_attachment(
            golden_doc,
            plaintext.encode("utf-8"),
            filename="article.txt",
            content_type="text/plain",
        )

    # Copy specified attachments from source
    for att_name in copy_attachments:
        if att_name in doc.get("_attachments", {}):
            att = source_db.get_attachment(doc_id, att_name)
            if att:
                content = att.read()
                golden_doc = golden_db[doc_id]
                content_type = doc["_attachments"][att_name].get(
                    "content_type", "application/octet-stream",
                )
                golden_db.put_attachment(
                    golden_doc,
                    content,
                    filename=att_name,
                    content_type=content_type,
                )

    if verbosity >= 2:
        print(f"  Copied {doc_id} to {golden_db.name}", file=sys.stderr)


def _save_annotation(
    ann_db,
    doc_id: str,
    yedda_text: str,
    source: str,
    source_database: str,
    verbosity: int,
) -> None:
    """Save a YEDDA annotation as a document with article.txt.ann attachment.

    Args:
        ann_db: Annotation CouchDB database.
        doc_id: Document ID.
        yedda_text: YEDDA-annotated text.
        source: Source identifier (e.g., "hand", "jats", "bioc").
        source_database: Name of the source database.
        verbosity: Logging verbosity.
    """
    ann_doc: Dict[str, Any] = {
        "_id": doc_id,
        "source": source,
        "source_database": source_database,
    }

    if doc_id in ann_db:
        existing = ann_db[doc_id]
        ann_doc["_rev"] = existing["_rev"]

    ann_db.save(ann_doc)

    ann_doc = ann_db[doc_id]
    ann_db.put_attachment(
        ann_doc,
        yedda_text.encode("utf-8"),
        filename="article.txt.ann",
        content_type="text/plain",
    )

    if verbosity >= 2:
        print(
            f"  Saved {source} annotation for {doc_id} "
            f"({len(yedda_text)} chars)",
            file=sys.stderr,
        )


def populate_golden_databases(
    server,
    training_db_name: str,
    dev_db_name: str,
    training_selections: List[Dict[str, Any]],
    jats_selections: List[Dict[str, Any]],
    training_plaintexts: Dict[str, Tuple[str, str]],
    config: Dict[str, Any],
    dry_run: bool,
    verbosity: int,
) -> Dict[str, int]:
    """Populate all golden databases.

    Args:
        server: CouchDB server.
        training_db_name: Training database name.
        dev_db_name: Dev database name.
        training_selections: Selected training doc infos.
        jats_selections: Selected JATS doc infos.
        training_plaintexts: {doc_id: (plaintext, source)} for training docs.
        config: Environment config.
        dry_run: If True, don't write anything.
        verbosity: Logging verbosity.

    Returns:
        Dict of counts: {database_name: num_docs_written}.
    """
    if dry_run:
        counts = {
            "skol_golden": len(training_selections) + len(jats_selections),
            "skol_golden_ann_hand": len(training_selections),
        }
        # Count JATS/BioC annotation candidates
        jats_ann = 0
        bioc_ann = 0
        for sel in training_selections:
            doc = sel["doc"]
            if doc.get("xml_available") and doc.get("xml_format") == "jats":
                jats_ann += 1
            if doc.get("bioc_json"):
                bioc_ann += 1
        for sel in jats_selections:
            jats_ann += 1  # All JATS selections have JATS XML
            if sel["doc"].get("bioc_json"):
                bioc_ann += 1
        counts["skol_golden_ann_jats"] = jats_ann
        counts["skol_golden_ann_bioc"] = bioc_ann
        return counts

    # Create/open golden databases
    golden_db = _get_or_create_db(server, "skol_golden")
    ann_hand_db = _get_or_create_db(server, "skol_golden_ann_hand")
    ann_jats_db = _get_or_create_db(server, "skol_golden_ann_jats")
    ann_bioc_db = _get_or_create_db(server, "skol_golden_ann_bioc")

    training_db = server[training_db_name]
    dev_db = server[dev_db_name]

    counts = {
        "skol_golden": 0,
        "skol_golden_ann_hand": 0,
        "skol_golden_ann_jats": 0,
        "skol_golden_ann_bioc": 0,
    }

    # Process training documents
    if verbosity >= 1:
        print("\nPopulating from training documents...", file=sys.stderr)

    for sel in training_selections:
        doc_id = sel["doc_id"]
        doc = sel["doc"]
        plaintext, pt_source = training_plaintexts.get(
            doc_id, ("", "none"),
        )

        golden_sources = {
            "hand_annotated": True,
            "jats_available": bool(
                doc.get("xml_available")
                and doc.get("xml_format") == "jats"
            ),
            "bioc_available": bool(doc.get("bioc_json")),
            "has_pdf": "article.pdf" in doc.get("_attachments", {}),
            "pmcid": doc.get("pmcid"),
            "plaintext_source": pt_source,
        }

        # Copy to skol_golden
        copy_atts = ["article.pdf", "article.xml"]
        _copy_doc_to_golden(
            training_db, golden_db, doc_id, doc,
            golden_sources, plaintext, copy_atts, verbosity,
        )
        counts["skol_golden"] += 1

        # Save hand annotation
        _save_annotation(
            ann_hand_db, doc_id, sel["yedda_text"],
            "hand", training_db_name, verbosity,
        )
        counts["skol_golden_ann_hand"] += 1

        # Generate JATS annotation if JATS XML available
        if golden_sources["jats_available"]:
            att = training_db.get_attachment(doc_id, "article.xml")
            if att:
                try:
                    xml_str = att.read().decode("utf-8")
                    jats_yedda = jats_xml_to_yedda(xml_str)
                    _save_annotation(
                        ann_jats_db, doc_id, jats_yedda,
                        "jats", training_db_name, verbosity,
                    )
                    counts["skol_golden_ann_jats"] += 1
                except Exception as exc:
                    if verbosity >= 1:
                        print(
                            f"  {doc_id}: JATS→YEDDA failed: {exc}",
                            file=sys.stderr,
                        )

        # Generate BioC annotation if BioC JSON available
        if golden_sources["bioc_available"]:
            try:
                bioc_yedda = bioc_json_to_yedda(doc.get("bioc_json"))
                _save_annotation(
                    ann_bioc_db, doc_id, bioc_yedda,
                    "bioc", training_db_name, verbosity,
                )
                counts["skol_golden_ann_bioc"] += 1
            except Exception as exc:
                if verbosity >= 1:
                    print(
                        f"  {doc_id}: BioC→YEDDA failed: {exc}",
                        file=sys.stderr,
                    )

    # Process JATS documents from dev
    if verbosity >= 1:
        print("\nPopulating from JATS articles...", file=sys.stderr)

    for sel in jats_selections:
        doc_id = sel["doc_id"]
        doc = sel["doc"]

        # Get plaintext from JATS XML
        att = dev_db.get_attachment(doc_id, "article.xml")
        if att is None:
            if verbosity >= 1:
                print(
                    f"  {doc_id}: no article.xml, skipping",
                    file=sys.stderr,
                )
            continue

        xml_str = att.read().decode("utf-8")

        try:
            plaintext = plaintext_from_jats(xml_str)
        except ValueError as exc:
            if verbosity >= 1:
                print(
                    f"  {doc_id}: JATS plaintext failed: {exc}",
                    file=sys.stderr,
                )
            continue

        # Also check for existing article.txt
        pt_source = "jats"
        existing_att = dev_db.get_attachment(doc_id, "article.txt")
        if existing_att:
            existing_text = existing_att.read().decode("utf-8")
            if existing_text.strip():
                plaintext = existing_text
                pt_source = "article.txt"

        golden_sources = {
            "hand_annotated": False,
            "jats_available": True,
            "bioc_available": bool(doc.get("bioc_json")),
            "has_pdf": "article.pdf" in doc.get("_attachments", {}),
            "pmcid": doc.get("pmcid"),
            "plaintext_source": pt_source,
        }

        # Copy to skol_golden
        copy_atts = ["article.pdf", "article.xml"]
        _copy_doc_to_golden(
            dev_db, golden_db, doc_id, doc,
            golden_sources, plaintext, copy_atts, verbosity,
        )
        counts["skol_golden"] += 1

        # Generate JATS annotation
        try:
            jats_yedda = jats_xml_to_yedda(xml_str)
            _save_annotation(
                ann_jats_db, doc_id, jats_yedda,
                "jats", dev_db_name, verbosity,
            )
            counts["skol_golden_ann_jats"] += 1
        except Exception as exc:
            if verbosity >= 1:
                print(
                    f"  {doc_id}: JATS→YEDDA failed: {exc}",
                    file=sys.stderr,
                )

        # Generate BioC annotation if available
        bioc_json = doc.get("bioc_json")
        if bioc_json:
            try:
                bioc_yedda = bioc_json_to_yedda(bioc_json)
                _save_annotation(
                    ann_bioc_db, doc_id, bioc_yedda,
                    "bioc", dev_db_name, verbosity,
                )
                counts["skol_golden_ann_bioc"] += 1
            except Exception as exc:
                if verbosity >= 1:
                    print(
                        f"  {doc_id}: BioC→YEDDA failed: {exc}",
                        file=sys.stderr,
                    )

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate a golden evaluation dataset.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        required=True,
        help="Process all eligible documents.",
    )
    parser.add_argument(
        "--min-tags",
        type=int,
        default=4,
        help="Minimum distinct YEDDA tag types for training docs (default: 4).",
    )
    parser.add_argument(
        "--hand-limit",
        type=int,
        default=30,
        help="Max hand-annotated docs for golden holdout (default: 30).",
    )
    parser.add_argument(
        "--jats-limit",
        type=int,
        default=75,
        help="Maximum JATS articles to include from skol_dev (default: 75).",
    )
    parser.add_argument(
        "--training-database",
        type=str,
        default=None,
        help="Training database name (default: from env).",
    )
    parser.add_argument(
        "--dev-database",
        type=str,
        default=None,
        help="Dev database name (default: from env).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output.",
    )

    args = parser.parse_args()

    verbosity = 0 if args.quiet else args.verbose
    config = get_env_config()

    training_db_name = (
        args.training_database or config.get("training_database", "skol_training")
    )
    dev_db_name = (
        args.dev_database or config.get("couchdb_database", "skol_dev")
    )

    server = _connect_server(config)

    if verbosity >= 1:
        print(
            f"Curating golden dataset from {training_db_name} "
            f"and {dev_db_name}",
            file=sys.stderr,
        )

    # Step 1: Select training documents (all qualifying)
    all_training = select_training_docs(
        server, training_db_name, args.min_tags, verbosity,
    )

    # Step 1b: Subsample to holdout size
    training_selections = subsample_training_docs(
        all_training, args.hand_limit, verbosity,
    )

    # Step 2: Obtain plaintext for training documents
    training_plaintexts: Dict[str, Tuple[str, str]] = {}
    pt_source_counts: Dict[str, int] = {}
    if verbosity >= 1:
        print("\nObtaining plaintext for training documents...",
              file=sys.stderr)

    for sel in training_selections:
        plaintext, source = obtain_plaintext(
            server, training_db_name, dev_db_name,
            sel, config, verbosity,
        )
        training_plaintexts[sel["doc_id"]] = (plaintext, source)
        pt_source_counts[source] = pt_source_counts.get(source, 0) + 1

    if verbosity >= 1:
        print("Plaintext sources:", file=sys.stderr)
        for src, n in sorted(pt_source_counts.items(), key=lambda x: -x[1]):
            print(f"  {src}: {n}", file=sys.stderr)

    # Step 3: Select JATS articles from dev
    exclude_ids = {sel["doc_id"] for sel in training_selections}
    jats_selections = select_jats_docs(
        server, dev_db_name, exclude_ids, args.jats_limit, verbosity,
    )

    # Step 4: Populate golden databases
    if verbosity >= 1:
        total = len(training_selections) + len(jats_selections)
        print(
            f"\nTotal articles for golden dataset: {total} "
            f"({len(training_selections)} training + "
            f"{len(jats_selections)} JATS)",
            file=sys.stderr,
        )

    if args.dry_run:
        counts = populate_golden_databases(
            server, training_db_name, dev_db_name,
            training_selections, jats_selections,
            training_plaintexts, config, dry_run=True,
            verbosity=verbosity,
        )
        print("\n=== DRY RUN ===")
        print("Would create/update:")
        for db_name, n in sorted(counts.items()):
            print(f"  {db_name}: {n} documents")
        return

    counts = populate_golden_databases(
        server, training_db_name, dev_db_name,
        training_selections, jats_selections,
        training_plaintexts, config, dry_run=False,
        verbosity=verbosity,
    )

    if verbosity >= 1:
        print("\n=== Golden dataset curated ===", file=sys.stderr)
        for db_name, n in sorted(counts.items()):
            print(f"  {db_name}: {n} documents", file=sys.stderr)


if __name__ == "__main__":
    main()
