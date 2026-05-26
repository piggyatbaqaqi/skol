#!/usr/bin/env python3
"""
Build ingestion sources statistics and store in Redis.

This script calculates statistics about ingestion sources (publications/journals)
from CouchDB and stores the results as JSON in Redis for fast retrieval by the
Django sources page.

Redis Key: skol:sources:stats
TTL: None (manual refresh via this script or cron)

Usage:
    build_sources_stats.py [options]

Options:
    --verbosity N       Output verbosity level (0=quiet, 1=normal, 2=verbose)
    --ingest-db-name    Name of the ingest database (default: skol_dev)
    --taxon-db-name     Name of the taxa database (default: skol_taxa_dev)
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for skol modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import re

import couchdb
from env_config import get_env_config, create_redis_client

# Default Redis key for sources statistics (v1 / anonymous-user
# fallback).  Experiment-scoped runs append ``:{experiment_name}``
# via :func:`redis_key_for_experiment`.
_DEFAULT_REDIS_KEY = 'skol:sources:stats'


# Regex tags applied to each Treatment's ``treatment`` text (the
# Nomenclature paragraph) to count newly described or sanctioned
# names per journal.  Surface on the Ingestion Sources page as
# additional badges next to the total Treatments count.
#
# ``sp. nov.`` / ``gen. nov.`` / ``comb. nov.`` are nomenclatural-act
# markers — papers establishing a new species, genus, or combination.
# Case-insensitive; tolerates optional trailing-period and the space
# between the abbreviation and ``nov.``.
_NEW_TAXON_RE = re.compile(
    r"\b(?:sp|gen|comb|nom)\.?\s+nov\.?\b",
    re.IGNORECASE,
)

# Sanctioning-author tags.  Fries' Systema Mycologicum (1821) and
# Persoon's Synopsis Methodica Fungorum (1801) are the two
# nomenclatural-priority anchors in mycology — names with these
# attributions get special legal status.  Common citation forms:
#   ``Name : Fr.`` (sanctioned by Fries)
#   ``Name (Fr.) NewAuthor`` (combination based on a Fries name)
#   ``Name ex Fries`` (validly published by Fries even if originally proposed by another)
#   ``Name : Pers.`` / ``(Pers.)`` / ``ex Persoon``
_SANCTIONED_RE = re.compile(
    r":\s*Fr\.|:\s*Pers\.|\(Fr\.\)|\(Pers\.\)|\bex\s+Fries\b|\bex\s+Persoon\b",
)


def count_new_taxon_acts(text):
    """Count nomenclatural-act markers (``sp. nov.``, ``gen. nov.``,
    ``comb. nov.``, ``nom. nov.``) in a Treatment's text."""
    if not text:
        return 0
    return len(_NEW_TAXON_RE.findall(text))


def count_sanctioned_markers(text):
    """Count Fries / Persoon sanctioning-author citations in a
    Treatment's text."""
    if not text:
        return 0
    return len(_SANCTIONED_RE.findall(text))


def redis_key_for_experiment(experiment_name):
    """Return the Redis key used by ``build_sources_stats`` for the
    given experiment.

    When *experiment_name* is empty / None, returns the legacy
    ``skol:sources:stats`` key — matches the v1 cron job and the
    Django sources view's default fallback (for anonymous users or
    users with no active experiment).

    When *experiment_name* is non-empty (e.g. ``production_v3_hand``),
    appends ``:{experiment_name}`` so each experiment's Sources page
    reads its own stats blob.  Mirrors the experiment-doc Redis-key
    convention used elsewhere (``skol:embedding:v3_hand`` etc.).
    """
    if not experiment_name:
        return _DEFAULT_REDIS_KEY
    return f'{_DEFAULT_REDIS_KEY}:{experiment_name}'


def get_publication_registry():
    """Load the PublicationRegistry if available."""
    try:
        ingestors_path = str(Path(__file__).resolve().parent.parent / 'ingestors')
        if ingestors_path not in sys.path:
            sys.path.insert(0, ingestors_path)
        from ingestors.publications import PublicationRegistry
        return PublicationRegistry
    except ImportError:
        return None


def build_sources_stats(
    couchdb_url: str,
    ingest_db_name: str,
    treatments_db_name: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    verbosity: int = 1,
) -> Dict[str, Any]:
    """
    Build statistics about ingestion sources.

    Args:
        couchdb_url: CouchDB server URL
        ingest_db_name: Name of the ingest database
        treatments_db_name: Name of the taxa database
        username: CouchDB username
        password: CouchDB password
        verbosity: Output verbosity level

    Returns:
        Dictionary with sources statistics ready for JSON serialization
    """
    PublicationRegistry = get_publication_registry()

    # Connect to CouchDB
    server = couchdb.Server(couchdb_url)
    if username and password:
        server.resource.credentials = (username, password)

    # Check if ingest database exists
    if ingest_db_name not in server:
        raise ValueError(f"Ingest database '{ingest_db_name}' not found")

    db = server[ingest_db_name]

    if verbosity >= 1:
        print(f"Scanning ingest database: {ingest_db_name}")

    # Collect statistics by source
    source_stats: Dict[str, Dict[str, int]] = {}
    doc_to_journal: Dict[str, str] = {}
    doc_count = 0

    # Get normalize function if available
    normalize_fn = None
    if PublicationRegistry:
        normalize_fn = PublicationRegistry.normalize_journal_name

    for doc_id in db:
        # Skip design documents
        if doc_id.startswith('_design/'):
            continue

        try:
            doc = db[doc_id]
            doc_count += 1

            if verbosity >= 2 and doc_count % 1000 == 0:
                print(f"  Processed {doc_count} ingest documents...")

            # Get the journal name from the document
            journal_name = doc.get('journal')
            if not journal_name:
                journal_name = 'Unknown'

            # Normalize journal name to canonical form
            if normalize_fn:
                journal_name = normalize_fn(journal_name)

            # Map doc_id to journal for taxa lookup
            doc_to_journal[doc_id] = journal_name

            # Initialize stats for this journal if not seen before.
            # ``new_taxa_acts`` counts ``sp. nov.`` / ``gen. nov.`` /
            # ``comb. nov.`` / ``nom. nov.`` markers across all
            # Treatments from this journal; ``sanctioned_markers``
            # counts Fries / Persoon sanction citations.
            if journal_name not in source_stats:
                source_stats[journal_name] = {
                    'total': 0,
                    'taxonomy': 0,
                    'treatments': 0,
                    'new_taxa_acts': 0,
                    'sanctioned_markers': 0,
                }

            # Increment total count
            source_stats[journal_name]['total'] += 1

            # Check if document has taxonomy
            if doc.get('taxonomy') is True:
                source_stats[journal_name]['taxonomy'] += 1

        except Exception:
            continue

    if verbosity >= 1:
        print(f"  Found {doc_count} ingest documents across {len(source_stats)} journals")

    # Count taxa records
    if treatments_db_name in server:
        taxa_db = server[treatments_db_name]
        taxa_count = 0

        if verbosity >= 1:
            print(f"Scanning treatments database: {treatments_db_name}")

        for taxa_doc_id in taxa_db:
            if taxa_doc_id.startswith('_design/'):
                continue

            try:
                taxa_doc = taxa_db[taxa_doc_id]
                taxa_count += 1

                if verbosity >= 2 and taxa_count % 1000 == 0:
                    print(f"  Processed {taxa_count} treatment documents...")

                # Get the ingest doc_id from the taxa document
                ingest = taxa_doc.get('ingest', {})
                ingest_doc_id = ingest.get('_id')
                if ingest_doc_id and ingest_doc_id in doc_to_journal:
                    journal_name = doc_to_journal[ingest_doc_id]
                    source_stats[journal_name]['treatments'] += 1
                    # Mycologist-facing counts derived from the
                    # Nomenclature text on each Treatment.
                    treatment_text = taxa_doc.get('treatment') or ''
                    source_stats[journal_name]['new_taxa_acts'] += (
                        count_new_taxon_acts(treatment_text)
                    )
                    source_stats[journal_name]['sanctioned_markers'] += (
                        count_sanctioned_markers(treatment_text)
                    )

            except Exception:
                continue

        if verbosity >= 1:
            print(f"  Found {taxa_count} treatment documents")
    else:
        if verbosity >= 1:
            print(f"Taxa database '{treatments_db_name}' not found, skipping taxa counts")

    # Build display list with publication details
    sources = []
    total_records = 0
    total_taxonomy_documents = 0
    total_treatments_records = 0
    total_new_taxa_acts = 0
    total_sanctioned_markers = 0

    for journal_name, stats in source_stats.items():
        source_info = {
            'key': journal_name,
            'name': journal_name,
            'publisher': 'Unknown',
            'website': None,
            'total_records': stats['total'],
            'taxonomy_records': stats['taxonomy'],
            'taxonomy_percentage': round(
                (stats['taxonomy'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                1
            ),
            'treatments_records': stats['treatments'],
            'new_taxa_acts': stats.get('new_taxa_acts', 0),
            'sanctioned_markers': stats.get('sanctioned_markers', 0),
        }

        # Try to get additional information from PublicationRegistry
        if PublicationRegistry:
            pub_config = PublicationRegistry.get_by_journal(journal_name)
            if pub_config:
                source_info['name'] = pub_config.get('name', journal_name)
                source_info['publisher'] = pub_config.get('source', 'Unknown')
                if pub_config.get('address'):
                    source_info['website'] = pub_config['address']
                else:
                    # Fallback: try to extract website from various URL fields
                    for url_field in ['rss_url', 'index_url', 'archives_url', 'issues_url', 'url_prefix']:
                        if url_field in pub_config and pub_config[url_field]:
                            url = pub_config[url_field]
                            from urllib.parse import urlparse
                            parsed = urlparse(url)
                            if parsed.netloc:
                                source_info['website'] = f"{parsed.scheme}://{parsed.netloc}"
                                break

        sources.append(source_info)
        total_records += stats['total']
        total_taxonomy_documents += stats['taxonomy']
        total_treatments_records += stats['treatments']
        total_new_taxa_acts += stats.get('new_taxa_acts', 0)
        total_sanctioned_markers += stats.get('sanctioned_markers', 0)

    # Sort sources by name
    sources.sort(key=lambda x: x['name'].lower())

    result = {
        'sources': sources,
        'total_records': total_records,
        'total_taxonomy_documents': total_taxonomy_documents,
        'total_treatments_records': total_treatments_records,
        'total_new_taxa_acts': total_new_taxa_acts,
        'total_sanctioned_markers': total_sanctioned_markers,
        'ingest_db_name': ingest_db_name,
        'treatments_db_name': treatments_db_name,
        'created_at': datetime.now().isoformat(),
        'version': '1.0',
    }

    if verbosity >= 1:
        print(f"\nSummary:")
        print(f"  Sources: {len(sources)}")
        print(f"  Total records: {total_records}")
        print(f"  Taxonomy documents: {total_taxonomy_documents}")
        print(f"  Treatments records: {total_treatments_records}")
        print(f"  New-taxon acts (sp/gen/comb/nom nov.): {total_new_taxa_acts}")
        print(f"  Sanctioned (Fries/Persoon) markers:    {total_sanctioned_markers}")

    return result


def save_to_redis(
    stats: Dict[str, Any],
    experiment_name: Optional[str] = None,
    verbosity: int = 1,
) -> None:
    """
    Save sources statistics to Redis.

    Args:
        stats: Statistics dictionary to save
        experiment_name: Active experiment ID; selects an
            experiment-scoped Redis key.  None / empty writes to the
            default ``skol:sources:stats`` key (v1 / anonymous-user
            fallback).
        verbosity: Output verbosity level
    """
    r = create_redis_client(decode_responses=True)
    key = redis_key_for_experiment(experiment_name)

    # Save as JSON
    json_data = json.dumps(stats)
    r.set(key, json_data)

    if verbosity >= 1:
        print(f"\nSaved to Redis key: {key}")
        print(f"  JSON size: {len(json_data)} bytes")


def main():
    """Main entry point."""
    import argparse

    config = get_env_config()

    parser = argparse.ArgumentParser(
        description="Build ingestion sources statistics and store in Redis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build stats with default databases
    build_sources_stats.py

    # Use specific databases
    build_sources_stats.py --ingest-db-name skol_prod --taxon-db-name skol_taxa_prod

    # Verbose output
    build_sources_stats.py --verbosity 2

Environment Variables:
    COUCHDB_URL         CouchDB server URL
    COUCHDB_USER        CouchDB username
    COUCHDB_PASSWORD    CouchDB password
    INGEST_DB_NAME      Ingest database name (default: skol_dev)
    TREATMENTS_DB_NAME  Taxa database name (default: skol_taxa_dev)
                        TAXON_DB_NAME accepted as deprecated fallback
    REDIS_HOST          Redis host
    REDIS_PORT          Redis port
    REDIS_PASSWORD      Redis password
"""
    )

    args, _ = parser.parse_known_args()

    # Get configuration
    couchdb_url = config['couchdb_url']
    ingest_db_name = config.get('ingest_db_name', 'skol_dev')
    treatments_db_name = config.get('treatments_db_name', 'skol_treatments_dev')
    username = config['couchdb_username']
    password = config['couchdb_password']
    verbosity = config['verbosity']
    # ``--experiment NAME`` (env_config.py:424) populates this; an empty
    # string means "no experiment selected" and the default Redis key
    # gets written for the v1 sources page.
    experiment_name = config.get('experiment_name') or None

    if verbosity >= 1:
        print("Building ingestion sources statistics...")
        print(f"  CouchDB URL: {couchdb_url}")
        print(f"  Experiment: {experiment_name or '(none — default Redis key)'}")
        print(f"  Ingest DB: {ingest_db_name}")
        print(f"  Treatments DB: {treatments_db_name}")
        print()

    try:
        # Build statistics
        stats = build_sources_stats(
            couchdb_url=couchdb_url,
            ingest_db_name=ingest_db_name,
            treatments_db_name=treatments_db_name,
            username=username,
            password=password,
            verbosity=verbosity,
        )

        # Save to Redis
        save_to_redis(
            stats, experiment_name=experiment_name, verbosity=verbosity,
        )

        if verbosity >= 1:
            print("\nDone!")

    except Exception as e:
        print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        if verbosity >= 2:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
