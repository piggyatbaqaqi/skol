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

import couchdb
from env_config import get_env_config, create_redis_client

# Redis key for sources statistics
REDIS_KEY = 'skol:sources:stats'


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
    taxa_db_name: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    verbosity: int = 1,
) -> Dict[str, Any]:
    """
    Build statistics about ingestion sources.

    Args:
        couchdb_url: CouchDB server URL
        ingest_db_name: Name of the ingest database
        taxa_db_name: Name of the taxa database
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

            # Initialize stats for this journal if not seen before
            if journal_name not in source_stats:
                source_stats[journal_name] = {
                    'total': 0,
                    'taxonomy': 0,
                    'taxa': 0,
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
    if taxa_db_name in server:
        taxa_db = server[taxa_db_name]
        taxa_count = 0

        if verbosity >= 1:
            print(f"Scanning taxa database: {taxa_db_name}")

        for taxa_doc_id in taxa_db:
            if taxa_doc_id.startswith('_design/'):
                continue

            try:
                taxa_doc = taxa_db[taxa_doc_id]
                taxa_count += 1

                if verbosity >= 2 and taxa_count % 1000 == 0:
                    print(f"  Processed {taxa_count} taxa documents...")

                # Get the ingest doc_id from the taxa document
                ingest = taxa_doc.get('ingest', {})
                ingest_doc_id = ingest.get('_id')
                if ingest_doc_id and ingest_doc_id in doc_to_journal:
                    journal_name = doc_to_journal[ingest_doc_id]
                    source_stats[journal_name]['taxa'] += 1

            except Exception:
                continue

        if verbosity >= 1:
            print(f"  Found {taxa_count} taxa documents")
    else:
        if verbosity >= 1:
            print(f"Taxa database '{taxa_db_name}' not found, skipping taxa counts")

    # Build display list with publication details
    sources = []
    total_records = 0
    total_taxonomy_documents = 0
    total_taxa_records = 0

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
            'taxa_records': stats['taxa'],
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
        total_taxa_records += stats['taxa']

    # Sort sources by name
    sources.sort(key=lambda x: x['name'].lower())

    result = {
        'sources': sources,
        'total_records': total_records,
        'total_taxonomy_documents': total_taxonomy_documents,
        'total_taxa_records': total_taxa_records,
        'ingest_db_name': ingest_db_name,
        'taxa_db_name': taxa_db_name,
        'created_at': datetime.now().isoformat(),
        'version': '1.0',
    }

    if verbosity >= 1:
        print(f"\nSummary:")
        print(f"  Sources: {len(sources)}")
        print(f"  Total records: {total_records}")
        print(f"  Taxonomy documents: {total_taxonomy_documents}")
        print(f"  Taxa records: {total_taxa_records}")

    return result


def save_to_redis(stats: Dict[str, Any], verbosity: int = 1) -> None:
    """
    Save sources statistics to Redis.

    Args:
        stats: Statistics dictionary to save
        verbosity: Output verbosity level
    """
    r = create_redis_client(decode_responses=True)

    # Save as JSON
    json_data = json.dumps(stats)
    r.set(REDIS_KEY, json_data)

    if verbosity >= 1:
        print(f"\nSaved to Redis key: {REDIS_KEY}")
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
    TAXON_DB_NAME       Taxa database name (default: skol_taxa_dev)
    REDIS_HOST          Redis host
    REDIS_PORT          Redis port
    REDIS_PASSWORD      Redis password
"""
    )

    args, _ = parser.parse_known_args()

    # Get configuration
    couchdb_url = config['couchdb_url']
    ingest_db_name = config.get('ingest_db_name', 'skol_dev')
    taxa_db_name = config.get('taxon_db_name', 'skol_taxa_dev')
    username = config['couchdb_username']
    password = config['couchdb_password']
    verbosity = config['verbosity']

    if verbosity >= 1:
        print("Building ingestion sources statistics...")
        print(f"  CouchDB URL: {couchdb_url}")
        print(f"  Ingest DB: {ingest_db_name}")
        print(f"  Taxa DB: {taxa_db_name}")
        print()

    try:
        # Build statistics
        stats = build_sources_stats(
            couchdb_url=couchdb_url,
            ingest_db_name=ingest_db_name,
            taxa_db_name=taxa_db_name,
            username=username,
            password=password,
            verbosity=verbosity,
        )

        # Save to Redis
        save_to_redis(stats, verbosity=verbosity)

        if verbosity >= 1:
            print("\nDone!")

    except Exception as e:
        print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        if verbosity >= 2:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
