#!/usr/bin/env python3
"""
Migrate skol_taxa_dev from Provenance-Based to Content-Based Document IDs

Current IDs: sha256(doc_id:url:line_number) -- provenance-based
New IDs:     sha256(taxon_text:description_text) -- content-based

This eliminates duplicate taxa that were ingested through different code paths.

The migration creates a mapping database (old_id -> new_id) and then migrates
both skol_taxa_dev and skol_taxa_full_dev in-place.

Phases:
  mapping   - Build old_id -> new_id mapping in migration database
  taxa      - Migrate skol_taxa_dev documents to new IDs
  taxa_full - Migrate skol_taxa_full_dev documents to new IDs
  all       - Run all phases in order

Usage:
    # Dry run to see stats
    python fixes/migrate_taxa_ids.py --phase mapping --dry-run

    # Run mapping phase
    python fixes/migrate_taxa_ids.py --phase mapping

    # Run all phases
    python fixes/migrate_taxa_ids.py --phase all

    # Production databases
    python fixes/migrate_taxa_ids.py --phase all \\
        --taxa-db skol_taxa --taxa-full-db skol_taxa_full \\
        --migration-db skol_taxa_migration
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent directory to path for env_config and ingestors
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_config import get_env_config
from ingestors.timestamps import set_timestamps


def generate_taxon_doc_id(taxon_text: str, description_text: str) -> str:
    """Generate a content-based, deterministic document ID for a taxon.

    Identical taxon+description content always produces the same ID,
    regardless of which ingest path produced it.

    Args:
        taxon_text: The nomenclature/taxon text
        description_text: The description text

    Returns:
        Deterministic document ID as 'taxon_<sha256_hex>'
    """
    content = (taxon_text or "").strip() + ":" + (description_text or "").strip()
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    return f"taxon_{hash_obj.hexdigest()}"


def score_document(doc: dict, taxa_full_ids: set) -> int:
    """Score a taxon document for winner selection during deduplication.

    Higher score = better candidate to be the canonical document.

    Args:
        doc: The taxon document from CouchDB
        taxa_full_ids: Set of document IDs that exist in skol_taxa_full_dev

    Returns:
        Integer score
    """
    score = 0

    # Has enriched JSON translation in taxa_full
    if doc['_id'] in taxa_full_ids:
        score += 100

    # Has both span types
    nom_spans = doc.get('nomenclature_spans')
    desc_spans = doc.get('description_spans')
    if nom_spans and desc_spans:
        score += 10

    # Has PDF page info
    pdf_page = doc.get('pdf_page')
    if pdf_page is not None and pdf_page != 0:
        score += 5

    # Count populated ingest fields
    ingest = doc.get('ingest', {})
    if isinstance(ingest, dict):
        for key, value in ingest.items():
            if value is not None and value != '' and key != '_id':
                score += 1

    return score


def phase_mapping(
    config: dict,
    taxa_db: str,
    taxa_full_db: str,
    migration_db: str,
    dry_run: bool = False,
    batch_size: int = 100,
    verbosity: int = 1
) -> dict:
    """Phase 1: Build the old_id -> new_id mapping database.

    Args:
        config: Environment configuration
        taxa_db: Source taxa database name
        taxa_full_db: Source taxa_full database name
        migration_db: Migration mapping database name
        dry_run: If True, print stats without creating the mapping database
        batch_size: Documents per bulk write batch
        verbosity: Verbosity level

    Returns:
        Dict with statistics
    """
    import couchdb

    couchdb_url = config['couchdb_url']
    username = config['couchdb_username']
    password = config['couchdb_password']

    print(f"\n{'='*70}")
    print(f"Phase 1: Build Migration Mapping")
    print(f"{'='*70}")
    print(f"CouchDB: {couchdb_url}")
    print(f"Taxa DB: {taxa_db}")
    print(f"Taxa Full DB: {taxa_full_db}")
    print(f"Migration DB: {migration_db}")
    if dry_run:
        print(f"Mode: DRY RUN")
    print()

    # Connect to CouchDB
    server = couchdb.Server(couchdb_url)
    if username and password:
        server.resource.credentials = (username, password)

    if taxa_db not in server:
        print(f"Error: Database '{taxa_db}' not found")
        return {'error': True}

    db = server[taxa_db]

    # Collect taxa_full IDs for scoring
    taxa_full_ids = set()
    if taxa_full_db in server:
        full_db = server[taxa_full_db]
        if verbosity >= 1:
            print("Loading taxa_full document IDs for scoring...")
        for doc_id in full_db:
            if not doc_id.startswith('_design/'):
                taxa_full_ids.add(doc_id)
        if verbosity >= 1:
            print(f"  Found {len(taxa_full_ids)} documents in {taxa_full_db}")
    else:
        print(f"Warning: {taxa_full_db} not found, scoring without json_annotated info")

    # Step 1: Iterate all taxa docs and compute new IDs
    if verbosity >= 1:
        print("\nReading all taxa documents...")

    # old_id -> (new_id, doc)
    old_to_new = {}
    # new_id -> [old_id, ...]
    new_id_groups = defaultdict(list)

    doc_ids = []
    for doc_id in db:
        if not doc_id.startswith('_design/'):
            doc_ids.append(doc_id)

    total = len(doc_ids)
    if verbosity >= 1:
        print(f"  Found {total} documents")

    errors = 0
    for i, old_id in enumerate(doc_ids):
        try:
            doc = db[old_id]
            taxon_text = doc.get('taxon', '')
            description_text = doc.get('description', '')
            new_id = generate_taxon_doc_id(taxon_text, description_text)

            old_to_new[old_id] = (new_id, doc)
            new_id_groups[new_id].append(old_id)

            if verbosity >= 2 and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{total}...")
        except Exception as e:
            errors += 1
            print(f"  ERROR reading {old_id}: {e}")

    unique_new_ids = len(new_id_groups)
    duplicates = {k: v for k, v in new_id_groups.items() if len(v) > 1}
    unchanged = sum(1 for old_id, (new_id, _) in old_to_new.items() if old_id == new_id)

    # Print statistics
    print(f"\n{'='*70}")
    print("Mapping Statistics")
    print(f"{'='*70}")
    print(f"Total documents:          {total}")
    print(f"Unique content hashes:    {unique_new_ids}")
    print(f"Duplicate groups:         {len(duplicates)}")
    print(f"Documents to deduplicate: {sum(len(v) for v in duplicates.values())}")
    print(f"Net reduction:            {total - unique_new_ids}")
    print(f"Already correct ID:       {unchanged}")
    print(f"Errors:                   {errors}")

    if duplicates and verbosity >= 1:
        # Show duplicate group size distribution
        sizes = defaultdict(int)
        for group in duplicates.values():
            sizes[len(group)] += 1
        print(f"\nDuplicate group sizes:")
        for size in sorted(sizes.keys()):
            print(f"  Size {size}: {sizes[size]} groups")

    if dry_run:
        print(f"\n[DRY RUN] No mapping database created")
        return {
            'total': total,
            'unique': unique_new_ids,
            'duplicate_groups': len(duplicates),
            'net_reduction': total - unique_new_ids,
            'errors': errors
        }

    # Step 2: Score and pick winners
    if verbosity >= 1:
        print("\nScoring documents and selecting winners...")

    # winner[new_id] = old_id of the winner
    winners = {}
    for new_id, old_ids in new_id_groups.items():
        if len(old_ids) == 1:
            winners[new_id] = old_ids[0]
        else:
            # Score each candidate
            scored = []
            for old_id in old_ids:
                _, doc = old_to_new[old_id]
                s = score_document(doc, taxa_full_ids)
                scored.append((s, old_id))
            # Sort by score descending, then by old_id ascending (tie-breaker)
            scored.sort(key=lambda x: (-x[0], x[1]))
            winners[new_id] = scored[0][1]

            if verbosity >= 2:
                print(f"  Group for {new_id[:50]}...")
                for s, oid in scored:
                    marker = " (WINNER)" if oid == scored[0][1] else ""
                    print(f"    score={s:4d} {oid[:50]}...{marker}")

    # Step 3: Create migration database and write mappings
    if verbosity >= 1:
        print(f"\nCreating migration database: {migration_db}")

    if migration_db not in server:
        server.create(migration_db)
    mig_db = server[migration_db]

    # Write mapping documents in batches
    if verbosity >= 1:
        print("Writing mapping documents...")

    now = datetime.now(timezone.utc).isoformat()
    batch = []
    written = 0

    for old_id, (new_id, doc) in old_to_new.items():
        group_size = len(new_id_groups[new_id])
        is_winner = (winners[new_id] == old_id)

        ingest = doc.get('ingest', {})
        mapping_doc = {
            '_id': old_id,
            'new_id': new_id,
            'is_winner': is_winner,
            'duplicate_group_size': group_size,
            'source_db': taxa_db,
            'migrated_at': now,
            'old_ingest_id': ingest.get('_id', '') if isinstance(ingest, dict) else '',
            'old_url': ingest.get('url', '') if isinstance(ingest, dict) else '',
        }

        # Check if mapping already exists (idempotent)
        if old_id in mig_db:
            existing = mig_db[old_id]
            mapping_doc['_rev'] = existing['_rev']

        batch.append(mapping_doc)

        if len(batch) >= batch_size:
            mig_db.update(batch)
            written += len(batch)
            if verbosity >= 2:
                print(f"  Written {written}/{total}...")
            batch = []

    if batch:
        mig_db.update(batch)
        written += len(batch)

    # Step 4: Create design document with by_new_id view
    design_doc_id = '_design/migration'
    design_doc = {
        '_id': design_doc_id,
        'views': {
            'by_new_id': {
                'map': 'function(doc) { if (doc.new_id) emit(doc.new_id, { old_id: doc._id, is_winner: doc.is_winner }); }'
            }
        }
    }
    if design_doc_id in mig_db:
        existing = mig_db[design_doc_id]
        design_doc['_rev'] = existing['_rev']
    mig_db.save(design_doc)

    print(f"\nMapping complete: {written} documents written to {migration_db}")
    return {
        'total': total,
        'unique': unique_new_ids,
        'duplicate_groups': len(duplicates),
        'net_reduction': total - unique_new_ids,
        'written': written,
        'errors': errors
    }


def phase_taxa(
    config: dict,
    taxa_db: str,
    migration_db: str,
    dry_run: bool = False,
    batch_size: int = 100,
    verbosity: int = 1
) -> dict:
    """Phase 2: Migrate skol_taxa_dev documents to new content-based IDs.

    Args:
        config: Environment configuration
        taxa_db: Taxa database name
        migration_db: Migration mapping database name
        dry_run: If True, print what would be done
        batch_size: Documents per batch
        verbosity: Verbosity level

    Returns:
        Dict with statistics
    """
    import couchdb

    couchdb_url = config['couchdb_url']
    username = config['couchdb_username']
    password = config['couchdb_password']

    print(f"\n{'='*70}")
    print(f"Phase 2: Migrate {taxa_db}")
    print(f"{'='*70}")
    if dry_run:
        print(f"Mode: DRY RUN")
    print()

    server = couchdb.Server(couchdb_url)
    if username and password:
        server.resource.credentials = (username, password)

    if taxa_db not in server:
        print(f"Error: Database '{taxa_db}' not found")
        return {'error': True}
    if migration_db not in server:
        print(f"Error: Migration database '{migration_db}' not found. Run --phase mapping first.")
        return {'error': True}

    db = server[taxa_db]
    mig_db = server[migration_db]

    # Load all mappings grouped by new_id
    if verbosity >= 1:
        print("Loading migration mappings...")

    # new_id -> [(old_id, is_winner), ...]
    new_id_groups = defaultdict(list)
    mapping_count = 0
    for doc_id in mig_db:
        if doc_id.startswith('_design/'):
            continue
        mapping = mig_db[doc_id]
        new_id = mapping['new_id']
        is_winner = mapping.get('is_winner', False)
        new_id_groups[new_id].append((doc_id, is_winner))
        mapping_count += 1

    if verbosity >= 1:
        print(f"  Loaded {mapping_count} mappings -> {len(new_id_groups)} unique new IDs")

    # Process each group
    created = 0
    deleted = 0
    skipped = 0
    errors = 0

    groups = list(new_id_groups.items())
    total_groups = len(groups)

    for gi, (new_id, members) in enumerate(groups):
        try:
            # Find the winner
            winner_old_id = None
            for old_id, is_winner in members:
                if is_winner:
                    winner_old_id = old_id
                    break
            if winner_old_id is None:
                # Fallback: pick first
                winner_old_id = members[0][0]

            # Check if new_id already exists (already migrated)
            if new_id in db:
                if verbosity >= 2:
                    print(f"  [{gi+1}/{total_groups}] SKIP {new_id[:50]}... (already exists)")
                skipped += 1
                continue

            # Check if winner doc still exists
            if winner_old_id not in db:
                if verbosity >= 2:
                    print(f"  [{gi+1}/{total_groups}] SKIP {new_id[:50]}... (winner {winner_old_id[:30]}... gone)")
                skipped += 1
                continue

            # Load winner document
            winner_doc = db[winner_old_id]

            # Build new document
            new_doc = dict(winner_doc)
            new_doc['_id'] = new_id
            if '_rev' in new_doc:
                del new_doc['_rev']

            # Build ingest_sources from all members
            if len(members) > 1:
                ingest_sources = []
                for old_id, _ in members:
                    try:
                        if old_id in db:
                            old_doc = db[old_id]
                            ingest_sources.append({
                                'old_id': old_id,
                                'ingest': old_doc.get('ingest', {}),
                            })
                    except Exception:
                        ingest_sources.append({'old_id': old_id, 'ingest': {}})
                new_doc['ingest_sources'] = ingest_sources

            if dry_run:
                created += 1
                deleted += len(members)
                if verbosity >= 2:
                    print(f"  [{gi+1}/{total_groups}] WOULD migrate {winner_old_id[:40]}... -> {new_id[:40]}...")
                continue

            # Save new document
            set_timestamps(new_doc, is_new=True)
            db.save(new_doc)
            created += 1

            # Delete all old documents in the group
            for old_id, _ in members:
                if old_id == new_id:
                    # The new doc overwrote this one (shouldn't happen, but be safe)
                    continue
                try:
                    if old_id in db:
                        old_doc = db[old_id]
                        db.delete(old_doc)
                        deleted += 1
                except Exception as e:
                    if verbosity >= 1:
                        print(f"    Warning: Could not delete {old_id}: {e}")

            if verbosity >= 2 and (gi + 1) % 100 == 0:
                print(f"  [{gi+1}/{total_groups}] Created {created}, deleted {deleted}...")

        except Exception as e:
            errors += 1
            print(f"  [{gi+1}/{total_groups}] ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"Taxa Migration {'(DRY RUN) ' if dry_run else ''}Complete")
    print(f"{'='*70}")
    print(f"Groups processed: {total_groups}")
    print(f"Documents created: {created}")
    print(f"Documents deleted: {deleted}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    return {
        'created': created,
        'deleted': deleted,
        'skipped': skipped,
        'errors': errors
    }


def phase_taxa_full(
    config: dict,
    taxa_full_db: str,
    migration_db: str,
    dry_run: bool = False,
    batch_size: int = 100,
    verbosity: int = 1
) -> dict:
    """Phase 3: Migrate skol_taxa_full_dev documents to new content-based IDs.

    Args:
        config: Environment configuration
        taxa_full_db: Taxa full database name
        migration_db: Migration mapping database name
        dry_run: If True, print what would be done
        batch_size: Documents per batch
        verbosity: Verbosity level

    Returns:
        Dict with statistics
    """
    import couchdb

    couchdb_url = config['couchdb_url']
    username = config['couchdb_username']
    password = config['couchdb_password']

    print(f"\n{'='*70}")
    print(f"Phase 3: Migrate {taxa_full_db}")
    print(f"{'='*70}")
    if dry_run:
        print(f"Mode: DRY RUN")
    print()

    server = couchdb.Server(couchdb_url)
    if username and password:
        server.resource.credentials = (username, password)

    if taxa_full_db not in server:
        print(f"Error: Database '{taxa_full_db}' not found")
        return {'error': True}
    if migration_db not in server:
        print(f"Error: Migration database '{migration_db}' not found. Run --phase mapping first.")
        return {'error': True}

    db = server[taxa_full_db]
    mig_db = server[migration_db]

    # Collect all doc IDs first
    doc_ids = []
    for doc_id in db:
        if not doc_id.startswith('_design/'):
            doc_ids.append(doc_id)

    total = len(doc_ids)
    if verbosity >= 1:
        print(f"Found {total} documents in {taxa_full_db}")

    migrated = 0
    skipped = 0
    errors = 0

    for i, old_id in enumerate(doc_ids):
        try:
            # Look up mapping
            if old_id not in mig_db:
                if verbosity >= 1:
                    print(f"  [{i+1}/{total}] SKIP {old_id[:50]}... (no mapping found)")
                skipped += 1
                continue

            mapping = mig_db[old_id]
            new_id = mapping['new_id']

            # Already migrated?
            if old_id == new_id:
                skipped += 1
                continue

            if new_id in db:
                # New ID already exists, just delete the old one
                if not dry_run:
                    old_doc = db[old_id]
                    db.delete(old_doc)
                skipped += 1
                continue

            if old_id not in db:
                skipped += 1
                continue

            doc = db[old_id]

            # Build new document
            new_doc = dict(doc)
            new_doc['_id'] = new_id
            if '_rev' in new_doc:
                del new_doc['_rev']

            if dry_run:
                migrated += 1
                if verbosity >= 2:
                    print(f"  [{i+1}/{total}] WOULD migrate {old_id[:40]}... -> {new_id[:40]}...")
                continue

            # Save new, delete old
            set_timestamps(new_doc, is_new=True)
            db.save(new_doc)
            old_doc = db[old_id]
            db.delete(old_doc)
            migrated += 1

            if verbosity >= 2 and (i + 1) % 100 == 0:
                print(f"  [{i+1}/{total}] Migrated {migrated}...")

        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{total}] ERROR {old_id[:50]}...: {e}")

    print(f"\n{'='*70}")
    print(f"Taxa Full Migration {'(DRY RUN) ' if dry_run else ''}Complete")
    print(f"{'='*70}")
    print(f"Total documents: {total}")
    print(f"Migrated: {migrated}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    return {'total': total, 'migrated': migrated, 'skipped': skipped, 'errors': errors}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Migrate skol_taxa_dev to content-based document IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  mapping   - Build old_id -> new_id mapping in migration database
  taxa      - Migrate skol_taxa_dev documents to new IDs
  taxa_full - Migrate skol_taxa_full_dev documents to new IDs
  all       - Run all phases in order

Examples:
    # Dry run to see deduplication stats
    python fixes/migrate_taxa_ids.py --phase mapping --dry-run

    # Run all phases on dev
    python fixes/migrate_taxa_ids.py --phase all

    # Run on production
    python fixes/migrate_taxa_ids.py --phase all \\
        --taxa-db skol_taxa --taxa-full-db skol_taxa_full \\
        --migration-db skol_taxa_migration
"""
    )

    parser.add_argument(
        '--phase',
        choices=['mapping', 'taxa', 'taxa_full', 'all'],
        required=True,
        help='Which migration phase to run'
    )
    parser.add_argument(
        '--taxa-db',
        default='skol_taxa_dev',
        help='Taxa database name (default: skol_taxa_dev)'
    )
    parser.add_argument(
        '--taxa-full-db',
        default='skol_taxa_full_dev',
        help='Taxa full database name (default: skol_taxa_full_dev)'
    )
    parser.add_argument(
        '--migration-db',
        default='skol_taxa_migration_dev',
        help='Migration mapping database name (default: skol_taxa_migration_dev)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Documents per batch (default: 100)'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Verbosity level: 0=silent, 1=info, 2=debug (default: 1)'
    )

    args = parser.parse_args()
    config = get_env_config()

    try:
        phases = ['mapping', 'taxa', 'taxa_full'] if args.phase == 'all' else [args.phase]
        all_results = {}

        for phase in phases:
            if phase == 'mapping':
                result = phase_mapping(
                    config=config,
                    taxa_db=args.taxa_db,
                    taxa_full_db=args.taxa_full_db,
                    migration_db=args.migration_db,
                    dry_run=args.dry_run,
                    batch_size=args.batch_size,
                    verbosity=args.verbosity
                )
            elif phase == 'taxa':
                result = phase_taxa(
                    config=config,
                    taxa_db=args.taxa_db,
                    migration_db=args.migration_db,
                    dry_run=args.dry_run,
                    batch_size=args.batch_size,
                    verbosity=args.verbosity
                )
            elif phase == 'taxa_full':
                result = phase_taxa_full(
                    config=config,
                    taxa_full_db=args.taxa_full_db,
                    migration_db=args.migration_db,
                    dry_run=args.dry_run,
                    batch_size=args.batch_size,
                    verbosity=args.verbosity
                )

            all_results[phase] = result
            if result.get('error'):
                print(f"\nPhase '{phase}' failed. Stopping.")
                sys.exit(1)

        if args.phase == 'all':
            print(f"\n{'='*70}")
            print("All Phases Complete")
            print(f"{'='*70}")
            for phase, result in all_results.items():
                print(f"  {phase}: {result}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
