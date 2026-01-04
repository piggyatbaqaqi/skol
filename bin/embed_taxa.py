#!/usr/bin/env python3
"""
Compute sBERT Embeddings for Taxa and Save to Redis

This standalone program loads taxa descriptions from CouchDB, computes sBERT
embeddings using the EmbeddingsComputer, and saves them to Redis.

Usage:
    python embed_taxa.py [--force] [--verbosity LEVEL] [--expire SECONDS]

Example:
    python embed_taxa.py --force --verbosity 2
    python embed_taxa.py --expire 604800  # 7 days
"""

import argparse
import os
import sys
from typing import Dict, Any

import redis

from dr_drafts_mycosearch.data import SKOL_TAXA
from dr_drafts_mycosearch.compute_embeddings import EmbeddingsComputer


# ============================================================================
# Environment Configuration
# ============================================================================

def get_env_config() -> Dict[str, Any]:
    """
    Get environment configuration from environment variables or defaults.

    Returns:
        Dictionary of configuration values
    """
    return {
        # CouchDB settings
        'couchdb_host': os.environ.get('COUCHDB_HOST', '127.0.0.1:5984'),
        'couchdb_username': os.environ.get('COUCHDB_USER', 'admin'),
        'couchdb_password': os.environ.get('COUCHDB_PASSWORD', 'SU2orange!'),
        'taxon_db_name': os.environ.get('TAXON_DB_NAME', 'skol_taxa_dev'),

        # Redis settings
        'redis_host': os.environ.get('REDIS_HOST', 'localhost'),
        'redis_port': int(os.environ.get('REDIS_PORT', '6379')),

        # Embedding settings
        'embedding_name': os.environ.get('EMBEDDING_NAME', 'skol:embedding:v1.1'),
        'embedding_expire': int(os.environ.get('EMBEDDING_EXPIRE', str(60 * 60 * 24 * 2))),  # 2 days default
    }


# ============================================================================
# Embedding Functions
# ============================================================================

def compute_and_save_embeddings(
    config: Dict[str, Any],
    force: bool = False,
    verbosity: int = 1,
    expire_override: int = None
) -> None:
    """
    Load taxa descriptions from CouchDB, compute embeddings, and save to Redis.

    Args:
        config: Environment configuration
        force: If True, compute embeddings even if they already exist in Redis
        verbosity: Verbosity level (0=silent, 1=info, 2=debug)
        expire_override: Optional expiration time override (seconds)
    """
    # Determine expiration time
    embedding_expire = expire_override if expire_override is not None else config['embedding_expire']

    # Build CouchDB URL
    couchdb_url = f"http://{config['couchdb_host']}"

    # Build Redis URL
    redis_url = f"redis://{config['redis_host']}:{config['redis_port']}"

    print(f"\n{'='*70}")
    print(f"Computing Taxa Embeddings")
    print(f"{'='*70}")
    print(f"CouchDB: {couchdb_url}")
    print(f"Database: {config['taxon_db_name']}")
    print(f"Redis: {redis_url}")
    print(f"Embedding key: {config['embedding_name']}")
    if embedding_expire:
        print(f"Expiration: {embedding_expire} seconds ({embedding_expire / 86400:.1f} days)")
    else:
        print(f"Expiration: None (never expires)")
    print()

    # Connect to Redis
    redis_client = redis.Redis(
        host=config['redis_host'],
        port=config['redis_port'],
        decode_responses=False  # We need bytes for embedding serialization
    )

    # Test Redis connection
    try:
        redis_client.ping()
        if verbosity >= 1:
            print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        sys.exit(1)

    # Check if embeddings already exist
    if not force and redis_client.exists(config['embedding_name']):
        print(f"\n✓ Embeddings already exist in Redis: {config['embedding_name']}")
        print(f"  Use --force to recompute them")
        return

    # Load descriptions from CouchDB
    if verbosity >= 1:
        print("\nLoading taxa descriptions from CouchDB...")

    try:
        skol_taxa = SKOL_TAXA(
            couchdb_url=couchdb_url,
            username=config['couchdb_username'],
            password=config['couchdb_password'],
            db_name=config['taxon_db_name']
        )
        descriptions = skol_taxa.get_descriptions()

        if verbosity >= 1:
            print(f"✓ Loaded {len(descriptions)} taxa descriptions")

        if len(descriptions) == 0:
            print("\n⚠ No descriptions found. Nothing to embed.")
            return

        # Show sample if verbose
        if verbosity >= 2:
            print("\nSample descriptions:")
            for i, desc in enumerate(descriptions[:3]):
                print(f"  {i+1}. {desc[:100]}...")

    except Exception as e:
        print(f"✗ Failed to load descriptions from CouchDB: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Compute embeddings
    if verbosity >= 1:
        print("\nComputing embeddings...")

    try:
        embedder = EmbeddingsComputer(
            idir='/dev/null',  # Not used when providing descriptions directly
            redis_url=redis_url,
            redis_expire=embedding_expire,
            embedding_name=config['embedding_name'],
        )

        embedding_result = embedder.run(descriptions)

        print(f"\n{'='*70}")
        print("Embedding Complete!")
        print(f"{'='*70}")
        print(f"✓ Embeddings saved to Redis")
        print(f"  Key: {config['embedding_name']}")
        print(f"  Number of embeddings: {len(descriptions)}")
        if hasattr(embedding_result, 'shape'):
            print(f"  Embedding shape: {embedding_result.shape}")

    except Exception as e:
        print(f"✗ Failed to compute embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# Main Program
# ============================================================================

def main():
    """Main entry point for the embedding program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute sBERT embeddings for taxa and save to Redis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  COUCHDB_HOST          CouchDB host (default: 127.0.0.1:5984)
  COUCHDB_USER          CouchDB username (default: admin)
  COUCHDB_PASSWORD      CouchDB password (default: SU2orange!)
  TAXON_DB_NAME         Taxa database name (default: skol_taxa_dev)
  REDIS_HOST            Redis host (default: localhost)
  REDIS_PORT            Redis port (default: 6379)
  EMBEDDING_NAME        Redis key for embeddings (default: skol:embedding:v1.1)
  EMBEDDING_EXPIRE      Expiration time in seconds (default: 172800 = 2 days)
"""
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Recompute embeddings even if they already exist in Redis'
    )

    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Verbosity level (0=silent, 1=info, 2=debug, default: 1)'
    )

    parser.add_argument(
        '--expire',
        type=int,
        default=None,
        metavar='SECONDS',
        help='Override expiration time in seconds (default: 172800 = 2 days, 0 = never)'
    )

    args = parser.parse_args()

    # Get configuration
    config = get_env_config()

    # Run embedding computation
    try:
        compute_and_save_embeddings(
            config=config,
            force=args.force,
            verbosity=args.verbosity,
            expire_override=args.expire
        )
    except KeyboardInterrupt:
        print("\n\n✗ Embedding computation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Embedding computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
