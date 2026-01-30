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
import sys
from typing import Dict, Any
from pathlib import Path

import redis

# Add parent directory to path for skol modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Python 3.11+ compatibility: Apply formatargspec shim before importing ML libraries
import skol_compat  # noqa: F401 (imported for side effects)

from dr_drafts_mycosearch.data import SKOL_TAXA
from dr_drafts_mycosearch.compute_embeddings import EmbeddingsComputer
from env_config import get_env_config


# ============================================================================
# Locking Constants (shared with Django views)
# ============================================================================

LOCK_KEY = 'skol:build:embedding:lock'
LOCK_TTL = 660  # 11 minutes


# ============================================================================
# Embedding Functions
# ============================================================================

def compute_and_save_embeddings(
    config: Dict[str, Any],
    force: bool = False,
    verbosity: int = 1,
    expire_override: int = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> None:
    """
    Load taxa descriptions from CouchDB, compute embeddings, and save to Redis.

    Args:
        config: Environment configuration
        force: If True, compute embeddings even if they already exist in Redis
        verbosity: Verbosity level (0=silent, 1=info, 2=debug)
        expire_override: Optional expiration time override (seconds)
        dry_run: If True, show what would be computed without saving
        skip_existing: If True, skip if embeddings already exist (default behavior)
    """
    # Determine expiration time
    embedding_expire = expire_override if expire_override is not None else config['embedding_expire']

    # Build CouchDB URL
    couchdb_url = f"http://{config['couchdb_host']}"

    # Build Redis URL
    redis_url = f"redis://{config['redis_host']}:{config['redis_port']}"

    if verbosity >= 2:
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
        if dry_run:
            print(f"Mode: DRY RUN (no changes will be saved)")
        if force:
            print(f"Mode: FORCE (recompute even if exists)")
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
    embedding_exists = redis_client.exists(config['embedding_name'])
    if skip_existing and not force and embedding_exists:
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
            db_name=config['taxon_db_name'],
            verbosity=verbosity
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
    if dry_run:
        print(f"\n[DRY RUN] Would compute embeddings for {len(descriptions)} taxa")
        print(f"[DRY RUN] Would save to Redis key: {config['embedding_name']}")
        if embedding_expire:
            print(f"[DRY RUN] With expiration: {embedding_expire} seconds")
        if embedding_exists:
            print(f"[DRY RUN] Would overwrite existing embeddings")
        return

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

def acquire_lock(config: Dict[str, Any], verbosity: int = 1) -> redis.Redis:
    """
    Acquire the build lock to prevent concurrent builds.

    Args:
        config: Environment configuration with Redis settings
        verbosity: Verbosity level

    Returns:
        Redis client if lock acquired, None if another build is running

    Raises:
        SystemExit: If lock cannot be acquired (another build in progress)
    """
    redis_client = redis.Redis(
        host=config['redis_host'],
        port=config['redis_port'],
        decode_responses=True
    )

    # Try to acquire lock (SETNX = SET if Not eXists)
    lock_acquired = redis_client.set(LOCK_KEY, 'building', nx=True, ex=LOCK_TTL)

    if not lock_acquired:
        if verbosity >= 1:
            print(f"✓ Another embedding build is already in progress (lock: {LOCK_KEY})")
            print("  Exiting gracefully. Try again later or check the other process.")
        sys.exit(0)  # Exit with success - not an error, just already running

    if verbosity >= 2:
        print(f"✓ Acquired build lock: {LOCK_KEY} (TTL: {LOCK_TTL}s)")

    return redis_client


def release_lock(redis_client: redis.Redis, verbosity: int = 1) -> None:
    """
    Release the build lock.

    Args:
        redis_client: Redis client
        verbosity: Verbosity level
    """
    try:
        redis_client.delete(LOCK_KEY)
        if verbosity >= 2:
            print(f"✓ Released build lock: {LOCK_KEY}")
    except Exception as e:
        # Don't fail if we can't release - TTL will handle it
        if verbosity >= 1:
            print(f"⚠ Could not release lock (will auto-expire): {e}")


def main():
    """Main entry point for the embedding program."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute sBERT embeddings for taxa and save to Redis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Work Control Options (from env_config):
  --dry-run             Preview what would be computed without saving
  --skip-existing       Skip if embeddings already exist (default behavior)
  --force               Recompute even if embeddings exist (overrides --skip-existing)

Environment Variables:
  DRY_RUN=1             Same as --dry-run
  SKIP_EXISTING=1       Same as --skip-existing (default)
  FORCE=1               Same as --force
  COUCHDB_HOST          CouchDB host (default: 127.0.0.1:5984)
  COUCHDB_USER          CouchDB username (default: admin)
  COUCHDB_PASSWORD      CouchDB password (default: SU2orange!)
  TAXON_DB_NAME         Taxa database name (default: skol_taxa_dev)
  REDIS_HOST            Redis host (default: localhost)
  REDIS_PORT            Redis port (default: 6379)
  EMBEDDING_NAME        Redis key for embeddings (default: skol:embedding:v1.1)
  EMBEDDING_EXPIRE      Expiration time in seconds (default: 172800 = 2 days)

Examples:
  python embed_taxa.py --force                  # Recompute with default expiration
  python embed_taxa.py --dry-run                # Preview without saving
  python embed_taxa.py --expire 604800          # Expire after 7 days
  python embed_taxa.py --expire None            # Never expire
  python embed_taxa.py --expire 0               # Never expire (same as None)
"""
    )

    def parse_expire(value):
        """Parse expire argument: integer seconds or 'None' for no expiration."""
        if value.lower() == 'none':
            return None
        try:
            expire_val = int(value)
            if expire_val < 0:
                raise argparse.ArgumentTypeError("expire must be non-negative or 'None'")
            return expire_val
        except ValueError:
            raise argparse.ArgumentTypeError(f"expire must be an integer or 'None', got: {value}")

    parser.add_argument(
        '--expire',
        type=parse_expire,
        default=argparse.SUPPRESS,  # Don't set attribute if not provided
        metavar='SECONDS',
        help='Expiration time in seconds, or "None" for no expiration (default: 172800 = 2 days)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be computed without saving'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip if embeddings already exist (default behavior)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Recompute embeddings even if they already exist in Redis'
    )

    args, _ = parser.parse_known_args()

    # Get configuration
    config = get_env_config()

    # Handle expire argument:
    # - If user didn't provide --expire: attribute doesn't exist, use config default
    # - If user provided --expire None: args.expire == None, explicitly set no expiration
    # - If user provided --expire N: args.expire == N, use that value
    expire_override = getattr(args, 'expire', None)

    # Merge work control options from command-line args and env_config
    # Command-line args take precedence over env_config
    dry_run = args.dry_run or config.get('dry_run', False)
    skip_existing = args.skip_existing or config.get('skip_existing', True)  # Default to True for this script
    force = args.force or config.get('force', False)

    verbosity = config['verbosity']

    # Acquire build lock (exits gracefully if another build is running)
    # Skip lock for dry-run since we're not actually building anything
    redis_client = None
    if not dry_run:
        redis_client = acquire_lock(config, verbosity)

    # Run embedding computation
    try:
        compute_and_save_embeddings(
            config=config,
            force=force,
            verbosity=verbosity,
            expire_override=expire_override,
            dry_run=dry_run,
            skip_existing=skip_existing,
        )
    except KeyboardInterrupt:
        print("\n\n✗ Embedding computation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Embedding computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Always release the lock
        if redis_client:
            release_lock(redis_client, verbosity)


if __name__ == '__main__':
    main()
