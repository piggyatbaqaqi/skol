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

from dr_drafts_mycosearch.data import SKOL_TAXA, SKOL_COLLECTIONS
from dr_drafts_mycosearch.compute_embeddings import EmbeddingsComputer
from env_config import get_env_config, create_redis_client, build_redis_url
import pandas as pd


# ============================================================================
# Locking Constants (shared with Django views)
# ============================================================================

LOCK_KEY = 'skol:build:embedding:lock'
LOCK_TTL = 7260  # 121 minutes


# ============================================================================
# Redis Helpers
# ============================================================================

# 64 MB chunks — well under typical TLS buffer limits
_CHUNK_SIZE = 64 * 1024 * 1024


def redis_set_chunked(
    r: redis.Redis,
    key: str,
    data: bytes,
    chunk_size: int = _CHUNK_SIZE,
    expire: int | None = None,
    verbosity: int = 1,
) -> None:
    """Write a large value to Redis using SETRANGE to avoid TLS limits.

    Falls back to plain SET for values smaller than *chunk_size*.
    """
    if len(data) <= chunk_size:
        r.set(key, data)
    else:
        # Delete first so SETRANGE doesn't append to old data
        r.delete(key)
        offset = 0
        while offset < len(data):
            end = min(offset + chunk_size, len(data))
            r.setrange(key, offset, data[offset:end])
            if verbosity >= 2:
                pct = end * 100 // len(data)
                print(f"  Writing {key}: {pct}%")
            offset = end
    if expire is not None and expire > 0:
        r.expire(key, expire)


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
    include_collections: bool = True,
    precision: str = "float32",
    backend: str = "torch",
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
        include_collections: If True, include user collections in embeddings
    """
    # Determine expiration time
    embedding_expire = expire_override if expire_override is not None else config['embedding_expire']

    # Build CouchDB URL
    couchdb_url = f"http://{config['couchdb_host']}"

    # Build Redis URL (respects REDIS_TLS setting)
    redis_url = build_redis_url()

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

    # Connect to Redis (respects REDIS_TLS and REDIS_PASSWORD settings)
    redis_client = create_redis_client(decode_responses=False)

    # Test Redis connection
    try:
        redis_client.ping()
        if verbosity >= 1:
            print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        sys.exit(1)

    # Check if embeddings already exist
    embedding_name = config['embedding_name']
    embedding_exists = redis_client.exists(embedding_name)

    # Back up existing embeddings before overwriting.
    # The backup key uses the same skol:embedding:* pattern so the
    # UI auto-discovers it as an available embedding (easy rollback).
    if embedding_exists and not dry_run:
        backup_key = f'{embedding_name}:backup'
        existing_data: bytes | None = redis_client.get(embedding_name)  # type: ignore[assignment]
        if existing_data:
            # Store backup without expiration for durability
            redis_set_chunked(
                redis_client, backup_key, existing_data,
                verbosity=verbosity,
            )
            if verbosity >= 1:
                size_mb = len(existing_data) / (1024 * 1024)
                print(
                    f"✓ Backed up existing embeddings to "
                    f"{backup_key} ({size_mb:.1f} MB)"
                )

    if skip_existing and not force and embedding_exists:
        print(f"\n✓ Embeddings already exist in Redis: {embedding_name}")
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

        # Load collections if enabled
        if include_collections:
            collections_db = config.get('collections_db_name', 'skol_collections_dev')
            if verbosity >= 1:
                print(f"\nLoading user collections from {collections_db}...")

            try:
                skol_collections = SKOL_COLLECTIONS(
                    couchdb_url=couchdb_url,
                    username=config['couchdb_username'],
                    password=config['couchdb_password'],
                    db_name=collections_db,
                    verbosity=verbosity
                )
                collection_descriptions = skol_collections.get_descriptions()

                if len(collection_descriptions) > 0:
                    if verbosity >= 1:
                        print(f"✓ Loaded {len(collection_descriptions)} collection descriptions")

                    # Concatenate taxa and collections
                    descriptions = pd.concat(
                        [descriptions, collection_descriptions],
                        ignore_index=True
                    )
                    if verbosity >= 1:
                        print(f"✓ Combined total: {len(descriptions)} descriptions")
                else:
                    if verbosity >= 1:
                        print("  No public collections found")

            except Exception as e:
                if verbosity >= 1:
                    print(f"⚠ Could not load collections (continuing with taxa only): {e}")

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
            redis_username=config['redis_username'],
            redis_password=config['redis_password'],
            redis_expire=embedding_expire,
            embedding_name=config['embedding_name'],
            precision=precision,
            backend=backend,
        )

        try:
            embedding_result = embedder.run(descriptions)
        except Exception as save_err:
            # The library's run() computes then saves via r.set().
            # If the save fails (e.g. TLS buffer limit on large
            # payloads), fall back to chunked SETRANGE writes.
            if embedder.result is None:
                raise  # Computation itself failed
            if verbosity >= 1:
                print(
                    f"  Library save failed ({save_err}), "
                    "retrying with chunked write..."
                )
            import pickle
            pickled = pickle.dumps(embedder.result)
            redis_set_chunked(
                redis_client, config['embedding_name'],
                pickled, expire=embedding_expire,
                verbosity=verbosity,
            )
            embedding_result = embedder.result

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
    # Use create_redis_client for proper TLS and auth configuration
    redis_client = create_redis_client(decode_responses=True)

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

    parser.add_argument(
        '--include-collections',
        action='store_true',
        default=True,
        help='Include user collections in embeddings (default: True)'
    )

    parser.add_argument(
        '--no-collections',
        action='store_true',
        help='Exclude user collections from embeddings'
    )

    parser.add_argument(
        '--skip-lock',
        action='store_true',
        help='Skip lock acquisition (caller already holds the lock)'
    )

    parser.add_argument(
        '--precision',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'int8', 'binary'],
        help='Embedding precision (default: float32)',
    )

    parser.add_argument(
        '--backend',
        type=str,
        default='torch',
        choices=['torch', 'onnx'],
        help='SentenceTransformer backend (default: torch)',
    )

    args, _ = parser.parse_known_args()

    # Get configuration
    config = get_env_config()

    # Handle expire argument:
    # - If user didn't provide --expire: attribute doesn't exist, use config default
    # - If user provided --expire None: args.expire == None, explicitly no expiration
    # - If user provided --expire N: args.expire == N, use that value
    _UNSET = object()
    expire_override = getattr(args, 'expire', _UNSET)
    if expire_override is _UNSET:
        expire_override = config['embedding_expire']

    # Merge work control options from command-line args and env_config
    # Command-line args take precedence over env_config
    dry_run = args.dry_run or config.get('dry_run', False)
    skip_existing = args.skip_existing or config.get('skip_existing', True)  # Default to True for this script
    force = args.force or config.get('force', False)
    include_collections = not args.no_collections  # Default True unless --no-collections

    verbosity = config['verbosity']

    # Acquire build lock (exits gracefully if another build is running)
    # Skip lock for dry-run since we're not actually building anything
    redis_client = None
    if not dry_run:
        if args.skip_lock:
            # Caller already holds the lock; get a client to release it later
            redis_client = create_redis_client(decode_responses=True)
        else:
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
            include_collections=include_collections,
            precision=args.precision,
            backend=args.backend,
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
