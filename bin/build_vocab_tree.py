#!/usr/bin/env python3
"""
Build Vocabulary Tree for UI Menus

This script reads JSON representations from CouchDB and organizes all vocabulary
into an efficient tree data structure for retrieving successive vocabularies
at each level. The tree is saved to Redis with a versioned key.

The tree structure:
- Level 1: All tokens appearing as top-level keys in JSON representations
- Level 2: Under each level1 token, all tokens appearing as level2 keys
- Continue recursively to the leaves (array values)

Usage:
    python bin/build_vocab_tree.py [options]

Examples:
    # Use defaults (skol_taxa_full_dev database)
    python bin/build_vocab_tree.py

    # Specify database
    python bin/build_vocab_tree.py --db skol_taxa_full

    # Custom Redis connection
    python bin/build_vocab_tree.py --redis-host localhost --redis-port 6379

    # Dry run (don't save to Redis)
    python bin/build_vocab_tree.py --dry-run
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import couchdb
import redis

# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import create_redis_client


# ============================================================================
# Locking Constants (shared with Django views)
# ============================================================================

LOCK_KEY = 'skol:build:vocab_tree:lock'
LOCK_TTL = 360  # 6 minutes


class VocabularyTree:
    """
    A tree data structure for organizing JSON vocabulary by levels.

    The tree supports efficient retrieval of vocabulary at any path:
    - get_children([]) returns all level-1 terms
    - get_children(["pileus"]) returns all level-2 terms under "pileus"
    - get_children(["pileus", "shape"]) returns all level-3 terms under pileus.shape

    Attributes:
        tree: Nested dict representing the vocabulary hierarchy
        stats: Statistics about the tree (depth, node counts, etc.)
    """

    def __init__(self):
        """Initialize an empty vocabulary tree."""
        self.tree: Dict[str, Any] = {}
        self.stats = {
            "total_nodes": 0,
            "max_depth": 0,
            "level_counts": defaultdict(int),
            "leaf_count": 0,
        }

    def _is_valid_term(self, term: str) -> bool:
        """
        Check if a term is valid vocabulary (not garbage/syntax).

        Filters out:
        - Punctuation-only strings
        - JSON syntax fragments
        - Very long strings (likely raw JSON or code)
        - Strings with too many special characters

        Args:
            term: The term to validate

        Returns:
            True if term is valid vocabulary
        """
        import re

        if not term or not term.strip():
            return False

        stripped = term.strip()

        # Skip very long strings (likely malformed JSON)
        if len(stripped) > 100:
            return False

        # Skip strings that look like raw JSON
        if stripped.startswith('{') or stripped.startswith('['):
            return False
        if '":' in stripped or '": ' in stripped:
            return False

        # Skip strings that are mostly punctuation/syntax
        if re.match(r'^[\[\]{}():,;\s\-\.\"\'`><=]+$', stripped):
            return False

        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', stripped):
            return False

        # Skip if more than 30% of chars are brackets/quotes
        syntax_chars = sum(1 for c in stripped if c in '[]{}()":,;')
        if len(stripped) > 0 and syntax_chars / len(stripped) > 0.3:
            return False

        return True

    def add_json(self, data: Any, path: Optional[List[str]] = None, depth: int = 0) -> None:
        """
        Add a JSON structure to the vocabulary tree.

        Recursively traverses the JSON and adds all keys and values to the tree
        at their appropriate positions.

        Args:
            data: JSON data (dict, list, or primitive)
            path: Current path in the tree (for recursion)
            depth: Current depth (for statistics)
        """
        if path is None:
            path = []

        if isinstance(data, dict):
            for key, value in data.items():
                # Normalize key to lowercase
                normalized_key = key.lower().strip() if isinstance(key, str) else str(key)

                # Skip invalid keys (garbage, JSON fragments, etc.)
                if not self._is_valid_term(normalized_key):
                    continue

                # Navigate to the current position in the tree
                current = self.tree
                for p in path:
                    if p not in current:
                        current[p] = {}
                    current = current[p]

                # Add this key if not present
                if normalized_key not in current:
                    current[normalized_key] = {}
                    self.stats["total_nodes"] += 1
                    self.stats["level_counts"][depth + 1] += 1

                # Recurse into the value
                self.add_json(value, path + [normalized_key], depth + 1)

        elif isinstance(data, list):
            for item in data:
                self.add_json(item, path, depth)

        elif isinstance(data, str):
            # Leaf value - add to tree
            normalized_value = data.lower().strip()

            # Skip invalid values
            if not self._is_valid_term(normalized_value):
                return

            # Navigate to current position
            current = self.tree
            for p in path:
                if p not in current:
                    current[p] = {}
                current = current[p]

            # Add leaf value
            if normalized_value not in current:
                current[normalized_value] = {}
                self.stats["total_nodes"] += 1
                self.stats["level_counts"][depth + 1] += 1
                self.stats["leaf_count"] += 1

            # Track max depth
            if depth + 1 > self.stats["max_depth"]:
                self.stats["max_depth"] = depth + 1

    def get_children(self, path: List[str]) -> List[str]:
        """
        Get all children (vocabulary) at the given path.

        Args:
            path: List of keys representing the path in the tree
                  Empty list returns top-level vocabulary

        Returns:
            Sorted list of child keys at the given path
        """
        current = self.tree
        for p in path:
            normalized_p = p.lower().strip()
            if normalized_p not in current:
                return []
            current = current[normalized_p]

        return sorted(current.keys())

    def get_subtree(self, path: List[str]) -> Dict[str, Any]:
        """
        Get the subtree rooted at the given path.

        Args:
            path: List of keys representing the path

        Returns:
            Subtree dict at the given path, or empty dict if path not found
        """
        current = self.tree
        for p in path:
            normalized_p = p.lower().strip()
            if normalized_p not in current:
                return {}
            current = current[normalized_p]
        return current

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tree to a dictionary for serialization.

        Returns:
            Dictionary with 'tree' and 'stats' keys
        """
        return {
            "tree": self.tree,
            "stats": {
                "total_nodes": self.stats["total_nodes"],
                "max_depth": self.stats["max_depth"],
                "level_counts": dict(self.stats["level_counts"]),
                "leaf_count": self.stats["leaf_count"],
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VocabularyTree":
        """
        Create a VocabularyTree from a dictionary.

        Args:
            data: Dictionary with 'tree' and 'stats' keys

        Returns:
            VocabularyTree instance
        """
        tree = cls()
        tree.tree = data.get("tree", {})
        stats = data.get("stats", {})
        tree.stats["total_nodes"] = stats.get("total_nodes", 0)
        tree.stats["max_depth"] = stats.get("max_depth", 0)
        tree.stats["level_counts"] = defaultdict(int, stats.get("level_counts", {}))
        tree.stats["leaf_count"] = stats.get("leaf_count", 0)
        return tree


def build_vocabulary_tree(
    couchdb_url: str,
    db_name: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    limit: Optional[int] = None,
    verbosity: int = 1
) -> VocabularyTree:
    """
    Build a vocabulary tree from JSON representations in CouchDB.

    Args:
        couchdb_url: CouchDB server URL
        db_name: Database name containing json_annotated documents
        username: CouchDB username
        password: CouchDB password
        limit: Maximum number of documents to process (None for all)
        verbosity: Verbosity level (0=quiet, 1=normal, 2=verbose)

    Returns:
        VocabularyTree containing merged vocabulary from all documents
    """
    # Connect to CouchDB
    if verbosity >= 1:
        print(f"Connecting to CouchDB at {couchdb_url}...")

    server = couchdb.Server(couchdb_url)
    if username and password:
        server.resource.credentials = (username, password)

    if db_name not in server:
        raise ValueError(f"Database '{db_name}' not found")

    db = server[db_name]

    # Build the tree
    tree = VocabularyTree()
    processed = 0
    skipped = 0

    if verbosity >= 1:
        print(f"Processing documents from {db_name}...")

    for doc_id in db:
        if doc_id.startswith('_design/'):
            continue

        if limit and processed >= limit:
            break

        try:
            doc = db[doc_id]
            json_annotated = doc.get('json_annotated')

            if not json_annotated or not isinstance(json_annotated, dict):
                skipped += 1
                continue

            # Add this document's JSON to the tree
            tree.add_json(json_annotated)
            processed += 1

            if verbosity >= 2 and processed % 100 == 0:
                print(f"  Processed {processed} documents...")
            elif verbosity >= 1 and processed % 500 == 0:
                print(f"  Processed {processed} documents...")

        except Exception as e:
            if verbosity >= 2:
                print(f"  Warning: Error processing {doc_id}: {e}")
            skipped += 1

    if verbosity >= 1:
        print(f"\nProcessed {processed} documents, skipped {skipped}")
        print("Tree statistics:")
        print(f"  Total nodes: {tree.stats['total_nodes']}")
        print(f"  Max depth: {tree.stats['max_depth']}")
        print(f"  Leaf nodes: {tree.stats['leaf_count']}")
        for level, count in sorted(tree.stats['level_counts'].items()):
            print(f"  Level {level}: {count} terms")

    return tree


def save_to_redis(
    tree: VocabularyTree,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    version: Optional[str] = None,
    ttl: Optional[int] = None,
    verbosity: int = 1
) -> str:
    """
    Save the vocabulary tree to Redis.

    Args:
        tree: VocabularyTree to save
        redis_host: Redis host
        redis_port: Redis port
        redis_db: Redis database number
        redis_password: Redis password (optional)
        version: Version string for the key (default: current timestamp)
        ttl: Time-to-live in seconds (optional, None means no expiration)
        verbosity: Verbosity level

    Returns:
        The Redis key where the tree was saved
    """
    # Generate version string
    if version is None:
        version = datetime.now().strftime("%Y_%m_%d_%H_%M")

    key = f"skol:ui:menus_{version}"

    # Connect to Redis (respects REDIS_TLS settings from env_config)
    if verbosity >= 1:
        print(f"\nConnecting to Redis at {redis_host}:{redis_port}...")

    r = create_redis_client(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=True
    )

    # Test connection
    try:
        r.ping()
    except redis.ConnectionError as e:
        raise ConnectionError(f"Failed to connect to Redis: {e}")

    # Serialize and save
    data = tree.to_dict()
    data["version"] = version
    data["created_at"] = datetime.now().isoformat()

    json_str = json.dumps(data, ensure_ascii=False)

    if verbosity >= 1:
        print(f"Saving tree to Redis key: {key}")
        print(f"  Data size: {len(json_str):,} bytes")
        if ttl:
            print(f"  TTL: {ttl} seconds ({ttl // 3600}h {(ttl % 3600) // 60}m)")

    # Save with optional TTL
    if ttl:
        r.set(key, json_str, ex=ttl)
    else:
        r.set(key, json_str)

    # Verify the data was actually saved before updating the pointer
    if not r.exists(key):
        raise RuntimeError(f"Failed to save vocabulary tree to Redis key: {key}")

    # Update the "latest" pointer
    # IMPORTANT: If data has TTL, pointer must have same TTL so they expire together.
    # A dangling pointer (pointing to expired data) causes confusing errors.
    latest_key = "skol:ui:menus_latest"
    if ttl:
        r.set(latest_key, key, ex=ttl)
        if verbosity >= 1:
            print(f"  WARNING: TTL is set. Both data and pointer will expire in {ttl}s.")
            print("           Run without --ttl for a persistent vocabulary tree.")
    else:
        r.set(latest_key, key)

    if verbosity >= 1:
        print(f"  Updated latest pointer: {latest_key} -> {key}")

    return key


def load_from_redis(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    version: Optional[str] = None
) -> Tuple[VocabularyTree, Dict[str, Any]]:
    """
    Load a vocabulary tree from Redis.

    Args:
        redis_host: Redis host
        redis_port: Redis port
        redis_db: Redis database number
        redis_password: Redis password
        version: Specific version to load (default: latest)

    Returns:
        Tuple of (VocabularyTree, metadata dict)
    """
    r = create_redis_client(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=True
    )

    # Determine key
    if version:
        key = f"skol:ui:menus_{version}"
    else:
        # Get latest
        latest_key = "skol:ui:menus_latest"
        key = r.get(latest_key)
        if not key:
            raise ValueError("No vocabulary tree found in Redis")

    # Load data
    json_str = r.get(key)
    if not json_str:
        raise ValueError(f"Key '{key}' not found in Redis")

    data = json.loads(json_str)
    tree = VocabularyTree.from_dict(data)

    metadata = {
        "key": key,
        "version": data.get("version"),
        "created_at": data.get("created_at"),
    }

    return tree, metadata


def acquire_lock(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    verbosity: int = 1
) -> Optional[redis.Redis]:
    """
    Acquire the build lock to prevent concurrent builds.

    Args:
        redis_host: Redis host
        redis_port: Redis port
        verbosity: Verbosity level

    Returns:
        Redis client if lock acquired

    Raises:
        SystemExit: If lock cannot be acquired (another build in progress)
    """
    redis_client = create_redis_client(
        host=redis_host,
        port=redis_port,
        decode_responses=True
    )

    # Try to acquire lock (SETNX = SET if Not eXists)
    lock_acquired = redis_client.set(LOCK_KEY, 'building', nx=True, ex=LOCK_TTL)

    if not lock_acquired:
        if verbosity >= 1:
            print(f"✓ Another vocab tree build is already in progress (lock: {LOCK_KEY})")
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


def print_tree_sample(tree: VocabularyTree, max_items: int = 10) -> None:
    """
    Print a sample of the tree structure for verification.

    Args:
        tree: VocabularyTree to display
        max_items: Maximum items to show at each level
    """
    print("\n" + "=" * 70)
    print("VOCABULARY TREE SAMPLE")
    print("=" * 70)

    # Get top-level terms
    level1_terms = tree.get_children([])[:max_items]
    print(f"\nLevel 1 terms ({len(tree.get_children([]))} total, showing {len(level1_terms)}):")

    for term in level1_terms:
        level2_terms = tree.get_children([term])
        print(f"\n  {term} ({len(level2_terms)} children)")

        for child in level2_terms[:5]:
            level3_terms = tree.get_children([term, child])
            if level3_terms:
                sample = level3_terms[:3]
                more = f"... +{len(level3_terms) - 3} more" if len(level3_terms) > 3 else ""
                print(f"    {child}: [{', '.join(sample)}{more}]")
            else:
                print(f"    {child}: (leaf)")

        if len(level2_terms) > 5:
            print(f"    ... +{len(level2_terms) - 5} more children")

    if len(tree.get_children([])) > max_items:
        print(f"\n  ... +{len(tree.get_children([])) - max_items} more level-1 terms")

    print("=" * 70)


def main():
    """Main entry point for the vocabulary tree builder."""
    parser = argparse.ArgumentParser(
        description="Build vocabulary tree from JSON representations and save to Redis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use defaults
  %(prog)s --db skol_taxa_full      # Specify database
  %(prog)s --dry-run                # Build tree but don't save
  %(prog)s --show-sample            # Display tree sample after building
        """
    )

    # CouchDB options
    parser.add_argument(
        '--db', '--database',
        default='skol_taxa_full_dev',
        help='CouchDB database name (default: skol_taxa_full_dev)'
    )
    parser.add_argument(
        '--couchdb-url',
        default=os.environ.get('COUCHDB_URL', 'http://localhost:5984'),
        help='CouchDB URL (default: $COUCHDB_URL or http://localhost:5984)'
    )
    parser.add_argument(
        '--couchdb-user',
        default=os.environ.get('COUCHDB_USER', 'admin'),
        help='CouchDB username (default: $COUCHDB_USER or admin)'
    )
    parser.add_argument(
        '--couchdb-password',
        default=os.environ.get('COUCHDB_PASSWORD', ''),
        help='CouchDB password (default: $COUCHDB_PASSWORD)'
    )

    # Redis options
    parser.add_argument(
        '--redis-host',
        default=os.environ.get('REDIS_HOST', 'localhost'),
        help='Redis host (default: $REDIS_HOST or localhost)'
    )
    parser.add_argument(
        '--redis-port',
        type=int,
        default=int(os.environ.get('REDIS_PORT', '6379')),
        help='Redis port (default: $REDIS_PORT or 6379)'
    )
    parser.add_argument(
        '--redis-db',
        type=int,
        default=int(os.environ.get('REDIS_DB', '0')),
        help='Redis database number (default: $REDIS_DB or 0)'
    )
    parser.add_argument(
        '--redis-password',
        default=os.environ.get('REDIS_PASSWORD'),
        help='Redis password (default: $REDIS_PASSWORD)'
    )

    # Processing options
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process'
    )
    parser.add_argument(
        '--version',
        default=None,
        help='Version string for Redis key (default: current timestamp YYYY_MM_DD_HH_MM)'
    )
    parser.add_argument(
        '--ttl',
        type=int,
        default=None,
        help='Time-to-live in seconds for Redis keys (default: no expiration). '
             'Examples: 3600 (1 hour), 86400 (1 day), 604800 (1 week)'
    )

    # Output options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build tree but do not save to Redis'
    )
    parser.add_argument(
        '--show-sample',
        action='store_true',
        help='Display a sample of the tree structure'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Also save tree to a JSON file'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=None,
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity (can be repeated)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress output (same as --verbosity 0)'
    )

    args = parser.parse_args()

    # Set verbosity: --verbosity takes precedence, then -q, then -v
    if args.verbosity is not None:
        verbosity = args.verbosity
    elif args.quiet:
        verbosity = 0
    else:
        verbosity = 1 + args.verbose  # Base level 1, plus any -v flags

    # Acquire build lock (exits gracefully if another build is running)
    # Skip lock for dry-run since we're not actually saving anything
    lock_client = None
    if not args.dry_run:
        lock_client = acquire_lock(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            verbosity=verbosity
        )

    try:
        # Build the tree
        if verbosity >= 1:
            print("=" * 70)
            print("Building Vocabulary Tree")
            print("=" * 70)

        tree = build_vocabulary_tree(
            couchdb_url=args.couchdb_url,
            db_name=args.db,
            username=args.couchdb_user,
            password=args.couchdb_password,
            limit=args.limit,
            verbosity=verbosity
        )

        # Show sample if requested
        if args.show_sample:
            print_tree_sample(tree)

        # Save to JSON file if requested
        if args.output_json:
            if verbosity >= 1:
                print(f"\nSaving tree to {args.output_json}...")

            data = tree.to_dict()
            data["version"] = args.version or datetime.now().strftime("%Y_%m_%d_%H_%M")
            data["created_at"] = datetime.now().isoformat()
            data["source_db"] = args.db

            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if verbosity >= 1:
                print(f"  Saved to {args.output_json}")

        # Save to Redis unless dry run
        if not args.dry_run:
            redis_key = save_to_redis(
                tree=tree,
                redis_host=args.redis_host,
                redis_port=args.redis_port,
                redis_db=args.redis_db,
                redis_password=args.redis_password,
                version=args.version,
                ttl=args.ttl,
                verbosity=verbosity
            )

            if verbosity >= 1:
                print(f"\nVocabulary tree saved to Redis: {redis_key}")
        else:
            if verbosity >= 1:
                print("\n[DRY RUN] Tree built but not saved to Redis")

        if verbosity >= 1:
            print("\nDone!")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if verbosity >= 2:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        # Always release the lock
        if lock_client:
            release_lock(lock_client, verbosity)


if __name__ == "__main__":
    sys.exit(main())
