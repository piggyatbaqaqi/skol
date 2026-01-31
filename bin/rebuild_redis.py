#!/usr/bin/env python3
"""
Rebuild All SKOL Redis Keys

This script rebuilds all skol:* keys in Redis by running the appropriate
bin/ programs in sequence. Use this after a Redis data loss or when setting
up a new environment.

Keys rebuilt:
  - skol:classifier:model:*    (from train_classifier.py)
  - skol:embedding:*           (from embed_taxa.py)
  - skol:ui:menus_*            (from build_vocab_tree.py)
  - skol:fungaria              (from manage_fungaria.py)

Usage:
    python bin/rebuild_redis.py              # Rebuild all keys
    python bin/rebuild_redis.py --list       # List what would be rebuilt
    python bin/rebuild_redis.py --dry-run    # Preview without executing

Examples:
    # Full rebuild
    python bin/rebuild_redis.py

    # Skip classifier training (slow)
    python bin/rebuild_redis.py --skip-classifier

    # Only rebuild specific components
    python bin/rebuild_redis.py --only embeddings,fungaria
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config, create_redis_client


# ============================================================================
# Component Definitions
# ============================================================================

COMPONENTS = {
    'classifier': {
        'name': 'Classifier Model',
        'keys': ['skol:classifier:model:*'],
        'script': 'train_classifier.py',
        'args': [],
        'description': 'Train the text classifier model (slow, requires Spark)',
    },
    'embeddings': {
        'name': 'Taxa Embeddings',
        'keys': ['skol:embedding:*'],
        'script': 'embed_taxa.py',
        'args': ['--force'],  # Force rebuild even if exists
        'description': 'Compute embeddings for taxa descriptions',
    },
    'vocab_tree': {
        'name': 'Vocabulary Tree',
        'keys': ['skol:ui:menus_*', 'skol:ui:menus_latest'],
        'script': 'build_vocab_tree.py',
        'args': [],
        'description': 'Build vocabulary tree for UI menus',
    },
    'fungaria': {
        'name': 'Fungaria Registry',
        'keys': ['skol:fungaria'],
        'script': 'manage_fungaria.py',
        'args': ['download'],
        'description': 'Download Index Herbariorum institution list',
    },
}

# Order matters: some components may depend on others
BUILD_ORDER = ['classifier', 'embeddings', 'vocab_tree', 'fungaria']


def list_existing_keys(redis_client, verbosity: int = 1) -> dict:
    """
    List existing skol:* keys in Redis.

    Args:
        redis_client: Redis client
        verbosity: Verbosity level

    Returns:
        Dict mapping component names to lists of matching keys
    """
    results = {}

    for component, config in COMPONENTS.items():
        matching_keys = []
        for pattern in config['keys']:
            keys = redis_client.keys(pattern)
            matching_keys.extend([k.decode() if isinstance(k, bytes) else k for k in keys])
        results[component] = sorted(set(matching_keys))

    return results


def run_component(
    component: str,
    bin_dir: Path,
    verbosity: int = 1,
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None
) -> bool:
    """
    Run a component's build script.

    Args:
        component: Component name
        bin_dir: Path to bin/ directory
        verbosity: Verbosity level
        dry_run: If True, print command but don't execute
        extra_args: Additional arguments to pass to the script

    Returns:
        True if successful, False otherwise
    """
    config = COMPONENTS[component]
    script = bin_dir / config['script']

    if not script.exists():
        print(f"  ✗ Script not found: {script}")
        return False

    # Build command
    cmd = [sys.executable, str(script)] + config['args']
    if extra_args:
        cmd.extend(extra_args)

    # Add verbosity flag if supported
    if verbosity >= 2:
        cmd.extend(['--verbosity', '2'])

    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    if verbosity >= 1:
        print(f"  Running: {' '.join(cmd)}")
        print()

    try:
        result = subprocess.run(
            cmd,
            cwd=bin_dir.parent,  # Run from project root
            check=False,
        )

        if result.returncode != 0:
            print(f"  ✗ {config['name']} failed with exit code {result.returncode}")
            return False

        print(f"  ✓ {config['name']} completed successfully")
        return True

    except Exception as e:
        print(f"  ✗ Error running {config['name']}: {e}")
        return False


def main():
    """Main entry point for the rebuild script."""
    parser = argparse.ArgumentParser(
        description='Rebuild all SKOL Redis keys',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Components:
""" + '\n'.join([
    f"  {name:12} - {config['description']}"
    for name, config in COMPONENTS.items()
]) + """

Examples:
  %(prog)s                          # Rebuild all
  %(prog)s --skip-classifier        # Skip slow classifier training
  %(prog)s --only embeddings        # Only rebuild embeddings
  %(prog)s --list                   # List existing keys
"""
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List existing skol:* keys and exit'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands but do not execute'
    )
    parser.add_argument(
        '--skip-classifier',
        action='store_true',
        help='Skip classifier training (slow, requires Spark)'
    )
    parser.add_argument(
        '--only',
        type=str,
        default=None,
        metavar='COMPONENTS',
        help='Only rebuild specific components (comma-separated: classifier,embeddings,vocab_tree,fungaria)'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity'
    )

    args = parser.parse_args()

    # Set verbosity
    verbosity = args.verbosity + args.verbose

    # Get configuration
    config = get_env_config()
    bin_dir = Path(__file__).resolve().parent

    # Connect to Redis
    if verbosity >= 1:
        print(f"Connecting to Redis at {config['redis_host']}:{config['redis_port']}...")

    try:
        redis_client = create_redis_client(decode_responses=False)
        redis_client.ping()
        if verbosity >= 1:
            print("✓ Connected to Redis\n")
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        sys.exit(1)

    # List mode
    if args.list:
        print("=" * 70)
        print("Existing SKOL Redis Keys")
        print("=" * 70)

        existing = list_existing_keys(redis_client, verbosity)

        for component, keys in existing.items():
            config_info = COMPONENTS[component]
            print(f"\n{config_info['name']} ({component}):")
            if keys:
                for key in keys:
                    # Get TTL
                    ttl = redis_client.ttl(key)
                    if ttl == -1:
                        ttl_str = "no expiration"
                    elif ttl == -2:
                        ttl_str = "key does not exist"
                    else:
                        hours = ttl // 3600
                        minutes = (ttl % 3600) // 60
                        ttl_str = f"expires in {hours}h {minutes}m"

                    # Get size
                    try:
                        size = redis_client.memory_usage(key)
                        if size:
                            if size > 1024 * 1024:
                                size_str = f"{size / 1024 / 1024:.1f} MB"
                            elif size > 1024:
                                size_str = f"{size / 1024:.1f} KB"
                            else:
                                size_str = f"{size} bytes"
                        else:
                            size_str = "unknown size"
                    except Exception:
                        size_str = "unknown size"

                    print(f"  {key} ({size_str}, {ttl_str})")
            else:
                print("  (no keys found)")

        print()
        return 0

    # Determine which components to rebuild
    if args.only:
        components = [c.strip() for c in args.only.split(',')]
        # Validate
        for c in components:
            if c not in COMPONENTS:
                print(f"✗ Unknown component: {c}")
                print(f"  Available: {', '.join(COMPONENTS.keys())}")
                sys.exit(1)
    else:
        components = BUILD_ORDER.copy()
        if args.skip_classifier:
            components.remove('classifier')

    # Rebuild
    print("=" * 70)
    print("Rebuilding SKOL Redis Keys")
    print("=" * 70)
    print(f"\nComponents to rebuild: {', '.join(components)}")

    if args.dry_run:
        print("\n[DRY RUN MODE - no changes will be made]\n")

    success_count = 0
    fail_count = 0

    for component in components:
        config_info = COMPONENTS[component]
        print(f"\n{'='*70}")
        print(f"Building: {config_info['name']}")
        print(f"Keys: {', '.join(config_info['keys'])}")
        print(f"{'='*70}\n")

        if run_component(component, bin_dir, verbosity, args.dry_run):
            success_count += 1
        else:
            fail_count += 1
            # Continue with other components

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")

    if fail_count > 0:
        print("\n⚠ Some components failed. Check the output above for details.")
        return 1
    else:
        print("\n✓ All components rebuilt successfully!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
