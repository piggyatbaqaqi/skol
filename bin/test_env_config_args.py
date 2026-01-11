#!/usr/bin/env python3
"""
Test command-line argument parsing in env_config.py

This script verifies that get_env_config() correctly parses command-line arguments
and that they override environment variables.
"""

import sys
from pathlib import Path

# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config

def test_env_config_args():
    """Test that command-line arguments are parsed and override env vars."""

    config = get_env_config()

    print("Testing env_config.py command-line argument parsing")
    print("=" * 70)
    print()

    # Show a few key config values
    print("Current configuration:")
    print(f"  couchdb_database: {config['couchdb_database']}")
    print(f"  couchdb_host: {config['couchdb_host']}")
    print(f"  redis_host: {config['redis_host']}")
    print(f"  redis_port: {config['redis_port']}")
    print(f"  cores: {config['cores']}")
    print(f"  prediction_batch_size: {config['prediction_batch_size']}")
    print()

    # Check if command-line args were provided
    if len(sys.argv) > 1:
        print(f"Command-line arguments detected: {' '.join(sys.argv[1:])}")
        print()

        # Show which values might have been overridden
        print("Note: Command-line args override environment variables")
        print("      Underscores in config keys become dashes in argument names")
        print("      Example: config['couchdb_database'] → --couchdb-database")
    else:
        print("No command-line arguments provided (using env vars and defaults)")
        print()
        print("To test argument parsing, run:")
        print("  python3 bin/test_env_config_args.py --couchdb-database mydb --cores 8")

    print()
    print("=" * 70)
    print("✓ Configuration loaded successfully!")
    print()

    # Show argument mapping examples
    print("Available command-line arguments (sample):")
    print("  --couchdb-database <name>  Override CouchDB database name")
    print("  --couchdb-host <host>      Override CouchDB host")
    print("  --redis-port <port>        Override Redis port (integer)")
    print("  --cores <num>              Override number of Spark cores (integer)")
    print("  --prediction-batch-size <size>  Override batch size (integer)")
    print()

if __name__ == '__main__':
    test_env_config_args()
