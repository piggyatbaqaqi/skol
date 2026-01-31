#!/usr/bin/env python3
"""
Manage Herbaria and Fungaria Registry in Redis

This standalone program downloads the Index Herbariorum list from the NYBG API
and stores it in Redis. It also supports adding entries from local JSON files
to include personal collections (e.g., Rod Tulloss's Amanita collection) that
are not tracked by Index Herbariorum.

Data Source:
    Index Herbariorum (IH) API by NYBG
    https://sweetgum.nybg.org/science/ih/
    API Documentation: https://github.com/nybgvh/IH-API/wiki

Usage:
    python manage_fungaria.py download           # Download from NYBG API
    python manage_fungaria.py add FILE           # Add entries from JSON file
    python manage_fungaria.py list               # List all entries
    python manage_fungaria.py search QUERY       # Search by code or name
    python manage_fungaria.py get CODE           # Get specific institution
    python manage_fungaria.py stats              # Show statistics
    python manage_fungaria.py clear              # Clear all entries

Examples:
    python manage_fungaria.py download --verbosity 2
    python manage_fungaria.py add personal_fungaria.json
    python manage_fungaria.py search "Tulloss"
    python manage_fungaria.py get NY
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import redis
import requests

# Add parent directory to path for skol modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add bin directory to path for env_config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import get_env_config, create_redis_client

# ============================================================================
# Constants
# ============================================================================

REDIS_KEY = 'skol:fungaria'
IH_API_BASE = 'http://sweetgum.nybg.org/science/api/v1'
IH_INSTITUTIONS_ENDPOINT = f'{IH_API_BASE}/institutions'

# Default expiration: 30 days (Index Herbariorum doesn't change frequently)
DEFAULT_EXPIRE_SECONDS = 60 * 60 * 24 * 30


# ============================================================================
# API Functions
# ============================================================================

def download_institutions(verbosity: int = 1) -> List[Dict[str, Any]]:
    """
    Download complete list of institutions from Index Herbariorum API.

    Args:
        verbosity: Verbosity level (0=silent, 1=info, 2=debug)

    Returns:
        List of institution dictionaries

    Raises:
        requests.RequestException: If API request fails
    """
    if verbosity >= 1:
        print(f"Downloading institutions from {IH_INSTITUTIONS_ENDPOINT}...")

    # Set headers to identify as a legitimate client
    headers = {
        'User-Agent': 'SKOL/1.0 (Synoptic Key of Life; https://synoptickeyof.life)',
        'Accept': 'application/json',
    }

    response = requests.get(IH_INSTITUTIONS_ENDPOINT, headers=headers, timeout=120)
    response.raise_for_status()

    data = response.json()

    # The API returns a structure with 'data' containing the institutions
    if isinstance(data, dict) and 'data' in data:
        institutions = data['data']
    elif isinstance(data, list):
        institutions = data
    else:
        raise ValueError(f"Unexpected API response structure: {type(data)}")

    if verbosity >= 1:
        print(f"Downloaded {len(institutions)} institutions")

    return institutions


def get_institution_by_code(code: str, verbosity: int = 1) -> Optional[Dict[str, Any]]:
    """
    Get a specific institution by its code from the API.

    Args:
        code: Institution code (e.g., 'NY', 'K', 'BPI')
        verbosity: Verbosity level

    Returns:
        Institution dictionary or None if not found
    """
    url = f"{IH_INSTITUTIONS_ENDPOINT}/{code}"

    if verbosity >= 2:
        print(f"Fetching {url}...")

    headers = {
        'User-Agent': 'SKOL/1.0 (Synoptic Key of Life; https://synoptickeyof.life)',
        'Accept': 'application/json',
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()

        # Extract institution from response
        if isinstance(data, dict) and 'data' in data:
            return data['data'][0] if data['data'] else None
        return data
    except requests.RequestException as e:
        if verbosity >= 1:
            print(f"Warning: Could not fetch institution {code}: {e}")
        return None


# ============================================================================
# Redis Functions
# ============================================================================

def get_redis_client(config: Dict[str, Any]) -> redis.Redis:
    """Create and return a Redis client with TLS and auth support."""
    return create_redis_client(decode_responses=True)


def save_to_redis(
    client: redis.Redis,
    institutions: List[Dict[str, Any]],
    source: str,
    expire: Optional[int] = DEFAULT_EXPIRE_SECONDS,
    verbosity: int = 1
) -> int:
    """
    Save institutions to Redis, merging with existing entries.

    Args:
        client: Redis client
        institutions: List of institution dictionaries
        source: Source identifier (e.g., 'nybg', 'local:filename.json')
        expire: Expiration time in seconds (None for no expiration)
        verbosity: Verbosity level

    Returns:
        Number of institutions added/updated
    """
    # Load existing data
    existing_raw = client.get(REDIS_KEY)
    if existing_raw:
        existing_data = json.loads(existing_raw)
    else:
        existing_data = {'institutions': {}, 'sources': [], 'metadata': {}}

    # Ensure structure
    if 'institutions' not in existing_data:
        existing_data['institutions'] = {}
    if 'sources' not in existing_data:
        existing_data['sources'] = []

    # Track source
    if source not in existing_data['sources']:
        existing_data['sources'].append(source)

    # Add/update institutions keyed by code
    count = 0
    for inst in institutions:
        code = inst.get('code') or inst.get('irn') or f"unknown_{count}"
        # Mark the source for this entry
        inst['_source'] = source
        existing_data['institutions'][code] = inst
        count += 1

    # Update metadata
    existing_data['metadata'] = {
        'total_count': len(existing_data['institutions']),
        'sources': existing_data['sources'],
        'last_updated': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc
        ).isoformat()
    }

    # Save to Redis
    client.set(REDIS_KEY, json.dumps(existing_data))

    if expire:
        client.expire(REDIS_KEY, expire)
        if verbosity >= 2:
            print(f"Set expiration: {expire} seconds ({expire / 86400:.1f} days)")

    if verbosity >= 1:
        print(f"Saved {count} institutions from {source}")
        print(f"Total institutions in registry: {len(existing_data['institutions'])}")

    return count


def load_from_redis(client: redis.Redis) -> Dict[str, Any]:
    """Load fungaria data from Redis."""
    raw = client.get(REDIS_KEY)
    if not raw:
        return {'institutions': {}, 'sources': [], 'metadata': {}}
    return json.loads(raw)


def clear_redis(client: redis.Redis, verbosity: int = 1) -> bool:
    """Clear all fungaria data from Redis."""
    result = client.delete(REDIS_KEY)
    if verbosity >= 1:
        if result:
            print(f"Cleared {REDIS_KEY} from Redis")
        else:
            print(f"Key {REDIS_KEY} did not exist")
    return bool(result)


# ============================================================================
# Local File Functions
# ============================================================================

def load_local_file(filepath: Path, verbosity: int = 1) -> List[Dict[str, Any]]:
    """
    Load institutions from a local JSON file.

    The file should contain either:
    - A list of institution objects
    - An object with a 'data' or 'institutions' key containing a list

    Each institution should have at minimum:
    - code: Unique identifier (e.g., 'TULLOSS' for a personal collection)
    - organization: Name of the institution or collection

    Example format (matching NYBG structure):
    [
        {
            "code": "TULLOSS",
            "organization": "Rod Tulloss Amanita Collection",
            "specimenTotal": 15000,
            "contact": {
                "firstName": "Rod",
                "lastName": "Tulloss",
                "email": "ret@pluto.njcc.com"
            },
            "address": {
                "city": "Roosevelt",
                "state": "New Jersey",
                "country": "U.S.A."
            }
        }
    ]

    Args:
        filepath: Path to JSON file
        verbosity: Verbosity level

    Returns:
        List of institution dictionaries
    """
    if verbosity >= 1:
        print(f"Loading institutions from {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different structures
    if isinstance(data, list):
        institutions = data
    elif isinstance(data, dict):
        if 'data' in data:
            institutions = data['data']
        elif 'institutions' in data:
            institutions = data['institutions']
            # Handle dict format
            if isinstance(institutions, dict):
                institutions = list(institutions.values())
        else:
            # Assume single institution
            institutions = [data]
    else:
        raise ValueError(f"Unexpected JSON structure in {filepath}")

    if verbosity >= 1:
        print(f"Loaded {len(institutions)} institutions from file")

    return institutions


# ============================================================================
# Display Functions
# ============================================================================

def display_institution(inst: Dict[str, Any], verbose: bool = False) -> None:
    """Display a single institution."""
    code = inst.get('code', 'N/A')
    org = inst.get('organization', 'Unknown')
    source = inst.get('_source', 'unknown')

    # Get location (API uses physicalCity etc., local files may use city etc.)
    addr = inst.get('address', {})
    if isinstance(addr, dict):
        city = addr.get('physicalCity') or addr.get('city', '')
        state = addr.get('physicalState') or addr.get('state', '')
        country = addr.get('physicalCountry') or addr.get('country', '')
        location = ', '.join(filter(None, [city, state, country]))
    else:
        location = ''

    # Get specimen count
    specimens = inst.get('specimenTotal', '')
    if specimens:
        specimens = f" ({specimens:,} specimens)" if isinstance(specimens, int) else f" ({specimens} specimens)"

    print(f"  [{code}] {org}")
    if location:
        print(f"         {location}{specimens}")
    if verbose:
        print(f"         Source: {source}")
        if inst.get('contact'):
            contact = inst['contact']
            name = f"{contact.get('firstName', '')} {contact.get('lastName', '')}".strip()
            email = contact.get('email', '')
            if name or email:
                print(f"         Contact: {name} <{email}>" if email else f"         Contact: {name}")


def display_stats(data: Dict[str, Any]) -> None:
    """Display statistics about the fungaria registry."""
    institutions = data.get('institutions', {})
    metadata = data.get('metadata', {})

    print(f"\nFungaria Registry Statistics")
    print(f"{'='*40}")
    print(f"Total institutions: {len(institutions)}")

    # Count by source
    sources = {}
    for inst in institutions.values():
        src = inst.get('_source', 'unknown')
        sources[src] = sources.get(src, 0) + 1

    print(f"\nBy source:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    # Count by country
    countries: dict[str, int] = {}
    for inst in institutions.values():
        addr = inst.get('address', {})
        if isinstance(addr, dict):
            # API uses physicalCountry, local files may use country
            country = (addr.get('physicalCountry') or
                       addr.get('country') or 'Unknown')
        else:
            country = 'Unknown'
        countries[country] = countries.get(country, 0) + 1

    print(f"\nTop 10 countries:")
    for country, count in sorted(countries.items(), key=lambda x: -x[1])[:10]:
        print(f"  {country}: {count}")

    # Last updated
    if metadata.get('last_updated'):
        print(f"\nLast updated: {metadata['last_updated']}")


# ============================================================================
# Command Handlers
# ============================================================================

def cmd_download(args, config: Dict[str, Any]) -> int:
    """Download institutions from NYBG API."""
    verbosity = config['verbosity']

    try:
        institutions = download_institutions(verbosity)
    except requests.RequestException as e:
        print(f"Error downloading from API: {e}")
        return 1

    client = get_redis_client(config)

    try:
        client.ping()
        if verbosity >= 1:
            print("Connected to Redis")
    except redis.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        return 1

    # Determine expiration
    expire = getattr(args, 'expire', DEFAULT_EXPIRE_SECONDS)

    if args.replace:
        clear_redis(client, verbosity=0)

    save_to_redis(client, institutions, 'nybg', expire=expire, verbosity=verbosity)

    return 0


def cmd_add(args, config: Dict[str, Any]) -> int:
    """Add institutions from a local JSON file."""
    verbosity = config['verbosity']
    filepath = Path(args.file)

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1

    try:
        institutions = load_local_file(filepath, verbosity)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON file: {e}")
        return 1

    client = get_redis_client(config)

    try:
        client.ping()
    except redis.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        return 1

    source = f"local:{filepath.name}"
    expire = getattr(args, 'expire', None)  # No expiration for local additions by default

    save_to_redis(client, institutions, source, expire=expire, verbosity=verbosity)

    return 0


def cmd_list(args, config: Dict[str, Any]) -> int:
    """List all institutions."""
    verbosity = config['verbosity']

    client = get_redis_client(config)
    data = load_from_redis(client)
    institutions = data.get('institutions', {})

    if not institutions:
        print("No institutions in registry. Run 'download' first.")
        return 0

    print(f"\nFungaria Registry ({len(institutions)} institutions)")
    print(f"{'='*50}")

    # Sort by code
    limit = getattr(args, 'limit', None) or len(institutions)
    for i, (code, inst) in enumerate(sorted(institutions.items())):
        if i >= limit:
            print(f"\n... and {len(institutions) - limit} more")
            break
        display_institution(inst, verbose=(verbosity >= 2))

    return 0


def cmd_search(args, config: Dict[str, Any]) -> int:
    """Search institutions by code or name."""
    verbosity = config['verbosity']
    query = args.query.lower()

    client = get_redis_client(config)
    data = load_from_redis(client)
    institutions = data.get('institutions', {})

    if not institutions:
        print("No institutions in registry. Run 'download' first.")
        return 0

    # Search in code and organization name
    matches = []
    for code, inst in institutions.items():
        org = inst.get('organization', '').lower()
        if query in code.lower() or query in org:
            matches.append((code, inst))

    if not matches:
        print(f"No matches found for '{args.query}'")
        return 0

    print(f"\nSearch results for '{args.query}' ({len(matches)} matches)")
    print(f"{'='*50}")

    for code, inst in sorted(matches):
        display_institution(inst, verbose=(verbosity >= 2))

    return 0


def cmd_get(args, config: Dict[str, Any]) -> int:
    """Get a specific institution by code."""
    verbosity = config['verbosity']
    code = args.code.upper()

    client = get_redis_client(config)
    data = load_from_redis(client)
    institutions = data.get('institutions', {})

    inst = institutions.get(code)

    if not inst:
        print(f"Institution '{code}' not found in local registry")
        # Try fetching from API
        if verbosity >= 1:
            print("Checking NYBG API...")
        inst = get_institution_by_code(code, verbosity)
        if inst:
            print(f"\nFound in NYBG API (not in local registry):")
        else:
            print(f"Institution '{code}' not found in NYBG API either")
            return 1

    print(f"\nInstitution: {code}")
    print(f"{'='*50}")
    print(json.dumps(inst, indent=2))

    return 0


def cmd_stats(args, config: Dict[str, Any]) -> int:
    """Show statistics about the registry."""
    client = get_redis_client(config)
    data = load_from_redis(client)

    if not data.get('institutions'):
        print("No institutions in registry. Run 'download' first.")
        return 0

    display_stats(data)
    return 0


def cmd_clear(args, config: Dict[str, Any]) -> int:
    """Clear all fungaria data from Redis."""
    verbosity = config['verbosity']

    if not args.yes:
        response = input(f"Are you sure you want to clear {REDIS_KEY}? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled")
            return 0

    client = get_redis_client(config)
    clear_redis(client, verbosity)
    return 0


# ============================================================================
# Main Program
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Manage Herbaria and Fungaria Registry in Redis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Sources:
  The primary source is Index Herbariorum (IH) maintained by NYBG.
  Additional sources can be added from local JSON files to include
  personal collections not tracked by IH.

Redis Key: skol:fungaria

Examples:
  %(prog)s download                    # Download from NYBG API
  %(prog)s download --replace          # Replace existing data
  %(prog)s add personal_fungaria.json  # Add from local file
  %(prog)s search "Tulloss"            # Search by name
  %(prog)s get NY                      # Get specific institution
  %(prog)s stats                       # Show statistics

Local File Format:
  JSON file should contain a list of institution objects with at minimum:
  - code: Unique identifier
  - organization: Name of the institution/collection

  See --help-format for full format documentation.
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # download command
    p_download = subparsers.add_parser('download', help='Download from NYBG API')
    p_download.add_argument('--replace', action='store_true',
                           help='Replace existing data instead of merging')
    p_download.add_argument('--expire', type=int, default=DEFAULT_EXPIRE_SECONDS,
                           help=f'Expiration in seconds (default: {DEFAULT_EXPIRE_SECONDS})')

    # add command
    p_add = subparsers.add_parser('add', help='Add from local JSON file')
    p_add.add_argument('file', help='Path to JSON file')
    p_add.add_argument('--expire', type=int, default=None,
                      help='Expiration in seconds (default: none)')

    # list command
    p_list = subparsers.add_parser('list', help='List all institutions')
    p_list.add_argument('--limit', type=int, default=50,
                       help='Maximum entries to show (default: 50)')

    # search command
    p_search = subparsers.add_parser('search', help='Search institutions')
    p_search.add_argument('query', help='Search query (code or name)')

    # get command
    p_get = subparsers.add_parser('get', help='Get specific institution')
    p_get.add_argument('code', help='Institution code (e.g., NY, K, BPI)')

    # stats command
    subparsers.add_parser('stats', help='Show statistics')

    # clear command
    p_clear = subparsers.add_parser('clear', help='Clear all data')
    p_clear.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt')

    # Global options
    parser.add_argument(
        '--verbosity',
        type=int,
        default=None,
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Get configuration
    config = get_env_config()

    # Apply verbosity override from command line
    if args.verbosity is not None:
        config['verbosity'] = args.verbosity

    # Dispatch to command handler
    handlers = {
        'download': cmd_download,
        'add': cmd_add,
        'list': cmd_list,
        'search': cmd_search,
        'get': cmd_get,
        'stats': cmd_stats,
        'clear': cmd_clear,
    }

    handler = handlers.get(args.command)
    if handler:
        try:
            return handler(args, config)
        except KeyboardInterrupt:
            print("\nInterrupted")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            if config['verbosity'] >= 2:
                import traceback
                traceback.print_exc()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
