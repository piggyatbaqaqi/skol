#!/usr/bin/env python3
"""
Main entry point for ingesting data into CouchDB.

This module provides a command-line interface for ingesting data from
various sources using different ingestor implementations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.robotparser import RobotFileParser

# Add parent directory to path for direct script execution
if __name__ == '__main__' and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

import couchdb

# Support both direct execution and module execution
try:
    from .crossref import CrossrefIngestor
    from .ingenta import IngentaIngestor
    from .local_ingenta import LocalIngentaIngestor
    from .local_mykoweb import LocalMykowebJournalsIngestor
    from .local_mykoweb_literature import LocalMykowebLiteratureIngestor
    from .mdpi import MdpiIngestor
    from .medwin_publishers import MedwinPublishersIngestor
    from .mycosphere import MycosphereIngestor
    from .pensoft import PensoftIngestor
    from .taylor_francis import TaylorFrancisIngestor
    from .publications import PublicationRegistry
except ImportError:
    from ingestors.crossref import CrossrefIngestor
    from ingestors.ingenta import IngentaIngestor
    from ingestors.local_ingenta import LocalIngentaIngestor
    from ingestors.local_mykoweb import LocalMykowebJournalsIngestor
    from ingestors.local_mykoweb_literature import LocalMykowebLiteratureIngestor
    from ingestors.mdpi import MdpiIngestor
    from ingestors.medwin_publishers import MedwinPublishersIngestor
    from ingestors.mycosphere import MycosphereIngestor
    from ingestors.pensoft import PensoftIngestor
    from ingestors.taylor_francis import TaylorFrancisIngestor
    from ingestors.publications import PublicationRegistry


# Registry mapping ingestor class names to actual classes
INGESTOR_CLASSES = {
    'CrossrefIngestor': CrossrefIngestor,
    'IngentaIngestor': IngentaIngestor,
    'LocalIngentaIngestor': LocalIngentaIngestor,
    'LocalMykowebJournalsIngestor': LocalMykowebJournalsIngestor,
    'LocalMykowebLiteratureIngestor': LocalMykowebLiteratureIngestor,
    'MdpiIngestor': MdpiIngestor,
    'MedwinPublishersIngestor': MedwinPublishersIngestor,
    'MycosphereIngestor': MycosphereIngestor,
    'PensoftIngestor': PensoftIngestor,
    'TaylorFrancisIngestor': TaylorFrancisIngestor,
}




def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Ingest data from web sources into CouchDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verbosity levels:
  0 - Silent (no output)
  1 - Warnings only (robot denials, errors)
  2 - Normal (default: skip and add messages)
  3 - Verbose (includes URLs and separators)

Examples:
  # Ingest from a specific publication source
  %(prog)s --publication mycotaxon
  %(prog)s --publication studies-in-mycology
  %(prog)s --publication ingenta-local

  # Ingest from all predefined sources
  %(prog)s --all

  # Ingest from custom RSS feed
  %(prog)s --source ingenta --rss https://api.ingentaconnect.com/content/mtax/mt?format=rss

  # Ingest from custom local directory
  %(prog)s --source ingenta --local /data/skol/www/www.ingentaconnect.com

  # With credentials from environment variables
  export COUCHDB_USER=myuser COUCHDB_PASSWORD=mypass
  %(prog)s --all

  # Silent mode
  %(prog)s --publication mycotaxon --verbosity 0

  # Verbose mode
  %(prog)s --all -v 3
        """
    )

    # CouchDB connection arguments
    parser.add_argument(
        '--couchdb-url',
        type=str,
        default=os.environ.get('COUCHDB_URL', 'http://localhost:5984'),
        help='CouchDB server URL (default: $COUCHDB_URL or http://localhost:5984)'
    )
    parser.add_argument(
        '--couchdb-username',
        type=str,
        default=os.environ.get('COUCHDB_USER'),
        help='CouchDB username (default: $COUCHDB_USER)'
    )
    parser.add_argument(
        '--couchdb-password',
        type=str,
        default=os.environ.get('COUCHDB_PASSWORD'),
        help='CouchDB password (default: $COUCHDB_PASSWORD)'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='skol_dev',
        help='CouchDB database name (default: skol_dev)'
    )

    # Ingestor selection (only required for --rss or --local)
    parser.add_argument(
        '--source',
        type=str,
        choices=['ingenta'],
        help='Data source to ingest from (required with --rss or --local)'
    )

    # Ingestion mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--rss',
        type=str,
        metavar='URL',
        help='RSS feed URL to ingest from'
    )
    mode_group.add_argument(
        '--local',
        type=Path,
        metavar='DIR',
        help='Local directory containing BibTeX files'
    )
    mode_group.add_argument(
        '--publication',
        type=str,
        metavar='KEY',
        choices=PublicationRegistry.list_publications(),
        help=f'Use predefined source: {", ".join(PublicationRegistry.list_publications())}'
    )
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='Ingest from all predefined sources'
    )
    mode_group.add_argument(
        '--list-publications',
        action='store_true',
        help='List all available publication sources and exit'
    )

    # Optional arguments
    parser.add_argument(
        '--user-agent',
        type=str,
        default='synoptickeyof.life',
        help='User agent string for HTTP requests (default: synoptickeyof.life)'
    )
    parser.add_argument(
        '--robots-url',
        type=str,
        help='URL to robots.txt (default: source-specific)'
    )
    parser.add_argument(
        '--bibtex-pattern',
        type=str,
        default='format=bib',
        help='Filename pattern for BibTeX files (default: format=bib)'
    )
    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help='Verbosity level (default: 2)'
    )

    return parser


def run_ingestion(
    db: couchdb.Database,
    config: Dict,
    user_agent: str = 'synoptickeyof.life',
    robots_url: Optional[str] = None,
    verbosity: int = 2
) -> None:
    """
    Run a single ingestion task.

    Args:
        db: CouchDB database instance
        config: Publication configuration dictionary from PublicationRegistry
        user_agent: User agent string
        robots_url: Custom robots.txt URL
        verbosity: Verbosity level
    """
    # Set up robot parser
    source = config.get('source', 'unknown')
    robots_url_final = PublicationRegistry.get_robots_url(source, robots_url)
    if verbosity >= 3:
        print(f"Loading robots.txt from {robots_url_final}")
    robot_parser = RobotFileParser()
    robot_parser.set_url(robots_url_final)
    robot_parser.read()

    # Get the ingestor class
    ingestor_class_name = config.get('ingestor_class')
    if not ingestor_class_name:
        raise ValueError(f"No ingestor_class specified for source '{source}'")

    ingestor_class = INGESTOR_CLASSES.get(ingestor_class_name)
    if not ingestor_class:
        raise ValueError(f"Unknown ingestor class: {ingestor_class_name}")

    # Prepare constructor arguments - pass everything from config plus common params
    # Each ingestor will ignore params it doesn't need via **kwargs
    constructor_args = {
        'db': db,
        'user_agent': user_agent,
        'robot_parser': robot_parser,
        'verbosity': verbosity,
        # Add all config values (mode, urls, paths, etc.)
        **{k: Path(v) if k in ('root', 'local_path') and isinstance(v, str) else v
           for k, v in config.items()
           if k not in ('name', 'source', 'ingestor_class')},
    }

    # Special handling for local PDF maps (for Mykoweb ingestors)
    if 'local_path_prefix' in config and 'url_prefix' in config:
        constructor_args['local_pdf_map'] = {
            config['url_prefix']: config['local_path_prefix']
        }

    # Create and run the ingestor
    ingestor = ingestor_class(**constructor_args)

    # Perform ingestion using polymorphic ingest() method
    ingestor.ingest()


def list_publications() -> None:
    """Print a table of all available publication sources."""
    print("\nAvailable publication Sources:")
    print("=" * 80)
    print(f"{'Key':<25} {'Name':<30} {'Type':<10} {'Details':<15}")
    print("-" * 80)

    for key, config in PublicationRegistry.get_all().items():
        source_type = config.get('mode', 'WEB').upper()
        if config.get('mode') == 'rss':
            details = config['rss_url'].split('/')[-1][:15]
        elif config.get('mode') == 'api':
            details = f"ISSN {config.get('issn', 'N/A')}"
        elif config.get('mode') == 'local':
            details = "Local files"
        else:
            details = "Web scraping"
        print(f"{key:<25} {config['name']:<30} {source_type:<10} {details:<15}")

    print("=" * 80)
    print(f"\nTotal: {len(PublicationRegistry.get_all())} publication sources")
    print("\nUsage:")
    print("  Single publication:  ./main.py --publication <key>")
    print("  All publications:    ./main.py --all")
    print()


def main() -> int:
    """
    Main entry point for the ingestor CLI.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Handle --list-publications
    if args.list_publications:
        list_publications()
        return 0

    # Validate --source is provided when needed
    if (args.rss or args.local) and not args.source:
        parser.error("--source is required when using --rss or --local")

    try:
        # Connect to CouchDB
        if args.verbosity >= 2:
            print(f"Connecting to CouchDB at {args.couchdb_url}...")

        # Create server connection with credentials if provided
        if args.couchdb_username and args.couchdb_password:
            couch = couchdb.Server(args.couchdb_url)
            couch.resource.credentials = (args.couchdb_username, args.couchdb_password)
            if args.verbosity >= 3:
                print(f"Using credentials for user: {args.couchdb_username}")
        else:
            couch = couchdb.Server(args.couchdb_url)

        db = couch[args.database]
        if args.verbosity >= 2:
            print(f"Using database: {args.database}")

        # Handle different ingestion modes
        if args.all:
            # Ingest from all predefined sources
            all_sources = PublicationRegistry.get_all()
            if args.verbosity >= 2:
                print(f"Ingesting from all {len(all_sources)} predefined sources...")
            for key, config in all_sources.items():
                if args.verbosity >= 2:
                    print(f"\n{'=' * 60}")
                    print(f"Processing: {config['name']} ({key})")
                    print(f"{'=' * 60}")

                run_ingestion(
                    db=db,
                    config=config,
                    user_agent=args.user_agent,
                    robots_url=args.robots_url,
                    verbosity=args.verbosity
                )
        elif args.publication:
            # Use predefined source
            config = PublicationRegistry.get(args.publication)
            if config is None:
                print(f"Error: Unknown publication '{args.publication}'", file=sys.stderr)
                return 1

            if args.verbosity >= 2:
                print(f"Using publication: {config['name']}")

            run_ingestion(
                db=db,
                config=config,
                user_agent=args.user_agent,
                robots_url=args.robots_url,
                verbosity=args.verbosity
            )
        elif args.rss:
            # Direct RSS mode - construct config dict
            config = {
                'source': args.source,
                'ingestor_class': 'IngentaIngestor',
                'mode': 'rss',
                'rss_url': args.rss,
                'bibtex_pattern': args.bibtex_pattern,
            }
            run_ingestion(
                db=db,
                config=config,
                user_agent=args.user_agent,
                robots_url=args.robots_url,
                verbosity=args.verbosity
            )
        elif args.local:
            # Direct local mode - construct config dict
            config = {
                'source': args.source,
                'ingestor_class': 'LocalIngentaIngestor',
                'mode': 'local',
                'local_path': str(args.local),
                'root': str(args.local),
                'bibtex_pattern': args.bibtex_pattern,
            }
            run_ingestion(
                db=db,
                config=config,
                user_agent=args.user_agent,
                robots_url=args.robots_url,
                verbosity=args.verbosity
            )

        if args.verbosity >= 2:
            print("\nIngestion complete!")

        return 0

    except couchdb.http.ResourceNotFound:
        print(f"Error: Database '{args.database}' not found", file=sys.stderr)
        return 1
    except couchdb.http.Unauthorized:
        print("Error: Unauthorized access to CouchDB", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbosity >= 1:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
