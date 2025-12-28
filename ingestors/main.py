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
from typing import Optional
from urllib.robotparser import RobotFileParser

# Add parent directory to path for direct script execution
if __name__ == '__main__' and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

import couchdb

# Support both direct execution and module execution
try:
    from .ingenta import IngentaIngestor
    from .local_ingenta import LocalIngentaIngestor
    from .local_mykoweb import LocalMykowebJournalsIngestor
    from .local_mykoweb_literature import LocalMykowebLiteratureIngestor
except ImportError:
    from ingestors.ingenta import IngentaIngestor
    from ingestors.local_ingenta import LocalIngentaIngestor
    from ingestors.local_mykoweb import LocalMykowebJournalsIngestor
    from ingestors.local_mykoweb_literature import LocalMykowebLiteratureIngestor


# Predefined ingestion sources from ist769_skol.ipynb
SOURCES = {
    'mycotaxon': {
        'name': 'Mycotaxon',
        'source': 'ingenta',
        'mode': 'rss',
        'rss_url': 'https://api.ingentaconnect.com/content/mtax/mt?format=rss',
    },
    'studies-in-mycology': {
        'name': 'Studies in Mycology',
        'source': 'ingenta',
        'mode': 'rss',
        'rss_url': 'https://api.ingentaconnect.com/content/wfbi/sim?format=rss',
    },
    'ingenta-local': {
        'name': 'Ingenta Local BibTeX Files',
        'source': 'ingenta',
        'mode': 'local',
        'local_path': '/data/skol/www/www.ingentaconnect.com',
    },
    'mykoweb-journals': {
        'name': 'Mykoweb Journals (Mycotaxon, Persoonia, Sydowia)',
        'source': 'mykoweb-journals',
        'mode': 'local',
        'local_path': '/data/skol/www/mykoweb.com/systematics/journals',
    },
    'mykoweb-literature': {
        'name': 'Mykoweb Literature/Books',
        'source': 'mykoweb-literature',
        'mode': 'local',
        'local_path': '/data/skol/www/mykoweb.com/systematics/literature',
        'url_prefix': 'https://mykoweb.com/systematics/literature',
    },
    'mykoweb-caf': {
        'name': 'Mykoweb CAF PDFs',
        'source': 'mykoweb-caf',
        'mode': 'local',
        'local_path': '/data/skol/www/mykoweb.com/CAF',
        'url_prefix': 'https://mykoweb.com/CAF',
    },
    'mykoweb-crepidotus': {
        'name': 'Mykoweb Crepidotus',
        'source': 'mykoweb-crepidotus',
        'mode': 'local',
        'local_path': '/data/skol/www/mykoweb.com/Crepidotus',
        'url_prefix': 'https://mykoweb.com/Crepidotus',
    },
    'mykoweb-oldbooks': {
        'name': 'Mykoweb Old Books',
        'source': 'mykoweb-oldbooks',
        'mode': 'local',
        'local_path': '/data/skol/www/mykoweb.com/OldBooks',
        'url_prefix': 'https://mykoweb.com/OldBooks',
    },
    'mykoweb-gsmnp': {
        'name': 'Mykoweb GSMNP',
        'source': 'mykoweb-gsmnp',
        'mode': 'local',
        'local_path': '/data/skol/www/mykoweb.com/GSMNP',
        'url_prefix': 'https://mykoweb.com/GSMNP',
    },
    'mykoweb-pholiota': {
        'name': 'Mykoweb Pholiota',
        'source': 'mykoweb-pholiota',
        'mode': 'local',
        'local_path': '/data/skol/www/mykoweb.com/Pholiota',
        'url_prefix': 'https://mykoweb.com/Pholiota',
    },
    'mykoweb-misc': {
        'name': 'Mykoweb Misc',
        'source': 'mykoweb-misc',
        'mode': 'local',
        'local_path': '/data/skol/www/mykoweb.com/misc',
        'url_prefix': 'https://mykoweb.com/misc',
    },
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
        choices=list(SOURCES.keys()),
        help=f'Use predefined source: {", ".join(SOURCES.keys())}'
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


def get_robots_url(source: str, custom_url: Optional[str]) -> str:
    """
    Get the robots.txt URL for a given source.

    Args:
        source: Name of the data source
        custom_url: Custom robots.txt URL (overrides default)

    Returns:
        URL to robots.txt file
    """
    if custom_url:
        return custom_url

    robots_urls = {
        'ingenta': 'https://www.ingentaconnect.com/robots.txt',
        'mykoweb-journals': 'https://mykoweb.com/robots.txt',
        'mykoweb-literature': 'https://mykoweb.com/robots.txt',
        'mykoweb-caf': 'https://mykoweb.com/robots.txt',
        'mykoweb-crepidotus': 'https://mykoweb.com/robots.txt',
        'mykoweb-oldbooks': 'https://mykoweb.com/robots.txt',
        'mykoweb-gsmnp': 'https://mykoweb.com/robots.txt',
        'mykoweb-pholiota': 'https://mykoweb.com/robots.txt',
        'mykoweb-misc': 'https://mykoweb.com/robots.txt',
    }

    return robots_urls.get(source, '')


def run_ingestion(
    db: couchdb.Database,
    source: str,
    mode: str,
    rss_url: Optional[str] = None,
    local_path: Optional[Path] = None,
    user_agent: str = 'synoptickeyof.life',
    robots_url: Optional[str] = None,
    bibtex_pattern: str = 'format=bib',
    verbosity: int = 2,
    url_prefix: Optional[str] = None
) -> None:
    """
    Run a single ingestion task.

    Args:
        db: CouchDB database instance
        source: Source type ('ingenta', etc.)
        mode: Ingestion mode ('rss' or 'local')
        rss_url: RSS feed URL (required if mode='rss')
        local_path: Local directory path (required if mode='local')
        user_agent: User agent string
        robots_url: Custom robots.txt URL
        bibtex_pattern: BibTeX filename pattern
        verbosity: Verbosity level
    """
    # Set up robot parser
    robots_url_final = get_robots_url(source, robots_url)
    if verbosity >= 3:
        print(f"Loading robots.txt from {robots_url_final}")
    robot_parser = RobotFileParser()
    robot_parser.set_url(robots_url_final)
    robot_parser.read()

    # Create appropriate ingestor based on source and mode
    if source == 'ingenta':
        if mode == 'local':
            ingestor = LocalIngentaIngestor(
                db=db,
                user_agent=user_agent,
                robot_parser=robot_parser,
                verbosity=verbosity
            )
        else:
            ingestor = IngentaIngestor(
                db=db,
                user_agent=user_agent,
                robot_parser=robot_parser,
                verbosity=verbosity
            )
    elif source == 'mykoweb-journals':
        # Configure local PDF mapping for Mykoweb journals
        local_pdf_map = {
            'https://mykoweb.com/systematics/journals': '/data/skol/www/mykoweb.com/systematics/journals'
        }
        ingestor = LocalMykowebJournalsIngestor(
            db=db,
            user_agent=user_agent,
            robot_parser=robot_parser,
            verbosity=verbosity,
            local_pdf_map=local_pdf_map
        )
    elif source == 'mykoweb-literature':
        # Configure local PDF mapping for Mykoweb literature
        local_pdf_map = {
            'https://mykoweb.com/systematics/literature': '/data/skol/www/mykoweb.com/systematics/literature'
        }
        ingestor = LocalMykowebLiteratureIngestor(
            db=db,
            user_agent=user_agent,
            robot_parser=robot_parser,
            verbosity=verbosity,
            local_pdf_map=local_pdf_map
        )
    elif source == 'mykoweb-caf':
        # Configure local PDF mapping for Mykoweb CAF PDFs
        local_pdf_map = {
            'https://mykoweb.com/CAF': '/data/skol/www/mykoweb.com/CAF'
        }
        ingestor = LocalMykowebLiteratureIngestor(
            db=db,
            user_agent=user_agent,
            robot_parser=robot_parser,
            verbosity=verbosity,
            local_pdf_map=local_pdf_map
        )
    elif source == 'mykoweb-crepidotus':
        # Configure local PDF mapping for Mykoweb Crepidotus
        local_pdf_map = {
            'https://mykoweb.com/Crepidotus': '/data/skol/www/mykoweb.com/Crepidotus'
        }
        ingestor = LocalMykowebLiteratureIngestor(
            db=db,
            user_agent=user_agent,
            robot_parser=robot_parser,
            verbosity=verbosity,
            local_pdf_map=local_pdf_map
        )
    elif source == 'mykoweb-oldbooks':
        # Configure local PDF mapping for Mykoweb Old Books
        local_pdf_map = {
            'https://mykoweb.com/OldBooks': '/data/skol/www/mykoweb.com/OldBooks'
        }
        ingestor = LocalMykowebLiteratureIngestor(
            db=db,
            user_agent=user_agent,
            robot_parser=robot_parser,
            verbosity=verbosity,
            local_pdf_map=local_pdf_map
        )
    elif source == 'mykoweb-gsmnp':
        # Configure local PDF mapping for Mykoweb GSMNP
        local_pdf_map = {
            'https://mykoweb.com/GSMNP': '/data/skol/www/mykoweb.com/GSMNP'
        }
        ingestor = LocalMykowebLiteratureIngestor(
            db=db,
            user_agent=user_agent,
            robot_parser=robot_parser,
            verbosity=verbosity,
            local_pdf_map=local_pdf_map
        )
    elif source == 'mykoweb-pholiota':
        # Configure local PDF mapping for Mykoweb Pholiota
        local_pdf_map = {
            'https://mykoweb.com/Pholiota': '/data/skol/www/mykoweb.com/Pholiota'
        }
        ingestor = LocalMykowebLiteratureIngestor(
            db=db,
            user_agent=user_agent,
            robot_parser=robot_parser,
            verbosity=verbosity,
            local_pdf_map=local_pdf_map
        )
    elif source == 'mykoweb-misc':
        # Configure local PDF mapping for Mykoweb misc
        local_pdf_map = {
            'https://mykoweb.com/misc': '/data/skol/www/mykoweb.com/misc'
        }
        ingestor = LocalMykowebLiteratureIngestor(
            db=db,
            user_agent=user_agent,
            robot_parser=robot_parser,
            verbosity=verbosity,
            local_pdf_map=local_pdf_map
        )
    else:
        raise ValueError(f"Unknown source '{source}'")

    # Perform ingestion
    if mode == 'rss':
        if not rss_url:
            raise ValueError("rss_url required for RSS mode")
        if verbosity >= 2:
            print(f"Ingesting from RSS feed: {rss_url}")
        ingestor.ingest_from_rss(rss_url=rss_url)
    elif mode == 'local':
        if not local_path:
            raise ValueError("local_path required for local mode")
        if verbosity >= 2:
            print(f"Ingesting from local directory: {local_path}")

        # Call appropriate method based on ingestor type
        if isinstance(ingestor, LocalMykowebJournalsIngestor):
            ingestor.ingest_from_local_journals(root=local_path)
        elif isinstance(ingestor, LocalMykowebLiteratureIngestor):
            # Pass url_prefix if provided
            if url_prefix:
                ingestor.ingest_from_local_literature(
                    root=local_path,
                    local_path_prefix=str(local_path),
                    url_prefix=url_prefix
                )
            else:
                ingestor.ingest_from_local_literature(root=local_path)
        else:
            ingestor.ingest_from_local_bibtex(
                root=local_path,
                bibtex_file_pattern=bibtex_pattern
            )
    else:
        raise ValueError(f"Unknown mode '{mode}'")


def list_publications() -> None:
    """Print a table of all available publication sources."""
    print("\nAvailable publication Sources:")
    print("=" * 80)
    print(f"{'Key':<25} {'Name':<30} {'Type':<10} {'Details':<15}")
    print("-" * 80)

    for key, config in SOURCES.items():
        source_type = config['mode'].upper()
        if config['mode'] == 'rss':
            details = config['rss_url'].split('/')[-1][:15]
        else:
            details = "Local files"
        print(f"{key:<25} {config['name']:<30} {source_type:<10} {details:<15}")

    print("=" * 80)
    print(f"\nTotal: {len(SOURCES)} publication sources")
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
            if args.verbosity >= 2:
                print(f"Ingesting from all {len(SOURCES)} predefined sources...")
            for key, config in SOURCES.items():
                if args.verbosity >= 2:
                    print(f"\n{'=' * 60}")
                    print(f"Processing: {config['name']} ({key})")
                    print(f"{'=' * 60}")

                run_ingestion(
                    db=db,
                    source=config['source'],
                    mode=config['mode'],
                    rss_url=config.get('rss_url'),
                    local_path=Path(config['local_path']) if config.get('local_path') else None,
                    user_agent=args.user_agent,
                    robots_url=args.robots_url,
                    bibtex_pattern=args.bibtex_pattern,
                    verbosity=args.verbosity,
                    url_prefix=config.get('url_prefix')
                )
        elif args.publication:
            # Use predefined source
            config = SOURCES[args.publication]
            if args.verbosity >= 2:
                print(f"Using publication: {config['name']}")

            run_ingestion(
                db=db,
                source=config['source'],
                mode=config['mode'],
                rss_url=config.get('rss_url'),
                local_path=Path(config['local_path']) if config.get('local_path') else None,
                user_agent=args.user_agent,
                robots_url=args.robots_url,
                bibtex_pattern=args.bibtex_pattern,
                verbosity=args.verbosity,
                url_prefix=config.get('url_prefix')
            )
        elif args.rss:
            # Direct RSS mode
            run_ingestion(
                db=db,
                source=args.source,
                mode='rss',
                rss_url=args.rss,
                user_agent=args.user_agent,
                robots_url=args.robots_url,
                bibtex_pattern=args.bibtex_pattern,
                verbosity=args.verbosity
            )
        elif args.local:
            # Direct local mode
            run_ingestion(
                db=db,
                source=args.source,
                mode='local',
                local_path=args.local,
                user_agent=args.user_agent,
                robots_url=args.robots_url,
                bibtex_pattern=args.bibtex_pattern,
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
        if args.verbosity >= 3:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
