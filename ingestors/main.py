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
except ImportError:
    from ingestors.ingenta import IngentaIngestor


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
  # Ingest from Ingenta RSS feed
  %(prog)s --source ingenta --rss https://api.ingentaconnect.com/content/mtax/mt?format=rss

  # Ingest from local BibTeX files
  %(prog)s --source ingenta --local /data/skol/www/www.ingentaconnect.com

  # With credentials from command line
  %(prog)s --source ingenta --rss <url> --couchdb-username user --couchdb-password pass

  # With credentials from environment variables
  export COUCHDB_USER=myuser COUCHDB_PASSWORD=mypass
  %(prog)s --source ingenta --rss <url>

  # Silent mode
  %(prog)s --source ingenta --rss <url> --verbosity 0

  # Verbose mode
  %(prog)s --source ingenta --rss <url> -v 3
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

    # Ingestor selection
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['ingenta'],
        help='Data source to ingest from'
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
    }

    return robots_urls.get(source, '')


def main() -> int:
    """
    Main entry point for the ingestor CLI.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args()

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

        # Set up robot parser
        robots_url = get_robots_url(args.source, args.robots_url)
        if args.verbosity >= 3:
            print(f"Loading robots.txt from {robots_url}")
        robot_parser = RobotFileParser()
        robot_parser.set_url(robots_url)
        robot_parser.read()

        # Create appropriate ingestor
        if args.source == 'ingenta':
            ingestor = IngentaIngestor(
                db=db,
                user_agent=args.user_agent,
                robot_parser=robot_parser,
                verbosity=args.verbosity
            )
        else:
            print(f"Error: Unknown source '{args.source}'", file=sys.stderr)
            return 1

        # Perform ingestion
        if args.rss:
            if args.verbosity >= 2:
                print(f"Ingesting from RSS feed: {args.rss}")
            ingestor.ingest_from_rss(rss_url=args.rss)
        elif args.local:
            if args.verbosity >= 2:
                print(f"Ingesting from local directory: {args.local}")
            ingestor.ingest_from_local_bibtex(
                root=args.local,
                bibtex_file_pattern=args.bibtex_pattern
            )

        if args.verbosity >= 2:
            print("Ingestion complete!")

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
