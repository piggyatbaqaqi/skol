#!/usr/bin/env python3
"""
Fetch all DOI numbers for articles in a journal using Crossref API via habanero.

Usage:
    ./get_journal_dois.py <ISSN> [--output file.txt] [--email your@email.com]
    ./get_journal_dois.py 2309-608X --output jof_dois.txt
    ./get_journal_dois.py 0027-5514 --email scholar@example.com

This script queries the Crossref API to retrieve all DOIs for articles published
in a journal identified by its ISSN or eISSN.
"""

import sys
import argparse
import time
from typing import List, Optional
from pathlib import Path

try:
    from habanero import Crossref
except ImportError:
    print("ERROR: habanero library not found.", file=sys.stderr)
    print("Install with: pip install habanero", file=sys.stderr)
    sys.exit(1)


def get_journal_dois(
    issn: str,
    mailto: Optional[str] = None,
    verbose: bool = True,
    rate_limit_delay: float = 0.1
) -> List[str]:
    """
    Fetch all DOI numbers for articles in a journal.

    Args:
        issn: ISSN or eISSN of the journal (with or without hyphen)
        mailto: Email address for polite API usage (recommended by Crossref)
        verbose: Print progress messages
        rate_limit_delay: Delay between requests in seconds (default: 0.1)

    Returns:
        List of DOI strings
    """
    # Normalize ISSN format (add hyphen if missing)
    if '-' not in issn and len(issn) == 8:
        issn = f"{issn[:4]}-{issn[4:]}"

    if verbose:
        print(f"Querying Crossref for journal ISSN: {issn}")
        if mailto:
            print(f"Using mailto: {mailto} (polite pool - faster API access)")

    # Initialize Crossref client
    cr = Crossref(mailto=mailto) if mailto else Crossref()

    if verbose:
        print("Fetching DOIs (this may take a while for large journals)...")

    dois = []
    cursor = '*'
    batch_num = 0
    per_page = 100  # Crossref allows up to 1000 results per request
    max_works = 200  # Safety limit to avoid infinite loops
    total_works = 0
    # Iterate through all results using cursor-based pagination
    while True:
        batch_num += 1

        try:
            results = cr.works(filter={'issn': issn}, limit=per_page, cursor=cursor, progress_bar=True)
        except Exception as e:
            print(f"ERROR: Failed to fetch batch {batch_num}: {e}", file=sys.stderr)
            break

        if verbose and batch_num > 1:
            print(f"  Batch {batch_num}: fetched {len(dois):,}, DOIs...")

        if verbose:
            print(f" {len(results):,} results in this batch")

        for result in results:
            if not result or 'message' not in result:
                break

            message = result['message']
            items = message.get('items', [])

            if not items:
                break

            # Extract DOIs from this batch
            for item in items:
                total_works += 1
                if total_works > max_works:
                    if verbose:
                        print(f"Reached maximum work limit of {max_works}. Stopping.", file=sys.stderr)
                    return dois
                doi = item.get('DOI')
                if doi:
                    dois.append(doi)

        # Get next cursor
        next_cursor = message.get('next-cursor')
        if not next_cursor or next_cursor == cursor:
            # No more pages
            break

        cursor  = next_cursor

        # Rate limiting - be polite to Crossref API
        time.sleep(rate_limit_delay)

    if verbose:
        print(f"\nCompleted! Retrieved {len(dois):,} DOIs")

    return dois


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch all DOI numbers for articles in a journal using Crossref API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch DOIs for Journal of Fungi (2309-608X)
  %(prog)s 2309-608X

  # Save to file
  %(prog)s 2309-608X --output jof_dois.txt

  # Use email for polite pool (faster API access)
  %(prog)s 2309-608X --email your@email.com --output dois.txt

  # Quiet mode (only output DOIs)
  %(prog)s 2309-608X --quiet > dois.txt

Notes:
  - Providing --email enables Crossref's "polite pool" for faster access
  - The script uses cursor-based pagination to handle journals with many articles
  - Rate limiting is built-in to avoid overwhelming the API
        """
    )

    parser.add_argument(
        'issn',
        help='ISSN or eISSN of the journal (e.g., 2309-608X or 2309608X)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file to save DOIs (one per line)'
    )
    parser.add_argument(
        '--email',
        type=str,
        help='Email address for polite API usage (recommended by Crossref)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages (only output DOIs)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.1,
        help='Delay between API requests in seconds (default: 0.1)'
    )

    args = parser.parse_args()

    # Fetch DOIs
    dois = get_journal_dois(
        issn=args.issn,
        mailto=args.email,
        verbose=not args.quiet,
        rate_limit_delay=args.rate_limit
    )

    if not dois:
        return 1

    # Output results
    if args.output:
        # Save to file
        try:
            with open(args.output, 'w') as f:
                for doi in dois:
                    f.write(f"{doi}\n")
            if not args.quiet:
                print(f"\nSaved {len(dois):,} DOIs to {args.output}")
        except IOError as e:
            print(f"ERROR: Failed to write to {args.output}: {e}", file=sys.stderr)
            return 1
    else:
        # Print to stdout
        for doi in dois:
            print(doi)

    return 0


if __name__ == '__main__':
    sys.exit(main())
