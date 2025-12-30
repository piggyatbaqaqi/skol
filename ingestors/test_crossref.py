#!/usr/bin/env python3
"""
Test program for CrossrefIngestor.

Retrieves a small number of articles from a journal by ISSN to test
the Crossref API integration, metadata extraction, and PDF downloading.

Usage:
    ./test_crossref.py [ISSN] [--count N] [--no-pdf]
    ./test_crossref.py                    # Default: 2309-608X, 2 articles
    ./test_crossref.py 1234-5678          # Different ISSN
    ./test_crossref.py --count 5          # Get 5 articles
    ./test_crossref.py --no-pdf           # Skip PDF downloads (faster)
"""

import sys
import argparse
import json
from pathlib import Path

try:
    from crossref import CrossrefIngestor
except ImportError:
    try:
        from ingestors.crossref import CrossrefIngestor
    except ImportError:
        print("ERROR: Could not import CrossrefIngestor", file=sys.stderr)
        sys.exit(1)

try:
    from mock_database import MockDatabase
except ImportError:
    try:
        from ingestors.mock_database import MockDatabase
    except ImportError:
        print("ERROR: Could not import MockDatabase", file=sys.stderr)
        sys.exit(1)

from urllib.robotparser import RobotFileParser


def test_crossref_ingestor(
    issn: str = '2309-608X',
    count: int = 2,
    skip_pdf: bool = False,
    verbosity: int = 3
):
    """
    Test CrossrefIngestor with a small number of articles.

    Args:
        issn: ISSN of the journal to test
        count: Number of articles to retrieve
        skip_pdf: If True, skip PDF downloads (faster testing)
        verbosity: Verbosity level (0-3)
    """
    print("=" * 80)
    print("CrossrefIngestor Test Program")
    print("=" * 80)
    print()
    print(f"ISSN: {issn}")
    print(f"Articles to retrieve: {count}")
    print(f"Skip PDF downloads: {skip_pdf}")
    print(f"Verbosity: {verbosity}")
    print()

    # Create mock database
    db = MockDatabase()

    # Create robot parser (allows everything for testing)
    robot_parser = RobotFileParser()
    robot_parser.set_url('https://api.crossref.org/robots.txt')
    robot_parser.read()

    # Create ingestor with test configuration
    print("Initializing CrossrefIngestor...")
    ingestor = CrossrefIngestor(
        db=db,
        user_agent='CrossrefIngestorTest/1.0 (testing)',
        robot_parser=robot_parser,
        issn=issn,
        mailto='piggy.yarroll+skol@gmail.com',
        max_articles=count,
        allow_scihub=not skip_pdf,  # Only allow Sci-Hub if we're downloading PDFs
        verbosity=verbosity
    )

    # Override _download_pdf_with_pypaperretriever if skipping PDFs
    if skip_pdf:
        print("PDF downloads disabled for faster testing")
        ingestor._download_pdf_with_pypaperretriever = lambda doi: None

    # Run ingestion
    print()
    print("-" * 80)
    print("Starting ingestion...")
    print("-" * 80)
    print()

    try:
        ingestor.ingest()
    except Exception as e:
        print(f"\nERROR during ingestion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Display results
    print()
    print("=" * 80)
    print("Ingestion Complete - Results Summary")
    print("=" * 80)
    print()

    documents = db.get_documents()

    if not documents:
        print("No documents were ingested.")
        return 1

    print(f"Total documents ingested: {len(documents)}")
    print()

    # Display each document
    for idx, doc in enumerate(documents, 1):
        print(f"Document {idx}:")
        print("-" * 80)
        print(f"  DOI:         {doc.get('doi', 'N/A')}")
        print(f"  Title:       {doc.get('title', 'N/A')[:70]}...")
        print(f"  Authors:     {doc.get('author', 'N/A')[:70]}...")
        print(f"  Year:        {doc.get('year', 'N/A')}")
        print(f"  Journal:     {doc.get('journal', 'N/A')}")
        print(f"  Volume:      {doc.get('volume', 'N/A')}")
        print(f"  Issue:       {doc.get('issue', 'N/A')}")
        print(f"  Pages:       {doc.get('pages', 'N/A')}")
        print(f"  PDF URL:     {doc.get('pdf_url', 'N/A')}")
        print(f"  Human URL:   {doc.get('human_url', 'N/A')}")
        print(f"  BibTeX URL:  {doc.get('bibtex_url', 'N/A')}")

        # Check for attachments
        if '_attachments' in doc:
            print(f"  Attachments:")
            for name, info in doc['_attachments'].items():
                size = info.get('length', 0)
                print(f"    - {name}: {size:,} bytes")
        else:
            print(f"  Attachments: None")

        print()

    # Save results to JSON file for inspection
    output_file = Path('test_crossref_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'issn': issn,
            'count_requested': count,
            'count_retrieved': len(documents),
            'documents': documents
        }, f, indent=2, default=str)

    print(f"Full results saved to: {output_file}")
    print()

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test CrossrefIngestor with a small number of articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default ISSN (Journal of Fungi) and 2 articles
  %(prog)s

  # Test with different ISSN
  %(prog)s 0093-4666

  # Get 5 articles
  %(prog)s --count 5

  # Skip PDF downloads for faster testing
  %(prog)s --no-pdf

  # Verbose output
  %(prog)s -v 3

  # Quiet mode
  %(prog)s -v 1

  # Test Mycotaxon with 3 articles, no PDFs
  %(prog)s 0093-4666 --count 3 --no-pdf
        """
    )

    parser.add_argument(
        'issn',
        nargs='?',
        default='2309-608X',
        help='ISSN or eISSN of the journal (default: 2309-608X for Journal of Fungi)'
    )
    parser.add_argument(
        '--count', '-c',
        type=int,
        default=2,
        help='Number of articles to retrieve (default: 2)'
    )
    parser.add_argument(
        '--no-pdf',
        action='store_true',
        help='Skip PDF downloads for faster testing'
    )
    parser.add_argument(
        '--verbosity', '-v',
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help='Verbosity level: 0=silent, 1=errors, 2=normal, 3=verbose (default: 2)'
    )

    args = parser.parse_args()

    return test_crossref_ingestor(
        issn=args.issn,
        count=args.count,
        skip_pdf=args.no_pdf,
        verbosity=args.verbosity
    )


if __name__ == '__main__':
    sys.exit(main())
