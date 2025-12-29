#!/usr/bin/env python3
"""
Test program for MycosphereIngestor.

This program tests the metadata extraction from Mycosphere journal pages
without actually ingesting into the database. It displays extracted metadata
from one index page and one issue page.
"""

import sys
import json
from pathlib import Path
from urllib.robotparser import RobotFileParser

# Add parent directory to path for imports
if __name__ == '__main__' and __package__ is None:
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# Import mock database
try:
    from ingestors.mock_database import MockDatabase
except ImportError:
    from mock_database import MockDatabase


# Import the ingestor
try:
    from ingestors.mycosphere import MycosphereIngestor
except ImportError:
    from mycosphere import MycosphereIngestor


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def print_article(article, index):
    """Print article metadata in a readable format."""
    print(f"\nArticle {index + 1}:")
    print("-" * 60)

    # Print each field
    for key in ['title', 'authors', 'volume', 'number', 'year',
                'publication_date', 'acceptance_date', 'receipt_date',
                'pages', 'issn', 'eissn', 'url', 'pdf_url']:
        if key in article:
            value = article[key]
            print(f"  {key:20s}: {value}")

    # Print abstract (truncated)
    if 'abstract' in article:
        abstract = article['abstract']
        if len(abstract) > 200:
            abstract = abstract[:200] + '...'
        print(f"  {'abstract':20s}: {abstract}")

    # Print keywords
    if 'keywords' in article:
        keywords = article['keywords']
        if len(keywords) > 100:
            keywords = keywords[:100] + '...'
        print(f"  {'keywords':20s}: {keywords}")


def test_volume_page(ingestor, volume_url, volume_num):
    """
    Test metadata extraction from a volume index page.

    Args:
        ingestor: MycosphereIngestor instance
        volume_url: URL of volume page
        volume_num: Volume number
    """
    print_separator()
    print(f"Testing Volume Index Page: {volume_url}")
    print_separator()

    # Fetch and parse page
    soup = ingestor._fetch_page(volume_url)
    if not soup:
        print("ERROR: Failed to fetch page")
        return []

    # Extract articles
    articles = ingestor._extract_article_metadata(soup, volume_num)

    print(f"\nFound {len(articles)} article(s)")

    # Print each article
    for idx, article in enumerate(articles):
        print_article(article, idx)

    return articles


def test_issue_page(ingestor, issue_url, volume_num, issue_num):
    """
    Test metadata extraction from an issue page.

    Args:
        ingestor: MycosphereIngestor instance
        issue_url: URL of issue page
        volume_num: Volume number
        issue_num: Issue number
    """
    print_separator()
    print(f"Testing Issue Page: {issue_url}")
    print_separator()

    # Fetch and parse page
    soup = ingestor._fetch_page(issue_url)
    if not soup:
        print("ERROR: Failed to fetch page")
        return []

    # Extract articles
    articles = ingestor._extract_article_metadata(soup, volume_num, issue_num)

    print(f"\nFound {len(articles)} article(s)")

    # Print each article
    for idx, article in enumerate(articles):
        print_article(article, idx)

    return articles


def test_archives_page(ingestor):
    """
    Test volume link extraction from archives page.

    Args:
        ingestor: MycosphereIngestor instance
    """
    print_separator()
    print(f"Testing Archives Page: {ingestor.ARCHIVES_URL}")
    print_separator()

    # Fetch and parse archives page
    soup = ingestor._fetch_page(ingestor.ARCHIVES_URL)
    if not soup:
        print("ERROR: Failed to fetch archives page")
        return []

    # Extract volume links
    volume_links = ingestor._extract_volume_links(soup)

    print(f"\nFound {len(volume_links)} volume link(s)")
    print("\nVolume Links:")
    print("-" * 60)

    for vol in volume_links:
        print(f"  Volume {vol['volume']:3s} ({vol['year']}): {vol['url']}")

    return volume_links


def test_complete_workflow(ingestor, volume_url, volume_num):
    """
    Test the complete workflow: extract from index, enhance with issue pages.

    Args:
        ingestor: MycosphereIngestor instance
        volume_url: URL of volume index page
        volume_num: Volume number
    """
    print_separator()
    print(f"Testing Complete Workflow on Volume {volume_num}")
    print_separator()

    # Step 1: Extract from volume index (gets page numbers)
    print(f"\nStep 1: Extracting from volume index: {volume_url}")
    vol_soup = ingestor._fetch_page(volume_url)
    if not vol_soup:
        print("ERROR: Failed to fetch volume page")
        return []

    articles = ingestor._extract_articles_from_volume_index(vol_soup, volume_num)
    print(f"Extracted {len(articles)} articles from volume index")

    # Show sample with page numbers
    print("\nSample articles with page numbers:")
    for i, article in enumerate(articles[:3]):
        print(f"  {i+1}. {article.get('title', '')[:50]}...")
        print(f"     Pages: {article.get('pages', 'NOT FOUND')}, Issue: {article.get('number', 'N/A')}")

    # Step 2: Check for issue pages
    issue_links = ingestor._extract_issue_links(vol_soup, volume_num)
    print(f"\nStep 2: Found {len(issue_links)} issue pages for metadata enhancement")

    # Step 3: Enhance with metadata from first issue
    if issue_links:
        first_issue = issue_links[0]
        print(f"\nStep 3: Enhancing with metadata from Issue {first_issue['number']}")

        issue_soup = ingestor._fetch_page(first_issue['url'])
        if issue_soup:
            issue_articles = ingestor._extract_article_metadata(
                issue_soup,
                first_issue['volume'],
                first_issue['number']
            )

            print(f"Extracted {len(issue_articles)} articles from issue page")

            # Create PDF map and enhance
            pdf_map = {article['pdf_url']: article for article in articles}
            enhanced_count = 0

            for issue_article in issue_articles:
                if 'pdf_url' in issue_article and issue_article['pdf_url'] in pdf_map:
                    article = pdf_map[issue_article['pdf_url']]

                    # Add metadata from issue page
                    for key in ['number', 'authors', 'abstract', 'keywords',
                               'receipt_date', 'acceptance_date', 'publication_date', 'year']:
                        if key in issue_article and issue_article[key]:
                            article[key] = issue_article[key]

                    enhanced_count += 1

            print(f"Enhanced {enhanced_count} articles with metadata")

    # Show final results
    print("\n" + "=" * 80)
    print("Final Results (first 3 articles):")
    print("=" * 80)

    for idx, article in enumerate(articles[:3]):
        print_article(article, idx)

    return articles


def save_to_json(articles, filename):
    """
    Save articles to JSON file.

    Args:
        articles: List of article dicts
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(articles, f, indent=2)
    print(f"\nSaved {len(articles)} article(s) to {filename}")


def main():
    """Main test function."""
    print("\nMycosphere Ingestor Test Program")
    print_separator('=')

    # Create mock database
    mock_db = MockDatabase()

    # Set up robot parser
    robot_parser = RobotFileParser()
    robot_parser.set_url('https://mycosphere.org/robots.txt')
    robot_parser.read()

    # Create ingestor with verbose output
    ingestor = MycosphereIngestor(
        db=mock_db,
        user_agent='synoptickeyof.life (testing)',
        robot_parser=robot_parser,
        verbosity=3
    )

    # Test 1: Extract volume links from archives
    print("\n\n" + "=" * 80)
    print("TEST 1: Extract Volume Links from Archives Page")
    print("=" * 80)
    volume_links = test_archives_page(ingestor)

    if not volume_links:
        print("\nERROR: No volume links found. Cannot proceed with further tests.")
        return 1

    # Select first volume for testing
    test_volume = volume_links[0]
    print(f"\n\nUsing Volume {test_volume['volume']} for testing: {test_volume['url']}")

    # Test 2: NEW - Complete workflow test
    print("\n\n" + "=" * 80)
    print("TEST 2: Complete Workflow (Index + Issue Enhancement)")
    print("=" * 80)

    final_articles = test_complete_workflow(
        ingestor,
        test_volume['url'],
        test_volume['volume']
    )

    if final_articles:
        save_to_json(final_articles, 'mycosphere_complete_test.json')

    print("\n\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
