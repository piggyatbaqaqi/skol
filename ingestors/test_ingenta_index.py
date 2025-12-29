#!/usr/bin/env python3
"""
Test program for IngentaIngestor index mode.

This program tests the metadata extraction from IngentaConnect index pages
without actually ingesting into the database. It displays extracted metadata
from the index page, issue pages, and individual articles.
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
    from ingestors.ingenta import IngentaIngestor
except ImportError:
    from ingenta import IngentaIngestor


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def print_article(article, index):
    """Print article metadata in a readable format."""
    print(f"\nArticle {index + 1}:")
    print("-" * 60)

    # Print each field
    for key in ['title', 'authors', 'volume', 'number', 'pages',
                'publication_date', 'doi', 'document_type',
                'url', 'itemtype']:
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


def test_index_page(ingestor):
    """
    Test volume/issue link extraction from index page.

    Args:
        ingestor: IngentaIngestor instance

    Returns:
        List of issue link dicts
    """
    print_separator()
    print(f"Testing Index Page: {ingestor.index_url}")
    print_separator()

    # Fetch and parse index page
    soup = ingestor._fetch_page(ingestor.index_url)
    if not soup:
        print("ERROR: Failed to fetch index page")
        return []

    # Extract volume/issue links
    issue_links = ingestor._extract_volume_issue_links(soup)

    print(f"\nFound {len(issue_links)} issue link(s)")
    print("\nIssue Links (first 10):")
    print("-" * 60)

    for issue in issue_links[:10]:
        print(f"  Volume {issue['volume']:3s}, "
              f"Issue {issue['issue']:3s}: {issue['url']}")

    if len(issue_links) > 10:
        print(f"  ... and {len(issue_links) - 10} more")

    return issue_links


def test_issue_page(ingestor, issue_url, volume, issue_num):
    """
    Test metadata extraction from an issue page.

    Args:
        ingestor: IngentaIngestor instance
        issue_url: URL of issue page
        volume: Volume number
        issue_num: Issue number

    Returns:
        List of article dicts
    """
    print_separator()
    print(f"Testing Issue Page: {issue_url}")
    print(f"Volume {volume}, Issue {issue_num}")
    print_separator()

    # Fetch and parse page
    soup = ingestor._fetch_page(issue_url)
    if not soup:
        print("ERROR: Failed to fetch page")
        return []

    # Extract articles (only open access)
    articles = ingestor._extract_articles_from_issue(soup, volume, issue_num)

    print(f"\nFound {len(articles)} open access article(s)")

    # Print each article
    for idx, article in enumerate(articles):
        print_article(article, idx)

    return articles


def test_article_metadata(ingestor, article_url):
    """
    Test metadata extraction from article page.

    Args:
        ingestor: IngentaIngestor instance
        article_url: URL of article page

    Returns:
        Dict with detailed metadata
    """
    print_separator()
    print(f"Testing Article Metadata: {article_url}")
    print_separator()

    # Extract metadata
    metadata = ingestor._extract_article_metadata(article_url)

    print("\nExtracted Metadata:")
    print("-" * 60)

    for key, value in metadata.items():
        if value:
            if key == 'abstract' and len(value) > 300:
                value = value[:300] + '...'
            print(f"  {key:20s}: {value}")
        else:
            print(f"  {key:20s}: (not found)")

    return metadata


def test_complete_workflow(ingestor, max_issues=1):
    """
    Test the complete workflow: extract from index through to articles.

    Args:
        ingestor: IngentaIngestor instance
        max_issues: Maximum number of issues to process

    Returns:
        List of all extracted articles
    """
    print_separator()
    print("Testing Complete Workflow")
    print_separator()

    # Step 1: Extract issue links from index
    print(f"\nStep 1: Extracting issue links from: {ingestor.index_url}")
    index_soup = ingestor._fetch_page(ingestor.index_url)
    if not index_soup:
        print("ERROR: Failed to fetch index page")
        return []

    issue_links = ingestor._extract_volume_issue_links(index_soup)
    print(f"Found {len(issue_links)} issue(s)")

    if not issue_links:
        print("ERROR: No issues found")
        return []

    # Step 2: Process first issue
    all_articles = []
    for idx, issue_info in enumerate(issue_links[:max_issues]):
        print(f"\nStep 2.{idx+1}: Processing Volume {issue_info['volume']}, "
              f"Issue {issue_info['issue']}")

        # Fetch issue page
        issue_soup = ingestor._fetch_page(issue_info['url'])
        if not issue_soup:
            print(f"ERROR: Failed to fetch issue page")
            continue

        # Extract articles from issue
        articles = ingestor._extract_articles_from_issue(
            issue_soup,
            issue_info['volume'],
            issue_info['issue']
        )

        print(f"Extracted {len(articles)} open access articles from issue page")
        all_articles.extend(articles)

    # Show final results
    print("\n" + "=" * 80)
    print(f"Final Results: {len(all_articles)} total open access articles")
    print("=" * 80)

    print("\nShowing first 3 articles:")
    for idx, article in enumerate(all_articles[:3]):
        print_article(article, idx)

    return all_articles


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
    print("\nIngenta Index Mode Ingestor Test Program")
    print("Testing Studies in Mycology")
    print_separator('=')

    # Create mock database
    mock_db = MockDatabase()

    # Set up robot parser
    robot_parser = RobotFileParser()
    robot_parser.set_url('https://www.ingentaconnect.com/robots.txt')
    robot_parser.read()

    # Create ingestor with verbose output
    ingestor = IngentaIngestor(
        db=mock_db,
        user_agent='synoptickeyof.life (testing)',
        robot_parser=robot_parser,
        verbosity=3,
        index_url='https://api.ingentaconnect.com/content/wfbi/sim?format=index',
        rate_limit_min_ms=1000,
        rate_limit_max_ms=5000
    )

    # Test 1: Extract issue links from index
    print("\n\n" + "=" * 80)
    print("TEST 1: Extract Issue Links from Index Page")
    print("=" * 80)
    issue_links = test_index_page(ingestor)

    if not issue_links:
        print("\nERROR: No issue links found. Cannot proceed with tests.")
        return 1

    # Select first issue for detailed testing
    test_issue = issue_links[0]
    print(f"\n\nUsing Volume {test_issue['volume']}, Issue {test_issue['issue']} "
          f"for detailed testing: {test_issue['url']}")

    # Test 2: Extract articles from first issue
    print("\n\n" + "=" * 80)
    print("TEST 2: Extract Open Access Articles from Issue Page")
    print("=" * 80)

    articles = test_issue_page(
        ingestor,
        test_issue['url'],
        test_issue['volume'],
        test_issue['issue']
    )

    if articles:
        save_to_json(articles, 'ingenta_index_test.json')

        # Test 3: Test article metadata extraction on first article
        if articles[0].get('url'):
            print("\n\n" + "=" * 80)
            print("TEST 3: Extract Detailed Metadata from Article Page")
            print("=" * 80)

            test_article_metadata(ingestor, articles[0]['url'])

    # Test 4: Complete workflow (single issue)
    print("\n\n" + "=" * 80)
    print("TEST 4: Complete Workflow (Index -> Issue -> Articles)")
    print("=" * 80)

    final_articles = test_complete_workflow(ingestor, max_issues=1)

    if final_articles:
        save_to_json(final_articles, 'ingenta_index_complete_test.json')

    print("\n\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
