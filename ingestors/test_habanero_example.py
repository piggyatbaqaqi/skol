#!/usr/bin/env python3
"""
Example showing how habanero works for fetching journal DOIs.

This is a demonstration script showing the basic concept.
For the full implementation, see get_journal_dois.py
"""

# Example of what habanero does (conceptual - requires habanero to be installed)
def example_habanero_usage():
    """
    Example showing how to use habanero to fetch DOIs from Crossref.

    This is the core concept used in get_journal_dois.py
    """
    print("Example habanero usage:")
    print("-" * 60)
    print()
    print("# Import habanero")
    print("from habanero import Crossref")
    print()
    print("# Initialize client (optionally with email for faster access)")
    print("cr = Crossref(mailto='your@email.com')")
    print()
    print("# Query for articles from a journal by ISSN")
    print("issn = '2309-608X'  # Journal of Fungi")
    print("result = cr.works(filter={'issn': issn}, limit=1000, cursor='*')")
    print()
    print("# Result structure:")
    print("{")
    print("  'message': {")
    print("    'total-results': 12345,")
    print("    'items': [")
    print("      {'DOI': '10.3390/jof12010028', 'title': [...], ...},")
    print("      {'DOI': '10.3390/jof12010027', 'title': [...], ...},")
    print("      ...")
    print("    ],")
    print("    'next-cursor': 'AoJ/rZR...'  # For pagination")
    print("  }")
    print("}")
    print()
    print("# Extract DOIs from results")
    print("dois = [item['DOI'] for item in result['message']['items']]")
    print()
    print("# Use cursor for next batch")
    print("next_cursor = result['message']['next-cursor']")
    print("result = cr.works(filter={'issn': issn}, limit=1000, cursor=next_cursor)")


def example_output():
    """Show example output format."""
    print("\n\nExample output from get_journal_dois.py:")
    print("=" * 60)
    print()
    print("$ ./get_journal_dois.py 2309-608X --email scholar@example.com")
    print()
    print("Querying Crossref for journal ISSN: 2309-608X")
    print("Using mailto: scholar@example.com (polite pool - faster API access)")
    print("Fetching first batch to determine total article count...")
    print("Found 1,234 total articles")
    print("Fetching DOIs (this may take a while for large journals)...")
    print("  Batch 2: fetched 1,000/1,234 DOIs...")
    print()
    print("Completed! Retrieved 1,234 DOIs")
    print()
    print("10.3390/jof12010028")
    print("10.3390/jof12010027")
    print("10.3390/jof12010026")
    print("10.3390/jof12010025")
    print("...")
    print()
    print("# With --output flag:")
    print("$ ./get_journal_dois.py 2309-608X -o jof_dois.txt --email scholar@example.com")
    print()
    print("Querying Crossref for journal ISSN: 2309-608X")
    print("...")
    print("Completed! Retrieved 1,234 DOIs")
    print()
    print("Saved 1,234 DOIs to jof_dois.txt")


def example_use_cases():
    """Show common use cases."""
    print("\n\nCommon use cases:")
    print("=" * 60)
    print()
    print("1. Get all DOIs for a journal:")
    print("   ./get_journal_dois.py 2309-608X > dois.txt")
    print()
    print("2. Count articles in a journal:")
    print("   ./get_journal_dois.py 2309-608X --quiet | wc -l")
    print()
    print("3. Get DOIs from multiple journals:")
    print("   for issn in 2309-608X 0093-4666 0031-5850; do")
    print("     ./get_journal_dois.py $issn >> all_dois.txt")
    print("   done")
    print()
    print("4. Check if a DOI exists in journal:")
    print("   ./get_journal_dois.py 2309-608X --quiet | grep '10.3390/jof12010028'")


if __name__ == '__main__':
    print("=" * 60)
    print("Habanero + Crossref API - Example Usage")
    print("=" * 60)

    example_habanero_usage()
    example_output()
    example_use_cases()

    print("\n\n" + "=" * 60)
    print("To use the actual script:")
    print("=" * 60)
    print("1. Install habanero: pip install habanero")
    print("2. Run: ./get_journal_dois.py <ISSN>")
    print("3. See GET_JOURNAL_DOIS_README.md for full documentation")
    print()
