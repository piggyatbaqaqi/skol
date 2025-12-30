#!/usr/bin/env python3
"""
Example program to retrieve all articles for a given journal ISSN
using the Crossref API via habanero.
This program demonstrates how to use the habanero library to fetch
all articles (works) for a specified journal ISSN, handling pagination
to retrieve the complete set of articles.
"""
from habanero import Crossref
import time

def get_journal_articles_by_issn(issn, mailto_email):
    """
    Retrieves all articles (works) for a given journal ISSN using the Crossref API via habanero.

    Args:
        issn (str): The ISSN of the journal.
        mailto_email (str): Your email address for the Crossref polite pool.

    Returns:
        list: A list of all article records (dictionaries).
    """
    cr = Crossref(mailto=mailto_email)
    all_articles = []
    cursor = "*"
    rows_per_page = 1000  # Max allowed by API

    print(f"Starting retrieval for ISSN: {issn}")

    limit = 10
    counter = 0
    # Loop to fetch all pages of results using the cursor
    while True:
        counter += 1
        if counter > limit:
            print("Reached limit of pages for testing purposes.")
            break
        try:
            results = cr.journals(
                issn=issn,
                works=True,
                cursor=cursor,
                rows=rows_per_page
            )

            # Extract the articles from the current page
            articles = results['message']['items']
            if not articles:
                break  # Exit loop if no more articles are returned

            all_articles.extend(articles)
            print(f"Fetched {len(articles)} articles in this page. Total so far: {len(all_articles)}")

            # Get the next cursor value
            cursor = results['message'].get('next-cursor')
            if not cursor:
                break

            # Be polite to the API (optional but recommended)
            time.sleep(0.5)

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    print(f"Finished retrieval. Total articles for ISSN {issn}: {len(all_articles)}")
    return all_articles

# --- Example Usage ---
# Replace with the actual ISSN and your email address
JOURNAL_ISSN = "2309-608X" # Example ISSN for Journal of Fungi
YOUR_EMAIL = "piggy.yarroll+skol@gmail.com" # Important for the "polite pool"

if __name__ == "__main__":
    articles_list = get_journal_articles_by_issn(JOURNAL_ISSN, YOUR_EMAIL)
    print(f"Total articles retrieved: {len(articles_list)}")
    print("Sample article data:")
    for article in articles_list[:5]:  # Print details for the first 5 articles
        print(f"DOI: {article.get('DOI')}, Title: {article.get('title', ['N/A'])[0]}")
