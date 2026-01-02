"""
PensoftIngestor for ingesting articles from Pensoft journals.

This module provides the PensoftIngestor class for scraping and ingesting
articles from Pensoft publishing platform journals.
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin
from datetime import datetime

from bs4 import BeautifulSoup

from .ingestor import Ingestor


class PensoftIngestor(Ingestor):
    """
    Ingestor for Pensoft journals (pensoft.net).

    This class scrapes Pensoft journal websites to extract article metadata
    and PDFs. It handles issue listing pages, individual issue pages with
    pagination, and article downloads.

    Journal information (MycoKeys example):
        ISSN: 1314-4057
        eISSN: 1314-4049
        URL: https://mycokeys.pensoft.net/

    Navigation:
        - Start: https://mycokeys.pensoft.net/issues
        - Browse issues with pagination: browse_journal_issues.php?p=[PAGE]
        - Issue pages: /issue/[ISSUE_ID]/
        - Issue pagination: browse_journal_issue_documents.php?issue_id=[ID]&p=[PAGE]
        - Article pages: /article/[ARTICLE_ID]/
        - PDF downloads: /article/[ARTICLE_ID]/download/pdf/[PDF_ID]

    Metadata extracted:
        - title
        - authors
        - volume, number (issue)
        - publication_date
        - year
        - doi
        - pages
        - article_type (Research Article, etc.)

    Note: Issue IDs are non-sequential database identifiers.
          PDF IDs differ from article IDs and must be extracted from links.
    """

    def __init__(
        self,
        journal_name: str = 'mycokeys',
        journal_id: str = '11',
        issn: Optional[str] = None,
        eissn: Optional[str] = None,
        issues_url: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the PensoftIngestor.

        Args:
            journal_name: Journal name for URLs (default: 'mycokeys')
            journal_id: Journal ID for pagination URLs (default: '11')
            issn: Journal ISSN
            eissn: Journal eISSN
            issues_url: Issues URL to scrape from (constructed from journal_name if not provided)
            **kwargs: Base class arguments (db, user_agent, robot_parser, etc.)
        """
        super().__init__(**kwargs)
        self.journal_name = journal_name
        self.journal_id = journal_id
        self.issn = issn
        self.eissn = eissn

        # Construct base URL and issues URL
        self.base_url = f'https://{journal_name}.pensoft.net'
        self.issues_url = issues_url if issues_url is not None else f'{self.base_url}/issues'

    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Scrapes and ingests articles from the Pensoft journal.
        """
        self.ingest_from_issues(issues_url=self.issues_url)

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL - already complete from scraping.

        Args:
            base: Dictionary containing the 'pdf_url' field

        Returns:
            The PDF URL
        """
        return base.get('pdf_url', '')

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL (article landing page).

        Args:
            base: Dictionary containing the 'url' field

        Returns:
            The human-readable URL (article page, not PDF)
        """
        return base.get('url', '')

    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse date string to ISO format (YYYY-MM-DD).

        Pensoft uses multiple date formats:
        - Issue pages: DD-MM-YYYY (e.g., "27-11-2025")
        - Article pages: DD Month YYYY (e.g., "27 November 2025")

        Args:
            date_str: Date string in various formats

        Returns:
            ISO formatted date or None
        """
        if not date_str:
            return None

        # Clean up the date string
        date_str = date_str.strip()

        # Try common date formats
        formats = [
            '%d-%m-%Y',      # 27-11-2025 (issue pages)
            '%d %B %Y',      # 27 November 2025 (article pages)
            '%d %b %Y',      # 27 Nov 2025
            '%B %d, %Y',     # November 27, 2025
            '%Y-%m-%d',      # 2025-11-27
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue

        if self.verbosity >= 2:
            print(f"    Warning: Could not parse date '{date_str}'")
        return date_str  # Return as-is if parsing fails

    def _extract_year(self, date_str: Optional[str]) -> Optional[str]:
        """
        Extract year from ISO date string.

        Args:
            date_str: ISO formatted date (YYYY-MM-DD)

        Returns:
            Year as string or None
        """
        if not date_str:
            return None

        # Try to extract YYYY from beginning of string
        match = re.search(r'^(\d{4})', date_str)
        if match:
            return match.group(1)

        return None

    def _extract_total_issues(self, soup: BeautifulSoup) -> Optional[int]:
        """
        Extract total number of issues from the sidebar.

        Looks for text pattern "from 1 to X" in the "Go to issue" sidebar.

        Args:
            soup: BeautifulSoup object of issues page

        Returns:
            Total number of issues, or None if not found
        """
        # Look for "from 1 to X" pattern in the page
        text = soup.get_text()
        match = re.search(r'from\s+1\s+to\s+(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Fallback: look for "X issues matching" pattern
        match = re.search(r'(\d+)\s+issues?\s+matching', text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def _extract_issue_ids_from_page(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract issue IDs and metadata from an issues listing page.

        Args:
            soup: BeautifulSoup object of issues listing page

        Returns:
            List of dicts with 'issue_id', 'url', 'number' keys
        """
        issues = []
        issues_dict = {}  # Track by issue_id to handle duplicates

        # Find all links and filter for issue links
        # Note: Issue links can be absolute (/issue/ID) or relative (issue/ID/)
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')

            # Check if this is an issue link (matches both absolute and relative)
            if not href or 'issue/' not in href:
                continue

            # Extract issue ID from URL (handle both /issue/ID and issue/ID/)
            match = re.search(r'issue/(\d+)', href)
            if not match:
                continue

            issue_id = match.group(1)
            full_url = urljoin(self.base_url, href)

            # Try to extract issue number from link text or alt attribute
            text = link.get_text(strip=True)

            # If no text (e.g., image link), check alt or title attributes
            if not text:
                img = link.find('img')
                if img:
                    text = img.get('alt', '') or img.get('title', '')

            # Look for "MycoKeys 126" or just "126" pattern
            number_match = re.search(r'(?:MycoKeys\s+)?(\d+)', text, re.IGNORECASE)
            issue_number = number_match.group(1) if number_match else None

            if self.verbosity >= 4:
                print(f"    Issue link: {href}")
                print(f"      Text: '{text}'")
                print(f"      Extracted number: {issue_number}")

            # Store or update issue info
            # Prefer entries with issue numbers over those without
            if issue_id not in issues_dict:
                issues_dict[issue_id] = {
                    'issue_id': issue_id,
                    'url': full_url,
                    'number': issue_number,
                    'text': text
                }
            elif issue_number and not issues_dict[issue_id]['number']:
                # Update with version that has an issue number
                issues_dict[issue_id]['number'] = issue_number
                issues_dict[issue_id]['text'] = text

        # Convert dict to list
        issues = list(issues_dict.values())

        return issues

    def _extract_articles_from_issue_page(
        self,
        soup: BeautifulSoup,
        issue_number: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract article metadata from an issue page.

        Args:
            soup: BeautifulSoup object of issue page
            issue_number: Issue number (optional)

        Returns:
            List of article metadata dicts
        """
        articles = []

        # Find all article containers (div.article in Pensoft)
        article_divs = soup.find_all('div', class_='article')

        if self.verbosity >= 3:
            print(f"    Found {len(article_divs)} article div(s)")

        for idx, article_div in enumerate(article_divs):
            # Find the article title link in div.articleHeadline
            title_link = None
            headline_div = article_div.find('div', class_='articleHeadline')
            if headline_div:
                for link in headline_div.find_all('a', href=True):
                    href = str(link.get('href', ''))
                    if '/article/' in href and '/download/' not in href:
                        title_link = link
                        break

            if not title_link:
                if self.verbosity >= 3:
                    print(f"      Article {idx + 1}: No title link found")
                continue

            href = str(title_link.get('href', ''))
            match = re.search(r'/article/(\d+)', href)
            if not match:
                if self.verbosity >= 3:
                    print(f"      Article {idx + 1}: Could not extract article ID from {href}")
                continue

            article_id = match.group(1)
            article_url = urljoin(self.base_url, href)

            # Get title text, preserving HTML entities and italics
            title = title_link.get_text(strip=True)

            if self.verbosity >= 4:
                print(f"      Article {idx + 1}: {title[:60]}...")

            # Initialize article dict
            article = {
                'url': article_url,
                'article_id': article_id,
                'title': title,
                'itemtype': 'article',
            }

            if issue_number:
                article['number'] = issue_number

            # Add ISSN if available
            if self.issn:
                article['issn'] = self.issn
            if self.eissn:
                article['eissn'] = self.eissn

            # Extract DOI from div.ArtDoi
            doi_div = article_div.find('div', class_='ArtDoi')
            if doi_div:
                doi_link = doi_div.find('a', href=True)
                if doi_link:
                    doi_text = doi_link.get_text(strip=True)
                    if doi_text and doi_text.startswith('10.'):
                        article['doi'] = doi_text

            # Extract publication date from DoiRow (format: DD-MM-YYYY)
            doi_row = article_div.find('div', class_='DoiRow')
            if doi_row:
                date_match = re.search(r'(\d{1,2}-\d{1,2}-\d{4})', doi_row.get_text())
                if date_match:
                    pub_date = self._parse_date(date_match.group(1))
                    if pub_date:
                        article['publication_date'] = pub_date
                        article['year'] = self._extract_year(pub_date)

            # Extract page range (look for pattern like "213-238")
            page_text = article_div.get_text()
            pages_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', page_text)
            if pages_match:
                article['pages'] = f"{pages_match.group(1)}–{pages_match.group(2)}"

            # Extract authors from div.authors_list_holder
            authors_div = article_div.find('div', class_='authors_list_holder')
            if authors_div:
                author_links = authors_div.find_all('a', class_='authors_list_holder')
                authors_list = []
                for a in author_links:
                    author_name = a.get_text(strip=True)
                    if author_name:
                        authors_list.append(author_name)
                if authors_list:
                    article['authors'] = ', '.join(authors_list)

            # Extract article type (e.g., "Research Article")
            type_match = re.search(
                r'(Research Article|Review Article|Short Communication|Data Paper)',
                article_div.get_text(),
                re.IGNORECASE
            )
            if type_match:
                article['article_type'] = type_match.group(1)

            # Find PDF download link in div.DownLink
            downlink_div = article_div.find('div', class_='DownLink')
            if downlink_div:
                for pdf_link in downlink_div.find_all('a', href=True):
                    pdf_href = str(pdf_link.get('href', ''))
                    if '/download/pdf/' in pdf_href:
                        pdf_url = urljoin(self.base_url, pdf_href)
                        article['pdf_url'] = pdf_url
                        if self.verbosity >= 4:
                            print(f"        PDF URL: {pdf_url}")
                        break

            if 'pdf_url' not in article and self.verbosity >= 3:
                print(f"      Article {idx + 1}: No PDF URL found")

            articles.append(article)

        return articles

    def _check_for_pagination(self, soup: BeautifulSoup) -> Optional[int]:
        """
        Check if there are pagination links and return the total number of pages.

        Args:
            soup: BeautifulSoup object

        Returns:
            Total number of pages, or None if no pagination found
        """
        # Look for pagination links (numbered page links)
        page_numbers = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if not href or ('?p=' not in href and '&p=' not in href):
                continue

            match = re.search(r'[&?]p=(\d+)', href)
            if match:
                page_numbers.append(int(match.group(1)))

        if not page_numbers:
            return None

        # Return max page number + 1 (since pages are 0-indexed)
        return max(page_numbers) + 1

    def ingest_from_issues(
        self,
        issues_url: str = None,
        max_issues: Optional[int] = None
    ) -> None:
        """
        Ingest articles from Pensoft journal issues.

        Strategy:
        1. Fetch issues page and determine total number of issues
        2. Paginate through issues listing pages
        3. For each issue, visit issue page
        4. Handle within-issue pagination if present
        5. Extract article metadata and PDF URLs
        6. Ingest articles

        Args:
            issues_url: URL of the issues page (default: self.issues_url)
            max_issues: Maximum number of issues to process (for testing)
        """
        if issues_url is None:
            issues_url = self.issues_url

        if self.verbosity >= 2:
            print(f"Fetching issues from: {issues_url}")

        # Fetch initial issues page
        issues_soup = self._fetch_page(issues_url)
        if not issues_soup:
            if self.verbosity >= 1:
                print("Failed to fetch issues page")
            return

        # Try to extract total number of issues
        total_issues = self._extract_total_issues(issues_soup)
        if total_issues and self.verbosity >= 2:
            print(f"Total issues: {total_issues}")

        # Collect all issue IDs by paginating through issues listing
        all_issues = []
        seen_issue_ids = set()
        page_num = 0

        # Keep fetching pages until we get no new issues
        while True:
            if page_num == 0:
                page_soup = issues_soup
            else:
                page_url = (f'{self.base_url}/browse_journal_issues.php?'
                          f'journal_name={self.journal_name}&lang=&'
                          f'journal_id={self.journal_id}&p={page_num}')

                if self.verbosity >= 3:
                    print(f"  Fetching issues page {page_num + 1}")

                page_soup = self._fetch_page(page_url)
                if not page_soup:
                    break

            # Extract issue IDs from this page
            page_issues = self._extract_issue_ids_from_page(page_soup)

            # Check if we found any new issues
            new_issues = [iss for iss in page_issues if iss['issue_id'] not in seen_issue_ids]

            if not new_issues:
                # No new issues on this page, we're done
                break

            # Add new issues
            for iss in new_issues:
                seen_issue_ids.add(iss['issue_id'])
                all_issues.append(iss)

            page_num += 1

            # Safety limit to prevent infinite loops
            if page_num > 50:
                if self.verbosity >= 1:
                    print("Warning: Reached pagination safety limit of 50 pages")
                break

        if self.verbosity >= 2:
            print(f"Found {len(all_issues)} issue(s)")

        # Process each issue
        for idx, issue_info in enumerate(all_issues):
            if max_issues and idx >= max_issues:
                break

            if self.verbosity >= 2:
                print(f"\n{'=' * 60}")
                print(f"Processing Issue {issue_info.get('number', issue_info['issue_id'])}")
                if self.verbosity >= 3:
                    print(f"  Issue ID: {issue_info['issue_id']}")
                    print(f"  Issue Number: {issue_info.get('number')}")
                    print(f"  URL: {issue_info['url']}")
                print(f"{'=' * 60}")

            # Collect all articles from this issue (handling pagination)
            issue_articles = []

            # Fetch first page of issue
            issue_soup = self._fetch_page(issue_info['url'])
            if not issue_soup:
                if self.verbosity >= 2:
                    print("  Failed to fetch issue page")
                continue

            # Check for within-issue pagination
            issue_pages = self._check_for_pagination(issue_soup)

            if issue_pages:
                if self.verbosity >= 3:
                    print(f"  Issue has {issue_pages} page(s) of articles")

                for page_num in range(issue_pages):
                    if page_num > 0:  # Already have first page
                        page_url = (f'{self.base_url}/browse_journal_issue_documents.php?'
                                  f'journal_name={self.journal_name}&issue_id={issue_info["issue_id"]}&'
                                  f'lang=&journal_id={self.journal_id}&p={page_num}')

                        page_soup = self._fetch_page(page_url)
                        if not page_soup:
                            continue
                    else:
                        page_soup = issue_soup

                    # Extract articles from this page
                    articles = self._extract_articles_from_issue_page(
                        page_soup,
                        issue_info.get('number')
                    )
                    issue_articles.extend(articles)
            else:
                # No pagination, extract from current page
                issue_articles = self._extract_articles_from_issue_page(
                    issue_soup,
                    issue_info.get('number')
                )

            if self.verbosity >= 2:
                print(f"  Found {len(issue_articles)} article(s)")

            if not issue_articles:
                continue

            # Filter to only articles with PDF URLs
            articles_with_pdfs = [a for a in issue_articles if 'pdf_url' in a]

            if self.verbosity >= 2:
                if len(articles_with_pdfs) < len(issue_articles):
                    print(f"  {len(articles_with_pdfs)} article(s) have PDF URLs")

            # Ingest articles
            if articles_with_pdfs:
                if self.verbosity >= 2:
                    print(f"  Ingesting {len(articles_with_pdfs)} article(s)")

                self._ingest_documents(
                    documents=articles_with_pdfs,
                    meta={
                        'source': 'pensoft',
                        'journal': self.journal_name,
                        'issue_id': issue_info['issue_id'],
                    },
                    bibtex_link=issue_info['url']
                )

        if self.verbosity >= 2:
            print(f"\n{'=' * 60}")
            print("Ingestion complete")
            print(f"{'=' * 60}")
