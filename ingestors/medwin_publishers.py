"""
MedwinPublishersIngestor for ingesting articles from Medwin Publishers journals.

This module provides the MedwinPublishersIngestor class for scraping and ingesting
articles from Medwin Publishers journal websites.
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime

from bs4 import BeautifulSoup

from .ingestor import Ingestor


class MedwinPublishersIngestor(Ingestor):
    """
    Ingestor for Medwin Publishers journals (medwinpublishers.com).

    This class scrapes Medwin Publishers websites to extract article metadata
    and PDFs. It handles archive pages, issue pages, and individual article pages.

    Journal information (OAJMMS example):
        ISSN: 2689-7822
        URL: https://www.medwinpublishers.com/OAJMMS/

    Navigation:
        - Start: https://www.medwinpublishers.com/OAJMMS/archive.php
        - Follow issue links (e.g., volume.php?volumeId=597&issueId=1860)
        - Follow article links (e.g., article-description.php?artId=12877)
        - Download PDF from "View PDF" button

    Metadata extracted:
        - title
        - authors (from issue page, not article page)
        - volume, number (issue)
        - publication_date
        - year
        - doi
        - abstract
        - keywords
        - article_type (Research Article, Mini Review, etc.)

    Note: Page numbers are not available (online-only journal format).
    """

    # Default configuration for OAJMMS (can be overridden)
    BASE_URL = 'https://www.medwinpublishers.com'
    ARCHIVES_URL = 'https://www.medwinpublishers.com/OAJMMS/archive.php'
    ISSN = '2689-7822'
    JOURNAL_NAME = 'Open Access Journal of Mycology & Mycological Sciences'

    def __init__(
        self,
        archives_url: Optional[str] = None,
        issn: Optional[str] = None,
        journal_name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MedwinPublishersIngestor.

        Args:
            archives_url: Archives URL to scrape from (default: ARCHIVES_URL)
            issn: Journal ISSN (default: ISSN)
            journal_name: Journal name (default: JOURNAL_NAME)
            **kwargs: Base class arguments (db, user_agent, robot_parser, etc.)
        """
        super().__init__(**kwargs)
        self.archives_url = archives_url if archives_url is not None else self.ARCHIVES_URL
        self.issn = issn if issn is not None else self.ISSN
        self.journal_name = journal_name if journal_name is not None else self.JOURNAL_NAME

    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Scrapes and ingests articles from the Medwin Publishers archives.
        """
        self.ingest_from_archives(archives_url=self.archives_url)

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

        Args:
            date_str: Date string in various formats

        Returns:
            ISO formatted date or None
        """
        if not date_str:
            return None

        # Clean up the date string
        date_str = date_str.strip()
        date_str = re.sub(r',\s*$', '', date_str)  # Remove trailing comma
        date_str = re.sub(r'\s+', ' ', date_str)   # Normalize whitespace

        # Try common date formats
        formats = [
            '%B %d, %Y',     # July 11, 2024
            '%d %B %Y',      # 11 July 2024
            '%b %d, %Y',     # Jul 11, 2024
            '%d %b %Y',      # 11 Jul 2024
            '%Y-%m-%d',      # 2024-07-11
            '%d/%m/%Y',      # 11/07/2024
            '%m/%d/%Y',      # 07/11/2024
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

    def _extract_issue_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract issue links from the archives page.

        Args:
            soup: BeautifulSoup object of archives page
            base_url: Base URL for resolving relative links (should be the archive page URL)

        Returns:
            List of dicts with 'url', 'volume', 'number', and 'year' keys
        """
        issue_links = []
        seen_urls = set()

        # Find all links to volume.php
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Match volume.php links
            if 'volume.php' in href:
                # Parse query parameters
                parsed = urlparse(href)
                params = parse_qs(parsed.query)

                if 'volumeId' in params and 'issueId' in params:
                    full_url = urljoin(base_url, href)

                    if full_url not in seen_urls:
                        seen_urls.add(full_url)

                        # Try to extract volume and issue from link text or surrounding context
                        text = link.get_text(strip=True)

                        # Pattern: "Volume X Issue Y" or "Volume X, Issue Y"
                        volume_match = re.search(r'Volume\s+(\d+)', text, re.IGNORECASE)
                        issue_match = re.search(r'Issue\s+(\d+)', text, re.IGNORECASE)

                        # Also try to find year from surrounding text
                        # Look at parent or nearby text
                        parent_text = ''
                        if link.parent:
                            parent_text = link.parent.get_text()
                        year_match = re.search(r'\((\d{4})\)', parent_text + text)

                        issue_links.append({
                            'url': full_url,
                            'volume': volume_match.group(1) if volume_match else 'unknown',
                            'number': issue_match.group(1) if issue_match else 'unknown',
                            'year': year_match.group(1) if year_match else None,
                            'text': text,
                            'volumeId': params['volumeId'][0],
                            'issueId': params['issueId'][0]
                        })

        return issue_links

    def _extract_article_links_from_issue(
        self,
        soup: BeautifulSoup,
        volume: str,
        number: str
    ) -> List[Dict[str, Any]]:
        """
        Extract article information from an issue page.

        Args:
            soup: BeautifulSoup object of issue page
            volume: Volume number
            number: Issue number

        Returns:
            List of dicts with article metadata including 'article_id', 'url', 'title', 'authors', 'doi'
        """
        articles = []

        # Find all article-description.php links
        for link in soup.find_all('a', href=re.compile(r'article-description\.php')):
            href = link['href']

            # Parse article ID
            parsed = urlparse(href)
            params = parse_qs(parsed.query)

            if 'artId' not in params:
                continue

            article_id = params['artId'][0]
            article_url = urljoin(self.BASE_URL, href)

            # Try to find the article block containing this link
            # Articles are typically in div or table structures
            article_block = link.find_parent(['div', 'tr', 'td'])

            if not article_block:
                # Fallback to just the link text
                article_block = link

            block_text = article_block.get_text()

            # Extract title (usually the link text itself)
            title = link.get_text(strip=True)

            # Try to find article type (Research Article, Mini Review, etc.)
            article_type = None
            type_match = re.search(r'(Research Article|Mini Review|Review Article|Case Report|Short Communication)',
                                  block_text, re.IGNORECASE)
            if type_match:
                article_type = type_match.group(1)

            # Try to extract authors from the block
            # Authors typically appear after the title, before DOI
            authors = None

            # Look for author patterns - names separated by commas or semicolons
            # Often marked with asterisks for corresponding authors
            # Pattern: Name1*, Name2, Name3*
            author_match = re.search(
                r'(?:Authors?[:\s]+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\*?(?:,|;)\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\*?)*)',
                block_text
            )
            if author_match:
                authors = author_match.group(1).strip()

            # Try to extract DOI
            doi = None
            doi_match = re.search(r'(?:DOI[:\s]+)?(10\.\d+/[^\s]+)', block_text, re.IGNORECASE)
            if doi_match:
                doi = doi_match.group(1).strip()

            articles.append({
                'article_id': article_id,
                'url': article_url,
                'title': title,
                'authors': authors,
                'doi': doi,
                'article_type': article_type,
                'volume': volume,
                'number': number,
            })

        return articles

    def _extract_article_metadata(
        self,
        soup: BeautifulSoup,
        base_article: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract detailed metadata from an article page.

        Args:
            soup: BeautifulSoup object of article page
            base_article: Base article dict with url, title, authors from issue page

        Returns:
            Enhanced article metadata dict
        """
        article = base_article.copy()

        # Add ISSN and journal name
        article['issn'] = self.issn
        article['journal'] = self.journal_name
        article['itemtype'] = 'article'

        # Extract abstract
        # Look for "Abstract" heading followed by text
        abstract_section = soup.find(['h2', 'h3', 'h4', 'strong', 'b'], string=re.compile(r'Abstract', re.IGNORECASE))
        if abstract_section:
            # Get the next sibling or parent's next sibling
            abstract_text = ''

            # Try to find the abstract content
            next_elem = abstract_section.find_next_sibling()
            if next_elem:
                abstract_text = next_elem.get_text(strip=True)
            else:
                # Try parent's next sibling
                parent = abstract_section.parent
                if parent:
                    next_elem = parent.find_next_sibling()
                    if next_elem:
                        abstract_text = next_elem.get_text(strip=True)

            if abstract_text:
                article['abstract'] = abstract_text

        # If abstract not found, try alternative approach: look for paragraph after "Abstract"
        if 'abstract' not in article:
            page_text = soup.get_text()
            abstract_match = re.search(
                r'Abstract[:\s]+(.+?)(?=Keywords?[:\s]|Introduction|$)',
                page_text,
                re.IGNORECASE | re.DOTALL
            )
            if abstract_match:
                abstract_text = abstract_match.group(1).strip()
                # Clean up whitespace
                abstract_text = re.sub(r'\s+', ' ', abstract_text)
                if len(abstract_text) > 20:  # Only use if substantial
                    article['abstract'] = abstract_text

        # Extract keywords
        keywords_section = soup.find(['h2', 'h3', 'h4', 'strong', 'b'], string=re.compile(r'Keywords?', re.IGNORECASE))
        if keywords_section:
            keywords_text = ''

            next_elem = keywords_section.find_next_sibling()
            if next_elem:
                keywords_text = next_elem.get_text(strip=True)
            else:
                parent = keywords_section.parent
                if parent:
                    next_elem = parent.find_next_sibling()
                    if next_elem:
                        keywords_text = next_elem.get_text(strip=True)

            if keywords_text:
                article['keywords'] = keywords_text

        # If keywords not found, try regex on page text
        if 'keywords' not in article:
            page_text = soup.get_text()
            keywords_match = re.search(
                r'Keywords?[:\s]+(.+?)(?=\n\n|Introduction|Abstract|$)',
                page_text,
                re.IGNORECASE | re.DOTALL
            )
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                keywords_text = re.sub(r'\s+', ' ', keywords_text)
                if keywords_text:
                    article['keywords'] = keywords_text

        # Extract publication date
        # Look for patterns like "Published: July 11, 2024" or just date text
        date_match = re.search(
            r'(?:Published|Publication Date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            soup.get_text(),
            re.IGNORECASE
        )
        if date_match:
            pub_date = self._parse_date(date_match.group(1))
            if pub_date:
                article['publication_date'] = pub_date
                article['year'] = self._extract_year(pub_date)

        # Extract DOI if not already present
        if 'doi' not in article or not article['doi']:
            doi_match = re.search(r'(?:DOI[:\s]+)?(10\.\d+/[^\s<]+)', soup.get_text(), re.IGNORECASE)
            if doi_match:
                article['doi'] = doi_match.group(1).strip()

        # Extract volume and issue if not already present
        if 'volume' not in article or article['volume'] == 'unknown':
            volume_match = re.search(r'Volume[:\s]+(\d+)', soup.get_text(), re.IGNORECASE)
            if volume_match:
                article['volume'] = volume_match.group(1)

        if 'number' not in article or article['number'] == 'unknown':
            issue_match = re.search(r'Issue[:\s]+(\d+)', soup.get_text(), re.IGNORECASE)
            if issue_match:
                article['number'] = issue_match.group(1)

        # Extract PDF URL from "View PDF" button or link
        pdf_link = soup.find('a', string=re.compile(r'View PDF|Download PDF|PDF', re.IGNORECASE))
        if not pdf_link:
            # Try finding by href pattern
            pdf_link = soup.find('a', href=re.compile(r'\.pdf$', re.IGNORECASE))

        if pdf_link and pdf_link.get('href'):
            pdf_url = urljoin(self.BASE_URL, pdf_link['href'])
            article['pdf_url'] = pdf_url

        return article

    def ingest_from_archives(
        self,
        archives_url: str = None,
        max_issues: Optional[int] = None
    ) -> None:
        """
        Ingest articles from Medwin Publishers archives.

        Strategy:
        1. Extract issue links from archive page
        2. For each issue, extract article links and basic metadata
        3. For each article, fetch detailed metadata and PDF URL
        4. Ingest articles with all metadata

        Args:
            archives_url: URL of the archives page (default: self.archives_url)
            max_issues: Maximum number of issues to process (for testing)
        """
        if archives_url is None:
            archives_url = self.archives_url

        if self.verbosity >= 2:
            print(f"Fetching archives from: {archives_url}")

        # Fetch archives page
        archives_soup = self._fetch_page(archives_url)
        if not archives_soup:
            if self.verbosity >= 1:
                print("Failed to fetch archives page")
            return

        # Extract issue links
        issue_links = self._extract_issue_links(archives_soup, archives_url)
        if self.verbosity >= 2:
            print(f"Found {len(issue_links)} issue(s)")

        # Process each issue
        for idx, issue_info in enumerate(issue_links):
            if max_issues and idx >= max_issues:
                break

            if self.verbosity >= 2:
                print(f"\n{'=' * 60}")
                print(f"Processing Volume {issue_info['volume']} Issue {issue_info['number']}")
                if issue_info['year']:
                    print(f"Year: {issue_info['year']}")
                print(f"{'=' * 60}")

            # Fetch issue page
            issue_soup = self._fetch_page(issue_info['url'])
            if not issue_soup:
                if self.verbosity >= 2:
                    print(f"  Failed to fetch issue page")
                continue

            # Extract article links from issue page
            articles = self._extract_article_links_from_issue(
                issue_soup,
                issue_info['volume'],
                issue_info['number']
            )

            if self.verbosity >= 2:
                print(f"  Found {len(articles)} article(s)")

            if not articles:
                continue

            # Enhance each article with detailed metadata from article page
            enhanced_articles = []
            for article in articles:
                if self.verbosity >= 3:
                    print(f"    Fetching metadata for: {article['title'][:60]}...")

                # Fetch article page
                article_soup = self._fetch_page(article['url'])
                if not article_soup:
                    if self.verbosity >= 2:
                        print(f"      Failed to fetch article page")
                    continue

                # Extract detailed metadata
                enhanced_article = self._extract_article_metadata(article_soup, article)

                # Only ingest if we have a PDF URL
                if 'pdf_url' in enhanced_article:
                    enhanced_articles.append(enhanced_article)
                    if self.verbosity >= 3:
                        print(f"      ✓ Metadata extracted")
                else:
                    if self.verbosity >= 2:
                        print(f"      ✗ No PDF URL found, skipping")

            # Ingest articles
            if enhanced_articles:
                if self.verbosity >= 2:
                    print(f"  Ingesting {len(enhanced_articles)} article(s)")

                self._ingest_documents(
                    documents=enhanced_articles,
                    meta={
                        'source': 'medwin-publishers',
                        'journal': self.journal_name,
                        'volume': issue_info['volume'],
                        'number': issue_info['number'],
                    },
                    bibtex_link=issue_info['url']
                )

        if self.verbosity >= 2:
            print(f"\n{'=' * 60}")
            print("Ingestion complete")
            print(f"{'=' * 60}")
