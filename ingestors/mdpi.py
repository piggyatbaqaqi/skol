"""
MDPI-specific ingestor implementation.

This module provides the MdpiIngestor class for ingesting data from
MDPI journals (mdpi.com) via RSS feeds and index pages.
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .ingestor import Ingestor


class MdpiIngestor(Ingestor):
    """
    Ingestor specialized for MDPI journals.

    Handles MDPI-specific URL formatting and metadata extraction.
    Supports multiple ingestion modes:
    - RSS: Ingest from RSS feeds
    - Index: Ingest from index pages (volume/issue navigation)
    """

    # Base URL for MDPI
    BASE_URL = 'https://www.mdpi.com'

    def __init__(
        self,
        rss_url: Optional[str] = None,
        index_url: Optional[str] = None,
        journal_code: Optional[str] = None,
        issn: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MdpiIngestor.

        Args:
            rss_url: RSS feed URL to ingest from (e.g., https://www.mdpi.com/rss/journal/jof)
            index_url: Index page URL to ingest from (e.g., https://www.mdpi.com/journal/jof)
            journal_code: Journal code (e.g., 'jof' for Journal of Fungi)
            issn: ISSN for the journal (e.g., '2309-608X')
            **kwargs: Base class arguments (db, user_agent, robot_parser, etc.)
        """
        super().__init__(**kwargs)
        self.rss_url = rss_url
        self.index_url = index_url
        self.journal_code = journal_code
        self.issn = issn

    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Ingests data from the RSS feed or index URL specified in the constructor.
        """
        if self.index_url is not None:
            self.ingest_from_index(self.index_url)
        elif self.rss_url is not None:
            # feedparser normalizes all RSS formats to feed.feed
            self.ingest_from_rss(self.rss_url)
        else:
            raise ValueError("Either rss_url or index_url must be provided for MdpiIngestor")

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL for MDPI by appending /pdf.

        Args:
            base: Dictionary containing the 'url' field

        Returns:
            URL with /pdf appended
        """
        url = base['url']
        # Remove trailing slash if present
        url = url.rstrip('/')
        return f"{url}/pdf"

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL for MDPI.

        Args:
            base: Dictionary containing the 'url' field

        Returns:
            The URL as-is
        """
        return base['url']

    def _extract_volume_issue_links(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract volume and issue links from the index page.

        The index page has a sidebar on the left with "Current Issue" link,
        followed by volume links. Volume pages have links to issues.

        Args:
            soup: BeautifulSoup object of index page

        Returns:
            List of dicts with 'url', 'volume', 'issue' keys
        """
        issue_links = []

        # Find the sidebar (typically has class like 'sidebar', 'left-sidebar', etc.)
        # Look for links to volume pages
        volume_links = []

        # Try to find volume links - they typically look like /journal/jof/volume/11
        all_links = soup.find_all('a', href=re.compile(r'/journal/[^/]+/volume/\d+'))

        for link in all_links:
            href = link.get('href', '')
            if href and href not in [v['url'] for v in volume_links]:
                full_url = urljoin(self.BASE_URL, href)
                # Extract volume number
                volume_match = re.search(r'/volume/(\d+)', href)
                if volume_match:
                    volume_links.append({
                        'url': full_url,
                        'volume': volume_match.group(1)
                    })

        if self.verbosity >= 3:
            print(f"    Found {len(volume_links)} volume page(s)")

        # Now fetch each volume page to get issue links
        for vol_info in volume_links:
            if self.verbosity >= 3:
                print(f"    Fetching volume {vol_info['volume']} page...")

            vol_soup = self._fetch_page(vol_info['url'])
            if not vol_soup:
                continue

            # Find issue links - they look like /2309-608X/11/1
            issue_link_elements = vol_soup.find_all('a', href=re.compile(r'/\d{4}-\d{3,4}X?/\d+/\d+'))

            for link in issue_link_elements:
                href = link.get('href', '')
                if not href:
                    continue

                full_url = urljoin(self.BASE_URL, href)

                # Extract volume and issue from URL pattern like /2309-608X/11/1
                issue_match = re.search(r'/(\d{4}-\d{3,4}X?)/(\d+)/(\d+)', href)
                if issue_match:
                    volume = issue_match.group(2)
                    issue_num = issue_match.group(3)

                    # Avoid duplicates
                    if not any(i['url'] == full_url for i in issue_links):
                        issue_links.append({
                            'url': full_url,
                            'volume': volume,
                            'issue': issue_num,
                        })

        return issue_links

    def _extract_articles_from_issue(
        self,
        soup: BeautifulSoup,
        volume: str,
        issue: str
    ) -> List[Dict[str, Any]]:
        """
        Extract article metadata from an issue page.

        Only extracts open access articles (marked with "Open Access" label).

        Args:
            soup: BeautifulSoup object of issue page
            volume: Volume number
            issue: Issue number

        Returns:
            List of article metadata dicts
        """
        articles = []

        # Find all "Open Access" labels - typically in a span or div with specific class
        oa_markers = soup.find_all(string=re.compile(r'Open\s+Access', re.IGNORECASE))

        for marker in oa_markers:
            # Find the article container (parent div/article element)
            article_container = marker.find_parent(['article', 'div', 'li'])
            if not article_container:
                continue

            # Find article title and URL
            title_link = article_container.find('a', href=re.compile(r'/\d{4}-\d{3,4}X?/\d+/\d+/\d+'))
            if not title_link:
                continue

            title = title_link.get_text(strip=True)
            href = title_link.get('href', '')
            article_url = urljoin(self.BASE_URL, href)

            # Extract authors - typically in a div/span with class like 'authors' or 'by'
            authors = None
            authors_elem = article_container.find(['div', 'span', 'p'], class_=re.compile(r'author|by', re.IGNORECASE))
            if authors_elem:
                authors = authors_elem.get_text(strip=True)
            else:
                # Try to find by text pattern (often starts with "by " or contains " and ")
                text_content = article_container.get_text()
                # Look for author line (often has "by" or multiple names with commas)
                lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                for line in lines:
                    if re.search(r'^by\s+', line, re.IGNORECASE) or (', ' in line and ' and ' in line.lower()):
                        authors = re.sub(r'^by\s+', '', line, flags=re.IGNORECASE).strip()
                        break

            # Extract DOI
            doi = None
            doi_link = article_container.find('a', href=re.compile(r'doi\.org'))
            if doi_link:
                doi_url = doi_link.get('href', '')
                doi_match = re.search(r'doi\.org/(.+?)(?:\?|$)', doi_url)
                if doi_match:
                    doi = doi_match.group(1)

            # Extract abstract
            abstract = None
            abstract_section = article_container.find(string=re.compile(r'Abstract:', re.IGNORECASE))
            if abstract_section:
                # Get the parent and find the abstract text
                abstract_parent = abstract_section.find_parent(['div', 'p'])
                if abstract_parent:
                    abstract_text = abstract_parent.get_text(strip=True)
                    # Remove "Abstract:" prefix
                    abstract = re.sub(r'^Abstract:\s*', '', abstract_text, flags=re.IGNORECASE)

            # Extract section - "(This article belongs to section ...)"
            section = None
            section_match = article_container.find(string=re.compile(r'\(This article belongs to [Ss]ection', re.IGNORECASE))
            if section_match:
                section_text = section_match.strip()
                # Extract section name from parenthetical
                match = re.search(r'\(This article belongs to [Ss]ection\s+(.+?)\)', section_text, re.IGNORECASE)
                if match:
                    section = match.group(1).strip()

            # Build article dict
            article = {
                'url': article_url,
                'title': title,
                'volume': volume,
                'number': issue,
                'itemtype': 'article',
            }

            if authors:
                article['authors'] = authors

            if doi:
                article['doi'] = doi

            if abstract:
                article['abstract'] = abstract

            if section:
                article['section'] = section

            articles.append(article)

        return articles

    def ingest_from_index(
        self,
        index_url: str,
        max_issues: Optional[int] = None
    ) -> None:
        """
        Ingest articles from MDPI journal index page.

        Args:
            index_url: URL of the index page (e.g., https://www.mdpi.com/journal/jof)
            max_issues: Maximum number of issues to process (for testing)
        """
        if self.verbosity >= 2:
            print(f"Fetching index from: {index_url}")

        # Fetch index page
        index_soup = self._fetch_page(index_url)
        if not index_soup:
            if self.verbosity >= 1:
                print("Failed to fetch index page")
            return

        # Extract volume/issue links
        issue_links = self._extract_volume_issue_links(index_soup)
        if self.verbosity >= 2:
            print(f"Found {len(issue_links)} issue(s)")

        # Process each issue
        for idx, issue_info in enumerate(issue_links):
            if max_issues and idx >= max_issues:
                break

            if self.verbosity >= 2:
                print(f"\n{'=' * 60}")
                print(f"Processing Volume {issue_info['volume']}, Issue {issue_info['issue']}")
                print(f"{'=' * 60}")

            # Fetch issue page
            issue_soup = self._fetch_page(issue_info['url'])
            if not issue_soup:
                continue

            # Extract articles from issue page (only OA articles)
            articles = self._extract_articles_from_issue(
                issue_soup,
                issue_info['volume'],
                issue_info['issue']
            )

            if self.verbosity >= 2:
                print(f"  Extracted {len(articles)} open access article(s)")

            if not articles:
                continue

            # Ingest articles
            self._ingest_documents(
                documents=articles,
                meta={
                    'source': 'mdpi',
                    'volume': issue_info['volume'],
                    'issue': issue_info['issue'],
                },
                bibtex_link=issue_info['url']
            )
