"""
IngentaConnect-specific ingestor implementation.

This module provides the IngentaIngestor class for ingesting data from
IngentaConnect RSS feeds, index pages, and local BibTeX files.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .ingestor import Ingestor


class IngentaIngestor(Ingestor):
    """
    Ingestor specialized for IngentaConnect data.

    Handles Ingenta-specific URL formatting and BibTeX content transformations.
    Supports multiple ingestion modes:
    - RSS: Ingest from RSS feeds
    - Index: Ingest from index pages (volume/issue navigation)
    - Local: Ingest from local BibTeX files
    """

    # Base URL for IngentaConnect
    BASE_URL = 'https://www.ingentaconnect.com'

    def __init__(
        self,
        rss_url: Optional[str] = None,
        index_url: Optional[str] = None,
        bibtex_url_template: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the IngentaIngestor.

        Args:
            rss_url: RSS feed URL to ingest from
            index_url: Index page URL to ingest from (e.g., ?format=index)
            bibtex_url_template: Template for BibTeX URLs
            **kwargs: Base class arguments (db, user_agent, robot_parser, etc.)
        """
        super().__init__(**kwargs)
        self.rss_url = rss_url
        self.index_url = index_url
        self.bibtex_url_template = bibtex_url_template

    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Ingests data from the RSS feed or index URL specified in the constructor.
        """
        if self.index_url is not None:
            self.ingest_from_index(self.index_url)
        elif self.rss_url is not None:
            self.ingest_from_rss(self.rss_url, self.bibtex_url_template)
        else:
            raise ValueError("Either rss_url or index_url must be provided for IngentaIngestor")

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL for Ingenta with crawler parameter.

        Args:
            base_url: The base URL from the BibTeX entry

        Returns:
            URL with ?crawler=true appended
        """
        return f"{base['url']}?crawler=true"

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL for Ingenta.

        Args:
            base_url: The base URL from the BibTeX entry
        """
        return base['url']

    def transform_bibtex_content(self, content: bytes) -> bytes:
        """
        Fix Ingenta-specific BibTeX syntax issues.

        Ingenta BibTeX files have syntax problems that need correction:
        - Missing commas between url and parent_itemid fields
        - Missing commas before 'parent' field
        - Embedded newlines that break parsing

        Args:
            content: Raw BibTeX content

        Returns:
            Corrected BibTeX content
        """
        return (
            content
            # Fix url field running into parent_itemid field
            .replace(b"\"\nparent_itemid", b"\",\nparent_itemid")
            .replace(b"\"\\nparent_itemid", b"\",\\nparent_itemid")
            # Fix other parent field issues
            .replace(b"\"\\nparent", b"\",\\nparent")
            # Remove remaining embedded newlines
            .replace(b"\\n", b"")
        )

    def ingest_from_local_bibtex(
        self,
        root: Path,
        bibtex_file_pattern: str = 'format=bib',
        url_prefix: str = 'https://www.ingentaconnect.com/'
    ) -> None:
        """
        Ingest from a local directory containing Ingenta BibTeX files.

        Args:
            root: Root directory to search for BibTeX files
            bibtex_file_pattern: Filename pattern to match BibTeX files
            url_prefix: URL prefix for Ingenta (default: https://www.ingentaconnect.com/)
        """
        super().ingest_from_local_bibtex(
            root=root,
            bibtex_file_pattern=bibtex_file_pattern,
            url_prefix=url_prefix
        )

    def _extract_volume_issue_links(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract volume and issue links from the index page.

        The index page has Volume headers as <li><strong>Volume X</strong></li>
        followed by sibling <li> elements containing issue links.

        Args:
            soup: BeautifulSoup object of index page

        Returns:
            List of dicts with 'url', 'volume', 'issue', and 'text' keys
        """
        issue_links = []

        # Find all strong tags that contain "Volume"
        all_strong = soup.find_all('strong')
        volume_headers = []
        for strong in all_strong:
            text = strong.get_text(strip=True)
            if re.match(r'Volume\s+\d+', text, re.IGNORECASE):
                volume_headers.append(strong)

        for volume_header in volume_headers:
            # Extract volume number from header text
            volume_match = re.search(r'Volume\s+(\d+)', volume_header.get_text(), re.IGNORECASE)
            if not volume_match:
                continue

            volume = volume_match.group(1)

            # Find the parent <li> of this volume header
            parent_li = volume_header.find_parent('li')
            if not parent_li:
                continue

            # Iterate through next sibling <li> elements until we hit another volume
            for sibling in parent_li.next_siblings:
                # Skip non-tag siblings (text nodes, etc.)
                if not hasattr(sibling, 'name') or sibling.name != 'li':
                    continue

                # Stop if we hit another volume header
                if sibling.find('strong'):
                    break

                # Find links in this issue <li>
                links = sibling.find_all('a', href=re.compile(r'/content/'))

                for link in links:
                    href = link.get('href', '')
                    text = link.get_text(strip=True)

                    # Clean session IDs from href
                    href_clean = href.split(';')[0]

                    # Extract issue number from link text
                    issue_match = re.search(r'Number\s+(\d+)', text, re.IGNORECASE)
                    if not issue_match:
                        # Try to extract from href (e.g., /content/wfbi/pimj/2025/00000054/00000001)
                        href_match = re.search(r'/(\d{8})$', href_clean)
                        if href_match:
                            issue_num = str(int(href_match.group(1)))  # Remove leading zeros
                        else:
                            continue
                    else:
                        issue_num = issue_match.group(1)

                    full_url = urljoin(self.BASE_URL, href_clean)

                    issue_links.append({
                        'url': full_url,
                        'volume': volume,
                        'issue': issue_num,
                        'text': text
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

        Only extracts open access articles (marked with OA icon).

        Args:
            soup: BeautifulSoup object of issue page
            volume: Volume number
            issue: Issue number

        Returns:
            List of article metadata dicts
        """
        articles = []

        # Find all open access icons
        oa_icons = soup.find_all('img', src=re.compile(r'icon_o_a_square\.gif'))

        for icon in oa_icons:
            # Find the article title link (next <a> tag after the icon)
            parent = icon.find_parent(['div', 'li', 'tr', 'td'])
            if not parent:
                continue

            # Find article link
            title_link = parent.find('a', href=re.compile(r'/contentone/'))
            if not title_link:
                continue

            title = title_link.get_text(strip=True)
            article_url = urljoin(self.BASE_URL, title_link.get('href', ''))

            # Extract page numbers
            pages = None
            pages_elem = parent.find(string=re.compile(r'pp\.\s*\d+'))
            if pages_elem:
                pages_match = re.search(r'pp\.\s*(\d+[-–]\d+)', pages_elem)
                if pages_match:
                    pages = pages_match.group(1)

            # Extract authors
            authors = None
            authors_elem = parent.find(string=re.compile(r'Authors?:', re.IGNORECASE))
            if authors_elem:
                # Authors text is after "Authors:"
                authors_match = re.search(r'Authors?:\s*(.+)', authors_elem, re.IGNORECASE)
                if authors_match:
                    authors = authors_match.group(1).strip()

            # Build article dict with basic metadata
            article = {
                'url': article_url,
                'title': title,
                'volume': volume,
                'number': issue,
                'itemtype': 'article',
            }

            if pages:
                article['pages'] = pages

            if authors:
                article['authors'] = authors

            # Fetch detailed metadata from article page
            if self.verbosity >= 3:
                print(f"      Fetching metadata for: {title[:50]}...")

            detailed_metadata = self._extract_article_metadata(article_url)
            article.update(detailed_metadata)

            articles.append(article)

        return articles

    def _extract_article_metadata(self, article_url: str) -> Dict[str, Optional[str]]:
        """
        Fetch the article page and extract detailed metadata.

        Args:
            article_url: URL to the article page

        Returns:
            Dict with 'abstract', 'keywords', 'document_type', 'publication_date', 'doi', etc.
        """
        result: Dict[str, Optional[str]] = {
            'abstract': None,
            'keywords': None,
            'document_type': None,
            'publication_date': None,
            'doi': None,
        }

        # Fetch the article page
        soup = self._fetch_page(article_url)
        if not soup:
            return result

        # Extract abstract
        # Look for section with "Abstract" or anchor #Abst
        abstract_section = soup.find('div', id='Abst')
        if not abstract_section:
            # Try finding by heading
            abstract_headers = soup.find_all(['h2', 'h3', 'h4'], string=re.compile(r'Abstract', re.IGNORECASE))
            for header in abstract_headers:
                # Get the next element(s) containing the abstract text
                abstract_section = header.find_next_sibling(['p', 'div'])
                if abstract_section:
                    break

        if abstract_section:
            # Extract text, but exclude "No Abstract" messages
            abstract_text = abstract_section.get_text(strip=True)
            if abstract_text and not re.match(r'No\s+Abstract', abstract_text, re.IGNORECASE):
                result['abstract'] = abstract_text

        # Extract keywords
        # Keywords are often in meta tags or as linked elements
        keywords_list = []

        # Try finding keyword links
        keyword_links = soup.find_all('a', href=re.compile(r'keyword'))
        for link in keyword_links:
            keyword = link.get_text(strip=True)
            if keyword and keyword not in keywords_list:
                keywords_list.append(keyword)

        if keywords_list:
            result['keywords'] = ', '.join(keywords_list)

        # Extract document type
        doc_type_elem = soup.find(string=re.compile(r'Document Type', re.IGNORECASE))
        if doc_type_elem:
            # Look for the value after this label
            parent = doc_type_elem.find_parent(['div', 'span', 'p', 'td'])
            if parent:
                # Try next sibling or text after the label
                next_elem = parent.find_next_sibling()
                if next_elem:
                    result['document_type'] = next_elem.get_text(strip=True)
                else:
                    # Try extracting from same element
                    text = parent.get_text()
                    match = re.search(r'Document Type[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
                    if match:
                        result['document_type'] = match.group(1).strip()

        # Extract publication date
        pub_date_elem = soup.find(string=re.compile(r'Publication date', re.IGNORECASE))
        if pub_date_elem:
            parent = pub_date_elem.find_parent(['div', 'span', 'p', 'td'])
            if parent:
                text = parent.get_text()
                # Look for date pattern (e.g., "01 June 2025")
                date_match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', text)
                if date_match:
                    result['publication_date'] = date_match.group(1)

        # Extract DOI
        doi_link = soup.find('a', href=re.compile(r'doi\.org'))
        if doi_link:
            doi_url = doi_link.get('href', '')
            doi_match = re.search(r'doi\.org/(.+?)(?:\?|$)', doi_url)
            if doi_match:
                result['doi'] = doi_match.group(1)

        return result

    def ingest_from_index(
        self,
        index_url: str,
        max_issues: Optional[int] = None
    ) -> None:
        """
        Ingest articles from IngentaConnect index page.

        Args:
            index_url: URL of the index page (e.g., with ?format=index)
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
                    'source': 'ingenta',
                    'volume': issue_info['volume'],
                    'issue': issue_info['issue'],
                },
                bibtex_link=issue_info['url']
            )
