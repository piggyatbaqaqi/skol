"""
TaylorFrancisIngestor for ingesting articles from Taylor & Francis journals.

This module provides the TaylorFrancisIngestor class for scraping and ingesting
articles from Taylor & Francis journals at tandfonline.com.
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .ingestor import Ingestor


class TaylorFrancisIngestor(Ingestor):
    """
    Ingestor for Taylor & Francis journals.

    This class scrapes Taylor & Francis journal websites to extract
    article metadata and PDFs. It can be used for any Taylor & Francis
    journal hosted at tandfonline.com.

    Example journals:
        - Mycology: https://www.tandfonline.com/loi/tmyc20
        - Other mycology journals hosted on the same platform

    Navigation:
        - Start: Journal archive page (e.g., /loi/tmyc20)
        - Volume headers contain Issue headers
        - Issue pages have Article blocks

    Metadata extracted:
        - title
        - authors
        - pages
        - publication_date (published online)
        - doi
        - abstract (from abstract page)
        - keywords (from abstract page)
    """

    # Base URL for Taylor & Francis
    BASE_URL = 'https://www.tandfonline.com'

    def __init__(
        self,
        archives_url: Optional[str] = None,
        journal_name: Optional[str] = None,
        issn: Optional[str] = None,
        eissn: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the TaylorFrancisIngestor.

        Args:
            archives_url: Archives URL to scrape from (e.g., https://www.tandfonline.com/loi/tmyc20)
            journal_name: Journal name for metadata (e.g., 'Mycology')
            issn: Print ISSN (e.g., '2150-1203')
            eissn: Electronic ISSN (e.g., '2150-1211')
            **kwargs: Base class arguments (db, user_agent, robot_parser, etc.)
        """
        super().__init__(**kwargs)
        self.archives_url = archives_url
        self.journal_name = journal_name
        self.issn = issn
        self.eissn = eissn

    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Scrapes and ingests articles from the Taylor & Francis journal archives.
        """
        if self.archives_url is None:
            raise ValueError(
                "archives_url must be provided for TaylorFrancisIngestor"
            )
        self.ingest_from_archives(archives_url=self.archives_url)

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL from the article URL.

        Converts /doi/full/ to /doi/pdf/ and adds ?download=true.

        Args:
            base: Dictionary containing the 'url' field

        Returns:
            The PDF URL
        """
        url = base.get('url', '')
        # Convert /doi/full/ to /doi/pdf/ and add download parameter
        pdf_url = url.replace('/doi/full/', '/doi/pdf/')
        if '?' not in pdf_url:
            pdf_url += '?download=true'
        return pdf_url

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL.

        Args:
            base: Dictionary containing the 'url' field

        Returns:
            The human-readable URL
        """
        return base.get('url', '')

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """
        Extract DOI from Taylor & Francis URL.

        Args:
            url: Article URL (e.g., https://www.tandfonline.com/doi/full/10.1080/21501203.2024.2316066)

        Returns:
            DOI string (e.g., 10.1080/21501203.2024.2316066) or None
        """
        # DOI is everything after /doi/full/ or /doi/pdf/
        match = re.search(r'/doi/(?:full|pdf)/(.+?)(?:\?|$)', url)
        if match:
            return match.group(1)
        return None

    def _extract_volume_issue_links(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract volume and issue links from the archives page.

        The archives page has Volume headers, under which are Issue headers with links.

        Args:
            soup: BeautifulSoup object of archives page

        Returns:
            List of dicts with 'url', 'volume', and 'issue' keys
        """
        issue_links = []

        # Find all volume sections
        # Look for headers like "Volume X (Year)"
        volume_headers = soup.find_all(['h2', 'h3', 'h4'], string=re.compile(r'Volume\s+\d+', re.IGNORECASE))

        for volume_header in volume_headers:
            # Extract volume number from header text
            volume_match = re.search(r'Volume\s+(\d+)', volume_header.get_text(), re.IGNORECASE)
            if not volume_match:
                continue

            volume = volume_match.group(1)

            # Find all issue links after this volume header
            # Look for siblings or descendants containing issue links
            # Issues are typically in a list or div after the volume header
            container = volume_header.find_next_sibling()
            if not container:
                # Try parent's next sibling
                parent = volume_header.parent
                if parent:
                    container = parent.find_next_sibling()

            if not container:
                continue

            # Find all issue links in the container
            # Look for links with patterns like "Issue X" or links to /toc/ paths
            issue_link_elements = container.find_all('a', href=re.compile(r'/toc/'))

            for link in issue_link_elements:
                href = link.get('href', '')
                text = link.get_text(strip=True)

                # Extract issue number from link text
                issue_match = re.search(r'Issue\s+(\d+)', text, re.IGNORECASE)
                if not issue_match:
                    # Try to extract from href (e.g., /toc/tmyc20/15/1)
                    href_match = re.search(r'/toc/[^/]+/\d+/(\d+)', href)
                    if href_match:
                        issue_num = href_match.group(1)
                    else:
                        continue
                else:
                    issue_num = issue_match.group(1)

                full_url = urljoin(self.BASE_URL, href)

                issue_links.append({
                    'url': full_url,
                    'volume': volume,
                    'issue': issue_num,
                    'text': text
                })

        return issue_links

    def _extract_abstract_and_keywords(self, article_url: str) -> Dict[str, Optional[str]]:
        """
        Fetch the abstract page and extract abstract and keywords.

        Args:
            article_url: URL to the article page

        Returns:
            Dict with 'abstract' and 'keywords' keys
        """
        result = {'abstract': None, 'keywords': None}

        # Fetch the article page
        soup = self._fetch_page(article_url)
        if not soup:
            return result

        # Find abstract section
        # Look for a section or div with heading "ABSTRACT"
        abstract_section = None
        abstract_headers = soup.find_all(['h2', 'h3', 'h4', 'div'], string=re.compile(r'ABSTRACT', re.IGNORECASE))

        for header in abstract_headers:
            # Get the next element(s) containing the abstract text
            # Abstract is typically in a <p> or <div> after the header
            if header.name in ['h2', 'h3', 'h4']:
                # Header element - get next sibling or parent's next sibling
                abstract_section = header.find_next_sibling(['p', 'div'])
            else:
                # Div element - abstract might be inside
                abstract_section = header

            if abstract_section:
                break

        if abstract_section:
            # Extract text until we hit KEYWORDS section
            abstract_text = []
            current = abstract_section

            while current:
                text = current.get_text(strip=True)

                # Stop if we hit KEYWORDS
                if re.search(r'KEYWORDS?:', text, re.IGNORECASE):
                    # Extract keywords from this element
                    keywords_match = re.search(r'KEYWORDS?:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                    if keywords_match:
                        result['keywords'] = keywords_match.group(1).strip()
                    break

                # Add text if it's not empty and not a header
                if text and not re.match(r'^ABSTRACT$', text, re.IGNORECASE):
                    abstract_text.append(text)

                # Move to next sibling
                current = current.find_next_sibling(['p', 'div'])

            if abstract_text:
                result['abstract'] = ' '.join(abstract_text)

        # If we didn't find keywords yet, search for them separately
        if not result['keywords']:
            keywords_headers = soup.find_all(['h2', 'h3', 'h4', 'div', 'p', 'span'],
                                            string=re.compile(r'KEYWORDS?:', re.IGNORECASE))

            for header in keywords_headers:
                text = header.get_text(strip=True)
                keywords_match = re.search(r'KEYWORDS?:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                if keywords_match:
                    result['keywords'] = keywords_match.group(1).strip()
                    break

        return result

    def _extract_articles_from_issue(
        self,
        soup: BeautifulSoup,
        volume: str,
        issue: str
    ) -> List[Dict[str, Any]]:
        """
        Extract article metadata from an issue page.

        Args:
            soup: BeautifulSoup object of issue page
            volume: Volume number
            issue: Issue number

        Returns:
            List of article metadata dicts
        """
        articles = []

        # Find all article blocks
        # Taylor & Francis uses class names like "art_title" or "issue-item"
        # Look for elements marked with "Article" or containing article links
        article_blocks = soup.find_all(['div', 'li'], class_=re.compile(r'article|issue-item|hlFld-Title', re.IGNORECASE))

        if not article_blocks:
            # Fallback: find all links to /doi/full/ or /doi/abs/
            if self.verbosity >= 3:
                print("    Fallback: searching for doi links")
            article_links = soup.find_all('a', href=re.compile(r'/doi/(full|abs)/'))

            # Create pseudo-blocks from links
            for link in article_links:
                # Get the parent container
                parent = link.find_parent(['div', 'li', 'article'])
                if parent and parent not in article_blocks:
                    article_blocks.append(parent)

        if self.verbosity >= 3:
            print(f"    Found {len(article_blocks)} article block(s)")

        for block in article_blocks:
            # Extract title and URL
            title_link = block.find('a', href=re.compile(r'/doi/(full|abs)/'))

            if not title_link:
                continue

            title = title_link.get_text(strip=True)
            article_url = urljoin(self.BASE_URL, title_link.get('href', ''))

            # Ensure we're using the full URL (not abs)
            article_url = article_url.replace('/doi/abs/', '/doi/full/')

            # Extract DOI
            doi = self._extract_doi_from_url(article_url)

            # Extract authors
            authors = None
            authors_elem = block.find(['span', 'div', 'p'], class_=re.compile(r'author|contrib', re.IGNORECASE))
            if authors_elem:
                authors = authors_elem.get_text(strip=True)
            else:
                # Try finding by text pattern (names often have "and" or commas)
                block_text = block.get_text()
                # Look for line after title that looks like author names
                lines = [line.strip() for line in block_text.split('\n') if line.strip()]
                for i, line in enumerate(lines):
                    if title in line and i + 1 < len(lines):
                        potential_authors = lines[i + 1]
                        # Check if it looks like author names (contains "and" or multiple commas)
                        if ' and ' in potential_authors.lower() or potential_authors.count(',') >= 1:
                            authors = potential_authors
                            break

            # Extract pages
            pages = None
            pages_elem = block.find(['span', 'div'], class_=re.compile(r'page', re.IGNORECASE))
            if pages_elem:
                pages_text = pages_elem.get_text(strip=True)
                # Extract just the page numbers
                pages_match = re.search(r'(\d+[-–]\d+|\d+)', pages_text)
                if pages_match:
                    pages = pages_match.group(1)
            else:
                # Try finding in block text
                pages_match = re.search(r'[Pp]ages?:?\s*(\d+[-–]\d+|\d+)', block.get_text())
                if pages_match:
                    pages = pages_match.group(1)

            # Extract publication date
            publication_date = None
            date_elem = block.find(['span', 'div'], class_=re.compile(r'date|published', re.IGNORECASE))
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                # Look for "Published online: DD Mon YYYY" or similar
                date_match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', date_text)
                if date_match:
                    publication_date = date_match.group(1)
            else:
                # Try finding in block text
                block_text = block.get_text()
                date_match = re.search(r'Published\s+online:?\s*(\d{1,2}\s+\w+\s+\d{4})', block_text, re.IGNORECASE)
                if date_match:
                    publication_date = date_match.group(1)

            # Build article dict
            article = {
                'url': article_url,
                'title': title,
                'volume': volume,
                'number': issue,
                'publisher': 'Taylor & Francis',
                'itemtype': 'article',
            }

            # Add journal name if provided
            if self.journal_name:
                article['journal'] = self.journal_name

            # Add ISSN values if provided
            if self.issn:
                article['issn'] = self.issn
            if self.eissn:
                article['eissn'] = self.eissn

            if doi:
                article['doi'] = doi

            if authors:
                article['authors'] = authors

            if pages:
                article['pages'] = pages

            if publication_date:
                article['publication_date'] = publication_date

            # Fetch abstract and keywords
            if self.verbosity >= 3:
                print(f"      Fetching abstract for: {title[:50]}...")

            abstract_data = self._extract_abstract_and_keywords(article_url)
            if abstract_data['abstract']:
                article['abstract'] = abstract_data['abstract']
            if abstract_data['keywords']:
                article['keywords'] = abstract_data['keywords']

            articles.append(article)

        return articles

    def ingest_from_archives(
        self,
        archives_url: str,
        max_issues: Optional[int] = None
    ) -> None:
        """
        Ingest articles from Taylor & Francis journal archives.

        Args:
            archives_url: URL of the archives page
            max_issues: Maximum number of issues to process (for testing)
        """
        if self.verbosity >= 2:
            print(f"Fetching archives from: {archives_url}")

        # Fetch archives page
        archives_soup = self._fetch_page(archives_url)
        if not archives_soup:
            if self.verbosity >= 1:
                print("Failed to fetch archives page")
            return

        # Extract volume/issue links
        issue_links = self._extract_volume_issue_links(archives_soup)
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

            # Extract articles from issue page
            articles = self._extract_articles_from_issue(
                issue_soup,
                issue_info['volume'],
                issue_info['issue']
            )

            if self.verbosity >= 2:
                print(f"  Extracted {len(articles)} article(s)")

            if not articles:
                continue

            # Ingest articles
            self._ingest_documents(
                documents=articles,
                meta={
                    'source': 'mycology-taylor-francis',
                    'volume': issue_info['volume'],
                    'issue': issue_info['issue'],
                },
                bibtex_link=issue_info['url']
            )
