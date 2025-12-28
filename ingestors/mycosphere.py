"""
MycosphereIngestor for ingesting articles from mycosphere.org.

This module provides the MycosphereIngestor class for scraping and ingesting
articles from the Mycosphere journal website.
"""

import re
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from .ingestor import Ingestor


class MycosphereIngestor(Ingestor):
    """
    Ingestor for Mycosphere journal (mycosphere.org).

    This class scrapes the Mycosphere website to extract article metadata
    and PDFs. It handles both volume index pages and individual issue pages.

    Journal information:
        ISSN: 2077-7000
        eISSN: 2077-7019
        URL: https://mycosphere.org

    Navigation:
        - Start: https://mycosphere.org/archives.php
        - Follow volume links (e.g., https://mycosphere.org/volume-12-2021/)
        - If volume has issue pages, follow those; otherwise parse articles from volume page

    Metadata extracted:
        - title (from PDF link name)
        - volume, number (issue)
        - authors
        - receipt_date (from "Received")
        - acceptance_date (from "Accepted")
        - publication_date (from "Published")
        - year (extracted from publication_date)
        - pages
        - abstract
        - keywords
    """

    # Journal metadata
    ISSN = '2077-7000'
    EISSN = '2077-7019'
    BASE_URL = 'https://mycosphere.org'
    ARCHIVES_URL = 'https://mycosphere.org/archives.php'

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL - already complete from scraping.

        Args:
            base: Dictionary containing the 'pdf_url' field

        Returns:
            The PDF URL
        """
        return base.get('pdf_url', base.get('url', ''))

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL.

        Args:
            base: Dictionary containing the 'url' field

        Returns:
            The human-readable URL
        """
        return base.get('url', base.get('pdf_url', ''))

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a web page.

        Checks robots.txt before fetching to ensure compliance.

        Args:
            url: URL to fetch

        Returns:
            BeautifulSoup object or None on error
        """
        # Check robots.txt before fetching
        if not self.robot_parser.can_fetch(self.user_agent, url):
            if self.verbosity >= 2:
                print(f"  Blocked by robots.txt: {url}")
            return None

        try:
            if self.verbosity >= 3:
                print(f"  Fetching: {url}")

            response = requests.get(url, headers={'User-Agent': self.user_agent})
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            if self.verbosity >= 1:
                print(f"  Error fetching {url}: {e}")
            return None

    def _extract_volume_links(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract volume links from the archives page.

        Ignores sidebar content and focuses on main content area.

        Args:
            soup: BeautifulSoup object of archives page

        Returns:
            List of dicts with 'url', 'volume', and 'year' keys
        """
        volume_links = []
        seen_urls = set()  # Track URLs to avoid duplicates

        # Find all links that match volume patterns
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)

            # Match patterns like "volume-12-2021" in href
            volume_match = re.search(r'volume-(\d+)-(\d{4})', href)
            if volume_match:
                volume_num = volume_match.group(1)
                year = volume_match.group(2)
                full_url = urljoin(self.BASE_URL, href)

                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    volume_links.append({
                        'url': full_url,
                        'volume': volume_num,
                        'year': year,
                        'text': text
                    })
            else:
                # Also match simple pattern like "volume-1/" or "/volume-1/"
                # These are typically older volumes
                simple_match = re.search(r'/volume-(\d+)/?$', href)
                if simple_match:
                    volume_num = simple_match.group(1)
                    # Try to extract year from link text
                    year_match = re.search(r'(\d{4})', text)
                    year = year_match.group(1) if year_match else None

                    full_url = urljoin(self.BASE_URL, href)

                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        volume_links.append({
                            'url': full_url,
                            'volume': volume_num,
                            'year': year or 'unknown',
                            'text': text
                        })

        return volume_links

    def _extract_issue_links(self, soup: BeautifulSoup, volume: str) -> List[Dict[str, Any]]:
        """
        Extract issue links from a volume page.

        Args:
            soup: BeautifulSoup object of volume page
            volume: Volume number

        Returns:
            List of dicts with 'url', 'volume', and 'number' keys
        """
        issue_links = []

        # Look for links to issue pages
        # Pattern might be like "Issue 1", "Issue 2", etc.
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)

            # Match issue patterns
            issue_match = re.search(r'issue[_-]?(\d+)', href, re.IGNORECASE)
            if not issue_match:
                # Try matching from link text
                issue_match = re.search(r'issue\s+(\d+)', text, re.IGNORECASE)

            if issue_match:
                issue_num = issue_match.group(1)
                full_url = urljoin(self.BASE_URL, href)

                issue_links.append({
                    'url': full_url,
                    'volume': volume,
                    'number': issue_num,
                    'text': text
                })

        return issue_links

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

        # Clean up the date string - remove trailing commas, extra spaces, etc.
        date_str = date_str.strip()
        date_str = re.sub(r',\s*$', '', date_str)  # Remove trailing comma
        date_str = re.sub(r'\s+', ' ', date_str)   # Normalize whitespace

        # Try common date formats
        formats = [
            '%d %B %Y',      # 15 January 2021
            '%B %d, %Y',     # January 15, 2021
            '%d %b %Y',      # 15 Jan 2021
            '%Y-%m-%d',      # 2021-01-15
            '%d/%m/%Y',      # 15/01/2021
            '%m/%d/%Y',      # 01/15/2021
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

    def _extract_articles_from_volume_index(
        self,
        soup: BeautifulSoup,
        volume: str
    ) -> List[Dict[str, Any]]:
        """
        Extract basic article information from volume index page.

        This extracts titles, PDF URLs, issue numbers, and page numbers from
        the volume index page. Page numbers come from citations like
        "Mycosphere 1(2), 89-101" which mean Volume 1, Issue 2, pages 89-101.

        Args:
            soup: BeautifulSoup object of volume index page
            volume: Volume number

        Returns:
            List of article metadata dicts with title, pdf_url, url, number, pages
        """
        articles = []

        # Find the main content area (ignore sidebar)
        # Look for div with id="divMain" or class="span9"
        main_content = soup.find('div', id='divMain')
        if not main_content:
            main_content = soup.find('div', class_=re.compile(r'span9|main|content'))
        if not main_content:
            if self.verbosity >= 2:
                print("    Warning: Could not find main content area, using whole page")
            main_content = soup

        # Find the volume header
        volume_pattern = rf'Volume\s+{re.escape(volume)}[^\n]*Index'
        volume_header = main_content.find(['h1', 'h2', 'h3'], string=re.compile(volume_pattern, re.IGNORECASE))

        if not volume_header:
            if self.verbosity >= 2:
                print(f"    Warning: Could not find volume {volume} header on index page")
            # Still try to extract, but may get wrong volume's articles
            search_area = main_content
        else:
            # Get content after the volume header until next volume header or end
            search_area = volume_header.find_parent()
            if not search_area:
                search_area = main_content

        # Find all PDF links in the search area
        pdf_links = search_area.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE))

        for pdf_link in pdf_links:
            pdf_url = urljoin(self.BASE_URL, pdf_link['href'])
            title = pdf_link.get_text(strip=True)

            # Clean title (remove .pdf extension if present)
            title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE).strip()

            if not title:
                continue

            # Get the immediate parent (should be <p> tag with the full article entry)
            parent = pdf_link.parent
            if not parent:
                continue

            parent_text = parent.get_text()

            # Extract citation: "Mycosphere X(Y), Z-W" where X=volume, Y=issue, Z-W=pages
            # Also handle format without parentheses: "Mycosphere X, Z-W"
            citation_match = re.search(
                r'Mycosphere\s+(\d+)\(([^\)]+)\),\s*(\d+)\s*[–-]\s*(\d+)',
                parent_text
            )

            article = {
                'title': title,
                'pdf_url': pdf_url,
                'url': pdf_url,  # Use PDF URL as the main URL
                'volume': volume,
                'issn': self.ISSN,
                'eissn': self.EISSN,
            }

            if citation_match:
                # Full citation with issue and pages
                cite_volume = citation_match.group(1)
                cite_issue = citation_match.group(2)
                page_start = citation_match.group(3)
                page_end = citation_match.group(4)

                article['number'] = cite_issue.strip()
                article['pages'] = f"{page_start}–{page_end}"

                if self.verbosity >= 4:
                    print(f"    Found citation: Mycosphere {cite_volume}({cite_issue}), {page_start}–{page_end}")
            else:
                # Try simpler pattern without issue number
                simple_citation = re.search(
                    r'Mycosphere\s+(\d+),\s*(\d+)\s*[–-]\s*(\d+)',
                    parent_text
                )
                if simple_citation:
                    cite_volume = simple_citation.group(1)
                    page_start = simple_citation.group(2)
                    page_end = simple_citation.group(3)

                    article['pages'] = f"{page_start}–{page_end}"

                    if self.verbosity >= 4:
                        print(f"    Found citation: Mycosphere {cite_volume}, {page_start}–{page_end}")

            articles.append(article)

        return articles

    def _extract_article_metadata(
        self,
        soup: BeautifulSoup,
        volume: str,
        number: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract article metadata from a page (either issue page or volume index page).

        Args:
            soup: BeautifulSoup object of the page
            volume: Volume number
            number: Issue number (optional)

        Returns:
            List of article metadata dicts
        """
        articles = []

        # Strategy: Get all the text content and split it into article sections
        # based on numbered patterns (1., 2., 3., etc.) or Keywords markers

        # First, collect all PDF links and their titles
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$', re.IGNORECASE))
        pdf_map = {}  # Map title to PDF URL

        for pdf_link in pdf_links:
            pdf_url = urljoin(self.BASE_URL, pdf_link['href'])
            title = pdf_link.get_text(strip=True)
            if title and not title.lower().endswith('.pdf'):
                # Clean title
                title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE).strip()
                if title:
                    pdf_map[title[:80]] = pdf_url  # Use first 80 chars as key

        # Get the main content text
        # Find a div or section that contains multiple articles
        main_content = soup.find('div', class_=re.compile(r'content|main|article', re.I))
        if not main_content:
            # Fallback: use body
            main_content = soup.find('body')
        if not main_content:
            return articles

        full_text = main_content.get_text()

        # Split by Keywords: to identify article boundaries
        # Each article should have exactly one "Keywords:" section
        article_sections = []

        # Use regex to find all article sections
        # Pattern: Article starts with optional number, title, authors, dates, abstract, keywords
        # Look for pattern: Keywords: ... (marks end of article)
        pattern = r'(\d+\.\s+.+?)(?=\d+\.\s+|\Z)'
        matches = re.finditer(pattern, full_text, re.DOTALL)

        for match in matches:
            section_text = match.group(1)
            # Check if this section has Keywords (valid article)
            if 'Keywords' in section_text and len(section_text) > 100:
                article_sections.append(section_text)

        if not article_sections:
            # Fallback: no numbered articles found, maybe different format
            # Try splitting by Keywords directly
            parts = re.split(r'Keywords:', full_text)
            for i in range(len(parts) - 1):
                # Reconstruct article: everything before this Keywords + Keywords + the keywords themselves
                section = parts[i] + 'Keywords:' + parts[i+1].split('\n\n')[0]
                if len(section) > 100:
                    article_sections.append(section)

        # Process each article section
        for section_text in article_sections:
            # Extract title - first line after number
            title_match = re.search(r'^\d+\.\s+(.+?)(?=\n|Author)', section_text, re.MULTILINE)
            if not title_match:
                title_match = re.search(r'^(.+?)(?=\n|Author)', section_text, re.MULTILINE)

            title = title_match.group(1).strip() if title_match else ""

            # Find matching PDF URL
            pdf_url = None
            for pdf_title, url in pdf_map.items():
                # Fuzzy match on first 50 chars
                if title[:50].lower() in pdf_title.lower() or pdf_title.lower() in title[:50].lower():
                    pdf_url = url
                    break

            if not pdf_url:
                # Skip if no PDF found
                continue

            container_text = section_text

            # Initialize article dict
            article = {
                'url': pdf_url,
                'pdf_url': pdf_url,
                'title': title,
                'volume': volume,
                'journal': 'Mycosphere',
                'issn': self.ISSN,
                'eissn': self.EISSN,
                'itemtype': 'article',
            }

            # Add issue number if available
            if number:
                article['number'] = number

            # Extract authors
            authors_match = re.search(
                r'(?:Author[s]?|By)[:\s]+([^\n]+?)(?=\n|Received|Accepted|Published|Pages|Abstract|$)',
                container_text,
                re.IGNORECASE | re.DOTALL
            )
            if authors_match:
                article['authors'] = authors_match.group(1).strip()

            # Extract received date (handle both "Received" and "Recieved" typo)
            received_match = re.search(
                r'Reci?eved[:\s]+([^,\n]+?)(?=,|;|Accepted|Published|\n|$)',
                container_text,
                re.IGNORECASE
            )
            if received_match:
                article['receipt_date'] = self._parse_date(received_match.group(1))

            # Extract accepted date
            accepted_match = re.search(
                r'Accepted[:\s]+([^,\n]+?)(?=,|;|Reci?eved|Published|\n|$)',
                container_text,
                re.IGNORECASE
            )
            if accepted_match:
                article['acceptance_date'] = self._parse_date(accepted_match.group(1))

            # Extract published date
            published_match = re.search(
                r'Published[:\s]+([^,\n]+?)(?=,|;|Reci?eved|Accepted|\n|$)',
                container_text,
                re.IGNORECASE
            )
            if published_match:
                pub_date = self._parse_date(published_match.group(1))
                article['publication_date'] = pub_date
                article['year'] = self._extract_year(pub_date)

            # Extract pages
            # First try explicit "Pages:" or "Pp." keyword
            pages_match = re.search(
                r'(?:Pages|Pp\.?|p\.?)[:\s]+(\d+[-–]\d+|\d+)',
                container_text,
                re.IGNORECASE
            )
            if pages_match:
                article['pages'] = pages_match.group(1).strip()
            else:
                # Try "Mycosphere <volume>, <pages>" format (from index pages)
                pages_fallback = re.search(
                    r'Mycosphere\s+\d+,\s*(\d+[-–]\d+)',
                    container_text,
                    re.IGNORECASE
                )
                if pages_fallback:
                    article['pages'] = pages_fallback.group(1).strip()

            # Extract abstract
            # First try explicit "Abstract:" keyword
            abstract_match = re.search(
                r'Abstract[:\s]+(.+?)(?=Keywords|Reci?eved|Accepted|Published|$)',
                container_text,
                re.IGNORECASE | re.DOTALL
            )
            if abstract_match:
                article['abstract'] = abstract_match.group(1).strip()
            else:
                # If no explicit "Abstract" keyword, try to extract text between dates line and keywords
                # Pattern: after the line with dates (ending with Published: date) until "Keywords:"
                # Handle both single-line dates (comma-separated) and multi-line dates
                abstract_fallback = re.search(
                    r'Published[:\s]+[^,\n]+(?:,|\n)(.+?)(?=Keywords|$)',
                    container_text,
                    re.IGNORECASE | re.DOTALL
                )
                if abstract_fallback:
                    abstract_text = abstract_fallback.group(1).strip()
                    # Clean up - remove extra whitespace and newlines
                    abstract_text = re.sub(r'\s+', ' ', abstract_text)
                    # Only use if substantial text (more than 20 chars)
                    if abstract_text and len(abstract_text) > 20:
                        article['abstract'] = abstract_text

            # Extract keywords
            keywords_match = re.search(
                r'Keywords?[:\s]+(.+?)(?=\n\n|Received|Accepted|Published|Abstract|$)',
                container_text,
                re.IGNORECASE | re.DOTALL
            )
            if keywords_match:
                article['keywords'] = keywords_match.group(1).strip()

            articles.append(article)

        return articles

    def ingest_from_archives(
        self,
        archives_url: str = ARCHIVES_URL,
        max_volumes: Optional[int] = None
    ) -> None:
        """
        Ingest articles from Mycosphere archives.

        Strategy:
        1. Always extract basic info (title, pages, PDF URL) from volume index pages
        2. Optionally enhance with detailed metadata (abstract, keywords, dates) from issue pages

        Args:
            archives_url: URL of the archives page
            max_volumes: Maximum number of volumes to process (for testing)
        """
        if self.verbosity >= 2:
            print(f"Fetching archives from: {archives_url}")

        # Fetch archives page
        archives_soup = self._fetch_page(archives_url)
        if not archives_soup:
            if self.verbosity >= 1:
                print("Failed to fetch archives page")
            return

        # Extract volume links
        volume_links = self._extract_volume_links(archives_soup)
        if self.verbosity >= 2:
            print(f"Found {len(volume_links)} volume(s)")

        # Process each volume
        for idx, vol_info in enumerate(volume_links):
            if max_volumes and idx >= max_volumes:
                break

            if self.verbosity >= 2:
                print(f"\n{'=' * 60}")
                print(f"Processing Volume {vol_info['volume']} ({vol_info['year']})")
                print(f"{'=' * 60}")

            # Fetch volume page
            vol_soup = self._fetch_page(vol_info['url'])
            if not vol_soup:
                continue

            # ALWAYS extract basic article info from volume index page
            # This gets us titles, PDF URLs, and page numbers
            articles = self._extract_articles_from_volume_index(vol_soup, vol_info['volume'])

            if self.verbosity >= 2:
                print(f"  Extracted {len(articles)} article(s) from volume index")

            if not articles:
                continue

            # Check for issue links - we'll use these to enhance metadata if available
            issue_links = self._extract_issue_links(vol_soup, vol_info['volume'])

            if issue_links and self.verbosity >= 2:
                print(f"  Found {len(issue_links)} issue page(s) for metadata enhancement")

            # Try to enhance articles with metadata from issue pages
            if issue_links:
                # Create a map of PDF URLs to articles for quick lookup
                pdf_map = {article['pdf_url']: article for article in articles}

                for issue_info in issue_links:
                    if self.verbosity >= 3:
                        print(f"    Enhancing metadata from Issue {issue_info['number']}")

                    issue_soup = self._fetch_page(issue_info['url'])
                    if not issue_soup:
                        continue

                    # Extract detailed metadata from issue page
                    issue_articles = self._extract_article_metadata(
                        issue_soup,
                        issue_info['volume'],
                        issue_info['number']
                    )

                    # Merge metadata: enhance our articles with data from issue pages
                    for issue_article in issue_articles:
                        # Try to match by PDF URL
                        if 'pdf_url' in issue_article and issue_article['pdf_url'] in pdf_map:
                            # Found a match - enhance the article
                            article = pdf_map[issue_article['pdf_url']]

                            # Add fields that are only available from issue pages
                            # Include 'number' to ensure articles get issue numbers even when
                            # not present in volume index citations (e.g., "Mycosphere 1, 1-9")
                            for key in ['number', 'authors', 'abstract', 'keywords',
                                       'receipt_date', 'acceptance_date', 'publication_date', 'year']:
                                if key in issue_article and issue_article[key]:
                                    article[key] = issue_article[key]

                            if self.verbosity >= 4:
                                print(f"      Enhanced: {article['title'][:50]}...")

            # Ingest all articles
            if self.verbosity >= 2:
                print(f"  Ingesting {len(articles)} article(s)")

            self._ingest_documents(
                documents=articles,
                meta={
                    'source': 'mycosphere',
                    'volume': vol_info['volume'],
                },
                bibtex_link=vol_info['url']
            )
