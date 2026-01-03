"""
Base class for ingesting web data into CouchDB.

This module provides the abstract Ingestor base class that defines the
interface for different data source ingestors.
"""

import os
import time
import random
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
from urllib.robotparser import RobotFileParser

import bibtexparser
import couchdb
import feedparser
import requests
from bs4 import BeautifulSoup
from uuid import uuid5, NAMESPACE_URL


class Ingestor(ABC):
    """
    Base class for ingesting web data or local copies of websites into CouchDB.

    This class provides common functionality for ingesting data from RSS feeds,
    BibTeX files, and local file systems. Subclasses should implement
    source-specific transformations and URL formatting.
    """

    db: couchdb.Database
    user_agent: str
    robot_parser: RobotFileParser
    verbosity: int
    local_pdf_map: Dict[str, str]
    rate_limit_min_ms: int
    rate_limit_max_ms: int
    last_fetch_time: Optional[float]
    suppressed_domains: Dict[str, str]

    def __init__(
        self,
        db: couchdb.Database,
        user_agent: str,
        robot_parser: RobotFileParser,
        verbosity: int = 2,
        local_pdf_map: Optional[Dict[str, str]] = None,
        rate_limit_min_ms: int = 1000,
        rate_limit_max_ms: int = 5000,
        max_retries: int = 3,
        retry_base_wait_time: int = 60,
        retry_backoff_multiplier: float = 2.0,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Ingestor.

        Args:
            db: CouchDB database instance for storing documents
            user_agent: User agent string for HTTP requests
            robot_parser: Robot file parser for checking crawl permissions
            verbosity: Verbosity level (0=silent, 1=warnings, 2=normal, 3=verbose)
            local_pdf_map: Optional mapping of URL prefixes to local directories.
                When fetching PDFs, if the URL starts with a prefix in this map,
                the PDF will be read from the corresponding local directory
                instead of being downloaded.
                Example: {'https://mykoweb.com/journals': '/data/skol/www/mykoweb.com/journals'}
            rate_limit_min_ms: Minimum delay between requests in milliseconds (default: 1000)
            rate_limit_max_ms: Maximum delay between requests in milliseconds (default: 5000)
            max_retries: Maximum retry attempts for 429 rate limit errors (default: 3)
            retry_base_wait_time: Initial wait time in seconds for exponential backoff (default: 60)
            retry_backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
            **kwargs: Additional parameters (ignored by base class, used by subclasses)
        """
        self.db = db
        self.user_agent = user_agent
        self.robot_parser = robot_parser
        self.verbosity = verbosity
        self.local_pdf_map = local_pdf_map if local_pdf_map is not None else {}
        self.rate_limit_min_ms = rate_limit_min_ms
        self.rate_limit_max_ms = rate_limit_max_ms
        self.max_retries = max_retries
        self.retry_base_wait_time = retry_base_wait_time
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.last_fetch_time = None
        self.suppressed_domains = {}  # Dict[domain, reason] for 403 forbidden domains
        self._session = requests.Session()

    @abstractmethod
    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Each subclass must implement this method to define its specific
        ingestion logic. This method should call the appropriate ingestion
        method (e.g., ingest_from_rss, ingest_from_local_bibtex, etc.)
        with the parameters stored in the instance.
        """
        raise NotImplementedError("Subclasses must implement ingest()")

    def _apply_rate_limit(self) -> None:
        """
        Apply rate limiting before making an HTTP request.

        Uses Crawl-Delay from robots.txt if available, otherwise uses a
        random delay between configured min/max bounds.
        """
        if self.last_fetch_time is None:
            return

        # Check if robots.txt specifies a Crawl-Delay
        crawl_delay = self.robot_parser.crawl_delay(self.user_agent)

        if crawl_delay is not None:
            # Use Crawl-Delay from robots.txt (in seconds)
            # crawl_delay can be int, float, or string - convert to float
            delay_seconds = float(crawl_delay)
            if self.verbosity >= 3:
                print(f"  Using Crawl-Delay from robots.txt: {delay_seconds}s")
        else:
            # Use random delay between configured bounds (convert ms to seconds)
            delay_seconds = random.uniform(
                self.rate_limit_min_ms / 1000.0,
                self.rate_limit_max_ms / 1000.0
            )
            if self.verbosity >= 3:
                print(f"  Using random delay: {delay_seconds:.2f}s")

        # Calculate time since last fetch
        elapsed = time.time() - self.last_fetch_time
        sleep_time = delay_seconds - elapsed

        if sleep_time > 0:
            if self.verbosity >= 3:
                print(f"  Sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

    def _check_suppression(self, url: str) -> Optional[requests.Response]:
        """
        Check if the domain of the given URL is suppressed.

        Args:
            url: URL to check

        Returns:
            Mock 403 Response if domain is suppressed, None otherwise
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        if domain in self.suppressed_domains:
            if self.verbosity >= 2:
                reason = self.suppressed_domains[domain]
                print(f"  Skipping suppressed domain: {domain} ({reason})")

            # Return a mock 403 response
            mock_response = requests.Response()
            mock_response.status_code = 403
            mock_response.url = url
            mock_response._content = b''
            return mock_response

        return None

    def _get_with_rate_limit(self, url: str, **kwargs: Any) -> requests.Response:
        """
        Wrapper for requests.get() that respects rate limiting.

        Handles HTTP 429 (Too Many Requests) responses by checking the
        Retry-After header and waiting the specified duration before retrying.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments to pass to requests.get()

        Returns:
            Response object from requests.get()
        """
        # Check if domain is suppressed (returns mock 403 if suppressed)
        suppression_response = self._check_suppression(url)
        if suppression_response is not None:
            return suppression_response

        self._apply_rate_limit()

        if self.verbosity >= 3:
            print(f"  Fetching: {url}")

        # Record fetch time before making request
        self.last_fetch_time = time.time()

        # Ensure User-Agent header is set
        headers = kwargs.get('headers', {})
        if 'User-Agent' not in headers:
            headers['User-Agent'] = self.user_agent
            kwargs['headers'] = headers

        response = self._session.get(url, **kwargs)

        # Log rate limit info if available (for debugging)
        if self.verbosity >= 4:
            rate_limit_headers = {
                'RateLimit-Limit': response.headers.get('RateLimit-Limit'),
                'RateLimit-Remaining': response.headers.get('RateLimit-Remaining'),
                'RateLimit-Reset': response.headers.get('RateLimit-Reset'),
                'X-RateLimit-Limit': response.headers.get('X-RateLimit-Limit'),
                'X-RateLimit-Remaining': (
                    response.headers.get('X-RateLimit-Remaining')
                ),
                'X-RateLimit-Reset': response.headers.get('X-RateLimit-Reset'),
            }
            present_headers = {
                k: v for k, v in rate_limit_headers.items() if v
            }
            if present_headers:
                print(f"  Rate limit headers: {present_headers}")

        # Handle HTTP 429 (Too Many Requests) with various headers
        if response.status_code == 429:
            wait_seconds = None
            header_used = None

            # 1. Check Retry-After header (RFC 7231)
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    # Try parsing as integer (seconds)
                    wait_seconds = int(retry_after)
                    header_used = 'Retry-After'
                except ValueError:
                    # Try parsing as HTTP date
                    try:
                        from email.utils import parsedate_to_datetime
                        retry_time = parsedate_to_datetime(retry_after)
                        wait_seconds = (
                            retry_time - datetime.now(retry_time.tzinfo)
                        ).total_seconds()
                        wait_seconds = max(0, wait_seconds)
                        header_used = 'Retry-After'
                    except (ValueError, TypeError):
                        wait_seconds = None

            # 2. Check RateLimit-Reset or X-RateLimit-Reset (IETF draft)
            if wait_seconds is None:
                reset_header = (
                    response.headers.get('RateLimit-Reset') or
                    response.headers.get('X-RateLimit-Reset')
                )
                if reset_header:
                    try:
                        reset_value = int(reset_header)
                        # Could be either:
                        # - Unix timestamp (> 1000000000)
                        # - Seconds until reset (< 1000000000)
                        current_time = time.time()
                        if reset_value > 1000000000:
                            # Unix timestamp
                            wait_seconds = max(0, reset_value - current_time)
                        else:
                            # Seconds until reset
                            wait_seconds = reset_value
                        header_used = (
                            'RateLimit-Reset'
                            if 'RateLimit-Reset' in response.headers
                            else 'X-RateLimit-Reset'
                        )
                    except (ValueError, TypeError):
                        wait_seconds = None

            # 3. Default if no headers found or parsing failed
            if wait_seconds is None:
                wait_seconds = 60
                header_used = 'default'
                if self.verbosity >= 1:
                    print(
                        "  Warning: No rate limit headers found, "
                        "using 60s default"
                    )
                    print("  All response headers:")
                    for header_name, header_value in response.headers.items():
                        print(f"    {header_name}: {header_value}")

            if self.verbosity >= 1:
                msg = (
                    f"  HTTP 429 (Too Many Requests) - "
                    f"waiting {wait_seconds:.0f}s before retry "
                    f"(from {header_used})"
                )
                print(msg)

            time.sleep(wait_seconds)

            # Update last fetch time after the wait
            self.last_fetch_time = time.time()

            # Retry the request
            if self.verbosity >= 3:
                print(f"  Retrying: {url}")

            response = requests.get(url, **kwargs)

        # Handle HTTP 403 (Forbidden) - add domain to suppression list
        if response.status_code == 403:
            self._register_suppression(url)

        # Log non-200 status codes
        if response.status_code != 200:
            if self.verbosity >= 1:
                msg = (
                    f"  Warning: Received status code "
                    f"{response.status_code} for URL: {url}"
                )
                print(msg)

        return response

    def _register_suppression(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        if domain not in self.suppressed_domains:
            reason = f"403 Forbidden at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self.suppressed_domains[domain] = reason

            if self.verbosity >= 1:
                print(f"  Domain suppressed due to 403 Forbidden: {domain}")

    def _retry_with_backoff(self, func, *args, operation_name: str = "operation", **kwargs):
        """
        Execute a function with exponential backoff retry logic for rate limit errors.

        Args:
            func: Callable to execute
            *args: Positional arguments to pass to func
            operation_name: Name of the operation for logging (default: "operation")
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result of func() or None if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                # Success - update last fetch time and return
                self.last_fetch_time = time.time()
                return result
            except Exception as e:
                error_msg = str(e).lower()
                # Check if error is rate-limit related
                is_rate_limit = (
                    '429' in error_msg or
                    'too many requests' in error_msg or
                    'rate limit' in error_msg
                )

                if is_rate_limit and attempt < self.max_retries - 1:
                    # Calculate exponential backoff wait time
                    wait_time = self.retry_base_wait_time * (self.retry_backoff_multiplier ** attempt)
                    if self.verbosity >= 2:
                        print(f"  {operation_name} hit rate limit, waiting {wait_time:.0f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    # Not a rate limit error, or we've exhausted retries
                    if self.verbosity >= 3:
                        error_type = "rate limit" if is_rate_limit else "error"
                        print(f"  {operation_name} {error_type}: {e}")
                    return None

        return None

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a web page.

        Checks robots.txt before fetching to ensure compliance.
        Implements rate limiting using Crawl-Delay from robots.txt if available,
        otherwise uses a random delay between configured min/max bounds.

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
            response = self._get_with_rate_limit(url)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            if self.verbosity >= 1:
                print(f"  Error fetching {url}: {e}")
            return None

    def _get_local_pdf_path(self, pdf_url: str) -> Optional[Path]:
        """
        Get the local filesystem path for a PDF URL if it exists in local_pdf_map.

        This method tries to find the local file in two ways:
        1. First with the URL path as-is (may contain URL encoding)
        2. If not found, with URL-decoded path (e.g., %20 -> space)

        Args:
            pdf_url: The PDF URL to check

        Returns:
            Path object if a local file exists, None otherwise
        """
        for url_prefix, local_dir in self.local_pdf_map.items():
            if pdf_url.startswith(url_prefix):
                # Replace URL prefix with local directory
                relative_path = pdf_url[len(url_prefix):]

                # Try with URL-encoded path first
                local_path = Path(local_dir) / relative_path.lstrip('/')
                if local_path.exists() and local_path.is_file():
                    return local_path

                # If not found, try with URL-decoded path
                # This handles cases like "Introduction%20to%20Mycology.pdf"
                # -> "Introduction to Mycology.pdf"
                decoded_relative_path = unquote(relative_path)
                if decoded_relative_path != relative_path:
                    local_path_decoded = Path(local_dir) / decoded_relative_path.lstrip('/')
                    if local_path_decoded.exists() and local_path_decoded.is_file():
                        return local_path_decoded

        return None

    def format_bibtex_url(self, base: Dict[str, str], bibtex_link: Optional[str]) -> Optional[str]:
        """
        Format the BibTeX URL for a given entry.

        Args:
            base: The base dictionary from the BibTeX entry
            bibtex_link: The original BibTeX link
        Returns:
            Formatted BibTeX URL
        """
        return bibtex_link

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format a PDF URL according to source-specific requirements.

        Args:
            base_url: The base URL from the BibTeX entry

        Returns:
            Formatted PDF URL ready for fetching
        """
        return base['url']

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL according to source-specific requirements.

        Args:
            base_url: The base URL from the BibTeX entry

        Returns:
            Formatted human-readable URL
        """
        return base['url']

    def transform_bibtex_content(self, content: bytes) -> bytes:
        """
        Apply source-specific transformations to BibTeX content.

        Some sources have syntax quirks that need to be fixed before parsing.

        Args:
            content: Raw BibTeX content as bytes

        Returns:
            Transformed BibTeX content ready for parsing
        """
        return content

    def _ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        meta: Dict[str, Any],
        bibtex_link: str
    ) -> None:
        """
        Process and ingest a list of document dictionaries.

        This method handles the core ingestion logic: checking for duplicates,
        verifying robot permissions, saving to CouchDB, and fetching PDFs.
        It treats the document dictionaries as opaque data structures.

        Args:
            documents: List of document dictionaries with arbitrary fields
            meta: Metadata dictionary to attach to each document
            bibtex_link: URL or path to the BibTeX file
        """
        for doc_dict in documents:
            # Create document with formatted URLs
            pdf_url = self.format_pdf_url(doc_dict)
            # Use UUID5 to create a consistent document ID based on the PDF URL
            doc = {
                '_id': uuid5(NAMESPACE_URL, pdf_url).hex,
                'meta': meta,
                'bibtex_url': self.format_bibtex_url(doc_dict, bibtex_link),
                'pdf_url': pdf_url,
                'human_url': self.format_human_url(doc_dict),
            }

            # Add all fields from source dictionary
            for k, v in doc_dict.items():
                doc[k] = v

            # Check if document already exists
            doc_exists = doc['_id'] in self.db
            if doc_exists:
                # Document exists - check if it has PDF attachment
                existing_doc = self.db[doc['_id']]
                has_pdf = '_attachments' in existing_doc and 'article.pdf' in existing_doc['_attachments']

                if has_pdf:
                    # Document has PDF, skip it
                    if self.verbosity >= 2:
                        print(f"Skipping {pdf_url} (already has PDF)")
                    continue
                else:
                    # Document exists but missing PDF - we'll add it below
                    if self.verbosity >= 2:
                        print(f"Adding PDF to existing record: {pdf_url}")
                    doc = existing_doc  # Use existing doc to preserve _rev
            else:
                # New document
                if self.verbosity >= 2:
                    print(f"Adding {pdf_url}")

            # Check robot permissions
            if not self.robot_parser.can_fetch(self.user_agent, pdf_url):
                # TODO(piggy): We should probably log blocked URLs.
                if self.verbosity >= 1:
                    print(f"Robot permission denied {pdf_url}")
                continue

            # Save document to CouchDB if it's new
            if not doc_exists:
                _doc_id, _doc_rev = self.db.save(doc)

            # Fetch PDF - check local first, then download if needed
            local_pdf_path = self._get_local_pdf_path(pdf_url)

            if local_pdf_path:
                # Read PDF from local filesystem
                if self.verbosity >= 3:
                    print(f"  Reading PDF from local file: {local_pdf_path}")
                with open(local_pdf_path, 'rb') as pdf_f:
                    pdf_doc = pdf_f.read()
            else:
                # Download PDF from URL with rate limiting
                if self.verbosity >= 3:
                    print(f"  Downloading PDF from: {pdf_url}")
                response = self._get_with_rate_limit(pdf_url, stream=False)
                if response.status_code != 200:
                    if self.verbosity >= 1:
                        print(f"  Failed to download PDF: {pdf_url} (status code {response.status_code})")
                    continue
                pdf_doc = response.content

            attachment_filename = 'article.pdf'
            attachment_content_type = 'application/pdf'
            attachment_file = BytesIO(pdf_doc)

            self.db.put_attachment(
                doc,
                attachment_file,
                attachment_filename,
                attachment_content_type
            )

            if self.verbosity >= 3:
                print("-" * 10)

    def _is_url_ingested(self, url: str) -> bool:
        """
        Check if a URL has already been ingested.

        We do this check twice so that we can bail out early if we already
        have the PDF attachment.

        Args:
            url: URL to check
        Returns:
            True if the URL has been fully ingested, False otherwise
        """
        doc_id = uuid5(NAMESPACE_URL, url).hex
        return (doc_id in self.db and
                '_attachments' in self.db[doc_id] and
                'article.pdf' in self.db[doc_id]['_attachments'])

    def ingest_from_bibtex(
        self,
        content: bytes,
        bibtex_link: str,
        meta: Dict[str, Any]
    ) -> None:
        """
        Load documents referenced in a BibTeX database.

        Args:
            content: BibTeX file content as bytes
            bibtex_link: URL or path to the BibTeX file
            meta: Metadata dictionary to attach to each document
        """
        # Apply source-specific transformations
        transformed_content = self.transform_bibtex_content(content)

        # Parse BibTeX
        bib_database = bibtexparser.parse_string(transformed_content.decode('utf-8'))

        # Extract all BibTeX fields into plain dictionaries first
        # This loop runs before processing to make documents opaque
        documents = []
        for bib_entry in bib_database.entries:
            doc_dict = {}
            for k in bib_entry.fields_dict.keys():
                doc_dict[k] = bib_entry[k]
            documents.append(doc_dict)

        # Process all documents through generic ingestion pipeline
        self._ingest_documents(documents, meta, bibtex_link)

    def ingest_from_rss(
        self,
        rss_url: str,
        bibtex_url_template: Optional[str] = None
    ) -> None:
        """
        Ingest documents from an RSS feed.

        Args:
            rss_url: URL of the RSS feed
            bibtex_url_template: Optional template for constructing BibTeX URLs.
                If None, uses entry.link + '?format=bib'
        """
        feed = feedparser.parse(rss_url)

        # Check if feedparser successfully parsed the feed
        if feed.bozo:
            # Parse error - likely got HTML error page instead of RSS
            error_msg = f"Failed to parse RSS feed from {rss_url}"
            if hasattr(feed, 'bozo_exception'):
                error_msg += f": {feed.bozo_exception}"
            # Check if we got an access denied page
            if hasattr(feed.feed, 'summary') and 'Access Denied' in feed.feed.get('summary', ''):
                error_msg = f"Access denied when fetching RSS feed from {rss_url}"
            if self.verbosity >= 1:
                print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

        # feedparser normalizes all feed formats to feed.feed
        # Use getattr with defaults for missing optional fields
        feed_meta = {
            'url': rss_url,
            'title': getattr(feed.feed, 'title', 'Unknown'),
            'link': getattr(feed.feed, 'link', rss_url),
            'description': getattr(feed.feed, 'description', ''),
        }

        for entry in feed.entries:
            entry_meta = {
                'title': entry.title,
                'link': entry.link,
            }
            if hasattr(entry, 'summary'):
                entry_meta['summary'] = entry.summary
            if hasattr(entry, 'description'):
                entry_meta['description'] = entry.description

            # Construct BibTeX URL
            if bibtex_url_template:
                bibtex_link = bibtex_url_template.format(link=entry.link)
            else:
                bibtex_link = f'{entry.link}?format=bib'

            if self.verbosity >= 3:
                print(f"bibtex_link: {bibtex_link}")

            # Check robot permissions
            if not self.robot_parser.can_fetch(self.user_agent, bibtex_link):
                if self.verbosity >= 1:
                    print(f"Robot permission denied {bibtex_link}")
                continue

            # Fetch BibTeX file with rate limiting
            bibtex_response = self._get_with_rate_limit(bibtex_link, stream=False)
            if bibtex_response.status_code != 200:
                if self.verbosity >= 1:
                    print(f"  Failed to download BibTeX: {bibtex_link} (status code {bibtex_response.status_code})")
                continue
            self.ingest_from_bibtex(
                content=bibtex_response.content,
                bibtex_link=bibtex_link,
                meta={
                    'feed': feed_meta,
                    'entry': entry_meta,
                }
            )
            if self.verbosity >= 3:
                print("=" * 20)

    def ingest_from_local_bibtex(
        self,
        root: Path,
        bibtex_file_pattern: str = 'format=bib',
        url_prefix: str = ''
    ) -> None:
        """
        Ingest from a local directory containing BibTeX files.

        Args:
            root: Root directory to search for BibTeX files
            bibtex_file_pattern: Filename pattern to match BibTeX files
            url_prefix: URL prefix to construct bibtex_link from file path
        """
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith(bibtex_file_pattern):
                    continue

                full_filepath = os.path.join(dirpath, filename)

                # Construct URL from file path
                if url_prefix:
                    relative_path = full_filepath[len(str(root)):]
                    bibtex_link = f"{url_prefix}{relative_path}"
                else:
                    bibtex_link = full_filepath

                # Read and process BibTeX file
                with open(full_filepath, 'rb') as f:
                    content = f.read()
                    self.ingest_from_bibtex(
                        content=content,
                        bibtex_link=bibtex_link,
                        meta={}
                    )
