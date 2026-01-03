"""
Ingestor for journals via Crossref API using habanero and pypaperretriever.

This ingestor retrieves article metadata and PDFs from journals indexed in Crossref
by querying with an ISSN/eISSN. It uses habanero to fetch DOIs and metadata,
then pypaperretriever to download PDFs.
"""

import time
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from io import BytesIO

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from habanero import Crossref
except ImportError:
    raise ImportError("habanero library required. Install with: pip install habanero")

try:
    from pypaperretriever import PaperRetriever
except ImportError:
    raise ImportError("pypaperretriever library required. Install with: pip install pypaperretriever")

try:
    from paperscraper.pdf import save_pdf
except ImportError:
    save_pdf = None  # Optional fallback

from .ingestor import Ingestor


class CrossrefIngestor(Ingestor):
    """
    Ingestor for journals via Crossref API.

    Uses habanero to query Crossref for articles by ISSN/eISSN, then downloads
    PDFs using pypaperretriever. PDFs are stored in CouchDB and removed from
    the filesystem immediately after ingestion.
    """

    def __init__(
        self,
        issn: Optional[str] = None,
        mailto: str = "piggy.yarroll+skol@gmail.com",
        max_articles: Optional[int] = None,
        allow_scihub: bool = True,
        api_batch_delay: float = 0.1,
        **kwargs: Any
    ) -> None:
        """
        Initialize the CrossrefIngestor.

        Args:
            issn: ISSN or eISSN of the journal (with or without hyphen)
            mailto: Email address for Crossref API polite pool (default: piggy.yarroll+skol@gmail.com)
            max_articles: Maximum number of articles to ingest (None = all)
            allow_scihub: Whether to allow pypaperretriever to use Sci-Hub as fallback
            api_batch_delay: Delay in seconds between Crossref API batch requests (default: 0.1)
            **kwargs: Additional parameters passed to Ingestor base class (including max_retries, retry_base_wait_time, retry_backoff_multiplier)
        """
        super().__init__(**kwargs)
        self.issn = issn
        self.mailto = mailto
        self.max_articles = max_articles
        self.allow_scihub = allow_scihub
        self.api_batch_delay = api_batch_delay

    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Queries Crossref for all articles in the journal by ISSN, then downloads
        PDFs using pypaperretriever.
        """
        if not self.issn:
            raise ValueError("ISSN must be provided for CrossrefIngestor")

        # Normalize ISSN format (add hyphen if missing)
        issn = self.issn
        if '-' not in issn and len(issn) == 8:
            issn = f"{issn[:4]}-{issn[4:]}"

        if self.verbosity >= 1:
            print(f"Querying Crossref for journal ISSN: {issn}")
            print(f"Using mailto: {self.mailto}")

        # Initialize Crossref client
        cr = Crossref(mailto=self.mailto)

        # Ingest each work from the generator
        work_count = 0
        for work in self._fetch_all_works(cr, issn):
            work_count += 1

            if self.verbosity >= 2:
                max_str = f"/{self.max_articles}" if self.max_articles else ""
                print(f"\n[{work_count}{max_str}] Processing work...")

            try:
                self._ingest_work(work)
            except Exception as e:
                if self.verbosity >= 1:
                    doi = work.get('DOI', 'unknown')
                    print(f"  ERROR ingesting work {doi}: {e}")
                    if self.verbosity >= 3:
                        import traceback
                        traceback.print_exc()

        if self.verbosity >= 1:
            print(f"\nIngestion complete. Processed {work_count} works.")

    def _fetch_all_works(self, cr: Crossref, issn: str) -> Iterator[Dict[str, Any]]:
        """
        Fetch all works for a journal using cursor pagination.

        Args:
            cr: Crossref client instance
            issn: ISSN of the journal

        Yields:
            Work dictionaries from Crossref
        """
        cursor = '*'
        batch_num = 0
        if self.max_articles is not None:
            per_page = min(1000, self.max_articles)
        else:
            per_page = 1000 # Crossref allows up to 1000
        total_yielded = 0

        if self.verbosity >= 2:
            print(f"Fetching works from Crossref (max {self.max_articles})...")

        while True:
            batch_num += 1

            try:
                # Query Crossref with ISSN filter and cursor pagination
                results = cr.works(
                    filter={'issn': issn},
                    limit=per_page,
                    cursor=cursor,
                    progress_bar=(self.verbosity >= 2)
                )
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"ERROR: Failed to fetch batch {batch_num}: {e}")
                break

            # results is an iterable that yields response dictionaries
            batch_count = 0
            next_cursor = None

            for result in results:
                if not result or 'message' not in result:
                    break

                message = result['message']
                items = message.get('items', [])

                if not items:
                    break

                # Yield each work
                for item in items:
                    yield item
                    total_yielded += 1
                    batch_count += 1

                    # Check if we've reached the max limit
                    if (self.max_articles is not None and
                            total_yielded >= self.max_articles):
                        if self.verbosity >= 2:
                            print(f"  Reached max_articles limit of {self.max_articles}")
                        return

                # Get next cursor for pagination
                next_cursor = message.get('next-cursor')
                if not next_cursor or next_cursor == cursor:
                    # No more pages
                    next_cursor = None
                    break

            if self.verbosity >= 2:
                print(f"  Batch {batch_num}: fetched {batch_count} works (total: {total_yielded})")

            if batch_count == 0 or not next_cursor:
                # No more works or no next page
                break

            cursor = next_cursor

            # Rate limiting between API batch requests
            time.sleep(self.api_batch_delay)

    def _ingest_work(self, work: Dict[str, Any]) -> None:
        """
        Ingest a single work from Crossref.

        Args:
            work: Work dictionary from Crossref API
        """
        doi = work.get('DOI')
        if not doi:
            if self.verbosity >= 2:
                print("  Skipping work without DOI")
            return

        # Extract metadata from Crossref record
        title = work.get('title', [''])[0] if work.get('title') else ''
        authors = self._format_authors(work.get('author', []))
        year = self._extract_year(work)
        journal = work.get('container-title', [''])[0] if work.get('container-title') else ''
        volume = work.get('volume', '')
        issue = work.get('issue', '')
        pages = work.get('page', '')

        if self.verbosity >= 2:
            print(f"  DOI: {doi}")
            print(f"  Title: {title[:80]}...")

        # Create document URL from DOI
        url = f"https://doi.org/{doi}"

        # Extract human-readable URL from Crossref record
        human_url = self._extract_human_url(work, url)

        # Create document ID from URL
        from uuid import uuid5, NAMESPACE_URL
        doc_id = str(uuid5(NAMESPACE_URL, url))

        # Check if already ingested
        if doc_id in self.db:
            existing_doc = self.db[doc_id]
            has_pdf = '_attachments' in existing_doc and 'article.pdf' in existing_doc['_attachments']
            if has_pdf:
                if self.verbosity >= 2:
                    print(f"  Already ingested with PDF, skipping")
                return
            else:
                if self.verbosity >= 2:
                    print(f"  Found existing record without PDF, will add PDF")
                doc = existing_doc
        else:
            # Create new document
            doc = {
                '_id': doc_id,
                'url': url,
                'doi': doi,
                'title': title,
                'author': authors,
                'year': year,
                'journal': journal,
                'volume': volume,
                'issue': issue,
                'pages': pages,
                'pdf_url': url,  # DOI resolves to PDF or article page
                'human_url': human_url,  # Publisher's human-readable URL
                'bibtex_url': f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex",
                'source': 'crossref',
            }

            # Save document
            _doc_id, _doc_rev = self.db.save(doc)

        # Download PDF using TDM links, pypaperretriever, or paperscraper
        attachment_data = self._download_pdf_with_pypaperretriever(doi, work)

        if attachment_data:
            content, filename, content_type = attachment_data
            # Attach content to document
            if self.verbosity >= 3:
                print(f"  Attaching {filename} ({len(content)} bytes)")

            self.db.put_attachment(
                doc,
                BytesIO(content),
                filename,
                content_type
            )

            if self.verbosity >= 2:
                print(f"  Successfully ingested with {filename}")
        else:
            if self.verbosity >= 2:
                print(f"  Could not download PDF or XML")

    def _download_pdf_with_pypaperretriever(self, doi: str, work: Dict[str, Any]) -> Optional[tuple]:
        """
        Download PDF using TDM links, pypaperretriever, or paperscraper fallback.

        Args:
            doi: DOI of the article
            work: Crossref work dictionary

        Returns:
            Tuple of (content_bytes, filename, content_type) or None if all methods fail
        """
        # Apply rate limiting before PDF download
        self._apply_rate_limit()

        # Construct DOI URL
        doi_url = f"https://doi.org/{doi}"

        # Try TDM link first (text-mining or tdm intended application)
        content = self._try_tdm_link(work)

        # If TDM failed, try citation_pdf_url from article landing page
        if content is None:
            content = self._try_citation_pdf_url(work, doi_url)

        # If citation_pdf_url failed, try pypaperretriever
        if content is None:
            content = self._try_pypaperretriever(doi)

        # If pypaperretriever failed, try paperscraper fallback
        if content is None:
            if self.verbosity >= 2:
                print("  Trying paperscraper fallback...")
            content = self._try_paperscraper(doi)

        if content is None:
            return None

        # Detect content type based on magic bytes
        filename, content_type = self._detect_content_type(content)

        if self.verbosity >= 3:
            print(f"  Detected content type: {content_type}")

        return (content, filename, content_type)

    def _try_tdm_link(self, work: Dict[str, Any]) -> Optional[bytes]:
        """
        Try to download from TDM (Text and Data Mining) link in Crossref record.

        Args:
            work: Crossref work dictionary

        Returns:
            Content as bytes, or None if no TDM link or download failed
        """
        # Check for links with TDM intended application
        if 'link' not in work:
            return None

        links = work['link']
        if not isinstance(links, list):
            return None

        # Look for TDM or text-mining link
        tdm_url = None
        for link in links:
            if not isinstance(link, dict):
                continue

            intended_app = link.get('intended-application', '')
            if intended_app in ('tdm', 'text-mining'):
                tdm_url = link.get('URL')
                if tdm_url:
                    if self.verbosity >= 2:
                        print(f"  Found TDM link: {tdm_url}")
                    break

        if not tdm_url:
            if self.verbosity >= 3:
                print(f"  No TDM link found in Crossref record")
            return None

        # Try to download from TDM URL
        try:
            if self.verbosity >= 3:
                print(f"  Downloading from TDM URL...")

            response = self._get_with_rate_limit(tdm_url, stream=False)
            self.last_fetch_time = time.time()

            if response.status_code == 200:
                content = response.content
                if self.verbosity >= 3:
                    print(f"  Downloaded {len(content)} bytes from TDM link")
                return content
            else:
                if self.verbosity >= 3:
                    print(f"  TDM download failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            if self.verbosity >= 3:
                print(f"  TDM download failed: {e}")
            return None

    def _try_citation_pdf_url(self, work: Dict[str, Any], doi_url: str) -> Optional[bytes]:
        """
        Try to extract and download from citation_pdf_url meta tag on article page.

        Args:
            work: Crossref work dictionary
            doi_url: DOI resolution URL (e.g., https://doi.org/...)

        Returns:
            PDF content as bytes, or None if extraction or download failed
        """
        # Check if BeautifulSoup is available
        if BeautifulSoup is None:
            return None

        # Get the article landing page URL
        landing_url = self._extract_human_url(work, doi_url)
        if not landing_url or landing_url == doi_url:
            # No separate landing page, skip
            return None

        try:
            if self.verbosity >= 3:
                print(f"  Checking for citation_pdf_url at: {landing_url}")

            # Fetch the landing page HTML
            response = self._get_with_rate_limit(landing_url, stream=False)
            if response.status_code != 200:
                if self.verbosity >= 3:
                    print(f"  Failed to fetch landing page: HTTP {response.status_code}")
                return None

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for citation_pdf_url meta tag
            pdf_meta = soup.find('meta', attrs={'name': 'citation_pdf_url'})
            if not pdf_meta or not pdf_meta.get('content'):
                if self.verbosity >= 3:
                    print("  No citation_pdf_url meta tag found")
                return None

            pdf_url = str(pdf_meta['content'])
            if self.verbosity >= 2:
                print(f"  Found citation_pdf_url: {pdf_url}")

            # Download from the citation_pdf_url
            pdf_response = self._get_with_rate_limit(pdf_url, stream=False)
            self.last_fetch_time = time.time()

            if pdf_response.status_code == 200:
                content = pdf_response.content
                if self.verbosity >= 3:
                    print(f"  Downloaded {len(content)} bytes from citation_pdf_url")
                return content
            else:
                if self.verbosity >= 3:
                    print(f"  citation_pdf_url download failed: HTTP {pdf_response.status_code}")
                return None

        except Exception as e:
            if self.verbosity >= 3:
                print(f"  citation_pdf_url extraction/download failed: {e}")
            return None

    def _try_pypaperretriever(self, doi: str) -> Optional[bytes]:
        """
        Try to download PDF using pypaperretriever with exponential backoff for 429 errors.

        Args:
            doi: DOI of the article

        Returns:
            PDF content as bytes, or None if download failed
        """
        def _download_with_habanero():
            """Inner function that performs the actual download."""
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix='crossref_ingest_')
            temp_path = Path(temp_dir)

            try:
                if self.verbosity >= 3:
                    print(f"  Using temp directory: {temp_dir}")

                # Download using pypaperretriever
                retriever = PaperRetriever(
                    email=self.mailto,
                    doi=doi,
                    download_directory=str(temp_path),
                    allow_scihub=self.allow_scihub
                )

                retriever.download()

                # Find the downloaded PDF file (search recursively)
                pdf_files = list(temp_path.glob('**/*.pdf'))

                if not pdf_files:
                    if self.verbosity >= 3:
                        print("  No PDF file found in temp directory")
                    return None

                # Read the first PDF file found
                pdf_file = pdf_files[0]
                if self.verbosity >= 3:
                    print(f"  Reading PDF from: {pdf_file.name}")

                with open(pdf_file, 'rb') as f:
                    return f.read()

            finally:
                # Always clean up temp directory
                try:
                    shutil.rmtree(temp_dir)
                    if self.verbosity >= 3:
                        print("  Cleaned up temp directory")
                except Exception as e:
                    if self.verbosity >= 1:
                        print(f"  WARNING: Failed to clean up temp directory {temp_dir}: {e}")

        # Use the base class retry wrapper
        return self._retry_with_backoff(
            _download_with_habanero,
            operation_name="pypaperretriever"
        )

    def _try_paperscraper(self, doi: str) -> Optional[bytes]:
        """
        Try to download PDF using paperscraper fallback with exponential backoff for 429 errors.

        Args:
            doi: DOI of the article

        Returns:
            Content as bytes (could be PDF or XML), or None if download failed
        """
        # Check if paperscraper is available
        if save_pdf is None:
            return None

        def _download_with_paperscraper():
            """Inner function that performs the actual download."""
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix='crossref_paperscraper_')
            temp_path = Path(temp_dir)

            try:
                if self.verbosity >= 3:
                    print(f"  Using paperscraper temp directory: {temp_dir}")

                # save_pdf returns the path to the downloaded file
                output_path = save_pdf(doi, filepath=str(temp_path))

                if output_path and Path(output_path).exists():
                    if self.verbosity >= 3:
                        print(f"  paperscraper downloaded: {output_path}")

                    with open(output_path, 'rb') as f:
                        return f.read()
                else:
                    if self.verbosity >= 3:
                        print("  paperscraper returned no file")
                    return None

            finally:
                # Always clean up temp directory
                try:
                    shutil.rmtree(temp_dir)
                    if self.verbosity >= 3:
                        print("  Cleaned up paperscraper temp directory")
                except Exception as e:
                    if self.verbosity >= 1:
                        print(f"  WARNING: Failed to clean up temp directory {temp_dir}: {e}")

        # Use the base class retry wrapper
        return self._retry_with_backoff(
            _download_with_paperscraper,
            operation_name="paperscraper"
        )

    def _detect_content_type(self, content: bytes) -> tuple:
        """
        Detect if content is PDF or XML based on magic bytes.

        Args:
            content: File content as bytes

        Returns:
            Tuple of (filename, content_type)
        """
        # Check for PDF magic bytes (%PDF)
        if content.startswith(b'%PDF'):
            return ('article.pdf', 'application/pdf')

        # Check for XML declaration or common XML start tags
        if (content.startswith(b'<?xml') or
            content.startswith(b'<article') or
            content.startswith(b'<xml') or
            content.lstrip().startswith(b'<?xml') or
            content.lstrip().startswith(b'<article')):
            return ('article.xml', 'application/xml')

        # Default to PDF if uncertain
        if self.verbosity >= 2:
            print(f"  Warning: Could not detect content type, defaulting to PDF")
        return ('article.pdf', 'application/pdf')

    def _format_authors(self, authors: list) -> str:
        """
        Format author list from Crossref data.

        Args:
            authors: List of author dictionaries from Crossref

        Returns:
            Formatted author string
        """
        if not authors:
            return ''

        author_names = []
        for author in authors:
            given = author.get('given', '')
            family = author.get('family', '')
            if family:
                name = f"{given} {family}".strip() if given else family
                author_names.append(name)

        return '; '.join(author_names)

    def _extract_year(self, work: Dict[str, Any]) -> str:
        """
        Extract publication year from Crossref work.

        Args:
            work: Work dictionary from Crossref

        Returns:
            Year as string, or empty string if not found
        """
        # Try different date fields
        for field in ['published-print', 'published-online', 'created']:
            if field in work:
                date_parts = work[field].get('date-parts', [[]])[0]
                if date_parts and len(date_parts) > 0:
                    return str(date_parts[0])

        return ''

    def _extract_human_url(self, work: Dict[str, Any], doi_url: str) -> str:
        """
        Extract human-readable URL from Crossref work record.

        Tries in order:
        1. resource.primary.URL field
        2. links list entry with content-type "text/html"
        3. Falls back to DOI URL

        Args:
            work: Work dictionary from Crossref
            doi_url: Fallback DOI URL (https://doi.org/...)

        Returns:
            Human-readable URL string
        """
        # Try resource.primary.URL first
        if 'resource' in work:
            resource = work['resource']
            if 'primary' in resource and 'URL' in resource['primary']:
                url = resource['primary']['URL']
                if url:
                    if self.verbosity >= 3:
                        print(f"  Using resource URL: {url}")
                    return url

        # Try links list for text/html content-type
        if 'link' in work:
            links = work['link']
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, dict):
                        content_type = link.get('content-type', '')
                        if 'text/html' in content_type and 'URL' in link:
                            url = link['URL']
                            if url:
                                if self.verbosity >= 3:
                                    print(f"  Using link URL: {url}")
                                return url

        # Fall back to DOI URL
        if self.verbosity >= 3:
            print(f"  Using DOI URL: {doi_url}")
        return doi_url

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL from base dictionary.

        For Crossref, the PDF URL is the DOI resolution URL.

        Args:
            base: Dictionary containing 'url' key

        Returns:
            PDF URL string
        """
        return base.get('url', '')

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL from base dictionary.

        For Crossref, this is the DOI resolution URL.

        Args:
            base: Dictionary containing 'url' key

        Returns:
            Human-readable URL string
        """
        return base.get('url', '')

    def format_bibtex_url(self, base: Dict[str, str], bibtex_link: str) -> str:
        """
        Format BibTeX URL from base dictionary.

        For Crossref, we construct the BibTeX API URL from the DOI.

        Args:
            base: Dictionary containing 'doi' key
            bibtex_link: Unused for Crossref

        Returns:
            Crossref BibTeX transformation API URL
        """
        doi = base.get('doi', '')
        if doi:
            return f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
        return ''
