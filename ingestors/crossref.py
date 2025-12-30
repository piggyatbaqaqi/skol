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
    from habanero import Crossref
except ImportError:
    raise ImportError("habanero library required. Install with: pip install habanero")

try:
    from pypaperretriever import PaperRetriever
except ImportError:
    raise ImportError("pypaperretriever library required. Install with: pip install pypaperretriever")

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
        **kwargs: Any
    ) -> None:
        """
        Initialize the CrossrefIngestor.

        Args:
            issn: ISSN or eISSN of the journal (with or without hyphen)
            mailto: Email address for Crossref API polite pool (default: piggy.yarroll+skol@gmail.com)
            max_articles: Maximum number of articles to ingest (None = all)
            allow_scihub: Whether to allow pypaperretriever to use Sci-Hub as fallback
            **kwargs: Additional parameters passed to Ingestor base class
        """
        super().__init__(**kwargs)
        self.issn = issn
        self.mailto = mailto
        self.max_articles = max_articles
        self.allow_scihub = allow_scihub

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
        per_page = 100  # Crossref allows up to 1000
        total_yielded = 0

        if self.verbosity >= 2:
            print("Fetching works from Crossref...")

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
                    if self.max_articles and total_yielded >= self.max_articles:
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

            # Rate limiting
            time.sleep(0.1)

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

        # Download PDF using pypaperretriever with temp directory
        pdf_content = self._download_pdf_with_pypaperretriever(doi)

        if pdf_content:
            # Attach PDF to document
            if self.verbosity >= 3:
                print(f"  Attaching PDF ({len(pdf_content)} bytes)")

            self.db.put_attachment(
                doc,
                BytesIO(pdf_content),
                'article.pdf',
                'application/pdf'
            )

            if self.verbosity >= 2:
                print(f"  Successfully ingested with PDF")
        else:
            if self.verbosity >= 2:
                print(f"  Could not download PDF")

    def _download_pdf_with_pypaperretriever(self, doi: str) -> Optional[bytes]:
        """
        Download PDF using pypaperretriever, then delete the file.

        Args:
            doi: DOI of the article

        Returns:
            PDF content as bytes, or None if download failed
        """
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

            try:
                retriever.download()
            except Exception as e:
                if self.verbosity >= 2:
                    print(f"  pypaperretriever failed: {e}")
                return None

            # Find the downloaded PDF file
            pdf_files = list(temp_path.glob('*.pdf'))

            if not pdf_files:
                if self.verbosity >= 3:
                    print(f"  No PDF file found in temp directory")
                return None

            # Read the first PDF file found
            pdf_file = pdf_files[0]
            if self.verbosity >= 3:
                print(f"  Reading PDF from: {pdf_file.name}")

            with open(pdf_file, 'rb') as f:
                pdf_content = f.read()

            return pdf_content

        finally:
            # Always clean up temp directory
            try:
                shutil.rmtree(temp_dir)
                if self.verbosity >= 3:
                    print(f"  Cleaned up temp directory")
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"  WARNING: Failed to clean up temp directory {temp_dir}: {e}")

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
