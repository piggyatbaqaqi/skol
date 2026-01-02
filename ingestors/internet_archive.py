"""
InternetArchiveIngestor for downloading journal issues from the Internet Archive.

This module provides the InternetArchiveIngestor class for downloading
and ingesting journal articles from Internet Archive collections using
the internetarchive library.
"""

import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from uuid import uuid5, NAMESPACE_URL

try:
    import internetarchive as ia
except ImportError as exc:
    raise ImportError(
        "internetarchive library required. "
        "Install with: pip install internetarchive"
    ) from exc

from .ingestor import Ingestor


class InternetArchiveIngestor(Ingestor):
    """
    Ingestor for journals from Internet Archive collections.

    This class downloads journal issues from the Internet Archive, extracting
    both PDF and XML (OCR text) files when available.

    Example collections:
        - Sydowia: pub_sydowia (https://archive.org/details/pub_sydowia)

    Files downloaded:
        - PDF: Main article content
        - djvu.xml: OCR text in XML format (when available)

    Metadata extracted:
        - title
        - volume
        - issue
        - date (publication date)
        - identifier (Internet Archive item ID)
    """

    def __init__(
        self,
        collection: Optional[str] = None,
        max_items: Optional[int] = None,
        download_xml: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize the InternetArchiveIngestor.

        Args:
            collection: Internet Archive collection identifier (e.g., 'pub_sydowia')
            max_items: Maximum number of items to download (None = all)
            download_xml: Whether to download XML files in addition to PDFs
            **kwargs: Base class arguments (db, user_agent, robot_parser, etc.)
        """
        super().__init__(**kwargs)
        self.collection = collection
        self.max_items = max_items
        self.download_xml = download_xml

    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Searches the Internet Archive collection and downloads all items,
        including PDFs and optionally XML files.
        """
        if not self.collection:
            raise ValueError("Collection identifier must be provided")

        if self.verbosity >= 1:
            print(f"Downloading from Internet Archive collection: {self.collection}")

        # Search for all items in the collection
        search_query = f'collection:{self.collection}'
        search_results = ia.search_items(search_query)

        if self.verbosity >= 2:
            print(f"Search query: {search_query}")

        # Process each item
        item_count = 0
        for item_result in search_results:
            if self.max_items and item_count >= self.max_items:
                if self.verbosity >= 1:
                    print(f"\nReached maximum item limit: {self.max_items}")
                break

            item_count += 1
            identifier = item_result['identifier']

            if self.verbosity >= 2:
                print(f"\n{'=' * 60}")
                print(f"[{item_count}] Processing item: {identifier}")
                print(f"{'=' * 60}")

            try:
                self._ingest_item(identifier)
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"  Error processing item {identifier}: {e}")
                if self.verbosity >= 3:
                    import traceback
                    traceback.print_exc()
                continue

        if self.verbosity >= 1:
            print(f"\n{'=' * 60}")
            print(f"Ingestion complete: {item_count} items processed")
            print(f"{'=' * 60}")

    def _ingest_item(self, identifier: str) -> None:
        """
        Ingest a single item from the Internet Archive.

        Args:
            identifier: Internet Archive item identifier
        """
        # Get item details
        item = ia.get_item(identifier)
        metadata = item.metadata

        if self.verbosity >= 2:
            title = metadata.get('title', 'N/A')
            print(f"  Title: {title}")
            print(f"  Volume: {metadata.get('volume', 'N/A')}")
            print(f"  Issue: {metadata.get('issue', 'N/A')}")
            print(f"  Date: {metadata.get('date', 'N/A')}")

        # Find PDF and XML files
        pdf_file = None
        xml_file = None

        for f in item.files:
            fname = f['name']
            # Main PDF file (not derivative files)
            excluded = ['_bw.pdf', '_text.pdf']
            if fname.endswith('.pdf') and not any(x in fname for x in excluded):
                pdf_file = f
            # DJVU XML file (OCR text)
            elif fname.endswith('_djvu.xml'):
                xml_file = f

        if not pdf_file:
            if self.verbosity >= 2:
                print("  No PDF file found, skipping")
            return

        if self.verbosity >= 3:
            pdf_size_mb = int(pdf_file.get('size', 0)) / 1024 / 1024
            print(f"  PDF file: {pdf_file['name']} ({pdf_size_mb:.2f} MB)")
            if xml_file:
                xml_size_mb = int(xml_file.get('size', 0)) / 1024 / 1024
                print(f"  XML file: {xml_file['name']} "
                      f"({xml_size_mb:.2f} MB)")

        # Prepare document metadata
        pdf_url = (f'https://archive.org/download/{identifier}/'
                   f'{pdf_file["name"]}')

        doc_metadata = {
            'title': metadata.get('title', ''),
            'itemtype': 'article',
            'url': f'https://archive.org/details/{identifier}',
            'pdf_url': pdf_url,
            'identifier': identifier,
        }

        # Add volume and issue if available
        if metadata.get('volume'):
            doc_metadata['volume'] = str(metadata['volume'])
        if metadata.get('issue'):
            doc_metadata['number'] = str(metadata['issue'])

        # Parse and add date
        date_str = metadata.get('date')
        if date_str:
            pub_date = self._parse_date(date_str)
            if pub_date:
                doc_metadata['publication_date'] = pub_date
                doc_metadata['year'] = self._extract_year(pub_date)

        # Add ISSN if available
        if metadata.get('issn'):
            doc_metadata['issn'] = metadata['issn']

        # Add publisher if available
        if metadata.get('publisher'):
            doc_metadata['publisher'] = metadata['publisher']

        # Add XML URL if available
        if xml_file:
            doc_metadata['xml_url'] = (
                f'https://archive.org/download/{identifier}/'
                f'{xml_file["name"]}'
            )

        # Create document ID from URL
        doc_id = uuid5(NAMESPACE_URL, pdf_url).hex

        # Create document with metadata
        doc = {
            '_id': doc_id,
            'pdf_url': pdf_url,
            'human_url': doc_metadata['url'],
        }

        # Add all fields from source dictionary
        for k, v in doc_metadata.items():
            doc[k] = v

        # Check if document already exists
        doc_exists = doc_id in self.db
        needs_pdf = True
        needs_xml = self.download_xml and xml_file is not None

        if doc_exists:
            # Document exists - check what attachments it has
            existing_doc = self.db[doc_id]
            attachments = existing_doc.get('_attachments', {})
            has_pdf = 'article.pdf' in attachments
            has_xml = 'article.xml' in attachments

            needs_pdf = not has_pdf
            needs_xml = needs_xml and not has_xml

            if not needs_pdf and not needs_xml:
                # Has everything we need
                if self.verbosity >= 2:
                    msg = "Skipping (already has PDF"
                    if self.download_xml and xml_file:
                        msg += " and XML"
                    msg += ")"
                    print(f"  {msg}")
                return

            # Use existing doc to preserve _rev
            doc = existing_doc

            if self.verbosity >= 2:
                parts = []
                if needs_pdf:
                    parts.append("PDF")
                if needs_xml:
                    parts.append("XML")
                print(f"  Adding {' and '.join(parts)} to existing record")
        else:
            # New document
            if self.verbosity >= 2:
                msg = "Adding new document"
                if self.download_xml and xml_file:
                    msg += " (with PDF and XML)"
                else:
                    msg += " (with PDF)"
                print(f"  {msg}")

        # Save document to CouchDB if it's new
        if not doc_exists:
            _doc_id, _doc_rev = self.db.save(doc)

        # Download and attach files
        with tempfile.TemporaryDirectory(prefix='ia_ingest_') as temp_dir:
            temp_path = Path(temp_dir)

            # Download and attach PDF if needed
            if needs_pdf:
                pdf_path = temp_path / pdf_file['name']
                if self.verbosity >= 2:
                    print("  Downloading PDF...")

                try:
                    item.download(
                        files=[pdf_file['name']],
                        destdir=str(temp_path),
                        verbose=self.verbosity >= 4,
                        no_directory=True
                    )
                except Exception as e:
                    if self.verbosity >= 1:
                        print(f"  Error downloading PDF: {e}")
                    return

                if not pdf_path.exists():
                    if self.verbosity >= 1:
                        msg = f"  PDF download failed: file not found at "
                        msg += f"{pdf_path}"
                        print(msg)
                    return

                # Attach PDF
                if self.verbosity >= 3:
                    print("  Attaching PDF to CouchDB...")

                with open(pdf_path, 'rb') as pdf_f:
                    pdf_content = BytesIO(pdf_f.read())
                    self.db.put_attachment(
                        doc,
                        pdf_content,
                        'article.pdf',
                        'application/pdf'
                    )

            # Download and attach XML if needed
            if needs_xml:
                xml_path = temp_path / xml_file['name']
                if self.verbosity >= 2:
                    print("  Downloading XML...")

                try:
                    item.download(
                        files=[xml_file['name']],
                        destdir=str(temp_path),
                        verbose=self.verbosity >= 4,
                        no_directory=True
                    )

                    if xml_path.exists():
                        # Attach XML
                        if self.verbosity >= 3:
                            print("  Attaching XML to CouchDB...")

                        with open(xml_path, 'rb') as xml_f:
                            xml_content = BytesIO(xml_f.read())
                            self.db.put_attachment(
                                doc,
                                xml_content,
                                'article.xml',
                                'application/xml'
                            )
                    else:
                        if self.verbosity >= 2:
                            msg = ("  Warning: XML download succeeded "
                                   "but file not found")
                            print(msg)

                except Exception as e:
                    if self.verbosity >= 2:
                        print(f"  Warning: Could not download XML: {e}")

    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse date string to ISO format (YYYY-MM-DD).

        Internet Archive dates can be in various formats:
        - YYYY-MM-DD
        - YYYY-MM
        - YYYY

        Args:
            date_str: Date string

        Returns:
            ISO formatted date or None
        """
        if not date_str:
            return None

        date_str = date_str.strip()

        # Try various date formats
        formats = [
            '%Y-%m-%d',  # 1903-01-15
            '%Y-%m',     # 1903-01
            '%Y',        # 1903
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue

        # If it's just a year, return as YYYY-01-01
        if date_str.isdigit() and len(date_str) == 4:
            return f"{date_str}-01-01"

        if self.verbosity >= 2:
            print(f"    Warning: Could not parse date '{date_str}'")
        return None

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

        # Extract YYYY from beginning of string
        if len(date_str) >= 4 and date_str[:4].isdigit():
            return date_str[:4]

        return None

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL from base metadata.

        Args:
            base: Dictionary containing metadata

        Returns:
            The PDF URL
        """
        return base.get('pdf_url', '')

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL (item page on archive.org).

        Args:
            base: Dictionary containing metadata

        Returns:
            The human-readable URL
        """
        return base.get('url', '')
