"""
LocalMykowebJournalsIngestor for ingesting from local Mykoweb journals mirror.

This module provides the LocalMykowebJournalsIngestor class for ingesting
journal PDF files from a local mirror of https://mykoweb.com/systematics/journals/
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .ingestor import Ingestor
from .publications import PublicationRegistry


class LocalMykowebJournalsIngestor(Ingestor):
    """
    Ingestor for local mirror of Mykoweb journals.

    This class handles ingestion from local PDF files that mirror the
    Mykoweb systematics journals directory structure.

    Path mapping:
        /data/skol/www/mykoweb.com/systematics/journals/ ->
        https://mykoweb.com/systematics/journals/

    Supported journals:
        - Mycotaxon
        - Persoonia
        - Sydowia

    ISSN metadata is retrieved from PublicationRegistry.SOURCES.

    File naming patterns:
        - "Mycotaxon v001n1.pdf" -> volume=1, number=1
        - "Mycotaxon v077.pdf" -> volume=77, number=1 (defaults to 1)
        - "Persoonia v01n2.pdf" -> volume=1, number=2
        - "Sydowia Vol20.pdf" -> volume=20, number=None
        - "Sydowia Vol22(1-4).pdf" -> volume=22, number="1-4"
    """

    # Default root for local mirror
    DEFAULT_ROOT = Path('/data/skol/www/mykoweb.com/systematics/journals')

    def __init__(
        self,
        root: Optional[Path] = None,
        local_path_prefix: str = '/data/skol/www/mykoweb.com/systematics/journals',
        url_prefix: str = 'https://mykoweb.com/systematics/journals',
        **kwargs: Any
    ) -> None:
        """
        Initialize the LocalMykowebJournalsIngestor.

        Args:
            root: Root directory to search for journal PDFs (default: DEFAULT_ROOT)
            local_path_prefix: Local path prefix to replace
            url_prefix: URL prefix to use as replacement
            **kwargs: Base class arguments (db, user_agent, robot_parser, etc.)
        """
        super().__init__(**kwargs)
        self.root = root if root is not None else self.DEFAULT_ROOT
        self.local_path_prefix = local_path_prefix
        self.url_prefix = url_prefix

    def ingest(self) -> None:
        """
        Perform the ingestion operation.

        Ingests data from the local journals directory specified in the constructor.
        """
        self.ingest_from_local_journals(
            root=self.root,
            local_path_prefix=self.local_path_prefix,
            url_prefix=self.url_prefix
        )

    def _get_journal_metadata(self, journal_name: str) -> Optional[Dict[str, str]]:
        """
        Get ISSN/eISSN metadata for a journal from SOURCES.

        Args:
            journal_name: Name of the journal (e.g., 'Mycotaxon', 'Persoonia', 'Sydowia')

        Returns:
            Dictionary with 'issn' and 'eissn' keys, or None if not found
        """
        # Map journal directory names to publication keys
        journal_key_map = {
            'Mycotaxon': 'mycotaxon',
            'Persoonia': 'persoonia',
            'Sydowia': 'sydowia-ia',
        }

        key = journal_key_map.get(journal_name)
        if not key:
            return None

        source_config = PublicationRegistry.SOURCES.get(key)
        if not source_config:
            return None

        return {
            'issn': source_config.get('issn', ''),
            'eissn': source_config.get('eissn', ''),
        }

    # Filename patterns
    # Pattern 1: "Journal vXXXnY.pdf" (Mycotaxon, Persoonia with number)
    PATTERN_VN = re.compile(
        r'^(?P<journal>\w+)\s+v(?P<volume>\d+)n(?P<number>\d+)\.pdf$',
        re.IGNORECASE
    )

    # Pattern 2: "Journal vXXX.pdf" (Mycotaxon, Persoonia without number)
    PATTERN_V = re.compile(
        r'^(?P<journal>\w+)\s+v(?P<volume>\d+)\.pdf$',
        re.IGNORECASE
    )

    # Pattern 3: "Journal VolXX.pdf" or "Journal VolXX(stuff).pdf" (Sydowia)
    PATTERN_VOL = re.compile(
        r'^(?P<journal>\w+)\s+Vol(?P<volume>\d+)(?:\((?P<number>[^)]+)\))?\.pdf$',
        re.IGNORECASE
    )

    def parse_filename(
        self,
        filename: str,
        journal_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a PDF filename to extract volume and number information.

        Args:
            filename: The PDF filename (e.g., "Mycotaxon v001n1.pdf")
            journal_name: The journal name from the directory

        Returns:
            Dictionary with journal, volume, and number, or None if unparseable
        """
        # Try "Journal vXXXnY.pdf" pattern (with number)
        match = self.PATTERN_VN.match(filename)
        if match:
            return {
                'journal': journal_name,
                'volume': str(int(match.group('volume'))),  # Remove leading zeros
                'number': str(int(match.group('number'))),  # Remove leading zeros
            }

        # Try "Journal vXXX.pdf" pattern (without number - default to 1)
        match = self.PATTERN_V.match(filename)
        if match:
            return {
                'journal': journal_name,
                'volume': str(int(match.group('volume'))),  # Remove leading zeros
                'number': '1',  # Default to issue 1 when not specified
            }

        # Try "Journal VolXX(Y).pdf" pattern
        match = self.PATTERN_VOL.match(filename)
        if match:
            result = {
                'journal': journal_name,
                'volume': str(int(match.group('volume'))),
            }
            # Number is optional (may be in parentheses or absent)
            number = match.group('number')
            if number:
                result['number'] = number  # Keep as-is (e.g., "1-4")
            return result

        # Unparseable filename (e.g., "contents" files)
        return None

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL - same as the base url for Mykoweb PDFs.

        Args:
            base: Dictionary containing the 'url' field

        Returns:
            The PDF URL (same as url)
        """
        return base['url']

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL - same as the base url.

        Args:
            base: Dictionary containing the 'url' field

        Returns:
            The human-readable URL (same as url)
        """
        return base['url']

    def ingest_from_local_journals(
        self,
        root: Path = DEFAULT_ROOT,
        local_path_prefix: str = '/data/skol/www/mykoweb.com/systematics/journals',
        url_prefix: str = 'https://mykoweb.com/systematics/journals'
    ) -> None:
        """
        Ingest from local Mykoweb journals directory.

        This method walks through journal subdirectories, parses PDF filenames,
        and creates documents with proper metadata.

        Args:
            root: Root directory to search for journal PDFs
                 (default: /data/skol/www/mykoweb.com/systematics/journals)
            local_path_prefix: Local path prefix to replace
            url_prefix: URL prefix to use as replacement
        """
        # Walk through journal subdirectories
        for journal_dir in os.listdir(root):
            journal_path = root / journal_dir

            # Skip if not a directory
            if not journal_path.is_dir():
                continue

            # Get journal metadata
            journal_metadata = self._get_journal_metadata(journal_dir)
            if not journal_metadata:
                if self.verbosity >= 1:
                    print(f"Warning: Unknown journal directory '{journal_dir}', skipping")
                continue

            if self.verbosity >= 2:
                print(f"Processing journal: {journal_dir}")

            # Process each PDF in the journal directory
            for filename in os.listdir(journal_path):
                # Skip non-PDF files
                if not filename.endswith('.pdf'):
                    continue

                # Parse filename
                parsed = self.parse_filename(filename, journal_dir)
                if not parsed:
                    if self.verbosity >= 1:
                        print(f"  Warning: Could not parse filename '{filename}', skipping")
                    continue

                # Build full file path and URL
                full_filepath = journal_path / filename
                relative_path = str(full_filepath)[len(local_path_prefix):]
                file_url = f"{url_prefix}{relative_path}"

                # Create document dictionary
                doc_dict = {
                    'url': file_url,
                    'journal': parsed['journal'],
                    'volume': parsed['volume'],
                    'issn': journal_metadata['issn'],
                    'eissn': journal_metadata['eissn'],
                    'itemtype': 'article',
                }

                # Add number if present
                if 'number' in parsed:
                    doc_dict['number'] = parsed['number']

                # Create metadata
                meta = {
                    'source': 'mykoweb',
                    'journal': journal_dir,
                }

                if self.verbosity >= 3:
                    print(f"  Processing: {filename}")
                    print(f"    Volume: {parsed['volume']}, Number: {parsed.get('number', 'N/A')}")
                    print(f"    URL: {file_url}")

                # Ingest single document
                self._ingest_documents(
                    documents=[doc_dict],
                    meta=meta,
                    bibtex_link=file_url  # Use PDF URL as bibtex_link
                )

                if self.verbosity >= 3:
                    print()
