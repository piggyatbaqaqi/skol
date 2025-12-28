"""
LocalMykowebLiteratureIngestor for ingesting literature from local Mykoweb mirror.

This module provides the LocalMykowebLiteratureIngestor class for ingesting
literature/book PDF files from a local mirror of Mykoweb.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .ingestor import Ingestor


class LocalMykowebLiteratureIngestor(Ingestor):
    """
    Ingestor for local mirror of Mykoweb literature/books.

    This class handles ingestion from local PDF files representing books
    or other literature from the Mykoweb collection. It uses the filename
    (without .pdf extension) as the title.

    Path mapping:
        /data/skol/www/mykoweb.com/systematics/literature/ ->
        https://mykoweb.com/systematics/literature/

    File naming:
        Any .pdf file - the basename becomes the title
        Example: "Introduction to Mycology.pdf" -> title="Introduction to Mycology"
    """

    # Default root for local mirror
    DEFAULT_ROOT = Path('/data/skol/www/mykoweb.com/systematics/literature')

    def __init__(
        self,
        root: Optional[Path] = None,
        local_path_prefix: str = '/data/skol/www/mykoweb.com/systematics/literature',
        url_prefix: str = 'https://mykoweb.com/systematics/literature',
        **kwargs: Any
    ) -> None:
        """
        Initialize the LocalMykowebLiteratureIngestor.

        Args:
            root: Root directory to search for literature PDFs (default: DEFAULT_ROOT)
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

        Ingests data from the local literature directory specified in the constructor.
        """
        self.ingest_from_local_literature(
            root=self.root,
            local_path_prefix=self.local_path_prefix,
            url_prefix=self.url_prefix
        )

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

    def ingest_from_local_literature(
        self,
        root: Path = DEFAULT_ROOT,
        local_path_prefix: str = '/data/skol/www/mykoweb.com/systematics/literature',
        url_prefix: str = 'https://mykoweb.com/systematics/literature'
    ) -> None:
        """
        Ingest from local Mykoweb literature directory.

        This method walks through the directory tree, finds all PDF files,
        and creates documents using the filename as the title.

        Args:
            root: Root directory to search for literature PDFs
                 (default: /data/skol/www/mykoweb.com/systematics/literature)
            local_path_prefix: Local path prefix to replace
            url_prefix: URL prefix to use as replacement
        """
        if self.verbosity >= 2:
            print(f"Processing literature from: {root}")

        # Walk through all subdirectories
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                # Skip non-PDF files
                if not filename.endswith('.pdf'):
                    continue

                # Build full file path and URL
                full_filepath = os.path.join(dirpath, filename)
                relative_path = full_filepath[len(local_path_prefix):]
                file_url = f"{url_prefix}{relative_path}"

                # Extract title from filename (remove .pdf extension)
                title = filename[:-4]  # Remove '.pdf'

                # Create document dictionary
                doc_dict = {
                    'url': file_url,
                    'title': title,
                    'itemtype': 'book',
                }

                # Create metadata
                meta = {
                    'source': 'mykoweb',
                    'type': 'literature',
                }

                if self.verbosity >= 3:
                    print(f"  Processing: {filename}")
                    print(f"    Title: {title}")
                    print(f"    URL: {file_url}")

                # Ingest single document
                self._ingest_documents(
                    documents=[doc_dict],
                    meta=meta,
                    bibtex_link=file_url  # Use PDF URL as bibtex_link
                )

                if self.verbosity >= 3:
                    print()
