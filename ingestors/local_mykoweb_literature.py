"""
LocalMykowebLiteratureIngestor for ingesting literature from local Mykoweb mirror.

This module provides the LocalMykowebLiteratureIngestor class for ingesting
literature/book PDF files from a local mirror of Mykoweb.

Curated bibliographic metadata (journal, volume, issue, pages, author,
year, human-edited title) comes from the JSON file produced by
``bin/extract_mykoweb_pdf_metadata.py`` — see
docs/mykoweb_pdf_metadata_extraction.md.  When the JSON is missing or
a PDF isn't in it, the ingestor falls back to filename-as-title /
itemtype='book'.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .ingestor import Ingestor
from .mykoweb_metadata import (
    load_metadata_index,
    lookup_pdf_metadata,
    metadata_to_doc_fields,
)


class LocalMykowebLiteratureIngestor(Ingestor):
    """
    Ingestor for local mirror of Mykoweb literature/books.

    This class handles ingestion from local PDF files representing books
    or journal articles from the Mykoweb collection.  When a curated
    metadata record is available (from the JSON produced by
    ``bin/extract_mykoweb_pdf_metadata.py``) the ingested doc carries
    ``itemtype='article'`` plus journal / volume / issue / pages /
    author / year for journal articles, or ``itemtype='book'`` plus
    title / author / year for books.  Without a metadata record the
    ingestor falls back to filename-as-title / itemtype='book'.

    Path mapping:
        /data/skol/www/mykoweb.com/systematics/literature/ ->
        https://mykoweb.com/systematics/literature/
    """

    # Default root for local mirror
    DEFAULT_ROOT = Path('/data/skol/www/mykoweb.com/systematics/literature')

    # Default site root (parent of systematics/) — used to compute
    # the relative-to-site key for metadata lookup.
    DEFAULT_SITE_ROOT = '/data/skol/www/mykoweb.com'

    # Default metadata JSON path — matches the canonical output of
    # bin/extract_mykoweb_pdf_metadata.py.
    DEFAULT_METADATA_PATH = Path(
        '/data/skol/www/mykoweb.com/systematics_pdf_metadata.json'
    )

    def __init__(
        self,
        root: Optional[Path] = None,
        local_path_prefix: str = '/data/skol/www/mykoweb.com/systematics/literature',
        url_prefix: str = 'https://mykoweb.com/systematics/literature',
        site_root: Optional[str] = None,
        metadata_path: Optional[Path] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the LocalMykowebLiteratureIngestor.

        Args:
            root: Root directory to search for literature PDFs (default: DEFAULT_ROOT)
            local_path_prefix: Local path prefix to replace
            url_prefix: URL prefix to use as replacement
            site_root: Site root used for metadata-lookup key (default:
                       DEFAULT_SITE_ROOT)
            metadata_path: Path to the systematics_pdf_metadata.json
                       produced by bin/extract_mykoweb_pdf_metadata.py
                       (default: DEFAULT_METADATA_PATH).  Pass ``None``
                       to disable metadata lookup entirely.
            **kwargs: Base class arguments (db, user_agent, robot_parser, etc.)
        """
        super().__init__(**kwargs)
        self.root = root if root is not None else self.DEFAULT_ROOT
        self.local_path_prefix = local_path_prefix
        self.url_prefix = url_prefix
        self.site_root = (site_root if site_root is not None
                          else self.DEFAULT_SITE_ROOT)
        self.metadata_path = (metadata_path
                              if metadata_path is not None
                              else self.DEFAULT_METADATA_PATH)
        self._metadata_index: Dict[str, Dict[str, Any]] = (
            load_metadata_index(self.metadata_path)
        )
        if self.verbosity >= 1:
            if self._metadata_index:
                print(f"Loaded {len(self._metadata_index)} metadata records "
                      f"from {self.metadata_path}")
            else:
                print(f"No metadata index loaded (path={self.metadata_path}) "
                      f"— falling back to filename titles")

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

                # Start from the legacy filename-as-title / book
                # defaults; merge curated metadata if available.
                doc_dict: Dict[str, Any] = {
                    'url': file_url,
                    'title': filename[:-4],  # Strip .pdf
                    'itemtype': 'book',
                }
                record = lookup_pdf_metadata(
                    full_filepath, self._metadata_index, self.site_root,
                )
                if record is not None:
                    doc_dict.update(metadata_to_doc_fields(record))

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
