"""
LocalIngentaIngestor for ingesting from local mirror of IngentaConnect.

This module provides the LocalIngentaIngestor class for ingesting data from
local BibTeX files that mirror the IngentaConnect website structure.
"""

from pathlib import Path

from .ingenta import IngentaIngestor


class LocalIngentaIngestor(IngentaIngestor):
    """
    Ingestor for local mirror of IngentaConnect data.

    This class handles ingestion from local BibTeX files that mirror the
    IngentaConnect website structure, converting local file paths to their
    corresponding web URLs.

    Path mapping:
        /data/skol/www/www.ingentaconnect.com/ ->
        https://www.ingentaconnect.com/

    The base Ingestor.ingest_from_local_bibtex() method strips the root
    path from each file path and prepends the url_prefix to construct the
    bibtex_link.

    Example:
        root: /data/skol/www/www.ingentaconnect.com
        file: /data/skol/www/www.ingentaconnect.com/content/mtax/mt/...
        relative_path: /content/mtax/mt/format=bib
        bibtex_link: https://www.ingentaconnect.com/content/mtax/mt/...
    """

    # Default root for local mirror
    DEFAULT_ROOT = Path('/data/skol/www/www.ingentaconnect.com')

    def ingest_from_local_bibtex(
        self,
        root: Path = DEFAULT_ROOT,
        bibtex_file_pattern: str = 'format=bib',
        url_prefix: str = 'https://www.ingentaconnect.com'
    ) -> None:
        """
        Ingest from a local directory mirroring IngentaConnect structure.

        This method converts local file paths to their corresponding web
        URLs by stripping the root path and prepending the URL prefix.

        Args:
            root: Root directory to search for BibTeX files
                 (default: /data/skol/www/www.ingentaconnect.com)
            bibtex_file_pattern: Filename pattern to match BibTeX files
                                (default: format=bib)
            url_prefix: URL prefix to prepend to relative paths
                       (default: https://www.ingentaconnect.com)
        """
        # Call parent which calls Ingestor.ingest_from_local_bibtex
        # Override url_prefix to exclude trailing slash
        super().ingest_from_local_bibtex(
            root=root,
            bibtex_file_pattern=bibtex_file_pattern,
            url_prefix=url_prefix
        )
