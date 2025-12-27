"""
IngentaConnect-specific ingestor implementation.

This module provides the IngentaIngestor class for ingesting data from
IngentaConnect RSS feeds and local BibTeX files.
"""

from pathlib import Path
from typing import Dict

from .ingestor import Ingestor


class IngentaIngestor(Ingestor):
    """
    Ingestor specialized for IngentaConnect data.

    Handles Ingenta-specific URL formatting and BibTeX content transformations.
    """

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """
        Format PDF URL for Ingenta with crawler parameter.

        Args:
            base_url: The base URL from the BibTeX entry

        Returns:
            URL with ?crawler=true appended
        """
        return f"{base['url']}?crawler=true"

    def format_human_url(self, base: Dict[str, str]) -> str:
        """
        Format human-readable URL for Ingenta.

        Args:
            base_url: The base URL from the BibTeX entry
        """
        return base['url']

    def transform_bibtex_content(self, content: bytes) -> bytes:
        """
        Fix Ingenta-specific BibTeX syntax issues.

        Ingenta BibTeX files have syntax problems that need correction:
        - Missing commas between url and parent_itemid fields
        - Missing commas before 'parent' field
        - Embedded newlines that break parsing

        Args:
            content: Raw BibTeX content

        Returns:
            Corrected BibTeX content
        """
        return (
            content
            # Fix url field running into parent_itemid field
            .replace(b"\"\nparent_itemid", b"\",\nparent_itemid")
            .replace(b"\"\\nparent_itemid", b"\",\\nparent_itemid")
            # Fix other parent field issues
            .replace(b"\"\\nparent", b"\",\\nparent")
            # Remove remaining embedded newlines
            .replace(b"\\n", b"")
        )

    def ingest_from_local_bibtex(
        self,
        root: Path,
        bibtex_file_pattern: str = 'format=bib',
        url_prefix: str = 'https://www.ingentaconnect.com/'
    ) -> None:
        """
        Ingest from a local directory containing Ingenta BibTeX files.

        Args:
            root: Root directory to search for BibTeX files
            bibtex_file_pattern: Filename pattern to match BibTeX files
            url_prefix: URL prefix for Ingenta (default: https://www.ingentaconnect.com/)
        """
        super().ingest_from_local_bibtex(
            root=root,
            bibtex_file_pattern=bibtex_file_pattern,
            url_prefix=url_prefix
        )
