"""
Base class for ingesting web data into CouchDB.

This module provides the abstract Ingestor base class that defines the
interface for different data source ingestors.
"""

import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.robotparser import RobotFileParser

import bibtexparser
import couchdb
import feedparser
import requests
from uuid import uuid4


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

    def __init__(
        self,
        db: couchdb.Database,
        user_agent: str,
        robot_parser: RobotFileParser,
        verbosity: int = 2
    ) -> None:
        """
        Initialize the Ingestor.

        Args:
            db: CouchDB database instance for storing documents
            user_agent: User agent string for HTTP requests
            robot_parser: Robot file parser for checking crawl permissions
            verbosity: Verbosity level (0=silent, 1=warnings, 2=normal, 3=verbose)
        """
        self.db = db
        self.user_agent = user_agent
        self.robot_parser = robot_parser
        self.verbosity = verbosity


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
            doc = {
                '_id': uuid4().hex,
                'meta': meta,
                'bibtex_url': self.format_bibtex_url(doc_dict, bibtex_link),
                'pdf_url': self.format_pdf_url(doc_dict),
                'human_url': self.format_human_url(doc_dict),
            }

            # Add all fields from source dictionary
            for k, v in doc_dict.items():
                doc[k] = v

            # Check if document already exists
            selector = {'selector': {'pdf_url': doc['pdf_url']}}
            found = False
            for _ in self.db.find(selector):
                found = True
                break
            if found:
                if self.verbosity >= 2:
                    print(f"Skipping {doc['pdf_url']}")
                continue

            # Check robot permissions
            if not self.robot_parser.can_fetch(self.user_agent, doc['pdf_url']):
                # TODO(piggy): We should probably log blocked URLs.
                if self.verbosity >= 1:
                    print(f"Robot permission denied {doc['pdf_url']}")
                continue

            if self.verbosity >= 2:
                print(f"Adding {doc['pdf_url']}")

            # Save document to CouchDB
            _doc_id, _doc_rev = self.db.save(doc)

            # Fetch and attach PDF
            with requests.get(doc['pdf_url'], stream=False) as pdf_f:
                pdf_f.raise_for_status()
                pdf_doc = pdf_f.content

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

        feed_meta = {
            'url': rss_url,
            'title': feed.feed.title,
            'link': feed.feed.link,
            'description': feed.feed.description,
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

            # Fetch BibTeX file
            with requests.get(bibtex_link, stream=False) as bibtex_f:
                bibtex_f.raise_for_status()

                self.ingest_from_bibtex(
                    content=bibtex_f.content,
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
