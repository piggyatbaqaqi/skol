"""
NaturalisIngestor for the Naturalis Biodiversity Center repository.

https://repository.naturalis.nl/ hosts open-access full text for the
journals Naturalis publishes or archives: Persoonia, Blumea, Zoologische
Mededelingen, Gorteria, Flora Malesiana Bulletin, and many others.  One
ingestor class serves all of them; a ``SOURCES`` row picks the journal by
its repository collection id (Persoonia is ``'1'``).

Navigation, chosen to respect robots.txt (``Crawl-delay: 2``;
``Disallow: /col/*`` rules out the collection pages and their RSS):

1. POST ``/search/query`` with a ``journal:<collection_id>`` filter.  The
   JSON reply holds ten hits per page and a ``paging.next`` cursor; follow
   it until ``paging.next.disabled``.
2. GET each hit's ``/pub/<id>/`` page and read its Highwire
   ``citation_*`` meta tags, which include ``citation_pdf_url``.
3. Hand the article to the base class, which stores it and attaches
   the PDF.
"""

from typing import Any, Dict, Iterator, List, Optional

import requests  # noqa: F401  (used once implemented)
from bs4 import BeautifulSoup

from .ingestor import Ingestor


class NaturalisIngestor(Ingestor):
    """Ingestor for one journal in repository.naturalis.nl."""

    BASE_URL = 'https://repository.naturalis.nl'
    SEARCH_PATH = '/search/query'

    collection_id: str
    base_url: str
    limit: Optional[int]
    dry_run: bool

    def __init__(
        self,
        collection_id: str = '1',
        base_url: Optional[str] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the NaturalisIngestor.

        Args:
            collection_id: Repository journal id, as in the search filter
                ``journal:<id>`` (``'1'`` is Persoonia).
            base_url: Repository root (default: BASE_URL).
            limit: Stop after this many articles have been handed to the
                base class (None for no limit).
            dry_run: Parse and report, but write nothing to CouchDB.
            **kwargs: Base class arguments (db, user_agent, robot_parser...)
        """
        super().__init__(**kwargs)
        self.collection_id = str(collection_id)
        self.base_url = (base_url or self.BASE_URL).rstrip('/')
        self.limit = limit
        self.dry_run = dry_run

    def ingest(self) -> None:
        """Ingest every article of the configured journal."""
        raise NotImplementedError

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """Return the PDF URL taken from ``citation_pdf_url``."""
        raise NotImplementedError

    def format_human_url(self, base: Dict[str, str]) -> str:
        """Return the repository's ``/pub/<id>/`` landing page."""
        raise NotImplementedError

    def search_url(self) -> str:
        """Return the absolute URL of the JSON search endpoint."""
        raise NotImplementedError

    def build_search_body(self, start: int) -> Dict[str, Any]:
        """
        Build the JSON body for one page of search results.

        Args:
            start: The ``from`` cursor (1 for the first page, then the
                value of ``paging.next.from`` from the previous reply).

        Returns:
            A request body filtered to this journal and sorted by year,
            oldest first, so that paging is stable.
        """
        raise NotImplementedError

    def _post_search(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        POST one search request, rate limited and robots-checked.

        Returns:
            The decoded JSON reply, or None if robots.txt forbids the
            endpoint or the request fails.
        """
        raise NotImplementedError

    def iter_search_hits(self) -> Iterator[Dict[str, Any]]:
        """
        Yield every search hit for this journal, each once.

        Follows ``paging.next.from`` until the reply marks it disabled,
        the reply has no hits, or a request fails.
        """
        raise NotImplementedError

    def parse_pub_page(
        self,
        soup: BeautifulSoup,
        pub_url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract article metadata from a ``/pub/<id>/`` page.

        Args:
            soup: The parsed landing page.
            pub_url: The page's URL, stored as the human URL.

        Returns:
            An article dict (title, author, journal, volume, number,
            pages, year, publication_date, issn, doi, abstract,
            keywords, pdf_url, url, naturalis_id, itemtype), or None
            when the page offers no PDF.
        """
        raise NotImplementedError

    @staticmethod
    def _meta_values(soup: BeautifulSoup, name: str) -> List[str]:
        """Return the stripped content of every ``<meta name=name>``."""
        raise NotImplementedError

    @staticmethod
    def _parse_citation_date(value: Optional[str]) -> Optional[str]:
        """Convert ``YYYY/MM/DD`` (or ``YYYY``) to ISO ``YYYY-MM-DD``."""
        raise NotImplementedError
