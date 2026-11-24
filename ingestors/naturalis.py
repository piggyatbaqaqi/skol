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

import re
from typing import Any, Dict, Iterator, List, Optional, Set

from bs4 import BeautifulSoup

from .ingestor import Ingestor


class NaturalisIngestor(Ingestor):
    """Ingestor for one journal in repository.naturalis.nl."""

    BASE_URL = 'https://repository.naturalis.nl'
    SEARCH_PATH = '/search/query'
    SEARCH_TIMEOUT_S = 60

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
        meta: Dict[str, Any] = {
            'source': 'naturalis',
            'collection_id': self.collection_id,
            'search_url': self.search_url(),
        }
        ingested = 0
        for hit in self.iter_search_hits():
            if self.limit is not None and ingested >= self.limit:
                if self.verbosity >= 2:
                    print(f"Reached limit of {self.limit}")
                break
            pub_url = hit.get('url') or f"{self.base_url}/pub/{hit['id']}/"
            soup = self._fetch_page(pub_url)
            if soup is None:
                continue
            article = self.parse_pub_page(soup, pub_url)
            if article is None:
                if self.verbosity >= 2:
                    print(f"  No PDF offered: {pub_url}")
                continue
            ingested += 1
            if self.dry_run:
                if self.verbosity >= 2:
                    print(f"[dry run] {article['pdf_url']} "
                          f"{article.get('title', '')[:60]}")
                continue
            self._ingest_documents([article], meta, pub_url)

    def format_pdf_url(self, base: Dict[str, str]) -> str:
        """Return the PDF URL taken from ``citation_pdf_url``."""
        return base['pdf_url']

    def format_human_url(self, base: Dict[str, str]) -> str:
        """Return the repository's ``/pub/<id>/`` landing page."""
        return base['url']

    def search_url(self) -> str:
        """Return the absolute URL of the JSON search endpoint."""
        return f'{self.base_url}{self.SEARCH_PATH}'

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
        return {
            'query': {
                'filters': {
                    'values': [
                        {'field_id': 'journal', 'query': self.collection_id},
                    ],
                },
                'sort': 'year_asc',
                'from': start,
            },
        }

    def _post_search(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        POST one search request through the rate-limited HTTP client.

        The client shares one per-domain clock between this POST and the
        pub-page and PDF GETs, so ``Crawl-delay`` holds across all three.

        Returns:
            The decoded JSON reply, or None if robots.txt forbids the
            endpoint or the request fails.
        """
        url = self.search_url()
        if not self.robot_parser.can_fetch(self.user_agent, url):
            if self.verbosity >= 1:
                print(f"Robot permission denied {url}")
            return None
        try:
            response = self.http_client.post(
                url,
                json=body,
                headers={'User-Agent': self.user_agent,
                         'Accept': 'application/json'},
                timeout=self.SEARCH_TIMEOUT_S,
            )
        except Exception as e:
            if self.verbosity >= 1:
                print(f"  Search request failed: {url} ({type(e).__name__})")
            return None
        if response.status_code != 200:
            if self.verbosity >= 1:
                print(f"  Search request failed: {url} "
                      f"(status code {response.status_code})")
            return None
        try:
            reply: Dict[str, Any] = response.json()
        except ValueError as e:
            if self.verbosity >= 1:
                print(f"  Search reply was not JSON: {url} ({e})")
            return None
        return reply

    def iter_search_hits(self) -> Iterator[Dict[str, Any]]:
        """
        Yield every search hit for this journal, each once.

        Follows ``paging.next.from`` until the reply marks it disabled,
        the reply has no hits, or a request fails.
        """
        seen: Set[str] = set()
        start = 1
        while True:
            reply = self._post_search(self.build_search_body(start))
            if reply is None:
                return
            hits = reply.get('hits') or []
            if not hits:
                return
            for hit in hits:
                hit_id = str(hit.get('id'))
                if hit_id in seen:
                    continue
                seen.add(hit_id)
                yield hit
            nxt = (reply.get('paging') or {}).get('next') or {}
            next_start = nxt.get('from')
            if nxt.get('disabled') or not isinstance(next_start, int) \
                    or next_start <= start:
                return
            start = next_start

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
        def first(name: str) -> Optional[str]:
            values = self._meta_values(soup, name)
            return values[0] if values else None

        pdf_url = first('citation_pdf_url')
        if not pdf_url:
            return None
        article: Dict[str, Any] = {
            'pdf_url': pdf_url,
            'url': pub_url,
            'itemtype': 'article',
        }
        id_match = re.search(r'/pub/(\d+)', pub_url)
        if id_match:
            article['naturalis_id'] = id_match.group(1)
        for field, meta in (('title', 'citation_title'),
                            ('journal', 'citation_journal_title'),
                            ('volume', 'citation_volume'),
                            ('number', 'citation_issue'),
                            ('issn', 'citation_issn'),
                            ('doi', 'citation_doi'),
                            ('abstract', 'citation_abstract')):
            value = first(meta)
            if value:
                article[field] = value
        for field, meta in (('author', 'citation_author'),
                            ('keywords', 'citation_keywords')):
            values = self._meta_values(soup, meta)
            if values:
                article[field] = '; '.join(values)
        first_page = first('citation_firstpage')
        last_page = first('citation_lastpage')
        if first_page:
            article['pages'] = (
                f"{first_page}–{last_page}"
                if last_page and last_page != first_page else first_page)
        date = self._parse_citation_date(
            first('citation_publication_date') or first('citation_date'))
        if date:
            article['publication_date'] = date
            article['year'] = date[:4]
        return article

    @staticmethod
    def _meta_values(soup: BeautifulSoup, name: str) -> List[str]:
        """Return the stripped content of every ``<meta name=name>``."""
        values: List[str] = []
        for tag in soup.find_all('meta', attrs={'name': name}):
            content = tag.get('content')
            if isinstance(content, str):
                text = ' '.join(content.split())
                if text:
                    values.append(text)
        return values

    @staticmethod
    def _parse_citation_date(value: Optional[str]) -> Optional[str]:
        """Convert ``YYYY/MM/DD`` (or ``YYYY``) to ISO ``YYYY-MM-DD``."""
        if not value:
            return None
        match = re.fullmatch(
            r'(\d{4})(?:[/-](\d{1,2})(?:[/-](\d{1,2}))?)?', value.strip())
        if not match:
            return None
        year, month, day = match.groups()
        return f"{year}-{int(month or 1):02d}-{int(day or 1):02d}"
