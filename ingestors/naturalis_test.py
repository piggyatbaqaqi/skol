"""
Unit tests for NaturalisIngestor (repository.naturalis.nl).

Fixtures are trimmed copies of real responses fetched 2026-11-23:
the ``/pub/531660/`` landing page (Persoonia 1(1), 1959) and the shape
of a ``/search/query`` JSON reply.
"""

from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from . import naturalis as naturalis_module
from .naturalis import NaturalisIngestor


def _make_ingestor(**overrides: Any) -> NaturalisIngestor:
    """Create a NaturalisIngestor with mocked dependencies."""
    robot_parser = overrides.pop('robot_parser', None)
    if robot_parser is None:
        robot_parser = MagicMock()
        robot_parser.can_fetch.return_value = True
    defaults: Dict[str, Any] = dict(
        db=overrides.pop('db', MagicMock()),
        user_agent='test-agent',
        robot_parser=robot_parser,
        verbosity=0,
        collection_id='1',
    )
    defaults.update(overrides)
    return NaturalisIngestor(**defaults)


PUB_URL = 'https://repository.naturalis.nl/pub/531660/'
PDF_URL = ('https://repository.naturalis.nl/pub/531660/'
           'PERS1959001001008.pdf')

PUB_HTML = f"""
<html><head>
<meta name="citation_title" content="The stipitate Hydnums of the
 Netherlands—IV. Auriscalpium S. F. Gray"/>
<meta name="citation_author" content="Maas Geesteranus, R.A."/>
<meta name="citation_author" content="Donk, M.A."/>
<meta name="citation_date" content="1959/01/01"/>
<meta name="citation_publication_date" content="1959/01/01"/>
<meta name="citation_journal_title"
 content="Persoonia - Molecular Phylogeny and Evolution of Fungi"/>
<meta name="citation_volume" content="1"/>
<meta name="citation_issue" content="1"/>
<meta name="citation_issn" content="0031-5850"/>
<meta name="citation_firstpage" content="115"/>
<meta name="citation_lastpage" content="147"/>
<meta name="citation_abstract" content="A revision is given of the
 genera Auriscalpium, Hericium, Hydnum, and Sistotrema."/>
<meta name="citation_pdf_url" content="{PDF_URL}"/>
</head><body></body></html>
"""

PUB_HTML_MODERN = """
<html><head>
<meta name="citation_title" content="The genus Entoloma in Europe"/>
<meta name="citation_author" content="Noordeloos, Machiel E."/>
<meta name="citation_publication_date" content="2026/02/13"/>
<meta name="citation_journal_title"
 content="Persoonia - Molecular Phylogeny and Evolution of Fungi"/>
<meta name="citation_volume" content="56"/>
<meta name="citation_issue" content="1"/>
<meta name="citation_firstpage" content="200"/>
<meta name="citation_lastpage" content="307"/>
<meta name="citation_doi" content="10.3114/persoonia.2026.56.03" />
<meta name="citation_keywords" content="Agaricales"/>
<meta name="citation_keywords" content="taxonomy"/>
<meta name="citation_pdf_url"
 content="https://repository.naturalis.nl/pub/802287/Noordeloos-2026.pdf"/>
</head><body></body></html>
"""

PUB_HTML_NO_PDF = """
<html><head>
<meta name="citation_title" content="Staff paper elsewhere"/>
<meta name="citation_publication_date" content="2019/01/01"/>
</head><body></body></html>
"""


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, 'html.parser')


def _reply(ids: List[str], next_from: int, disabled: bool) -> Dict[str, Any]:
    """A /search/query reply with the given hit ids and paging cursor."""
    return {
        'hits': [
            {'id': i, 'url': f'https://repository.naturalis.nl/pub/{i}'}
            for i in ids
        ],
        'paging': {'next': {'from': next_from, 'disabled': disabled}},
    }


class TestConstruction:
    def test_defaults(self) -> None:
        ing = _make_ingestor()
        assert ing.collection_id == '1'
        assert ing.base_url == 'https://repository.naturalis.nl'
        assert ing.limit is None
        assert ing.dry_run is False

    def test_collection_id_coerced_to_str_and_base_url_trimmed(self) -> None:
        ing = _make_ingestor(collection_id=7,
                             base_url='https://example.org/')
        assert ing.collection_id == '7'
        assert ing.base_url == 'https://example.org'

    def test_ignores_unknown_kwargs_from_sources_row(self) -> None:
        # bin/ingest.py passes every SOURCES field through.
        ing = _make_ingestor(mode='web', journal='persoonia',
                             skip_existing=True)
        assert ing.collection_id == '1'


class TestHelpers:
    def test_search_url(self) -> None:
        assert (_make_ingestor().search_url()
                == 'https://repository.naturalis.nl/search/query')

    def test_meta_values_strips_and_keeps_order(self) -> None:
        soup = _soup(PUB_HTML)
        assert NaturalisIngestor._meta_values(soup, 'citation_author') == [
            'Maas Geesteranus, R.A.', 'Donk, M.A.']
        assert NaturalisIngestor._meta_values(soup, 'citation_doi') == []

    def test_parse_citation_date(self) -> None:
        parse = NaturalisIngestor._parse_citation_date
        assert parse('1959/01/01') == '1959-01-01'
        assert parse('2026/2/3') == '2026-02-03'
        assert parse('1987') == '1987-01-01'
        assert parse('') is None
        assert parse(None) is None
        assert parse('not a date') is None

    def test_format_urls(self) -> None:
        ing = _make_ingestor()
        base = {'pdf_url': PDF_URL, 'url': PUB_URL}
        assert ing.format_pdf_url(base) == PDF_URL
        assert ing.format_human_url(base) == PUB_URL


class TestSearchBody:
    def test_filters_to_journal_and_sorts_oldest_first(self) -> None:
        body = _make_ingestor(collection_id='3').build_search_body(1)
        query = body['query']
        assert query['filters']['values'] == [
            {'field_id': 'journal', 'query': '3'}]
        assert query['sort'] == 'year_asc'
        assert query['from'] == 1

    def test_start_is_passed_through(self) -> None:
        body = _make_ingestor().build_search_body(40)
        assert body['query']['from'] == 40


POST_VIA_CLIENT = pytest.mark.xfail(
    reason='_post_search still calls requests.post directly', strict=True)


def _with_client(**overrides: Any) -> 'tuple[NaturalisIngestor, MagicMock]':
    """An ingestor whose RateLimitedHttpClient is a mock."""
    ing = _make_ingestor(**overrides)
    client = MagicMock()
    ing.http_client = client
    return ing, client


class TestPostSearch:
    @pytest.fixture(autouse=True)
    def _no_network(self) -> Iterator[None]:
        """No test here may reach the real repository."""
        with patch.object(naturalis_module.requests, 'post',
                          side_effect=AssertionError('real POST')):
            yield

    @POST_VIA_CLIENT
    def test_posts_json_through_the_rate_limited_client(self) -> None:
        ing, client = _with_client()
        response = MagicMock(status_code=200)
        response.json.return_value = {'hits': []}
        client.post.return_value = response
        assert ing._post_search({'query': {}}) == {'hits': []}
        args, kwargs = client.post.call_args
        assert args[0] == 'https://repository.naturalis.nl/search/query'
        assert kwargs['json'] == {'query': {}}
        assert kwargs['headers']['Accept'] == 'application/json'

    def test_robots_denial_returns_none_without_posting(self) -> None:
        robot_parser = MagicMock()
        robot_parser.can_fetch.return_value = False
        ing, client = _with_client(robot_parser=robot_parser)
        with patch.object(naturalis_module.requests, 'post') as post:
            assert ing._post_search({}) is None
        client.post.assert_not_called()
        post.assert_not_called()

    @POST_VIA_CLIENT
    def test_client_robots_block_451_returns_none(self) -> None:
        ing, client = _with_client()
        client.post.return_value = MagicMock(status_code=451)
        assert ing._post_search({}) is None
        client.post.assert_called_once()

    @POST_VIA_CLIENT
    def test_http_error_returns_none(self) -> None:
        ing, client = _with_client()
        client.post.return_value = MagicMock(status_code=500)
        assert ing._post_search({}) is None
        client.post.assert_called_once()

    @POST_VIA_CLIENT
    def test_network_error_returns_none(self) -> None:
        ing, client = _with_client()
        client.post.side_effect = ConnectionError('boom')
        assert ing._post_search({}) is None
        client.post.assert_called_once()

    @POST_VIA_CLIENT
    def test_bad_json_returns_none(self) -> None:
        ing, client = _with_client()
        response = MagicMock(status_code=200)
        response.json.side_effect = ValueError('not json')
        client.post.return_value = response
        assert ing._post_search({}) is None
        client.post.assert_called_once()


class TestIterSearchHits:
    def test_follows_next_until_disabled(self) -> None:
        ing = _make_ingestor()
        ing._post_search = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                _reply(['a', 'b'], 10, False),
                _reply(['c'], 20, True),
            ])
        ids = [h['id'] for h in ing.iter_search_hits()]
        assert ids == ['a', 'b', 'c']
        starts = [c.args[0]['query']['from']
                  for c in ing._post_search.call_args_list]
        assert starts == [1, 10]

    def test_duplicate_hits_yielded_once(self) -> None:
        ing = _make_ingestor()
        ing._post_search = MagicMock(  # type: ignore[method-assign]
            side_effect=[
                _reply(['a', 'b'], 10, False),
                _reply(['b', 'c'], 20, True),
            ])
        assert [h['id'] for h in ing.iter_search_hits()] == ['a', 'b', 'c']

    def test_stops_on_failed_request(self) -> None:
        ing = _make_ingestor()
        ing._post_search = MagicMock(  # type: ignore[method-assign]
            side_effect=[_reply(['a'], 10, False), None])
        assert [h['id'] for h in ing.iter_search_hits()] == ['a']

    def test_stops_on_empty_page(self) -> None:
        ing = _make_ingestor()
        ing._post_search = MagicMock(  # type: ignore[method-assign]
            side_effect=[_reply(['a'], 10, False), _reply([], 20, False)])
        assert [h['id'] for h in ing.iter_search_hits()] == ['a']
        assert ing._post_search.call_count == 2

    def test_stops_if_cursor_does_not_advance(self) -> None:
        # Guards against an endless loop if the server ignores 'from'.
        ing = _make_ingestor()
        ing._post_search = MagicMock(  # type: ignore[method-assign]
            side_effect=[_reply(['a'], 1, False), _reply(['a'], 1, False)])
        assert [h['id'] for h in ing.iter_search_hits()] == ['a']
        assert ing._post_search.call_count == 1


class TestParsePubPage:
    def test_historic_article(self) -> None:
        art = _make_ingestor().parse_pub_page(_soup(PUB_HTML), PUB_URL)
        assert art is not None
        assert art['title'] == (
            'The stipitate Hydnums of the Netherlands—IV. '
            'Auriscalpium S. F. Gray')
        assert art['author'] == 'Maas Geesteranus, R.A.; Donk, M.A.'
        assert art['journal'] == (
            'Persoonia - Molecular Phylogeny and Evolution of Fungi')
        assert art['volume'] == '1'
        assert art['number'] == '1'
        assert art['pages'] == '115–147'
        assert art['year'] == '1959'
        assert art['publication_date'] == '1959-01-01'
        assert art['issn'] == '0031-5850'
        assert art['abstract'].startswith('A revision is given of the genera')
        assert art['pdf_url'] == PDF_URL
        assert art['url'] == PUB_URL
        assert art['naturalis_id'] == '531660'
        assert art['itemtype'] == 'article'
        assert 'doi' not in art
        assert 'keywords' not in art

    def test_modern_article_has_doi_and_keywords(self) -> None:
        url = 'https://repository.naturalis.nl/pub/802287'
        art = _make_ingestor().parse_pub_page(_soup(PUB_HTML_MODERN), url)
        assert art is not None
        assert art['doi'] == '10.3114/persoonia.2026.56.03'
        assert art['keywords'] == 'Agaricales; taxonomy'
        assert art['year'] == '2026'
        assert art['publication_date'] == '2026-02-13'
        assert art['naturalis_id'] == '802287'
        assert 'issn' not in art
        assert 'abstract' not in art

    def test_single_page_article(self) -> None:
        html = PUB_HTML.replace('content="147"', 'content="115"')
        art = _make_ingestor().parse_pub_page(_soup(html), PUB_URL)
        assert art is not None
        assert art['pages'] == '115'

    def test_no_pdf_returns_none(self) -> None:
        assert _make_ingestor().parse_pub_page(
            _soup(PUB_HTML_NO_PDF), PUB_URL) is None


class TestIngest:
    sink: MagicMock

    def _ingestor_with_hits(self, n: int, **kw: Any) -> NaturalisIngestor:
        ing = _make_ingestor(**kw)
        hits = [{'id': str(i), 'url': f'https://repository.naturalis.nl/'
                 f'pub/{i}'} for i in range(n)]
        ing.iter_search_hits = MagicMock(  # type: ignore[method-assign]
            return_value=iter(hits))
        ing._fetch_page = MagicMock(  # type: ignore[method-assign]
            return_value=_soup(PUB_HTML))
        self.sink = MagicMock()
        ing._ingest_documents = self.sink  # type: ignore[method-assign]
        return ing

    def test_each_article_goes_to_base_pipeline(self) -> None:
        ing = self._ingestor_with_hits(2)
        ing.ingest()
        assert self.sink.call_count == 2
        docs, meta, bibtex_link = self.sink.call_args.args
        assert docs[0]['pdf_url'] == PDF_URL
        assert meta['source'] == 'naturalis'
        assert meta['collection_id'] == '1'
        assert bibtex_link == 'https://repository.naturalis.nl/pub/1'

    def test_limit(self) -> None:
        ing = self._ingestor_with_hits(5, limit=3)
        ing.ingest()
        assert self.sink.call_count == 3

    def test_dry_run_writes_nothing(self) -> None:
        ing = self._ingestor_with_hits(2, dry_run=True)
        ing.ingest()
        self.sink.assert_not_called()

    def test_pages_without_pdf_or_unfetchable_are_skipped(self) -> None:
        ing = self._ingestor_with_hits(3)
        ing._fetch_page = MagicMock(  # type: ignore[method-assign]
            side_effect=[_soup(PUB_HTML_NO_PDF), None, _soup(PUB_HTML)])
        ing.ingest()
        assert self.sink.call_count == 1

    def test_skipped_pages_do_not_count_toward_limit(self) -> None:
        ing = self._ingestor_with_hits(3, limit=1)
        ing._fetch_page = MagicMock(  # type: ignore[method-assign]
            side_effect=[_soup(PUB_HTML_NO_PDF), _soup(PUB_HTML),
                         _soup(PUB_HTML)])
        ing.ingest()
        assert self.sink.call_count == 1
