"""
Unit tests for RateLimitedHttpClient.post().

post() is the adapter's counterpart of get(): it forwards to
pypaperretriever.HttpClient.post(), which shares get()'s robots.txt,
suppression, per-domain rate limit, and backoff, and it gives callers a
Response in every case (a synthetic 451 when robots.txt forbids the URL).
"""

from unittest.mock import MagicMock

from .rate_limited_client import RateLimitedHttpClient


URL = 'https://repository.naturalis.nl/search/query'


def _client() -> RateLimitedHttpClient:
    client = RateLimitedHttpClient(user_agent='test-agent',
                                   robot_parser=MagicMock(), timeout=30)
    client._client = MagicMock()
    return client


class TestPost:
    def test_forwards_to_http_client_with_default_timeout(self) -> None:
        client = _client()
        response = MagicMock(status_code=200)
        client._client.post.return_value = response
        assert client.post(URL, json={'q': 1}) is response
        args, kwargs = client._client.post.call_args
        assert args[0] == URL
        assert kwargs['json'] == {'q': 1}
        assert kwargs['timeout'] == 30

    def test_caller_timeout_wins(self) -> None:
        client = _client()
        client._client.post.return_value = MagicMock(status_code=200)
        client.post(URL, timeout=5)
        assert client._client.post.call_args[1]['timeout'] == 5

    def test_robots_block_becomes_451(self) -> None:
        client = _client()
        client._client.post.return_value = None
        response = client.post(URL)
        assert response.status_code == 451
        assert response.url == URL

    def test_records_last_fetch_time(self) -> None:
        client = _client()
        client._client.post.return_value = MagicMock(status_code=200)
        assert client.last_fetch_time is None
        client.post(URL)
        assert isinstance(client.last_fetch_time, float)
