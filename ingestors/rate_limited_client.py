"""
Rate-limited HTTP client — adapter over pypaperretriever.HttpClient.

Historically this module had its own ~360-line implementation of rate
limiting, robots.txt parsing, 429 retry, and 403 domain suppression.
pypaperretriever now ships `HttpClient` with the same feature set plus
three correctness gaps the skol version had:

  * **doi.org / dx.doi.org exemption** from 403 suppression — a 403 reached
    via DOI resolution actually came from the destination publisher, not
    doi.org itself.  Without exemption a single bad publisher response
    poisoned every subsequent DOI lookup for the session.
  * **503 / 504 exponential backoff** with linear decay on success — skol's
    version treated 5xx as opaque non-200 warnings; HttpClient retries them
    with backoff and gradually returns to full speed.
  * **Multi-attempt 429 retry** — skol's version retried once per call;
    HttpClient retries up to `max_retries` (default 3).

This file is now a thin compatibility shim: its constructor signature and
the `.get()`, `.suppressed_domains`, `.retry_with_backoff()` API surface
match the legacy interface, but the underlying network behaviour is
HttpClient's.  Callers in `bin/` don't need to change.
"""

import time
from typing import Any, Callable, Dict, Optional
from urllib.robotparser import RobotFileParser

import requests
from pypaperretriever import HttpClient


class RateLimitedHttpClient:
    """
    Rate-limited HTTP client, delegating to pypaperretriever.HttpClient.

    Construction kwargs match the legacy class.  Internally, milliseconds
    are converted to seconds, and the presence of a `robot_parser` (the
    legacy mechanism for opting into robots.txt) is mapped to
    `respect_robots_txt=True` on the wrapped client.  HttpClient maintains
    its own robots.txt cache, so the caller's `RobotFileParser` instance is
    not actually consulted — it's used only as a boolean opt-in signal.
    """

    def __init__(
        self,
        user_agent: str,
        robot_parser: Optional[RobotFileParser] = None,
        verbosity: int = 1,
        rate_limit_min_ms: int = 1000,
        rate_limit_max_ms: int = 5000,
        max_retries: int = 3,
        retry_base_wait_time: int = 60,
        retry_backoff_multiplier: float = 2.0,
        timeout: int = 60,
    ) -> None:
        # Map legacy ms units → HttpClient's seconds.
        self._client = HttpClient(
            user_agent=user_agent,
            respect_robots_txt=robot_parser is not None,
            # Explicit per-domain delay always applies — matches legacy
            # behaviour of inserting random delays regardless of robots.
            delay_min_s=rate_limit_min_ms / 1000.0,
            delay_max_s=rate_limit_max_ms / 1000.0,
            max_retries=max_retries,
            verbosity=verbosity,
        )
        # Exception-retry parameters used by retry_with_backoff() only —
        # distinct from HttpClient's HTTP-level 429/5xx retry.
        self.user_agent = user_agent
        self.verbosity = verbosity
        self.rate_limit_min_ms = rate_limit_min_ms
        self.rate_limit_max_ms = rate_limit_max_ms
        self.max_retries = max_retries
        self.retry_base_wait_time = retry_base_wait_time
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.timeout = timeout
        self.last_fetch_time: Optional[float] = None

    @property
    def suppressed_domains(self) -> Dict[str, str]:
        """Forwards to HttpClient's blocklist (doi.org-exempt)."""
        return self._client.suppressed_domains

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """
        GET with rate limiting, retry, and domain suppression.

        Returns a `requests.Response` in all cases.  HttpClient returns
        `None` only when respect_robots_txt=True AND the URL is disallowed
        by robots.txt; in that case we synthesise an HTTP 451 ("Unavailable
        For Legal Reasons") response so callers expecting a Response object
        don't crash.  A 451 status code reads naturally as "we chose not to
        fetch this URL" — distinct from a 403 (server refused) or a real
        404 (origin not found).
        """
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
        response = self._client.get(url, **kwargs)
        if response is None:
            mock = requests.Response()
            mock.status_code = 451
            mock.url = url
            mock._content = b'Blocked by robots.txt'
            return mock
        self.last_fetch_time = time.time()
        return response

    def retry_with_backoff(
        self,
        func: Callable[..., Any],
        *args: Any,
        operation_name: str = "operation",
        **kwargs: Any,
    ) -> Any:
        """
        Exception-based retry wrapper for arbitrary callables.

        Distinct from HttpClient's response-based 429 retry: this catches
        *exceptions* whose stringified form contains '429', 'too many
        requests', or 'rate limit', and retries the callable with
        exponential backoff.  Used by Ingestor base class to wrap
        operations (e.g. publisher-specific PDF downloaders) that may
        raise on rate limits rather than returning a 429 status code.

        Returns the callable's result on success, or None if all retries
        fail (including non-rate-limit exceptions).
        """
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                self.last_fetch_time = time.time()
                return result
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = (
                    '429' in error_msg
                    or 'too many requests' in error_msg
                    or 'rate limit' in error_msg
                )
                if is_rate_limit and attempt < self.max_retries - 1:
                    wait_time = (
                        self.retry_base_wait_time
                        * (self.retry_backoff_multiplier ** attempt)
                    )
                    if self.verbosity >= 2:
                        print(
                            f"  {operation_name} hit rate limit, "
                            f"waiting {wait_time:.0f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                    time.sleep(wait_time)
                    continue
                if self.verbosity >= 3:
                    error_type = "rate limit" if is_rate_limit else "error"
                    print(f"  {operation_name} {error_type}: {e}")
                return None
        return None
