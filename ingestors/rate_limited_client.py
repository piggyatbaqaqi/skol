"""
Rate-limited HTTP client for respectful web scraping.

This module provides a RateLimitedHttpClient class that handles rate limiting,
domain suppression, and retry logic for HTTP requests.
"""

import time
import random
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests


class RateLimitedHttpClient:
    """
    HTTP client with built-in rate limiting and retry logic.

    Features:
    - Respects robots.txt Crawl-Delay directives
    - Random delay between configurable min/max bounds
    - Handles HTTP 429 (Too Many Requests) with automatic retry
    - Suppresses domains that return 403 Forbidden
    - Exponential backoff for rate limit errors
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
        """
        Initialize the RateLimitedHttpClient.

        Args:
            user_agent: User agent string for HTTP requests
            robot_parser: Optional robot file parser for checking Crawl-Delay
            verbosity: Verbosity level (0=silent, 1=warnings, 2=normal, 3=verbose)
            rate_limit_min_ms: Minimum delay between requests in milliseconds
            rate_limit_max_ms: Maximum delay between requests in milliseconds
            max_retries: Maximum retry attempts for 429 rate limit errors
            retry_base_wait_time: Initial wait time in seconds for exponential backoff
            retry_backoff_multiplier: Multiplier for exponential backoff
            timeout: Request timeout in seconds
        """
        self.user_agent = user_agent
        self.robot_parser = robot_parser
        self.verbosity = verbosity
        self.rate_limit_min_ms = rate_limit_min_ms
        self.rate_limit_max_ms = rate_limit_max_ms
        self.max_retries = max_retries
        self.retry_base_wait_time = retry_base_wait_time
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.timeout = timeout
        self.last_fetch_time: Optional[float] = None
        self.suppressed_domains: Dict[str, str] = {}
        self._session = requests.Session()

    def _apply_rate_limit(self) -> None:
        """
        Apply rate limiting before making an HTTP request.

        Uses Crawl-Delay from robots.txt if available, otherwise uses a
        random delay between configured min/max bounds.
        """
        if self.last_fetch_time is None:
            return

        delay_seconds = None

        # Check if robots.txt specifies a Crawl-Delay
        if self.robot_parser is not None:
            crawl_delay = self.robot_parser.crawl_delay(self.user_agent)
            if crawl_delay is not None:
                delay_seconds = float(crawl_delay)
                if self.verbosity >= 3:
                    print(f"  Using Crawl-Delay from robots.txt: {delay_seconds}s")

        if delay_seconds is None:
            # Use random delay between configured bounds (convert ms to seconds)
            delay_seconds = random.uniform(
                self.rate_limit_min_ms / 1000.0,
                self.rate_limit_max_ms / 1000.0
            )
            if self.verbosity >= 3:
                print(f"  Using random delay: {delay_seconds:.2f}s")

        # Calculate time since last fetch
        elapsed = time.time() - self.last_fetch_time
        sleep_time = delay_seconds - elapsed

        if sleep_time > 0:
            if self.verbosity >= 3:
                print(f"  Sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

    def _check_suppression(self, url: str) -> Optional[requests.Response]:
        """
        Check if the domain of the given URL is suppressed.

        Args:
            url: URL to check

        Returns:
            Mock 403 Response if domain is suppressed, None otherwise
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        if domain in self.suppressed_domains:
            if self.verbosity >= 2:
                reason = self.suppressed_domains[domain]
                print(f"  Skipping suppressed domain: {domain} ({reason})")

            # Return a mock 403 response
            mock_response = requests.Response()
            mock_response.status_code = 403
            mock_response.url = url
            mock_response._content = b''
            return mock_response

        return None

    def _register_suppression(self, url: str) -> None:
        """Register a domain as suppressed due to 403 Forbidden."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        if domain not in self.suppressed_domains:
            reason = f"403 Forbidden at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            self.suppressed_domains[domain] = reason

            if self.verbosity >= 1:
                print(f"  Domain suppressed due to 403 Forbidden: {domain}")

    def _handle_429(self, response: requests.Response, url: str) -> Optional[float]:
        """
        Parse 429 response headers to determine wait time.

        Args:
            response: The 429 response
            url: The URL that was requested

        Returns:
            Wait time in seconds, or None if unable to determine
        """
        wait_seconds = None
        header_used = None

        # 1. Check Retry-After header (RFC 7231)
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                # Try parsing as integer (seconds)
                wait_seconds = int(retry_after)
                header_used = 'Retry-After'
            except ValueError:
                # Try parsing as HTTP date
                try:
                    from email.utils import parsedate_to_datetime
                    retry_time = parsedate_to_datetime(retry_after)
                    wait_seconds = (
                        retry_time - datetime.now(retry_time.tzinfo)
                    ).total_seconds()
                    wait_seconds = max(0, wait_seconds)
                    header_used = 'Retry-After'
                except (ValueError, TypeError):
                    wait_seconds = None

        # 2. Check RateLimit-Reset or X-RateLimit-Reset (IETF draft)
        if wait_seconds is None:
            reset_header = (
                response.headers.get('RateLimit-Reset') or
                response.headers.get('X-RateLimit-Reset')
            )
            if reset_header:
                try:
                    reset_value = int(reset_header)
                    current_time = time.time()
                    if reset_value > 1000000000:
                        # Unix timestamp
                        wait_seconds = max(0, reset_value - current_time)
                    else:
                        # Seconds until reset
                        wait_seconds = reset_value
                    header_used = (
                        'RateLimit-Reset'
                        if 'RateLimit-Reset' in response.headers
                        else 'X-RateLimit-Reset'
                    )
                except (ValueError, TypeError):
                    wait_seconds = None

        # 3. Default if no headers found or parsing failed
        if wait_seconds is None:
            wait_seconds = 60
            header_used = 'default'
            if self.verbosity >= 1:
                print(
                    "  Warning: No rate limit headers found, "
                    "using 60s default"
                )
                if self.verbosity >= 2:
                    print("  All response headers:")
                    for header_name, header_value in response.headers.items():
                        print(f"    {header_name}: {header_value}")

        if self.verbosity >= 1:
            print(
                f"  HTTP 429 (Too Many Requests) - "
                f"waiting {wait_seconds:.0f}s before retry "
                f"(from {header_used})"
            )

        return wait_seconds

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """
        Make a GET request with rate limiting and retry logic.

        Handles HTTP 429 (Too Many Requests) responses by checking the
        Retry-After header and waiting the specified duration before retrying.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments to pass to requests.get()

        Returns:
            Response object from requests.get()
        """
        # Check if domain is suppressed (returns mock 403 if suppressed)
        suppression_response = self._check_suppression(url)
        if suppression_response is not None:
            return suppression_response

        self._apply_rate_limit()

        if self.verbosity >= 3:
            print(f"  Fetching: {url}")

        # Record fetch time before making request
        self.last_fetch_time = time.time()

        # Ensure User-Agent header is set
        headers = kwargs.get('headers', {})
        if 'User-Agent' not in headers:
            headers['User-Agent'] = self.user_agent
            kwargs['headers'] = headers

        # Set timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        response = self._session.get(url, **kwargs)

        # Log rate limit info if available (for debugging)
        if self.verbosity >= 4:
            rate_limit_headers = {
                'RateLimit-Limit': response.headers.get('RateLimit-Limit'),
                'RateLimit-Remaining': response.headers.get('RateLimit-Remaining'),
                'RateLimit-Reset': response.headers.get('RateLimit-Reset'),
                'X-RateLimit-Limit': response.headers.get('X-RateLimit-Limit'),
                'X-RateLimit-Remaining': response.headers.get('X-RateLimit-Remaining'),
                'X-RateLimit-Reset': response.headers.get('X-RateLimit-Reset'),
            }
            present_headers = {k: v for k, v in rate_limit_headers.items() if v}
            if present_headers:
                print(f"  Rate limit headers: {present_headers}")

        # Handle HTTP 429 (Too Many Requests) with various headers
        if response.status_code == 429:
            wait_seconds = self._handle_429(response, url)
            time.sleep(wait_seconds)

            # Update last fetch time after the wait
            self.last_fetch_time = time.time()

            # Retry the request
            if self.verbosity >= 3:
                print(f"  Retrying: {url}")

            response = self._session.get(url, **kwargs)

        # Handle HTTP 403 (Forbidden) - add domain to suppression list
        if response.status_code == 403:
            self._register_suppression(url)

        # Log non-200 status codes
        if response.status_code != 200:
            if self.verbosity >= 1:
                print(f"  Warning: Received status code {response.status_code} for URL: {url}")

        return response

    def retry_with_backoff(
        self,
        func,
        *args,
        operation_name: str = "operation",
        **kwargs
    ):
        """
        Execute a function with exponential backoff retry logic for rate limit errors.

        Args:
            func: Callable to execute
            *args: Positional arguments to pass to func
            operation_name: Name of the operation for logging
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result of func() or None if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                # Success - update last fetch time and return
                self.last_fetch_time = time.time()
                return result
            except Exception as e:
                error_msg = str(e).lower()
                # Check if error is rate-limit related
                is_rate_limit = (
                    '429' in error_msg or
                    'too many requests' in error_msg or
                    'rate limit' in error_msg
                )

                if is_rate_limit and attempt < self.max_retries - 1:
                    # Calculate exponential backoff wait time
                    wait_time = (
                        self.retry_base_wait_time *
                        (self.retry_backoff_multiplier ** attempt)
                    )
                    if self.verbosity >= 2:
                        print(
                            f"  {operation_name} hit rate limit, waiting {wait_time:.0f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})"
                        )
                    time.sleep(wait_time)
                    continue
                else:
                    # Not a rate limit error, or we've exhausted retries
                    if self.verbosity >= 3:
                        error_type = "rate limit" if is_rate_limit else "error"
                        print(f"  {operation_name} {error_type}: {e}")
                    return None

        return None
