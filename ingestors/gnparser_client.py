"""Thin client for the gnparser authorship-parsing REST API.

gnparser (https://parser.globalnames.org/) parses scientific names and
extracts structured authorship, year, and canonical-name components.

This module uses gnparser to extract the author/year string that
immediately follows a taxon name already located by gnfinder.  The
caller passes a short text window (``text[name_end : name_end + 80]``)
and receives a :class:`ParsedAuthorship` describing the author span
within that window.

Default endpoint: ``https://parser.globalnames.org/api/v1``
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

_DEFAULT_URL = "https://parser.globalnames.org/api/v1"
_DEFAULT_TIMEOUT = 30  # seconds
_DEFAULT_RETRIES = 3
_BACKOFF_BASE = 1.0


@dataclass
class ParsedAuthorship:
    """Authorship string extracted by gnparser from a text window.

    Attributes:
        verbatim: The raw authorship text found in the window
            (e.g. ``"(L.) Lam., 1783"``).
        offset_in_window: Start character offset of *verbatim* within
            the window string that was passed in.
        length: Length of *verbatim* in characters.
        year: Publication year string (e.g. ``"1783"``), or empty if absent.
        authors: List of author surname strings.
    """

    verbatim: str
    offset_in_window: int
    length: int
    year: str = ""
    authors: List[str] = field(default_factory=list)


def parse_authorship_after_name(
    window: str,
    gnparser_url: str = _DEFAULT_URL,
    timeout: int = _DEFAULT_TIMEOUT,
    retries: int = _DEFAULT_RETRIES,
) -> Optional[ParsedAuthorship]:
    """Extract the authorship string from *window* using gnparser.

    *window* should be a short text slice starting immediately after a
    taxon name, e.g.::

        window = text[name_end : name_end + 80]

    gnparser is fed a synthetic name ``"X " + window`` so it can
    recognise the trailing authorship context.  The resulting authorship
    offset is adjusted back to the original window coordinates.

    Args:
        window: Text starting right after the taxon name.
        gnparser_url: Base URL of the gnparser API.
        timeout: Per-request timeout in seconds.
        retries: Number of retry attempts after the initial try.

    Returns:
        :class:`ParsedAuthorship` if authorship was found, else ``None``.

    Raises:
        requests.HTTPError: If all retries are exhausted.
    """
    if not window.strip():
        return None

    # Prepend a dummy uninomial so gnparser sees "Name <authorship>"
    synthetic = "Xus " + window
    results = _batch_parse([synthetic], gnparser_url=gnparser_url,
                           timeout=timeout, retries=retries)
    if not results:
        return None

    parsed = results[0]
    authorship = parsed.get("authorship") or {}
    verbatim_auth: str = authorship.get("verbatim", "") or ""
    if not verbatim_auth:
        return None

    # Locate the authorship in the synthetic string to compute offset
    try:
        syn_offset = synthetic.index(verbatim_auth)
    except ValueError:
        # authorship verbatim not found literally — fall back to end-of-dummy
        syn_offset = len("Xus ")

    # Adjust: subtract the dummy prefix length ("Xus " = 4 chars)
    offset_in_window = max(0, syn_offset - len("Xus "))

    authors: List[str] = _extract_authors(authorship)
    year: str = authorship.get("year", {}).get("year", "") or ""

    return ParsedAuthorship(
        verbatim=verbatim_auth,
        offset_in_window=offset_in_window,
        length=len(verbatim_auth),
        year=year,
        authors=authors,
    )


def _batch_parse(
    names: List[str],
    gnparser_url: str = _DEFAULT_URL,
    timeout: int = _DEFAULT_TIMEOUT,
    retries: int = _DEFAULT_RETRIES,
) -> List[Dict[str, Any]]:
    """POST a list of name strings to gnparser and return parsed dicts.

    Args:
        names: List of scientific name strings.
        gnparser_url: Base URL of the gnparser API.
        timeout: Per-request timeout in seconds.
        retries: Number of retries on transient failures.

    Returns:
        List of parsed result dicts (one per name).
    """
    url = gnparser_url.rstrip("/") + "/parse"
    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))
        try:
            resp = requests.post(url, json=names, timeout=timeout)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc

    raise last_exc  # type: ignore[misc]


def _extract_authors(authorship: Dict[str, Any]) -> List[str]:
    """Flatten gnparser authorship combinedAuthors into a surname list.

    Args:
        authorship: The ``"authorship"`` sub-dict from a gnparser result.

    Returns:
        List of author surname strings; empty if not present.
    """
    authors: List[str] = []
    for combo_key in ("combinedAuthors", "authors"):
        if combo_key in authorship:
            for author in authorship[combo_key]:
                name = author.get("name", "") or ""
                if name:
                    authors.append(name)
            break
    return authors
