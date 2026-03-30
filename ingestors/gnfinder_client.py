"""Thin client for the gnfinder taxon-name recognition REST API.

gnfinder (https://finder.globalnames.org/) finds scientific names in text
using Bayesian and heuristic methods.  This module wraps the v1 REST
endpoint and maps its response to :class:`NameSpan` objects compatible
with the :mod:`ingestors.spans` layer.

Default endpoint: ``https://finder.globalnames.org/api/v1/find``

Example::

    from ingestors.gnfinder_client import find_names

    spans = find_names("Amanita muscaria (L.) Lam. is a well-known fungus.")
    for ns in spans:
        print(ns.canonical, ns.start, ns.end, ns.annot_nomen_type)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

_DEFAULT_URL = "https://finder.globalnames.org/api/v1/find"
_DEFAULT_TIMEOUT = 30  # seconds
_DEFAULT_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds; actual delay = _BACKOFF_BASE * 2**attempt


@dataclass
class NameSpan:
    """A scientific name detected by gnfinder.

    Attributes:
        start: Inclusive start character offset in the input text.
        end: Exclusive end character offset in the input text.
        verbatim: Raw text as it appears in the input.
        canonical: Canonical name without authorship.
        cardinality: 0 = unknown, 1 = uninomial, 2 = binomial, 3 = trinomial.
        odds_log10: log₁₀ of the Bayes odds ratio (higher = more confident).
        annot_nomen: Nomenclatural annotation string (e.g. ``"sp. nov."``).
        annot_nomen_type: Annotation type code: ``"SP_NOV"``, ``"COMB_NOV"``,
            or ``"NO_ANNOT"``.
    """

    start: int
    end: int
    verbatim: str
    canonical: str
    cardinality: int
    odds_log10: float
    annot_nomen: str = ""
    annot_nomen_type: str = "NO_ANNOT"


def find_names(
    text: str,
    gnfinder_url: str = _DEFAULT_URL,
    verify: bool = False,
    timeout: int = _DEFAULT_TIMEOUT,
    retries: int = _DEFAULT_RETRIES,
) -> List[NameSpan]:
    """Find scientific names in *text* using the gnfinder REST API.

    Sends a POST request with ``{"text": text}`` and parses the
    ``"names"`` array from the response.  Retries on transient HTTP
    errors and timeouts with exponential back-off.

    Args:
        text: Plaintext to search for scientific names.
        gnfinder_url: Base URL of the gnfinder API endpoint.
        verify: If False, skip TLS certificate verification (useful for
            self-hosted instances with self-signed certs).
        timeout: Per-request timeout in seconds.
        retries: Number of retry attempts after the initial try.

    Returns:
        List of :class:`NameSpan` objects sorted by start offset.

    Raises:
        requests.HTTPError: If all retries are exhausted and the last
            response had a non-2xx status code.
        requests.ConnectionError: If the server is unreachable.
    """
    payload: Dict[str, Any] = {"text": text}
    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):
        if attempt > 0:
            delay = _BACKOFF_BASE * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            resp = requests.post(
                gnfinder_url,
                json=payload,
                timeout=timeout,
                verify=verify,
            )
            resp.raise_for_status()
            return _parse_response(resp.json())
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc

    raise last_exc  # type: ignore[misc]


def _parse_response(body: Dict[str, Any]) -> List[NameSpan]:
    """Parse the gnfinder JSON response into :class:`NameSpan` objects.

    Args:
        body: Parsed JSON response dict from the gnfinder API.

    Returns:
        List of :class:`NameSpan` objects sorted by start offset.
    """
    names = body.get("names") or []
    result: List[NameSpan] = []
    for entry in names:
        start: int = entry.get("start", 0)
        # gnfinder "end" is exclusive (past-the-end)
        end: int = entry.get("end", start)
        verbatim: str = entry.get("verbatim", "")
        # best match is in "bestResult" sub-dict when present
        best = entry.get("bestResult") or {}
        canonical: str = best.get("name", verbatim)
        cardinality: int = best.get("cardinality", 0)
        odds: float = float(entry.get("oddsLog10", 0.0))
        annot: str = entry.get("annotNomen", "") or ""
        annot_type: str = entry.get("annotNomenType", "NO_ANNOT") or "NO_ANNOT"

        result.append(
            NameSpan(
                start=start,
                end=end,
                verbatim=verbatim,
                canonical=canonical,
                cardinality=cardinality,
                odds_log10=odds,
                annot_nomen=annot,
                annot_nomen_type=annot_type,
            )
        )
    return sorted(result, key=lambda n: n.start)
