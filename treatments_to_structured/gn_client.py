"""gnfinder + gnparser HTTP clients + authored-binomial detector.

Isolated from ``triage_signals`` so the pure-Python signal
predicates stay free of network dependencies.  ``triage_signals``
accepts the detector's boolean result as an optional keyword arg
via ``treatment_signals(..., authored_binomial_in_desc=...)``.

Default URLs match the ``skol-gnservices`` deb's localhost ports;
override via ``env_config``'s ``gnfinder_url`` / ``gnparser_url``.
"""

import urllib.error
import urllib.parse
import urllib.request
import json
from typing import Any, Dict, List


DEFAULT_GNFINDER_URL = 'http://localhost:9080/api/v1/find'
DEFAULT_GNPARSER_URL = 'http://localhost:9081/api/v1'


class GnServiceUnavailable(Exception):
    """Raised when gnfinder or gnparser can't be reached (network
    error, timeout, or non-2xx response).  Callers should catch
    this to degrade gracefully — the detector reports None (not
    fired) rather than crashing the whole triage run.
    """


def find_names(
    text: str,
    url: str = DEFAULT_GNFINDER_URL,
    *,
    timeout: float = 8.0,
    words_around: int = 4,
) -> List[Dict[str, Any]]:
    """Call gnfinder's ``/find`` endpoint and return the list of
    detected names.

    Each returned dict carries at least ``verbatim`` (the raw
    matched string), ``start``/``end`` byte offsets, and
    ``wordsAfter`` (list of tokens immediately following the
    name — needed to reconstruct the authorship suffix for
    ``parse_name``).
    """
    if not text:
        return []
    payload = json.dumps({
        'text': text,
        'wordsAround': words_around,
        'returnContent': False,
    }).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GnServiceUnavailable(
            f'gnfinder at {url} unavailable: {exc}'
        ) from exc
    data = json.loads(body)
    return list(data.get('names') or [])


def parse_name(
    name: str,
    url: str = DEFAULT_GNPARSER_URL,
    *,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Call gnparser's GET endpoint and return the parsed result
    for a single name.  Returns an empty dict when the name
    doesn't parse — callers check ``result.get('authorship',
    {}).get('normalized')`` to decide if authorship is present.
    """
    if not name:
        return {}
    encoded = urllib.parse.quote(name, safe='')
    full_url = f'{url.rstrip("/")}/{encoded}'
    try:
        with urllib.request.urlopen(full_url, timeout=timeout) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GnServiceUnavailable(
            f'gnparser at {url} unavailable: {exc}'
        ) from exc
    data = json.loads(body)
    # gnparser returns a JSON array; single-name query returns a
    # one-element list.  Return the first element or empty dict.
    if isinstance(data, list) and data:
        return dict(data[0])
    return {}


def _has_authorship(parsed: Dict[str, Any]) -> bool:
    """True if gnparser's parsed result carries a non-empty
    ``authorship.normalized`` field.  Isolates the exact key
    structure so tests can rely on this predicate."""
    authorship = parsed.get('authorship') or {}
    normalized = authorship.get('normalized') or ''
    return bool(normalized.strip())


def authored_binomial_in_text(
    text: str,
    gnfinder_url: str = DEFAULT_GNFINDER_URL,
    gnparser_url: str = DEFAULT_GNPARSER_URL,
    *,
    timeout: float = 8.0,
) -> bool:
    """True if ``text`` contains at least one authored binomial
    citation.

    Composed pipeline:
      1. gnfinder locates candidate scientific names in the text.
      2. For each candidate: reconstruct ``verbatim + wordsAfter``
         and pass to gnparser.
      3. If gnparser identifies non-empty ``authorship`` for any
         candidate, return True — the text contains a formal
         citation.  Per §1 rule, a Description that contains a
         formal authored citation is a merge/leak signal.

    Early-exits on the first authored match to skip unnecessary
    gnparser calls.

    ``GnServiceUnavailable`` propagates from ``find_names`` — the
    triage CLI catches it and degrades gracefully.
    Individual gnparser failures on a specific name are treated
    as unauthored (conservative — don't fire on unverified
    names).
    """
    if not text:
        return False
    names = find_names(text, gnfinder_url, timeout=timeout)
    for entry in names:
        verbatim = entry.get('verbatim') or ''
        words_after = entry.get('wordsAfter') or []
        if not verbatim:
            continue
        # Try progressively longer candidates: verbatim + 1 word,
        # +2, ..., +N.  gnparser can misparse when too much trailing
        # prose is appended after the authorship (it treats
        # subsequent tokens as extra epithets and drops the
        # authorship parse).  Return True on the FIRST candidate
        # that yields non-empty authorship.  taxon_83e36037's
        # `Trichaptum perrottetii (Lév.) Ryvarden The basidiocarps`
        # parses correctly at 2-3 words after but not at 4.
        found_authorship = False
        for n_words in range(1, len(words_after) + 1):
            candidate = ' '.join([verbatim, *words_after[:n_words]])
            try:
                parsed = parse_name(
                    candidate, gnparser_url, timeout=timeout,
                )
            except GnServiceUnavailable:
                # Individual gnparser call failed — treat as
                # unauthored for this candidate length; keep
                # trying other lengths.
                continue
            if _has_authorship(parsed):
                found_authorship = True
                break
        if found_authorship:
            return True
    return False


__all__ = (
    'DEFAULT_GNFINDER_URL',
    'DEFAULT_GNPARSER_URL',
    'GnServiceUnavailable',
    'find_names',
    'parse_name',
    'authored_binomial_in_text',
)
