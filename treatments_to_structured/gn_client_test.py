"""Tests for treatments_to_structured.gn_client.

Uses monkeypatch to stub urllib.request.urlopen so tests don't
require the gnfinder / gnparser services to be running.  The
integration behaviour on live services is verified end-to-end
via the triage CLI's live regression check.
"""

import io
import json
import urllib.error
from typing import Any, Dict, List

import pytest

from treatments_to_structured.gn_client import (
    GnServiceUnavailable,
    authored_binomial_in_text,
)


# ---------------------------------------------------------------------------
# Fake urlopen — records calls and returns pre-programmed JSON bodies
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal urlopen-response stand-in supporting context-manager
    protocol + .read()."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> '_FakeResponse':
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeUrlopen:
    """Pluggable urlopen stub.  Given a sequence of responses keyed
    by URL substring, returns the matching body when called.

    Usage:
      fake = _FakeUrlopen({
          '9080': gnfinder_response_bytes,
          '9081': gnparser_response_bytes,
      })
      monkeypatch.setattr(urllib.request, 'urlopen', fake)
    """

    def __init__(self, responses: Dict[str, Any]) -> None:
        # Values may be bytes (single response) or callables that
        # take the request-URL and return bytes.
        self.responses = responses
        self.calls: List[str] = []

    def __call__(self, req: Any, timeout: float = 8.0) -> _FakeResponse:
        # `req` is either a Request (POST) or a URL string (GET).
        if hasattr(req, 'full_url'):
            url = req.full_url
        else:
            url = str(req)
        self.calls.append(url)
        for match, resp in self.responses.items():
            if match in url:
                body = resp(url) if callable(resp) else resp
                return _FakeResponse(body)
        raise AssertionError(
            f'no fake response matched URL {url!r}'
        )


def _gnfinder_body(names: List[Dict[str, Any]]) -> bytes:
    """Build a gnfinder-shaped JSON response body carrying the
    supplied `names` list.  Metadata is minimal — the client only
    reads `names`."""
    return json.dumps({
        'metadata': {'totalNames': len(names)},
        'names': names,
    }).encode('utf-8')


def _gnparser_body(authorship_normalized: str) -> bytes:
    """Build a gnparser-shaped JSON response body carrying an
    ``authorship.normalized`` value.  Empty string = bare name."""
    parsed_flag = bool(authorship_normalized)
    return json.dumps([{
        'parsed': parsed_flag,
        'canonical': {'simple': 'Xy zz'},
        'authorship': (
            {'normalized': authorship_normalized}
            if authorship_normalized else {}
        ),
    }]).encode('utf-8')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAuthoredBinomialInText:
    def test_empty_text_false(self) -> None:
        # No HTTP call; short-circuit on empty.
        assert not authored_binomial_in_text('')

    def test_authored_binomial_fires(self, monkeypatch: Any) -> None:
        """The taxon_83e36037 canonical case: gnfinder finds
        `Trichaptum perrottetii` with wordsAfter carrying the
        authorship; gnparser resolves the authorship."""
        fake = _FakeUrlopen({
            'find': _gnfinder_body([{
                'verbatim': 'Trichaptum perrottetii',
                'wordsAfter': ['(Lév.)', 'Ryvarden,'],
            }]),
            '9081': _gnparser_body('(Lév.) Ryvarden'),
        })
        monkeypatch.setattr(
            'urllib.request.urlopen', fake,
        )
        assert authored_binomial_in_text(
            'Some text with Trichaptum perrottetii (Lév.) Ryvarden here.'
        )

    def test_bare_binomial_does_not_fire(
        self, monkeypatch: Any,
    ) -> None:
        """Habitat / host mentions like `Quercus alba` are legit
        in Description; gnparser reports empty authorship; detector
        stays quiet."""
        fake = _FakeUrlopen({
            'find': _gnfinder_body([{
                'verbatim': 'Quercus alba',
                'wordsAfter': ['trees', 'near'],
            }]),
            '9081': _gnparser_body(''),  # no authorship
        })
        monkeypatch.setattr('urllib.request.urlopen', fake)
        assert not authored_binomial_in_text(
            'on Quercus alba trees near the pond'
        )

    def test_mixed_bare_and_authored_fires(
        self, monkeypatch: Any,
    ) -> None:
        """One bare + one authored → True.  Verifies iteration
        over multiple names and early-exit on the authored one."""
        gnparser_bodies = iter([
            _gnparser_body(''),  # first name: bare
            _gnparser_body('Author'),  # second name: authored
        ])
        fake = _FakeUrlopen({
            'find': _gnfinder_body([
                {'verbatim': 'Quercus alba',
                 'wordsAfter': ['is', 'common']},
                {'verbatim': 'Foo bar',
                 'wordsAfter': ['Author,', '1900']},
            ]),
            '9081': lambda url: next(gnparser_bodies),
        })
        monkeypatch.setattr('urllib.request.urlopen', fake)
        assert authored_binomial_in_text('any text')

    def test_no_names_found_false(self, monkeypatch: Any) -> None:
        """gnfinder returns an empty names list → False.  No
        gnparser call needed."""
        fake = _FakeUrlopen({'find': _gnfinder_body([])})
        monkeypatch.setattr('urllib.request.urlopen', fake)
        assert not authored_binomial_in_text('no names here')

    def test_gnfinder_timeout_raises(
        self, monkeypatch: Any,
    ) -> None:
        """gnfinder network failure propagates as
        GnServiceUnavailable — the triage CLI catches this and
        degrades gracefully."""
        def _raise(*args: Any, **kwargs: Any) -> Any:
            raise urllib.error.URLError('connection refused')
        monkeypatch.setattr('urllib.request.urlopen', _raise)
        with pytest.raises(GnServiceUnavailable):
            authored_binomial_in_text('any text with Foo bar name')

    def test_gnparser_timeout_treated_as_unauthored(
        self, monkeypatch: Any,
    ) -> None:
        """gnparser failure on a specific name is treated as
        unauthored (conservative — don't fire §6:authored_binomial
        on unverified names)."""
        def _routed(req: Any, timeout: float = 8.0) -> _FakeResponse:
            if hasattr(req, 'full_url'):
                url = req.full_url
            else:
                url = str(req)
            if 'find' in url:
                return _FakeResponse(_gnfinder_body([{
                    'verbatim': 'Foo bar',
                    'wordsAfter': ['Author,', '1900'],
                }]))
            # gnparser call fails
            raise urllib.error.URLError('gnparser down')
        monkeypatch.setattr('urllib.request.urlopen', _routed)
        # gnfinder found a candidate, gnparser failed — conservative
        # verdict: False.
        assert not authored_binomial_in_text('Foo bar Author, 1900')

    def test_ocr_corrupt_authority_still_fires(
        self, monkeypatch: Any,
    ) -> None:
        """gnfinder tolerates authority-suffix garbage (empirically
        verified 2026-07-02).  taxon_572d470e's `Brumm., spec. llOU.`
        pattern: binomial intact, authority OCR-corrupted.
        gnparser still extracts an authorship — fires True."""
        fake = _FakeUrlopen({
            'find': _gnfinder_body([{
                'verbatim': 'Saccobolus sphaerosporus',
                'wordsAfter': ['Brumm.,', 'spec.', 'llOU.'],
            }]),
            '9081': _gnparser_body('Brumm.'),
        })
        monkeypatch.setattr('urllib.request.urlopen', fake)
        assert authored_binomial_in_text(
            'Saccobolus sphaerosporus Brumm., spec. llOU.'
        )
