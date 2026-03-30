"""Tests for ingestors/gnparser_client.py.

Uses mocked HTTP responses; no live network required.

Run with: python -m pytest ingestors/gnparser_client_test.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.gnparser_client import (
    ParsedAuthorship,
    _batch_parse,
    _extract_authors,
    parse_authorship_after_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(body, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status = MagicMock()
    return resp


# A realistic gnparser response for "Xus (L.) Lam."
_PARSED_WITH_AUTHORSHIP = [
    {
        "id": "1",
        "verbatim": "Xus (L.) Lam.",
        "canonicalName": {"full": "Xus", "simple": "Xus"},
        "authorship": {
            "verbatim": "(L.) Lam.",
            "year": {},
            "combinedAuthors": [
                {"name": "L."},
                {"name": "Lam."},
            ],
        },
    }
]

_PARSED_WITH_YEAR = [
    {
        "id": "1",
        "verbatim": "Xus Lam. 1783",
        "canonicalName": {"full": "Xus"},
        "authorship": {
            "verbatim": "Lam. 1783",
            "year": {"year": "1783"},
            "combinedAuthors": [{"name": "Lam."}],
        },
    }
]

_PARSED_NO_AUTHORSHIP = [
    {
        "id": "1",
        "verbatim": "Xus",
        "canonicalName": {"full": "Xus"},
        "authorship": {},
    }
]


# ---------------------------------------------------------------------------
# _extract_authors
# ---------------------------------------------------------------------------


class TestExtractAuthors(unittest.TestCase):

    def test_extracts_combined_authors(self) -> None:
        auth = {"combinedAuthors": [{"name": "L."}, {"name": "Lam."}]}
        self.assertEqual(_extract_authors(auth), ["L.", "Lam."])

    def test_empty_authorship(self) -> None:
        self.assertEqual(_extract_authors({}), [])

    def test_authors_key_fallback(self) -> None:
        auth = {"authors": [{"name": "Banks"}]}
        self.assertEqual(_extract_authors(auth), ["Banks"])

    def test_combined_authors_takes_priority_over_authors(self) -> None:
        auth = {
            "combinedAuthors": [{"name": "A"}],
            "authors": [{"name": "B"}],
        }
        self.assertEqual(_extract_authors(auth), ["A"])


# ---------------------------------------------------------------------------
# parse_authorship_after_name (mocked)
# ---------------------------------------------------------------------------


class TestParseAuthorshipAfterName(unittest.TestCase):
    """parse_authorship_after_name wraps gnparser to find authorship windows."""

    def _call(self, window: str, parsed_body) -> object:
        resp = _mock_response(parsed_body)
        with patch("ingestors.gnparser_client.requests") as mock_req:
            mock_req.post.return_value = resp
            mock_req.HTTPError = Exception
            mock_req.Timeout = TimeoutError
            mock_req.ConnectionError = ConnectionError
            return parse_authorship_after_name(window, retries=0)

    def test_returns_parsed_authorship(self) -> None:
        result = self._call("(L.) Lam. is a fungus", _PARSED_WITH_AUTHORSHIP)
        self.assertIsInstance(result, ParsedAuthorship)

    def test_verbatim_matches_authorship(self) -> None:
        result = self._call("(L.) Lam. is a fungus", _PARSED_WITH_AUTHORSHIP)
        self.assertEqual(result.verbatim, "(L.) Lam.")

    def test_authors_extracted(self) -> None:
        result = self._call("(L.) Lam. is a fungus", _PARSED_WITH_AUTHORSHIP)
        self.assertIn("L.", result.authors)
        self.assertIn("Lam.", result.authors)

    def test_year_extracted(self) -> None:
        result = self._call("Lam. 1783 was described", _PARSED_WITH_YEAR)
        self.assertEqual(result.year, "1783")

    def test_no_authorship_returns_none(self) -> None:
        result = self._call("grows in forests", _PARSED_NO_AUTHORSHIP)
        self.assertIsNone(result)

    def test_empty_window_returns_none(self) -> None:
        # Should not call the API at all
        with patch("ingestors.gnparser_client.requests") as mock_req:
            mock_req.post.return_value = _mock_response([])
            mock_req.HTTPError = Exception
            mock_req.Timeout = TimeoutError
            mock_req.ConnectionError = ConnectionError
            result = parse_authorship_after_name("   ", retries=0)
        self.assertIsNone(result)
        mock_req.post.assert_not_called()

    def test_sends_synthetic_name_with_dummy_prefix(self) -> None:
        resp = _mock_response(_PARSED_WITH_AUTHORSHIP)
        with patch("ingestors.gnparser_client.requests") as mock_req:
            mock_req.post.return_value = resp
            mock_req.HTTPError = Exception
            mock_req.Timeout = TimeoutError
            mock_req.ConnectionError = ConnectionError
            parse_authorship_after_name("(L.) Lam.", retries=0)
            payload = mock_req.post.call_args[1]["json"]
        self.assertTrue(payload[0].startswith("Xus "))


# ---------------------------------------------------------------------------
# _batch_parse retry logic
# ---------------------------------------------------------------------------


class TestBatchParseRetry(unittest.TestCase):

    def test_retries_on_failure_then_succeeds(self) -> None:
        fail_resp = _mock_response({}, 500)
        ok_resp = _mock_response(_PARSED_WITH_AUTHORSHIP)
        with patch("ingestors.gnparser_client.requests") as mock_req:
            with patch("ingestors.gnparser_client.time.sleep"):
                mock_req.post.side_effect = [fail_resp, ok_resp]
                mock_req.HTTPError = Exception
                mock_req.Timeout = TimeoutError
                mock_req.ConnectionError = ConnectionError
                result = _batch_parse(["Xus (L.) Lam."], retries=1)
        self.assertEqual(len(result), 1)

    def test_raises_after_all_retries_exhausted(self) -> None:
        fail_resp = _mock_response({}, 500)
        with patch("ingestors.gnparser_client.requests") as mock_req:
            with patch("ingestors.gnparser_client.time.sleep"):
                mock_req.post.return_value = fail_resp
                mock_req.HTTPError = Exception
                mock_req.Timeout = TimeoutError
                mock_req.ConnectionError = ConnectionError
                with self.assertRaises(Exception):
                    _batch_parse(["Xus"], retries=1)


if __name__ == "__main__":
    unittest.main()
