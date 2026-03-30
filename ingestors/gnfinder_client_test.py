"""Tests for ingestors/gnfinder_client.py.

Uses mocked HTTP responses; no live network required.

Run with: python -m pytest ingestors/gnfinder_client_test.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.gnfinder_client import NameSpan, _parse_response, find_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(body: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status = MagicMock()
    return resp


_SAMPLE_BODY = {
    "names": [
        {
            "start": 0,
            "end": 16,
            "verbatim": "Amanita muscaria",
            "oddsLog10": 8.3,
            "annotNomen": "sp. nov.",
            "annotNomenType": "SP_NOV",
            "bestResult": {
                "name": "Amanita muscaria",
                "cardinality": 2,
            },
        },
        {
            "start": 50,
            "end": 65,
            "verbatim": "Amanita phalloides",
            "oddsLog10": 7.1,
            "annotNomen": "",
            "annotNomenType": "NO_ANNOT",
            "bestResult": {
                "name": "Amanita phalloides",
                "cardinality": 2,
            },
        },
    ]
}


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse(unittest.TestCase):
    """_parse_response maps gnfinder JSON to NameSpan objects."""

    def test_returns_list_of_name_spans(self) -> None:
        result = _parse_response(_SAMPLE_BODY)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(n, NameSpan) for n in result))

    def test_parses_start_end_verbatim(self) -> None:
        result = _parse_response(_SAMPLE_BODY)
        self.assertEqual(result[0].start, 0)
        self.assertEqual(result[0].end, 16)
        self.assertEqual(result[0].verbatim, "Amanita muscaria")

    def test_parses_canonical_from_best_result(self) -> None:
        result = _parse_response(_SAMPLE_BODY)
        self.assertEqual(result[0].canonical, "Amanita muscaria")

    def test_parses_cardinality(self) -> None:
        result = _parse_response(_SAMPLE_BODY)
        self.assertEqual(result[0].cardinality, 2)

    def test_parses_odds(self) -> None:
        result = _parse_response(_SAMPLE_BODY)
        self.assertAlmostEqual(result[0].odds_log10, 8.3)

    def test_parses_annot_nomen_and_type(self) -> None:
        result = _parse_response(_SAMPLE_BODY)
        self.assertEqual(result[0].annot_nomen, "sp. nov.")
        self.assertEqual(result[0].annot_nomen_type, "SP_NOV")

    def test_sorted_by_start(self) -> None:
        body = {
            "names": [
                {"start": 50, "end": 65, "verbatim": "B", "bestResult": {"name": "B", "cardinality": 1}},
                {"start": 0, "end": 14, "verbatim": "A", "bestResult": {"name": "A", "cardinality": 1}},
            ]
        }
        result = _parse_response(body)
        self.assertEqual(result[0].start, 0)
        self.assertEqual(result[1].start, 50)

    def test_empty_names_array(self) -> None:
        result = _parse_response({"names": []})
        self.assertEqual(result, [])

    def test_missing_names_key(self) -> None:
        result = _parse_response({})
        self.assertEqual(result, [])

    def test_no_best_result_falls_back_to_verbatim(self) -> None:
        body = {
            "names": [
                {"start": 0, "end": 5, "verbatim": "Amanita"},
            ]
        }
        result = _parse_response(body)
        self.assertEqual(result[0].canonical, "Amanita")

    def test_no_annot_nomen_defaults(self) -> None:
        body = {
            "names": [
                {"start": 0, "end": 5, "verbatim": "Amanita",
                 "bestResult": {"name": "Amanita", "cardinality": 1}},
            ]
        }
        result = _parse_response(body)
        self.assertEqual(result[0].annot_nomen, "")
        self.assertEqual(result[0].annot_nomen_type, "NO_ANNOT")


# ---------------------------------------------------------------------------
# find_names (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFindNames(unittest.TestCase):
    """find_names sends POST and returns parsed NameSpan objects."""

    def _call(self, resp: MagicMock, **kwargs) -> list:
        with patch("ingestors.gnfinder_client.requests") as mock_req:
            mock_req.post.return_value = resp
            mock_req.HTTPError = Exception
            mock_req.Timeout = TimeoutError
            mock_req.ConnectionError = ConnectionError
            return find_names("some text", retries=0, **kwargs)

    def test_returns_name_spans(self) -> None:
        resp = _mock_response(_SAMPLE_BODY)
        result = self._call(resp)
        self.assertEqual(len(result), 2)

    def test_posts_to_url(self) -> None:
        resp = _mock_response(_SAMPLE_BODY)
        with patch("ingestors.gnfinder_client.requests") as mock_req:
            mock_req.post.return_value = resp
            mock_req.HTTPError = Exception
            mock_req.Timeout = TimeoutError
            mock_req.ConnectionError = ConnectionError
            find_names("text", gnfinder_url="http://myserver/api/v1/find", retries=0)
            url = mock_req.post.call_args[0][0]
            self.assertEqual(url, "http://myserver/api/v1/find")

    def test_payload_contains_text(self) -> None:
        resp = _mock_response(_SAMPLE_BODY)
        with patch("ingestors.gnfinder_client.requests") as mock_req:
            mock_req.post.return_value = resp
            mock_req.HTTPError = Exception
            mock_req.Timeout = TimeoutError
            mock_req.ConnectionError = ConnectionError
            find_names("my article text", retries=0)
            kwargs = mock_req.post.call_args[1]
            self.assertEqual(kwargs["json"]["text"], "my article text")

    def test_retry_on_error_then_success(self) -> None:
        fail_resp = _mock_response({}, 500)
        ok_resp = _mock_response(_SAMPLE_BODY, 200)
        with patch("ingestors.gnfinder_client.requests") as mock_req:
            with patch("ingestors.gnfinder_client.time.sleep"):
                mock_req.post.side_effect = [fail_resp, ok_resp]
                mock_req.HTTPError = Exception
                mock_req.Timeout = TimeoutError
                mock_req.ConnectionError = ConnectionError
                result = find_names("text", retries=1)
        self.assertEqual(len(result), 2)

    def test_raises_after_exhausting_retries(self) -> None:
        fail_resp = _mock_response({}, 500)
        with patch("ingestors.gnfinder_client.requests") as mock_req:
            with patch("ingestors.gnfinder_client.time.sleep"):
                mock_req.post.return_value = fail_resp
                mock_req.HTTPError = Exception
                mock_req.Timeout = TimeoutError
                mock_req.ConnectionError = ConnectionError
                with self.assertRaises(Exception):
                    find_names("text", retries=1)


if __name__ == "__main__":
    unittest.main()
