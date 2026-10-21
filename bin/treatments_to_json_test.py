"""Tests for bin/treatments_to_json.py.

Covers the ``--max-description-tokens`` threshold: how its value is
resolved from the CLI and environment, and that the flag reaches
``translate_treatments_to_json``.

Run with: python -m pytest bin/treatments_to_json_test.py -v
"""

import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatments_to_json import (  # type: ignore[import]  # noqa: E402
    resolve_max_description_tokens,
    translate_treatments_to_json,
)

NOT_YET = pytest.mark.xfail(
    strict=True, reason="description token threshold not implemented yet")

SCRIPT = Path(__file__).resolve().parent / "treatments_to_json.py"


@NOT_YET
class TestResolveMaxDescriptionTokens(unittest.TestCase):

    def test_unset_means_no_threshold(self):
        self.assertIsNone(resolve_max_description_tokens(None, {}))

    def test_cli_value(self):
        self.assertEqual(resolve_max_description_tokens(4800, {}), 4800)

    def test_environment_used_when_cli_absent(self):
        self.assertEqual(
            resolve_max_description_tokens(
                None, {"MAX_DESCRIPTION_TOKENS": "2000"}),
            2000)

    def test_cli_beats_environment(self):
        self.assertEqual(
            resolve_max_description_tokens(
                4800, {"MAX_DESCRIPTION_TOKENS": "2000"}),
            4800)

    def test_empty_environment_value_means_unset(self):
        self.assertIsNone(
            resolve_max_description_tokens(
                None, {"MAX_DESCRIPTION_TOKENS": ""}))

    def test_rejects_non_positive_values(self):
        for cli, env in ((0, {}), (-5, {}),
                         (None, {"MAX_DESCRIPTION_TOKENS": "0"}),
                         (None, {"MAX_DESCRIPTION_TOKENS": "-1"})):
            with self.subTest(cli=cli, env=env):
                with self.assertRaises(ValueError):
                    resolve_max_description_tokens(cli, env)

    def test_rejects_non_integer_environment_value(self):
        with self.assertRaises(ValueError):
            resolve_max_description_tokens(
                None, {"MAX_DESCRIPTION_TOKENS": "4.8k"})


@NOT_YET
class TestFlagWiring(unittest.TestCase):

    def test_help_lists_the_flag(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, timeout=180, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-description-tokens", result.stdout)

    def test_translate_accepts_the_threshold(self):
        params = inspect.signature(translate_treatments_to_json).parameters
        self.assertIn("max_description_tokens", params)
        self.assertIsNone(params["max_description_tokens"].default)


if __name__ == "__main__":
    unittest.main()
