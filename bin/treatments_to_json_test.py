"""Tests for bin/treatments_to_json.py.

Covers the ``--max-description-tokens`` threshold and the
``--max-new-tokens`` generation cap: how each value is resolved from the
CLI and environment, and that each flag reaches
``translate_treatments_to_json``.

Run with: python -m pytest bin/treatments_to_json_test.py -v
"""

import inspect
import subprocess
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatments_to_json import (  # type: ignore[import]  # noqa: E402
    resolve_max_description_tokens,
    resolve_max_new_tokens,
    translate_treatments_to_json,
)
from treatments_json_translator import (  # noqa: E402
    TreatmentsJSONTranslator,
)


SCRIPT = Path(__file__).resolve().parent / "treatments_to_json.py"


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


class TestResolveMaxNewTokens(unittest.TestCase):

    def test_unset_means_translator_default(self):
        self.assertEqual(
            resolve_max_new_tokens(None, {}),
            TreatmentsJSONTranslator.DEFAULT_MAX_NEW_TOKENS)

    def test_cli_value(self):
        self.assertEqual(resolve_max_new_tokens(3072, {}), 3072)

    def test_environment_used_when_cli_absent(self):
        self.assertEqual(
            resolve_max_new_tokens(None, {"MAX_NEW_TOKENS": "2048"}), 2048)

    def test_cli_beats_environment(self):
        self.assertEqual(
            resolve_max_new_tokens(3072, {"MAX_NEW_TOKENS": "2048"}), 3072)

    def test_empty_environment_value_means_unset(self):
        self.assertEqual(
            resolve_max_new_tokens(None, {"MAX_NEW_TOKENS": " "}),
            TreatmentsJSONTranslator.DEFAULT_MAX_NEW_TOKENS)

    def test_rejects_non_positive_values(self):
        for cli, env in ((0, {}), (-5, {}),
                         (None, {"MAX_NEW_TOKENS": "0"})):
            with self.subTest(cli=cli, env=env):
                with self.assertRaises(ValueError):
                    resolve_max_new_tokens(cli, env)

    def test_rejects_non_integer_environment_value(self):
        with self.assertRaises(ValueError):
            resolve_max_new_tokens(None, {"MAX_NEW_TOKENS": "3k"})


class TestMaxNewTokensWiring(unittest.TestCase):

    def test_help_lists_the_flag(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, timeout=180, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-new-tokens", result.stdout)

    def test_translate_accepts_the_cap(self):
        params = inspect.signature(translate_treatments_to_json).parameters
        self.assertIn("max_new_tokens", params)
        self.assertEqual(
            params["max_new_tokens"].default,
            TreatmentsJSONTranslator.DEFAULT_MAX_NEW_TOKENS)


if __name__ == "__main__":
    unittest.main()
