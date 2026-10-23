"""Tests for bin/treatments_to_json.py.

Covers the ``--max-description-tokens`` threshold and the
``--max-new-tokens`` generation cap: how each value is resolved from the
CLI and environment, and that each flag reaches
``translate_treatments_to_json``.

Run with: python -m pytest bin/treatments_to_json_test.py -v
"""

import inspect
import subprocess
from types import SimpleNamespace
import sys
import unittest
from pathlib import Path
from typing import Any

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatments_to_json import (  # type: ignore[import]  # noqa: E402
    parse_batch_sizes,
    validate_backend_args,
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


def backend_args(**overrides: Any) -> SimpleNamespace:
    base = dict(backend='modal', gpu='H100', gpus=1, max_cost=5.0,
                modal_batch_size=64, dry_run=False,
                use_ontology_context=False)
    base.update(overrides)
    return SimpleNamespace(**base)


class TestValidateBackendArgs(unittest.TestCase):

    def test_complete_modal_args(self):
        validate_backend_args(backend_args())

    def test_local_defaults(self):
        validate_backend_args(backend_args(backend='local', gpu=None,
                                           max_cost=None))

    def test_modal_needs_a_gpu(self):
        with self.assertRaisesRegex(ValueError, '--gpu'):
            validate_backend_args(backend_args(gpu=None))

    def test_modal_needs_max_cost_to_spend(self):
        with self.assertRaisesRegex(ValueError, '--max-cost'):
            validate_backend_args(backend_args(max_cost=None))

    def test_dry_run_needs_no_max_cost(self):
        validate_backend_args(backend_args(max_cost=None, dry_run=True))

    def test_positive_values(self):
        for bad in (dict(max_cost=0.0), dict(max_cost=-1.0), dict(gpus=0),
                    dict(modal_batch_size=0)):
            with self.subTest(**bad):
                with self.assertRaises(ValueError):
                    validate_backend_args(backend_args(**bad))

    def test_ontology_context_is_local_only(self):
        with self.assertRaisesRegex(ValueError, 'ontology'):
            validate_backend_args(backend_args(use_ontology_context=True))

    def test_modal_flags_refused_with_local_backend(self):
        for flag in (dict(gpu='H100'), dict(max_cost=5.0)):
            with self.subTest(**flag):
                args = backend_args(backend='local', gpu=None,
                                    max_cost=None)
                for key, value in flag.items():
                    setattr(args, key, value)
                with self.assertRaises(ValueError):
                    validate_backend_args(args)


class TestModalFlagWiring(unittest.TestCase):

    def test_help_lists_the_flags(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, timeout=180, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--backend', '--gpu', '--gpus', '--max-cost',
                     '--modal-batch-size'):
            self.assertIn(flag, result.stdout)
        for gpu in ('H100', 'A100-40GB', 'L4'):
            self.assertIn(gpu, result.stdout)

    def test_translate_accepts_the_backend_settings(self):
        params = inspect.signature(translate_treatments_to_json).parameters
        self.assertEqual(params['backend'].default, 'local')
        self.assertIsNone(params['gpu'].default)
        self.assertEqual(params['gpus'].default, 1)
        self.assertIsNone(params['max_cost'].default)
        self.assertEqual(params['modal_batch_size'].default, 64)


CALIBRATE_NOT_YET = pytest.mark.xfail(
    strict=True, reason="--calibrate not implemented yet")


@CALIBRATE_NOT_YET
class TestParseBatchSizes(unittest.TestCase):

    def test_list(self):
        self.assertEqual(parse_batch_sizes('64,128,256'), (64, 128, 256))

    def test_spaces_and_single(self):
        self.assertEqual(parse_batch_sizes(' 32 , 64'), (32, 64))
        self.assertEqual(parse_batch_sizes('128'), (128,))

    def test_rejects(self):
        for bad in ('', '64,', '64,abc', '0', '64,-1', '1.5'):
            with self.subTest(text=bad):
                with self.assertRaises(ValueError):
                    parse_batch_sizes(bad)


class TestValidateCalibrateArgs(unittest.TestCase):

    def test_calibrate_with_modal_and_a_budget(self):
        validate_backend_args(backend_args(calibrate=True,
                                           calibrate_records=256))

    @CALIBRATE_NOT_YET
    def test_calibrate_needs_modal(self):
        with self.assertRaisesRegex(ValueError, '--calibrate'):
            validate_backend_args(backend_args(
                backend='local', gpu=None, max_cost=None, calibrate=True,
                calibrate_records=256))

    @CALIBRATE_NOT_YET
    def test_calibrate_spends_so_no_dry_run(self):
        with self.assertRaisesRegex(ValueError, '--dry-run'):
            validate_backend_args(backend_args(
                calibrate=True, calibrate_records=256, dry_run=True,
                max_cost=None))

    @CALIBRATE_NOT_YET
    def test_calibrate_records_positive(self):
        with self.assertRaises(ValueError):
            validate_backend_args(backend_args(calibrate=True,
                                               calibrate_records=0))


@CALIBRATE_NOT_YET
class TestCalibrateFlagWiring(unittest.TestCase):

    def test_help_lists_the_flags(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, timeout=180, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--calibrate', '--calibrate-records',
                     '--calibrate-batch-sizes', '--calibration-file'):
            self.assertIn(flag, result.stdout)

    def test_translate_accepts_the_calibration_settings(self):
        params = inspect.signature(translate_treatments_to_json).parameters
        self.assertIs(params['calibrate'].default, False)
        self.assertEqual(params['calibrate_records'].default, 256)
        self.assertEqual(params['calibrate_batch_sizes'].default,
                         (64, 128, 256))
        self.assertIsNone(params['calibration_file'].default)


if __name__ == "__main__":
    unittest.main()
