"""Tests for treatments_to_structured/modal_calibration.py.

Run with: python -m pytest treatments_to_structured/modal_calibration_test.py -v
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest

from treatments_to_structured import modal_inference as mi
from treatments_to_structured.modal_calibration import (
    CALIBRATION_FILE,
    DEFAULT_BATCH_SIZES,
    FALLBACK_BATCH_SIZE,
    SCHEMA_VERSION,
    best_config,
    calibration_key,
    compare_gpus,
    estimate,
    find_calibration,
    load_calibrations,
    resolve_batch_size,
    sample_stratified,
    save_calibration,
    summarize,
)
from treatments_to_structured.modal_models import MODEL_ID, MODEL_REVISION


def raw(tokens_in: int, tokens_out: int,
        finish: str = 'stop') -> Dict[str, Any]:
    return {'_id': 'x', 'text': '{}', 'finish_reason': finish,
            'tokens_in': tokens_in, 'tokens_out': tokens_out}


def entry(gpu: str, key: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    return {'gpu': gpu, 'key': key, 'measured_at': '2026-10-23T00:00:00',
            'startup_s': 60.0, 'configs': [], **extra}


class TestConstants:

    def test_file_lives_in_the_package(self) -> None:
        assert CALIBRATION_FILE.name == 'modal_calibration.json'
        assert CALIBRATION_FILE.parent.name == 'treatments_to_structured'

    def test_default_batch_sizes(self) -> None:
        assert DEFAULT_BATCH_SIZES == (64, 128, 256)


class TestCalibrationKey:

    def test_names_what_speed_depends_on(self) -> None:
        assert calibration_key(3072) == {
            'engine': mi.ENGINE, 'engine_version': mi.VLLM_VERSION,
            'image': mi.VLLM_BASE_IMAGE, 'model': MODEL_ID,
            'model_revision': MODEL_REVISION, 'dtype': mi.DTYPE,
            'decoding': mi.DECODING, 'max_tokens': 3072,
            'max_model_len': mi.MAX_MODEL_LEN,
        }

    def test_max_tokens_matters(self) -> None:
        assert calibration_key(3072) != calibration_key(1024)


class TestSampleStratified:

    ITEMS = [{'_id': f'i{n:03}'} for n in range(100)]
    COUNTS = list(range(100))           # i000 shortest ... i099 longest

    def test_size_and_determinism(self) -> None:
        a = sample_stratified(self.ITEMS, 20, self.COUNTS, seed=7)
        b = sample_stratified(self.ITEMS, 20, self.COUNTS, seed=7)
        assert len(a) == 20 and a == b
        assert len({i['_id'] for i in a}) == 20

    def test_seed_changes_the_draw(self) -> None:
        assert (sample_stratified(self.ITEMS, 20, self.COUNTS, seed=1)
                != sample_stratified(self.ITEMS, 20, self.COUNTS, seed=2))

    def test_every_length_bin_contributes(self) -> None:
        picked = sample_stratified(self.ITEMS, 20, self.COUNTS, seed=3,
                                   bins=10)
        bins = sorted({int(i['_id'][1:]) // 10 for i in picked})
        assert bins == list(range(10))

    def test_n_at_least_len_returns_everything(self) -> None:
        picked = sample_stratified(self.ITEMS[:5], 9, self.COUNTS[:5],
                                   seed=0)
        assert sorted(i['_id'] for i in picked) == [
            'i000', 'i001', 'i002', 'i003', 'i004']

    def test_zero(self) -> None:
        assert sample_stratified(self.ITEMS, 0, self.COUNTS, seed=0) == []


class TestSummarize:

    def test_metrics(self) -> None:
        raws = [raw(1000, 400), raw(2000, 600), raw(500, 3072, 'length'),
                raw(1500, 528)]
        assert summarize(raws, elapsed_s=10.0, batch_size=64) == {
            'batch_size': 64, 'records': 4, 'elapsed_s': 10.0,
            'records_per_s': 0.4, 'output_tokens_per_s': 460.0,
            'cut_off_rate': 0.25, 'mean_output_tokens': 1150.0,
            'mean_input_tokens': 1250.0,
        }

    def test_refuses_empty_or_untimed(self) -> None:
        with pytest.raises(ValueError):
            summarize([], elapsed_s=1.0, batch_size=64)
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                summarize([raw(1, 1)], elapsed_s=bad, batch_size=64)


class TestFile:

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert load_calibrations(tmp_path / 'none.json') == []

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / 'cal.json'
        e = entry('H100', {'k': 1})
        save_calibration(e, path)
        assert load_calibrations(path) == [e]
        on_disk = json.loads(path.read_text())
        assert on_disk['_schema_version'] == SCHEMA_VERSION

    def test_new_measurement_replaces_the_gpus_entry(
            self, tmp_path: Path) -> None:
        path = tmp_path / 'cal.json'
        save_calibration(entry('L4', {'k': 1}), path)
        save_calibration(entry('H100', {'k': 1}), path)
        save_calibration(entry('H100', {'k': 2}, startup_s=30.0), path)
        loaded = load_calibrations(path)
        assert [e['gpu'] for e in loaded] == ['H100', 'L4']
        assert loaded[0]['key'] == {'k': 2}
        assert loaded[0]['startup_s'] == 30.0

    def test_no_temporary_file_left(self, tmp_path: Path) -> None:
        path = tmp_path / 'cal.json'
        save_calibration(entry('H100', {'k': 1}), path)
        assert sorted(p.name for p in tmp_path.iterdir()) == ['cal.json']

    def test_other_schema_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / 'cal.json'
        path.write_text(json.dumps({'_schema_version': 99, 'entries': []}))
        with pytest.raises(ValueError):
            load_calibrations(path)


class TestFindCalibration:

    ENTRIES = [entry('H100', {'k': 1}), entry('L4', {'k': 1})]

    def test_ok(self) -> None:
        assert find_calibration(self.ENTRIES, 'H100', {'k': 1}) == (
            'ok', self.ENTRIES[0])

    def test_stale(self) -> None:
        """Measured under another vLLM, model or decoding setting."""
        assert find_calibration(self.ENTRIES, 'H100', {'k': 2}) == (
            'stale', self.ENTRIES[0])

    def test_missing(self) -> None:
        assert find_calibration(self.ENTRIES, 'A100-40GB', {'k': 1}) == (
            'missing', None)


class TestBestConfig:

    def test_most_records_per_second(self) -> None:
        e = entry('H100', {}, configs=[
            {'batch_size': 64, 'records_per_s': 1.0},
            {'batch_size': 256, 'records_per_s': 2.5},
            {'batch_size': 128, 'records_per_s': 2.0}])
        assert best_config(e)['batch_size'] == 256

    def test_tie_goes_to_the_smaller_batch(self) -> None:
        e = entry('H100', {}, configs=[
            {'batch_size': 128, 'records_per_s': 2.0},
            {'batch_size': 64, 'records_per_s': 2.0}])
        assert best_config(e)['batch_size'] == 64

    def test_no_configs(self) -> None:
        with pytest.raises(ValueError):
            best_config(entry('H100', {}))


# ---------------------------------------------------------------------------
# Tranche 4: estimates for --dry-run
# ---------------------------------------------------------------------------

ESTIMATE_NOT_YET = pytest.mark.xfail(
    strict=True, reason="estimate / compare_gpus / resolve_batch_size "
                        "not implemented yet")


def measured(gpu: str, key: Dict[str, Any], startup_s: float = 100.0,
             best_rps: float = 5.0) -> Dict[str, Any]:
    return entry(gpu, key, startup_s=startup_s, configs=[
        {'batch_size': 64, 'records_per_s': best_rps / 2,
         'cut_off_rate': 0.02},
        {'batch_size': 256, 'records_per_s': best_rps,
         'cut_off_rate': 0.01}])


@ESTIMATE_NOT_YET
class TestEstimate:

    def test_arithmetic(self) -> None:
        est = estimate(1000, 'H100', 2, measured('H100', {'k': 1}))
        rate = mi.rate_per_second('H100')
        work_s = 1000 / 5.0                        # 200 s
        gpu_s = work_s + 2 * (100.0 + mi.SCALEDOWN_WINDOW_S)
        assert est['records'] == 1000
        assert est['gpu'] == 'H100' and est['gpus'] == 2
        assert est['batch_size'] == 256
        assert est['records_per_s'] == 5.0
        assert est['gpu_hours'] == pytest.approx(gpu_s / 3600)
        assert est['wall_hours'] == pytest.approx((work_s / 2 + 100) / 3600)
        assert est['cost'] == pytest.approx(gpu_s * rate)
        assert est['expected_cut_offs'] == pytest.approx(10.0)
        assert est['suggested_max_cost'] == math.ceil(
            est['cost'] * 1.25)
        assert est['prices_read_on'] == mi.PRICES_READ_ON
        assert est['measured_at'] == '2026-10-23T00:00:00'

    def test_more_gpus_cost_more_and_finish_sooner(self) -> None:
        e = measured('H100', {'k': 1})
        one, four = estimate(10000, 'H100', 1, e), estimate(
            10000, 'H100', 4, e)
        assert four['cost'] > one['cost']
        assert four['wall_hours'] < one['wall_hours']

    def test_nothing_to_do(self) -> None:
        est = estimate(0, 'H100', 1, measured('H100', {'k': 1}))
        assert est['cost'] == 0.0 and est['gpu_hours'] == 0.0
        assert est['wall_hours'] == 0.0

    def test_rejects_bad_settings(self) -> None:
        with pytest.raises(ValueError):
            estimate(100, 'T4', 1, measured('T4', {'k': 1}))
        with pytest.raises(ValueError):
            estimate(100, 'H100', 0, measured('H100', {'k': 1}))


@ESTIMATE_NOT_YET
class TestCompareGpus:

    def test_cheapest_first_with_status(self) -> None:
        key = {'k': 1}
        entries = [measured('A100-40GB', {'k': 0}, best_rps=2.3),
                   measured('H100', key, best_rps=6.2)]
        rows = compare_gpus(42000, 1, entries, key)
        assert [r['gpu'] for r in rows] == ['H100', 'A100-40GB']
        assert [r['status'] for r in rows] == ['ok', 'stale']
        assert rows[0]['cost'] < rows[1]['cost']

    def test_uncalibrated_gpus_are_left_out(self) -> None:
        rows = compare_gpus(100, 1, [measured('H100', {'k': 1})], {'k': 1})
        assert [r['gpu'] for r in rows] == ['H100']

    def test_nothing_calibrated(self) -> None:
        assert compare_gpus(100, 1, [], {'k': 1}) == []


@ESTIMATE_NOT_YET
class TestResolveBatchSize:

    def test_command_line_wins(self) -> None:
        assert resolve_batch_size(32, 'ok', measured('H100', {})) == 32

    def test_calibrated_best(self) -> None:
        assert resolve_batch_size(None, 'ok', measured('H100', {})) == 256

    def test_stale_calibration_still_informs(self) -> None:
        assert resolve_batch_size(None, 'stale',
                                  measured('H100', {})) == 256

    def test_fallback(self) -> None:
        assert resolve_batch_size(None, 'missing', None) == (
            FALLBACK_BATCH_SIZE)
        assert FALLBACK_BATCH_SIZE == 64
