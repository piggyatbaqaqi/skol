"""Tests for treatments_to_structured/modal_inference.py.

Run with: python -m pytest treatments_to_structured/modal_inference_test.py -v
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from treatments_to_structured.modal_inference import (
    CACHE_MOUNT,
    GPU_PRICES,
    SpendGuard,
    cache_env,
    interpret_output,
    make_batches,
    provenance,
    rate_per_second,
    raw_from_vllm,
    run_batches,
    structured_doc,
    taxon_doc_id,
)

FEATURES = {"pileus": {"shape": ["convex"]}}


class FakeClock:

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def accept(features: Dict[str, Any]) -> bool:
    return True


def reject(features: Dict[str, Any]) -> bool:
    raise ValueError("schema says no")


def raw(text: str, finish_reason: str = "stop") -> Dict[str, Any]:
    return {"_id": "t", "text": text, "finish_reason": finish_reason,
            "tokens_in": 10, "tokens_out": 5}


# ---------------------------------------------------------------------------
# Prices and the spending guard
# ---------------------------------------------------------------------------


class TestPrices:

    def test_table_holds_the_candidate_gpus(self) -> None:
        assert set(GPU_PRICES) == {'H100', 'A100-80GB', 'A100-40GB', 'L40S',
                                   'A10G', 'L4'}

    def test_rate_adds_cpu_and_memory(self) -> None:
        cpu_mem = 2 * 0.0000131 + 16 * 0.00000222
        assert rate_per_second('H100') == pytest.approx(0.001097 + cpu_mem)

    def test_unknown_gpu(self) -> None:
        with pytest.raises(ValueError):
            rate_per_second('T4')


class TestSpendGuard:

    def test_nothing_spent_before_start(self) -> None:
        clock = FakeClock()
        guard = SpendGuard(5.0, 'H100', 2, clock=clock)
        clock.now += 3600
        assert guard.spent_upper_bound() == 0.0
        assert guard.allows_more() is True

    def test_bound_is_elapsed_times_gpus_times_rate(self) -> None:
        clock = FakeClock()
        guard = SpendGuard(100.0, 'L4', 3, clock=clock)
        guard.start()
        clock.now += 600
        assert guard.spent_upper_bound() == pytest.approx(
            600 * 3 * rate_per_second('L4'))

    def test_stops_at_the_limit(self) -> None:
        clock = FakeClock()
        guard = SpendGuard(1.0, 'H100', 1, clock=clock)
        guard.start()
        seconds_to_limit = 1.0 / rate_per_second('H100')
        clock.now += seconds_to_limit - 1
        assert guard.allows_more() is True
        clock.now += 2
        assert guard.allows_more() is False

    def test_rejects_bad_settings(self) -> None:
        for args in ((0.0, 'H100', 1), (-1.0, 'H100', 1),
                     (5.0, 'T4', 1), (5.0, 'H100', 0)):
            with pytest.raises(ValueError):
                SpendGuard(*args)


# ---------------------------------------------------------------------------
# Batching and vLLM output
# ---------------------------------------------------------------------------


class TestMakeBatches:

    def test_consecutive_batches(self) -> None:
        items = [{"_id": str(i)} for i in range(5)]
        batches = make_batches(items, 2)
        assert [[i["_id"] for i in b] for b in batches] == [
            ["0", "1"], ["2", "3"], ["4"]]

    def test_empty(self) -> None:
        assert make_batches([], 3) == []

    def test_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            make_batches([{"_id": "a"}], 0)


class TestRawFromVllm:

    def test_fields(self) -> None:
        output = SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            outputs=[SimpleNamespace(text='{"a": ["b"]}',
                                     finish_reason='stop',
                                     token_ids=[7, 8])])
        assert raw_from_vllm('taxon_a', output) == {
            '_id': 'taxon_a', 'text': '{"a": ["b"]}',
            'finish_reason': 'stop', 'tokens_in': 3, 'tokens_out': 2}


class TestInterpretOutput:

    def test_ok(self) -> None:
        result = interpret_output(raw(json.dumps(FEATURES)), accept)
        assert result == {'features': FEATURES, 'json_valid': True,
                          'outcome': 'ok'}

    def test_cut_off_at_max_tokens_saves_empty(self) -> None:
        """Decision 5 of the design: a cut-off record is saved as {}."""
        result = interpret_output(raw('{"pileus": {', 'length'), accept)
        assert result == {'features': {}, 'json_valid': False,
                          'outcome': 'cut_off'}

    def test_cut_off_even_if_the_text_happens_to_parse(self) -> None:
        result = interpret_output(raw(json.dumps(FEATURES), 'length'),
                                  accept)
        assert result['outcome'] == 'cut_off'
        assert result['features'] == {}

    def test_unparseable(self) -> None:
        result = interpret_output(raw('not json'), accept)
        assert result == {'features': {}, 'json_valid': False,
                          'outcome': 'invalid'}

    def test_not_an_object(self) -> None:
        assert interpret_output(raw('[1, 2]'), accept)['outcome'] == (
            'invalid')

    def test_empty_object_is_not_valid(self) -> None:
        """The local path's is_valid_json treats {} as invalid."""
        result = interpret_output(raw('{}'), accept)
        assert result['json_valid'] is False

    def test_schema_rejection(self) -> None:
        result = interpret_output(raw(json.dumps(FEATURES)), reject)
        assert result == {'features': {}, 'json_valid': False,
                          'outcome': 'invalid'}

    def test_schema_returning_false_is_rejection(self) -> None:
        """TaxonomySchema.validate mostly raises, but one branch returns
        _basic_validate's boolean."""
        result = interpret_output(raw(json.dumps(FEATURES)),
                                  lambda features: False)
        assert result['outcome'] == 'invalid'


# ---------------------------------------------------------------------------
# run_batches
# ---------------------------------------------------------------------------


class AlwaysAllow:

    def allows_more(self) -> bool:
        return True


class AllowFirst:
    """Allows the first ``n`` submissions."""

    def __init__(self, n: int) -> None:
        self.left = n

    def allows_more(self) -> bool:
        self.left -= 1
        return self.left >= 0


def batches_of(n: int) -> List[List[Dict[str, Any]]]:
    return [[{"_id": f"b{i}"}] for i in range(n)]


def echo(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [raw(json.dumps(FEATURES)) | {"_id": item["_id"]}
            for item in batch]


class TestRunBatches:

    def test_every_batch_reaches_on_batch(self) -> None:
        seen: List[str] = []
        counts = run_batches(
            batches_of(5), echo, AlwaysAllow(),  # type: ignore[arg-type]
            on_batch=lambda b, r: seen.append(b[0]["_id"]),
            on_failure=lambda b, e: None, max_in_flight=2)
        assert sorted(seen) == ["b0", "b1", "b2", "b3", "b4"]
        assert counts == {'submitted': 5, 'completed': 5, 'failed': 0,
                          'not_submitted': 0}

    def test_failed_batch_goes_to_on_failure(self) -> None:
        def flaky(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if batch[0]["_id"] == "b1":
                raise RuntimeError("preempted")
            return echo(batch)

        saved: List[str] = []
        failed: List[str] = []
        counts = run_batches(
            batches_of(3), flaky, AlwaysAllow(),  # type: ignore[arg-type]
            on_batch=lambda b, r: saved.append(b[0]["_id"]),
            on_failure=lambda b, e: failed.append(b[0]["_id"]),
            max_in_flight=1)
        assert sorted(saved) == ["b0", "b2"]
        assert failed == ["b1"]
        assert counts['failed'] == 1 and counts['completed'] == 2

    def test_guard_stops_new_submissions(self) -> None:
        calls: List[str] = []

        def record(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            calls.append(batch[0]["_id"])
            return echo(batch)

        counts = run_batches(
            batches_of(6), record, AllowFirst(2),  # type: ignore[arg-type]
            on_batch=lambda b, r: None, on_failure=lambda b, e: None,
            max_in_flight=1)
        assert len(calls) == 2
        assert counts == {'submitted': 2, 'completed': 2, 'failed': 0,
                          'not_submitted': 4}

    def test_never_more_than_max_in_flight(self) -> None:
        lock = threading.Lock()
        state = {'now': 0, 'peak': 0}

        def slow(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            with lock:
                state['now'] += 1
                state['peak'] = max(state['peak'], state['now'])
            time.sleep(0.02)
            with lock:
                state['now'] -= 1
            return echo(batch)

        run_batches(batches_of(8), slow, AlwaysAllow(),  # type: ignore
                    on_batch=lambda b, r: None,
                    on_failure=lambda b, e: None, max_in_flight=3)
        assert 1 < state['peak'] <= 3


# ---------------------------------------------------------------------------
# Provenance and the saved document
# ---------------------------------------------------------------------------


class TestProvenance:

    def test_fields(self) -> None:
        assert provenance('modal', 'vllm', '0.32.0', 'H100',
                          'mistralai/Mistral-7B-Instruct-v0.3', 'c170c708',
                          'bfloat16', 'greedy', 3072) == {
            'backend': 'modal', 'engine': 'vllm', 'engine_version': '0.32.0',
            'gpu': 'H100', 'model': 'mistralai/Mistral-7B-Instruct-v0.3',
            'model_revision': 'c170c708', 'dtype': 'bfloat16',
            'decoding': 'greedy', 'max_tokens': 3072,
        }


class TestStructuredDoc:

    def test_id_matches_the_local_path(self) -> None:
        """Same rule as translate_and_save_streaming: sha256 of the
        stripped treatment and description joined by ':'."""
        assert taxon_doc_id(' Russula x\n', ' Pileus convex. ') == (
            'taxon_' + hashlib.sha256(
                'Russula x:Pileus convex.'.encode()).hexdigest())

    def test_shape(self) -> None:
        item = {'_id': 'src', 'treatment': 'Russula x',
                'description': 'Pileus convex.',
                'ingest': {'doi': '10.1/x'}, 'line_number': 4,
                'paragraph_number': 2, 'page_number': '7',
                'empirical_page_number': '123', 'prompt': 'P'}
        prov = {'backend': 'modal'}
        doc = structured_doc(item, FEATURES, True, prov)
        assert doc == {
            '_id': taxon_doc_id('Russula x', 'Pileus convex.'),
            'treatment': 'Russula x', 'description': 'Pileus convex.',
            'ingest': {'doi': '10.1/x'}, 'line_number': 4,
            'paragraph_number': 2, 'page_number': '7',
            'empirical_page_number': '123',
            'json_annotated': FEATURES, 'json_valid': True,
            'provenance': prov,
        }

    def test_missing_optional_fields(self) -> None:
        doc = structured_doc({'_id': 's', 'treatment': 'T',
                              'description': 'D'}, {}, False, {})
        assert doc['ingest'] == {}
        assert doc['line_number'] is None
        assert 'prompt' not in doc


# ---------------------------------------------------------------------------
# Compile and kernel caches on a Volume
# ---------------------------------------------------------------------------

CACHE_NOT_YET = pytest.mark.xfail(
    strict=True, reason="cache_env not implemented yet")


def test_cache_volume_mount() -> None:
    assert CACHE_MOUNT == '/cache'


@CACHE_NOT_YET
class TestCacheEnv:

    def test_sets_the_three_caches(self) -> None:
        env = cache_env('/cache', '0.32.0')
        assert set(env) == {'VLLM_CACHE_ROOT', 'FLASHINFER_WORKSPACE_BASE',
                            'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR'}

    def test_all_under_a_version_directory_on_the_mount(self) -> None:
        for path in cache_env('/cache', '0.32.0').values():
            assert path.startswith('/cache/vllm-0.32.0/')

    def test_caches_do_not_share_a_directory(self) -> None:
        paths = list(cache_env('/cache', '0.32.0').values())
        assert len(set(paths)) == len(paths)
        for a in paths:
            for b in paths:
                assert a == b or not b.startswith(a + '/')

    def test_an_upgrade_gets_fresh_directories(self) -> None:
        old = set(cache_env('/cache', '0.32.0').values())
        new = set(cache_env('/cache', '0.33.0').values())
        assert old.isdisjoint(new)
