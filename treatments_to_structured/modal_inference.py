"""Local-side logic for the treatments_to_json Modal backend.

Pure Python: no ``modal``, ``vllm`` or ``torch`` import, so it is
testable anywhere and also importable inside the Modal containers.  See
docs/modal_backend_design.md (tranche 2).
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional, Sequence

# Modal list prices, $/second, read from https://modal.com/pricing on
# PRICES_READ_ON: preemptible, no region selection.  Keys are Modal's
# GPU names.  A price change means editing this table, not recalibrating.
PRICES_READ_ON = '2026-10-22'
GPU_PRICES: Dict[str, float] = {
    'H100': 0.001097,
    'A100-80GB': 0.000694,
    'A100-40GB': 0.000583,
    'L40S': 0.000542,
    'A10G': 0.000306,
    'L4': 0.000222,
}
# Per GPU container: 2 physical cores and 16 GiB of memory.
CONTAINER_CORES = 2
CONTAINER_MEMORY_GIB = 16
CPU_PRICE_PER_CORE_SECOND = 0.0000131
MEMORY_PRICE_PER_GIB_SECOND = 0.00000222

ENGINE = 'vllm'
VLLM_VERSION = '0.32.0'
DTYPE = 'bfloat16'
DECODING = 'greedy'
# Longest run of whitespace the JSON grammar allows.  Without a cap,
# xgrammar's JSON grammar permits unlimited whitespace and Mistral can run
# into it until max_tokens: 25 of 200 outputs in the 2026-10-23
# comparison, every cut-off.  Four-space indentation at the schema's
# maximum depth needs at most 17.
MAX_WHITESPACE_CNT = 20

# Compile and kernel caches live on their own Volume, so containers after
# the first skip torch.compile and FlashInfer's JIT build (~2.5 of the
# ~4 minutes the first smoke run spent starting up).
CACHE_VOLUME_NAME = 'skol-vllm-cache'
CACHE_MOUNT = '/cache'

Raw = Dict[str, Any]          # one record's result from the GPU class
Item = Dict[str, Any]         # one record to translate, with 'prompt'


def rate_per_second(gpu: str) -> float:
    """$/second for one container of ``gpu``, including CPU and memory.

    Raises:
        ValueError: For a GPU not in ``GPU_PRICES``.
    """
    if gpu not in GPU_PRICES:
        raise ValueError(
            f'unknown GPU {gpu!r}; choose from {", ".join(GPU_PRICES)}')
    return (GPU_PRICES[gpu]
            + CONTAINER_CORES * CPU_PRICE_PER_CORE_SECOND
            + CONTAINER_MEMORY_GIB * MEMORY_PRICE_PER_GIB_SECOND)


class SpendGuard:
    """Upper bound on spend: elapsed time x GPUs x rate.

    Needs no calibration.  It over-counts start-up and idle time, which
    is the safe direction.
    """

    def __init__(self, max_cost: float, gpu: str, gpus: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_cost <= 0:
            raise ValueError(f'max_cost must be positive, got {max_cost}')
        if gpus < 1:
            raise ValueError(f'gpus must be at least 1, got {gpus}')
        self.max_cost = max_cost
        self.gpus = gpus
        self.rate = rate_per_second(gpu)
        self._clock = clock
        self._started: Optional[float] = None

    def start(self) -> None:
        """Begin timing.  Before ``start`` nothing has been spent."""
        self._started = self._clock()

    def spent_upper_bound(self) -> float:
        """Dollars spent at most so far."""
        if self._started is None:
            return 0.0
        return (self._clock() - self._started) * self.gpus * self.rate

    def allows_more(self) -> bool:
        """True while the bound is below ``max_cost``."""
        return self.spent_upper_bound() < self.max_cost


def cache_env(mount: str, vllm_version: str) -> Dict[str, str]:
    """Environment variables pointing vLLM's and FlashInfer's caches at
    ``mount``, under a directory named for ``vllm_version`` so an
    upgrade starts a fresh cache rather than mixing versions.

    vLLM routes the Inductor and Triton caches under ``VLLM_CACHE_ROOT``
    itself.
    """
    root = f'{mount.rstrip("/")}/vllm-{vllm_version}'
    return {
        'VLLM_CACHE_ROOT': f'{root}/vllm',
        'FLASHINFER_WORKSPACE_BASE': f'{root}/flashinfer',
        'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR': f'{root}/flashinfer-autotune',
    }


def json_grammar(schema: Dict[str, Any], xgr: Any) -> str:
    """Grammar text (EBNF) for ``schema`` with whitespace runs capped at
    ``MAX_WHITESPACE_CNT``, for ``StructuredOutputsParams(grammar=...)``.

    ``xgr`` is the ``xgrammar`` module, passed in because it exists only
    in the GPU image.
    """
    raise NotImplementedError


def sampling_options(max_tokens: int) -> Dict[str, Any]:
    """Keyword arguments for vLLM's ``SamplingParams``: greedy decoding
    with an explicit ``max_tokens`` (vLLM's default is 16).

    Raises:
        ValueError: If ``max_tokens`` is less than 1.
    """
    raise NotImplementedError


def make_batches(items: Sequence[Item], size: int) -> List[List[Item]]:
    """Consecutive batches of at most ``size`` items."""
    if size < 1:
        raise ValueError(f'batch size must be at least 1, got {size}')
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def raw_from_vllm(doc_id: str, output: Any) -> Raw:
    """One vLLM ``RequestOutput`` as ``{_id, text, finish_reason,
    tokens_in, tokens_out}``."""
    completion = output.outputs[0]
    return {
        '_id': doc_id,
        'text': completion.text,
        'finish_reason': completion.finish_reason,
        'tokens_in': len(output.prompt_token_ids or []),
        'tokens_out': len(completion.token_ids or []),
    }


def interpret_output(
    raw: Raw,
    validate: Callable[[Dict[str, Any]], Any],
) -> Dict[str, Any]:
    """Turn one raw result into what gets saved.

    Returns ``{features, json_valid, outcome}`` with ``outcome`` one of
    ``ok``, ``cut_off`` (stopped at max_tokens) or ``invalid``
    (unparseable, not an object, or rejected by ``validate``).  Only
    ``ok`` keeps the features; the others save ``{}``, as on the local
    path.
    """
    def rejected(outcome: str) -> Dict[str, Any]:
        return {'features': {}, 'json_valid': False, 'outcome': outcome}

    if raw.get('finish_reason') == 'length':
        return rejected('cut_off')
    try:
        features = json.loads(raw.get('text') or '')
    except ValueError:
        return rejected('invalid')
    if not isinstance(features, dict) or not features:
        return rejected('invalid')
    try:
        if validate(features) is False:
            return rejected('invalid')
    except ValueError:
        return rejected('invalid')
    return {'features': features, 'json_valid': True, 'outcome': 'ok'}


def run_batches(
    batches: Sequence[List[Item]],
    call: Callable[[List[Item]], List[Raw]],
    guard: SpendGuard,
    on_batch: Callable[[List[Item], List[Raw]], None],
    on_failure: Callable[[List[Item], BaseException], None],
    max_in_flight: int,
) -> Dict[str, int]:
    """Send batches through ``call`` with at most ``max_in_flight`` at
    once, stopping new submissions once ``guard`` disallows more.

    Batches already submitted finish.  A failed batch goes to
    ``on_failure`` and nothing of it is saved.

    Returns ``{submitted, completed, failed, not_submitted}``.
    """
    counts = {'submitted': 0, 'completed': 0, 'failed': 0,
              'not_submitted': 0}
    pending = list(batches)
    in_flight: Dict[Future, List[Item]] = {}
    stopped = False
    with ThreadPoolExecutor(max_workers=max(1, max_in_flight)) as pool:
        while pending or in_flight:
            while (pending and not stopped
                   and len(in_flight) < max_in_flight):
                if not guard.allows_more():
                    stopped = True
                    break
                batch = pending.pop(0)
                in_flight[pool.submit(call, batch)] = batch
                counts['submitted'] += 1
            if stopped:
                counts['not_submitted'] += len(pending)
                pending = []
            if not in_flight:
                break
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                batch = in_flight.pop(future)
                try:
                    results = future.result()
                except Exception as exc:
                    counts['failed'] += 1
                    on_failure(batch, exc)
                    continue
                counts['completed'] += 1
                on_batch(batch, results)
    return counts


def provenance(
    backend: str,
    engine: str,
    engine_version: Optional[str],
    gpu: Optional[str],
    model: str,
    model_revision: Optional[str],
    dtype: str,
    decoding: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """The ``provenance`` field stored on every structured treatment."""
    return {
        'backend': backend,
        'engine': engine,
        'engine_version': engine_version,
        'gpu': gpu,
        'model': model,
        'model_revision': model_revision,
        'dtype': dtype,
        'decoding': decoding,
        'max_tokens': max_tokens,
    }


def taxon_doc_id(treatment: str, description: str) -> str:
    """Deterministic structured-treatment id, as the local path makes it."""
    content = (treatment or '').strip() + ':' + (description or '').strip()
    return 'taxon_' + hashlib.sha256(content.encode('utf-8')).hexdigest()


def structured_doc(
    item: Item,
    features: Dict[str, Any],
    json_valid: bool,
    prov: Dict[str, Any],
) -> Dict[str, Any]:
    """The CouchDB document for one translated treatment."""
    return {
        '_id': taxon_doc_id(item.get('treatment', ''),
                            item.get('description', '')),
        'treatment': item.get('treatment', ''),
        'description': item.get('description', ''),
        'ingest': item.get('ingest') or {},
        'line_number': item.get('line_number'),
        'paragraph_number': item.get('paragraph_number'),
        'page_number': item.get('page_number'),
        'empirical_page_number': item.get('empirical_page_number'),
        'json_annotated': features,
        'json_valid': json_valid,
        'provenance': prov,
    }
