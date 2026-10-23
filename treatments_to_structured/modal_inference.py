"""Local-side logic for the treatments_to_json Modal backend.

Pure Python: no ``modal``, ``vllm`` or ``torch`` import, so it is
testable anywhere and also importable inside the Modal containers.  See
docs/modal_backend_design.md (tranche 2).
"""
from __future__ import annotations

import time
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

Raw = Dict[str, Any]          # one record's result from the GPU class
Item = Dict[str, Any]         # one record to translate, with 'prompt'


def rate_per_second(gpu: str) -> float:
    """$/second for one container of ``gpu``, including CPU and memory.

    Raises:
        ValueError: For a GPU not in ``GPU_PRICES``.
    """
    raise NotImplementedError


class SpendGuard:
    """Upper bound on spend: elapsed time x GPUs x rate.

    Needs no calibration.  It over-counts start-up and idle time, which
    is the safe direction.
    """

    def __init__(self, max_cost: float, gpu: str, gpus: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        raise NotImplementedError

    def start(self) -> None:
        """Begin timing.  Before ``start`` nothing has been spent."""
        raise NotImplementedError

    def spent_upper_bound(self) -> float:
        raise NotImplementedError

    def allows_more(self) -> bool:
        """True while the bound is below ``max_cost``."""
        raise NotImplementedError


def make_batches(items: Sequence[Item], size: int) -> List[List[Item]]:
    """Consecutive batches of at most ``size`` items."""
    raise NotImplementedError


def raw_from_vllm(doc_id: str, output: Any) -> Raw:
    """One vLLM ``RequestOutput`` as ``{_id, text, finish_reason,
    tokens_in, tokens_out}``."""
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def taxon_doc_id(treatment: str, description: str) -> str:
    """Deterministic structured-treatment id, as the local path makes it."""
    raise NotImplementedError


def structured_doc(
    item: Item,
    features: Dict[str, Any],
    json_valid: bool,
    prov: Dict[str, Any],
) -> Dict[str, Any]:
    """The CouchDB document for one translated treatment."""
    raise NotImplementedError
