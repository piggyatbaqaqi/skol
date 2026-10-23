"""Calibration of the treatments_to_json Modal backend.

Pure Python: no ``modal`` or ``vllm`` import.  Calibration measures how
fast a GPU type translates real records at several batch sizes, and
records the result in ``CALIBRATION_FILE`` (in the repo) for ``--dry-run``
to read.  See docs/modal_backend_design.md (tranche 3).

One entry per GPU type::

    {"gpu": "A100-40GB", "key": {...what the speed depends on...},
     "measured_at": "<ISO time>", "startup_s": 58.6,
     "configs": [{"batch_size": 64, "records": 256, "elapsed_s": ...,
                  "records_per_s": ..., "output_tokens_per_s": ...,
                  "cut_off_rate": ..., "mean_output_tokens": ...,
                  "mean_input_tokens": ...}, ...]}

An entry whose ``key`` differs from today's (a new vLLM, model revision,
decoding setting, ...) is stale.  Prices are not part of the key: they
live in modal_inference.GPU_PRICES.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CALIBRATION_FILE = Path(__file__).resolve().parent / 'modal_calibration.json'
SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZES: Tuple[int, ...] = (64, 128, 256)
DEFAULT_RECORDS_PER_CONFIG = 256

Entry = Dict[str, Any]


def calibration_key(max_tokens: int) -> Dict[str, Any]:
    """Everything a measured speed depends on, apart from the GPU type."""
    raise NotImplementedError


def sample_stratified(
    items: Sequence[Dict[str, Any]],
    n: int,
    token_counts: Sequence[int],
    seed: int,
    bins: int = 10,
) -> List[Dict[str, Any]]:
    """``n`` items spread evenly across the description-length range.

    Items are sorted by ``token_counts`` and cut into ``bins`` equal
    bins; each bin contributes an equal share, drawn with ``seed``.
    All items when ``n`` is at least ``len(items)``.
    """
    raise NotImplementedError


def summarize(
    raws: Sequence[Dict[str, Any]],
    elapsed_s: float,
    batch_size: int,
) -> Dict[str, Any]:
    """One config's measurements from its raw results and wall time.

    Raises:
        ValueError: With no results or a non-positive ``elapsed_s``.
    """
    raise NotImplementedError


def load_calibrations(path: Path = CALIBRATION_FILE) -> List[Entry]:
    """All entries; ``[]`` when the file does not exist.

    Raises:
        ValueError: For a file of another schema version.
    """
    raise NotImplementedError


def save_calibration(entry: Entry, path: Path = CALIBRATION_FILE) -> None:
    """Store ``entry``, replacing any earlier entry for its GPU.

    Entries are kept sorted by GPU, and the file is replaced atomically.
    """
    raise NotImplementedError


def find_calibration(
    entries: Sequence[Entry], gpu: str, key: Dict[str, Any],
) -> Tuple[str, Optional[Entry]]:
    """``('ok', entry)``, ``('stale', entry)`` when measured under a
    different key, or ``('missing', None)``."""
    raise NotImplementedError


def best_config(entry: Entry) -> Dict[str, Any]:
    """The config with the most records per second; ties go to the
    smaller batch.

    Raises:
        ValueError: When the entry has no configs.
    """
    raise NotImplementedError
