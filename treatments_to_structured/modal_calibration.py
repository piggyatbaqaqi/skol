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

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CALIBRATION_FILE = Path(__file__).resolve().parent / 'modal_calibration.json'
SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZES: Tuple[int, ...] = (64, 128, 256)
DEFAULT_RECORDS_PER_CONFIG = 256

Entry = Dict[str, Any]


def calibration_key(max_tokens: int) -> Dict[str, Any]:
    """Everything a measured speed depends on, apart from the GPU type."""
    from treatments_to_structured import modal_inference as mi
    from treatments_to_structured.modal_models import (
        MODEL_ID, MODEL_REVISION,
    )

    return {
        'engine': mi.ENGINE,
        'engine_version': mi.VLLM_VERSION,
        'image': mi.VLLM_BASE_IMAGE,
        'model': MODEL_ID,
        'model_revision': MODEL_REVISION,
        'dtype': mi.DTYPE,
        'decoding': mi.DECODING,
        'max_tokens': max_tokens,
        'max_model_len': mi.MAX_MODEL_LEN,
    }


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
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    order = sorted(range(len(items)), key=lambda i: (token_counts[i], i))
    bins = max(1, min(bins, len(order)))
    edges = [len(order) * b // bins for b in range(bins + 1)]
    groups = [order[edges[b]:edges[b + 1]] for b in range(bins)]
    quota = [n // bins] * bins
    extra = n - sum(quota)
    # Spread the remainder evenly across the length range.
    for k in range(extra):
        quota[int((k + 0.5) * bins / extra)] += 1
    rng = random.Random(seed)
    picked: List[int] = []
    for group, want in zip(groups, quota):
        picked += rng.sample(group, min(want, len(group)))
    if len(picked) < n:
        taken = set(picked)
        rest = [i for i in order if i not in taken]
        picked += rng.sample(rest, n - len(picked))
    picked.sort(key=lambda i: (token_counts[i], i))
    return [items[i] for i in picked]


def summarize(
    raws: Sequence[Dict[str, Any]],
    elapsed_s: float,
    batch_size: int,
) -> Dict[str, Any]:
    """One config's measurements from its raw results and wall time.

    Raises:
        ValueError: With no results or a non-positive ``elapsed_s``.
    """
    if not raws:
        raise ValueError('no results to summarize')
    if elapsed_s <= 0:
        raise ValueError(f'elapsed_s must be positive, got {elapsed_s}')
    n = len(raws)
    tokens_out = sum(r['tokens_out'] for r in raws)
    return {
        'batch_size': batch_size,
        'records': n,
        'elapsed_s': elapsed_s,
        'records_per_s': n / elapsed_s,
        'output_tokens_per_s': tokens_out / elapsed_s,
        'cut_off_rate': sum(r['finish_reason'] == 'length'
                            for r in raws) / n,
        'mean_output_tokens': tokens_out / n,
        'mean_input_tokens': sum(r['tokens_in'] for r in raws) / n,
    }


def load_calibrations(path: Path = CALIBRATION_FILE) -> List[Entry]:
    """All entries; ``[]`` when the file does not exist.

    Raises:
        ValueError: For a file of another schema version.
    """
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return []
    if data.get('_schema_version') != SCHEMA_VERSION:
        raise ValueError(
            f'{path}: schema version {data.get("_schema_version")!r}, '
            f'expected {SCHEMA_VERSION}')
    return list(data.get('entries', []))


def save_calibration(entry: Entry, path: Path = CALIBRATION_FILE) -> None:
    """Store ``entry``, replacing any earlier entry for its GPU.

    Entries are kept sorted by GPU, and the file is replaced atomically.
    """
    path = Path(path)
    entries = [e for e in load_calibrations(path)
               if e['gpu'] != entry['gpu']] + [entry]
    entries.sort(key=lambda e: e['gpu'])
    text = json.dumps({'_schema_version': SCHEMA_VERSION,
                       'entries': entries}, indent=1, sort_keys=True) + '\n'
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def find_calibration(
    entries: Sequence[Entry], gpu: str, key: Dict[str, Any],
) -> Tuple[str, Optional[Entry]]:
    """``('ok', entry)``, ``('stale', entry)`` when measured under a
    different key, or ``('missing', None)``."""
    for entry in entries:
        if entry['gpu'] == gpu:
            return ('ok' if entry['key'] == key else 'stale'), entry
    return 'missing', None


def best_config(entry: Entry) -> Dict[str, Any]:
    """The config with the most records per second; ties go to the
    smaller batch.

    Raises:
        ValueError: When the entry has no configs.
    """
    configs = entry.get('configs') or []
    if not configs:
        raise ValueError(f'no configs measured for {entry.get("gpu")}')
    return max(configs,
               key=lambda c: (c['records_per_s'], -c['batch_size']))
