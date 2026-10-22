"""Model weights on Modal storage for the treatments_to_json Modal backend.

Pure Python, with no ``modal`` import, so it is testable anywhere and
importable inside the Modal containers.  See
docs/modal_backend_design.md (tranche 1).

Layout on the Volume (mounted at ``VOLUME_MOUNT``)::

    <root>/models/<org>--<name>/<revision>/      Hugging Face files
    <root>/models/<org>--<name>/<revision>/.skol_prepared.json

The marker is written only after a complete download, so a partial
download never counts as prepared.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.3'
# Commit pinned 2026-10-22; preparation and inference must use the same.
MODEL_REVISION = 'c170c708c41dac9275d15a8fff4eca08d52bab71'

VOLUME_NAME = 'skol-models'
VOLUME_MOUNT = '/models'
# Optional: the repo is not gated today, but a token avoids rate limits.
HF_SECRET_NAME = 'huggingface'

# Hugging Face format only.  The repo also ships consolidated.safetensors
# and params.json (Mistral's own format, a second 14.5 GB copy of the
# weights), which vLLM does not need here.
ALLOW_PATTERNS: List[str] = [
    'config.json',
    'generation_config.json',
    'model-*.safetensors',
    'model.safetensors.index.json',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer.model',
    'tokenizer_config.json',
]

# Preparation runs on CPU only: no GPU is billed while downloading.
PREPARE_RESOURCES: Dict[str, Any] = {
    'gpu': None,
    'cpu': 2.0,
    'memory_mib': 4096,
    'timeout_s': 3600,
}

MARKER_NAME = '.skol_prepared.json'

# download(model_id, revision, dest_dir, allow_patterns) fetches the files;
# huggingface_hub.snapshot_download in production.
Downloader = Callable[[str, str, Path, List[str]], None]


def model_dir(root: Path, model_id: str, revision: str) -> Path:
    """Directory holding ``model_id`` at ``revision`` under ``root``."""
    raise NotImplementedError


def read_marker(directory: Path) -> Optional[Dict[str, Any]]:
    """The preparation marker in ``directory``, or None."""
    raise NotImplementedError


def is_prepared(root: Path, model_id: str, revision: str) -> bool:
    """True when the marker exists, names this model and revision, and
    every file it lists is present."""
    raise NotImplementedError


def prepare(
    root: Path,
    model_id: str,
    revision: str,
    download: Downloader,
    now: str,
) -> str:
    """Make sure the weights are on the Volume.

    Returns ``'present'`` when already prepared (nothing downloaded) or
    ``'downloaded'``.  The marker is written only after ``download``
    returns; if it raises, no marker is written and the error
    propagates.
    """
    raise NotImplementedError


def check(root: Path, model_id: str, revision: str) -> Dict[str, Any]:
    """Report ``{model_id, revision, path, prepared}`` without downloading."""
    raise NotImplementedError
