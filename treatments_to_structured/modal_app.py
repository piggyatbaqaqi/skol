"""Modal definitions for the treatments_to_json Modal backend.

Thin wrappers: each Modal function calls the tested logic in
``treatments_to_structured.modal_models``.  Run ephemerally with
``app.run()`` from local commands (bin/prepare_modal_models); nothing
is deployed.  See docs/modal_backend_design.md.

Tranche 1 defines the CPU-only model preparation; the vLLM GPU image
and inference class arrive in tranche 2.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import modal

from treatments_to_structured.modal_models import (
    PREPARE_RESOURCES,
    VOLUME_MOUNT,
    VOLUME_NAME,
    check,
    prepare,
)

APP_NAME = 'skol-treatments-json'
HUGGINGFACE_HUB_VERSION = '0.36.2'

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

prepare_image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install(f'huggingface_hub=={HUGGINGFACE_HUB_VERSION}')
    .add_local_python_source('treatments_to_structured')
)

_cpu_function = app.function(
    image=prepare_image,
    volumes={VOLUME_MOUNT: volume},
    cpu=PREPARE_RESOURCES['cpu'],
    memory=PREPARE_RESOURCES['memory_mib'],
    timeout=PREPARE_RESOURCES['timeout_s'],
)


@_cpu_function
def check_models(model_id: str, revision: str) -> Dict[str, Any]:
    """Report whether the weights are on the Volume; download nothing."""
    return check(Path(VOLUME_MOUNT), model_id, revision)


@_cpu_function
def prepare_models(model_id: str, revision: str) -> str:
    """Download the weights into the Volume unless already prepared."""
    from huggingface_hub import snapshot_download

    def download(model: str, rev: str, dest: Path,
                 allow_patterns: List[str]) -> None:
        snapshot_download(repo_id=model, revision=rev, local_dir=dest,
                          allow_patterns=allow_patterns)

    status = prepare(Path(VOLUME_MOUNT), model_id, revision, download,
                     datetime.now(timezone.utc).isoformat())
    volume.commit()
    return status
