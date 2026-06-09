"""Per-family pipeline registry.

Each family lives in its own module (``v3_logistic.py``,
``v4_crf.py``, …) and exports ``PIPELINE: tuple[PipelineStep, ...]``.
``load(name)`` picks one by name; ``available()`` enumerates the
registry so error messages can name what the operator should pick.
"""
from __future__ import annotations

import importlib
from typing import Tuple

from bin.pipelines.base import PipelineStep


# The canonical registry.  Adding a family = adding a new module
# under ``bin/pipelines/`` AND an entry here.  The two-step pattern
# keeps the registry explicit (operators see the list in errors)
# without paying for filesystem scans at import time.
_REGISTRY: Tuple[str, ...] = (
    'v3_logistic',
    'v4_crf',
)


def available() -> Tuple[str, ...]:
    """Names of every pipeline the dispatcher knows about."""
    return _REGISTRY


def load(name: str) -> Tuple[PipelineStep, ...]:
    """Return the ``PIPELINE`` tuple from ``bin/pipelines/{name}.py``.

    Raises ``ValueError`` with the available list on any unknown
    name — error messages should let the operator copy-paste a fix.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown pipeline {name!r}.  Available: "
            f"{', '.join(_REGISTRY)}",
        )
    module = importlib.import_module(f'bin.pipelines.{name}')
    pipeline: Tuple[PipelineStep, ...] = module.PIPELINE
    return pipeline
