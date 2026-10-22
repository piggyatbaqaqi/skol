#!/usr/bin/env python3
"""Make sure the model weights are on Modal storage, using CPU only.

Runs ``treatments_to_structured.modal_models.prepare`` in a CPU-only
Modal container, which downloads the pinned Hugging Face revision into
the ``skol-models`` Volume if it is not already there.  ``--check``
reports without downloading.  No GPU is billed.

Usage::

    bin/prepare_modal_models --check
    bin/prepare_modal_models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatments_to_structured.modal_models import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
)


class Remote(Protocol):
    """The Modal side: CPU-only functions run against the Volume."""

    def check(self, model_id: str, revision: str) -> Dict[str, Any]: ...

    def prepare(self, model_id: str, revision: str) -> str: ...


def validate_revision(revision: str) -> str:
    """``revision`` if it is a full 40-hex commit.

    Raises:
        ValueError: For a branch or tag such as ``main``, which can move
            and would let preparation and inference load different
            weights.
    """
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    from env_config import common_parser

    parser = argparse.ArgumentParser(
        parents=[common_parser()],
        description='Download the pinned model weights into Modal '
                    'storage (CPU only) if they are not already there.',
    )
    parser.add_argument('--check', action='store_true',
                        help='Report whether the weights are present; '
                             'download nothing.')
    parser.add_argument('--model', default=MODEL_ID, metavar='MODEL_ID',
                        help=f'Hugging Face model id (default: {MODEL_ID})')
    parser.add_argument('--revision', default=MODEL_REVISION,
                        metavar='COMMIT',
                        help='Full 40-hex commit to pin (default: the '
                             'pinned revision)')
    return parser


def run(args: argparse.Namespace, remote: Remote) -> int:
    """Carry out ``args`` against ``remote``; return the exit status.

    ``--check``: 0 when prepared, 1 when not.  Otherwise 0 after
    preparing.  2 on an invalid revision or a remote failure.
    """
    raise NotImplementedError


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
