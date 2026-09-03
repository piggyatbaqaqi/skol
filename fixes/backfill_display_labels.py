#!/usr/bin/env python3
"""Backfill display labels onto annotations that predate the capture.

Skeleton: see the xfailed tests.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def choose_display_labels(
        annotations: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError
