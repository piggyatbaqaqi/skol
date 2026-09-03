#!/usr/bin/env python3
"""Build the canonical feature-label DB from the candidate DB.

Skeleton: see the xfailed tests in ``build_canonical_labels_test.py``.
"""

import argparse
import collections
import sys
from pathlib import Path
from typing import Any, Container, Dict, Iterable, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_config import common_parser  # noqa: E402


def canonicalize_all(
    annotations: Iterable[Mapping[str, Any]],
    *,
    known: Mapping[str, str],
    established: Mapping[str, str],
    protected: Container[str],
    source_db: str,
) -> Tuple[List[Dict[str, Any]], 'collections.Counter[str]']:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[common_parser()],
    )
    parser.parse_args()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
