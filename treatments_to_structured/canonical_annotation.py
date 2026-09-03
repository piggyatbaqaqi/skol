#!/usr/bin/env python3
"""Raw bootstrap annotations to canonical ones.

Skeleton: see the xfailed tests in ``canonical_annotation_test.py``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class CanonicalLabel:
    """One canonical label plus the dimensions taken out of its name.

    Skeleton: see the xfailed tests.
    """

    label: str
    sub_attribute: Optional[str] = None
    media: Tuple[str, ...] = ()
    condition: Optional[str] = None
    transforms: Tuple[str, ...] = field(default=())


def fold_case(label: str, known: Mapping[str, str]) -> str:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError


def split_condition(label: str) -> Tuple[str, Tuple[str, ...], Optional[str]]:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError


def strip_sub_attribute(
    label: str,
    established: Mapping[str, str],
) -> Tuple[str, Optional[str]]:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError


def split_compound(
    label: str,
    known: Mapping[str, str],
) -> Optional[List[str]]:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError


def presence_from_span(text: str) -> Optional[str]:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError


def canonicalize_label(
    label: str,
    *,
    known: Mapping[str, str],
    established: Mapping[str, str],
) -> List[CanonicalLabel]:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError


def canonical_records(
    annotation: Mapping[str, Any],
    *,
    known: Mapping[str, str],
    established: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """Skeleton: see the xfailed tests."""
    raise NotImplementedError
