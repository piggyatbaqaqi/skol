#!/usr/bin/env python3
"""XML format identity for ingested articles — one registry.

Skeleton: the registry is real, the behaviour is not yet written.
See the xfailed tests in ``xml_formats_test.py``.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

JATS = 'jats'
TAXPUB = 'taxpub'


def _unimplemented(header: str) -> bool:
    raise NotImplementedError


@dataclass(frozen=True)
class XmlFormat:
    """One recognised XML format.

    ``jats_family`` is the membership answer callers used to restate
    for themselves; ``matches`` decides detection from the header.
    """

    name: str
    jats_family: bool
    matches: Callable[[str], bool]


# Ordered by specificity: detection returns the first match, so a
# profile of a format must precede the format it profiles.
FORMATS: Tuple[XmlFormat, ...] = (
    XmlFormat(name=TAXPUB, jats_family=True, matches=_unimplemented),
    XmlFormat(name=JATS, jats_family=True, matches=_unimplemented),
)


def detect(content: bytes) -> Optional[str]:
    """Name of the format ``content`` is in, or ``None``."""
    raise NotImplementedError


def is_jats_family(xml_format: Optional[str]) -> bool:
    """True when ``xml_format`` names a JATS-family format."""
    raise NotImplementedError


def is_plain_jats(xml_format: Optional[str]) -> bool:
    """True for JATS and nothing else — not TaxPub."""
    raise NotImplementedError


def is_taxpub(xml_format: Optional[str]) -> bool:
    """True for TaxPub exactly."""
    raise NotImplementedError
