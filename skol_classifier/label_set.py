"""Versioned label sets (Experiment 6 plan §8.5, §8.7 step 0).

A label set is the tag list, the label→field maps, the projections into
each model's label space, and the JATS projections, held as a dated,
loadable object rather than module constants.  Each lives in
``skol_classifier/label_sets/<version>.json``.

Every model artifact declares the label set it was built against
(:meth:`LabelSet.declaration`), and loading refuses a mismatch
(:func:`check_declaration`) rather than projecting one vocabulary
through a model trained on another.  There is no global "current"
label set: the version is bound to the artifact.

SKELETON (E6-0): signatures only, so the xfail tests can import.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

DECLARATION_KEYS: Tuple[str, ...] = (
    'label_set', 'label_set_digest', 'projection',
)
"""Metadata keys a model artifact carries to declare its label set."""


class LabelSetMismatch(ValueError):
    """A model artifact was built against a different label set."""


@dataclasses.dataclass(frozen=True)
class Projection:
    """A projection of the label set into one model's label space.

    ``members`` pass through (matched case-insensitively, returned in
    canonical spelling); anything else maps to ``fallback``.  The
    output labels, in index order, are ``members`` followed by
    ``fallback`` when it is not itself a member.
    """

    name: str
    members: Tuple[str, ...]
    fallback: str

    @property
    def labels(self) -> Tuple[str, ...]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class LabelSet:
    """One version of the label set.  Built by :func:`load_label_set`."""

    version: str
    date: str
    tags: Tuple[str, ...]
    deprecated: Mapping[str, Optional[str]]
    fields: Mapping[str, str]
    spans_fields: Mapping[str, str]
    projections: Mapping[str, Projection]
    jats_emit: Tuple[str, ...]
    jats_sec_type: str
    digest: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'LabelSet':
        """Build from the parsed JSON; ``digest`` is computed here."""
        raise NotImplementedError

    def project(self, projection: str, tag: str) -> str:
        """Map ``tag`` into ``projection``'s label space."""
        raise NotImplementedError

    def label_map(self, projection: str) -> Dict[str, int]:
        """Label → index for ``projection``, in index order."""
        raise NotImplementedError

    def declaration(self, projection: str) -> Dict[str, str]:
        """The metadata a model on ``projection`` records."""
        raise NotImplementedError

    def sec_type_to_tag(self) -> Callable[[str], str]:
        """The TaxPub ``sec-type`` projection this set names."""
        raise NotImplementedError


def available_label_sets() -> List[str]:
    """Versions shipped in ``skol_classifier/label_sets/``."""
    raise NotImplementedError


def load_label_set(version: str) -> LabelSet:
    """Load one version.  Unknown version: ``KeyError`` naming those
    available."""
    raise NotImplementedError


def check_declaration(
    metadata: Mapping[str, Any],
    label_set: LabelSet,
    projection: str,
) -> str:
    """Refuse a model whose metadata declares another label set.

    Returns ``'declared'`` when the metadata names this version, digest
    and projection.  Metadata with no declaration (models saved before
    E6-0) is accepted as ``'inferred'`` only when ``label_set`` is v4
    and the stored ``label_map`` equals v4's for ``projection``.
    Anything else raises :class:`LabelSetMismatch`.
    """
    raise NotImplementedError
