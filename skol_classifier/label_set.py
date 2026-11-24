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

"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

DECLARATION_KEYS: Tuple[str, ...] = (
    'label_set', 'label_set_digest', 'projection',
)
"""Metadata keys a model artifact carries to declare its label set."""

LABEL_SET_DIR = Path(__file__).resolve().parent / 'label_sets'

# The one version whose models predate declarations (E6-0).
_UNDECLARED_VERSION = 'v4'


class LabelSetMismatch(ValueError):
    """A model artifact was built against a different label set."""


def _jats_v4(sec_type: str) -> str:
    from ingestors.jats_to_yedda import sec_type_to_tag
    return sec_type_to_tag(sec_type).value


# Projections that are code rather than data, by the name a label set
# gives them.  A new mapping gets a new name; an existing name never
# changes behaviour.
_SEC_TYPE_PROJECTIONS: Dict[str, Callable[[str], str]] = {
    'jats_v4': _jats_v4,
}


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
        if self.fallback in self.members:
            return self.members
        return self.members + (self.fallback,)

    def project(self, tag: str) -> str:
        for member in self.members:
            if member.lower() == tag.lower():
                return member
        return self.fallback


def _digest(data: Mapping[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


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
        sec_type = str(data['jats']['sec_type'])
        if sec_type not in _SEC_TYPE_PROJECTIONS:
            raise KeyError(
                f'unknown sec-type projection {sec_type!r}; known: '
                f'{sorted(_SEC_TYPE_PROJECTIONS)}')
        return cls(
            version=str(data['version']),
            date=str(data['date']),
            tags=tuple(data['tags']),
            deprecated=dict(data['deprecated']),
            fields=dict(data['fields']),
            spans_fields=dict(data['spans_fields']),
            projections={
                name: Projection(name=name,
                                 members=tuple(p['members']),
                                 fallback=str(p['fallback']))
                for name, p in data['projections'].items()
            },
            jats_emit=tuple(data['jats']['emit']),
            jats_sec_type=sec_type,
            digest=_digest(data),
        )

    def _projection(self, projection: str) -> Projection:
        try:
            return self.projections[projection]
        except KeyError:
            raise KeyError(
                f'label set {self.version} has no projection '
                f'{projection!r}; it has {sorted(self.projections)}'
            ) from None

    def project(self, projection: str, tag: str) -> str:
        """Map ``tag`` into ``projection``'s label space."""
        return self._projection(projection).project(tag)

    def label_map(self, projection: str) -> Dict[str, int]:
        """Label → index for ``projection``, in index order."""
        return {label: i for i, label
                in enumerate(self._projection(projection).labels)}

    def declaration(self, projection: str) -> Dict[str, str]:
        """The metadata a model on ``projection`` records."""
        self._projection(projection)
        return {
            'label_set': self.version,
            'label_set_digest': self.digest,
            'projection': projection,
        }

    def sec_type_to_tag(self) -> Callable[[str], str]:
        """The TaxPub ``sec-type`` projection this set names."""
        return _SEC_TYPE_PROJECTIONS[self.jats_sec_type]


def available_label_sets() -> List[str]:
    """Versions shipped in ``skol_classifier/label_sets/``."""
    return sorted(p.stem for p in LABEL_SET_DIR.glob('*.json'))


def load_label_set(version: str) -> LabelSet:
    """Load one version.  Unknown version: ``KeyError`` naming those
    available."""
    path = LABEL_SET_DIR / f'{version}.json'
    if not path.is_file():
        raise KeyError(f'no label set {version!r}; available: '
                       f'{available_label_sets()}')
    label_set = LabelSet.from_dict(json.loads(path.read_text('utf-8')))
    if label_set.version != version:
        raise ValueError(f'{path} declares version {label_set.version!r}')
    return label_set


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
    expected = label_set.declaration(projection)
    if any(key in metadata for key in DECLARATION_KEYS):
        found = {key: metadata.get(key) for key in DECLARATION_KEYS}
        if found != expected:
            raise LabelSetMismatch(
                f'model declares {found}; this pipeline uses {expected}')
        return 'declared'
    if (label_set.version == _UNDECLARED_VERSION
            and metadata.get('label_map') == label_set.label_map(projection)):
        return 'inferred'
    raise LabelSetMismatch(
        f'model declares no label set and its label_map does not match '
        f'{label_set.version}/{projection}; refusing to load it')
