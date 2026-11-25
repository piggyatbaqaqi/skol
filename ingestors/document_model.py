"""The document model every Experiment 6 layer reads (E6-1a-1).

Experiment 6 plan §8.8 item 1, and §8.1 for provenance.  A document is
a sequence of blocks in reading order.  Each block has a kind, its
text, its section path (heading ancestry), where it came from in the
source, and zero or more **assertions**: a layer's output that the
source itself states, with the origin that stated it.

Readers (structural OCR, E6-1a-3; JATS, E6-1b) produce this model.
Markdown is one way to print it (:meth:`Document.to_markdown`), for
display and prompts; it is not the interchange format.  The
interchange format is :meth:`Document.to_dict`, plain JSON-compatible
data tagged with :data:`SCHEMA`.

SKELETON: the types exist so tests can import them; behaviour lands
after the tests are approved.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = 'document-model/1'
"""Tag written by :meth:`Document.to_dict`; :meth:`Document.from_dict`
refuses any other."""


class BlockKind(str, enum.Enum):
    """What a block is, typographically.  Not what it means: meaning is
    an assertion or a layer's output."""

    HEADING = 'heading'
    PARAGRAPH = 'paragraph'
    LIST_ITEM = 'list_item'
    TABLE = 'table'
    FIGURE = 'figure'
    CAPTION = 'caption'
    KEY_COUPLET = 'key_couplet'


class SourceKind(str, enum.Enum):
    """How the source document reached us."""

    BORN_DIGITAL = 'born_digital'   # PDF with a real text layer
    SCAN = 'scan'                   # page images, OCRed
    JATS = 'jats'                   # publisher XML


@dataclasses.dataclass(frozen=True)
class Box:
    """A rectangle on one page, in the reader's page coordinates.
    Pages are numbered from 1."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclasses.dataclass(frozen=True)
class SourceSpan:
    """Where a block came from.  At least one of: page boxes (OCR; a
    block may cross a page), a 1-based inclusive line range (plain
    text), or an element id or path (JATS)."""

    boxes: Tuple[Box, ...] = ()
    lines: Optional[Tuple[int, int]] = None
    element: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Assertion:
    """A layer's output that the source states.

    ``layer`` names the layer it answers for (``layout``, ``article``,
    ``treatment``, ``block-label``, …), ``value`` the answer, and
    ``origin`` who stated it (``taxpub``, ``jats-sec-type``,
    ``ocr-layout``, ``hand-v2``, …).  A label value names the label set
    it belongs to, as models do (plan §8.5).
    """

    layer: str
    value: str
    origin: str
    label_set: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Block:
    """One block in reading order.  ``level`` is required for a
    heading (1 = top) and forbidden otherwise.  ``confidence`` is the
    engine's, in [0, 1], when it gives one."""

    kind: BlockKind
    text: str
    source: SourceSpan
    section_path: Tuple[str, ...] = ()
    level: Optional[int] = None
    assertions: Tuple[Assertion, ...] = ()
    confidence: Optional[float] = None
    language: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Provenance:
    """How a document's text was produced (plan §8.1, §8.12).

    ``engine`` and ``reader`` with versions; ``languages`` detected;
    mean engine ``confidence`` in [0, 1] when given; ``min_ppi`` of the
    page images for a scan; ``text_layer_discarded`` when an existing
    text layer was thrown away in favour of fresh OCR.
    """

    source_kind: SourceKind
    engine: str
    engine_version: str
    reader: str
    reader_version: str
    recorded_at: str
    languages: Tuple[str, ...] = ()
    confidence: Optional[float] = None
    min_ppi: Optional[int] = None
    text_layer_discarded: bool = False


@dataclasses.dataclass(frozen=True)
class Document:
    """A source document as blocks in reading order."""

    doc_id: str
    blocks: Tuple[Block, ...]
    provenance: Provenance

    def assertions_for(self, layer: str) -> List[Tuple[int, Assertion]]:
        """(block index, assertion) for every assertion on ``layer``,
        in reading order."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible form, tagged with :data:`SCHEMA`."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'Document':
        """Inverse of :meth:`to_dict`.  ``ValueError`` on another
        schema tag or an unknown block or source kind."""
        raise NotImplementedError

    def to_markdown(self) -> str:
        """Print for display and prompts.  Lossy by design."""
        raise NotImplementedError


def assign_section_paths(blocks: Sequence[Block]) -> Tuple[Block, ...]:
    """Fill each block's ``section_path`` from the headings before it.

    A block's path is the text of its enclosing headings, outermost
    first.  A heading's own path excludes itself.  A heading at level
    n closes every open heading at level n or deeper; skipped levels
    are allowed.
    """
    raise NotImplementedError
