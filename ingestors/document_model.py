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

Every type validates itself on construction, so a block that can't be
what it claims is refused at once, whether it was built by a reader or
loaded back from storage.
"""
from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = 'document-model/1'
"""Tag written by :meth:`Document.to_dict`; :meth:`Document.from_dict`
refuses any other."""


def _check_probability(name: str, value: Optional[float]) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be in [0, 1], got {value!r}')


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

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f'page is numbered from 1, got {self.page}')
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError(f'inverted box {self!r}')


@dataclasses.dataclass(frozen=True)
class SourceSpan:
    """Where a block came from.  At least one of: page boxes (OCR; a
    block may cross a page), a 1-based inclusive line range (plain
    text), or an element id or path (JATS)."""

    boxes: Tuple[Box, ...] = ()
    lines: Optional[Tuple[int, int]] = None
    element: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.boxes and self.lines is None and not self.element:
            raise ValueError('a source span needs boxes, lines or an element')
        if self.lines is not None:
            first, last = self.lines
            if first < 1 or first > last:
                raise ValueError(
                    f'line range must be 1-based and ordered, '
                    f'got {self.lines!r}')


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

    def __post_init__(self) -> None:
        if self.kind is BlockKind.HEADING:
            if self.level is None or self.level < 1:
                raise ValueError(
                    f'a heading needs a level of 1 or more, got {self.level!r}')
        elif self.level is not None:
            raise ValueError(f'only a heading has a level, not a {self.kind.value}')
        _check_probability('block confidence', self.confidence)


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

    def __post_init__(self) -> None:
        _check_probability('provenance confidence', self.confidence)
        if self.min_ppi is not None and self.min_ppi <= 0:
            raise ValueError(f'min_ppi must be positive, got {self.min_ppi}')


@dataclasses.dataclass(frozen=True)
class Document:
    """A source document as blocks in reading order."""

    doc_id: str
    blocks: Tuple[Block, ...]
    provenance: Provenance

    def assertions_for(self, layer: str) -> List[Tuple[int, Assertion]]:
        """(block index, assertion) for every assertion on ``layer``,
        in reading order."""
        return [(i, a) for i, block in enumerate(self.blocks)
                for a in block.assertions if a.layer == layer]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible form, tagged with :data:`SCHEMA`."""
        prov = dataclasses.asdict(self.provenance)
        prov['source_kind'] = self.provenance.source_kind.value
        prov['languages'] = list(self.provenance.languages)
        return {
            'schema': SCHEMA,
            'doc_id': self.doc_id,
            'provenance': prov,
            'blocks': [_block_to_dict(b) for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'Document':
        """Inverse of :meth:`to_dict`.  ``ValueError`` on another
        schema tag or an unknown block or source kind."""
        schema = data.get('schema')
        if schema != SCHEMA:
            raise ValueError(
                f'expected schema {SCHEMA!r}, got {schema!r}')
        prov = dict(data['provenance'])
        prov['source_kind'] = SourceKind(prov['source_kind'])
        prov['languages'] = tuple(prov.get('languages', ()))
        return cls(
            doc_id=data['doc_id'],
            blocks=tuple(_block_from_dict(b) for b in data['blocks']),
            provenance=Provenance(**prov),
        )

    def to_markdown(self) -> str:
        """Print for display and prompts.  Lossy by design: source
        spans, assertions, confidence and provenance are not printed."""
        return '\n\n'.join(_block_to_markdown(b) for b in self.blocks)


def assign_section_paths(blocks: Sequence[Block]) -> Tuple[Block, ...]:
    """Fill each block's ``section_path`` from the headings before it.

    A block's path is the text of its enclosing headings, outermost
    first.  A heading's own path excludes itself.  A heading at level
    n closes every open heading at level n or deeper; skipped levels
    are allowed.
    """
    open_headings: List[Tuple[int, str]] = []
    result: List[Block] = []
    for block in blocks:
        if block.kind is BlockKind.HEADING:
            assert block.level is not None   # Block.__post_init__ checked
            while open_headings and open_headings[-1][0] >= block.level:
                open_headings.pop()
        path = tuple(text for _level, text in open_headings)
        result.append(dataclasses.replace(block, section_path=path))
        if block.kind is BlockKind.HEADING:
            assert block.level is not None
            open_headings.append((block.level, block.text))
    return tuple(result)


def _block_to_dict(block: Block) -> Dict[str, Any]:
    src = block.source
    return {
        'kind': block.kind.value,
        'text': block.text,
        'source': {
            'boxes': [dataclasses.asdict(b) for b in src.boxes],
            'lines': list(src.lines) if src.lines is not None else None,
            'element': src.element,
        },
        'section_path': list(block.section_path),
        'level': block.level,
        'assertions': [dataclasses.asdict(a) for a in block.assertions],
        'confidence': block.confidence,
        'language': block.language,
    }


def _block_from_dict(data: Mapping[str, Any]) -> Block:
    src = data['source']
    lines = src.get('lines')
    return Block(
        kind=BlockKind(data['kind']),
        text=data['text'],
        source=SourceSpan(
            boxes=tuple(Box(**b) for b in src.get('boxes', ())),
            lines=(int(lines[0]), int(lines[1])) if lines is not None
            else None,
            element=src.get('element'),
        ),
        section_path=tuple(data.get('section_path', ())),
        level=data.get('level'),
        assertions=tuple(Assertion(**a) for a in data.get('assertions', ())),
        confidence=data.get('confidence'),
        language=data.get('language'),
    )


# A line that Markdown would read as a heading, list item or blockquote.
_BLOCK_MARKER = re.compile(r'^(\s*)(#|[-*+](?=\s)|>|\d+(?=[.)]\s))')


def _escape_line(line: str) -> str:
    match = _BLOCK_MARKER.match(line)
    if match is None:
        return line
    indent, marker = match.group(1), match.group(2)
    rest = line[match.end():]
    if marker[0].isdigit():         # '2. x' -> '2\. x'
        return f'{indent}{marker}\\{rest}'
    return f'{indent}\\{marker}{rest}'


def _escape(text: str) -> str:
    return '\n'.join(_escape_line(line) for line in text.split('\n'))


def _block_to_markdown(block: Block) -> str:
    kind = block.kind
    if kind is BlockKind.HEADING:
        assert block.level is not None
        return f"{'#' * block.level} {block.text}"
    if kind is BlockKind.LIST_ITEM:
        return f'- {block.text}'
    if kind is BlockKind.FIGURE:
        return f'![{block.text}]()'
    if kind is BlockKind.CAPTION:
        return f'*{block.text}*'
    if kind in (BlockKind.TABLE, BlockKind.KEY_COUPLET):
        return block.text
    return _escape(block.text)
