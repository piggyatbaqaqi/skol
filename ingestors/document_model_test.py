"""Tests for ingestors/document_model.py (E6-1a-1, Experiment 6 plan §8.8
item 1 and §8.1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.document_model import (  # noqa: E402
    SCHEMA,
    Assertion,
    Block,
    BlockKind,
    Box,
    Document,
    Provenance,
    SourceKind,
    SourceSpan,
    assign_section_paths,
)

NOT_YET = pytest.mark.xfail(strict=True, reason='E6-1a-1 not implemented')

LINE = SourceSpan(lines=(1, 1))


def _block(kind: BlockKind = BlockKind.PARAGRAPH, text: str = 'Pileus 3–5 cm.',
           level: Optional[int] = None, **kw: Any) -> Block:
    return Block(kind=kind, text=text, source=kw.pop('source', LINE),
                 level=level, **kw)


def _heading(text: str, level: int) -> Block:
    return _block(BlockKind.HEADING, text, level=level)


def _provenance(**kw: Any) -> Provenance:
    fields: Dict[str, Any] = dict(
        source_kind=SourceKind.SCAN, engine='olmocr', engine_version='0.0',
        reader='olmocr-markdown', reader_version='1',
        recorded_at='2026-11-24T00:00:00Z')
    fields.update(kw)
    return Provenance(**fields)


def _document() -> Document:
    """One of everything, so round trips cover every field."""
    blocks = (
        _heading('Taxonomy', 1),
        _block(BlockKind.PARAGRAPH, 'Amanita nova sp. nov.',
               source=SourceSpan(boxes=(Box(3, 10.0, 20.0, 300.0, 40.0),)),
               assertions=(Assertion('treatment', 'start', 'taxpub'),
                           Assertion('block-label', 'Nomenclature',
                                     'jats-sec-type', label_set='v4')),
               confidence=0.91, language='la'),
        _heading('Key to species', 2),
        _block(BlockKind.KEY_COUPLET, '1. Spores smooth ... 2',
               source=SourceSpan(element='/article/body/sec[3]/p[1]')),
        _block(BlockKind.LIST_ITEM, 'Holotype: NY 123'),
        _block(BlockKind.TABLE, '| a | b |\n|---|---|\n| 1 | 2 |',
               source=SourceSpan(boxes=(Box(4, 0, 0, 1, 1),
                                        Box(5, 0, 0, 1, 1)))),
        _block(BlockKind.FIGURE, 'Basidiospores'),
        _block(BlockKind.CAPTION, 'Fig. 1. Basidiospores. Bar = 10 µm.'),
    )
    return Document(
        doc_id='doc-1', blocks=assign_section_paths(blocks),
        provenance=_provenance(languages=('en', 'la'), confidence=0.87,
                               min_ppi=300, text_layer_discarded=True))


# ---------------------------------------------------------------------
# Validation: a block that can't be what it claims is refused at once
# ---------------------------------------------------------------------

@NOT_YET
def test_heading_requires_a_level() -> None:
    with pytest.raises(ValueError):
        _block(BlockKind.HEADING, 'Taxonomy')


@NOT_YET
@pytest.mark.parametrize('level', [0, -1])
def test_heading_level_is_positive(level: int) -> None:
    with pytest.raises(ValueError):
        _heading('Taxonomy', level)


@NOT_YET
def test_level_is_forbidden_off_a_heading() -> None:
    with pytest.raises(ValueError):
        _block(BlockKind.PARAGRAPH, level=2)


@NOT_YET
@pytest.mark.parametrize('confidence', [-0.01, 1.01])
def test_block_confidence_is_a_probability(confidence: float) -> None:
    with pytest.raises(ValueError):
        _block(confidence=confidence)


@NOT_YET
def test_source_span_must_say_something() -> None:
    with pytest.raises(ValueError):
        SourceSpan()


@NOT_YET
@pytest.mark.parametrize('lines', [(0, 1), (3, 2)])
def test_line_range_is_one_based_and_ordered(lines: Any) -> None:
    with pytest.raises(ValueError):
        SourceSpan(lines=lines)


@NOT_YET
@pytest.mark.parametrize('box', [(0, 0, 0, 1, 1), (1, 2, 0, 1, 1),
                                 (1, 0, 2, 1, 1)])
def test_box_is_on_a_real_page_and_not_inverted(box: Any) -> None:
    with pytest.raises(ValueError):
        Box(*box)


@NOT_YET
@pytest.mark.parametrize('confidence', [-0.5, 1.5])
def test_provenance_confidence_is_a_probability(confidence: float) -> None:
    with pytest.raises(ValueError):
        _provenance(confidence=confidence)


@NOT_YET
def test_provenance_ppi_is_positive() -> None:
    with pytest.raises(ValueError):
        _provenance(min_ppi=0)


def test_kinds_are_the_plans_seven() -> None:
    # Plan §8.8 item 1: heading, paragraph, list item, table, figure,
    # caption, key couplet.
    assert {k.value for k in BlockKind} == {
        'heading', 'paragraph', 'list_item', 'table', 'figure',
        'caption', 'key_couplet'}


# ---------------------------------------------------------------------
# Section paths
# ---------------------------------------------------------------------

@NOT_YET
def test_section_paths_follow_heading_nesting() -> None:
    blocks = assign_section_paths([
        _block(text='before any heading'),
        _heading('Results', 1),
        _block(text='r'),
        _heading('Taxonomy', 2),
        _block(text='t'),
        _heading('Discussion', 1),
        _block(text='d'),
    ])
    assert [b.section_path for b in blocks] == [
        (),
        (),
        ('Results',),
        ('Results',),
        ('Results', 'Taxonomy'),
        (),
        ('Discussion',),
    ]


@NOT_YET
def test_section_paths_allow_skipped_levels() -> None:
    blocks = assign_section_paths([
        _heading('A', 1),
        _heading('C', 3),
        _block(text='under C'),
        _heading('B', 2),
        _block(text='under B'),
    ])
    assert [b.section_path for b in blocks] == [
        (), ('A',), ('A', 'C'), ('A',), ('A', 'B')]


@NOT_YET
def test_assign_section_paths_keeps_everything_else() -> None:
    doc = _document()
    again = assign_section_paths(doc.blocks)
    assert again == doc.blocks


# ---------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------

@NOT_YET
def test_assertions_for_a_layer_in_reading_order() -> None:
    doc = _document()
    assert doc.assertions_for('treatment') == [
        (1, Assertion('treatment', 'start', 'taxpub'))]
    assert doc.assertions_for('block-label') == [
        (1, Assertion('block-label', 'Nomenclature', 'jats-sec-type',
                      label_set='v4'))]
    assert doc.assertions_for('layout') == []


# ---------------------------------------------------------------------
# Serialization: the interchange format
# ---------------------------------------------------------------------

@NOT_YET
def test_round_trip_is_exact() -> None:
    doc = _document()
    assert Document.from_dict(doc.to_dict()) == doc


@NOT_YET
def test_round_trip_survives_json() -> None:
    # CouchDB stores JSON, so tuples come back as lists.
    doc = _document()
    data = json.loads(json.dumps(doc.to_dict(), ensure_ascii=False))
    assert Document.from_dict(data) == doc


@NOT_YET
def test_to_dict_is_tagged_with_the_schema() -> None:
    assert _document().to_dict()['schema'] == SCHEMA


@NOT_YET
def test_from_dict_refuses_another_schema() -> None:
    data = _document().to_dict()
    data['schema'] = 'document-model/2'
    with pytest.raises(ValueError, match='document-model/2'):
        Document.from_dict(data)


@NOT_YET
def test_from_dict_refuses_an_unknown_block_kind() -> None:
    data = _document().to_dict()
    data['blocks'][0]['kind'] = 'sidebar'
    with pytest.raises(ValueError):
        Document.from_dict(data)


@NOT_YET
def test_from_dict_refuses_an_unknown_source_kind() -> None:
    data = _document().to_dict()
    data['provenance']['source_kind'] = 'fax'
    with pytest.raises(ValueError):
        Document.from_dict(data)


@NOT_YET
def test_from_dict_validates_blocks() -> None:
    # Stored data gets the same checks as constructed data.
    data = _document().to_dict()
    data['blocks'][0]['level'] = None   # a heading with no level
    with pytest.raises(ValueError):
        Document.from_dict(data)


# ---------------------------------------------------------------------
# Markdown: a printing, not the interchange format
# ---------------------------------------------------------------------

@NOT_YET
def test_markdown_prints_each_kind() -> None:
    md = _document().to_markdown()
    assert md.split('\n\n') == [
        '# Taxonomy',
        'Amanita nova sp. nov.',
        '## Key to species',
        '1. Spores smooth ... 2',
        '- Holotype: NY 123',
        '| a | b |\n|---|---|\n| 1 | 2 |',
        '![Basidiospores]()',
        '*Fig. 1. Basidiospores. Bar = 10 µm.*',
    ]


@NOT_YET
def test_markdown_escapes_text_that_would_change_kind() -> None:
    # A paragraph starting with '#', '- ' or '2. ' must not print as a
    # heading or a list.
    doc = Document(
        doc_id='d',
        blocks=(_block(text='# not a heading'), _block(text='- not a list'),
                _block(text='2. not a list')),
        provenance=_provenance())
    assert doc.to_markdown().split('\n\n') == [
        '\\# not a heading', '\\- not a list', '2\\. not a list']


@NOT_YET
def test_markdown_of_an_empty_document_is_empty() -> None:
    assert Document('d', (), _provenance()).to_markdown() == ''
