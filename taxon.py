"""Represent Nomenclature paragraphs and matching Descriptions."""
import itertools
from typing import Any, Dict, Iterable, Iterator, List, Optional

from paragraph import Paragraph
from label import Label
from line import Line


def get_ingest_field(record: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Get a field from a taxa record using INGEST field names.

    Callers should always use original ingest field names:
    - '_id' for document ID (not 'doc_id')
    - 'url' for human URL (not 'human_url')
    - 'pdf_url' for PDF URL

    Tries 'ingest' first (new format), falls back to 'source' (old format)
    with automatic name translation for backward compatibility.

    Args:
        record: Taxa record dict (may have 'ingest', 'source', or both)
        *keys: Field path using INGEST names (e.g., 'url', '_id', 'pdf_url')
        default: Value to return if field not found

    Returns:
        Field value or default

    Examples:
        get_ingest_field(record, '_id')      # Gets ingest._id or source.doc_id
        get_ingest_field(record, 'url')      # Gets ingest.url or source.human_url
        get_ingest_field(record, 'pdf_url')  # Gets ingest.pdf_url or source.pdf_url
    """
    # Try ingest first (new format) - use keys directly
    ingest = record.get('ingest')
    if ingest is not None:
        result: Any = ingest
        for key in keys:
            if not isinstance(result, dict):
                result = None
                break
            result = result.get(key)
            if result is None:
                break
        if result is not None:
            return result

    # Fall back to source (old format) - translate ingest names to source names
    source = record.get('source')
    if source is not None and keys:
        # Map ingest field names → source field names
        ingest_to_source: Dict[str, str] = {
            '_id': 'doc_id',
            'url': 'human_url',
            'pdf_url': 'pdf_url',
        }
        mapped_key = ingest_to_source.get(keys[0], keys[0])
        if isinstance(source, dict):
            return source.get(mapped_key, default)

    return default


class Taxon(object):
    FIELDNAMES = [
        'serial_number',
        'filename', 'human_url', 'pdf_url', 'label', 'paragraph_number', 'pdf_page',
        'pdf_label', 'empirical_page_number', 'body'
    ]
    LONG_GAP = 6  # 6 Paragraphs is long enough to give up.


    _nomenclatures: List[Paragraph]
    _descriptions: List[Paragraph]
    _serial: int = 0

    def __init__(self):
        self.__class__._serial += 1
        self._serial = self.__class__._serial
        self.reset()

    def __repr__(self) -> str:
        return repr(list(self.dictionaries()))

    def reset(self):
        self._nomenclatures = []
        self._descriptions = []

    def add_nomenclature(self, pp: Paragraph) -> None:
        self._nomenclatures.append(pp)

    def add_description(self, pp: Paragraph) -> None:
        self._descriptions.append(pp)

    def been_too_long(self, pp: Paragraph) -> bool:
        if self._descriptions:
            pp_num = self._descriptions[-1].paragraph_number
        elif self._nomenclatures:
            pp_num = self._nomenclatures[-1].paragraph_number
        else:
            return False
        return pp.paragraph_number - pp_num > self.LONG_GAP

    def has_nomenclature(self) -> bool:
        return bool(self._nomenclatures)

    def has_description(self) -> bool:
        return bool(self._descriptions)

    def doc_id(self) -> str | None:
        '''Return the doc_id from the first nomenclature paragraph, if any.'''
        if self._nomenclatures:
            first_line = self._nomenclatures[0].first_line
            return first_line.doc_id if first_line else None
        return None

    def dictionaries(self) -> Iterator[Dict[str, str]]:
        for pp in itertools.chain(self._nomenclatures, self._descriptions):
            d = pp.as_dict()
            d['serial_number'] = str(self._serial)
            yield d

    def human_url(self) -> str | None:
        '''Return the human_url from the first nomenclature paragraph, if any.'''
        return self.as_row().get('source', {}).get('human_url')

    def as_row(self) -> Dict[str, None | str | int | Dict[str, None | str | int]]:
        '''Convert this Taxon to a dictionary suitable for output.'''

        # Pull other fields from self._nomenclatures[0]
        pp = self._nomenclatures[0]
        first_line = pp.first_line
        assert first_line is not None, "Nomenclature paragraph must have at least one line"
        source_doc_id = first_line.doc_id or "unknown"
        source_url = first_line.human_url
        source_pdf_url = first_line.pdf_url
        source_db_name = first_line.db_name or "unknown"
        line_number = first_line.line_number
        ingest = first_line.ingest

        retval: Dict[str, None | str | int | Dict[str, Any]] = {
            'taxon': "\n".join((str(pp) for pp in self._nomenclatures)),
            'description': "\n".join((str(pp) for pp in self._descriptions)),
            'source': {
                'doc_id': source_doc_id,
                'human_url': source_url,
                'pdf_url': source_pdf_url,
                'db_name': source_db_name,
            },
            'ingest': ingest,
            'line_number': line_number,
            'paragraph_number': pp.paragraph_number,
            # pdf_page comes from "--- PDF Page N Label L---" markers in text (from pdf_section_extractor.py)
            # Will be 0 if markers are not present in the text
            'pdf_page': pp.pdf_page,
            'pdf_label': pp.pdf_label,
            'empirical_page_number': str(pp.empirical_page_number) if pp.empirical_page_number is not None else None,
        }
        return retval



def group_paragraphs(paragraphs: Iterable[Paragraph]) -> Iterator[Taxon]:
    nomenclature = Label('Nomenclature')
    description = Label('Description')
    state = 'Start a Taxon'
    taxon = Taxon()
    state = 'Look for Nomenclatures'
    for pp in paragraphs:
        if state == 'Look for Nomenclatures':
            if pp.top_label() == nomenclature:
                taxon.add_nomenclature(pp)
                continue
            if pp.top_label() == description:
                if taxon.has_nomenclature():
                    # Check if description is from same document as nomenclature
                    pp_doc_id = pp.first_line.doc_id if pp.first_line else None
                    taxon_doc_id = taxon.doc_id()
                    if pp_doc_id and taxon_doc_id and pp_doc_id != taxon_doc_id:
                        # Different document - reset and skip this description
                        taxon.reset()
                        continue
                    state = 'Look for Descriptions'
                    # Fall through to the description handling below.
                else:
                    # Found a Description without a preceding Nomenclature
                    # Create a stub nomenclature paragraph with "Nomen undetected"
                    stub_line = Line("Nomen undetected")
                    stub_paragraph = Paragraph(labels=[nomenclature], lines=[stub_line],
                                               paragraph_number=pp.paragraph_number)
                    taxon.add_nomenclature(stub_paragraph)
                    state = 'Look for Descriptions'
                    # Fall through to the description handling below.
            if taxon.been_too_long(pp):
                taxon.reset()
                continue
            # Fall through in case we just found a description.
        if state == 'Look for Descriptions':
            # Check if we've crossed a document boundary
            pp_doc_id = pp.first_line.doc_id if pp.first_line else None
            taxon_doc_id = taxon.doc_id()
            if pp_doc_id and taxon_doc_id and pp_doc_id != taxon_doc_id:
                # Document boundary crossed - yield current taxon and reset
                if taxon and taxon.has_description() and taxon.has_nomenclature():
                    yield taxon
                taxon = Taxon()
                state = 'Look for Nomenclatures'
                # Re-process this paragraph in the new state
                if pp.top_label() == nomenclature:
                    taxon.add_nomenclature(pp)
                continue

            if pp.top_label() == description:
                taxon.add_description(pp)
                continue
            if pp.top_label() == nomenclature:
                if taxon.has_description():
                    # We have a complete taxon, yield it and start a new one
                    if taxon.has_nomenclature():
                        yield taxon
                    taxon = Taxon()
                    taxon.add_nomenclature(pp)
                    state = 'Look for Nomenclatures'
                else:
                    # No description yet - add this nomenclature to the current taxon
                    # This handles cases where multiple nomenclature blocks appear
                    # before the description
                    taxon.add_nomenclature(pp)
                continue

            if taxon.been_too_long(pp):
                if taxon and taxon.has_description() and taxon.has_nomenclature():
                    yield taxon
                taxon = Taxon()
                state = 'Look for Nomenclatures'
                continue

    if taxon and taxon.has_description() and taxon.has_nomenclature():
        yield taxon
