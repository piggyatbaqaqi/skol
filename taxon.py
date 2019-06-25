"""Represent Nomenclature paragraphs and matching Descriptions."""
import itertools
from typing import Dict, Iterable, Iterator, List

from paragraph import Paragraph
from label import Label

LONG_GAP = 20  # 20 Paragraphs is long enough to give up.

class Taxon(object):
    FIELDNAMES = [
        'serial_number',
        'filename', 'label', 'paragraph_number', 'page_number',
        'empirical_page_number', 'body'
    ]

    _nomenclatures: List[Paragraph]
    _descriptions: List[Paragraph]
    _serial: int = 0

    def __init__(self):
        self.__class__._serial += 1
        self.reset()

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
        return pp.paragraph_number - pp_num > LONG_GAP

    def has_nomenclature(self) -> bool:
        return bool(self._nomenclatures)
    
    def has_description(self) -> bool:
        return bool(self._descriptions)
    
    def dictionaries(self) -> Iterator[Dict[str, str]]:
        for pp in itertools.chain(self._nomenclatures, self._descriptions):
            d = pp.as_dict()
            d['serial_number'] = str(self._serial)
            yield d


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
                    state = 'Look for Descriptions'
            if taxon.been_too_long(pp):
                taxon.reset()
                continue
        if state == 'Look for Descriptions':
            if pp.top_label() == description:
                taxon.add_description(pp)
                continue
            if pp.top_label() == nomenclature:
                yield taxon
                taxon = Taxon()
                taxon.add_nomenclature(pp)
                state = 'Look for Nomenclatures'
                continue

            if taxon.been_too_long(pp):
                if taxon and taxon.has_description():
                    yield taxon
                taxon = Taxon()
                state = 'Look for Nomenclatures'
                continue

    if taxon and taxon.has_description():
        yield taxon
