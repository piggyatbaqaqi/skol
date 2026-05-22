'''A mixin class for adding fields to catalog elements.

Vendored from autonlab/ngautonml@main (Apache-2.0) — see
``ngautonml/catalog/catalog_element_mixin.py``.  Identical content
except for the import path of CatalogNameError.
'''

# Copyright (c) 2023 Carnegie Mellon University
# Vendored under Apache-2.0; see catalog.py.

from typing import Dict, List, Optional

from .catalog import CatalogNameError


class CatalogElementMixin():
    '''Contains properties required for all catalog elements'''
    _name: Optional[str] = None
    _tags: Optional[Dict[str, List[str]]] = None

    def __init__(self,
                 name: Optional[str] = None,
                 tags: Optional[Dict[str, List[str]]] = None,
                 **unused_kwargs):
        super().__init__()
        tmp_name = name or self._name
        if tmp_name is not None:
            self._name = tmp_name.lower()
        if self._name is None:
            raise CatalogNameError(f'{self.__class__.__name__} needs to define _name.')

        if self._tags is None:
            tmp_tags = tags or {}
        else:
            tmp_tags = self._tags.copy()
            if tags is not None:
                tmp_tags.update(tags)
        self._tags = {}
        for key, val in tmp_tags.items():
            if isinstance(val, str):
                val = [val]
            if not isinstance(val, list):
                raise NotImplementedError
            self._tags[key.lower()] = [vv.lower() for vv in val]

    @property
    def name(self) -> str:
        '''The catalog name of the object'''
        assert self._name is not None
        return self._name

    @property
    def tags(self) -> Dict[str, List[str]]:
        '''The catalog tags of the object'''
        assert self._tags is not None
        return self._tags.copy()

    def _tag_to_bool(self, tagname: str, default: bool) -> bool:
        '''Interprets a list of tag values as a boolean.

        As of 01-03-2024, only looks at first value in list,
        and checks if it equals 'true' (case-insensitive).

        If the tag is not set, will return default.
        '''
        assert self._tags is not None
        if tagname in self._tags:
            return self._tags[tagname][0].lower() == 'true'
        return default
