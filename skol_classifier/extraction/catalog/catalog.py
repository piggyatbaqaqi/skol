'''Base module for catalogs in the skol extraction pipeline.

Vendored from autonlab/ngautonml@main (Apache-2.0) — see
``ngautonml/catalog/catalog.py`` in that repository.  Adapted for
skol with no behavioural changes; only import paths differ.
'''

# Copyright (c) 2023 Carnegie Mellon University
# This code is subject to the license terms contained in the LICENSE file
# in the autonlab/ngautonml repository (Apache-2.0).
# Vendored into skol with no behavioural changes.

import abc
from typing import Dict, Generic, Iterable, List, Set, Tuple, TypeVar, Union

T = TypeVar("T")


class CatalogError(Exception):
    '''Base error class for the catalog module'''


class CatalogLookupError(CatalogError, LookupError):
    '''The requested key is not present.'''


class CatalogNameError(CatalogError, LookupError):
    '''The object does not have a name attribute.'''


class CatalogValueError(CatalogError, ValueError):
    '''An improper value has been sent.'''


class Catalog(Generic[T], metaclass=abc.ABCMeta):
    '''Generic Catalog class for managing swappable components.

    Allows search by name of the component, and a lookup
    by classification or property.
    '''

    @property
    @abc.abstractmethod
    def tagtypes(self) -> Set[str]:
        '''Retrieve list of all tag types present in catalog'''

    @abc.abstractmethod
    def all_objects(self) -> Iterable[T]:
        '''Retrieve all registered objects.'''

    @abc.abstractmethod
    def items(self) -> Iterable[Tuple[str, T]]:
        '''List all the items in the catalog.'''

    @abc.abstractmethod
    def tagvals(self, tagtype: str) -> Set[str]:
        '''Retrieve all tag values present for a given tag type'''

    @abc.abstractmethod
    def lookup_by_name(self, name: str) -> T:
        '''Find an object registered under name.'''

    @abc.abstractmethod
    def lookup_by_tag_and(self, **tags: Union[str, Iterable[str]]) -> Dict[str, T]:
        '''Find objects with a specified tag for all specified tag types.

        \\*\\*tags looks like:

            tag_type = tag_value
            OR
            tag_type = (tag_value1, tag_value2)

        Needs to match at least one specified tag value for all specified tag types.
        '''


def upcast(tags: Dict[str, List[str]]) -> Dict[str, Union[str, List[str]]]:
    '''Cast a Dict[str, List[str]] into a Dict[str, Union[str, List[str]]]

    This is only needed to work within typing constraints.
    '''
    return dict(tags.items())
