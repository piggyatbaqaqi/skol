'''In-memory catalog implementation.

Vendored from autonlab/ngautonml@main (Apache-2.0) — see
``ngautonml/catalog/memory_catalog.py``.  Two changes from upstream:

1. The ``CATALOG_IGNORE`` constant (upstream
   ``ngautonml.wrangler.constants.CATALOG_IGNORE``, value ``'impl'``)
   is inlined as a module-level constant here.
2. The custom upstream ``Logger`` is replaced with Python's standard
   ``logging`` module.

Otherwise identical.
'''

# Copyright (c) 2023 Carnegie Mellon University
# Vendored under Apache-2.0; see catalog.py.

import logging
from collections import defaultdict
from importlib import util
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union

from .catalog import (Catalog, CatalogError, CatalogLookupError,
                      CatalogNameError, CatalogValueError)


# Directory name treated as private inside an autoloaded tree.  Files
# under any directory with this name are skipped.  Upstream this is
# imported from ngautonml.wrangler.constants.
CATALOG_IGNORE = 'impl'

log = logging.getLogger(__name__)

T = TypeVar("T")


class CatalogNoRegister(CatalogError):
    '''A module lacks a register function.'''


class CatalogDuplicateError(CatalogError):
    '''We attempted to insert an element with a duplicate key.'''


class MemoryCatalog(Catalog[T]):
    '''In memory Generic Catalog class for managing swappable components.

    This version of Catalog allows registering an element and loading a directory.

    Allows search by name of the component, and a lookup
    by classification or property.
    '''
    _kwargs: Dict[str, Any]
    __cat_names: Dict[str, T]
    __tags: Dict[str, Dict[str, List[Tuple[str, T]]]]
    # "tag_name": {
    #     "tag_value"{
    #         [
    #             ("modelname1", Model1),
    #             ("modelname2", Model2)
    #         ]
    #     }
    # }
    __names_by_tag: Dict[str, Dict[str, List[str]]]
    # "tag_name": {
    #     "tag_value"{
    #         [
    #             "modelname1",
    #             "modelname2"
    #         ]
    #     }
    # }
    __tags_by_name: Dict[str, Dict[str, List[str]]]
    # "modelname1": {
    #     "tag_name"{
    #         [
    #             "tag_value1",
    #             "tag_value2"
    #         ]
    #     }
    # }

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.__cat_names = {}
        self.__tags = defaultdict(lambda: defaultdict(list))
        self.__names_by_tag = defaultdict(lambda: defaultdict(list))
        self.__tags_by_name = defaultdict(lambda: defaultdict(list))

    def register(self, obj: T, name: Optional[str] = None,
                 tags: Optional[Dict[str, Union[str, List[str]]]] = None) -> str:
        '''Register an object under the name.'''
        obj_name = name
        if obj_name is None:
            if hasattr(obj, 'name'):
                obj_name = getattr(obj, 'name')
            else:
                if isinstance(obj, type):
                    obj_signifier = obj.__name__
                else:
                    obj_signifier = obj.__class__.__name__

                raise CatalogNameError(
                    'Name must be defined for Catalog Object: '
                    f'{obj_signifier}')
        obj_name = obj_name.lower()

        if obj_name in self.__cat_names:
            raise CatalogDuplicateError(obj_name)

        self.__cat_names[obj_name] = obj

        if tags is None:
            tags = getattr(obj, 'tags', None)

        if tags is not None:
            for ttype, tvals in tags.items():
                if isinstance(tvals, str):
                    tvals = [tvals]
                for tval in tvals:
                    if not isinstance(tval, str):
                        raise CatalogValueError(f'{ttype}:{tval} is a {type(tval)}, not string')
                    self.__tags[ttype.lower()][tval.lower()].append((obj_name, obj))
                    self.__names_by_tag[ttype.lower()][tval.lower()].append(obj_name)
                    self.__tags_by_name[obj_name][ttype.lower()].append(tval.lower())
        return obj_name

    @property
    def tagtypes(self) -> Set[str]:
        '''Retrieve list of all tag types present in catalog'''
        return set(self.__tags.keys())

    def all_objects(self) -> Iterable[T]:
        '''Retrieve all registered objects.'''
        return self.__cat_names.values()

    def tagvals(self, tagtype: str) -> Set[str]:
        '''Retrieve all tag values present for a given tag type'''
        if tagtype in self.__tags.keys():
            return set(self.__tags[tagtype].keys())
        return set()

    def lookup_by_name(self, name: str) -> T:
        '''Find an object registered under name.'''
        obj_name = name.lower()
        if obj_name in self.__cat_names:
            return self.__cat_names[obj_name]

        raise CatalogLookupError(name)

    def lookup_by_tag_and(self, **tags: Union[str, Iterable[str]]) -> Dict[str, T]:
        '''Find objects with a specified tag for all specified tag types.

        \\*\\*tags looks like:
            tag_type = tag_value
            OR
            tag_type = (tag_value1, tag_value2)

        Needs to match at least one specified tag value for all specified tag types.
        '''
        result: Dict[str, T] = self.__cat_names.copy()
        for tag_type, tag_values in tags.items():
            if isinstance(tag_values, str):
                tag_values = (tag_values,)

            result = {
                name: obj
                for name, obj in result.items()
                if name in self._union_tag_search(tag_type, tag_values)
            }

        return result

    def _union_tag_search(self,
                          tag_type: str,
                          tag_values: Iterable[str]) -> List[str]:
        '''Return all the names in the catalog that match at least 1 provided tag value'''
        found_names: Set[str] = set()
        tag_type = tag_type.lower()
        for tag_value in tag_values:
            tag_value = tag_value.lower()
            found_names.update(set(self.__names_by_tag[tag_type][tag_value]))
        return list(found_names)

    def load(self, module_directory: Path):
        '''Load all the catalog modules from a directory hierarchy.

        Normal usage is to set module_directory to Path(__file__).parents[1].

        Each module in the hierarchy needs a "register(catalog, \\*arg, \\*\\*kwargs)"
        function that will register all the catalog objects in the module.
        '''
        module_directory = module_directory.resolve()
        root = python_root(str(module_directory))
        filepaths = []

        # Build up the list of files to examine.
        for rootdir, dirs, filenames in os.walk(module_directory, topdown=True):
            if CATALOG_IGNORE in dirs:
                log.debug('dropping %s/%s', rootdir, CATALOG_IGNORE)
                dirs[:] = [d for d in dirs if d != CATALOG_IGNORE]
            for filename in filenames:
                if self._skip(rootdir, filename):
                    continue
                filepaths.append(os.path.abspath(os.path.join(rootdir, filename)))
        # Load all the modules calling register (if present).
        for filepath in filepaths:
            modname = module_name_from_filepath(filepath, root)

            module = self._load_module(filepath, modname)

            # At this point we think we have a valid module.
            if hasattr(module, 'register'):
                module.register(self, **self._kwargs)
            else:
                raise CatalogNoRegister(
                    f'autoloader: module {module.__name__} '
                    f'has no register, dir(module): {dir(module)}')

    def _skip(self, rootdir, filename) -> bool:
        '''Returns True if the autoloader should skip this file, and logs it.'''
        if not filename.endswith('.py'):
            log.debug('autoloader dropping non-python file %s/%s', rootdir, filename)
            return True
        if filename.startswith('__'):
            log.debug('autoloader dropping special file %s/%s', rootdir, filename)
            return True
        if filename.endswith('_test.py'):
            log.debug('autoloader dropping test path %s/%s', rootdir, filename)
            return True
        return False

    def _load_module(self, filepath, modname):
        '''Load the specified file as a module.'''
        if modname in sys.modules:
            # If module is already loaded: don't reload it.
            log.debug('%s is already loaded', modname)
            module = sys.modules[modname]

        else:
            # Load the file as a module.
            spec = util.spec_from_file_location(modname, filepath)
            if spec is None:
                raise ImportError(f"can't make spec for module {modname} at {filepath}")
            module = util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f'loader for module {modname} is None')
            spec.loader.exec_module(module)
            sys.modules[modname] = module
        return module

    def items(self) -> Iterable[Tuple[str, T]]:
        return self.__cat_names.items()


def module_name_from_filepath(filepath: str, root_path: str) -> str:
    """Converts a path to a .py file into a Python module name.

    E.g. for a root_path of "/home/piggy/src/autonml" and a filepath of
    "/home/piggy/src/autonml/ngautonml/algorithms/connect.py", we get
    a module name of "ngautonml.algorithms.connect".

    Args:
        filepath: The path to the .py file.
        root_path: The path to the root of the Python tree that the module lives in.

    Returns:
        The full name of the module pointed at by filepath.
    """
    filepath = os.path.abspath(filepath)
    root_path = os.path.abspath(root_path)

    # Get the relative path to the .py file from the root of the Python tree.
    relative_path = os.path.relpath(path=filepath, start=root_path)

    # Split the relative path on the last "/".
    path_components = relative_path.split(os.sep)

    # Strip off the .py extension from the last component of the path, if it exists.
    if path_components[-1].endswith(".py"):
        path_components[-1] = path_components[-1][:-3]

    # Join the path components with dots.
    module_name = ".".join(path_components)

    return module_name


def python_root(filepath: str) -> str:
    """Find the top directory with an __init__.py in it.

    Args:
        filepath: The path to a Python file.

    Returns:
        The prefix that points to the highest directory with an __init__.py.
    """

    filepath = os.path.abspath(filepath)
    path_components = filepath.split(os.sep)

    # Find the first directory with an __init__.py in it.
    for i in range(len(path_components)):
        if os.path.exists(os.path.join(*path_components[:i], "__init__.py")):
            break

    # Remove all directories before the directory with an __init__.py from the path.
    # Please don't put __init__.py in /.
    path_components = path_components[:i - 1]

    # Join the path components with os.path.join.
    prefix = os.path.sep + os.path.join(*path_components)

    return prefix
