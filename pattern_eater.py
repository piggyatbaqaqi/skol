"""Process a list in a file into a large regex."""

import abc
import csv
import re
from typing import Any, Iterable, List, Optional

class Eater(object):

    _metaclass_ = abc.ABCMeta
    _file = ...  # type: Optional[Iterable[str]]
    _data = ...  # type: Optional[Iterable[str]]
    _pattern = ...  # type: re._pattern_type
    # These go at the beginning of the regex.
    _extra_regex = ...  # type: List[str]
    _filename = ...  # type: Optional[str]

    def __init__(self, filename: Optional[str] = None, data: Optional[Iterable[str]] = None):
        if filename is None:
            filename = self._filename
        if filename:
            self._file = open(filename, 'r')
        else:
            self._file = None
        self._data = data

        self._pattern = '|'.join(
            self._extra_regex + 
            sorted(set([self.make_pattern(word) for word in self.read_records()]), reverse=True))

        self._file.close()

    def contents(self):
        return self._file or self._data
    
    @abc.abstractmethod
    def read_records(self) -> Iterable[str]:
        """Generator that returns strings from self.contents()"""
        return []

    def make_pattern(self, word: str) -> str:
        """Convert word into a pattern fragment."""
        pattern = word.lower().replace('.', '\\.')
        return pattern

    def match(self, word: str):
        return(re.match(self._pattern, word.lower()))

    def eat_tokens(self, line: str) -> (Any, str):
        """Consume tokens that match the pattern.

        Also consumes whitespace.
        Args:
          line - The input fragment to tokenize.
        Returns:
          (re match object, unmatched remainder)
        """
        match = self.match(line)
        while line and match:
            line = line[match.end():]
            ws = re.match('[\s]+', line)
            if ws:
                line = line[ws.end():]
            yield (match, line)
            match = self.match(line)
