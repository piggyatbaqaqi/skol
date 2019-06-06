"""Process a list in a file into a large regex."""

import abc
import csv
import re
from typing import Any, Iterable, Iterator, List, Optional, Set, Tuple, Union

class Tokenizer(object):

    _metaclass_ = abc.ABCMeta
    _file: Optional[Iterable[str]]
    _pattern: Union[Any, Set[str]]
    # These go at the beginning of the regex.
    _extra_regex: List[str]

    _filename: Optional[str]
    _data: Optional[Iterable[str]]

    def __init__(self, filename: Optional[str] = None, data: Optional[Iterable[str]] = None) -> None:
        if filename is None:
            filename = self._filename
        if filename:
            self._file = open(filename, 'r')
        else:
            self._file = None
        self._data = data

        self._pattern = self.build_pattern()

        if self._file is not None:
            self._file.close()

    def build_pattern(self) -> Union[Any, Set[str]]:
        return '|'.join(
            self._extra_regex +
            sorted(set([self.make_pattern(word) for word in self.read_records()]),
                   reverse=True))

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

    def tokenize(self, line: str) -> Iterator[Tuple[Any, str]]:
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


class HashTokenizer(Tokenizer):

    def build_pattern(self):
        return set(self.read_records())

    def make_pattern(self, word: str) -> str:
        """Convert word into a pattern fragment."""
        return word

    def split(self, line: str) -> List[str]:
        return line.split()

    @abc.abstractmethod
    def read_records(self) -> Iterable[str]:
        """Generator that returns strings from self.contents()"""
        return []

    def tokenize(self, line: str) -> Iterator[Tuple[Any, str]]:
        """Consume tokens that match the pattern.

        Also consumes whitespace.
        Args:
          line - The input fragment to tokenize.
        Returns:
          (matched token (str), unmatched remainder)
        """
        tokens = self.split(line)
        while tokens:
            if tokens[0].lower() in self._pattern:
                retval = tokens.pop(0)
                yield(retval, ' '.join(tokens))
            else:
                yield(None, ' '.join(tokens))
                break
