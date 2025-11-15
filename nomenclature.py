"""Find nomenclature lines."""

import argparse
import numpy  # type: ignore
import regex as re  # type: ignore
import sys
import time
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import mycobank_authors
import mycobank_species
import indexfungorum_authors
import iso4
import tokenizer

class Nov(tokenizer.HashTokenizer):

    _filename = None
    _words = [
        'comb.', 'fam.', 'gen.', 'ined.', 'var.', 'subg.', 'subsp.', 'sp.',
        'f.', 'nov.'
    ]

    def read_records(self) -> Iterator[str]:
        for word in self._words:
            yield(word)


class Nomenclature(object):
    """Recursive descent parser for a nomenclature line."""
    _species = mycobank_species.MycoBankSpecies(name='name')
    _m_authors = mycobank_authors.MycoBankAuthors(name='author')
    _i_authors = indexfungorum_authors.IndexFungorumAuthors(name='author')
    _journals = iso4.LTWA(name='journal')
    _nov = Nov(name='novum')

    def _tokenize(self,
                  tokenizer: tokenizer.Tokenizer,
                  s: str,
                  to_raise: bool = False) -> Optional[str]:
        seen = False
        l = None
        for m, l in tokenizer.tokenize(s):
            if not seen and m is not None:
                seen = True
        if not seen:
            if to_raise:
                raise SyntaxError('No %s in %s' % (tokenizer.name, l))
            else:
                return None
        return l


    def name(self, s: str, to_raise: bool = False) -> Optional[str]:
        return self._tokenize(self._species, s, to_raise)

    def optional_not_name(self, s: str, to_raise: bool = False) -> Optional[str]:
        if self.name(s):
            return s
        m = re.match(r'\S+\s(.*)$', s, flags=re.DOTALL)
        if m:
            return m.group(1)
        else:
            return ''

    def authors(self, s: str, to_raise: bool = False) -> Optional[str]:
        seen = False
        l = s
        while True:
            for m, l in self._m_authors.tokenize(l):
                if not seen and m is not None:
                    seen = True
            for m, l in self._i_authors.tokenize(l):
                if not seen and m is not None:
                    seen = True
            if not l:
                if seen:
                    return ''
                else:
                    if to_raise:
                        raise SyntaxError('No authors in %s' % l)
                    else:
                        return None

            if m is None:
                if seen:
                    return l
                else:
                    if to_raise:
                        raise SyntaxError('No authors in %s' % l)
                    else:
                        return None
        return None

    def citation(self, s: str, to_raise: bool = False) -> Optional[str]:
        return self._tokenize(self._journals, s, to_raise)

    def nov(self, s: str, to_raise: bool = False) -> Optional[str]:
        return self._tokenize(self._nov, s, to_raise)

    def nomenclature(self, line: str, to_raise: bool = False) -> Optional[str]:
        s = self.optional_not_name(line)
        s = self.name(s, to_raise=True)
        author_rest = self.authors(s, to_raise=True)
        s = self.citation(author_rest)
        if s is None:
            s = self.nov(author_rest)
            if s is None:
                if to_raise:
                    raise SyntaxError('Neither citations nor novum in "%s"' % author_rest)
                else:
                    return None
        return s

    def nomenclature_list(self, line: str) -> Optional[str]:
        s = self.nomenclature(line, to_raise=True)
        while s:
            m = re.match(r'^\s*[≡=;]', s)
            if not m:
                return s
            rest = s[m.end():]
            s = self.nomenclature(rest)
            if s is None:
                return rest
        return s


def sample(files: List[str]) -> Iterator[str]:
    for filename in files:
        result = ''
        f = open(filename, 'r')
        for l in f:
            # Remove hyphenation.
            if l.endswith('-\n'):
                l = l[:-2]
            if l.strip():
                result += l
            else:
                retval = result
                result = ''
                yield retval


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='+', help='the file to search for nomenclatures')
    return parser.parse_args()


def main():
    args = define_args()

    n = Nomenclature()

    for s in sample(args.file):
        print(s.strip())
        try:
            rest = n.nomenclature_list(s)
        except SyntaxError as e:
            print('SyntaxError', e)
            print()
            continue
        print('rest: %s' % rest)
        print()

if __name__ == '__main__':
    main()
