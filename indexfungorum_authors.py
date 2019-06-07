"""Process the author list from Index Fungorum.

This includes author abbreviations, unlike the MycoBank data.

$ wget http:/www.indexfungorum.org/FungalNameAuthors.rtf
$ unrtf --text FungalNameAuthors.rtf | iconv -f ISO-8859-15 -t utf-8 > FungalNameAuthors.txt
"""

import argparse
import csv
import re
import sys
from typing import Iterator, List, Optional

import tokenizer

class IndexFungorumAuthors(tokenizer.HashTokenizer):

    _filename = 'data/authors/FungalNameAuthors.txt'

    _extra_words: List[str] = ['apud', 'ex', 'in', ':']

    def split(self, line: str) -> List[str]:
        return line.translate(str.maketrans(',[]()-', '      ')).split()

    def read_records(self) -> Iterator[str]:
        for word in self._extra_words:
            yield(word)
        start = True
        for line in self.contents():
            if start:
                if line == '-----------------\n':
                    start = False
                continue
            # Remove trailing publication data and forename notation.
            line = re.sub(r'\([^)]*\)$|\[no forename[s]*\]', '', line.lower().strip())
            # Catch alternative spellings.
            m = re.match(r'\((.*)\) see (.*)', line)
            if m:
                line = ' '.join(m.groups())
            line = line.translate(str.maketrans(',[]()-', '      '))
            for name in line.split():
                yield(name)
                if '.' in name:
                    for initial in name.split('.'):
                        if initial:
                            yield(initial + '.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file',
                        type=str,
                        help='filename of the UTF-8 csv MycoBank Export')
    parser.add_argument('names',
                        nargs='*',
                        type=str,
                        help='list of names to check for --tokenize')
    parser.add_argument('--tokenize',
                        help='Process tokens from the command line.',
                        action='store_true')
    args = parser.parse_args()

    indexfungorum = IndexFungorumAuthors(args.file)

    if args.tokenize:
        for name in args.names:
            for (m, l) in indexfungorum.tokenize(name):
                if m:
                    print('m: %s line: %s' % (m, l))
                else:
                    print('m: None line: %s' % l)


if __name__ == '__main__':
    main()
