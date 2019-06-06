"""Process the author list from MycoBank.

$ wget https://www.mycobank.org/localfiles/MBList.zip
# Unzip and convert to csv with soffice.
"""

import argparse
import csv
import re
import sys
from typing import Iterator, Optional

import tokenizer

class MycoBankAuthors(tokenizer.HashTokenizer):

    _filename = 'data/species/Export.csv'

    _extra_regex = []

    def read_records(self) -> Iterator[str]:
        for record in csv.DictReader(self.contents()):
            for name in record['Authors'].lower().split():
                name = name.strip().translate(str.maketrans('', '', '()&"'))
                if name:
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

    mycobank = MycoBankAuthors(args.file)

    if args.tokenize:
        for name in args.names:
            for (m, l) in mycobank.tokenize(name):
                if m:
                    print('m: %s line: %s' % (m, l))
                else:
                    print('m: None line: %s' % l)


if __name__ == '__main__':
    main()
