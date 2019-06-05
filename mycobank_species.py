"""Process the species list from MycoBank.

$ wget https://www.mycobank.org/localfiles/MBList.zip
# Unzip and convert to csv with soffice.
"""

import argparse
import csv
import re
import sys
from typing import Iterator, Optional

import tokenizer

class MycoBankSpecies(tokenizer.HashTokenizer):

    _filename = 'data/species/Export.csv'

    _extra_regex = []

    def read_records(self) -> Iterator[str]:
        for record in csv.DictReader(self.contents()):
            for name in record['Taxon_name'].replace('*', '').split():
                if name in ['[unranked]', '?']:
                    continue
                if name:
                    yield(name)

    def make_pattern(self, word: str) -> str:
        pattern = super(MycoBankSpecies, self).make_pattern(word)
        return pattern.replace('(', r'\(').replace(')', r'\)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file',
                        type=str,
                        help='filename of the UTF-8 csv MycoBank Export')
    parser.add_argument('taxa',
                        nargs='*',
                        type=str,
                        help='list of taxa to check for --tokenize')
    parser.add_argument('--tokenize',
                        help='Process tokens from the command line.',
                        action='store_true')
    args = parser.parse_args()

    mycobank = MycoBankSpecies(args.file)

    if args.tokenize:
        for taxon in args.taxa:
            for (m, l) in mycobank.tokenize(taxon):
                if m:
                    print('m: %s line: %s' % (m, l))
                else:
                    print('m: None line: %s' % l)


if __name__ == '__main__':
    main()
