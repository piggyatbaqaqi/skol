"""Process the ISO 4 abbreviation list.

$ wget --no-check-certificate https://www.issn.org/wp-content/uploads/2013/09/LTWA_20160915.txt
# LTWA is in UTF-16. The csv module only understands utf-8.
$ cat LTWA_20160915.txt | iconv -f utf-16 -t utf-8 > LTWA_20160915.txt
"""

import argparse
import csv
from typing import Iterator, Optional

import tokenizer

class LTWA(tokenizer.Tokenizer):

    _filename = 'data/LTWA_20160915.txt'

    # These go at the beginning of the regex.
    _extra_regex = [
        r'\b[mdclxvi]+\b',  # Roman numerals.
        r'[-(),&.:\']',     # Extra punctuation.
        r'\d+',             # Numbers.
        r'\w+ea',
    ]

    # I'd like to infer these.
    _places = [
        'andaman',
        'byochugaizashi',
        'corboda',
        'cornell',
        'formosa',
        'gifalnye',
        'java',
        'matsushima',
        'oesterreich',
        'rico',
        'roma',
        'sada',
        'sappro.',
        'siena',
        'ssr',
        'sssr',
        'taihoku',
        'tokyo',
        'torrey',
        'tottori',
        'ussr',
        'wien',
        'wisc.',
        'yamagata',
        'yokohama',
    ]

    _small_words = [
        'atti',
        'aric.',
        '-ales',
        '-ana',
        '-eae',
        '-ia',
        '-tax-',
        '-um',
        'algen',
        'arts',
        'griby',  # Russian 'mushrooms'
        'biology',
        'club',
        'crittog.',
        'crypt.',
        'cryptog.',
        'de',
        'fl.',
        'fór',
        'fungi',
        'fung.',
        'genus',
        'in',
        'microfung.',
        'national',
        'niz.',  # Russian
        'nova',
        'nuova',
        'of',
        'orto',  # Italian 'garden'
        'para',
        'pilz',  # German 'mushroom'
        'sta',
        'staz.',
        'sti.',  # Romanian 'sci.'
        'the',
        'u.',
        'u.s.d.a.',
        'vereins',  # German "club's"
    ]

    def read_records(self) -> Iterator[str]:
        for word in self._places:
            yield(word)
        for word in self._small_words:
            yield(word)
        for record in csv.DictReader(self.contents(), delimiter='\t'):
            yield(record['WORD'])
            if record['ABBREVIATIONS'] != 'n.a.':
                yield(record['ABBREVIATIONS'])
                # Allow abbreviations with spaces removed.
                if ' ' in record['ABBREVIATIONS']:
                    yield(record['ABBREVIATIONS'].replace(' ', ''))
                    yield(record['ABBREVIATIONS'].replace(' ', '').replace('.', ''))

    def make_pattern(self, word: str) -> str:
        pattern = super(LTWA, self).make_pattern(word)
        if pattern.startswith('-'):
            pattern = r'\w*' + pattern[1:]
        if pattern.endswith('-'):
            pattern = pattern[:-1] + r'\w*'
        if not pattern.endswith('.'):
            pattern = pattern + r'\b'
        return r'\b' + pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file',
                        type=str,
                        help='filename of the UTF-8 tab-delimited LTWA')
    parser.add_argument('abbrevs',
                        nargs='*',
                        type=str,
                        help='list of abbreviations to check for --tokenize')
    parser.add_argument('--tokenize',
                        help='Process tokens from the command line.',
                        action='store_true')
    parser.add_argument('--grep_v_file',
                        type=str,
                        help='Figure out how much of each line can be recognized.',
                        default=[],
                        action='append')
    args = parser.parse_args()

    ltwa = LTWA(args.file)
    if args.tokenize:
        for abbrev in args.abbrevs:
            for (m, l) in ltwa.tokenize(abbrev):
                print('m: %s line: %s' % (m, l))

    for filename in args.grep_v_file:
        f = open(filename, 'r')
        for line in f:
            for (m, l) in ltwa.tokenize(line):
                if m is None:
                    break
            print('%s -> %s' % (line.strip(), l))
            l = ''


if __name__ == '__main__':
    main()
