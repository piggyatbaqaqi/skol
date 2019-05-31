"""Process the ISO 4 abbreviation list.

$ wget --no-check-certificate https://www.issn.org/wp-content/uploads/2013/09/LTWA_20160915.txt
# LTWA is in UTF-16. The csv module only understands utf-8.
$ cat LTWA_20160915.txt | iconv -f utf-16 -t utf-8 > LTWA_20160915.txt
"""

import argparse
import csv
import re
from typing import Any, Iterable, List, Optional

class LTWA(object):

    _file = ...  # type: Optional[str]
    _data = ...  # type: Optional[Iterable[str]]
    _pattern = ...  # type: re._pattern_type

    def __init__(self, filename: Optional[str] = 'data/LTWA_20160915.txt', data: Optional[Iterable[str]] = None):
        if filename:
            self._file = open(filename, 'r')
        else:
            self._file = None
        self._data = data
        
        self._pattern = '|'.join([self.make_pattern(word) for word in self.read_records()])
            
        self._file.close()

    def contents(self):
        return self._file or self._data

    def read_records(self):
        for record in csv.DictReader(self.contents(), delimiter='\t'):
            if record['ABBREVIATIONS'] == 'n.a.':
                yield(record['WORD'])
            else:
                yield(record['ABBREVIATIONS'])

    def make_pattern(self, word: str):
        pattern = word.replace('.', '\\.')
        if pattern.startswith('-'):
            pattern = r'\w+' + pattern[1:]
        if pattern.endswith('-'):
            pattern = pattern[:-1] + '\w+'
        return pattern

    def match(self, word: str):
        return(re.match(self._pattern, word))

    def eat_tokens(self, line: str) -> (Any, str):
        match = self.match(line)
        while line and match:
            line = line[match.end():]
            ws = re.match('\s+', line)
            if ws:
                line = line[ws.end():]
            yield (match, line)
            match = self.match(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='filename of the UTF-8 tab-delimited LTWA')
    parser.add_argument('abbrevs', type=str, help='list of abbreviations to check')
    args = parser.parse_args()

    ltwa = LTWA(args.file)
    for (m, l) in ltwa.eat_tokens(args.abbrevs):
        if m:
            print('m: %s line: %s' % (m, l))
        else:
            print('m: None line: %s' % l)


if __name__ == '__main__':
    main()
