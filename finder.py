"""Find species descriptions."""

import argparse
import re
import sys
from typing import Iterable, List, Optional

class Paragraph(object):
    short_line = ...  # type: int
    lines = ...  # type: Iterable[str]
    successor = ...  # type: Optional[str]

    def __init__(self, short_line=40):
        self.short_line = short_line
        self.lines = []
        self.successor = None

    def append(self, line: str):
        self.lines.append(line)

    def append_ahead(self, line: str):
        if self.successor is not None:
            self.lines.append(self.successor)
        self.successor = line

    def __str__(self) -> str:
        return '\n'.join(self.lines) + '\n'

    def __repr__(self) -> str:
        return 'Paragraph(%s)\n' % repr(str(self))

    def is_header(self) -> bool:
        return self.lines and self.lines[0].startswith('')

    def startswith(self, tokens: List[str]) -> bool:
        if not self.lines:
            return False
        tokenized = self.lines[0].strip().split()
        if not tokenized:
            return False
        first_token = tokenized[0].lower()
        return first_token in tokens

    def is_figure(self):
        return self.startswith([
            'fig', 'fig.', 'figs', 'figure', 'figures', 'plate', 'plates',
        ])

    def is_table(self):
        return self.startswith([
            'table', 'tbl.', 'tbl'
        ])

    def is_blank(self):
        return self.lines and all(not line for line in self.lines)

    def last_line(self):
        if not self.lines:
            return None
        return self.lines[-1]

    def close(self):
        if self.successor:
            self.lines.append(self.successor)
            self.successor = None

    def endswith(self, s: str):
        last_line = self.last_line()
        return last_line and last_line.endswith(s)


def parse_paragraphs(contents: Iterable[str]):
    pp = Paragraph()
    for line in contents:
        line = line.rstrip('\n')
        pp.append_ahead(line)
        # Tables continue to grow as long as we have short lines.
        if pp.is_table():
            if len(line) < pp.short_line:
                continue
            retval = pp
            pp = Paragraph()
            pp.append(retval.successor)
            yield retval
            continue

        # Blocks of blank lines are a paragraph.
        if pp.is_blank():
            if not line:
                continue
            retval = pp
            pp = Paragraph()
            pp.append(retval.successor)
            yield retval
            continue

        # Figures end with a blank line, or a period at the end of a
        # line.
        if pp.is_figure():
            if line and not pp.endswith('.'):
               continue
            retval = pp
            pp = Paragraph()
            pp.append(retval.successor)
            yield retval
            continue

        # A period before a newline marks the end of a paragraph.
        if pp.endswith('.'):
            retval = pp
            pp = Paragraph()
            pp.append(retval.successor)
            yield retval
            continue

        # A blank line ends a paragraph.
        if line == '':
            retval = pp
            pp = Paragraph()
            pp.append(retval.successor)
            yield retval
            continue

    pp.close()
    yield pp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="the file to search for descriptions")
    parser.add_argument("--dump_phase1", help="Dump the output of phase 1 and exit.", action="store_true")
    parser.add_argument("--dump_phase2", help="Dump the output of phase 2 and exit.", action="store_true")
    args = parser.parse_args()
    print(args.file)

    contents = open(args.file, 'r')

    output = []

    for paragraph in parse_paragraphs(contents):
        if (paragraph.is_header() or
            paragraph.is_figure() or
            paragraph.is_table()):
            continue

        output.append(paragraph)

    if args.dump_phase1:
        print(repr(list(output)))
        sys.exit(0)

    if args.dump_phase2:
        print(repr(list(output)))
        sys.exit(0)
        

if __name__ == '__main__':
    main()
