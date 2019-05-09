"""Find species descriptions."""

import argparse
import re
import sys
from typing import Iterable, List, Optional

class Label(object):
    _label_value = ...  # type: Optional[str]
    
    def __init__(self, label_value: Optional[str] = None):
        self._label_value = label_value

    def __eq__(self, other: 'Label') -> bool:
        return self._label_value == other._label_value

    def __repr__(self):
        return 'Label(%r)' % self._label_value

    def set_label(self, label_value: str):
        if self._label_value is not None:
            raise ValueError('Label already has value: %s' % self._label_value)
        self._label_value = label_value


class Paragraph(object):
    short_line = ...  # type: int
    lines = ...  # type: Iterable[str]
    next_line = ...  # type: Optional[str]
    next_label = ...  # type: Optional[Label]
    _labels = ...  # type: List[Label]

    def __init__(self, short_line=40, labels=None):
        self.short_line = short_line
        self.lines = []
        self.next_line = None
        if labels:
            self._labels = labels[:]
        else:
            self._labels = []

    def append(self, line: str, label: Optional[Label] = None) -> None:
       self.lines.append(line)
       if label is not None:
           self._labels.append(label)

    def append_ahead(self, line: str, label: Optional[Label] = None) -> None:
        if self.next_line is not None:
            self.append(self.next_line, self.next_label)
            self.next_label = None
        self.next_line = line
        self.next_label = label

    def next_paragraph(self, labels: List[Label]) -> ('Paragraph', 'Paragraph'):
        pp = Paragraph(labels=labels)
        pp.append(self.next_line, self.next_label)
        return (self, pp)

    def __str__(self) -> str:
        return '\n'.join(self.lines) + '\n'

    def __repr__(self) -> str:
        return 'Labels(%s), Paragraph(%r)\n' % (self._labels, str(self))

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

    def is_figure(self) -> bool:
        return self.startswith([
            'fig', 'fig.', 'figs', 'figure', 'figures', 'plate', 'plates',
        ])

    def is_table(self) -> bool:
        return self.startswith([
            'table', 'tbl.', 'tbl'
        ])

    def is_blank(self) -> bool:
        return self.lines and all(not line for line in self.lines)

    @property
    def last_line(self) -> str:
        if not self.lines:
            return None
        return self.lines[-1]

    def close(self) -> None:
        if self.next_line:
            self.append(self.next_line, self.next_label)
            self.next_line = None
            self.next_label = None

    def endswith(self, s: str) -> bool:
        last_line = self.last_line
        return last_line and last_line.endswith(s)

    @property
    def labels(self) -> List[str]:
        return self._labels[:]

    def add_label(self, label: Optional[Label] = None) -> Label:
        if label is None:
            label = Label()
        self._labels.append(label)
        return label


def strip_label_start(line: str) -> (bool, str):
    if line.startswith('[@'):
        return (True, line[2:])
    else:
        return (False, line)


def strip_label_end(line: str) -> (Optional[str], str):
    match = re.search(r'(?P<line>.*)\#(?P<label_value>.*)\*\]$', line)
    if not match:
        return (None, line)
    return (match.group('label_value'), match.group('line'))


def parse_paragraphs(contents: Iterable[str]) -> Iterable[Paragraph]:
    label_stack = []
    label = None
    pp = Paragraph(labels=label_stack)
    for line in contents:
        (starts, line) = strip_label_start(line)
        if starts:
            label = Label()
            label_stack.append(label)
        else:
            label = None

        (label_value, line) = strip_label_end(line)
        if label_value:
            label_stack.pop().set_label(label_value)

        pp.append_ahead(line, label)

        # Tables continue to grow as long as we have short lines.
        if pp.is_table():
            if len(line) < pp.short_line:
                continue
            (retval, pp) = pp.next_paragraph(labels=label_stack)
            yield retval
            continue

        # Blocks of blank lines are a paragraph.
        if pp.is_blank():
            if not line:
                continue
            retval = pp
            (retval, pp) = pp.next_paragraph(labels=label_stack)
            yield retval
            continue

        # Figures end with a blank line, or a period at the end of a
        # line.
        if pp.is_figure():
            if line and not pp.endswith('.'):
               continue
            (retval, pp) = pp.next_paragraph(labels=label_stack)
            yield retval
            continue

        # A period before a newline marks the end of a paragraph.
        if pp.endswith('.'):
            (retval, pp) = pp.next_paragraph(labels=label_stack)
            yield retval
            continue

        # A blank line ends a paragraph.
        if line == '':
            (retval, pp) = pp.next_paragraph(labels=label_stack)
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
