"""Find species descriptions."""

import argparse
import re
import sys
from typing import Iterable, List, Optional

class Line(object):
    _value = ...  # type: str
    _label_start = ...  # type: bool
    _label_end = ...  # type: Optional[str]

    def __init__(self, line: str):
        self._value = line.strip(' \n')
        self.strip_label_start()
        self.strip_label_end()

    def __repr__(self) -> str:
        return 'start: %s end: %s value: %r' % (
            self._label_start, self._label_end, self._value)

    def line(self):
        return self._value

    def strip_label_start(self) -> None:
        if self.startswith('[@'):
            self._label_start = True
            self._value = self._value[2:]
            print("DEBUG: strip_label_start", self._value)
        else:
            self._label_start = False
            print("DEBUG: no label to start")

    def strip_label_end(self) -> None:
        match = re.search(r'(?P<line>.*)\#(?P<label_value>.*)\*\]$', self._value)
        if not match:
            self._label_end = None
            print("DEBUG: no label end: %r" % self)
            return
        print("DEBUG: strip_label_end, match.groups:", match.groups())
        (self._value, self._label_end) = match.groups()

    def startswith(self, *args, **kwargs) -> bool:
        return self._value.startswith(*args, **kwargs)

    def endswith(self, *args, **kwargs) -> bool:
        return self._value.endswith(*args, **kwargs)

    def is_short(self, short_line: int) -> bool:
        return len(self._value) < short_line

    def is_blank(self) -> bool:
        return self._value == ''

    def contains_start(self) -> bool:
        return self._label_start

    def end_label(self) -> Optional[str]:
        return self._label_end


class Label(object):
    _value = ...  # type: Optional[str]

    def __init__(self, value: Optional[str] = None):
        self._value = value

    def __eq__(self, other: 'Label') -> bool:
        return self._value == other._value

    def __repr__(self):
        return 'Label(%r)' % self._value

    def set_label(self, label_value: str):
        if self.assigned():
            raise ValueError('Label already has value: %s' % self._value)
        self._value = label_value

    def assigned(self):
        return self._value is not None

class Paragraph(object):
    short_line = ...  # type: int
    _lines = ...  # type: Iterable[Line]
    _next_line = ...  # type: Optional[Line]
    _labels = ...  # type: List[Label]

    def __init__(self, short_line=60, labels: Optional[List[Label]] = None):
        self.short_line = short_line
        self.lines = []
        self._next_line = None
        if labels:
            self._labels = labels[:]
        else:
            self._labels = []

    def append(self, line: str) -> None:
        print("DEBUG: append(line: %r)" % line)
        if line.contains_start():
            print("DEBUG: pushing label")
            self.push_label()
        if line.end_label():
            print("DEBUG: labeling top label: ", line.end_label())
            self.top_label().set_label(line.end_label())
        self.lines.append(line)

    def top_label(self) -> Optional[Label]:
        if self._labels:
            return self._labels[-1]
        else:
            return None

    def append_ahead(self, line: Line) -> None:
        print("DEBUG: append_ahead(line: %r)" % line)
        print("DEBUG: self._next_line: %r" % self._next_line)
        if self._next_line is not None:
            self.append(self._next_line)
        self._next_line = line

    def push_label(self) -> None:
        self._labels.append(Label())

    def pop_label(self) -> Label:
        retval = self._labels.pop()
        print("DEBUG: popping label: ", retval)
        return retval

    def next_line(self) -> Line:
        return self._next_line

    def next_paragraph(self) -> ('Paragraph', 'Paragraph'):
        print("DEBUG: next_paragraph")
        pp = Paragraph(labels=self._labels)
        # Remove labels which ended with the previous line.
        while pp.top_label() and pp.top_label().assigned():
            pp.pop_label()
        pp.append_ahead(self._next_line)
        self._next_line = None
        print("DEBUG:     current pp: %r" % self)
        print("DEBUG:     next pp: %r" % pp)
        return (self, pp)

    def __str__(self) -> str:
        return '\n'.join([l.line() for l in self.lines]) + '\n'

    def __repr__(self) -> str:
        return 'Labels(%s), Paragraph(%r), Pending(%r)\n' % (self._labels, str(self), self._next_line)

    def startswith(self, tokens: List[str]) -> bool:
        if not self.lines:
            return False
        tokenized = self.lines[0].line().strip().split()
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
        if self.lines:
            return all(line.is_blank() for line in self.lines)
        return False  # Empty paragraph is not blank yet.

    @property
    def last_line(self) -> str:
        if not self.lines:
            return None
        return self.lines[-1]

    def close(self) -> None:
        if self._next_line:
            self.append(self._next_line)
            self.next_line = None

    def endswith(self, s: str) -> bool:
        last_line = self.last_line
        return last_line and last_line.endswith(s)

    @property
    def labels(self) -> List[str]:
        return self._labels[:]


def parse_paragraphs(contents: Iterable[str]) -> Iterable[Paragraph]:
    label = None
    pp = Paragraph()
    for line_value in contents:
        line = Line(line_value)
        print("DEBUG: line: %r" % line)
        print("DEBUG: line.is_blank()=%r" % line.is_blank())
        # Strip page headers.
        if line.startswith(''):
            print("DEBUG: skipping page header: %r" % line)
            continue

        pp.append_ahead(line)

        print("DEBUG: pp: %r" % pp)

        # Tables continue to grow as long as we have short lines.
        if pp.is_table():
            print("DEBUG: table")
            if line.is_short(pp.short_line):
                continue
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Blocks of blank lines are a paragraph.
        if pp.is_blank():
            print("DEBUG: blanks")
            if line.is_blank():
                continue
            retval = pp
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Figures end with a blank line, or a period at the end of a
        # line.
        if pp.is_figure():
            print("DEBUG: figure")
            if not line.is_blank() and not pp.endswith('.'):
               continue
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # A period before a newline marks the end of a paragraph.
        if pp.endswith('.'):
            print("DEBUG: period")
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # A short line ends a paragraph.
        if pp.last_line and pp.last_line.is_short(pp.short_line):
            print("DEBUG: short line")
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # A blank line ends a paragraph.
        if line.is_blank():
            print("DEBUG: ending blank")
            (retval, pp) = pp.next_paragraph()
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
        if (paragraph.is_figure() or
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
