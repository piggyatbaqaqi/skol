"""Find species descriptions."""

import argparse
import re
import sys
from typing import Iterable, List, Optional

from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import *
import pandas

class Line(object):
    _value = ...  # type: Optional[str]
    _filename = ...  # type: str
    _label_start = ...  # type: bool
    _label_end = ...  # type: Optional[str]
    _line_number = ...  # type: int
    _file = None
    _count = 0

    def __init__(self, line: str, filename: Optional[str] = None):
        if self.__class__._file != filename:
            self.__class__._file = filename
            self.__class__._count = 0
        self.__class__._count += 1
        self._value = line.strip(' \n')
        self._filename = filename
        self._line_number = self._count
        self.strip_label_start()
        self.strip_label_end()

    def __repr__(self) -> str:
        return '%s:%d: start: %s end: %s value: %r' % (
            self._filename, self._line_number, self._label_start, self._label_end, self._value)

    def line(self):
        return self._value

    def strip_label_start(self) -> None:
        if self.startswith('[@'):
            self._label_start = True
            self._value = self._value[2:]
        else:
            self._label_start = False
        if '[@' in self._value:
            raise ValueError('Label open not at start of line: %s' % self)

    def strip_label_end(self) -> None:
        match = re.search(r'(?P<line>.*)\#(?P<label_value>.*)\*\]$', self._value)
        if not match:
            self._label_end = None
        else:
            (self._value, self._label_end) = match.groups()
        match = re.search(r'(?P<line>.*)\#(?P<label_value>.*)\*\]', self._value)
        if match:
            raise ValueError('Label close not at end of line: %r' % self)

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

    def __eq__(self, other: Optional['Label']) -> bool:
        if other is None:
            return self is None
        return self._value == other._value

    def __repr__(self):
        return 'Label(%r)' % self._value

    def __str__(self) -> str:
        return self._value
    
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

    def __init__(self, short_line=60, labels: Optional[List[Label]] = None,
                 lines: Optional[List[Line]] = None):
        self.short_line = short_line
        if lines:
            self._lines = lines[:]
        else:
            self._lines = []
        self._next_line = None
        if labels:
            self._labels = labels[:]
        else:
            self._labels = []

    def __str__(self) -> str:
        return '\n'.join([l.line() for l in self._lines]) + '\n'

    def __repr__(self) -> str:
        return 'Labels(%s), Paragraph(%r), Pending(%r)\n' % (self._labels, str(self), self._next_line)

    def append(self, line: Line) -> None:
        if line.contains_start():
            self.push_label()
        if line.end_label():
            if self.top_label() is None:
                raise ValueError('label close without open: %r' % line)
            try:
                self.top_label().set_label(line.end_label())
            except ValueError as e:
                raise ValueError('%s: %r' % (e, line))
        self._lines.append(line)

    def append_ahead(self, line: Line) -> None:
        if self._next_line is not None:
            self.append(self._next_line)
        self._next_line = line

    def top_label(self) -> Optional[Label]:
        if self._labels:
            return self._labels[-1]
        else:
            return None

    def push_label(self) -> None:
        self._labels.append(Label())

    def pop_label(self) -> Label:
        retval = self._labels.pop()
        return retval

    def next_line(self) -> Line:
        return self._next_line

    def next_paragraph(self) -> ('Paragraph', 'Paragraph'):
        pp = Paragraph(labels=self._labels)
        # Remove labels which ended with the previous line.
        while pp.top_label() and pp.top_label().assigned():
            pp.pop_label()
        pp.append_ahead(self._next_line)
        self._next_line = None
        return (self, pp)

    def replace_labels(self, labels: List[Label]) -> 'Paragraph':
        pp = Paragraph(labels=labels, lines=self._lines)
        return pp

    def startswith(self, tokens: List[str]) -> bool:
        if not self._lines:
            return False
        tokenized = self._lines[0].line().strip().split()
        if not tokenized:
            return False
        first_token = tokenized[0].lower()
        return first_token in tokens

    def is_figure(self) -> bool:
        return self.startswith([
            'fig', 'fig.', 'figs', 'figs.', 'figure', 'photo', 'plate',
        ])

    def is_table(self) -> bool:
        return self.startswith([
            'table', 'tbl.', 'tbl'
        ])

    def is_blank(self) -> bool:
        if self._lines:
            return all(line.is_blank() for line in self._lines)
        return False  # Empty paragraph is not blank yet.

    @property
    def last_line(self) -> str:
        if not self._lines:
            return None
        return self._lines[-1]

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

def paragraphs_to_dataframe(paragraphs: List[Paragraph]):
    return pandas.DataFrame(data={
            'v1': [str(pp.top_label()) for pp in paragraphs],
            'v2': [str(pp) for pp in paragraphs]
        })

def parse_paragraphs(contents: Iterable[Line]) -> Iterable[Paragraph]:
    label = None
    pp = Paragraph()
    for line in contents:
        # Strip page headers.
        if line.startswith(''):
            continue

        pp.append_ahead(line)

        # Tables continue to grow as long as we have short lines.
        if pp.is_table():
            if line.is_short(pp.short_line):
                continue
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Blocks of blank lines are a paragraph.
        if pp.is_blank():
            if line.is_blank():
                continue
            retval = pp
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Figures end with a blank line, or a period at the end of a
        # line.
        if pp.is_figure():
            if not line.is_blank() and not pp.endswith('.'):
               continue
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # A period before a newline marks the end of a paragraph.
        if pp.endswith('.'):
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # A short line ends a paragraph.
        if pp.last_line and pp.last_line.is_short(pp.short_line):
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # A blank line ends a paragraph.
        if line.is_blank():
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

    pp.close()
    yield pp


def remove_interstitials(paragraphs: Iterable[Paragraph]) -> Iterable[Paragraph]:
    for pp in paragraphs:
        if (pp.is_figure() or
            pp.is_table() or
            pp.is_blank()):
            continue
        yield(pp)


def target_classes(paragraphs: Iterable[Paragraph],
                   default: Label,
                   keep: List[Label]) -> Iterable[Paragraph]:
    for pp in paragraphs:
        if pp.top_label() in keep:
            yield pp
            continue
        yield pp.replace_labels([default])


def read_files(files: List[str]) -> Iterable[str]:
    for f in files:
        for line in open(f, 'r'):
            yield Line(line, filename=f)


def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
      for vectorizer in vectorizers:
        string = ''
        string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

        # train
        vectorize_text = vectorizer.fit_transform(train_data.v2)
        classifier.fit(vectorize_text, train_data.v1)

        # score
        vectorize_text = vectorizer.transform(test_data.v2)
        score = classifier.score(vectorize_text, test_data.v1)
        string += '. Has score: ' + str(score)
        print(string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, nargs='+', help="the file to search for descriptions")
    parser.add_argument("--dump_phase", help="Dump the output of these phases and exit.", default=[], type=int, action="append")
    args = parser.parse_args()
    print(args.file)

    contents = read_files(args.file)

    phase1 = parse_paragraphs(contents)

    if 1 in args.dump_phase:
        print('Phase 1')
        print('=======')
        phase1 = list(phase1)
        print(repr(phase1))
        if 1 == max(args.dump_phase):
            sys.exit(0)

    phase2 = remove_interstitials(phase1)

    if 2 in args.dump_phase:
        print('Phase 2')
        print('=======')
        phase2 = list(phase2)
        print(repr(phase2))
        if 2 == max(args.dump_phase):
            sys.exit(0)

    # All labels need to be resolved for this phase. The easiest way
    # to assure this is to convert to list.
    phase3 = target_classes(
        list(phase2),
        default=Label('Misc-exposition'),
        keep=[Label('Taxonomy'), Label('Description')]
    )

    if 3 in args.dump_phase:
        print('Phase 3')
        print('=======')
        phase3 = list(phase3)
        print(repr(phase3))
        if 3 == max(args.dump_phase):
            sys.exit(0)

    phase3 = list(phase3)
    sample_size = len(phase3)

    cutoff = int(sample_size * 0.70)
    learn = paragraphs_to_dataframe(phase3[:cutoff])
    test = paragraphs_to_dataframe(phase3[cutoff:])

    perform(
        [
            BernoulliNB(),
            RandomForestClassifier(n_estimators=100, n_jobs=-1),
            AdaBoostClassifier(),
            BaggingClassifier(),
            ExtraTreesClassifier(),
            GradientBoostingClassifier(),
            DecisionTreeClassifier(),
            CalibratedClassifierCV(),
            DummyClassifier(),
            PassiveAggressiveClassifier(),
            RidgeClassifier(),
            RidgeClassifierCV(),
            SGDClassifier(),
            OneVsRestClassifier(SVC(kernel='linear')),
            OneVsRestClassifier(LogisticRegression()),
            KNeighborsClassifier()
        ],
        [
            CountVectorizer(),
            TfidfVectorizer(),
            HashingVectorizer()
        ],
        learn,
        test
    )


if __name__ == '__main__':
    main()
