"""Find species descriptions."""

import argparse
import numpy  # type: ignore
import re
import sys
import time
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

from sklearn.naive_bayes import BernoulliNB  # type: ignore
from sklearn.dummy import DummyClassifier  # type: ignore
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier, LogisticRegression  # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score  # type: ignore
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.svm import *  # type: ignore
import pandas  # type: ignore

SEED=12345

class File(object):
    _filename: str
    _page_number: int
    _line_number: int
    _empirical_page_number: Optional[str]

    def __init__(
            self,
            filename: Optional[str] = None,
            contents: Optional[List[str]] = None) -> None:
        self._filename = filename
        self._page_number = 1
        self._line_number = 0
        if filename:
            self._file = open(filename, 'r')
            self._contents = None
        else:
            self._contents = contents
            self._file = None
        self._empirical_line_number = None

    def _set_empirical_page(self, l: str, first: bool = False) -> None:
        match = re.search(r'(^(?P<leading>[mdclxvi\d]+\b))|(?P<trailing>\b[mdclxvi\d]+$)', l)
        if not match:
            self._empirical_page_number = None
        else:
            self._empirical_page_number = (
                match.group('leading') or match.group('trailing')
            )

    def contents(self):
        return self._file or self._contents

    def read_line(self) -> Iterator['Line']:
        for l_str in self.contents():
            self._line_number += 1
            # First line of first page does not have a form feed.
            if self._line_number == 1 and self._page_number == 1:
                self._set_empirical_page(l_str)
            if l_str.startswith(''):
                self._page_number += 1
                self._line_number = 1
                # Strip the form feed.
                self._set_empirical_page(l_str[1:])
            l = Line(l_str, self)
            yield l

    @property
    def line_number(self) -> int:
        return self._line_number

    @property
    def page_number(self) -> int:
        return self._page_number

    @property
    def empirical_page_number(self) -> Optional[str]:
        return self._empirical_page_number

    @property
    def filename(self):
        return self._filename

class Line(object):
    _value: Optional[str]
    _filename: Optional[str]
    _label_start: bool
    _label_end: Optional[str]
    _line_number: int
    _empirical_page_number: Optional[str]
    _file = None

    _TABLE = [
        'table', 'tab.', 'tab', 'tbl.', 'tbl',
    ]

    def __init__(self, line: str, fileobj: Optional[File] = None) -> None:
        self._value = line.strip(' \n')
        self._filename = None
        self._page_number = None
        self._empirical_page_number = None
        self._line_number = 0
        self._label_start = False
        self._label_end = None
        if fileobj:
            self._filename = fileobj.filename
            self._line_number = fileobj.line_number
            self._page_number = fileobj.page_number
            self._empirical_page_number = fileobj.empirical_page_number
        self.strip_label_start()
        self.strip_label_end()

    def __repr__(self) -> str:
        return '%s:%d: start: %s end: %s value: %r' % (
            self._filename, self._line_number, self._label_start, self._label_end, self._value)

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def line_number(self) -> int:
        return self._line_number

    @property
    def page_number(self) -> int:
        return self._page_number

    @property
    def empirical_page_number(self) -> Optional[str]:
        return self._empirical_page_number

    @property
    def line(self) -> str:
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

    def startswith(self, tokens: Union[str, List[str]]) -> bool:
        if not self._value:
            return False
        if isinstance(tokens, str):
            return self._value.startswith(tokens)
        tokenized = self._value.strip().split()
        if not tokenized:
            return False
        first_token = tokenized[0].lower()
        return first_token in tokens

    def endswith(self, *args, **kwargs) -> bool:
        return self._value.endswith(*args, **kwargs)

    def search(self, *args, **kwargs):
        return re.search(*args, **kwargs, string=self._value)

    def is_short(self, short_line: int) -> bool:
        return len(self._value) < short_line

    def is_blank(self) -> bool:
        return self._value == ''

    def is_table(self) -> bool:
        return self.startswith(self._TABLE)

    def contains_start(self) -> bool:
        return self._label_start

    def end_label(self) -> Optional[str]:
        return self._label_end


class Label(object):
    _value: Optional[str]

    def __init__(self, value: Optional[str] = None) -> None:
        self._value = value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Label):
            return self._value == other._value
        else:
            return False

    def __repr__(self) -> str:
        return 'Label(%r)' % self._value

    def __str__(self) -> str:
        return self._value

    def set_label(self, label_value: str) -> None:
        if self.assigned():
            raise ValueError('Label already has value: %s' % self._value)
        self._value = label_value

    def assigned(self) -> bool:
        return self._value is not None


class Paragraph(object):
    short_line: int
    _lines: List[Line]
    _next_line: Optional[Line]
    _labels: List[Label]
    _reinterpret: List[str]

    # These are abbreviations which should not end a taxon paragraph.

    # Note that 'illeg.', 'illegit.', 'ined.', 'inval.', 'nov.',
    # 'nud.', 'press.' (in press), 'sp.', 'str.' can end a taxon paragraph.
    _KNOWN_ABBREVS = [
        'monogr.', 'uredin.', 'fam.',
        'acad.', 'agric.', 'akad.', 'al.', 'alt.', 'am.', 'amer.', 'ann.',
        'apr.', 'arg.', 'arkiv.', 'atk.', 'auct.', 'aug.', 'ave.', 'beauv.',
        'beitr.', 'bihar.', 'biol.', 'bot.', 'br.', 'bull.', 'burds.', 'ca.',
        'can.', 'carol.', 'ce.', 'cf.', 'cfr.', 'cienc.', 'cit.', 'cm.',
        'co.', 'comb.', 'crittog.', 'cunn.', 'dec.', 'del.', 'dept.', 'det.',
        'diam.', 'dis.', 'disc.', 'doc.', 'dr.', 'econ.', 'ed.', 'elev.',
        'entomol.', 'etc.', 'eur.', 'europ.', 'exot.', 'exp.', 'far.', 'feb.',
        'fenn.', 'fi.', 'fig.', 'figs.', 'fl.', 'fn.', 'fr.', 'fung.',
        'gard.', 'ges.', 'hb.', 'hedw.', 'henn.', 'herb.', 'hist.', 'hiver.',
        'holme.', 'hym.', 'ibid.', 'ica.', 'ind.', 'ined.', 'inst.', 'ist.',
        'ital.', 'jan.', 'jard.', 'jul.', 'jum.', 'jun.', 'kl.', 'kll.',
        'kon.', 'kérb.', 'lat.', 'later.', 'leafl.', 'leg.', 'lett.', 'li.',
        'linn.', 'loc.', 'lt.', 'magn.', 'mar.', 'mass.', 'mat.',
        'math.-naturwiss.', 'medit.', 'mi.', 'mich.', 'micol.', 'mikol.',
        'mr.', 'ms.', 'mt.', 'mu.', 'mucor.', 'mus.', 'mycol.', 'nat.',
        'naturk.', 'ned', 'ned.', 'neotrop.', 'no.', 'nom.', 'nsw.', 'nyl.',
        'oct.', 'pap.', 'par.', 'pers.', 'pg.', 'pl.', 'pls.', 'pp.', 'proc.',
        'prof.', 'prov.', 'publ.', 'rept.', 'repub.', 'res.', 'rim.', 'roxb.',
        'rupr.', 'sac.', 'schw.', 'sci.', 'sep.', 'ser.', 'sist.', 'snp.',
        'soc.', 'sp.', 'spor.', 'spp.', 'st.', 'sta.', 'stat.', 'surv.',
        'syn.', 'syst.', 'sér.', 'tax.', 'taxa.', 'tr.', 'trab.', 'tracts.',
        'trans.', 'univ.', 'var.', 'vary.', 'veg.', 'ven.', 'ver.', 'vic.',
        'wiss.', 'yum.', 'zool.',
    ]


    _SUFFIX_RE = (
        r'(ae|ana|ata|ca|cota|cys|derma|ea|ella|ense|es|forma|ia|ii'
        r'|ista|is|ix|i|oda|ola|oma|phora|sis|spora|tina|ula|um|us|zoa)\b'
    )
    _PUNCTUATION_RE = r'[().;:,≡=&]'
    _PUNCTUATION = {
        '(': 'PLPAREN',
        ')': 'PRPAREN',
        '.': 'PDOT',
        ';': 'PSEMI',
        ':': 'PCOLON',
        ',': 'PCOMMA',
        '≡': 'PEQUIV',
        '=': 'PEQUAL',
        '&': 'PAMPERSAND',
    }
    _YEAR_RE = r'\b[12]\d\d\d\b'
    _ABBREV_RE = '\b\w{1,5}\.'

    def __init__(self, short_line=45, labels: Optional[List[Label]] = None,
                 lines: Optional[List[Line]] = None) -> None:
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
        return '\n'.join([l.line for l in self._lines]) + '\n'

    def as_annotated(self) -> str:
        label = self.top_label()
        retval = str(self)[:-1]
        if label is not None and not self.is_blank():
            retval = '[@' + retval + '#' + str(label) + '*]'
        return retval

    def __repr__(self) -> str:
        return 'Labels(%s), Paragraph(%r), Pending(%r)\n' % (self._labels, str(self), self._next_line)

    @classmethod
    def set_reinterpretations(self, reinterpret: List[str]) -> None:
        self._reinterpret = reinterpret

    def suffixes(self) -> str:
        return ' ' + ' '.join(re.findall(self._SUFFIX_RE, str(self)))

    def punctuation(self) -> str:
        return ' ' + ' '.join([
            self._PUNCTUATION[p]
            for p in re.findall(self._PUNCTUATION_RE, str(self))
        ])

    def years(self) -> str:
        return ' ' + ' '.join(['PYEAR' for y in re.findall(self._YEAR_RE, str(self))])

    def abbrevs(self) -> str:
        return ' ' + ' '.join(['PABBREV' for y in re.findall(self._ABBREV_RE, str(self))])

    def reinterpret(self) -> str:
        def replace(m) -> str:
            retval = []
            if m.group('suffix'):
                if 'latinate' in self._reinterpret:
                    retval.append('PLATINATE')
                if 'suffix' in self._reinterpret:
                    retval.append(m.group('suffix'))
            if m.group('punctuation') and 'punctuation' in self._reinterpret:
                retval.append(self._PUNCTUATION[m.group('punctuation')])
            if m.group('year') and 'year' in self._reinterpret:
                retval.append('PYEAR')
            if m.group('abbrev') and 'abbrev' in self._reinterpret:
                retval.append('PABBREV')
            return ' ' + ' '.join(retval)
        r = (
            r'(?P<suffix>' + self._SUFFIX_RE + ')'
            r'|(?P<punctuation>' + self._PUNCTUATION_RE + ')'
            r'|(?P<year>' + self._YEAR_RE + ')'
            r'|(?P<abbrev>' + self._ABBREV_RE + ')'
        )

        return re.sub(r, replace, str(self))

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

    def next_paragraph(self) -> Tuple['Paragraph', 'Paragraph']:
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

    def startswith(self, tokens: Union[str, List[str]]) -> bool:
        if not self._lines:
            return False
        if isinstance(tokens, str):
            return self._lines[0].line.lower().startswith(tokens)
        tokenized = self._lines[0].line.strip().split()
        if not tokenized:
            return False
        first_token = tokenized[0].lower()
        return first_token in tokens

    def is_figure(self) -> bool:
        return self.startswith([
            'fig', 'fig.', 'figg.', 'figs', 'figs.', 'figure', 'photo', 'plate', 'plates',
        ])

    def is_table(self) -> bool:
        if not self._lines:
            return False
        return self._lines[0].is_table()

    def is_key(self) -> bool:
        return self.startswith('key to')

    def is_mycobank(self) -> bool:
        return self.startswith('mycobank')

    def is_all_long(self) -> bool:
        return all(not l.is_short(self.short_line) for l in self._lines)

    def is_blank(self) -> bool:
        if self._lines:
            return all(line.is_blank() for line in self._lines)
        return False  # Empty paragraph is not blank yet.

    def is_page_header(self):
        return self.startswith('')

    @property
    def next_line(self) -> Line:
        return self._next_line

    @property
    def last_line(self) -> Optional[Line]:
        if not self._lines:
            return None
        return self._lines[-1]

    def close(self) -> None:
        if self._next_line:
            self.append(self._next_line)
            self._next_line = None

    def endswith(self, s: str) -> bool:
        last_line = self.last_line
        return bool(last_line) and last_line.endswith(s)

    def detect_period(self) -> bool:
        last_line = self.last_line
        if last_line is None:
            return False
        if not last_line.endswith('.'):
            return False
        # A single initial except "'s" or similar.
        # I really want \w without \d.
        match = re.search(r"[^']\b\w\.$", last_line.line)
        if match:
            return False
        # : xxx.
        match = re.search(r'\b: \d+\.$', last_line.line)
        if match:
            return False
        # p. xxx.
        match = re.search(r'\bp\. \d+\.$', last_line.line)
        if match:
            return False
        match = re.search(r'\b(?P<abbrev>\w+\.)$', last_line.line)
        if match:
            if match.group('abbrev').lower() in self._KNOWN_ABBREVS:
                return False

        return True

    @property
    def labels(self) -> List[Label]:
        return self._labels[:]


def paragraphs_to_dataframe(paragraphs: List[Paragraph], suppress_text=False):
    if suppress_text:
        v2 = [pp.reinterpret() for pp in paragraphs]
    else:
        v2 = [str(pp) + ' ' + pp.reinterpret() for pp in paragraphs]
    return pandas.DataFrame(data={
        'v1': [str(pp.top_label()) for pp in paragraphs],
        # 'v2': [str(pp) + pp.suffixes() + pp.punctuation() + pp.years() + pp.abbrevs()
        #        for pp in paragraphs]
        'v2': v2,
        })



def parse_annotated(contents: Iterable[Line]) -> Iterable[Paragraph]:
    """Return paragraphs in annotated block form.

    Do not apply heuristic methods to divide paragraphs."""
    pp = Paragraph()
    for line in contents:
        pp.append_ahead(line)

        if line.contains_start():
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        if pp.last_line and pp.last_line.end_label() is not None:
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue


def parse_paragraphs(contents: Iterable[Line]) -> Iterable[Paragraph]:
    pp = Paragraph()
    for line in contents:
        pp.append_ahead(line)

        # New document triggers a new paragraph.
        if pp.last_line and pp.last_line.filename != line.filename:
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Page break triggers a new paragraph.
        if line.startswith(''):
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Page break is a whole paragraph.
        if pp.is_page_header():
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Tables start with a few long lines and
        # continue to grow as long as we have short lines.
        if pp.is_table():
            if line.is_short(pp.short_line):
                continue
            else:
                if pp.is_all_long():
                    continue
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Blocks of blank lines are a paragraph.
        if pp.is_blank():
            if line.is_blank():
                continue
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Figures end with a blank line, or period or colon at the end
        # of a line.
        if pp.is_figure():
            if (not line.is_blank() and
                not pp.detect_period() and
                not pp.endswith(':')):
               continue
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Leading hyphen triggers a new paragraph.
        if line.startswith('-'):
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # A table starts a new paragraph.
        if pp.next_line.is_table():
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # Synonymy reference ends a taxon.
        if pp.last_line and pp.last_line.search(r'\([Ss]yn.*\)$'):
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        # A taxon ends in nov., nov. comb., nov. sp., ined.,
        # emend. (followed by emender), nom. sanct. or a year within 3
        # characters of the end followed by an optional figure
        # specifier.
        if pp.last_line and pp.last_line.search(
                r'(nov\.|nov\.\s?(comb\.|sp\.)|[(]?in\.?\s?ed\.[)]?|'
                r'[(]?nom\.\s?sanct\.[)]?|emend\..*|\b[12]\d{3}\b.{0,3})'
                r'[-\s—]*([[(]?(Fig|Plate)[^])]*[])]?)?$'):
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


def read_files(files: List[str]) -> Iterable[Line]:
    for f in files:
        file_object = File(f)
        for line in file_object.read_line():
            yield line


def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
      for vectorizer in vectorizers:
        string = ''
        string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

        numpy.random.seed(SEED)

        start = time.time()
        # train
        vectorize_text = vectorizer.fit_transform(train_data.v2)
        classifier.fit(vectorize_text, train_data.v1)

        # score
        vectorize_text = vectorizer.transform(test_data.v2)
        score = classifier.score(vectorize_text, test_data.v1)
        string += '. Has score: ' + str(score)
        end = time.time()
        string += ' elapsed time ' + str(end - start)
        print(string)


def perform_confusion_matrix(classifiers, vectorizers, train_data, test_data, emit_csv: bool):
    if emit_csv:
        print('classifier,vectorizer,time,label,precision,recall,f1,support')

    for classifier in classifiers:
      for vectorizer in vectorizers:
        numpy.random.seed(SEED)

        start = time.time()
        # train
        vectorize_text = vectorizer.fit_transform(train_data.v2)
        classifier.fit(vectorize_text, train_data.v1)

        # Build the confusion matrix.
        transformed_text = vectorizer.transform(test_data.v2)
        predicted_labels = classifier.predict(transformed_text)
        end = time.time()
        elapsed_time = end - start
        cm = confusion_matrix(test_data.v1, predicted_labels)
        if emit_csv:
            print(csv_report(test_data, elapsed_time, predicted_labels, cm,
                             classifier.__class__.__name__,
                             vectorizer.__class__.__name__)
            )
        else:
            print(human_report(test_data, elapsed_time, predicted_labels, cm,
                               classifier.__class__.__name__,
                               vectorizer.__class__.__name__))


def human_report(test_data: pandas.core.frame.DataFrame,
                 elapsed_time: float,
                 predicted_labels: numpy.ndarray,
                 cm: numpy.ndarray,
                 classifier_name: str,
                 vectorizer_name: str) -> str:
    string = ''
    string += classifier_name + ' with ' + vectorizer_name

    cr = classification_report(test_data.v1, predicted_labels)
    string += ' elapsed time ' + str(elapsed_time)
    string += '\n' + str(cr)
    string += '\nConfusion matrix\n' + str(cm)
    return string


def csv_report(test_data: pandas.core.frame.DataFrame,
               elapsed_time: float,
               predicted_labels: numpy.ndarray,
               cm: numpy.ndarray,
               classifier_name: str,
               vectorizer_name: str) -> str:
    result = []
    u = numpy.unique(test_data.v1,  return_counts=True)
    labels = u[0]
    support = u[1]
    precision = precision_score(test_data.v1, predicted_labels, average = None)
    recall = recall_score(test_data.v1, predicted_labels, average = None)
    f1 = f1_score(test_data.v1, predicted_labels, average = None)

    for i in range(len(labels)):
        result.append(
            '{classifier},{vectorizer},{time:f},'
            '{label},{precision},{recall},{f1},{support}'.format(
            classifier=classifier_name,
            vectorizer=vectorizer_name,
            time=elapsed_time,
            label=labels[i],
            precision=precision[i],
            recall=recall[i],
            f1=f1[i],
            support=support[i]
        ))
    return '\n'.join(result)


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='+', help='the file to search for descriptions')
    # Actions
    parser.add_argument(
        '--dump_phase',
        help='Dump the output of these phases and exit.',
        default=[], type=int, action='append')
    parser.add_argument(
        '--dump_files',
        help='Dump lists of files to process.',
        action='store_true')
    parser.add_argument(
        '--test_classifiers',
        help='Test a set of classifiers against the input files.',
        action='store_true')
    parser.add_argument(
        '--test_classifiers_by_label',
        help='Test a set of classifiers against the input files, reporting by label.',
        action='store_true')
    parser.add_argument(
        '--output_annotated',
        help='Output YEDDA-annotated file.',
        action='store_true')
    # Control options
    parser.add_argument(
        '--reinterpret',
        help='Append reinterpretations of various elements. Values={suffix, latinate, punctuation, year, abbrev}.',
        default=[], type=str, action='append')
    parser.add_argument(
        '--classifier',
        help='Which classifier should we use for actual runs?',
        type=str, default='CalibratedClassifierCV')
    parser.add_argument(
        '--vectorizer',
        help='Which vectorizer should we use for actual runs?',
        type=str, default='TfidfVectorizer')
    parser.add_argument(
        '--keep_interstitials',
        help='Keep figures, tables, and blanks.',
        action='store_true')
    parser.add_argument(
        '--fast',
        help='Skip slower vectorizers and classifiers.',
        action='store_true')
    parser.add_argument(
        '--suppress_text',
        help='Suppress raw text. Evaluate only reinterpreted text.',
        action='store_true')
    parser.add_argument(
        '--label',
        default=[],
        help='Labels to retain for training purposes.',
        type=str,
        action='append')
    parser.add_argument(
        '--output_label',
        default=[],
        help='Labels to output.',
        type=str,
        action='append')
    parser.add_argument(
        '--annotated_paragraphs',
        help='Use paragraph boundaries as annotated, not the heuristic boundaries.',
        action='store_true')
    parser.add_argument(
        '--csv',
        help='In test_classifiers_by_label, emit a csv.',
        action='store_true')

    return parser.parse_args()


def main():
    args = define_args()

    Paragraph.set_reinterpretations(args.reinterpret)

    if not args.label:
        labels = ['Nomenclature', 'Description']
    else:
        labels = args.label

    output_labels = [Label(l) for l in args.output_label]

    try:
        i = args.file.index('evaluate')
        training_files = args.file[:i]
        evaluate_files = args.file[i+1:]
    except ValueError:
        training_files = args.file
        evaluate_files = []


    if args.dump_files:
        print('\ntraining_files:', training_files)
        print('\nevaluate_files:', evaluate_files)

    classifiers = [
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
    ]
    vectorizers = [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ]

    fast_classifiers = [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        # BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        # RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        # KNeighborsClassifier()  # Actually not slow, but we run out of memory.
    ]
    fast_vectorizers = [
        CountVectorizer(),
        TfidfVectorizer(),
        # HashingVectorizer()
    ]

    if args.fast:
        classifiers = fast_classifiers
        vectorizers = fast_vectorizers
    try:
        i = [c.__class__.__name__ for c in classifiers].index(args.classifier)
    except ValueError:
        raise ValueError('Unknown classifier %s' % args.classifier)
    classifier = classifiers[i]

    try:
        i = [v.__class__.__name__ for v in vectorizers].index(args.vectorizer)
    except ValueError:
        raise ValueError('Unknown vectorizer %s' % args.vectorizer)
    vectorizer = vectorizers[i]

    contents = read_files(training_files)

    if args.annotated_paragraphs:
        phase1 = parse_annotated(contents)
    else:
        phase1 = parse_paragraphs(contents)

    if 1 in args.dump_phase:
        print('Phase 1')
        print('=======')
        phase1 = list(phase1)
        print(repr(phase1))
        if 1 == max(args.dump_phase):
            sys.exit(0)

    phase2 = remove_interstitials(phase1)
    phase1 = None  # Potentially recover memory.

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
        keep=[Label(l) for l in labels]
    )

    phase2 = None

    if 3 in args.dump_phase:
        print('Phase 3')
        print('=======')
        phase3 = list(phase3)
        print(repr(phase3))
        if 3 == max(args.dump_phase):
            sys.exit(0)

    phase3 = list(phase3)
    sample_size = len(phase3)

    numpy.random.seed(SEED)
    cutoff = int(sample_size * 0.70)
    permutation = numpy.random.permutation(phase3)
    phase3 = None
    learn = paragraphs_to_dataframe(permutation[:cutoff], args.suppress_text)
    test = paragraphs_to_dataframe(permutation[cutoff:], args.suppress_text)

    if args.test_classifiers:
        perform(
            classifiers,
            vectorizers,
            learn,
            test
        )
        sys.exit(0)

    if args.test_classifiers_by_label:
        perform_confusion_matrix(
            classifiers,
            vectorizers,
            learn,
            test,
            emit_csv=args.csv
        )
        sys.exit(0)

    phase4 = []
    if evaluate_files:
        # train
        vectorize_text = vectorizer.fit_transform(learn.v2)
        classifier.fit(vectorize_text, learn.v1)

        # predict
        if args.keep_interstitials:
            evaluated = parse_paragraphs(read_files(evaluate_files))
        else:
            evaluated = remove_interstitials(parse_paragraphs(read_files(evaluate_files)))
        for pp in evaluated:
            text = str(pp)
            vectorize_text = vectorizer.transform([text])
            predict = classifier.predict(vectorize_text)[0]
            phase4.append(pp.replace_labels(labels=[Label(predict)]))

        if args.output_annotated:
            if not output_labels:
                print('\n'.join([pp.as_annotated() for pp in phase4]))
            else:
                print('\n'.join([pp.as_annotated()
                                 for pp in phase4
                                 if pp.top_label() in output_labels]))

    if 4 in args.dump_phase:
        print('Phase 4')
        print('=======')
        print(repr(phase4))
        if 4 == max(args.dump_phase):
            sys.exit(0)



if __name__ == '__main__':
    main()
