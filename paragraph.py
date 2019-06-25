"""Represent a paragraph.

This is the primary classified unit.
"""
import pandas  # type: ignore
import re
from typing import List, Optional, Tuple, Union

from label import Label
from line import Line

class Paragraph(object):
    short_line: int
    _lines: List[Line]
    _next_line: Optional[Line]
    _labels: List[Label]
    _reinterpret: List[str]
    _paragraph_number: int

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
                 lines: Optional[List[Line]] = None,
                 paragraph_number: int = 0) -> None:
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
        self._paragraph_number = paragraph_number

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
        pp = Paragraph(labels=self._labels,
                       paragraph_number=self.paragraph_number + 1)
        # Remove labels which ended with the previous line.
        while pp.top_label() and pp.top_label().assigned():
            pp.pop_label()
        pp.append_ahead(self._next_line)
        self._next_line = None
        return (self, pp)

    def replace_labels(self, labels: List[Label]) -> 'Paragraph':
        pp = Paragraph(labels=labels, lines=self._lines,
                       paragraph_number=self.paragraph_number)
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

    @property
    def paragraph_number(self) -> int:
        return self._paragraph_number

    @property
    def filename(self) -> Optional[str]:
        return self.last_line.filename

    @property
    def page_number(self) -> int:
        return self.last_line.page_number

    @property
    def empirical_page_number(self) -> Optional[str]:
        return self.last_line.empirical_page_number

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


def to_dataframe(paragraphs: List[Paragraph], suppress_text=False):
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
