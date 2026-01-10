"""Represent a paragraph.

This is the primary classified unit.
"""
import pandas  # type: ignore
import regex as re  # type: ignore
from typing import Dict, List, Optional, Tuple, Union

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


    _PREFIX_RE = (
        r'\b('
        r'[cC]on'
        r')'
    )

    _SUFFIX_RE = (
        r'('
        r'ae|al|am|an|ar|ax|ba|be|bi|ca|ch|ci|ck|da|di|ea|ed|ei|en|er|es|ev|gi'
        r'|ha|he|ia|ic|id|ii|is|it|ix|íz|la|le|li|ll|ma|me|na|nd|ni|ns|o|oa|oé'
        r'|of|oi|on|or|os|ox|pa|ph|ps|ra|re|ri|rt|sa|se|si|ta|te|ti|ts|ty'
        r'|ua|ud|um|up|us|va|vá|xa|ya|yi|ys|yx|za|zi'
        r')\b'
    )
    _NOMENCLATURE_RE = (
        r'^([-\w≡=.*|:]*\s+)?'  # Optional first word.
        r'(([A-Z]\w*' + _SUFFIX_RE + r')|(' + _PREFIX_RE + r'\w*))' # Genus
        r'\s((\w+' + _SUFFIX_RE + r')|(' + _PREFIX_RE + r'\w*))?' # species
        r'.*'
        r'('
        r'nov\.|nov\.\s?(comb\.|sp\.)|[(]?in\.?\s?ed\.[)]?|'
        r'[(]?nom\.\s?(prov\.|sanct\.)[)]?|emend\..*|'  # Indications of changes.
        r'[[(]?\b[12]\d{3}\b(\s+.[12]\d{3}\b.)?[])]?'  # Publication year
        r')'
        r'[^\n]*$'  # Any trailing content
    )
    _PUNCTUATION_RE = r'[)(.;:,≡=&×]'
    _PUNCTUATION = {
        '&': 'PAMPERSAND',
        '(': 'PLPAREN',
        ')': 'PRPAREN',
        ',': 'PCOMMA',
        '.': 'PDOT',
        ':': 'PCOLON',
        ';': 'PSEMI',
        '=': 'PEQUAL',
        '×': 'PTIMES',
        '≡': 'PEQUIV',
    }
    _YEAR_RE = r'\b[12]\d\d\d\b'
    _ABBREV_RE = r'\b[[:alpha:]]{1,5}\.'

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

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            'filename': self.filename,
            'human_url': self.human_url,
            'pdf_url': self.pdf_url,
            'label': str(self.top_label()),
            'paragraph_number': self.paragraph_number,
            'page_number': self.page_number,
            'empirical_page_number': str(self.empirical_page_number),
            'body': str(self)
        }

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
        def t(m, kind: str) -> bool:
            return kind in self._reinterpret and m.group(kind)

        def replace(m) -> str:
            if 'latinate' in self._reinterpret:
                if m.group('latinate'):
                    if 'suffix' in self._reinterpret:
                        return ' PLATINATE ' + m.group('suffix') + ' '
                    return ' PLATINATE '
                else:
                    if 'suffix' in self._reinterpret:
                        return ' ' + m.group('suffix') + ' '
            # Handle the case with just the suffix.
            if t(m, 'suffix'):
                return ' ' + m.group('suffix') + ' '
            if t(m, 'punctuation'):
                return ' ' + self._PUNCTUATION[m.group('punctuation')] + ' '
            if t(m, 'year'):
                return ' PYEAR '
            if t(m, 'abbrev'):
                return ' PABBREV '
            return m.captures()[0]

        def append_pat(r: str, kind: str, pattern: str) -> str:
            if kind in self._reinterpret:
                if r:
                    return r + '|' + pattern
                else:
                    return pattern
            else:
                return r

        r = ''
        r = append_pat(r, 'latinate', r'(?P<latinate>\w+(?P<suffix>' + self._SUFFIX_RE + '))')
        r = append_pat(r, 'suffix', r'(?P<latinate>\w+(?P<suffix>' + self._SUFFIX_RE + '))')
        r = append_pat(r, 'punctuation', r'(?P<punctuation>' + self._PUNCTUATION_RE + ')')
        r = append_pat(r, 'year', r'(?P<year>' + self._YEAR_RE + ')')
        r = append_pat(r, 'abbrev', r'(?P<abbrev>' + self._ABBREV_RE + ')')

        retval = ''
        if 'nomenclature' in self._reinterpret and self.contains_nomenclature():
            retval += ' PNOMENCLATURE '
        return retval + re.sub(r, replace, str(self), flags=re.MULTILINE | re.DOTALL)

    def append(self, line: Line) -> None:
        if line.contains_start():
            self.push_label()
        if line.end_label():
            if self.top_label() is None:
                return
                # raise ValueError('label close without open: %r' % line)
            try:
                self.top_label().set_label(line.end_label())
            except ValueError as e:
                raise ValueError('%s: %r' % (e, line))
        self._lines.append(line)

    def prepend(self, line: Line) -> None:
        self._lines.insert(0, line)

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

    def is_empty(self) -> bool:
        return not self._lines

    def is_page_header(self):
        return self.startswith('')

    @property
    def next_line(self) -> Line:
        return self._next_line

    @property
    def first_line(self) -> Optional[Line]:
        if not self._lines:
            return None
        return self._lines[0]

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
        if self.last_line is None:
            return None
        return self.last_line.filename

    @property
    def page_number(self) -> int:
        if self.last_line is None:
            return 0
        return self.last_line.page_number

    @property
    def pdf_page(self) -> int:
        if self.last_line is None:
            return 0
        return self.last_line.pdf_page

    @property
    def empirical_page_number(self) -> Optional[str]:
        if self.last_line is None:
            return None
        return self.last_line.empirical_page_number

    @property
    def human_url(self) -> Optional[str]:
        if self.last_line is None:
            return None
        return self.last_line.human_url

    @property
    def pdf_url(self) -> Optional[str]:
        if self.last_line is None:
            return None
        return self.last_line.pdf_url

    def close(self) -> None:
        if self._next_line:
            self.append(self._next_line)
            self._next_line = None

    def endswith(self, s: str) -> bool:
        last_line = self.last_line
        return bool(last_line) and last_line.endswith(s)

    # Have we accumulated a nomenclature?
    def contains_nomenclature(self) -> bool:
        return bool(re.search(
            self._NOMENCLATURE_RE,
            str(self),
            flags=re.MULTILINE | re.DOTALL
        ))

    def split_at_nomenclature(self) -> Optional['Paragraph']:
        """Pull off a trailing nomenclature."""
        if not self.contains_nomenclature():
            return None
        pp = Paragraph(labels=self._labels,
                       paragraph_number=self.paragraph_number + 1)
        if self._next_line:
            pp.append_ahead(self._next_line)
            self._next_line = None
        while not pp.contains_nomenclature():
            l = self._lines.pop()
            pp.prepend(l)

        return pp

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
