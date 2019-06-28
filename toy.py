import regex as re  # type: ignore
import textwrap
import paragraph

class Toy(object):
    def __init__(self, reinterpret):
        self._reinterpret = reinterpret

    foo = 9
    r = (
        # r'(?P<latinate>\w+(?P<suffix>' + paragraph.Paragraph._SUFFIX_RE + '))'
        r'(?P<punctuation>' + paragraph.Paragraph._PUNCTUATION_RE + ')'
        #r'|(?P<year>' + paragraph.Paragraph._YEAR_RE + ')'
        #r'|(?P<abbrev>' + paragraph.Paragraph._ABBREV_RE + ')'
    )
    data = textwrap.dedent("""\
    Julella sublactea (Nylander) R.C. Harris in Egan, Bryologist 90: 163. 1987;
    Verrucaria sublactea Nylander, Flora 69: 464. 1886. syn. nov.
    """)

    def get_replace(self):
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
            if 'suffix' in self._reinterpret:
                if m.group('suffix'):
                    return ' ' + m.group('suffix') + ' '
            if 'punctuation' in self._reinterpret:
                if m.group('punctuation'):
                    return ' ' + paragraph.Paragraph._PUNCTUATION[m.group('punctuation')] + ' '
            if 'year' in self._reinterpret:
                if m.group('year'):
                    return ' PYEAR '
            if 'abbrev' in self._reinterpret:
                if m.group('abbrev'):
                    return ' PABBREV '
            return m.captures()[0]
        return replace
