import re
from typing import Iterator, List, Optional

from line import Line
from fileobj import FileObject

class File(FileObject):
    _filename: Optional[str]
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
