"""Handle individual text lines."""

import regex as re  # type: ignore
from typing import List, Optional, Union

from fileobj import FileObject


class Line(object):
    _value: Optional[str]
    _filename: Optional[str]
    _label_start: bool
    _label_end: Optional[str]
    _line_number: int
    _pdf_page: int
    _empirical_page_number: Optional[str]
    _file = None
    _is_page_marker: bool  # True if this line is a PDF page marker
    # CouchDB metadata (optional)
    _doc_id: Optional[str]
    _attachment_name: Optional[str]
    _db_name: Optional[str]
    _human_url: Optional[str]
    _pdf_url: Optional[str]

    _TABLE = [
        'table', 'tab.', 'tab', 'tbl.', 'tbl',
    ]

    def __init__(self, line: str, fileobj: Optional[FileObject] = None, is_page_marker: bool = False) -> None:
        self._value = line.strip(' \n')
        self._filename = None
        self._page_number = None
        self._pdf_page = 0  # Default to 0 if no PDF page markers present
        self._empirical_page_number = None
        self._line_number = 0
        self._label_start = False
        self._label_end = None
        self._is_page_marker = is_page_marker  # Mark if this is a PDF page marker
        # Initialize optional CouchDB metadata
        self._doc_id = None
        self._attachment_name = None
        self._db_name = None
        self._human_url = None
        self._pdf_url = None

        if fileobj:
            self._filename = fileobj.filename
            self._line_number = fileobj.line_number
            self._page_number = fileobj.page_number
            self._pdf_page = fileobj.pdf_page
            self._empirical_page_number = fileobj.empirical_page_number

            # Check if fileobj has CouchDB metadata (duck typing)
            if hasattr(fileobj, 'doc_id'):
                self._doc_id = fileobj.doc_id
            if hasattr(fileobj, 'attachment_name'):
                self._attachment_name = fileobj.attachment_name
            if hasattr(fileobj, 'db_name'):
                self._db_name = fileobj.db_name
            if hasattr(fileobj, 'human_url'):
                self._human_url = fileobj.human_url
            if hasattr(fileobj, 'pdf_url'):
                self._pdf_url = fileobj.pdf_url

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
    def pdf_page(self) -> int:
        return self._pdf_page

    @property
    def empirical_page_number(self) -> Optional[str]:
        return self._empirical_page_number

    @property
    def line(self) -> str:
        return self._value

    @property
    def is_page_marker(self) -> bool:
        """True if this line is a PDF page marker (--- PDF Page N ---)."""
        return self._is_page_marker

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

    @property
    def doc_id(self) -> Optional[str]:
        """CouchDB document ID (optional)."""
        return self._doc_id

    @property
    def attachment_name(self) -> Optional[str]:
        """CouchDB attachment filename (optional)."""
        return self._attachment_name

    @property
    def db_name(self) -> Optional[str]:
        """Database name - ingest_db_name (optional)."""
        return self._db_name

    @property
    def human_url(self) -> Optional[str]:
        """URL from the source (optional)."""
        return self._human_url

    @property
    def pdf_url(self) -> Optional[str]:
        """PDF URL from the source (optional)."""
        return self._pdf_url
