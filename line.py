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
    _pdf_label: Optional[str]
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
        self._pdf_label = None
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
            self._pdf_label = fileobj.pdf_label
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

        had_label_start = self.strip_label_start()

        # Handle form feed that was hidden inside annotation start marker
        # e.g., "[@\fdolor sit..." -> after stripping "[@", we have "\fdolor sit..."
        # Only do this if we actually stripped a label start marker (to avoid double-counting
        # form feeds that were already handled by FileObject.read_line())
        if had_label_start and self._value and self._value.startswith('\f') and fileobj:
            fileobj._page_number += 1
            fileobj._line_number = 1
            self._value = self._value[1:]  # Strip the form feed
            self._page_number = fileobj.page_number
            self._line_number = fileobj.line_number
            # Extract empirical page number from content after form feed
            fileobj._set_empirical_page(self._value)
            self._empirical_page_number = fileobj.empirical_page_number

        had_label_end = self.strip_label_end()

        # After stripping annotation end marker, check if a page number is now visible
        # (e.g., "dolor sit  xii  #Header*]" -> "dolor sit  xii  " exposes trailing "xii")
        # Only do this when we actually stripped an annotation marker
        if had_label_end:
            self._check_empirical_page_after_strip(fileobj)

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
    def pdf_label(self) -> Optional[str]:
        return self._pdf_label

    @property
    def empirical_page_number(self) -> Optional[str]:
        return self._empirical_page_number

    @property
    def line(self) -> str:
        return self._value

    @property
    def is_page_marker(self) -> bool:
        """True if this line is a PDF page marker (--- PDF Page N Label L ---)."""
        return self._is_page_marker

    def strip_label_start(self) -> bool:
        """Strip label start marker from line. Returns True if a label was stripped."""
        if self.startswith('[@'):
            self._label_start = True
            self._value = self._value[2:]
            if '[@' in self._value:
                raise ValueError('Label open not at start of line: %s' % self)
            return True
        else:
            self._label_start = False
            if self._value and '[@' in self._value:
                raise ValueError('Label open not at start of line: %s' % self)
            return False

    def strip_label_end(self) -> bool:
        """Strip label end marker from line. Returns True if a label was stripped."""
        if not self._value:
            self._label_end = None
            return False
        match = re.search(r'(?P<line>.*)\#(?P<label_value>.*)\*\]$', self._value)
        if not match:
            self._label_end = None
            # Check if there's a label close marker in the middle (not at end) - that's an error
            if re.search(r'\#.*\*\]', self._value):
                raise ValueError('Label close not at end of line: %r' % self)
            return False
        else:
            (self._value, self._label_end) = match.groups()
        if self._value and re.search(r'(?P<line>.*)\#(?P<label_value>.*)\*\]', self._value):
            raise ValueError('Label close not at end of line: %r' % self)
        return True

    def _check_empirical_page_after_strip(self, fileobj: Optional[FileObject]) -> None:
        """
        Check for empirical page number after stripping annotation markers.

        Annotated blocks like "dolor sit  xii  #Header*]" have the page number
        hidden in the middle. After stripping "#Header*]", the "xii" becomes
        visible at the trailing position and should be detected.

        Updates both this Line's and the FileObject's empirical_page_number
        so subsequent lines inherit the new page number.
        """
        if not self._value or not fileobj:
            return

        # Use the FileObject's method to extract and set the empirical page number
        # This updates the FileObject so subsequent lines inherit the new page number
        fileobj._set_empirical_page(self._value)
        self._empirical_page_number = fileobj.empirical_page_number

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
