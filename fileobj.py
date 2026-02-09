from abc import ABC, abstractmethod
from typing import Iterator, Optional
import regex as re  # type: ignore

import constants

class FileObject(ABC):
    """
    Abstract base class for file-like objects.

    Provides common functionality for reading files and tracking
    line numbers, page numbers, character offsets, and empirical page numbers.
    """

    _line_number: int
    _page_number: int
    _pdf_page: int
    _pdf_label: Optional[str]
    _empirical_page_number: Optional[str]
    _char_offset: int  # Cumulative character position in source text

    def _set_empirical_page(self, l: str, first: bool = False) -> None:
        """
        Extract empirical page number from line content.

        Looks for Roman numerals or digits at the start or end of the line.

        Args:
            l: Line string to extract page number from
            first: Unused parameter for compatibility
        """
        match = re.search(
            r'(^\s*(?P<leading>[mdclxvi\d]+\b))|((?P<trailing>\b[mdclxvi\d]+)\s*$)',
            l
        )
        if not match:
            self._empirical_page_number = None
        else:
            self._empirical_page_number = (
                match.group('leading') or match.group('trailing')
            )

    @abstractmethod
    def _get_content_iterator(self) -> Iterator[str]:
        """
        Get an iterator over the content lines.

        Subclasses must implement this to provide their specific
        content source (file, list, etc.).

        Returns:
            Iterator yielding line strings
        """
        pass

    def read_line(self) -> Iterator['Line']:
        """
        Read lines from the content source.

        This template method handles common line processing logic:
        - Tracking line and page numbers
        - Tracking character offsets for span highlighting
        - Detecting page breaks (form feed characters)
        - Detecting PDF page markers (--- PDF Page N Label L ---)
        - Extracting empirical page numbers
        - Creating Line objects

        Returns:
            Iterator yielding Line objects
        """
        from line  import Line # Import here to avoid circular imports
        for l_str in self._get_content_iterator():
            # Capture character offset at start of this line
            line_start_char = self._char_offset
            # Calculate end_char (position after last char of line content)
            line_end_char = line_start_char + len(l_str)
            # Update cumulative offset: content length + 1 for newline
            # (newlines are stripped by split('\n') but exist in source file)
            self._char_offset = line_end_char + 1

            self._line_number += 1

            # First line of first page does not have a form feed
            if self._line_number == 1 and self._page_number == 1:
                self._set_empirical_page(l_str)

            # Check for page break (form feed character)
            if l_str.startswith('\f'):
                self._page_number += 1
                self._line_number = 1
                # Strip the form feed
                self._set_empirical_page(l_str[1:])

            if pdf_page_match := re.match(
                constants.pdf_page_pattern, l_str.strip()
            ):
                self._pdf_page = int(pdf_page_match.group(1))
                self._pdf_label = pdf_page_match.group(3)  # May be None if no label
                # Create a special Line object for the page marker
                # This line will be preserved in output but not classified
                l = Line(l_str, self, is_page_marker=True,
                         start_char=line_start_char, end_char=line_end_char)
                yield l
                continue

            # Create Line object with file metadata and character offsets
            l = Line(l_str, self,
                     start_char=line_start_char, end_char=line_end_char)
            yield l

    @property
    def line_number(self) -> int:
        """Current line number."""
        return self._line_number

    @property
    def page_number(self) -> int:
        """Current page number."""
        return self._page_number

    @property
    def pdf_page(self) -> int:
        """PDF page number from PDF page markers (--- PDF Page N Label L ---), or 0 if not present."""
        return self._pdf_page

    @property
    def pdf_label(self) -> Optional[str]:
        """PDF page number from PDF page markers (--- PDF Page N Label L ---), or 0 if not present."""
        return self._pdf_label

    @property
    def empirical_page_number(self) -> Optional[str]:
        """Empirical page number extracted from document."""
        return self._empirical_page_number

    @property
    def char_offset(self) -> int:
        """Current cumulative character offset in source text."""
        return self._char_offset

    @property
    @abstractmethod
    def filename(self) -> Optional[str]:
        """Filename or identifier for this file object."""
        return None

    def human_url(self) -> Optional[str]:
        """URL or source location for this file object."""
        return None
