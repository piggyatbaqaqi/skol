from abc import ABC, abstractmethod
from typing import Iterator, Optional
import regex as re  # type: ignore


class FileObject(ABC):
    """
    Abstract base class for file-like objects.

    Provides common functionality for reading files and tracking
    line numbers, page numbers, and empirical page numbers.
    """

    _line_number: int
    _page_number: int
    _empirical_page_number: Optional[str]

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
        - Detecting page breaks (form feed characters)
        - Extracting empirical page numbers
        - Creating Line objects

        Returns:
            Iterator yielding Line objects
        """
        from line  import Line # Import here to avoid circular imports
        for l_str in self._get_content_iterator():
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

            # TODO(piggy): Implement pdf_page_number extraction logic here if needed

            # Create Line object with file metadata
            l = Line(l_str, self)
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
    def empirical_page_number(self) -> Optional[str]:
        """Empirical page number extracted from document."""
        return self._empirical_page_number

    @property
    @abstractmethod
    def filename(self) -> Optional[str]:
        """Filename or identifier for this file object."""
        return None

    def human_url(self) -> Optional[str]:
        """URL or source location for this file object."""
        return self.filename
