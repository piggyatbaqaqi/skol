from typing import Iterator, List, Optional

from line import Line
from fileobj import FileObject


class File(FileObject):
    """File-like object for reading local files."""

    _filename: Optional[str]
    _file: Optional[Iterator[str]]
    _contents: Optional[List[str]]

    def __init__(
            self,
            filename: Optional[str] = None,
            contents: Optional[List[str]] = None) -> None:
        """
        Initialize File from filename or content list.

        Args:
            filename: Path to file to read (optional)
            contents: List of line strings (optional)
        """
        self._filename = filename
        self._line_number = 0
        self._page_number = 1
        self._empirical_page_number = None

        if filename:
            self._file = open(filename, 'r', encoding='utf-8')
            self._contents = None
        else:
            self._contents = contents
            self._file = None

    def _get_content_iterator(self) -> Iterator[str]:
        """Get iterator over file contents."""
        return self._file or iter(self._contents)

    @property
    def filename(self) -> Optional[str]:
        """Filename for this file object."""
        return self._filename

    @property
    def url(self) -> Optional[str]:
        """URL is the filename for local files."""
        return self._filename


def read_files(files: List[str]) -> Iterator[Line]:
    for f in files:
        file_object = File(f)
        for line in file_object.read_line():
            yield line
