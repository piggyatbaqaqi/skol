from abc import ABC, abstractmethod
from typing import Optional

class FileObject(ABC):

    @property
    @abstractmethod
    def line_number(self) -> int:
        return 0

    @property
    @abstractmethod
    def page_number(self) -> int:
        return 0

    @property
    @abstractmethod
    def empirical_page_number(self) -> Optional[str]:
        return None

    @property
    @abstractmethod
    def filename(self) -> Optional[str]:
        return None

    @property
    @abstractmethod
    def url(self) -> Optional[str]:
        return None
