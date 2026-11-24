"""
BhlIngestor: whole volumes of one Biodiversity Heritage Library title.

Metadata comes from BHL's API v3 (an API key is required, read from
``BHL_API_KEY``).  BHL's own download endpoints (partpdf, itempdf,
itemtext) sit behind a browser challenge, so the page scans come from
each volume's Internet Archive copy (``SourceIdentifier``).

One CouchDB document per volume (``bhl_item_id``), holding the IA PDF.
BHL's article list for the volume is stored on that document as
``bhl_parts``: each article's metadata, its licence, and its PDF page
range (1-based, matching the ``--- PDF Page N ---`` markers in
article.txt), derived from BHL's ordered page list.  A treatment's
``pdf_page`` therefore names its article.

Licences are checked against ``data/approved_licenses.json``
(docs/approved-licenses.md) before anything is downloaded.  A page is
kept only if every article touching it is approved; pages outside any
article follow the volume's licence.  Withheld pages are replaced by
blank pages of the same size, so page numbers still line up.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from .ingestor import Ingestor
from .licenses import LicensePolicy


class BhlIngestor(Ingestor):
    """Ingestor for every volume of one BHL bibliography (title)."""

    API_URL = 'https://www.biodiversitylibrary.org/api3'
    IA_DOWNLOAD = 'https://archive.org/download'

    bibliography_id: str
    api_key: str
    license_policy: LicensePolicy
    limit: Optional[int]
    dry_run: bool

    def __init__(
        self,
        bibliography_id: str,
        api_key: Optional[str] = None,
        license_policy: Optional[LicensePolicy] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Args:
            bibliography_id: BHL TitleID (the number in
                ``/bibliography/<id>``).
            api_key: BHL API key; default ``$BHL_API_KEY``.  Raises
                ValueError if neither is set.
            license_policy: Approved-licence policy; default the shipped
                ``approved_licenses.json``.
            limit: Stop after this many volumes are stored.
            dry_run: Decide and report; download and write nothing.
            **kwargs: Base class arguments (db, user_agent, robot_parser...)
        """
        super().__init__(**kwargs)
        self.bibliography_id = str(bibliography_id)
        key = api_key or _env_api_key()
        if not key:
            raise ValueError(
                'BHL API key missing: set BHL_API_KEY (e.g. in '
                '~/.skol_env) or pass api_key.')
        self.api_key = key
        self.license_policy = (license_policy if license_policy is not None
                               else LicensePolicy.default())
        self.limit = limit
        self.dry_run = dry_run

    def ingest(self) -> None:
        """Store every volume of the title that has an approved page."""
        raise NotImplementedError

    def api(self, op: str, **params: Any) -> Optional[Any]:
        """
        Call one API v3 operation; return its ``Result``.

        Returns None on a robots denial, a transport or HTTP error, or a
        ``Status`` other than ``ok``.  The API key is never printed.
        """
        raise NotImplementedError

    def ia_pdf_url(self, item: Dict[str, Any]) -> Optional[str]:
        """The Internet Archive PDF for an item, or None if the item was
        not scanned by IA."""
        raise NotImplementedError

    @staticmethod
    def page_index(item: Dict[str, Any]) -> Dict[int, int]:
        """Map each PageID of the item to its 1-based PDF page."""
        raise NotImplementedError

    def item_license(
        self, item: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """(license_id, raw, approved) from LicenseUrl, then
        CopyrightStatus."""
        raise NotImplementedError

    def part_record(
        self,
        part: Dict[str, Any],
        pages: Dict[int, int],
        item_license: Tuple[Optional[str], Optional[str], bool],
    ) -> Dict[str, Any]:
        """
        One ``bhl_parts`` entry from GetPartMetadata.

        The licence comes from the part's LicenseUrl, then RightsStatus,
        falling back to the item's when both are blank.  PDF pages are
        None when a part page is not in ``pages``.
        """
        raise NotImplementedError

    @staticmethod
    def kept_pages(
        n_pages: int,
        parts: List[Dict[str, Any]],
        item_approved: bool,
    ) -> List[bool]:
        """
        Per PDF page (index 0 is page 1): keep it?

        A page inside one or more parts is kept only if all of them are
        approved; a page in no part is kept if the item is approved.
        """
        raise NotImplementedError

    @staticmethod
    def withheld_ranges(keep: List[bool]) -> List[Tuple[int, int]]:
        """Inclusive 1-based (first, last) runs of withheld pages."""
        raise NotImplementedError

    @staticmethod
    def pdf_page_count(pdf_bytes: bytes) -> int:
        """Number of pages in a PDF."""
        raise NotImplementedError

    @staticmethod
    def redact_pdf(pdf_bytes: bytes, keep: List[bool]) -> bytes:
        """Replace every page not kept with a blank page of the same
        size, carrying no text."""
        raise NotImplementedError

    def volume_doc(
        self,
        title: Dict[str, Any],
        item: Dict[str, Any],
        parts: List[Dict[str, Any]],
        item_license: Tuple[Optional[str], Optional[str], bool],
        pdf_url: str,
    ) -> Dict[str, Any]:
        """The CouchDB document for one volume (no _id or timestamps)."""
        raise NotImplementedError

    def _store_volume(self, doc: Dict[str, Any], pdf_bytes: bytes) -> None:
        """Save ``doc`` (new) and attach ``pdf_bytes`` as article.pdf."""
        raise NotImplementedError


def _env_api_key() -> Optional[str]:
    """``$BHL_API_KEY``, or None when unset or blank."""
    value = os.environ.get('BHL_API_KEY', '').strip()
    return value or None
