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

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import NAMESPACE_URL, uuid5

from bin.env_config import get_env_config

from .ingestor import Ingestor
from .licenses import LicensePolicy
from .timestamps import set_timestamps

License = Tuple[Optional[str], Optional[str], bool]


class BhlIngestor(Ingestor):
    """Ingestor for every volume of one BHL bibliography (title)."""

    API_URL = 'https://www.biodiversitylibrary.org/api3'
    IA_DOWNLOAD = 'https://archive.org/download'

    bibliography_id: str
    api_key: str
    license_policy: LicensePolicy
    limit: Optional[int]
    dry_run: bool
    www_dir: Path

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
            api_key: BHL API key; default ``BHL_API_KEY`` from the
                environment or ~/.skol_env (env_config).  Raises
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
        titles = self.api('GetTitleMetadata', id=self.bibliography_id,
                          items='t')
        if not titles:
            if self.verbosity >= 1:
                print(f"BHL title {self.bibliography_id}: no metadata")
            return
        title = titles[0]
        stored = 0
        for summary in title.get('Items') or []:
            if self.limit is not None and stored >= self.limit:
                if self.verbosity >= 2:
                    print(f"Reached limit of {self.limit}")
                break
            if self._ingest_item(title, str(summary['ItemID'])):
                stored += 1

    def _ingest_item(self, title: Dict[str, Any], item_id: str) -> bool:
        """Ingest one volume; True if it was stored (or would be, in a
        dry run)."""
        def skip(reason: str) -> bool:
            if self.verbosity >= 2:
                print(f"BHL item {item_id}: skipped, {reason}")
            return False

        items = self.api('GetItemMetadata', id=item_id, pages='t',
                         parts='t')
        if not items:
            return skip('no item metadata')
        item = items[0]
        pdf_url = self.ia_pdf_url(item)
        if pdf_url is None:
            return skip(f"not an Internet Archive scan "
                        f"(Source {item.get('Source')!r})")
        if self._is_url_ingested(pdf_url):
            return skip(f"already stored ({pdf_url})")

        pages = self.page_index(item)
        item_lic = self.item_license(item)
        parts: List[Dict[str, Any]] = []
        for summary in item.get('Parts') or []:
            meta = self.api('GetPartMetadata', id=str(summary['PartID']),
                            pages='t')
            if meta:
                parts.append(self.part_record(meta[0], pages, item_lic))

        unplaced = [p['bhl_part_id'] for p in parts
                    if not p['approved'] and p['pdf_page_start'] is None]
        if unplaced:
            return skip(f"unapproved parts with no page range {unplaced}; "
                        f"cannot withhold them")
        keep = self.kept_pages(len(pages), parts, item_lic[2])
        if not any(keep):
            return skip(f"no approved page (volume licence "
                        f"{item_lic[0] or item_lic[1]!r})")
        if self.dry_run:
            if self.verbosity >= 2:
                withheld = self.withheld_ranges(keep)
                print(f"[dry run] {pdf_url} ({len(parts)} parts, licence "
                      f"{item_lic[0]}, withheld {withheld})")
            return True

        try:
            response = self._get_with_rate_limit(pdf_url)
        except Exception as e:
            return skip(f"download failed ({type(e).__name__})")
        if response.status_code != 200:
            return skip(f"download failed (HTTP {response.status_code})")
        pdf_bytes: bytes = response.content
        if not pdf_bytes.startswith(b'%PDF'):
            return skip('download is not a PDF')

        doc = self.volume_doc(title, item, parts, item_lic, pdf_url)
        if self.pdf_page_count(pdf_bytes) != len(pages):
            if not all(keep):
                return skip('PDF pages do not match BHL pages; cannot '
                            'withhold pages safely')
            doc['page_alignment'] = 'mismatch'
            for part in parts:
                part['pdf_page_start'] = None
                part['pdf_page_end'] = None
        elif not all(keep):
            pdf_bytes = self.redact_pdf(pdf_bytes, keep)
            doc['withheld_pages'] = [
                list(r) for r in self.withheld_ranges(keep)]
        if self.verbosity >= 2:
            print(f"Adding {pdf_url} ({len(parts)} parts)")
        self._store_volume(doc, pdf_bytes)
        return True

    def api(self, op: str, **params: Any) -> Optional[Any]:
        """
        Call one API v3 operation; return its ``Result``.

        Returns None on a robots denial, a transport or HTTP error, or a
        ``Status`` other than ``ok``.  The API key is never printed.
        """
        if not self.robot_parser.can_fetch(self.user_agent, self.API_URL):
            if self.verbosity >= 1:
                print(f"Robot permission denied {self.API_URL}")
            return None
        query: Dict[str, Any] = {'op': op, **params, 'format': 'json',
                                 'apikey': self.api_key}
        # Messages name the operation only: the URL, exception text and
        # the API's ErrorMessage can all carry the key.
        try:
            response = self.http_client.get(self.API_URL, params=query)
        except Exception as e:
            if self.verbosity >= 1:
                print(f"  BHL {op} failed ({type(e).__name__})")
            return None
        if response.status_code != 200:
            if self.verbosity >= 1:
                print(f"  BHL {op} failed (HTTP {response.status_code})")
            return None
        try:
            reply = response.json()
        except ValueError:
            if self.verbosity >= 1:
                print(f"  BHL {op} reply was not JSON")
            return None
        if not isinstance(reply, dict) or reply.get('Status') != 'ok':
            if self.verbosity >= 1:
                status = reply.get('Status') if isinstance(reply, dict) \
                    else None
                print(f"  BHL {op} returned status {status!r}")
            return None
        return reply.get('Result')

    def ia_pdf_url(self, item: Dict[str, Any]) -> Optional[str]:
        """The Internet Archive PDF for an item, or None if the item was
        not scanned by IA."""
        identifier = (item.get('SourceIdentifier') or '').strip()
        if item.get('Source') != 'Internet Archive' or not identifier:
            return None
        return f"{self.IA_DOWNLOAD}/{identifier}/{identifier}.pdf"

    @staticmethod
    def page_index(item: Dict[str, Any]) -> Dict[int, int]:
        """Map each PageID of the item to its 1-based PDF page."""
        return {int(page['PageID']): i + 1
                for i, page in enumerate(item.get('Pages') or [])}

    def item_license(
        self, item: Dict[str, Any]
    ) -> License:
        """(license_id, raw, approved) from LicenseUrl, then
        CopyrightStatus."""
        return self.license_policy.evaluate(
            [item.get('LicenseUrl'), item.get('CopyrightStatus')])

    def part_record(
        self,
        part: Dict[str, Any],
        pages: Dict[int, int],
        item_license: License,
    ) -> Dict[str, Any]:
        """
        One ``bhl_parts`` entry from GetPartMetadata.

        The licence comes from the part's LicenseUrl, then RightsStatus,
        falling back to the item's when both are blank.  PDF pages are
        None when a part page is not in ``pages``.
        """
        raws = [part.get('LicenseUrl'), part.get('RightsStatus')]
        if any(r is not None and r.strip() for r in raws):
            license_id, raw, approved = self.license_policy.evaluate(raws)
        else:
            license_id, raw, approved = item_license
        indices = [pages.get(int(p['PageID']))
                   for p in part.get('Pages') or []]
        if indices and all(i is not None for i in indices):
            placed = [i for i in indices if i is not None]
            start: Optional[int] = min(placed)
            end: Optional[int] = max(placed)
        else:
            start = end = None
        date = part.get('Date')
        year = re.search(r'\d{4}', date or '')
        record: Dict[str, Any] = {
            'bhl_part_id': str(part['PartID']),
            'title': part.get('Title'),
            'author': '; '.join(a['Name'] for a in part.get('Authors') or []
                                if a.get('Name')),
            'genre': part.get('Genre'),
            'date': date,
            'year': year.group(0) if year else None,
            'pages': (part.get('PageRange') or '').replace('--', '–')
            or None,
            'start_page_number': part.get('StartPageNumber'),
            'end_page_number': part.get('EndPageNumber'),
            'pdf_page_start': start,
            'pdf_page_end': end,
            'url': part.get('PartUrl'),
            'license_id': license_id,
            'license_raw': raw,
            'approved': approved,
        }
        for ident in part.get('Identifiers') or []:
            if (ident.get('IdentifierName') or '').upper() == 'DOI':
                record['doi'] = ident.get('IdentifierValue')
                break
        return record

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
        covering: List[List[bool]] = [[] for _ in range(n_pages)]
        for part in parts:
            start, end = part.get('pdf_page_start'), part.get('pdf_page_end')
            if start is None or end is None:
                continue
            for i in range(start - 1, end):
                if 0 <= i < n_pages:
                    covering[i].append(bool(part.get('approved')))
        return [all(c) if c else item_approved for c in covering]

    @staticmethod
    def withheld_ranges(keep: List[bool]) -> List[Tuple[int, int]]:
        """Inclusive 1-based (first, last) runs of withheld pages."""
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i, kept in enumerate(keep, 1):
            if not kept and start is None:
                start = i
            elif kept and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(keep)))
        return runs

    @staticmethod
    def pdf_page_count(pdf_bytes: bytes) -> int:
        """Number of pages in a PDF."""
        import pymupdf  # optional [pdf] dependency; needed only here
        with pymupdf.open(stream=pdf_bytes, filetype='pdf') as doc:
            count: int = doc.page_count
        return count

    @staticmethod
    def redact_pdf(pdf_bytes: bytes, keep: List[bool]) -> bytes:
        """Replace every page not kept with a blank page of the same
        size, carrying no text."""
        import pymupdf  # optional [pdf] dependency; needed only here
        with pymupdf.open(stream=pdf_bytes, filetype='pdf') as doc:
            if doc.page_count != len(keep):
                raise ValueError(
                    f"keep has {len(keep)} entries for {doc.page_count} pages")
            for i, kept in enumerate(keep):
                if kept:
                    continue
                rect = doc[i].rect
                doc.delete_page(i)
                doc.new_page(pno=i, width=rect.width, height=rect.height)
            # garbage=3 drops the deleted pages' now-unreferenced objects,
            # so their images and text do not survive in the file.
            out: bytes = doc.tobytes(garbage=3, deflate=True)
        return out

    def volume_doc(
        self,
        title: Dict[str, Any],
        item: Dict[str, Any],
        parts: List[Dict[str, Any]],
        item_license: License,
        pdf_url: str,
    ) -> Dict[str, Any]:
        """The CouchDB document for one volume (no _id or timestamps)."""
        volume = re.sub(r'^v\.?\s*', '', str(item.get('Volume') or ''),
                        flags=re.I) or None
        full_title = title.get('FullTitle') or title.get('ShortTitle')
        return {
            'title': f"{full_title}, v. {volume}" if volume else full_title,
            'author': '; '.join(a['Name'] for a in title.get('Authors') or []
                                if a.get('Name')),
            'journal': title.get('ShortTitle'),
            'volume': volume,
            'year': item.get('Year'),
            'publisher_place': title.get('PublisherPlace'),
            'doi': title.get('Doi'),
            'itemtype': 'volume',
            'language': item.get('Language'),
            'copy_note': item.get('CopySpecificInformation'),
            'holding_institution': item.get('HoldingInstitution'),
            'bhl_title_id': str(title.get('TitleID')),
            'bhl_item_id': str(item.get('ItemID')),
            'ia_identifier': item.get('SourceIdentifier'),
            'pdf_url': pdf_url,
            'url': item.get('ItemUrl'),
            'human_url': item.get('ItemUrl'),
            'source': 'bhl',
            'license_id': item_license[0],
            'license_raw': item_license[1],
            'bhl_parts': parts,
            'withheld_pages': [],
            'page_alignment': 'ok',
        }

    def _save_complete_volume(
        self, pdf_url: str, pdf_bytes: bytes
    ) -> Dict[str, Any]:
        """
        Keep the unredacted download under ``www_dir`` at its mirror
        path (web_mirror), outside CouchDB, where no SKOL web view can
        serve it.

        Returns:
            ``{'path', 'sha256', 'bytes'}`` for the volume document.
        """
        raise NotImplementedError

    def _store_volume(self, doc: Dict[str, Any], pdf_bytes: bytes) -> None:
        """Save ``doc`` (new) and attach ``pdf_bytes`` as article.pdf."""
        doc['_id'] = uuid5(NAMESPACE_URL, doc['pdf_url']).hex
        set_timestamps(doc, is_new=True)
        self.db.save(doc)
        self.db.put_attachment(doc, BytesIO(pdf_bytes), 'article.pdf',
                               'application/pdf')


def _env_api_key() -> Optional[str]:
    """``BHL_API_KEY`` via env_config (environment, then .skol_env), or
    None when unset or blank."""
    value = get_env_config().get('bhl_api_key') or ''
    return value.strip() or None
