"""
BhlIngestor: whole volumes of one Biodiversity Heritage Library title,
or every volume holding one creator's (author's) work.

Metadata comes from BHL's API v3 (an API key is required, read from
``BHL_API_KEY``).  BHL's own download endpoints (partpdf, itempdf,
itemtext) sit behind a browser challenge, so the page scans come from
each volume's Internet Archive copy (``SourceIdentifier``).

One CouchDB document per volume (``bhl_item_id``), holding the IA PDF.
BHL's article list for the volume is stored on that document as
``bhl_parts``: each article's metadata, its licence, and its PDF pages
as runs (``pdf_page_ranges``, 1-based, matching the ``--- PDF Page N
---`` markers in article.txt), derived from BHL's ordered page list.
Articles need not be contiguous: plates are often bound at the back.
A treatment's ``pdf_page`` therefore names its article.

Licences are checked against ``data/approved_licenses.json``
(docs/approved-licenses.md) before anything is downloaded.  A page is
kept only if every article touching it is approved; pages outside any
article follow the volume's licence.  In ``article.pdf``, which is all
the pipeline and the web site see, withheld pages are blank pages of the
same size, so page numbers still line up.  The complete download of such
a mixed-rights volume is kept outside CouchDB, at its mirror path under
``SKOL_WWW_DIR`` (web_mirror; default /data/skol/www).
"""

import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from uuid import NAMESPACE_URL, uuid5

from bin.env_config import get_env_config

from .ingestor import Ingestor
from .licenses import LicensePolicy
from .timestamps import set_timestamps
from .web_mirror import save_mirror_copy

License = Tuple[Optional[str], Optional[str], bool]


class BhlIngestor(Ingestor):
    """Ingestor for every volume of one BHL bibliography (title), or of
    one creator: the volumes of the creator's titles and the volume that
    holds each of the creator's parts (articles)."""

    API_URL = 'https://www.biodiversitylibrary.org/api3'
    IA_DOWNLOAD = 'https://archive.org/download'

    bibliography_id: Optional[str]
    creator_id: Optional[str]
    title_ids: Optional[Set[str]]
    api_key: str
    license_policy: LicensePolicy
    limit: Optional[int]
    dry_run: bool
    www_dir: Path

    def __init__(
        self,
        bibliography_id: Optional[str] = None,
        creator_id: Optional[str] = None,
        title_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        license_policy: Optional[LicensePolicy] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
        www_dir: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Args:
            bibliography_id: BHL TitleID (the number in
                ``/bibliography/<id>``).
            creator_id: BHL AuthorID (the number in ``/creator/<id>``).
                Exactly one of bibliography_id and creator_id is given.
            title_ids: Creator mode only: ingest only these of the
                creator's titles (the creator's parts are kept whatever
                title holds them).  None keeps every title; [] keeps
                only parts.
            api_key: BHL API key; default ``BHL_API_KEY`` from the
                environment or ~/.skol_env (env_config).  Raises
                ValueError if neither is set.
            license_policy: Approved-licence policy; default the shipped
                ``approved_licenses.json``.
            limit: Stop after this many volumes are stored.
            dry_run: Decide and report; download and write nothing.
            www_dir: Root for complete copies of mixed-rights volumes;
                default ``SKOL_WWW_DIR`` via env_config, else
                /data/skol/www.
            **kwargs: Base class arguments (db, user_agent, robot_parser...)
        """
        super().__init__(**kwargs)
        if (bibliography_id is None) == (creator_id is None):
            raise ValueError(
                'BhlIngestor needs exactly one of bibliography_id and '
                'creator_id.')
        self.bibliography_id = (str(bibliography_id)
                                if bibliography_id is not None else None)
        self.creator_id = str(creator_id) if creator_id is not None else None
        if title_ids is not None and self.creator_id is None:
            raise ValueError('title_ids applies to creator_id only.')
        self.title_ids = ({str(t) for t in title_ids}
                          if title_ids is not None else None)
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
        self.www_dir = Path(www_dir if www_dir is not None
                            else get_env_config()['www_dir'])

    def ingest(self) -> None:
        """Store every volume of the title (or creator) that has an
        approved page."""
        if self.creator_id is not None:
            self._ingest_creator()
            return
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

    def _ingest_creator(self) -> None:
        """
        Store each volume holding the creator's work, once: every volume
        of each of the creator's titles, then the volume holding each of
        the creator's parts, with that volume's own title as its title.
        Parts with no BHL item are skipped.  ``limit`` counts stored
        volumes.
        """
        stored = 0
        for title, item_id in self._creator_volumes():
            if self.limit is not None and stored >= self.limit:
                if self.verbosity >= 2:
                    print(f"Reached limit of {self.limit}")
                break
            if self._ingest_item(title, item_id):
                stored += 1

    def _creator_volumes(self) -> Iterator[Tuple[Dict[str, Any], str]]:
        """Yield (title, item_id) for each volume holding the creator's
        work, each volume once, lazily so ``limit`` saves API calls."""
        authors = self.api('GetAuthorMetadata', id=self.creator_id,
                           pubs='t')
        if not authors:
            if self.verbosity >= 1:
                print(f"BHL creator {self.creator_id}: no metadata")
            return
        publications = authors[0].get('Publications') or []
        titles: Dict[str, Dict[str, Any]] = {}
        seen: Set[str] = set()

        def title_for(title_id: str) -> Optional[Dict[str, Any]]:
            if title_id not in titles:
                found = self.api('GetTitleMetadata', id=title_id)
                if not found:
                    return None
                titles[title_id] = found[0]
            return titles[title_id]

        for pub in publications:
            if pub.get('BHLType') != 'Title':
                continue
            title_id = str(pub.get('TitleID'))
            if self.title_ids is not None and title_id not in self.title_ids:
                if self.verbosity >= 2:
                    print(f"BHL title {title_id}: skipped, not in "
                          f"title_ids ({pub.get('Title')!r})")
                continue
            found = self.api('GetTitleMetadata', id=title_id, items='t')
            if not found:
                continue
            titles[title_id] = found[0]
            for summary in found[0].get('Items') or []:
                item_id = str(summary['ItemID'])
                if item_id not in seen:
                    seen.add(item_id)
                    yield found[0], item_id

        for pub in publications:
            if pub.get('BHLType') != 'Part':
                continue
            part_id = str(pub.get('PartID'))
            meta = self.api('GetPartMetadata', id=part_id)
            raw_item = meta[0].get('ItemID') if meta else None
            if not raw_item:
                if self.verbosity >= 2:
                    print(f"BHL part {part_id}: skipped, no BHL item "
                          f"({pub.get('ContainerTitle')!r})")
                continue
            item_id = str(raw_item)
            if item_id in seen:
                continue
            seen.add(item_id)
            items = self.api('GetItemMetadata', id=item_id)
            if not items or not items[0].get('TitleID'):
                if self.verbosity >= 1:
                    print(f"BHL item {item_id}: skipped, no title")
                continue
            title = title_for(str(items[0]['TitleID']))
            if title is not None:
                yield title, item_id

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
                parts.append(self.part_record(
                    meta[0], pages, item_lic,
                    item_year=self.volume_year(item)))

        unplaced = [p['bhl_part_id'] for p in parts
                    if not p['approved'] and p['pdf_page_ranges'] is None]
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
        doc.update(self.ia_metadata(str(item['SourceIdentifier']).strip()))
        expected_md5 = doc.get('ia_pdf_md5')
        if expected_md5 and hashlib.md5(pdf_bytes).hexdigest() != expected_md5:
            return skip('download does not match IA\'s md5 (corrupt or '
                        'truncated); a rerun retries it')
        if self.pdf_page_count(pdf_bytes) != len(pages):
            if not all(keep):
                return skip('PDF pages do not match BHL pages; cannot '
                            'withhold pages safely')
            doc['page_alignment'] = 'mismatch'
            for part in parts:
                part['pdf_page_ranges'] = None
        elif not all(keep):
            try:
                doc['complete_volume'] = self._save_complete_volume(
                    pdf_url, pdf_bytes)
            except (OSError, ValueError) as e:
                return skip(f"could not keep the complete copy "
                            f"({type(e).__name__}: {e})")
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

    IA_METADATA = 'https://archive.org/metadata'

    def ia_metadata(self, identifier: str) -> Dict[str, Any]:
        """
        Scan provenance from IA's metadata API, for telling copies apart
        (Layered Extraction's duplicate detection, Experiment 6 §8.9).

        Returns any of ``ia_contributor``, ``ia_scanning_center``,
        ``ia_scan_date`` (ISO when IA gives 14 digits), ``ia_image_count``
        and ``ia_pdf_md5`` (of ``<identifier>.pdf``, the file
        downloaded); ``{}`` if the lookup fails.
        """
        url = f"{self.IA_METADATA}/{identifier}"
        try:
            response = self._get_with_rate_limit(url)
        except Exception as e:
            if self.verbosity >= 1:
                print(f"  IA metadata failed for {identifier} "
                      f"({type(e).__name__})")
            return {}
        if response.status_code != 200:
            if self.verbosity >= 1:
                print(f"  IA metadata failed for {identifier} "
                      f"(HTTP {response.status_code})")
            return {}
        try:
            reply = response.json()
        except ValueError:
            return {}
        return parse_ia_metadata(reply, identifier)

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

    @staticmethod
    def volume_year(item: Dict[str, Any]) -> Optional[int]:
        """A volume's latest year: EndYear if present, else Year."""
        for key in ('EndYear', 'Year'):
            found = re.search(r'\d{4}', str(item.get(key) or ''))
            if found:
                return int(found.group(0))
        return None

    @staticmethod
    def part_year(part: Dict[str, Any]) -> Optional[int]:
        """The first four-digit year in a part's Date, if any."""
        found = re.search(r'\d{4}', str(part.get('Date') or ''))
        return int(found.group(0)) if found else None

    def item_license(
        self, item: Dict[str, Any]
    ) -> License:
        """(license_id, raw, approved) from LicenseUrl, then
        CopyrightStatus."""
        return self.license_policy.evaluate(
            [item.get('LicenseUrl'), item.get('CopyrightStatus')],
            year=self.volume_year(item))

    def part_record(
        self,
        part: Dict[str, Any],
        pages: Dict[int, int],
        item_license: License,
        item_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        One ``bhl_parts`` entry from GetPartMetadata.

        The licence comes from the part's LicenseUrl, then RightsStatus,
        falling back to the item's when both are blank.  PDF pages are
        ``pdf_page_ranges`` is None when a part page is not in ``pages``.
        """
        raws = [part.get('LicenseUrl'), part.get('RightsStatus')]
        if any(r is not None and r.strip() for r in raws):
            dated = self.part_year(part)
            license_id, raw, approved = self.license_policy.evaluate(
                raws, year=dated if dated is not None else item_year)
        else:
            license_id, raw, approved = item_license
        indices = [pages.get(int(p['PageID']))
                   for p in part.get('Pages') or []]
        ranges: Optional[List[List[int]]] = None
        if indices and all(i is not None for i in indices):
            ranges = []
            for i in sorted({i for i in indices if i is not None}):
                if ranges and ranges[-1][1] == i - 1:
                    ranges[-1][1] = i
                else:
                    ranges.append([i, i])
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
            'pdf_page_ranges': ranges,
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
            for start, end in part.get('pdf_page_ranges') or []:
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
        return save_mirror_copy(pdf_url, pdf_bytes, self.www_dir)

    def _store_volume(self, doc: Dict[str, Any], pdf_bytes: bytes) -> None:
        """Save ``doc`` (new) and attach ``pdf_bytes`` as article.pdf."""
        doc['_id'] = uuid5(NAMESPACE_URL, doc['pdf_url']).hex
        set_timestamps(doc, is_new=True)
        self.db.save(doc)
        self.db.put_attachment(doc, BytesIO(pdf_bytes), 'article.pdf',
                               'application/pdf')


def parse_ia_metadata(reply: Any, identifier: str) -> Dict[str, Any]:
    """The ``ia_*`` provenance fields from an archive.org metadata API
    reply for ``identifier`` (see BhlIngestor.ia_metadata); ``{}`` for a
    reply that is not a JSON object."""
    if not isinstance(reply, dict):
        return {}
    meta = reply.get('metadata') or {}
    out: Dict[str, Any] = {}
    for field, key in (('ia_contributor', 'contributor'),
                       ('ia_scanning_center', 'scanningcenter')):
        if meta.get(key):
            out[field] = meta[key]
    scandate = str(meta.get('scandate') or '')
    if re.fullmatch(r'\d{14}', scandate):
        out['ia_scan_date'] = (
            f"{scandate[0:4]}-{scandate[4:6]}-{scandate[6:8]}T"
            f"{scandate[8:10]}:{scandate[10:12]}:{scandate[12:14]}")
    elif scandate:
        out['ia_scan_date'] = scandate
    if str(meta.get('imagecount') or '').isdigit():
        out['ia_image_count'] = int(meta['imagecount'])
    for entry in reply.get('files') or []:
        if entry.get('name') == f"{identifier}.pdf" and entry.get('md5'):
            out['ia_pdf_md5'] = entry['md5']
            break
    return out


def _env_api_key() -> Optional[str]:
    """``BHL_API_KEY`` via env_config (environment, then .skol_env), or
    None when unset or blank."""
    value = get_env_config().get('bhl_api_key') or ''
    return value.strip() or None
