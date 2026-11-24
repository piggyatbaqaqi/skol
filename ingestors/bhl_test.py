"""
Unit tests for BhlIngestor.

Fixtures are trimmed from real API v3 replies fetched 2026-11-23 for
C. G. Lloyd, *Mycological writings* (TitleID 19766), volume 2 (ItemID
59611, IA mycologicalwriti02lloyiala): field names and values are the
API's, page lists cut to six pages and two parts.
"""

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock
from uuid import NAMESPACE_URL, uuid5

import pymupdf
import pytest

# skol_share is a top-level module of the checkout (as in
# particle_detector_test.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .bhl import BhlIngestor, parse_ia_metadata  # noqa: E402
from .licenses import LicensePolicy  # noqa: E402

KEY = 'test-key-0123456789'

MINI_POLICY: Dict[str, Any] = {
    'version': 1,
    'licenses': [
        {'id': 'public-domain', 'approved': True,
         'strings': ['NOT_IN_COPYRIGHT'], 'urls': []},
        {'id': 'unspecified', 'approved': False,
         'strings': ['', 'Not specified', 'Other'], 'urls': []},
    ],
    'creative_commons': {'by': True, 'by-nc-sa': True, 'by-nd': False},
}

TITLE: Dict[str, Any] = {
    'TitleID': 19766,
    'FullTitle': 'Mycological writings of C. G. Lloyd',
    'ShortTitle': 'Mycological writings of C. G. Lloyd',
    'PublisherPlace': 'Cincinnati, O',
    'PublicationDate': '[1898]-1929',
    'Doi': '10.5962/bhl.title.19766',
    'TitleUrl': 'https://www.biodiversitylibrary.org/bibliography/19766',
    'Authors': [{'Name': 'Lloyd, C. G. (Curtis Gates), 1859-1926.'}],
    'Items': [
        {'ItemID': 59611, 'Volume': '2', 'Year': '1898',
         'ItemUrl': 'https://www.biodiversitylibrary.org/item/59611'},
    ],
}

PAGE_IDS = [17880445, 17880446, 17880415, 17880416, 17880417, 17880418]

ITEM: Dict[str, Any] = {
    'ItemID': 59611,
    'TitleID': '19766',
    'Source': 'Internet Archive',
    'SourceIdentifier': 'mycologicalwriti02lloyiala',
    'Volume': '2',
    'Year': '1898',
    'CopySpecificInformation': 'Includes Mycological Notes, Nos. 19-31',
    'HoldingInstitution': 'University of California Libraries',
    'Language': 'English',
    'CopyrightStatus': 'NOT_IN_COPYRIGHT',
    'ItemUrl': 'https://www.biodiversitylibrary.org/item/59611',
    'Pages': [{'PageID': pid} for pid in PAGE_IDS],
    'Parts': [
        {'PartID': 328814, 'Title': '[Introductory Materials]'},
        {'PartID': 328815, 'Title': 'Mycological Notes, No. 19'},
    ],
}

PART_INTRO: Dict[str, Any] = {
    'PartUrl': 'https://www.biodiversitylibrary.org/part/328814',
    'PartID': 328814,
    'ItemID': '59611',
    'StartPageID': '17880445',
    'Genre': 'Article',
    'Title': '[Introductory Materials including Indices, Vol. 2]',
    'Volume': '2',
    'Date': '1898',
    'PageRange': '1--25',
    'StartPageNumber': '1',
    'EndPageNumber': '25',
    'RightsStatus': 'NOT_IN_COPYRIGHT',
    'Authors': [{'Name': 'Lloyd, Curtis Gates'}],
    'Identifiers': [],
    'Pages': [{'PageID': 17880445}, {'PageID': 17880446}],
}

PART_NOTES: Dict[str, Any] = {
    'PartUrl': 'https://www.biodiversitylibrary.org/part/328815',
    'PartID': 328815,
    'ItemID': '59611',
    'StartPageID': '17880415',
    'Genre': 'Article',
    'Title': 'Mycological Notes, No. 19',
    'Volume': '2',
    'Date': '1905-05',
    'PageRange': '205--220',
    'StartPageNumber': '205',
    'EndPageNumber': '220',
    'RightsStatus': 'NOT_IN_COPYRIGHT',
    'Authors': [{'Name': 'Lloyd, Curtis Gates'}],
    'Identifiers': [{'IdentifierName': 'DOI',
                     'IdentifierValue': '10.5962/p.328815'}],
    'Pages': [{'PageID': 17880415}, {'PageID': 17880416},
              {'PageID': 17880417}],
}

IA_PDF = ('https://archive.org/download/mycologicalwriti02lloyiala/'
          'mycologicalwriti02lloyiala.pdf')


def _policy() -> LicensePolicy:
    return LicensePolicy(MINI_POLICY)


def _make(**overrides: Any) -> BhlIngestor:
    robot_parser = overrides.pop('robot_parser', None)
    if robot_parser is None:
        robot_parser = MagicMock()
        robot_parser.can_fetch.return_value = True
    defaults: Dict[str, Any] = dict(
        db=overrides.pop('db', MagicMock()),
        user_agent='test-agent',
        robot_parser=robot_parser,
        verbosity=0,
        bibliography_id='19766',
        api_key=KEY,
        license_policy=overrides.pop('license_policy', MagicMock()),
    )
    defaults.update(overrides)
    return BhlIngestor(**defaults)


def _pdf(n: int, size: tuple = (300, 400)) -> bytes:
    """An n-page PDF whose page i carries the text 'page <i+1>'."""
    doc = pymupdf.open()
    for i in range(n):
        page = doc.new_page(width=size[0], height=size[1])
        page.insert_text((50, 50), f'page {i + 1}')
    data: bytes = doc.tobytes()
    return data


def _texts(pdf_bytes: bytes) -> List[str]:
    with pymupdf.open(stream=pdf_bytes, filetype='pdf') as doc:
        return [page.get_text().strip() for page in doc]


# ---------------------------------------------------------------------------


BY_DATE_POLICY: Dict[str, Any] = dict(MINI_POLICY, public_domain_by_date={
    'id': 'public-domain-by-date', 'years': 95,
    'overrides': ['unspecified', 'no-known-copyright', 'unrecognised']})

NOT_PROVIDED = ('Not provided. Contact Holding Institution to verify '
                'copyright status.')

# Trimmed from archive.org/metadata/researchesonfung01bulliala,
# 2026-11-23 (Buller, Researches on fungi v.1, UC Libraries' copy).
IA_META: Dict[str, Any] = {
    'metadata': {
        'identifier': 'researchesonfung01bulliala',
        'contributor': 'University of California Libraries',
        'scanningcenter': 'iala',
        'scandate': '20061122194442',
        'imagecount': '296',
        'sponsor': 'MSN',
        'volume': '1',
    },
    'files': [
        {'name': 'researchesonfung01bulliala_bw.pdf',
         'format': 'Grayscale PDF',
         'md5': '00e4f6e497bc629a4788d7bfa694a16a'},
        {'name': 'researchesonfung01bulliala.pdf', 'format': 'Text PDF',
         'md5': '3f598735597b34ce2d679f46b3150401'},
        {'name': 'researchesonfung01bulliala_djvu.txt',
         'md5': 'ffffffffffffffffffffffffffffffff'},
    ],
}

# Trimmed from GetAuthorMetadata(id=3484, pubs=t), 2026-11-23: C. H.
# Kauffman, 13 publications (3 titles, 10 parts in other titles).
AUTHOR: Dict[str, Any] = {
    'AuthorID': '3484',
    'Name': 'Kauffman, C. H.',
    'FullerForm': '(Calvin Henry),',
    'Dates': '1869-1931',
    'CreatorUrl': 'https://www.biodiversitylibrary.org/creator/3484',
    'Publications': [
        {'BHLType': 'Title', 'TitleID': '136929',
         'Title': 'The genus cortinarius', 'PublicationDate': '1905'},
        {'BHLType': 'Part', 'PartID': '246166',
         'Title': 'The genus Cortinarius: a preliminary study',
         'ContainerTitle': 'Bulletin of the Torrey Botanical Club',
         'Volume': '32', 'Issue': '6', 'Date': '1905',
         'PageRange': '301--325'},
        {'BHLType': 'Part', 'PartID': '246177',
         'Title': 'Tennessee and Kentucky fungi',
         'ContainerTitle': 'Mycologia', 'Date': '1917-05'},
    ],
}


class TestConstruction:
    def test_explicit_key(self) -> None:
        ing = _make()
        assert ing.api_key == KEY
        assert ing.bibliography_id == '19766'
        assert ing.limit is None and ing.dry_run is False

    @pytest.fixture
    def no_skol_env(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Hide the operator's ~/.skol_env and BHL_API_KEY (as in
        pmc_test.py); return the env_config module to stub."""
        import bin.env_config as env_config_module
        monkeypatch.setattr(env_config_module, '_skol_env_cache', {})
        monkeypatch.delenv('BHL_API_KEY', raising=False)
        monkeypatch.delenv('SKOL_WWW_DIR', raising=False)
        return env_config_module

    def test_key_from_environment(self, no_skol_env: Any,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('BHL_API_KEY', ' env-key ')
        assert _make(api_key=None).api_key == 'env-key'

    def test_key_from_skol_env(self, no_skol_env: Any) -> None:
        no_skol_env._skol_env_cache = {'BHL_API_KEY': 'file-key'}
        assert _make(api_key=None).api_key == 'file-key'

    def test_missing_key_is_an_error_naming_the_variable(
            self, no_skol_env: Any) -> None:
        with pytest.raises(ValueError, match='BHL_API_KEY'):
            _make(api_key=None)

    def test_www_dir_default(self, no_skol_env: Any) -> None:
        assert _make().www_dir == Path('/data/skol/www')

    def test_www_dir_from_environment(
            self, no_skol_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SKOL_WWW_DIR', '/srv/www')
        assert _make().www_dir == Path('/srv/www')

    def test_www_dir_argument_wins(
            self, no_skol_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('SKOL_WWW_DIR', '/srv/www')
        assert _make(www_dir='/x').www_dir == Path('/x')

    def test_bibliography_id_coerced_and_unknown_kwargs_ignored(
            self) -> None:
        # Restored: lost by an edit in e300640.
        ing = _make(bibliography_id=19766, mode='api', journal='x')
        assert ing.bibliography_id == '19766'

    def test_creator_id(self) -> None:
        ing = _make(bibliography_id=None, creator_id=3484)
        assert ing.creator_id == '3484'
        assert ing.bibliography_id is None

    def test_title_ids_coerced_to_a_set_of_strings(self) -> None:
        ing = _make(bibliography_id=None, creator_id='1881',
                    title_ids=[5397, '5397', 221079])
        assert ing.title_ids == {'5397', '221079'}

    def test_title_ids_default_none(self) -> None:
        assert _make(bibliography_id=None, creator_id='1881').title_ids \
            is None

    def test_title_ids_need_a_creator(self) -> None:
        with pytest.raises(ValueError, match='creator_id'):
            _make(title_ids=['5397'])

    @pytest.mark.parametrize('ids', [
        {'bibliography_id': None, 'creator_id': None},
        {'bibliography_id': '19766', 'creator_id': '3484'},
    ])
    def test_exactly_one_of_title_and_creator(
            self, ids: Dict[str, Any]) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            _make(**ids)


class TestApi:
    def _ing(self, reply: Any = None, status: int = 200,
             **kw: Any) -> BhlIngestor:
        ing = _make(**kw)
        client = MagicMock()
        response = MagicMock(status_code=status)
        response.json.return_value = reply
        client.get.return_value = response
        ing.http_client = client
        return ing

    def test_sends_op_params_key_and_json_format(self) -> None:
        ing = self._ing({'Status': 'ok', 'Result': [TITLE]})
        assert ing.api('GetTitleMetadata', id='19766', items='t') == [TITLE]
        args, kwargs = ing.http_client.get.call_args  # type: ignore
        assert args[0] == BhlIngestor.API_URL
        assert kwargs['params'] == {
            'op': 'GetTitleMetadata', 'id': '19766', 'items': 't',
            'format': 'json', 'apikey': KEY}

    def test_status_not_ok(self) -> None:
        ing = self._ing({'Status': 'unauthorized', 'ErrorMessage': 'bad',
                         'Result': None})
        assert ing.api('GetTitleMetadata', id='1') is None

    def test_http_error(self) -> None:
        assert self._ing(None, status=500).api('X') is None

    def test_transport_error(self) -> None:
        ing = self._ing()
        ing.http_client.get.side_effect = ConnectionError('boom')  # type: ignore
        assert ing.api('X') is None

    def test_robots_denial_makes_no_request(self) -> None:
        robot_parser = MagicMock()
        robot_parser.can_fetch.return_value = False
        ing = self._ing(robot_parser=robot_parser)
        assert ing.api('X') is None
        ing.http_client.get.assert_not_called()  # type: ignore

    def test_key_never_printed(self, capsys: pytest.CaptureFixture[str]
                               ) -> None:
        ing = self._ing({'Status': 'error', 'ErrorMessage': 'nope'},
                        verbosity=3)
        ing.api('GetItemMetadata', id='1')
        ing.http_client.get.side_effect = ConnectionError(  # type: ignore
            f'failed for {BhlIngestor.API_URL}?apikey={KEY}')
        ing.api('GetItemMetadata', id='1')
        out = capsys.readouterr()
        assert KEY not in out.out + out.err


class TestItemHelpers:
    def test_ia_pdf_url(self) -> None:
        assert _make().ia_pdf_url(ITEM) == IA_PDF

    def test_ia_pdf_url_none_for_other_sources(self) -> None:
        item = dict(ITEM, Source='Smithsonian Libraries')
        assert _make().ia_pdf_url(item) is None
        assert _make().ia_pdf_url(dict(ITEM, SourceIdentifier='')) is None

    def test_page_index_is_one_based_in_order(self) -> None:
        assert BhlIngestor.page_index(ITEM) == {
            pid: i + 1 for i, pid in enumerate(PAGE_IDS)}

    def test_item_license_prefers_license_url(self) -> None:
        ing = _make(license_policy=_policy())
        assert ing.item_license(ITEM) == (
            'public-domain', 'NOT_IN_COPYRIGHT', True)
        item = dict(ITEM, LicenseUrl=(
            'https://creativecommons.org/licenses/by-nd/4.0/'))
        assert ing.item_license(item)[0] == 'CC-BY-ND-4.0'
        assert ing.item_license(item)[2] is False

    def test_item_license_blank_is_unspecified(self) -> None:
        ing = _make(license_policy=_policy())
        item = dict(ITEM, CopyrightStatus='')
        assert ing.item_license(item) == ('unspecified', None, False)


class TestParseIaMetadata:
    """The module-level parser, shared with
    fixes/backfill_bhl_ia_provenance.py."""

    def test_same_fields_as_the_method(self) -> None:
        assert parse_ia_metadata(IA_META, 'researchesonfung01bulliala') == {
            'ia_contributor': 'University of California Libraries',
            'ia_scanning_center': 'iala',
            'ia_scan_date': '2006-11-22T19:44:42',
            'ia_image_count': 296,
            'ia_pdf_md5': '3f598735597b34ce2d679f46b3150401',
        }

    def test_non_object_reply(self) -> None:
        assert parse_ia_metadata(['x'], 'id') == {}


class TestIaMetadata:
    def _ing(self, reply: Any = IA_META, status: int = 200,
             error: Any = None) -> BhlIngestor:
        ing = _make()
        response = MagicMock(status_code=status)
        response.json.return_value = reply
        ing._get_with_rate_limit = MagicMock(  # type: ignore[method-assign]
            return_value=response, side_effect=error)
        return ing

    def test_fields(self) -> None:
        ing = self._ing()
        assert ing.ia_metadata('researchesonfung01bulliala') == {
            'ia_contributor': 'University of California Libraries',
            'ia_scanning_center': 'iala',
            'ia_scan_date': '2006-11-22T19:44:42',
            'ia_image_count': 296,
            'ia_pdf_md5': '3f598735597b34ce2d679f46b3150401',
        }
        url = ing._get_with_rate_limit.call_args.args[0]  # type: ignore
        assert url == ('https://archive.org/metadata/'
                       'researchesonfung01bulliala')

    def test_missing_and_odd_fields_are_left_out(self) -> None:
        reply = {'metadata': {'contributor': 'X', 'imagecount': 'many',
                              'scandate': '2006'},
                 'files': [{'name': 'other.pdf', 'md5': 'abc'}]}
        assert self._ing(reply).ia_metadata('researchesonfung01bulliala') \
            == {'ia_contributor': 'X', 'ia_scan_date': '2006'}

    def test_http_error_gives_nothing(self) -> None:
        assert self._ing(status=503).ia_metadata('x') == {}

    def test_transport_error_gives_nothing(self) -> None:
        assert self._ing(error=ConnectionError('boom')).ia_metadata('x') \
            == {}

    def test_bad_json_gives_nothing(self) -> None:
        ing = self._ing()
        ing._get_with_rate_limit.return_value.json.side_effect = (  # type: ignore
            ValueError('not json'))
        assert ing.ia_metadata('x') == {}
        assert self._ing(reply=['not', 'a', 'dict']).ia_metadata('x') == {}


class TestPublicDomainByDate:
    """McIlvaine's 1900 first edition (title 217265, item 347956) states
    'Not provided. Contact Holding Institution ...'."""

    def _ing(self) -> BhlIngestor:
        return _make(license_policy=LicensePolicy(BY_DATE_POLICY))

    def test_volume_year_prefers_end_year(self) -> None:
        assert BhlIngestor.volume_year({'Year': '1898',
                                        'EndYear': '1905'}) == 1905
        assert BhlIngestor.volume_year({'Year': '1900'}) == 1900
        assert BhlIngestor.volume_year({'Year': 'c1902'}) == 1902
        assert BhlIngestor.volume_year({}) is None

    def test_part_year(self) -> None:
        assert BhlIngestor.part_year({'Date': '1905-05'}) == 1905
        assert BhlIngestor.part_year({'Date': ''}) is None
        assert BhlIngestor.part_year({}) is None

    def test_old_volume_without_rights_is_public_domain_by_date(
            self) -> None:
        item = dict(ITEM, CopyrightStatus=NOT_PROVIDED, Year='1900')
        item.pop('EndYear', None)
        lic = self._ing().item_license(item)
        assert lic[0] == 'public-domain-by-date' and lic[2] is True

    def test_volume_ending_after_the_cutoff_is_not(self) -> None:
        item = dict(ITEM, CopyrightStatus=NOT_PROVIDED, Year='1909',
                    EndYear='1930')
        assert self._ing().item_license(item)[2] is True
        item['EndYear'] = '1950'
        assert self._ing().item_license(item)[2] is False

    def test_explicit_licence_on_an_old_volume_stands(self) -> None:
        item = dict(ITEM, Year='1900', LicenseUrl=(
            'https://creativecommons.org/licenses/by-nd/4.0/'))
        assert self._ing().item_license(item)[2] is False

    def test_part_uses_its_own_date(self) -> None:
        part = dict(PART_NOTES, RightsStatus=NOT_PROVIDED, Date='1905-05')
        rec = self._ing().part_record(
            part, BhlIngestor.page_index(ITEM),
            ('unspecified', None, False), item_year=1950)
        assert (rec['license_id'], rec['approved']) == (
            'public-domain-by-date', True)

    def test_part_without_a_date_uses_the_volume_year(self) -> None:
        part = dict(PART_NOTES, RightsStatus=NOT_PROVIDED, Date='')
        rec = self._ing().part_record(
            part, BhlIngestor.page_index(ITEM),
            ('unspecified', None, False), item_year=1900)
        assert rec['approved'] is True


class TestPartRecord:
    def _rec(self, part: Dict[str, Any],
             item_lic: Any = ('public-domain', 'NOT_IN_COPYRIGHT', True)
             ) -> Dict[str, Any]:
        ing = _make(license_policy=_policy())
        return ing.part_record(part, BhlIngestor.page_index(ITEM), item_lic)

    def test_fields(self) -> None:
        rec = self._rec(PART_NOTES)
        assert rec == {
            'bhl_part_id': '328815',
            'title': 'Mycological Notes, No. 19',
            'author': 'Lloyd, Curtis Gates',
            'genre': 'Article',
            'date': '1905-05',
            'year': '1905',
            'pages': '205–220',
            'start_page_number': '205',
            'end_page_number': '220',
            'pdf_page_ranges': [[3, 5]],
            'doi': '10.5962/p.328815',
            'url': 'https://www.biodiversitylibrary.org/part/328815',
            'license_id': 'public-domain',
            'license_raw': 'NOT_IN_COPYRIGHT',
            'approved': True,
        }

    def test_no_doi_key_when_absent(self) -> None:
        assert 'doi' not in self._rec(PART_INTRO)

    def test_license_url_beats_rights_status(self) -> None:
        part = dict(PART_NOTES, LicenseUrl=(
            'https://creativecommons.org/licenses/by-nc-sa/4.0/'))
        rec = self._rec(part)
        assert rec['license_id'] == 'CC-BY-NC-SA-4.0'
        assert rec['approved'] is True

    def test_blank_part_rights_fall_back_to_item(self) -> None:
        part = dict(PART_NOTES, RightsStatus='')
        rec = self._rec(part, ('CC-BY-ND-4.0', 'x', False))
        assert (rec['license_id'], rec['approved']) == ('CC-BY-ND-4.0',
                                                        False)

    def test_page_outside_item_gives_no_pdf_range(self) -> None:
        part = copy.deepcopy(PART_NOTES)
        part['Pages'].append({'PageID': 999})
        rec = self._rec(part)
        assert rec['pdf_page_ranges'] is None

    def test_non_contiguous_pages_give_several_ranges(self) -> None:
        # Lloyd v.2, part 328815: text on PDF pages 35-50, its plates
        # bound at the back on 473-502 (live API, 2026-11-23).
        part = copy.deepcopy(PART_NOTES)
        part['Pages'] = [{'PageID': PAGE_IDS[i]} for i in (2, 3, 5)]
        assert self._rec(part)['pdf_page_ranges'] == [[3, 4], [6, 6]]

    def test_ranges_follow_volume_order_not_part_order(self) -> None:
        part = copy.deepcopy(PART_NOTES)
        part['Pages'] = [{'PageID': PAGE_IDS[i]} for i in (5, 2, 3)]
        assert self._rec(part)['pdf_page_ranges'] == [[3, 4], [6, 6]]


class TestKeptPages:
    def _part(self, ranges: Any, approved: bool) -> Dict[str, Any]:
        return {'pdf_page_ranges': ranges, 'approved': approved}

    def test_all_approved(self) -> None:
        parts = [self._part([[1, 2]], True), self._part([[3, 5]], True)]
        assert BhlIngestor.kept_pages(6, parts, True) == [True] * 6

    def test_unapproved_part_withheld(self) -> None:
        parts = [self._part([[1, 2]], True), self._part([[3, 5]], False)]
        assert BhlIngestor.kept_pages(6, parts, True) == [
            True, True, False, False, False, True]

    def test_shared_page_withheld_if_any_part_unapproved(self) -> None:
        parts = [self._part([[1, 3]], True), self._part([[3, 5]], False)]
        assert BhlIngestor.kept_pages(5, parts, True) == [
            True, True, False, False, False]

    def test_pages_outside_parts_follow_the_item(self) -> None:
        parts = [self._part([[2, 3]], True)]
        assert BhlIngestor.kept_pages(4, parts, False) == [
            False, True, True, False]

    def test_non_contiguous_part_withholds_only_its_own_pages(self) -> None:
        # An unapproved article with plates at the back must not take
        # the articles bound between its text and its plates with it.
        parts = [self._part([[1, 2], [6, 6]], False),
                 self._part([[3, 5]], True)]
        assert BhlIngestor.kept_pages(6, parts, True) == [
            False, False, True, True, True, False]

    def test_part_without_pdf_range_is_ignored(self) -> None:
        parts = [self._part(None, False)]
        assert BhlIngestor.kept_pages(2, parts, True) == [True, True]

    def test_withheld_ranges(self) -> None:
        keep = [True, False, False, True, False, True]
        assert BhlIngestor.withheld_ranges(keep) == [(2, 3), (5, 5)]
        assert BhlIngestor.withheld_ranges([True, True]) == []


class TestPdf:
    def test_page_count(self) -> None:
        assert BhlIngestor.pdf_page_count(_pdf(4)) == 4

    def test_redact_blanks_withheld_pages_keeps_count_and_size(self) -> None:
        out = BhlIngestor.redact_pdf(_pdf(4, (300, 400)),
                                     [True, False, True, False])
        assert _texts(out) == ['page 1', '', 'page 3', '']
        with pymupdf.open(stream=out, filetype='pdf') as doc:
            assert doc.page_count == 4
            assert tuple(doc[1].rect)[2:] == (300, 400)

    def test_redact_all_kept_is_unchanged_text(self) -> None:
        out = BhlIngestor.redact_pdf(_pdf(2), [True, True])
        assert _texts(out) == ['page 1', 'page 2']


class TestVolumeDoc:
    def test_fields(self) -> None:
        ing = _make(license_policy=_policy())
        parts = [{'bhl_part_id': '328815'}]
        doc = ing.volume_doc(TITLE, ITEM, parts,
                             ('public-domain', 'NOT_IN_COPYRIGHT', True),
                             IA_PDF)
        assert doc == {
            'title': 'Mycological writings of C. G. Lloyd, v. 2',
            'author': 'Lloyd, C. G. (Curtis Gates), 1859-1926.',
            'journal': 'Mycological writings of C. G. Lloyd',
            'volume': '2',
            'year': '1898',
            'publisher_place': 'Cincinnati, O',
            'doi': '10.5962/bhl.title.19766',
            'itemtype': 'volume',
            'language': 'English',
            'copy_note': 'Includes Mycological Notes, Nos. 19-31',
            'holding_institution': 'University of California Libraries',
            'bhl_title_id': '19766',
            'bhl_item_id': '59611',
            'ia_identifier': 'mycologicalwriti02lloyiala',
            'pdf_url': IA_PDF,
            'url': 'https://www.biodiversitylibrary.org/item/59611',
            'human_url': 'https://www.biodiversitylibrary.org/item/59611',
            'source': 'bhl',
            'license_id': 'public-domain',
            'license_raw': 'NOT_IN_COPYRIGHT',
            'bhl_parts': parts,
            'withheld_pages': [],
            'page_alignment': 'ok',
        }


class TestIngest:
    """End-to-end over mocks: API replies, the IA download and the db."""

    @pytest.fixture(autouse=True)
    def _www_dir(self, tmp_path: Path) -> None:
        self.www_dir = tmp_path / 'www'

    def _ing(self, parts: List[Dict[str, Any]], pdf_pages: int = 6,
             item: Dict[str, Any] = ITEM, **kw: Any) -> BhlIngestor:
        db = MagicMock()
        db.__contains__.return_value = False
        kw.setdefault('www_dir', self.www_dir)
        ing = _make(db=db, license_policy=_policy(), **kw)
        by_part = {str(p['PartID']): p for p in parts}
        item = dict(item, Parts=[{'PartID': p['PartID']} for p in parts])

        def api(op: str, **params: Any) -> Any:
            if op == 'GetTitleMetadata':
                return [TITLE]
            if op == 'GetItemMetadata':
                return [item]
            if op == 'GetPartMetadata':
                return [by_part[str(params['id'])]]
            raise AssertionError(op)

        ing.api = MagicMock(side_effect=api)  # type: ignore[method-assign]
        response = MagicMock(status_code=200, content=_pdf(pdf_pages))
        ing._get_with_rate_limit = MagicMock(  # type: ignore[method-assign]
            return_value=response)
        self.stored: List[Any] = []
        ing._store_volume = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda doc, pdf: self.stored.append((doc, pdf)))
        self.pdf_bytes: bytes = response.content
        ing.ia_metadata = MagicMock(  # type: ignore[method-assign]
            return_value=self.ia if self.ia is not None else {})
        return ing

    ia: Any = None

    def test_public_domain_volume_stored_whole(self) -> None:
        ing = self._ing([PART_INTRO, PART_NOTES])
        ing.ingest()
        assert len(self.stored) == 1
        doc, pdf = self.stored[0]
        assert ing._get_with_rate_limit.call_args.args[0] == IA_PDF  # type: ignore
        assert _texts(pdf) == [f'page {i}' for i in range(1, 7)]
        assert [p['bhl_part_id'] for p in doc['bhl_parts']] == [
            '328814', '328815']
        assert doc['bhl_parts'][1]['pdf_page_ranges'] == [[3, 5]]
        assert doc['withheld_pages'] == []

    def test_part_metadata_requested_with_pages(self) -> None:
        ing = self._ing([PART_NOTES])
        ing.ingest()
        calls = [c for c in ing.api.call_args_list  # type: ignore
                 if c.args[0] == 'GetPartMetadata']
        assert calls[0].kwargs == {'id': '328815', 'pages': 't'}

    def test_mixed_rights_volume_is_redacted(self) -> None:
        nd = dict(PART_NOTES, LicenseUrl=(
            'https://creativecommons.org/licenses/by-nd/4.0/'))
        ing = self._ing([PART_INTRO, nd])
        ing.ingest()
        doc, pdf = self.stored[0]
        assert _texts(pdf) == ['page 1', 'page 2', '', '', '', 'page 6']
        assert doc['withheld_pages'] == [[3, 5]]
        assert doc['bhl_parts'][1]['approved'] is False

    def test_mixed_rights_volume_keeps_the_complete_copy_on_disk(
            self) -> None:
        nd = dict(PART_NOTES, LicenseUrl=(
            'https://creativecommons.org/licenses/by-nd/4.0/'))
        ing = self._ing([PART_INTRO, nd])
        ing.ingest()
        doc, _redacted = self.stored[0]
        # Mirrors the download URL, as elsewhere under /data/skol/www.
        path = (self.www_dir / 'archive.org' / 'download' /
                'mycologicalwriti02lloyiala' /
                'mycologicalwriti02lloyiala.pdf')
        complete = path.read_bytes()
        assert _texts(complete) == [f'page {i}' for i in range(1, 7)]
        assert doc['complete_volume'] == {
            'path': str(path),
            'sha256': hashlib.sha256(complete).hexdigest(),
            'bytes': len(complete),
        }

    def test_all_approved_volume_keeps_no_extra_copy(self) -> None:
        ing = self._ing([PART_INTRO, PART_NOTES])
        ing.ingest()
        doc, _pdf = self.stored[0]
        assert 'complete_volume' not in doc
        assert not self.www_dir.exists()

    def test_complete_copy_failure_stores_nothing(self) -> None:
        nd = dict(PART_NOTES, LicenseUrl=(
            'https://creativecommons.org/licenses/by-nd/4.0/'))
        ing = self._ing([PART_INTRO, nd])
        ing._save_complete_volume = MagicMock(  # type: ignore[method-assign]
            side_effect=OSError('disk full'))
        # (the ingestor's thin wrapper over web_mirror.save_mirror_copy)
        ing.ingest()
        assert self.stored == []

    def test_nothing_approved_downloads_nothing(self) -> None:
        item = dict(ITEM, CopyrightStatus='Not specified')
        intro = dict(PART_INTRO, RightsStatus='Other')
        ing = self._ing([intro], item=item)
        ing.ingest()
        ing._get_with_rate_limit.assert_not_called()  # type: ignore
        assert self.stored == []

    def test_page_mismatch_all_approved_stores_without_ranges(self) -> None:
        ing = self._ing([PART_INTRO, PART_NOTES], pdf_pages=8)
        ing.ingest()
        doc, _pdf_bytes = self.stored[0]
        assert doc['page_alignment'] == 'mismatch'
        assert all(p['pdf_page_ranges'] is None for p in doc['bhl_parts'])

    def test_page_mismatch_with_withheld_pages_is_skipped(self) -> None:
        nd = dict(PART_NOTES, LicenseUrl=(
            'https://creativecommons.org/licenses/by-nd/4.0/'))
        ing = self._ing([PART_INTRO, nd], pdf_pages=8)
        ing.ingest()
        assert self.stored == []

    def test_unapproved_part_without_page_range_skips_volume(self) -> None:
        # Added during implementation: such a part cannot be withheld.
        nd = copy.deepcopy(PART_NOTES)
        nd['LicenseUrl'] = 'https://creativecommons.org/licenses/by-nd/4.0/'
        nd['Pages'] = [{'PageID': 999}]
        ing = self._ing([PART_INTRO, nd])
        ing.ingest()
        ing._get_with_rate_limit.assert_not_called()  # type: ignore
        assert self.stored == []

    def test_non_ia_item_is_skipped(self) -> None:
        ing = self._ing([PART_NOTES], item=dict(ITEM, Source='Other'))
        ing.ingest()
        ing._get_with_rate_limit.assert_not_called()  # type: ignore
        assert self.stored == []

    def test_volume_already_stored_is_not_downloaded(self) -> None:
        ing = self._ing([PART_NOTES])
        ing._is_url_ingested = MagicMock(  # type: ignore[method-assign]
            return_value=True)
        ing.ingest()
        ing._get_with_rate_limit.assert_not_called()  # type: ignore
        assert self.stored == []

    def test_download_that_is_not_a_pdf_is_skipped(self) -> None:
        ing = self._ing([PART_NOTES])
        ing._get_with_rate_limit.return_value = MagicMock(  # type: ignore
            status_code=200, content=b'<html>nope</html>')
        ing.ingest()
        assert self.stored == []

    def test_scan_provenance_is_stored_on_the_volume(self) -> None:
        self.ia = {'ia_contributor': 'University of California Libraries',
                   'ia_scan_date': '2006-11-22T19:44:42'}
        ing = self._ing([PART_NOTES])
        ing.ingest()
        doc, _pdf = self.stored[0]
        assert doc['ia_contributor'] == 'University of California Libraries'
        assert doc['ia_scan_date'] == '2006-11-22T19:44:42'
        ing.ia_metadata.assert_called_once_with(  # type: ignore
            'mycologicalwriti02lloyiala')

    def test_failed_provenance_lookup_still_stores(self) -> None:
        self.ia = {}
        ing = self._ing([PART_NOTES])
        ing.ingest()
        assert len(self.stored) == 1

    def test_matching_md5_is_stored(self) -> None:
        ing = self._ing([PART_NOTES])
        ing.ia_metadata.return_value = {  # type: ignore
            'ia_pdf_md5': hashlib.md5(self.pdf_bytes).hexdigest()}
        ing.ingest()
        assert self.stored[0][0]['ia_pdf_md5'] == \
            hashlib.md5(self.pdf_bytes).hexdigest()

    def test_md5_mismatch_means_a_corrupt_download_and_is_skipped(
            self) -> None:
        ing = self._ing([PART_NOTES])
        ing.ia_metadata.return_value = {  # type: ignore
            'ia_pdf_md5': '0' * 32}
        ing.ingest()
        assert self.stored == []

    def test_dry_run_looks_up_nothing(self) -> None:
        ing = self._ing([PART_NOTES], dry_run=True)
        ing.ingest()
        ing.ia_metadata.assert_not_called()  # type: ignore

    def test_dry_run(self) -> None:
        ing = self._ing([PART_NOTES], dry_run=True)
        ing.ingest()
        ing._get_with_rate_limit.assert_not_called()  # type: ignore
        assert self.stored == []
        assert not self.www_dir.exists()

    def test_limit_counts_stored_volumes(self) -> None:
        title = dict(TITLE, Items=[dict(TITLE['Items'][0]),
                                   dict(TITLE['Items'][0], ItemID=59612)])
        ing = self._ing([PART_NOTES], limit=1)
        original = ing.api.side_effect  # type: ignore

        def api(op: str, **params: Any) -> Any:
            return [title] if op == 'GetTitleMetadata' else original(
                op, **params)

        ing.api.side_effect = api  # type: ignore
        ing.ingest()
        assert len(self.stored) == 1


class TestSaveCompleteVolume:
    def test_saves_the_download_at_its_mirror_path(
            self, tmp_path: Path) -> None:
        ing = _make(www_dir=tmp_path)
        info = ing._save_complete_volume(IA_PDF, b'%PDF-1.5 whole')
        path = (tmp_path / 'archive.org' / 'download' /
                'mycologicalwriti02lloyiala' /
                'mycologicalwriti02lloyiala.pdf')
        assert path.read_bytes() == b'%PDF-1.5 whole'
        assert info['path'] == str(path)


class TestCreatorMode:
    """Which volumes a creator ingest visits, and with which title."""

    def _ing(self, author: Dict[str, Any] = AUTHOR,
             part_items: Any = None, item_titles: Any = None,
             stored: bool = True, title_items: Any = None,
             **kw: Any) -> BhlIngestor:
        # part -> ItemID; item -> TitleID; title 136929 has two items.
        part_items = part_items if part_items is not None else {
            '246166': '1101', '246177': '1201'}
        item_titles = item_titles if item_titles is not None else {
            '1101': '7400', '1201': '10776', '2001': '136929',
            '2002': '136929'}
        ing = _make(bibliography_id=None, creator_id='3484', **kw)

        def api(op: str, **params: Any) -> Any:
            if op == 'GetAuthorMetadata':
                return [author] if author else None
            if op == 'GetTitleMetadata':
                tid = str(params['id'])
                title: Dict[str, Any] = {
                    'TitleID': int(tid), 'FullTitle': f'title {tid}',
                    'ShortTitle': f'title {tid}'}
                if params.get('items') == 't':
                    title['Items'] = [
                        {'ItemID': i} for i in
                        (title_items or {}).get(tid, (2001, 2002))]
                return [title]
            if op == 'GetPartMetadata':
                item = part_items.get(str(params['id']))
                return [{'PartID': params['id'],
                         **({'ItemID': item} if item else {})}]
            if op == 'GetItemMetadata':
                return [{'ItemID': int(params['id']),
                         'TitleID': item_titles[str(params['id'])]}]
            raise AssertionError(op)

        ing.api = MagicMock(side_effect=api)  # type: ignore[method-assign]
        self.visits: List[Any] = []

        def visit(title: Dict[str, Any], item_id: str) -> bool:
            self.visits.append((str(title['TitleID']), item_id,
                                title['FullTitle']))
            return stored

        ing._ingest_item = MagicMock(  # type: ignore[method-assign]
            side_effect=visit)
        return ing

    def test_titles_first_then_the_volume_holding_each_part(self) -> None:
        ing = self._ing()
        ing.ingest()
        assert self.visits == [
            ('136929', '2001', 'title 136929'),
            ('136929', '2002', 'title 136929'),
            ('7400', '1101', 'title 7400'),
            ('10776', '1201', 'title 10776'),
        ]

    def test_author_requested_with_publications(self) -> None:
        ing = self._ing()
        ing.ingest()
        first = ing.api.call_args_list[0]  # type: ignore[attr-defined]
        assert first.args == ('GetAuthorMetadata',)
        assert first.kwargs == {'id': '3484', 'pubs': 't'}

    def test_volume_reached_twice_is_visited_once(self) -> None:
        # The Torrey part sits in a volume of one of the creator's titles.
        ing = self._ing(part_items={'246166': '2002', '246177': '1201'})
        ing.ingest()
        assert [v[1] for v in self.visits] == ['2001', '2002', '1201']

    def test_part_without_an_item_is_skipped(self) -> None:
        ing = self._ing(part_items={'246166': None, '246177': '1201'})
        ing.ingest()
        assert [v[1] for v in self.visits] == ['2001', '2002', '1201']

    def test_host_title_fetched_once_per_title(self) -> None:
        ing = self._ing(item_titles={'1101': '10776', '1201': '10776',
                                     '2001': '136929', '2002': '136929'})
        ing.ingest()
        title_calls = [c for c in ing.api.call_args_list  # type: ignore
                       if c.args[0] == 'GetTitleMetadata'
                       and c.kwargs['id'] == '10776']
        assert len(title_calls) == 1

    def test_limit_counts_stored_volumes(self) -> None:
        ing = self._ing(limit=3)
        ing.ingest()
        assert len(self.visits) == 3

    def test_volumes_not_stored_do_not_count_toward_limit(self) -> None:
        ing = self._ing(limit=1, stored=False)
        ing.ingest()
        assert len(self.visits) == 4

    def test_no_author_metadata_visits_nothing(self) -> None:
        ing = self._ing(author={})
        ing.ingest()
        assert self.visits == []

    # -- title_ids (Buller: keep Researches on fungi, not Essays on wheat)

    def _two_titles(self, **kw: Any) -> BhlIngestor:
        author = copy.deepcopy(AUTHOR)
        author['Publications'].insert(1, {
            'BHLType': 'Title', 'TitleID': '17712',
            'Title': 'Essays on wheat', 'PublicationDate': '1919'})
        return self._ing(author=author,
                         title_items={'136929': (2001, 2002),
                                      '17712': (3001,)}, **kw)

    def test_without_title_ids_every_title_is_visited(self) -> None:
        ing = self._two_titles()
        ing.ingest()
        assert [v[1] for v in self.visits] == [
            '2001', '2002', '3001', '1101', '1201']

    def test_title_ids_keep_only_listed_titles_and_all_parts(
            self) -> None:
        ing = self._two_titles(title_ids=['136929'])
        ing.ingest()
        assert [v[1] for v in self.visits] == [
            '2001', '2002', '1101', '1201']

    def test_unlisted_title_is_not_requested(self) -> None:
        ing = self._two_titles(title_ids=['136929'])
        ing.ingest()
        asked = [c.kwargs['id'] for c in ing.api.call_args_list  # type: ignore
                 if c.args[0] == 'GetTitleMetadata']
        assert '17712' not in asked

    def test_empty_title_ids_keep_only_parts(self) -> None:
        ing = self._two_titles(title_ids=[])
        ing.ingest()
        assert [v[1] for v in self.visits] == ['1101', '1201']

    def test_skipped_title_is_reported(
            self, capsys: pytest.CaptureFixture[str]) -> None:
        ing = self._two_titles(title_ids=['136929'], verbosity=2)
        ing.ingest()
        out = capsys.readouterr().out
        assert '17712' in out and 'Essays on wheat' in out


class TestStoreVolume:
    def test_saves_new_doc_with_id_and_attaches_pdf(self) -> None:
        db = MagicMock()
        ing = _make(db=db, license_policy=_policy())
        doc = {'pdf_url': IA_PDF, 'title': 't'}
        ing._store_volume(doc, b'%PDF-1.5 x')
        saved = db.save.call_args.args[0]
        assert saved['_id'] == uuid5(NAMESPACE_URL, IA_PDF).hex
        assert 'create_time' in saved
        assert saved['title'] == 't'
        att = db.put_attachment.call_args
        assert att.args[2] == 'article.pdf'
        assert att.args[3] == 'application/pdf'
        assert att.args[1].read() == b'%PDF-1.5 x'
