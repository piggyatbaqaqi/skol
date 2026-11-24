"""Tests for ingestors/licenses.py and the shipped approved-licence list."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from .licenses import LicensePolicy

NOT_YET = pytest.mark.xfail(reason='LicensePolicy not implemented',
                            strict=True)

SHIPPED = Path(__file__).resolve().parent.parent / 'data' / \
    'approved_licenses.json'

MINI: Dict[str, Any] = {
    'version': 1,
    'licenses': [
        {'id': 'public-domain', 'approved': True,
         'strings': ['NOT_IN_COPYRIGHT'],
         'urls': ['creativecommons.org/publicdomain/mark/1.0']},
        {'id': 'unspecified', 'approved': False,
         'strings': ['', 'Other'], 'urls': []},
    ],
    'creative_commons': {'by': True, 'by-nd': False},
}


class TestShippedFile:
    def test_is_valid_json_with_the_expected_shape(self) -> None:
        policy = json.loads(SHIPPED.read_text())
        assert policy['version'] == 1
        ids = [entry['id'] for entry in policy['licenses']]
        assert len(ids) == len(set(ids))
        assert set(policy['creative_commons']) == {
            'by', 'by-sa', 'by-nc', 'by-nc-sa', 'by-nd', 'by-nc-nd'}


@NOT_YET
class TestShippedPolicy:
    """Pins the operator's rule of 2026-11-23 (docs/approved-licenses.md)."""

    @pytest.fixture
    def policy(self) -> LicensePolicy:
        return LicensePolicy.from_file(SHIPPED)

    @pytest.mark.parametrize('raw,expected', [
        ('NOT_IN_COPYRIGHT', 'public-domain'),
        ('https://creativecommons.org/publicdomain/zero/1.0/', 'CC0-1.0'),
        ('http://creativecommons.org/licenses/by/4.0/', 'CC-BY-4.0'),
        ('https://creativecommons.org/licenses/by-sa/3.0/', 'CC-BY-SA-3.0'),
        ('https://creativecommons.org/licenses/by-nc/2.5/', 'CC-BY-NC-2.5'),
        ('https://creativecommons.org/licenses/by-nc-sa/4.0/',
         'CC-BY-NC-SA-4.0'),
    ])
    def test_approved(self, policy: LicensePolicy, raw: str,
                      expected: str) -> None:
        assert policy.classify(raw) == expected
        assert policy.is_approved(expected)

    @pytest.mark.parametrize('raw,expected', [
        ('https://creativecommons.org/licenses/by-nd/4.0/', 'CC-BY-ND-4.0'),
        ('https://creativecommons.org/licenses/by-nc-nd/3.0/',
         'CC-BY-NC-ND-3.0'),
        ('No known copyright restrictions as determined by scanning '
         'institution.', 'no-known-copyright'),
        ('Not specified', 'unspecified'),
        ('Other', 'unspecified'),
        ('', 'unspecified'),
        (None, 'unspecified'),
    ])
    def test_rejected(self, policy: LicensePolicy, raw: str,
                      expected: str) -> None:
        assert policy.classify(raw) == expected
        assert not policy.is_approved(expected)

    def test_bhl_in_copyright_statement_is_unrecognised(
            self, policy: LicensePolicy) -> None:
        # BHL pairs this status with a LicenseUrl; the URL decides.
        raw = ('In copyright. Digitized with the permission of the '
               'rights holder.')
        assert policy.classify(raw) is None


@NOT_YET
class TestClassify:
    @pytest.fixture
    def policy(self) -> LicensePolicy:
        return LicensePolicy(MINI)

    @pytest.mark.parametrize('raw', [
        'https://creativecommons.org/licenses/by/4.0/',
        'http://creativecommons.org/licenses/by/4.0',
        'https://www.creativecommons.org/licenses/by/4.0/',
        'https://creativecommons.org/licenses/by/4.0/legalcode',
        'https://creativecommons.org/licenses/by/4.0/deed.en',
        'HTTPS://CreativeCommons.org/licenses/BY/4.0/',
        '  https://creativecommons.org/licenses/by/4.0/  ',
        'CC-BY-4.0',
        'cc-by-4.0',
    ])
    def test_cc_by_spellings(self, policy: LicensePolicy, raw: str) -> None:
        assert policy.classify(raw) == 'CC-BY-4.0'

    def test_cc_family_missing_from_policy_is_unrecognised(
            self, policy: LicensePolicy) -> None:
        assert policy.classify(
            'https://creativecommons.org/licenses/by-sa/4.0/') is None

    def test_strings_match_case_and_space_insensitively(
            self, policy: LicensePolicy) -> None:
        assert policy.classify('  not_in_copyright ') == 'public-domain'

    def test_url_matcher(self, policy: LicensePolicy) -> None:
        assert policy.classify(
            'https://creativecommons.org/publicdomain/mark/1.0/') \
            == 'public-domain'

    def test_unknown_url_and_text(self, policy: LicensePolicy) -> None:
        assert policy.classify('https://example.org/licence') is None
        assert policy.classify('All rights reserved') is None

    def test_is_approved(self, policy: LicensePolicy) -> None:
        assert policy.is_approved('public-domain')
        assert policy.is_approved('CC-BY-4.0')
        assert not policy.is_approved('CC-BY-ND-4.0')
        assert not policy.is_approved('unspecified')
        assert not policy.is_approved(None)
        assert not policy.is_approved('CC-BY-SA-4.0')  # family not listed
        assert not policy.is_approved('made-up')


@NOT_YET
class TestNormalizeUrl:
    def test_strips_scheme_www_suffix_and_slash(self) -> None:
        norm = LicensePolicy._normalize_url
        assert norm('https://www.creativecommons.org/licenses/by/4.0/'
                    'legalcode.de') == 'creativecommons.org/licenses/by/4.0'
        assert norm('http://Example.org/Path/') == 'example.org/path'
        assert norm('NOT_IN_COPYRIGHT') is None


@NOT_YET
class TestEvaluate:
    @pytest.fixture
    def policy(self) -> LicensePolicy:
        return LicensePolicy(MINI)

    def test_first_non_blank_decides(self, policy: LicensePolicy) -> None:
        assert policy.evaluate(
            [None, '', 'https://creativecommons.org/licenses/by-nd/4.0/',
             'NOT_IN_COPYRIGHT']) == (
            'CC-BY-ND-4.0',
            'https://creativecommons.org/licenses/by-nd/4.0/', False)

    def test_approved(self, policy: LicensePolicy) -> None:
        assert policy.evaluate([None, 'NOT_IN_COPYRIGHT']) == (
            'public-domain', 'NOT_IN_COPYRIGHT', True)

    def test_all_blank_is_unspecified(self, policy: LicensePolicy) -> None:
        assert policy.evaluate([None, '  ']) == ('unspecified', None, False)

    def test_unrecognised(self, policy: LicensePolicy) -> None:
        assert policy.evaluate(['Rights reserved']) == (
            None, 'Rights reserved', False)


@NOT_YET
class TestLoading:
    def test_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'p.json'
        path.write_text(json.dumps(MINI))
        assert LicensePolicy.from_file(path).classify(
            'NOT_IN_COPYRIGHT') == 'public-domain'

    def test_default_reads_the_share_file(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / 'approved_licenses.json').write_text(json.dumps(MINI))
        monkeypatch.setenv('SKOL_SHARE', str(tmp_path))
        policy = LicensePolicy.default()
        assert policy.classify('Other') == 'unspecified'
        assert policy.classify(
            'https://creativecommons.org/licenses/by-sa/4.0/') is None
