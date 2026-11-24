"""Tests for ingestors/licenses.py and the shipped approved-licence list."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# skol_share is a top-level module of the checkout (as in
# particle_detector_test.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .licenses import LicensePolicy  # noqa: E402

BY_DATE = pytest.mark.xfail(reason='public_domain_by_date not implemented',
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


class TestShippedPolicy:
    """Pins the operator's rule of 2026-11-23 (docs/approved-licenses.md)."""

    @pytest.fixture
    def policy(self) -> LicensePolicy:
        return LicensePolicy.from_file(SHIPPED)

    @pytest.mark.parametrize('raw,expected', [
        ('NOT_IN_COPYRIGHT', 'public-domain'),
        # BHL's wording on Lloyd v.1 and v.6 (items 245896, 246297).
        ('Public domain. The BHL considers that this work is no longer '
         'under copyright protection.', 'public-domain'),
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

    @BY_DATE
    def test_public_domain_by_date_cutoff_is_95_years(
            self, policy: LicensePolicy) -> None:
        # Operator, 2026-11-23: works published before current year - 95
        # are public domain (US term for published works; author death
        # dates are not looked up).
        raw = ('Not provided. Contact Holding Institution to verify '
               'copyright status.')
        assert policy.evaluate([raw], year=1930, current_year=2026) == (
            'public-domain-by-date', raw, True)
        assert policy.evaluate([raw], year=1931, current_year=2026) == (
            None, raw, False)

    def test_bhl_in_copyright_statement_is_unrecognised(
            self, policy: LicensePolicy) -> None:
        # BHL pairs this status with a LicenseUrl; the URL decides.
        raw = ('In copyright. Digitized with the permission of the '
               'rights holder.')
        assert policy.classify(raw) is None


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


class TestNormalizeUrl:
    def test_strips_scheme_www_suffix_and_slash(self) -> None:
        norm = LicensePolicy._normalize_url
        assert norm('https://www.creativecommons.org/licenses/by/4.0/'
                    'legalcode.de') == 'creativecommons.org/licenses/by/4.0'
        assert norm('http://Example.org/Path/') == 'example.org/path'
        assert norm('NOT_IN_COPYRIGHT') is None


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


BY_DATE_POLICY: Dict[str, Any] = dict(MINI, public_domain_by_date={
    'id': 'public-domain-by-date', 'years': 95,
    'overrides': ['unspecified', 'no-known-copyright', 'unrecognised']})
BY_DATE_POLICY['licenses'] = MINI['licenses'] + [
    {'id': 'no-known-copyright', 'approved': False,
     'strings': ['No known copyright restrictions'], 'urls': []}]


class TestPublicDomainByDate:
    @pytest.fixture
    def policy(self) -> LicensePolicy:
        return LicensePolicy(BY_DATE_POLICY)

    @BY_DATE
    @pytest.mark.parametrize('raws', [
        [None, 'Other'],                          # unspecified
        [None, None],                             # all blank
        ['No known copyright restrictions'],      # no-known-copyright
        ['Not in copyright. The BHL knows of no copyright restrictions '
         'on this item.'],                        # unrecognised
    ])
    def test_old_work_with_unestablished_rights_is_public_domain(
            self, policy: LicensePolicy, raws: Any) -> None:
        license_id, _raw, approved = policy.evaluate(
            raws, year=1905, current_year=2026)
        assert (license_id, approved) == ('public-domain-by-date', True)

    @BY_DATE
    def test_raw_statement_is_kept(self, policy: LicensePolicy) -> None:
        assert policy.evaluate(['Other'], year=1905, current_year=2026) \
            == ('public-domain-by-date', 'Other', True)

    def test_cutoff_is_strictly_before(self, policy: LicensePolicy) -> None:
        # 2026 - 95 = 1931: 1931 is not before it.
        assert policy.evaluate(['Other'], year=1931, current_year=2026) \
            == ('unspecified', 'Other', False)

    @BY_DATE
    def test_year_just_before_the_cutoff(self,
                                         policy: LicensePolicy) -> None:
        assert policy.evaluate(['Other'], year=1930,
                               current_year=2026)[2] is True

    def test_positive_licences_keep_their_verdict(
            self, policy: LicensePolicy) -> None:
        nd = 'https://creativecommons.org/licenses/by-nd/4.0/'
        assert policy.evaluate([nd], year=1900, current_year=2026) == (
            'CC-BY-ND-4.0', nd, False)
        assert policy.evaluate(['NOT_IN_COPYRIGHT'], year=1900,
                               current_year=2026) == (
            'public-domain', 'NOT_IN_COPYRIGHT', True)

    def test_no_year_no_rule(self, policy: LicensePolicy) -> None:
        assert policy.evaluate(['Other'], year=None, current_year=2026) \
            == ('unspecified', 'Other', False)

    def test_policy_without_the_block_has_no_rule(self) -> None:
        assert LicensePolicy(MINI).evaluate(
            ['Other'], year=1800, current_year=2026)[2] is False

    @BY_DATE
    def test_current_year_defaults_to_today(
            self, policy: LicensePolicy) -> None:
        import datetime
        this_year = datetime.date.today().year
        assert policy.evaluate(['Other'], year=this_year - 96)[2] is True
        assert policy.evaluate(['Other'], year=this_year - 95)[2] is False


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
