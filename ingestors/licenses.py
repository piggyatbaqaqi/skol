"""
Content licence policy: classify a rights statement and decide whether
SKOL may ingest it.

The policy lives in ``data/approved_licenses.json`` (found through
``skol_share``); the rule behind it is in ``docs/approved-licenses.md``.
A licence is identified by a stable id (SPDX where one exists), never
by the raw statement.
"""

import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from skol_share import share_file

POLICY_FILE = 'approved_licenses.json'

# Verdicts whose rights statement establishes nothing (blank, "Not
# specified", "No known copyright restrictions"); None, an unrecognised
# statement, is one too.  Only these may be overridden by the date rule
# or by an operator's licence assertion.
UNESTABLISHED = frozenset({'unspecified', 'no-known-copyright'})

# creativecommons.org/licenses/<family>/<version>
_CC_URL = re.compile(r'^creativecommons\.org/licenses/([a-z-]+)/(\d+(?:\.\d+)?)$')
# SPDX-style CC id: CC-BY-NC-SA-4.0
_CC_SPDX = re.compile(r'^cc-((?:by)(?:-(?:nc|sa|nd))*)-(\d+(?:\.\d+)?)$')
# Trailing parts of a CC URL that do not change the licence.
_URL_SUFFIX = re.compile(r'/(legalcode|deed)(\.[a-z_-]+)?$')


class LicensePolicy:
    """Classify rights statements against an approved-licence list."""

    _approved: Dict[str, bool]
    _strings: Dict[str, str]
    _urls: Dict[str, str]
    _cc_families: Dict[str, bool]
    _by_date: Optional[Dict[str, Any]]

    def __init__(self, policy: Dict[str, Any]) -> None:
        """Build the matcher tables from a parsed policy document."""
        self._approved = {}
        self._strings = {}
        self._urls = {}
        for entry in policy.get('licenses', []):
            license_id = entry['id']
            self._approved[license_id] = bool(entry['approved'])
            for raw in entry.get('strings', []):
                self._strings[self._normalize_string(raw)] = license_id
            for url in entry.get('urls', []):
                norm = self._normalize_url(url) or url.strip().lower()
                self._urls[norm] = license_id
        self._cc_families = {
            family.lower(): bool(ok)
            for family, ok in policy.get('creative_commons', {}).items()}
        self._by_date = policy.get('public_domain_by_date')

    @classmethod
    def from_file(cls, path: Path) -> 'LicensePolicy':
        """Load the policy from a JSON file."""
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    @classmethod
    def default(cls) -> 'LicensePolicy':
        """Load the shipped policy (``skol_share``'s approved_licenses.json)."""
        return cls.from_file(share_file(POLICY_FILE))

    @staticmethod
    def _normalize_url(raw: str) -> Optional[str]:
        """``host/path`` without scheme, ``www.``, deed/legalcode suffix
        or trailing slash, lower-cased; None if ``raw`` is not a URL."""
        text = raw.strip().lower()
        match = re.match(r'^https?://(.*)$', text)
        if not match:
            return None
        rest = match.group(1)
        if rest.startswith('www.'):
            rest = rest[4:]
        rest = rest.rstrip('/')
        rest = _URL_SUFFIX.sub('', rest).rstrip('/')
        return rest

    @staticmethod
    def _normalize_string(raw: str) -> str:
        """Lower-cased, whitespace-collapsed statement."""
        return ' '.join(raw.split()).lower()

    def _cc_id(self, family: str, version: str) -> Optional[str]:
        """``CC-<FAMILY>-<version>`` for a family the policy lists."""
        if family not in self._cc_families:
            return None
        return f'CC-{family.upper()}-{version}'

    def classify(self, raw: Optional[str]) -> Optional[str]:
        """
        Return the licence id for a rights statement.

        Blank or None is ``'unspecified'``.  Creative Commons URLs and
        SPDX-style CC ids map to ``CC-<FAMILY>-<version>``.  Returns None
        for a statement the policy does not recognise.
        """
        if raw is None or not raw.strip():
            return 'unspecified'
        text = self._normalize_string(raw)
        if text in self._strings:
            return self._strings[text]
        url = self._normalize_url(raw)
        if url is not None:
            if url in self._urls:
                return self._urls[url]
            cc = _CC_URL.match(url)
            if cc:
                return self._cc_id(cc.group(1), cc.group(2))
            return None
        spdx = _CC_SPDX.match(text)
        if spdx:
            return self._cc_id(spdx.group(1), spdx.group(2))
        return None

    @staticmethod
    def is_unestablished(license_id: Optional[str]) -> bool:
        """True for a verdict that establishes nothing (see
        UNESTABLISHED), including None (unrecognised)."""
        raise NotImplementedError

    def is_approved(self, license_id: Optional[str]) -> bool:
        """True only for a recognised, approved licence id."""
        if license_id is None:
            return False
        if license_id in self._approved:
            return self._approved[license_id]
        spdx = _CC_SPDX.match(license_id.lower())
        if spdx:
            return self._cc_families.get(spdx.group(1), False)
        return False

    def evaluate(
        self,
        raws: Iterable[Optional[str]],
        year: Optional[int] = None,
        current_year: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Decide from statements in priority order (e.g. LicenseUrl, then
        CopyrightStatus): the first non-blank one decides.

        If that verdict is one the policy's ``public_domain_by_date``
        overrides (unspecified, no-known-copyright, unrecognised) and
        ``year`` is before ``current_year`` (default: this year) minus
        its ``years``, the work is public domain by date instead.

        Returns:
            (license_id, raw statement used, approved).  All blank gives
            ``('unspecified', None, False)``; an unrecognised statement
            gives ``(None, raw, False)``; the date rule gives
            ``('public-domain-by-date', raw, True)``.
        """
        verdict: Tuple[Optional[str], Optional[str], bool] = (
            'unspecified', None, False)
        for raw in raws:
            if raw is None or not raw.strip():
                continue
            license_id = self.classify(raw)
            verdict = (license_id, raw, self.is_approved(license_id))
            break
        return self._apply_date_rule(verdict, year, current_year)

    def _apply_date_rule(
        self,
        verdict: Tuple[Optional[str], Optional[str], bool],
        year: Optional[int],
        current_year: Optional[int],
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Public domain by date for an old work whose stated rights
        establish nothing (see evaluate)."""
        rule = self._by_date
        license_id, raw, approved = verdict
        if rule is None or year is None or approved:
            return verdict
        kind = 'unrecognised' if license_id is None else license_id
        if kind not in rule.get('overrides', []):
            return verdict
        now = (current_year if current_year is not None
               else datetime.date.today().year)
        if year < now - int(rule['years']):
            return rule['id'], raw, True
        return verdict
