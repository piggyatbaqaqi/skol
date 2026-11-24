"""
Content licence policy: classify a rights statement and decide whether
SKOL may ingest it.

The policy lives in ``data/approved_licenses.json`` (found through
``skol_share``); the rule behind it is in ``docs/approved-licenses.md``.
A licence is identified by a stable id (SPDX where one exists), never
by the raw statement.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from skol_share import share_file

POLICY_FILE = 'approved_licenses.json'

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
        raws: Iterable[Optional[str]]
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Decide from statements in priority order (e.g. LicenseUrl, then
        CopyrightStatus): the first non-blank one decides.

        Returns:
            (license_id, raw statement used, approved).  All blank gives
            ``('unspecified', None, False)``; an unrecognised statement
            gives ``(None, raw, False)``.
        """
        for raw in raws:
            if raw is None or not raw.strip():
                continue
            license_id = self.classify(raw)
            return license_id, raw, self.is_approved(license_id)
        return 'unspecified', None, False
