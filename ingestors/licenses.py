"""
Content licence policy: classify a rights statement and decide whether
SKOL may ingest it.

The policy lives in ``data/approved_licenses.json`` (found through
``skol_share``); the rule behind it is in ``docs/approved-licenses.md``.
A licence is identified by a stable id (SPDX where one exists), never
by the raw statement.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

POLICY_FILE = 'approved_licenses.json'


class LicensePolicy:
    """Classify rights statements against an approved-licence list."""

    _approved: Dict[str, bool]
    _strings: Dict[str, str]
    _urls: Dict[str, str]
    _cc_families: Dict[str, bool]

    def __init__(self, policy: Dict[str, Any]) -> None:
        """Build the matcher tables from a parsed policy document."""
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: Path) -> 'LicensePolicy':
        """Load the policy from a JSON file."""
        raise NotImplementedError

    @classmethod
    def default(cls) -> 'LicensePolicy':
        """Load the shipped policy (``skol_share``'s approved_licenses.json)."""
        raise NotImplementedError

    @staticmethod
    def _normalize_url(raw: str) -> Optional[str]:
        """``host/path`` without scheme, ``www.``, deed/legalcode suffix
        or trailing slash, lower-cased; None if ``raw`` is not a URL."""
        raise NotImplementedError

    def classify(self, raw: Optional[str]) -> Optional[str]:
        """
        Return the licence id for a rights statement.

        Blank or None is ``'unspecified'``.  Creative Commons URLs and
        SPDX-style CC ids map to ``CC-<FAMILY>-<version>``.  Returns None
        for a statement the policy does not recognise.
        """
        raise NotImplementedError

    def is_approved(self, license_id: Optional[str]) -> bool:
        """True only for a recognised, approved licence id."""
        raise NotImplementedError

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
        raise NotImplementedError
