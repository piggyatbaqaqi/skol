"""Tests that the skol deb ships every data/ file its code opens.

Shipped code used to find data/ files relative to the source checkout
(``Path(__file__).parent.parent / 'data'``, or the current directory).
An installed package has neither, and on production (2026-11-18)
``/opt/skol/venv/lib/python3.14/site-packages/data`` did not exist: the
canonicalization map, ocr_damage's word lists, particle_detector's
personal fungaria and the nomenclature tokenizers' files were all
missing.  ocr_damage skipped its lists without a word.

Now (Trello EMbuJc0r) they are found through ``skol_share``, and the
deb carries them:

* ``build-deb.sh`` lists them in ``SHARE_FILES`` (paths relative to
  ``data/``) and stages them under ``${VERSION_DIR}/share/``;
* postinst links ``/opt/skol/share`` to that version's ``share/`` in
  the atomic switch, beside ``/opt/skol/bin``;
* ``skol-switch-version`` switches both links.

"Opened by shipped code" is decided statically: a string constant in
a shipped source that equals a tracked data/ path, with or without the
``data/`` prefix.  A constant that only mentions a file (a docstring,
help text) is not one.

Run with: python -m pytest tests/share_staging_test.py -v
"""
from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wheel_modules_test import ROOT, shipped_sources  # noqa: E402

BUILD_DEB = ROOT / 'build-deb.sh'
POSTINST = ROOT / 'debian' / 'postinst.template'
SWITCH_VERSION = ROOT / 'bin' / 'skol-switch-version.py'

# Data files shipped code names but the deb deliberately does not
# carry, with the reason.  Export.csv: user ruling, 2026-11-18.
NOT_SHIPPED: Dict[str, str] = {
    'species/Export.csv': (
        '95 MB MycoBank export (2019), read only by the nomenclature '
        'tokenizers, which nothing imports.  A copy per installed '
        'version would cost 95 MB each.'),
}


def tracked_data_files() -> Set[str]:
    """Paths of the files git tracks under data/, relative to data/."""
    out = subprocess.run(
        ['git', 'ls-files', '-z', 'data'], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout
    return {p[len('data/'):] for p in out.split('\0') if p}


def named_data_files() -> Dict[str, Set[str]]:
    """Tracked data/ files named by shipped code, mapped to the files
    naming them."""
    tracked = tracked_data_files()
    named: Dict[str, Set[str]] = {}
    for source in shipped_sources():
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Constant) and isinstance(
                    node.value, str):
                name = node.value.removeprefix('data/')
                if name in tracked:
                    named.setdefault(name, set()).add(
                        str(source.relative_to(ROOT)))
    return named


def share_files() -> List[str]:
    """The ``SHARE_FILES=( ... )`` array in build-deb.sh."""
    m = re.search(r'^SHARE_FILES=\((.*?)\)', BUILD_DEB.read_text(),
                  re.S | re.M)
    if not m:
        return []
    return [line.split('#')[0].strip()
            for line in m.group(1).splitlines()
            if line.split('#')[0].strip()]


def expand(value: str, variables: Dict[str, str]) -> str:
    """Expand ``${NAME}`` and ``$NAME`` from ``variables``."""
    for _ in range(10):
        new = re.sub(r'\$\{(\w+)\}|\$(\w+)',
                     lambda m: variables.get(m.group(1) or m.group(2),
                                             m.group(0)),
                     value)
        if new == value:
            break
        value = new
    return value


def postinst_links() -> Dict[str, str]:
    """``ln -sfn TARGET LINK`` after the atomic-switch marker, as
    {expanded link: expanded target}, with __VERSION__ left as is."""
    text = POSTINST.read_text()
    variables = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', text, re.M))
    marker = text.find('# ATOMIC SWITCH')
    assert marker >= 0, 'postinst.template lost its atomic-switch marker'
    links: Dict[str, str] = {}
    for target, link in re.findall(r'^ln -sfn "([^"]+)" "([^"]+)"$',
                                   text[marker:], re.M):
        links[expand(link, variables)] = expand(target, variables)
    return links


class TestScan:
    def test_scan_sees_ocr_damage(self) -> None:
        """Guard against the scan passing vacuously."""
        assert ('treatments_to_structured/ocr_damage.py'
                in named_data_files()['corpus_vocabulary.txt'])

    def test_not_shipped_entries_are_named(self) -> None:
        """An exemption for a file no code names is stale."""
        named = named_data_files()
        assert sorted(n for n in NOT_SHIPPED if n not in named) == []

    def test_staged_files_are_tracked(self) -> None:
        """A SHARE_FILES entry git does not track would break the build
        on a fresh clone.  (Vacuous until SHARE_FILES exists.)"""
        tracked = tracked_data_files()
        assert sorted(f for f in share_files() if f not in tracked) == []

    def test_not_shipped_files_are_not_staged(self) -> None:
        """(Vacuous until SHARE_FILES exists.)"""
        assert sorted(set(share_files()) & set(NOT_SHIPPED)) == []


class TestShareFiles:
    def test_shipped_code_names_only_staged_files(self) -> None:
        staged = set(share_files())
        missing = {name: sorted(files)
                   for name, files in named_data_files().items()
                   if name not in staged and name not in NOT_SHIPPED}
        assert missing == {}

    def test_staged_into_the_version_share_dir(self) -> None:
        assert 'staging${VERSION_DIR}/share' in BUILD_DEB.read_text()


class TestInstall:
    def test_postinst_switches_share_with_bin(self) -> None:
        links = postinst_links()
        version_dir = '/opt/skol/versions/__VERSION__'
        assert links.get('/opt/skol/bin') == f'{version_dir}/bin'
        assert links.get('/opt/skol/share') == f'{version_dir}/share'

    def test_switch_version_switches_share(self) -> None:
        spec = importlib.util.spec_from_file_location(
            'skol_switch_version', SWITCH_VERSION)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        symlinks = module.PACKAGE_CONFIG['skol']['symlinks']
        assert symlinks[0] == ('/opt/skol/bin', 'bin'), (
            'get_current_version reads the first symlink')
        assert ('/opt/skol/share', 'share') in symlinks
