"""Tests for the venv handling in debian/postinst.template and
django/debian/postinst.template.

On skol.synoptickeyof.life the skol venv was created with python3.13
in Feb 2026.  postinst later ran ``python3.14 -m venv /opt/skol/venv``
over it; ``venv`` adds ``bin/python3.14`` but leaves an existing
``bin/python`` alone, so pip installed into the 3.14 site-packages
while ``with_skol`` ran ``python`` (3.13) and every cron job died with
``No module named 'treatments_to_structured'`` (2026-11-13).

The venv step lives between ``# BEGIN ensure-venv`` and
``# END ensure-venv`` markers, reading ``${PYTHON}`` and ``${VENV}``;
these tests run that block with a stub interpreter.

Run with: python -m pytest tests/postinst_venv_test.py -v
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
POSTINSTS = {
    'skol': ROOT / 'debian' / 'postinst.template',
    'skol-django': ROOT / 'django' / 'debian' / 'postinst.template',
}
BEGIN = '# BEGIN ensure-venv'
END = '# END ensure-venv'


def ensure_venv_block(path: Path) -> str:
    """The lines between the ensure-venv markers."""
    text = path.read_text()
    m = re.search(rf'^{re.escape(BEGIN)}\n(.*?)^{re.escape(END)}$',
                  text, re.S | re.M)
    assert m, f'{path}: no {BEGIN} ... {END} block'
    return m.group(1)


def postinst_python(path: Path) -> str:
    """The interpreter a postinst names in ``PYTHON=``."""
    m = re.search(r'^PYTHON=(\S+)$', path.read_text(), re.M)
    assert m, f'{path}: no PYTHON= line'
    return m.group(1)


def run_block(block: str, tmp: Path, venv_python_target: str) -> List[str]:
    """Run ``block`` with a stub ``python3.14`` that logs its argv.

    ``venv_python_target``: ``'none'`` (no venv yet), ``'same'``
    (``bin/python`` -> the stub) or ``'other'`` (-> a python3.13).
    Returns the stub's logged invocations, one string each.
    """
    stub_dir = tmp / 'stubbin'
    stub_dir.mkdir()
    log = tmp / 'calls.log'
    stub = stub_dir / 'python3.14'
    stub.write_text(f'#!/bin/sh\necho "$*" >> {log}\n')
    stub.chmod(0o755)
    other = tmp / 'usr' / 'python3.13'
    other.parent.mkdir()
    other.write_text('#!/bin/sh\n')
    other.chmod(0o755)

    venv = tmp / 'venv'
    if venv_python_target != 'none':
        (venv / 'bin').mkdir(parents=True)
        target = stub if venv_python_target == 'same' else other
        (venv / 'bin' / 'python').symlink_to(target)

    env = dict(os.environ,
               PATH=f'{stub_dir}:/usr/bin:/bin',
               PYTHON='python3.14', VENV=str(venv))
    subprocess.run(['bash', '-ec', block], env=env, check=True,
                   capture_output=True, text=True)
    return log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize('pkg', sorted(POSTINSTS))
class TestEnsureVenv:
    def test_wrong_interpreter_recreates_with_clear(
            self, pkg: str, tmp_path: Path) -> None:
        """The production failure: bin/python -> python3.13."""
        calls = run_block(ensure_venv_block(POSTINSTS[pkg]), tmp_path,
                          'other')
        assert calls == [f'-m venv --clear {tmp_path / "venv"}']

    def test_right_interpreter_keeps_venv(
            self, pkg: str, tmp_path: Path) -> None:
        """A normal upgrade must not wipe the venv (torch etc.)."""
        calls = run_block(ensure_venv_block(POSTINSTS[pkg]), tmp_path,
                          'same')
        assert calls == [f'-m venv {tmp_path / "venv"}']

    def test_fresh_install_creates_venv(
            self, pkg: str, tmp_path: Path) -> None:
        calls = run_block(ensure_venv_block(POSTINSTS[pkg]), tmp_path,
                          'none')
        assert calls == [f'-m venv {tmp_path / "venv"}']

    def test_block_sets_venv_before_running(self, pkg: str) -> None:
        """The block reads ${VENV}; postinst must assign it first."""
        text = POSTINSTS[pkg].read_text()
        assert re.search(r'^VENV=\$\{SKOL_\w*VENV\}\n' + re.escape(BEGIN),
                         text, re.M)


class TestBlocksAgree:
    def test_both_postinsts_share_the_block(self) -> None:
        blocks = {ensure_venv_block(p) for p in POSTINSTS.values()}
        assert len(blocks) == 1


class TestWithSkolInterpreter:
    """with_skol runs scripts with the venv's exact interpreter, not
    whatever ``python`` happens to resolve to."""

    def test_no_bare_python_exec(self) -> None:
        text = POSTINSTS['skol'].read_text()
        assert not re.search(r'^\s*exec python\b', text, re.M)

    def test_exec_names_postinst_python(self) -> None:
        path = POSTINSTS['skol']
        python = postinst_python(path)
        assert (f'exec "${{VIRTUAL_ENV}}/bin/{python}" '
                '"${SKOL_HOME}/bin/${SCRIPT_NAME}.py" "$@"') in path.read_text()
