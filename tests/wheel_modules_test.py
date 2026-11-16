"""Tests that the skol wheel ships every top-level module its code imports.

Root-level modules reach the wheel only through ``py-modules`` in
pyproject.toml.  ``span_resolver`` and ``treatments_json_translator``
were missing, so on production (2026-11-15) ``verify_spans`` died with
``No module named 'span_resolver'``; ``assess_treatments`` and
``treatments_to_json`` import the other.  setup.py carried a second,
drifted module list (it had ``treatments_json_translator``) that the
build ignores: pyproject.toml's ``[tool.setuptools]`` wins.

Run with: python -m pytest tests/wheel_modules_test.py -v
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
SETUPTOOLS = tomllib.loads(
    (ROOT / 'pyproject.toml').read_text())['tool']['setuptools']
PY_MODULES: Set[str] = set(SETUPTOOLS['py-modules'])
PACKAGES: Set[str] = {
    p.split('.')[0] for p in SETUPTOOLS['packages']['find']['include']}


def local_top_level() -> Set[str]:
    """Names importable from the repo root: modules and packages."""
    return ({p.stem for p in ROOT.glob('*.py')}
            | {p.name for p in ROOT.iterdir()
               if (p / '__init__.py').exists()})


def imported_names(path: Path) -> Set[str]:
    """Top-level names of the absolute imports in ``path``."""
    names: Set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split('.')[0] for a in node.names)
        elif (isinstance(node, ast.ImportFrom) and node.level == 0
              and node.module):
            names.add(node.module.split('.')[0])
    return names


def shipped_sources() -> List[Path]:
    """Non-test .py files the wheel ships: listed modules + packages."""
    files = [ROOT / f'{m}.py' for m in PY_MODULES
             if (ROOT / f'{m}.py').exists()]
    for pkg in PACKAGES:
        files.extend((ROOT / pkg).rglob('*.py'))
    return [f for f in files if not f.stem.endswith('_test')]


def unshipped_imports() -> Dict[str, Set[str]]:
    """Local top-level modules imported by shipped code (transitively)
    but not shipped, mapped to the files importing them."""
    local = local_top_level()
    bin_scripts = {p.stem for p in (ROOT / 'bin').glob('*.py')}
    missing: Dict[str, Set[str]] = {}
    todo = shipped_sources()
    seen: Set[Path] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in imported_names(path):
            if (name in local and name not in PY_MODULES
                    and name not in PACKAGES and name not in bin_scripts):
                missing.setdefault(name, set()).add(
                    str(path.relative_to(ROOT)))
                module = ROOT / f'{name}.py'
                if module.exists():
                    todo.append(module)
    return missing


class TestPyModules:
    def test_listed_modules_exist(self) -> None:
        """A py-modules entry with no file (``taxa_json_translator``)
        is a stale name."""
        assert sorted(m for m in PY_MODULES
                      if not (ROOT / f'{m}.py').exists()) == []

    def test_shipped_code_imports_only_shipped_modules(self) -> None:
        assert unshipped_imports() == {}

    def test_scan_sees_bin(self) -> None:
        """Guard against the scan passing vacuously."""
        assert 'span_resolver' in imported_names(
            ROOT / 'bin' / 'verify_spans.py')


class TestSingleModuleList:
    def test_setup_py_declares_no_modules_or_packages(self) -> None:
        """pyproject.toml is the one list; setup.py's copy drifted."""
        tree = ast.parse((ROOT / 'setup.py').read_text())
        kwargs = {kw.arg for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and getattr(node.func, 'id', None) == 'setup'
                  for kw in node.keywords}
        assert kwargs & {'py_modules', 'packages'} == set()
