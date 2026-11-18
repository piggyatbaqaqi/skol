"""Tests for skol_share.py: where shipped read-only data is found.

Trello EMbuJc0r.  Priority: explicit value, then ``SKOL_SHARE``, then
the source checkout's ``data/``, then ``/opt/skol/share``.

Run with: python -m pytest skol_share_test.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from skol_share import (  # noqa: E402
    INSTALLED_SHARE,
    SHARE_ENV,
    checkout_data_dir,
    share_dir,
    share_file,
)

PENDING = pytest.mark.xfail(
    strict=True, reason='skol_share is a stub until the tests are approved')


@pytest.fixture(autouse=True)
def _no_share_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own SKOL_SHARE must not leak into these tests."""
    monkeypatch.delenv(SHARE_ENV, raising=False)


def _checkout(tmp_path: Path) -> Path:
    """A fake source checkout; returns its skol_share.py path."""
    root = tmp_path / 'repo'
    (root / 'data').mkdir(parents=True)
    (root / 'pyproject.toml').write_text('[project]\nname = "skol"\n')
    return root / 'skol_share.py'


def _site_packages(tmp_path: Path) -> Path:
    """A fake installed module, as the wheel puts it; no pyproject.toml."""
    site = tmp_path / 'venv' / 'lib' / 'python3.14' / 'site-packages'
    site.mkdir(parents=True)
    return site / 'skol_share.py'


def test_constants() -> None:
    assert SHARE_ENV == 'SKOL_SHARE'
    assert INSTALLED_SHARE == Path('/opt/skol/share')


@PENDING
class TestCheckoutDataDir:
    def test_a_checkout_has_one(self, tmp_path: Path) -> None:
        module = _checkout(tmp_path)
        assert checkout_data_dir(module) == module.parent / 'data'

    def test_an_installed_module_has_none(self, tmp_path: Path) -> None:
        assert checkout_data_dir(_site_packages(tmp_path)) is None

    def test_pyproject_without_data_is_not_a_checkout(
            self, tmp_path: Path) -> None:
        """Both markers are required: a sdist build tree has
        pyproject.toml but may lack data/."""
        root = tmp_path / 'sdist'
        root.mkdir()
        (root / 'pyproject.toml').write_text('')
        assert checkout_data_dir(root / 'skol_share.py') is None

    def test_this_checkout(self) -> None:
        """With no argument, the module's own location decides."""
        assert checkout_data_dir() == ROOT / 'data'


@PENDING
class TestShareDir:
    """The four steps, each beating the ones after it."""

    def test_explicit_value_beats_everything(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SHARE_ENV, str(tmp_path / 'from_env'))
        assert (share_dir(tmp_path / 'from_cli',
                          module_file=_checkout(tmp_path))
                == tmp_path / 'from_cli')

    def test_explicit_value_may_be_a_string(self, tmp_path: Path) -> None:
        assert share_dir(str(tmp_path / 'x')) == tmp_path / 'x'

    def test_env_beats_the_checkout(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SHARE_ENV, str(tmp_path / 'from_env'))
        assert (share_dir(module_file=_checkout(tmp_path))
                == tmp_path / 'from_env')

    def test_env_beats_the_installed_default(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SHARE_ENV, str(tmp_path / 'from_env'))
        assert (share_dir(module_file=_site_packages(tmp_path))
                == tmp_path / 'from_env')

    def test_checkout_beats_the_installed_default(
            self, tmp_path: Path) -> None:
        """A dev box with the deb installed still reads its checkout."""
        module = _checkout(tmp_path)
        assert share_dir(module_file=module) == module.parent / 'data'

    def test_installed_default(self, tmp_path: Path) -> None:
        assert (share_dir(module_file=_site_packages(tmp_path))
                == INSTALLED_SHARE)

    def test_this_checkout(self) -> None:
        assert share_dir() == ROOT / 'data'


@PENDING
class TestShareFile:
    def test_joins_onto_the_share_dir(self, tmp_path: Path) -> None:
        assert (share_file('corpus_vocabulary.txt', tmp_path)
                == tmp_path / 'corpus_vocabulary.txt')

    def test_keeps_subdirectories(self, tmp_path: Path) -> None:
        assert (share_file('authors/FungalNameAuthors.txt', tmp_path)
                == tmp_path / 'authors' / 'FungalNameAuthors.txt')

    def test_follows_the_same_priority(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SHARE_ENV, str(tmp_path))
        assert share_file('x.json') == tmp_path / 'x.json'

    def test_installed(self, tmp_path: Path) -> None:
        assert (share_file('x.json', module_file=_site_packages(tmp_path))
                == INSTALLED_SHARE / 'x.json')
