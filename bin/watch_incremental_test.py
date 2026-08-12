#!/usr/bin/env python3
"""Tests for watch_incremental.py."""
import subprocess
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent))
import watch_incremental  # noqa: E402


def _deb(tmp_path: Path, name: str) -> Path:
    """Create a fake deb file and return its path."""
    path = tmp_path / name
    path.write_text("fake deb")
    return path


class TestGetMatchingFiles:
    """`get_matching_files` expands one glob pattern."""

    def test_matches_versioned_debs(self, tmp_path: Path) -> None:
        a = _deb(tmp_path, "skol_0.9.0-134_all.deb")
        b = _deb(tmp_path, "skol_0.9.0-135_all.deb")

        found = watch_incremental.get_matching_files(
            str(tmp_path / "skol_*_all.deb"))

        assert found == {a, b}

    def test_no_match_is_empty_set(self, tmp_path: Path) -> None:
        assert watch_incremental.get_matching_files(
            str(tmp_path / "skol_*_all.deb")) == set()

    def test_patterns_do_not_bleed_into_each_other(
        self, tmp_path: Path
    ) -> None:
        """skol_*_all.deb must not swallow skol-django_*_all.deb --
        each pattern stands for exactly one package."""
        skol = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        _deb(tmp_path, "skol-django_0.9.0-114_all.deb")

        found = watch_incremental.get_matching_files(
            str(tmp_path / "skol_*_all.deb"))

        assert found == {skol}


class TestGetLatestFile:
    """`get_latest_file` chooses which of several matches to install."""

    def test_empty_returns_none(self) -> None:
        assert watch_incremental.get_latest_file(set()) is None

    def test_single_file(self, tmp_path: Path) -> None:
        f = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        assert watch_incremental.get_latest_file({f}) == f

    def test_picks_higher_build_number(self, tmp_path: Path) -> None:
        a = _deb(tmp_path, "skol_0.9.0-135_all.deb")
        b = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        assert watch_incremental.get_latest_file({a, b}) == b


class TestInstallPackages:
    """`install_packages` shells out to dpkg and then to postinstall."""

    @patch('watch_incremental.subprocess.run')
    def test_basic_install(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        f = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        mock_run.return_value = MagicMock(returncode=0)

        ok = watch_incremental.install_packages(
            files=[f], install_cmd='dpkg -i', install_args=[],
            postinstall=None, verbosity=0)

        assert ok is True
        argv = mock_run.call_args[0][0]
        assert 'dpkg' in argv and '-i' in argv and str(f) in argv

    @patch('watch_incremental.subprocess.run')
    def test_extra_install_args_are_forwarded(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        f = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        mock_run.return_value = MagicMock(returncode=0)

        watch_incremental.install_packages(
            files=[f], install_cmd='dpkg -i', install_args=['--force-all'],
            postinstall=None, verbosity=0)

        assert '--force-all' in mock_run.call_args[0][0]

    @patch('watch_incremental.subprocess.run')
    def test_postinstall_runs_after_install(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        f = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        mock_run.return_value = MagicMock(returncode=0)

        ok = watch_incremental.install_packages(
            files=[f], install_cmd='dpkg -i', install_args=[],
            postinstall='systemctl restart skol-django', verbosity=0)

        assert ok is True
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[1][1].get('shell') is True

    @patch('watch_incremental.subprocess.run')
    def test_install_failure_reported(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        f = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        mock_run.side_effect = subprocess.CalledProcessError(1, 'dpkg')

        assert watch_incremental.install_packages(
            files=[f], install_cmd='dpkg -i', install_args=[],
            postinstall=None, verbosity=0) is False

    @patch('watch_incremental.subprocess.run')
    def test_missing_install_command_reported(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        f = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        mock_run.side_effect = FileNotFoundError('no dpkg')

        assert watch_incremental.install_packages(
            files=[f], install_cmd='dpkg -i', install_args=[],
            postinstall=None, verbosity=0) is False

    @patch('watch_incremental.subprocess.run')
    def test_postinstall_failure_reported(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        f = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        mock_run.side_effect = [
            MagicMock(returncode=0),
            subprocess.CalledProcessError(1, 'systemctl'),
        ]

        assert watch_incremental.install_packages(
            files=[f], install_cmd='dpkg -i', install_args=[],
            postinstall='systemctl restart skol-django',
            verbosity=0) is False

    @patch('watch_incremental.subprocess.run')
    def test_several_packages_go_in_one_dpkg_call(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """skol and skol-django must reach a single dpkg invocation so
        they can satisfy each other's dependencies."""
        a = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        b = _deb(tmp_path, "skol-django_0.9.0-115_all.deb")
        mock_run.return_value = MagicMock(returncode=0)

        watch_incremental.install_packages(
            files=[a, b], install_cmd='dpkg -i', install_args=[],
            postinstall=None, verbosity=0)

        mock_run.assert_called_once()
        argv = mock_run.call_args[0][0]
        assert str(a) in argv and str(b) in argv


def _run_loop_with_sleeps(
    patterns: List[str], sleeps: List[Any], **kwargs: Any
) -> None:
    """Drive watch_and_install for a fixed number of polls.

    Each entry in `sleeps` is called in place of time.sleep; the loop is
    then broken with KeyboardInterrupt, which the function handles.
    """
    effects = list(sleeps) + [KeyboardInterrupt()]

    def fake_sleep(_seconds: float) -> None:
        effect = effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        effect()

    with patch('watch_incremental.time.sleep', side_effect=fake_sleep):
        watch_incremental.watch_and_install(
            patterns=patterns,
            install_cmd=kwargs.get('install_cmd', 'true'),
            install_args=[],
            postinstall=None,
            interval=0.01,
            verbosity=kwargs.get('verbosity', 0),
        )


class TestWatchAndInstall:
    """The watch loop itself."""

    @patch('watch_incremental.install_packages')
    def test_startup_files_are_not_installed(
        self, mock_install: MagicMock, tmp_path: Path
    ) -> None:
        """The documented contract: what is already on disk when the
        watcher starts is a baseline, not something to reinstall."""
        _deb(tmp_path, "skol_0.9.0-136_all.deb")

        _run_loop_with_sleeps([str(tmp_path / "skol_*_all.deb")], [])

        mock_install.assert_not_called()

    @patch('watch_incremental.install_packages')
    def test_new_file_is_installed(
        self, mock_install: MagicMock, tmp_path: Path
    ) -> None:
        _deb(tmp_path, "skol_0.9.0-136_all.deb")
        pattern = str(tmp_path / "skol_*_all.deb")
        new = tmp_path / "skol_0.9.0-137_all.deb"

        _run_loop_with_sleeps(
            [pattern], [lambda: new.write_text("fake deb")])

        mock_install.assert_called_once()
        assert mock_install.call_args[0][0] == [new]

    @patch('watch_incremental.install_packages')
    def test_unchanged_directory_installs_nothing(
        self, mock_install: MagicMock, tmp_path: Path
    ) -> None:
        _deb(tmp_path, "skol_0.9.0-136_all.deb")

        _run_loop_with_sleeps(
            [str(tmp_path / "skol_*_all.deb")],
            [lambda: None, lambda: None])

        mock_install.assert_not_called()

    @patch('watch_incremental.install_packages')
    def test_two_patterns_batch_into_one_install(
        self, mock_install: MagicMock, tmp_path: Path
    ) -> None:
        _deb(tmp_path, "skol_0.9.0-136_all.deb")
        _deb(tmp_path, "skol-django_0.9.0-114_all.deb")
        p_skol = str(tmp_path / "skol_*_all.deb")
        p_django = str(tmp_path / "skol-django_*_all.deb")
        a = tmp_path / "skol_0.9.0-137_all.deb"
        b = tmp_path / "skol-django_0.9.0-115_all.deb"

        def land_both() -> None:
            a.write_text("fake deb")
            b.write_text("fake deb")

        _run_loop_with_sleeps([p_skol, p_django], [land_both])

        mock_install.assert_called_once()
        assert set(mock_install.call_args[0][0]) == {a, b}

    @patch('watch_incremental.install_packages')
    def test_pattern_with_no_files_yet_is_tolerated(
        self, mock_install: MagicMock, tmp_path: Path
    ) -> None:
        """Starting the watcher before the build finishes is the normal
        case, not an error."""
        pattern = str(tmp_path / "skol_*_all.deb")
        new = tmp_path / "skol_0.9.0-137_all.deb"

        _run_loop_with_sleeps(
            [pattern], [lambda: new.write_text("fake deb")])

        mock_install.assert_called_once()
        assert mock_install.call_args[0][0] == [new]


_XFAIL_VERSION_SORT = pytest.mark.xfail(
    reason=(
        "2026-08-12: get_latest_file sorts lexicographically, so build "
        "99 beats build 100; version-aware ordering lands in the "
        "follow-up commit."
    ),
    strict=True,
)

_XFAIL_HEARTBEAT = pytest.mark.xfail(
    reason=(
        "2026-08-12: heartbeat_due / format_heartbeat not yet "
        "implemented; lands in the follow-up commit."
    ),
    strict=True,
)


class TestVersionOrdering:
    """Build numbers are decimal, not lexicographic.

    The docstring claims deb names 'sort correctly by filename'.  They
    do not: '99' > '100' as strings, so at the 99->100 rollover the
    watcher installs the OLDER package.  Currently latent only because
    builds 130-136 all have three digits; it returns at 999->1000.
    """

    @_XFAIL_VERSION_SORT
    def test_build_100_beats_build_99(self, tmp_path: Path) -> None:
        old = _deb(tmp_path, "skol_0.9.0-99_all.deb")
        new = _deb(tmp_path, "skol_0.9.0-100_all.deb")
        assert watch_incremental.get_latest_file({old, new}) == new

    @_XFAIL_VERSION_SORT
    def test_build_1000_beats_build_999(self, tmp_path: Path) -> None:
        old = _deb(tmp_path, "skol_0.9.0-999_all.deb")
        new = _deb(tmp_path, "skol_0.9.0-1000_all.deb")
        assert watch_incremental.get_latest_file({old, new}) == new

    @_XFAIL_VERSION_SORT
    def test_upstream_version_ordering(self, tmp_path: Path) -> None:
        old = _deb(tmp_path, "skol_0.9.0-9_all.deb")
        new = _deb(tmp_path, "skol_0.10.0-1_all.deb")
        assert watch_incremental.get_latest_file({old, new}) == new


class TestHeartbeat:
    """While waiting, the watcher must look alive.

    A silent terminal is indistinguishable from a wedged one -- the
    ambiguity that made a working watcher look broken.
    """

    @_XFAIL_HEARTBEAT
    def test_not_due_before_the_interval(self) -> None:
        assert watch_incremental.heartbeat_due(
            now=100.0, last_emit=80.0, interval=60.0) is False

    @_XFAIL_HEARTBEAT
    def test_due_once_the_interval_has_passed(self) -> None:
        assert watch_incremental.heartbeat_due(
            now=141.0, last_emit=80.0, interval=60.0) is True

    @_XFAIL_HEARTBEAT
    def test_zero_interval_disables_heartbeat(self) -> None:
        assert watch_incremental.heartbeat_due(
            now=1e9, last_emit=0.0, interval=0.0) is False

    @_XFAIL_HEARTBEAT
    def test_message_reports_wait_and_current_newest(
        self, tmp_path: Path
    ) -> None:
        f = _deb(tmp_path, "skol_0.9.0-136_all.deb")
        pattern = str(tmp_path / "skol_*_all.deb")

        msg = watch_incremental.format_heartbeat(
            [pattern], {pattern: {f}}, waited=125.0)

        assert "skol_0.9.0-136_all.deb" in msg
        assert "2m" in msg or "125" in msg

    @_XFAIL_HEARTBEAT
    def test_message_names_patterns_still_waiting(
        self, tmp_path: Path
    ) -> None:
        pattern = str(tmp_path / "skol_*_all.deb")

        msg = watch_incremental.format_heartbeat(
            [pattern], {pattern: set()}, waited=30.0)

        assert "skol_*_all.deb" in msg


class TestIntegration:
    """End-to-end checks against the real script."""

    def test_help_works(self) -> None:
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).parent / 'watch_incremental.py'), '--help'],
            capture_output=True, text=True)

        assert result.returncode == 0
        assert '--postinstall' in result.stdout
        assert '--interval' in result.stdout

    @_XFAIL_HEARTBEAT
    def test_heartbeat_flag_is_documented(self) -> None:
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).parent / 'watch_incremental.py'), '--help'],
            capture_output=True, text=True)

        assert '--heartbeat' in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
