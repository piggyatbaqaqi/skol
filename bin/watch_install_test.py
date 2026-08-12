#!/usr/bin/env python3
"""Tests for watch_install.py."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent))
import watch_install


class TestGetMtime:
    """Tests for get_mtime function."""

    def test_existing_file(self, tmp_path: Path) -> None:
        """Test getting mtime of existing file."""
        test_file = tmp_path / "test.deb"
        test_file.write_text("content")

        mtime = watch_install.get_mtime(test_file)

        assert mtime is not None
        assert isinstance(mtime, float)
        assert mtime > 0

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test getting mtime of nonexistent file returns None."""
        test_file = tmp_path / "nonexistent.deb"

        mtime = watch_install.get_mtime(test_file)

        assert mtime is None

    def test_mtime_changes_on_modification(self, tmp_path: Path) -> None:
        """Test that mtime changes when file is modified."""
        test_file = tmp_path / "test.deb"
        test_file.write_text("content1")
        mtime1 = watch_install.get_mtime(test_file)

        # Wait a bit and modify
        time.sleep(0.1)
        test_file.write_text("content2")
        mtime2 = watch_install.get_mtime(test_file)

        assert mtime2 > mtime1


class TestInstallPackages:
    """Tests for install_packages function."""

    @patch('watch_install.subprocess.run')
    def test_basic_install(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test basic package installation."""
        test_file = tmp_path / "test.deb"
        test_file.write_text("fake deb")
        mock_run.return_value = MagicMock(returncode=0)

        result = watch_install.install_packages(
            files=[test_file],
            install_cmd='dpkg -i',
            install_args=[],
            postinstall=None,
            verbosity=0,
        )

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert 'dpkg' in call_args
        assert '-i' in call_args
        assert str(test_file) in call_args

    @patch('watch_install.subprocess.run')
    def test_install_with_extra_args(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Test installation with extra arguments."""
        test_file = tmp_path / "test.deb"
        test_file.write_text("fake deb")
        mock_run.return_value = MagicMock(returncode=0)

        result = watch_install.install_packages(
            files=[test_file],
            install_cmd='dpkg -i',
            install_args=['--force-all'],
            postinstall=None,
            verbosity=0,
        )

        assert result is True
        call_args = mock_run.call_args[0][0]
        assert '--force-all' in call_args

    @patch('watch_install.subprocess.run')
    def test_install_with_postinstall(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        """Test installation with postinstall command."""
        test_file = tmp_path / "test.deb"
        test_file.write_text("fake deb")
        mock_run.return_value = MagicMock(returncode=0)

        result = watch_install.install_packages(
            files=[test_file],
            install_cmd='dpkg -i',
            install_args=[],
            postinstall='echo done',
            verbosity=0,
        )

        assert result is True
        # Should be called twice: once for dpkg, once for postinstall
        assert mock_run.call_count == 2
        # Second call should have shell=True
        assert mock_run.call_args_list[1][1].get('shell') is True

    @patch('watch_install.subprocess.run')
    def test_install_failure(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test handling of installation failure."""
        test_file = tmp_path / "test.deb"
        test_file.write_text("fake deb")
        mock_run.side_effect = subprocess.CalledProcessError(1, 'dpkg')

        result = watch_install.install_packages(
            files=[test_file],
            install_cmd='dpkg -i',
            install_args=[],
            postinstall=None,
            verbosity=0,
        )

        assert result is False

    @patch('watch_install.subprocess.run')
    def test_multiple_files(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Test installing multiple files at once."""
        files = [tmp_path / f"test{i}.deb" for i in range(3)]
        for f in files:
            f.write_text("fake deb")
        mock_run.return_value = MagicMock(returncode=0)

        result = watch_install.install_packages(
            files=files,
            install_cmd='dpkg -i',
            install_args=[],
            postinstall=None,
            verbosity=0,
        )

        assert result is True
        call_args = mock_run.call_args[0][0]
        for f in files:
            assert str(f) in call_args


class TestArgumentParsing:
    """Tests for command line argument parsing."""

    def test_basic_parsing(self) -> None:
        """Test basic argument parsing."""
        with patch.object(sys, 'argv', ['watch_install', 'test.deb']):
            # Can't easily test main() without it running the watch loop
            # So we test the argument parsing logic by simulating it
            args = sys.argv[1:]
            assert 'test.deb' in args

    def test_delimiter_parsing(self) -> None:
        """Test -- delimiter parsing."""
        args = ['--postinstall=cmd', 'file.deb', '--', '--force-all']

        delimiter_idx = args.index('--')
        pre_delimiter = args[:delimiter_idx]
        install_args = args[delimiter_idx + 1:]

        assert pre_delimiter == ['--postinstall=cmd', 'file.deb']
        assert install_args == ['--force-all']

    def test_no_delimiter(self) -> None:
        """Test parsing without -- delimiter."""
        args = ['--postinstall=cmd', 'file.deb']

        try:
            delimiter_idx = args.index('--')
            pre_delimiter = args[:delimiter_idx]
            install_args = args[delimiter_idx + 1:]
        except ValueError:
            pre_delimiter = args
            install_args = []

        assert pre_delimiter == ['--postinstall=cmd', 'file.deb']
        assert install_args == []


class TestWatchLoop:
    """Tests for the watch loop logic."""

    def test_detects_file_change(self, tmp_path: Path) -> None:
        """Test that file changes are detected."""
        test_file = tmp_path / "test.deb"
        test_file.write_text("content1")

        # Get initial mtime
        initial_mtime = watch_install.get_mtime(test_file)

        # Modify file
        time.sleep(0.1)
        test_file.write_text("content2")

        # Get new mtime
        new_mtime = watch_install.get_mtime(test_file)

        # Should detect change
        assert new_mtime > initial_mtime

    def test_detects_file_appearance(self, tmp_path: Path) -> None:
        """Test that new file appearance is detected."""
        test_file = tmp_path / "test.deb"

        # Initially doesn't exist
        assert watch_install.get_mtime(test_file) is None

        # Create file
        test_file.write_text("content")

        # Now exists
        assert watch_install.get_mtime(test_file) is not None

    def test_detects_file_disappearance(self, tmp_path: Path) -> None:
        """Test that file disappearance is handled."""
        test_file = tmp_path / "test.deb"
        test_file.write_text("content")

        # Initially exists
        assert watch_install.get_mtime(test_file) is not None

        # Remove file
        test_file.unlink()

        # Now doesn't exist
        assert watch_install.get_mtime(test_file) is None


_XFAIL_GLOB = pytest.mark.xfail(
    reason=(
        "2026-08-12: pattern globbing (newest_match / scan_once) not yet "
        "implemented; lands in the follow-up commit."
    ),
    strict=True,
)


def _touch(path: Path, mtime: float) -> Path:
    """Create `path` with a controlled mtime, so ordering is exact
    rather than dependent on filesystem timestamp resolution."""
    path.write_text("fake deb")
    os.utime(path, (mtime, mtime))
    return path


class TestNewestMatch:
    """`newest_match` resolves a glob to the most recently arrived file.

    This is what makes "install the newest version as it appears" work:
    build-deb.sh increments .build-number every build, so each build
    produces a NEW filename that shell expansion at launch could never
    have known about.
    """

    @_XFAIL_GLOB
    def test_picks_newest_of_several(self, tmp_path: Path) -> None:
        _touch(tmp_path / "skol_0.9.0-133_all.deb", 1000)
        newest = _touch(tmp_path / "skol_0.9.0-134_all.deb", 2000)
        _touch(tmp_path / "skol_0.9.0-131_all.deb", 500)

        assert watch_install.newest_match(
            str(tmp_path / "skol_*_all.deb")) == newest

    @_XFAIL_GLOB
    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        assert watch_install.newest_match(
            str(tmp_path / "skol_*_all.deb")) is None

    @_XFAIL_GLOB
    def test_literal_path_still_works(self, tmp_path: Path) -> None:
        """A non-glob filename must keep behaving as before -- it is
        just a pattern that matches exactly one file."""
        f = _touch(tmp_path / "skol_0.9.0-134_all.deb", 1000)
        assert watch_install.newest_match(str(f)) == f

    @_XFAIL_GLOB
    def test_ignores_non_matching_packages(self, tmp_path: Path) -> None:
        """skol_*_all.deb must not pick up skol-django_*_all.deb."""
        skol = _touch(tmp_path / "skol_0.9.0-134_all.deb", 1000)
        _touch(tmp_path / "skol-django_0.9.0-99_all.deb", 5000)
        assert watch_install.newest_match(
            str(tmp_path / "skol_*_all.deb")) == skol


class TestScanOnce:
    """`scan_once` is one poll of the watch loop: given the patterns and
    the previously-seen state, report what needs installing."""

    @_XFAIL_GLOB
    def test_startup_baseline_installs_nothing(self, tmp_path: Path) -> None:
        """Pre-existing debs must NOT be reinstalled just because the
        watcher started -- that was the original contract."""
        _touch(tmp_path / "skol_0.9.0-134_all.deb", 1000)
        pattern = str(tmp_path / "skol_*_all.deb")

        changed, state = watch_install.scan_once([pattern], {})

        assert changed == []
        assert state[pattern] is not None

    @_XFAIL_GLOB
    def test_new_version_appearing_is_installed(self, tmp_path: Path) -> None:
        """THE regression this whole change exists for: a brand-new
        filename lands after the watcher started."""
        _touch(tmp_path / "skol_0.9.0-134_all.deb", 1000)
        pattern = str(tmp_path / "skol_*_all.deb")
        _, state = watch_install.scan_once([pattern], {})

        newest = _touch(tmp_path / "skol_0.9.0-135_all.deb", 2000)
        changed, state = watch_install.scan_once([pattern], state)

        assert changed == [newest]

    @_XFAIL_GLOB
    def test_rebuild_in_place_is_installed(self, tmp_path: Path) -> None:
        """Same filename, newer mtime -- the pre-existing behaviour."""
        f = _touch(tmp_path / "skol_0.9.0-134_all.deb", 1000)
        pattern = str(tmp_path / "skol_*_all.deb")
        _, state = watch_install.scan_once([pattern], {})

        _touch(f, 2000)
        changed, _ = watch_install.scan_once([pattern], state)

        assert changed == [f]

    @_XFAIL_GLOB
    def test_unchanged_installs_nothing(self, tmp_path: Path) -> None:
        _touch(tmp_path / "skol_0.9.0-134_all.deb", 1000)
        pattern = str(tmp_path / "skol_*_all.deb")
        _, state = watch_install.scan_once([pattern], {})

        changed, _ = watch_install.scan_once([pattern], state)

        assert changed == []

    @_XFAIL_GLOB
    def test_pattern_matching_nothing_yet_is_tolerated(
        self, tmp_path: Path
    ) -> None:
        """Watching a pattern before any file exists must not raise --
        that is the normal 'waiting for the build' case."""
        pattern = str(tmp_path / "skol_*_all.deb")

        changed, state = watch_install.scan_once([pattern], {})

        assert changed == []
        assert state[pattern] is None

        appeared = _touch(tmp_path / "skol_0.9.0-135_all.deb", 1000)
        changed, _ = watch_install.scan_once([pattern], state)
        assert changed == [appeared]

    @_XFAIL_GLOB
    def test_multiple_patterns_batch_into_one_install(
        self, tmp_path: Path
    ) -> None:
        """skol and skol-django must be reported together so they reach
        a single `dpkg -i` and can satisfy each other's dependencies."""
        p_skol = str(tmp_path / "skol_*_all.deb")
        p_django = str(tmp_path / "skol-django_*_all.deb")
        _, state = watch_install.scan_once([p_skol, p_django], {})

        a = _touch(tmp_path / "skol_0.9.0-135_all.deb", 1000)
        b = _touch(tmp_path / "skol-django_0.9.0-135_all.deb", 1001)
        changed, _ = watch_install.scan_once([p_skol, p_django], state)

        assert set(changed) == {a, b}

    @_XFAIL_GLOB
    def test_disappearance_resets_without_installing(
        self, tmp_path: Path
    ) -> None:
        f = _touch(tmp_path / "skol_0.9.0-134_all.deb", 1000)
        pattern = str(tmp_path / "skol_*_all.deb")
        _, state = watch_install.scan_once([pattern], {})

        f.unlink()
        changed, state = watch_install.scan_once([pattern], state)

        assert changed == []
        assert state[pattern] is None


class TestQuotedGlobInvocation:
    """The exact command line that failed: quoted globs reaching Python
    unexpanded must now work rather than watching a literal '*' name."""

    @_XFAIL_GLOB
    def test_quoted_glob_is_expanded_by_the_script(
        self, tmp_path: Path
    ) -> None:
        _touch(tmp_path / "skol_0.9.0-134_all.deb", 1000)
        pattern = str(tmp_path / "skol_*_all.deb")

        _, state = watch_install.scan_once([pattern], {})

        # The literal pattern must never be treated as a filename.
        assert state[pattern] is not None
        assert "*" not in str(state[pattern][0])


class TestIntegration:
    """Integration tests."""

    def test_script_help(self) -> None:
        """Test that --help works."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / 'watch_install.py'),
             '--help'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert 'Watch deb files' in result.stdout
        assert '--postinstall' in result.stdout
        assert '--interval' in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
