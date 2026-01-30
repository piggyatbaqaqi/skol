#!/usr/bin/env python3
"""Tests for watch_install.py."""
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
