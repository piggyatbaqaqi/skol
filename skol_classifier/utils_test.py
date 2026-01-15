"""
Tests for utils.py module.

Run with: pytest skol_classifier/utils_test.py -v
"""

import os
import tempfile
import pytest

from .utils import get_file_list


class TestGetFileList:
    """Tests for get_file_list function."""

    def test_get_file_list_basic(self, tmp_path):
        """Test basic file listing with default pattern."""
        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt.ann").write_text("content2")
        (tmp_path / "file3.pdf").write_text("content3")

        result = get_file_list(str(tmp_path), pattern="*.txt*")

        # Should find file1.txt and file2.txt.ann
        assert len(result) == 2
        assert any("file1.txt" in f for f in result)
        assert any("file2.txt.ann" in f for f in result)

    def test_get_file_list_recursive(self, tmp_path):
        """Test recursive file listing."""
        # Create nested directory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "file1.txt").write_text("content1")
        (subdir / "file2.txt").write_text("content2")

        result = get_file_list(str(tmp_path), pattern="**/*.txt*")

        # Should find both files
        assert len(result) == 2

    def test_get_file_list_exclude_pattern(self, tmp_path):
        """Test exclude pattern functionality."""
        # Create test files
        (tmp_path / "good_file.txt").write_text("content1")
        (tmp_path / "Sydowia_file.txt").write_text("content2")

        result = get_file_list(str(tmp_path), pattern="*.txt", exclude_pattern="Sydowia")

        # Should only find good_file.txt
        assert len(result) == 1
        assert "good_file.txt" in result[0]

    def test_get_file_list_custom_exclude(self, tmp_path):
        """Test custom exclude pattern."""
        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "excluded_file.txt").write_text("content2")

        result = get_file_list(str(tmp_path), pattern="*.txt", exclude_pattern="excluded")

        assert len(result) == 1
        assert "file1.txt" in result[0]

    def test_get_file_list_no_matches(self, tmp_path):
        """Test when no files match the pattern."""
        # Create test file with different extension
        (tmp_path / "file.pdf").write_text("content")

        result = get_file_list(str(tmp_path), pattern="*.txt")

        assert len(result) == 0

    def test_get_file_list_empty_directory(self, tmp_path):
        """Test with empty directory."""
        result = get_file_list(str(tmp_path), pattern="*.txt")

        assert len(result) == 0

    def test_get_file_list_nonexistent_folder(self):
        """Test with non-existent folder."""
        with pytest.raises(FileNotFoundError):
            get_file_list("/nonexistent/path/that/does/not/exist", pattern="*.txt")

    def test_get_file_list_returns_full_paths(self, tmp_path):
        """Test that returned paths are full paths."""
        (tmp_path / "file.txt").write_text("content")

        result = get_file_list(str(tmp_path), pattern="*.txt")

        assert len(result) == 1
        assert os.path.isabs(result[0])
        assert os.path.exists(result[0])
