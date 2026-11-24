"""Tests for ingestors/web_mirror.py: files kept under /data/skol/www."""

import hashlib
from pathlib import Path

import pytest

from .web_mirror import mirror_path, save_mirror_copy

URL = ('https://archive.org/download/mycologicalwriti02lloyiala/'
       'mycologicalwriti02lloyiala.pdf')


class TestMirrorPath:
    def test_host_then_path(self, tmp_path: Path) -> None:
        assert mirror_path(URL, tmp_path) == (
            tmp_path / 'archive.org' / 'download' /
            'mycologicalwriti02lloyiala' / 'mycologicalwriti02lloyiala.pdf')

    def test_http_and_port_kept_in_host(self, tmp_path: Path) -> None:
        assert mirror_path('http://example.org:8080/a/b.pdf', tmp_path) == (
            tmp_path / 'example.org:8080' / 'a' / 'b.pdf')

    def test_host_lower_cased_path_kept_as_given(
            self, tmp_path: Path) -> None:
        assert mirror_path('https://Example.ORG/A%20B/C.pdf', tmp_path) == (
            tmp_path / 'example.org' / 'A%20B' / 'C.pdf')

    @pytest.mark.parametrize('url', [
        'ftp://example.org/a.pdf',
        'file:///etc/passwd',
        'https:///a.pdf',
        'https://example.org/',
        'https://example.org',
        'https://example.org/a/',
        'https://example.org/a.pdf?x=1',
        'https://example.org/a.pdf#p2',
        'https://example.org/../etc/passwd',
        'https://example.org/a/./b.pdf',
        'https://example.org/a//b.pdf',
        'https://example.org/a/..',
    ])
    def test_refuses_urls_that_do_not_name_one_file(
            self, tmp_path: Path, url: str) -> None:
        with pytest.raises(ValueError):
            mirror_path(url, tmp_path)


class TestSaveMirrorCopy:
    def test_writes_and_describes_the_file(self, tmp_path: Path) -> None:
        info = save_mirror_copy(URL, b'%PDF-1.5 whole', tmp_path)
        path = mirror_path(URL, tmp_path)
        assert path.read_bytes() == b'%PDF-1.5 whole'
        assert info == {
            'path': str(path),
            'sha256': hashlib.sha256(b'%PDF-1.5 whole').hexdigest(),
            'bytes': 14,
        }

    def test_creates_subdirectories_and_leaves_no_temp_file(
            self, tmp_path: Path) -> None:
        save_mirror_copy(URL, b'x', tmp_path)
        files = [p for p in tmp_path.rglob('*') if p.is_file()]
        assert files == [mirror_path(URL, tmp_path)]

    def test_overwrites_an_existing_copy(self, tmp_path: Path) -> None:
        save_mirror_copy(URL, b'old', tmp_path)
        save_mirror_copy(URL, b'new', tmp_path)
        assert mirror_path(URL, tmp_path).read_bytes() == b'new'

    def test_new_directories_inherit_the_setgid_group(
            self, tmp_path: Path) -> None:
        # /data/skol/www is setgid group skol; new subdirectories must
        # keep that, not get the writer's primary group.
        tmp_path.chmod(0o2775)
        save_mirror_copy(URL, b'x', tmp_path)
        new_dir = mirror_path(URL, tmp_path).parent
        assert new_dir.stat().st_gid == tmp_path.stat().st_gid
        assert new_dir.stat().st_mode & 0o2000

    def test_bad_url_writes_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            save_mirror_copy('https://example.org/../x', b'x', tmp_path)
        assert list(tmp_path.iterdir()) == []
