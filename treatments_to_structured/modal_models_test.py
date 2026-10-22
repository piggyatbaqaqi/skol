"""Tests for treatments_to_structured/modal_models.py.

Run with: python -m pytest treatments_to_structured/modal_models_test.py -v
"""
from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import List

import pytest

from treatments_to_structured.modal_models import (
    ALLOW_PATTERNS,
    MARKER_NAME,
    MODEL_ID,
    MODEL_REVISION,
    PREPARE_RESOURCES,
    check,
    is_prepared,
    model_dir,
    prepare,
    read_marker,
)

NOW = '2026-10-22T00:00:00+00:00'
REV = 'c170c708c41dac9275d15a8fff4eca08d52bab71'
OTHER_REV = '0' * 40

# The pinned revision's file list (HfApi.model_info, 2026-10-22).
REPO_FILES = [
    '.gitattributes', 'README.md', 'config.json',
    'consolidated.safetensors', 'generation_config.json',
    'model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors',
    'model-00003-of-00003.safetensors', 'model.safetensors.index.json',
    'params.json', 'special_tokens_map.json', 'tokenizer.json',
    'tokenizer.model', 'tokenizer.model.v3', 'tokenizer_config.json',
]


def selected(files: List[str]) -> List[str]:
    return [f for f in files
            if any(fnmatch.fnmatch(f, p) for p in ALLOW_PATTERNS)]


class FakeDownloader:
    """Writes the allowed files, as snapshot_download would."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: List[tuple] = []
        self.fail = fail

    def __call__(self, model_id: str, revision: str, dest: Path,
                 allow_patterns: List[str]) -> None:
        self.calls.append((model_id, revision, dest, list(allow_patterns)))
        dest.mkdir(parents=True, exist_ok=True)
        for name in [f for f in REPO_FILES
                     if any(fnmatch.fnmatch(f, p) for p in allow_patterns)]:
            (dest / name).write_text('x')
            if self.fail:
                raise OSError('connection reset')


# ---------------------------------------------------------------------------
# Constants: pinning, file selection, CPU-only preparation
# ---------------------------------------------------------------------------


class TestConstants:

    def test_model_is_pinned_to_a_commit(self) -> None:
        assert MODEL_ID == 'mistralai/Mistral-7B-Instruct-v0.3'
        assert re.fullmatch(r'[0-9a-f]{40}', MODEL_REVISION)

    def test_downloads_hugging_face_format_only(self) -> None:
        """consolidated.safetensors is a second 14.5 GB copy of the
        weights in Mistral's own format; vLLM uses the sharded files."""
        chosen = selected(REPO_FILES)
        assert 'consolidated.safetensors' not in chosen
        assert 'params.json' not in chosen
        for needed in ('config.json', 'model-00001-of-00003.safetensors',
                       'model-00003-of-00003.safetensors',
                       'model.safetensors.index.json', 'tokenizer.json',
                       'tokenizer_config.json'):
            assert needed in chosen

    def test_preparation_uses_no_gpu(self) -> None:
        assert PREPARE_RESOURCES['gpu'] is None


# ---------------------------------------------------------------------------
# Layout and marker
# ---------------------------------------------------------------------------


class TestModelDir:

    def test_layout(self, tmp_path: Path) -> None:
        assert model_dir(tmp_path, MODEL_ID, REV) == (
            tmp_path / 'models' / 'mistralai--Mistral-7B-Instruct-v0.3'
            / REV)

    def test_revisions_do_not_share_a_directory(self,
                                                tmp_path: Path) -> None:
        assert (model_dir(tmp_path, MODEL_ID, REV)
                != model_dir(tmp_path, MODEL_ID, OTHER_REV))


class TestPrepare:

    def test_downloads_when_absent_and_writes_marker(
            self, tmp_path: Path) -> None:
        download = FakeDownloader()
        status = prepare(tmp_path, MODEL_ID, REV, download, NOW)
        assert status == 'downloaded'
        model_id, revision, dest, patterns = download.calls[0]
        assert (model_id, revision) == (MODEL_ID, REV)
        assert dest == model_dir(tmp_path, MODEL_ID, REV)
        assert patterns == ALLOW_PATTERNS
        marker = read_marker(dest)
        assert marker is not None
        assert marker['model_id'] == MODEL_ID
        assert marker['revision'] == REV
        assert marker['prepared_at'] == NOW
        assert sorted(marker['files']) == sorted(selected(REPO_FILES))

    def test_present_when_already_prepared(self, tmp_path: Path) -> None:
        prepare(tmp_path, MODEL_ID, REV, FakeDownloader(), NOW)
        again = FakeDownloader()
        assert prepare(tmp_path, MODEL_ID, REV, again, NOW) == 'present'
        assert again.calls == []

    def test_failed_download_writes_no_marker(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            prepare(tmp_path, MODEL_ID, REV, FakeDownloader(fail=True), NOW)
        assert read_marker(model_dir(tmp_path, MODEL_ID, REV)) is None
        assert is_prepared(tmp_path, MODEL_ID, REV) is False

    def test_partial_download_is_redone(self, tmp_path: Path) -> None:
        """Files without a marker (an interrupted download) are not
        trusted."""
        dest = model_dir(tmp_path, MODEL_ID, REV)
        dest.mkdir(parents=True)
        (dest / 'config.json').write_text('x')
        download = FakeDownloader()
        assert prepare(tmp_path, MODEL_ID, REV, download, NOW) == (
            'downloaded')
        assert len(download.calls) == 1


class TestIsPrepared:

    def test_nothing_there(self, tmp_path: Path) -> None:
        assert is_prepared(tmp_path, MODEL_ID, REV) is False

    def test_other_revision_prepared(self, tmp_path: Path) -> None:
        prepare(tmp_path, MODEL_ID, OTHER_REV, FakeDownloader(), NOW)
        assert is_prepared(tmp_path, MODEL_ID, REV) is False

    def test_marker_for_another_revision(self, tmp_path: Path) -> None:
        """A marker copied into the wrong directory does not count."""
        prepare(tmp_path, MODEL_ID, REV, FakeDownloader(), NOW)
        dest = model_dir(tmp_path, MODEL_ID, REV)
        marker = json.loads((dest / MARKER_NAME).read_text())
        marker['revision'] = OTHER_REV
        (dest / MARKER_NAME).write_text(json.dumps(marker))
        assert is_prepared(tmp_path, MODEL_ID, REV) is False

    def test_listed_file_missing(self, tmp_path: Path) -> None:
        prepare(tmp_path, MODEL_ID, REV, FakeDownloader(), NOW)
        dest = model_dir(tmp_path, MODEL_ID, REV)
        (dest / 'model-00002-of-00003.safetensors').unlink()
        assert is_prepared(tmp_path, MODEL_ID, REV) is False

    def test_unreadable_marker(self, tmp_path: Path) -> None:
        dest = model_dir(tmp_path, MODEL_ID, REV)
        dest.mkdir(parents=True)
        (dest / MARKER_NAME).write_text('{not json')
        assert read_marker(dest) is None
        assert is_prepared(tmp_path, MODEL_ID, REV) is False


class TestCheck:

    def test_reports_without_downloading(self, tmp_path: Path) -> None:
        report = check(tmp_path, MODEL_ID, REV)
        assert report == {
            'model_id': MODEL_ID, 'revision': REV,
            'path': str(model_dir(tmp_path, MODEL_ID, REV)),
            'prepared': False,
        }
        assert not model_dir(tmp_path, MODEL_ID, REV).exists()

    def test_reports_prepared(self, tmp_path: Path) -> None:
        prepare(tmp_path, MODEL_ID, REV, FakeDownloader(), NOW)
        assert check(tmp_path, MODEL_ID, REV)['prepared'] is True
