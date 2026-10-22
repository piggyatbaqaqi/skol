"""Tests for bin/prepare_modal_models.py.

Run with: python -m pytest bin/prepare_modal_models_test.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_modal_models import (  # type: ignore[import]  # noqa: E402
    build_parser,
    run,
    validate_revision,
)
from treatments_to_structured.modal_models import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
)

NOT_YET = pytest.mark.xfail(
    strict=True, reason="prepare_modal_models not implemented yet")


class FakeRemote:
    """Records calls; stands in for the Modal functions."""

    def __init__(self, prepared: bool = False, fail: bool = False) -> None:
        self.prepared = prepared
        self.fail = fail
        self.calls: List[tuple] = []

    def check(self, model_id: str, revision: str) -> Dict[str, Any]:
        self.calls.append(('check', model_id, revision))
        if self.fail:
            raise RuntimeError('modal: unauthorized')
        return {'model_id': model_id, 'revision': revision,
                'path': '/models/x', 'prepared': self.prepared}

    def prepare(self, model_id: str, revision: str) -> str:
        self.calls.append(('prepare', model_id, revision))
        if self.fail:
            raise RuntimeError('modal: unauthorized')
        return 'present' if self.prepared else 'downloaded'


def parse(*argv: str) -> Any:
    return build_parser().parse_args(list(argv))


class TestParser:

    def test_defaults_are_the_pinned_model(self) -> None:
        args = parse()
        assert args.model == MODEL_ID
        assert args.revision == MODEL_REVISION
        assert args.check is False

    def test_rejects_unknown_flags(self) -> None:
        with pytest.raises(SystemExit):
            parse('--chek')


@NOT_YET
class TestValidateRevision:

    def test_full_commit(self) -> None:
        assert validate_revision(MODEL_REVISION) == MODEL_REVISION

    def test_refuses_branches_tags_and_short_hashes(self) -> None:
        for rev in ('main', 'v0.3', 'c170c708', 'C170C708' + 'C' * 32):
            with pytest.raises(ValueError):
                validate_revision(rev)


@NOT_YET
class TestRun:

    def test_check_does_not_prepare(self, capsys: Any) -> None:
        remote = FakeRemote(prepared=False)
        assert run(parse('--check'), remote) == 1
        assert remote.calls == [('check', MODEL_ID, MODEL_REVISION)]
        assert 'not prepared' in capsys.readouterr().out

    def test_check_when_prepared(self, capsys: Any) -> None:
        remote = FakeRemote(prepared=True)
        assert run(parse('--check'), remote) == 0
        assert 'prepared' in capsys.readouterr().out

    def test_prepare_downloads(self, capsys: Any) -> None:
        remote = FakeRemote(prepared=False)
        assert run(parse(), remote) == 0
        assert remote.calls == [('prepare', MODEL_ID, MODEL_REVISION)]
        assert 'downloaded' in capsys.readouterr().out

    def test_prepare_when_present(self, capsys: Any) -> None:
        assert run(parse(), FakeRemote(prepared=True)) == 0
        assert 'present' in capsys.readouterr().out

    def test_explicit_model_and_revision(self) -> None:
        remote = FakeRemote()
        rev = 'a' * 40
        run(parse('--model', 'org/other', '--revision', rev), remote)
        assert remote.calls == [('prepare', 'org/other', rev)]

    def test_invalid_revision_never_reaches_modal(self, capsys: Any) -> None:
        remote = FakeRemote()
        assert run(parse('--revision', 'main'), remote) == 2
        assert remote.calls == []
        assert 'main' in capsys.readouterr().err

    def test_remote_failure_is_reported(self, capsys: Any) -> None:
        assert run(parse(), FakeRemote(fail=True)) == 2
        assert 'unauthorized' in capsys.readouterr().err
