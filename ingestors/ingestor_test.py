"""Tests for ingestors/ingestor.py: the base class's delegation to its
HTTP client."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestors.rate_limited_client import RateLimitedHttpClient  # noqa: E402

INGESTOR_SOURCE = Path(__file__).resolve().parent / 'ingestor.py'


def _http_client_attributes() -> List[str]:
    """Every ``self.http_client.<name>`` that ingestor.py uses."""
    tree = ast.parse(INGESTOR_SOURCE.read_text())
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == 'http_client'
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == 'self'):
            names.add(node.attr)
    return sorted(names)


def test_finds_the_delegations() -> None:
    # Guards the parametrized test below against vacuously passing.
    assert {'get', 'retry_with_backoff'} <= set(_http_client_attributes())


# f6b9b48 (2026-06-24) removed these from the client, leaving the base
# class delegating to nothing.
DEAD = {'_apply_rate_limit', '_check_suppression', '_register_suppression'}


@pytest.mark.parametrize('name', [
    pytest.param(n, marks=pytest.mark.xfail(
        strict=True, reason='dead delegation, fix pending'))
    if n in DEAD else n
    for n in _http_client_attributes()])
def test_every_delegation_names_a_real_client_attribute(name: str) -> None:
    client = RateLimitedHttpClient(user_agent='test-agent')
    assert hasattr(client, name)
