#!/usr/bin/env python3
"""One-time fix: delete orphaned test collections from the local CouchDB.

Stub: the tests in expunge_orphan_collections_test.py are written first.
Trello 9ZkM5Wuq; see docs/django-test-isolation.md.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

DEFAULT_DJANGO_DB = str(
    Path(__file__).resolve().parent.parent / 'django' / 'db.sqlite3')


class RefusedError(Exception):
    """The live-collection list cannot be trusted; nothing is deleted."""


@dataclass
class Plan:
    """What a run would delete, per database."""
    orphans: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    kept: Dict[str, int] = field(default_factory=dict)
    by_owner: Counter[str] = field(default_factory=Counter)


def load_live_ids(django_db: str) -> Set[int]:
    raise NotImplementedError


def collection_id_of(doc_id: str) -> Optional[int]:
    raise NotImplementedError


def make_plan(dbs: Dict[str, Any], live_ids: Set[int]) -> Plan:
    raise NotImplementedError


def apply_plan(plan: Plan, dbs: Dict[str, Any], batch: int = 500) -> int:
    raise NotImplementedError


def report(plan: Plan) -> str:
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    raise NotImplementedError


def main() -> int:
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
