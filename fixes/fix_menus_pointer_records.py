#!/usr/bin/env python3
"""One-time fix: make every experiment record name its menus tree key.

``redis_keys.menus`` names the Redis key holding the vocabulary tree
(docs/plans/experiment-record-canonical.md, Step 1).  Older records
name a *pointer* key instead (``skol:ui:menus_latest``, or the
2026-11-13 stopgap ``skol:ui:menus_production_v4_1_pointer``) whose
value is the tree key.  This rewrites each such record to the
pointer's target, then deletes the pointer keys no record names.

Usage::

    fixes/fix_menus_pointer_records.py --dry-run
    fixes/fix_menus_pointer_records.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bin'))

LATEST = 'skol:ui:menus_latest'


@dataclass(frozen=True)
class MenusFix:
    """One record whose ``redis_keys.menus`` names a pointer."""
    doc_id: str
    old: str
    new: str


def is_pointer(value: Optional[str]) -> bool:
    """True when a stored menus value is a key name, not tree JSON."""
    raise NotImplementedError


def plan_menus_fix(records: Iterable[Dict[str, Any]], r: Any) -> List[MenusFix]:
    """The records whose menus key holds a pointer, with its target.

    Only a pointer whose target holds a tree is rewritten.  Records
    with no menus key, a key holding a tree, a key holding nothing, or
    a pointer to a missing tree are left alone.
    """
    raise NotImplementedError


def pointer_keys_to_delete(r: Any, fixes: Sequence[MenusFix],
                           records: Iterable[Dict[str, Any]]) -> List[str]:
    """Pointer keys safe to delete once ``fixes`` are applied: each
    fixed record's old key, plus ``skol:ui:menus_latest`` if it is a
    pointer, excluding any key a record still names afterwards."""
    raise NotImplementedError


def apply_menus_fix(db: Any, fixes: Sequence[MenusFix]) -> int:
    """Rewrite each record's ``redis_keys.menus``; return the count."""
    raise NotImplementedError


def main(argv: Optional[List[str]] = None) -> int:
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
