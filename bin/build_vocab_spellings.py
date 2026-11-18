#!/usr/bin/env python3
"""Generate ``data/vocab_key_spellings.json``: each key group's spelling.

Proposal 2a of docs/plans/vocab-tree-narrowing.md (Trello AZeJMCR3).
Every JSON key, at every depth, of an experiment's ``json_annotated``
documents is counted once per document under its group
(``vocab_canonical.group_key``) and its ``spelling_candidate``.  A
group's spelling is its most frequent candidate (ties: alphabetical),
counted in canonical space (the 2026-08-17 rule).  Groups found in
fewer than ``--min-support`` documents are left out; their keys stay
as written.

The output is git-tracked and reviewed as a diff, then shipped through
``SHARE_FILES``; changing it changes the map version records pin.

Usage::

    build_vocab_spellings --experiment production_v4_1 [--min-support 2]

Skeleton: the functions are stubs until the tests are approved.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MIN_SUPPORT = 2


def count_spellings(docs: Iterable[Mapping[str, Any]]
                    ) -> Dict[str, Counter[str]]:
    """Per group, how many documents use each spelling candidate."""
    raise NotImplementedError


def choose_spellings(counts: Mapping[str, Counter[str]],
                     min_support: int = DEFAULT_MIN_SUPPORT
                     ) -> Dict[str, str]:
    """Each supported group's most frequent candidate."""
    raise NotImplementedError


def table_text(spellings: Mapping[str, str],
               generated: Mapping[str, Any]) -> str:
    """The table file's text: deterministic, so a regeneration diffs
    cleanly."""
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    """Strict CLI: script flags plus env_config's ``common_parser()``."""
    raise NotImplementedError


def main() -> int:
    """Count an experiment's keys and write the table."""
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
