#!/usr/bin/env python3
"""Write ``json_canonical`` beside ``json_annotated`` in an experiment's
structured treatments database.

Proposal 2a of docs/plans/vocab-tree-narrowing.md (Trello AZeJMCR3).
Runs after ``treatments_to_json``, as the pipeline step
``canonicalize_json``.

The experiment record pins the map version it applies::

    "vocab_canonical": {"synonyms": "<blob hash>",
                        "spellings": "<blob hash>"}

``--pin`` writes the installed files' hashes into the record; it is the
only thing that changes the pin.  Otherwise the run refuses when the
installed files do not match the pin, and rewrites only the documents
that are stale: no ``json_canonical``, another map version, or a
``json_annotated`` changed since.  ``--force`` rewrites every document.

Usage::

    canonicalize_json --experiment production_v4_1 --pin
    canonicalize_json --experiment production_v4_1 [--dry-run] [--force]

Skeleton: the functions are stubs until the tests are approved.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from treatments_to_structured.vocab_canonical import (  # noqa: E402
    Spellings,
    Synonyms,
    Version,
)

RECORD_FIELD = 'vocab_canonical'


class PinError(Exception):
    """The record's pin is missing or does not match the installed map."""


def build_parser() -> argparse.ArgumentParser:
    """Strict CLI: script flags plus env_config's ``common_parser()``."""
    raise NotImplementedError


def pin_record(record: Mapping[str, Any],
               installed: Version) -> Dict[str, Any]:
    """A copy of the record pinned to ``installed``."""
    raise NotImplementedError


def check_pin(record: Mapping[str, Any], installed: Version) -> Version:
    """The record's pin, if it matches ``installed``; else PinError."""
    raise NotImplementedError


def plan_updates(docs: Iterable[Mapping[str, Any]], synonyms: Synonyms,
                 spellings: Spellings, version: Version,
                 force: bool = False
                 ) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    """The documents to write (each a full copy with the canonical
    fields set) and counts: ``checked``, ``updated``, ``fresh`` and
    ``unusable`` (no usable ``json_annotated``)."""
    raise NotImplementedError


def apply_updates(db: Any, docs: List[Dict[str, Any]],
                  batch: int = 500) -> int:
    """Save with bulk updates; returns how many were saved."""
    raise NotImplementedError


def main() -> int:
    """Pin, or bring the experiment's documents up to the pin."""
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
