#!/usr/bin/env python3
"""Skeleton — implementation lands in the next commit (TDD)."""

from typing import Any, Dict, List, Set, TextIO, Tuple


def field_tokens(doc: Dict[str, Any], english: Set[str]) -> Set[str]:
    raise NotImplementedError


def document_frequencies(db: Any, english: Set[str]) -> Tuple[Dict[str, int], int]:
    raise NotImplementedError


def select_vocabulary(freqs: Dict[str, int], threshold: int) -> List[str]:
    raise NotImplementedError


def write_vocabulary(words: List[str], stream: TextIO) -> None:
    raise NotImplementedError
