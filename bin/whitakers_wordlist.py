#!/usr/bin/env python3
"""Skeleton — implementation lands in the next commit (TDD)."""

import dataclasses
from typing import Dict, List, Optional, Set, Tuple


@dataclasses.dataclass(frozen=True)
class Entry:
    stems: List[str]
    pos: str
    codes: List[str]
    age: str


def parse_inflections(text: str) -> Dict[Tuple[str, str, str], list]:
    raise NotImplementedError


def endings_for(infl, pos: str, decl: str, var: str) -> list:
    raise NotImplementedError


def parse_dictline(line: str) -> Optional[Entry]:
    raise NotImplementedError


def forms_for_entry(entry, infl) -> Set[str]:
    raise NotImplementedError


def fold_form(form: str) -> Optional[str]:
    raise NotImplementedError


def build_wordlist(dictline: str, inflects: str,
                   ages: Optional[Set[str]] = None) -> List[str]:
    raise NotImplementedError
